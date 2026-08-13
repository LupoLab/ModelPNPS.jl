# =============================================================================
# GPU (CUDA) go/no-go benchmark for the TG-FROG propagation core.
#
# Standalone — needs only CUDA.jl, not Luna/ModelPNPS. Times the computational
# primitives of one RK4(5) propagation step in FP64 at the real campaign array
# shapes, and assembles the per-step cost model:
#
#   step ≈ 14×FFT + 14×propagator + 6×Kerr + 14×fused-broadcast passes
#   (12 FFTs from the 6 RHS evaluations + 2 for the per-step apodisation;
#    the Raman arm adds 6 × [2 batched doubled-grid FFTs + 3 broadcasts])
#
# Compare the printed per-point estimate against the measured CPU numbers from
# the timing_N640_*t / verifyAB_* jobs (Luna prints "Propagation finished in
# X seconds, N steps" in those logs — use the same step count here via ARGS).
#
# This is a KERNEL benchmark, not a port: a real GPU engine would add
# host<->device transfers at the z-saves and the extraction pass (both minor at
# these sizes over PCIe 4), plus 2–3 weeks of engineering. The point of this
# script is to decide whether that investment could ever pay off.
#
# One-time environment (login node is fine; artifacts finalise on first GPU use):
#   julia -e 'using Pkg; Pkg.activate(ENV["HOME"]*"/perfstack/cudaenv");
#             Pkg.add("CUDA")'
#
# Run (adjust partition/gres names to the cluster's GPU queue):
#   sbatch <<'EOF'
#   #!/bin/bash
#   #SBATCH --job-name=gpubench
#   #SBATCH --partition=<gpu-partition>
#   #SBATCH --gres=gpu:1
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=4
#   #SBATCH --mem=64G
#   #SBATCH --time=01:00:00
#   julia --project=$HOME/perfstack/cudaenv \
#         $HOME/perfstack/ModelPNPS/examples/gpu_bench.jl 1000
#   EOF
#
# ARGS[1] (optional): RK steps per delay point for the per-point estimate
# (default 1000; read the true value from a CPU job log for the same config).
# =============================================================================

using CUDA
using Printf

const NSTEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000

# (Nt, N) shapes: 40 µm campaigns have Nt=128, the 100 µm campaign Nt=512;
# N=640 is the optimised transverse grid, N=1024 the legacy/pinned one.
const SHAPES = [(128, 640), (128, 1024), (512, 640), (512, 1024)]

"Median-of-n GPU timing (CUDA.@elapsed uses events and synchronises)."
function gputime(f; n=7)
    f(); CUDA.synchronize() # warm-up + compile
    ts = [Float64(CUDA.@elapsed f()) for _ in 1:n]
    return minimum(ts)
end

gib(bytes) = bytes / 2^30

println("Device: ", CUDA.name(CUDA.device()))
println("Memory: ", round(gib(CUDA.total_memory()); digits=1), " GiB total, ",
        round(gib(CUDA.available_memory()); digits=1), " GiB available")
println("FP64 benchmark; per-point estimates use $NSTEPS RK steps (ARGS[1] to change)\n")

for (Nt, N) in SHAPES
    field_bytes = 16 * Nt * N * N
    # state + one scratch + factored-linop factors (tiny); materialised linop and the
    # doubled Raman buffer are optional extras checked against free memory below
    base_need = 2 * field_bytes + 2^30 # + FFT workspace headroom
    if CUDA.available_memory() < base_need
        @printf("(%d, %d, %d): skipped — needs ≳%.1f GiB free\n\n", Nt, N, N, gib(base_need))
        continue
    end
    @printf("=== shape (%d, %d, %d) — field array %.2f GiB ===\n", Nt, N, N, gib(field_bytes))

    E = CUDA.rand(ComplexF64, Nt, N, N)
    K = CUDA.rand(ComplexF64, Nt, N, N)

    # -- 3-D c2c FFT (in-place; fwd and inv cost the same) --------------------
    p = CUDA.CUFFT.plan_fft!(E, (1, 2, 3))
    t_fft = gputime(() -> p * E)
    @printf("  3-D FFT (c2c, FP64):        %8.2f ms\n", 1e3*t_fft)

    # -- propagator kernel, factored form (sqrt + cis on the fly) -------------
    dz = 1e-7
    β1 = 5e-9
    ωv = reshape(CuArray(collect(range(1e15, 4e15, length=Nt))), Nt, 1, 1)
    k2v = (ωv .* (1.5/2.99792458e8)).^2
    kperp2 = reshape(CUDA.rand(Float64, N, N) .* 1e13, 1, N, N)
    prop_fac!() = (@. E *= cis(-(sqrt(abs(k2v - kperp2)) - β1*ωv)*dz); nothing)
    t_prop_fac = gputime(prop_fac!)
    @printf("  propagator (factored):      %8.2f ms\n", 1e3*t_prop_fac)

    # -- propagator kernel, materialised linop (if memory allows) -------------
    t_prop_mat = NaN
    if CUDA.available_memory() > field_bytes + 2^30
        L = CUDA.rand(ComplexF64, Nt, N, N) .* (-0.01 - 1e6im)
        prop_mat!() = (@. E *= exp(L*dz); nothing)
        t_prop_mat = gputime(prop_mat!)
        @printf("  propagator (materialised): %8.2f ms\n", 1e3*t_prop_mat)
        CUDA.unsafe_free!(L)
    end

    # -- Kerr response (pointwise broadcast, accumulate form) -----------------
    fac = 1e-30
    kerr!() = (@. K += fac*abs2(E)*E; nothing)
    t_kerr = gputime(kerr!)
    @printf("  Kerr broadcast:             %8.2f ms\n", 1e3*t_kerr)

    # -- fused Butcher-stage pass (representative 3-array broadcast) ----------
    c1 = 0.3
    axpy!() = (@. K = E + c1*K; nothing)
    t_axpy = gputime(axpy!)
    @printf("  fused stage pass:           %8.2f ms\n", 1e3*t_axpy)

    # -- batched Raman: 2 FFTs on the doubled time grid + broadcasts ----------
    t_raman = NaN
    raman_bytes = 2 * field_bytes
    if CUDA.available_memory() > raman_bytes + 2^30
        B = CUDA.rand(ComplexF64, 2Nt, N, N)
        pB = CUDA.CUFFT.plan_fft!(B, 1)
        hω = CUDA.rand(ComplexF64, 2Nt)
        hr = reshape(hω, 2Nt, 1, 1)
        raman!() = begin
            pB * B
            @. B *= hr
            pB * B # stand-in for the inverse (same cost)
            nothing
        end
        t_raman = gputime(raman!)
        @printf("  Raman (2 batched FFTs + scale): %6.2f ms\n", 1e3*t_raman)
        CUDA.unsafe_free!(B)
    else
        println("  Raman buffer skipped (insufficient free memory)")
    end

    # -- per-step and per-point model ------------------------------------------
    t_prop = isnan(t_prop_mat) ? t_prop_fac : min(t_prop_fac, t_prop_mat)
    step_kerr = 14*t_fft + 14*t_prop + 6*t_kerr + 14*t_axpy
    @printf("  => Kerr step ≈ %.1f ms; per point (%d steps) ≈ %.1f min\n",
            1e3*step_kerr, NSTEPS, NSTEPS*step_kerr/60)
    if !isnan(t_raman)
        step_raman = step_kerr + 6*(t_raman + 2*t_kerr)
        @printf("  => Raman step ≈ %.1f ms; per point (%d steps) ≈ %.1f min\n",
                1e3*step_raman, NSTEPS, NSTEPS*step_raman/60)
    end
    println()
    CUDA.unsafe_free!(E)
    CUDA.unsafe_free!(K)
    GC.gc(); CUDA.reclaim()
end

println("Compare the per-point estimates against the CPU wall times from the")
println("timing_N640_*t / verifyAB_* job logs (same step count). Rule of thumb:")
println("a GPU engine is worth the 2-3 week port if (points per campaign) ×")
println("(CPU-GPU time difference) × (number of future campaigns) is large")
println("against the whole 128-core CPU allocation running concurrent points.")
