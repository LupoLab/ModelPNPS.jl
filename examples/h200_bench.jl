# =============================================================================
# ModelPNPS TG-FROG delay points on an H100/H200: accuracy A/B, then timing.
#
# Companion to Luna's test/manual/hpc_gpu_bench.jl (which benchmarks the MODAL
# transform). This one exercises the 3-D free-space path — `Grid.EnvGrid` ×
# `Grid.FreeGrid`, one `(Nω, N, N)` ComplexF64 state — at the shapes the
# production campaigns actually use.
#
# Run through examples/h200_modelpnps_suite.sh, or directly:
#   julia --project=/workspace/code/dev h200_bench.jl --mode=accuracy
#   julia --project=/workspace/code/dev h200_bench.jl --mode=bench --cases=dd05,04
#
# Options
#   --mode=accuracy|bench   accuracy: GPU-only self-consistency + rtol convergence
#                           bench:    GPU delay points at production shapes
#   --case=NAME             accuracy case (default dd05)
#   --cpu=1                 add a host reference to the accuracy run (slow; use the HPC)
#   --cases=a,b,c           bench cases, see CASES below (default dd05,04,dd20,100um)
#   --points=N              delay points per case (default 1)
#   --out=FILE              append a CSV row per point
#   --extract=on|off|both   save-time extraction (default on; `both` A/Bs it)
#
# CUDA is imported at the top level here deliberately. The cluster scripts must
# NOT do that (their login node faults precompiling CUDA.jl, hence the lazy
# `arraytype=:cuda` symbol and its world-age dance); a rented pod has the GPU
# directly attached, so the simple thing is also the correct one.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import CUDA
import Printf: @printf, @sprintf
import Dates

# ------------------------------------------------------------------ options --
opt(name, default) = begin
    p = "--$name="
    i = findfirst(a -> startswith(a, p), ARGS)
    isnothing(i) ? default : ARGS[i][length(p)+1:end]
end
MODE     = opt("mode", "accuracy")
POINTS   = parse(Int, opt("points", "1"))
OUT      = opt("out", "")
EXTRACT  = opt("extract", "on")

# --------------------------------------------------------- shared physics ----
# Identical to the production drivers (04 / 05 / 07): 1 fs, 0.1 µJ, 260 nm into
# 40 µm (or 100 µm) of SiO2 through the boxcars mask. Only the GRID differs
# between cases, which is the whole point — grid shape is what sets cost.
const TWIN_SAVES_ONLY = 1_000_000_000

function setup_args_for(; gap_mm, N, trange, thickness)
    gap  = gap_mm * 1e-3
    d    = gap/2 + 1.0e-3/2
    beam = TS.HE11Beam(125e-6, 5.0, 0.1)
    win  = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                                 zmask=0.1, apod=:tanh)
    return (; λ0=260e-9, τfwhm=1.0e-15, energy=0.1e-6,
              material=:SiO2, thickness=thickness,
              mask_diam=1.0e-3, mask_spacing=gap, λlims=(143e-9, 600e-9),
              beam=beam, window=win, apod=:supergauss, apod_param=16,
              trange=trange, store_window=false, R=366.0e-6, N=N)
end

const ZSAVE_40  = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
                   12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6
const ZSAVE_100 = [0.0, 2.0, 4.0, 8.0, 9.5, 16.0, 25.0, 40.0, 60.0, 100.0] .* 1e-6

# Production shapes. `field_gib` is one ComplexF64 (Nω, N, N) array; the solver
# holds 9 of them plus one transform buffer, so device memory ≈ 10 × that
# (measured exactly on an A40 at the `04` shape).
const CASES = Dict(
  # name      gap  N     trange   thickness  zsave      what it is
  "dd05"  => (gap_mm=0.5, N=640,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="07 d/D series, gap 0.5 mm"),
  "04"    => (gap_mm=1.0, N=768,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="04 Kerr production — the A40 reference shape"),
  "dd20"  => (gap_mm=2.0, N=1024, trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="07 d/D series, gap 2.0 mm — marginal on a 48 GB card"),
  "100um" => (gap_mm=1.0, N=640,  trange=220e-15, thickness=100e-6, zsave=ZSAVE_100,
              note="05 anchor: Nω=512, 2.5× the propagation distance"),
)
CASE_NAMES = split(opt("cases", "dd05,04,dd20,100um"), ",")

# The production solver settings. These are not tuning knobs: rtol=1e-7 and
# saves-only windowing are what the campaigns are validated at, and changing
# either changes the answer by far more than anything measured here.
const SOLVER = (; init_dz=5e-7, rtol=1e-7, max_dz=2e-6, twin_period=TWIN_SAVES_ONLY)

field_gib(Nω, N) = 16 * Nω * N * N / 2^30
devfree() = (s = Luna.device_memory_status(); isnothing(s) ? NaN : s[1]/2^30)
gib(x) = round(x; digits=2)

function banner()
    d = CUDA.device()
    @printf("host %s   julia %s   threads %d\n", gethostname(), VERSION, Threads.nthreads())
    @printf("device %s   %.1f GiB   driver CUDA %s   runtime %s\n",
            CUDA.name(d), CUDA.total_memory()/2^30,
            CUDA.driver_version(), CUDA.runtime_version())
end

csv_row(f, row) = isempty(OUT) || open(OUT, "a") do io; println(io, join(row, ",")); end

# ============================================================================ #
#  ACCURACY — GPU only
# ============================================================================ #
#
# No CPU reference is computed here. A CPU delay point at a production shape is
# ~50 min on 8 threads, this is rented GPU time, and the host-vs-device comparison
# is already covered where it belongs: ModelPNPS's hardware-gated CUDA testset
# (small grids, seconds) and, at production scale, `verify_against_collected`
# against a bit-identical CPU reference on the HPC.
#
# What is left is what only this machine can answer, and both parts are pure GPU:
#
#   1. Save-time extraction vs saving the stack, same backend, same grid. These
#      reduce the same slices by different routes; a difference is the extraction,
#      not the propagation. Must agree to rounding.
#   2. The solver's own convergence at this shape: production `rtol` against a
#      decade tighter. This is the number that actually bounds the campaign, and
#      it is NOT a backend question — it is worth measuring at any grid the
#      campaigns have not already validated, which is most of them.
#
# Pass --cpu=1 to add a host reference anyway (slow; prefer the HPC).
function run_accuracy()
    name = get(CASES, opt("case", "dd05"), CASES["dd05"])
    c = name
    println("\n=== accuracy (GPU only): $(opt("case", "dd05")) — $(c.note) ===")
    sa = setup_args_for(gap_mm=c.gap_mm, N=c.N, trange=c.trange, thickness=c.thickness)
    setup = TS.build_setup(; sa..., arraytype=CUDA.CuArray)
    Nω = length(setup.grid.ω)
    @printf("grid (%d, %d, %d)  field %.2f GiB  %d z-slices\n",
            Nω, c.N, c.N, field_gib(Nω, c.N), length(c.zsave))
    τ = -8e-15   # off centre: a wing point exercises the weak-signal regime

    ok = true
    println("\n1. save-time extraction vs saving the stack (same GPU, same grid)")
    t1 = @elapsed o1 = TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                               extract_on_save=true)
    t2 = @elapsed o2 = TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                               extract_on_save=false)
    # NB `@printf` reads its format at MACRO-EXPANSION time, so it must be a single
    # string literal — a `"a" * "b"` concatenation is an expression and fails at include
    # time, taking the whole script with it. Split long messages into separate calls.
    @printf("   on-save %.1f s   save-the-stack %.1f s\n", t1, t2)
    @printf("   (the difference is the %d × %.2f GiB of temp-file traffic the on-save\n",
            length(c.zsave), field_gib(Nω, c.N))
    println("    route does not do)")
    for k in keys(o1)
        k === :zsave && continue
        r = maximum(abs.(o1[k] .- o2[k])) / maximum(abs, o2[k])
        @printf("   %-22s %10.3e\n", k, r)
        ok &= r < 1e-10
    end

    println("\n2. solver convergence at this shape: rtol $(SOLVER.rtol) vs 1e-8")
    t3 = @elapsed o3 = TS.simulate_delay_point(setup, τ; zsave=c.zsave,
                                               init_dz=SOLVER.init_dz, rtol=1e-8,
                                               max_dz=SOLVER.max_dz,
                                               twin_period=SOLVER.twin_period)
    @printf("   rtol 1e-8 took %.1f s (%.2f× the production run)\n", t3, t3/t1)
    for k in keys(o1)
        k === :zsave && continue
        r = maximum(abs.(o1[k] .- o3[k])) / maximum(abs, o3[k])
        rz = maximum(abs.(o1[k][:, end] .- o3[k][:, end])) / maximum(abs, o3[k][:, end])
        @printf("   %-22s %10.3e   (final z-slice %10.3e)\n", k, r, rz)
    end
    println("   The campaign's own pass criterion is every z-slice of Iω_win within")
    println("   1e-3 relative of an rtol=1e-8 run. A wing point is the hard case:")
    println("   `weaknorm` measures error against the pump-dominated whole field, so")
    println("   the weak signal's own relative error is ~rtol × ‖pump‖/‖signal‖.")

    if opt("cpu", "0") == "1"
        println("\n3. host reference (requested with --cpu=1; slow)")
        sh = TS.build_setup(; sa...)
        th = @elapsed oh = TS.simulate_delay_point(sh, τ; zsave=c.zsave, SOLVER...)
        @printf("   cpu %.1f s on %d threads (%.1f× the GPU)\n", th, Threads.nthreads(), th/t1)
        for k in keys(oh)
            k === :zsave && continue
            @printf("   %-22s %10.3e\n", k,
                    maximum(abs.(o1[k] .- oh[k])) / maximum(abs, oh[k]))
        end
    end

    @printf("\n   %s\n", ok ? "PASS (extraction routes agree to < 1e-10)" :
                              "CHECK — the two extraction routes disagree")
    return ok
end

# ============================================================================ #
#  BENCH — GPU delay points at production shapes
# ============================================================================ #
const SUMMARY = Tuple{String,Float64,Float64}[]

function run_bench()
    println("\n=== bench: GPU delay points at production shapes ===")
    isempty(OUT) || csv_row(OUT, ["case","N","Nomega","field_GiB","point","wall_s",
                                  "device_GiB","host_GiB","extract_on_save"])
    # Warm up FIRST. CUDA kernels compile per TYPE, not per size, so a tiny propagation
    # compiles everything the real cases need for a couple of seconds. Without this the
    # first case carries ~11 s of compilation and reads as 2-3x slower than it is —
    # which is exactly what happened on the first H200 run.
    print("    warm-up (compiles the device kernels): ")
    let wa = setup_args_for(gap_mm=1.0, N=32, trange=110e-15, thickness=40e-6)
        tw = @elapsed begin
            ws = TS.build_setup(; wa..., arraytype=CUDA.CuArray)
            TS.simulate_delay_point(ws, 0.0; zsave=[0.0, 40e-6], SOLVER...)
        end
        @printf("%.1f s\n", tw)
    end
    GC.gc(); Luna.device_reclaim()
    for name in CASE_NAMES
        haskey(CASES, name) || (println("unknown case $name, skipping"); continue)
        c = CASES[name]
        sa = setup_args_for(gap_mm=c.gap_mm, N=c.N, trange=c.trange, thickness=c.thickness)
        Nω = round(Int, 0)  # filled in after the setup exists
        println("\n--- $name: $(c.note)")
        GC.gc(); Luna.device_reclaim()
        free0 = devfree()
        tsetup = @elapsed setup = TS.build_setup(; sa..., arraytype=CUDA.CuArray)
        Nω = length(setup.grid.ω)
        @printf("    grid (%d, %d, %d)  field %.2f GiB  setup %.1f s  device after setup %.1f GiB\n",
                Nω, c.N, c.N, field_gib(Nω, c.N), tsetup, free0 - devfree())

        for ip in 1:POINTS
            # A spread of delays: the far wing and near zero differ in step count,
            # and the campaign cost is the average over the scan, not the best case.
            τ = (ip == 1) ? 0.0 : (-25e-15 + 50e-15 * (ip - 1) / max(POINTS - 1, 1))
            GC.gc(); Luna.device_reclaim()
            f0 = devfree()
            t = @elapsed TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                                 extract_on_save=(EXTRACT != "off"))
            dev = f0 - devfree()
            host = Sys.maxrss()/2^30
            # Cost per GiB of state is the saturation signal: a delay point's work is
            # proportional to the field size at fixed step count, so if the card is
            # saturated this is roughly CONSTANT across cases whose fields differ by
            # 2.6×. If it falls as the field grows, the smaller cases are not filling
            # the card and there is headroom to exploit (see the suite's `share` step).
            @printf("    point %d  τ %+7.2f fs   wall %7.1f s   device %5.1f GiB   host %5.1f GiB   %6.2f s/GiB\n",
                    ip, τ*1e15, t, dev, host, t/field_gib(Nω, c.N))
            push!(SUMMARY, (name, field_gib(Nω, c.N), t))
            csv_row(OUT, [name, c.N, Nω, gib(field_gib(Nω, c.N)), ip, round(t; digits=2),
                          gib(dev), gib(host), EXTRACT != "off"])
        end
        setup = nothing; GC.gc(); Luna.device_reclaim()
    end
    println("\n  Device memory should be ≈ 10 × the field size (9 solver registers + 1")
    println("  transform buffer); more means the cuFFT workspace is not negligible at")
    println("  that shape, which changes how many points fit on the card at once.")
    if length(SUMMARY) > 1
        println("\n  saturation check — cost per GiB of state across cases:")
        println("    case       field GiB    wall s    s/GiB     scan (200 pts)")
        for (nm, f, t) in SUMMARY
            @printf("    %-10s %8.2f %9.1f %8.2f     %5.1f min\n", nm, f, t, t/f, 200t/60)
        end
        v = [t/f for (_, f, t) in SUMMARY]
        spread = maximum(v) / minimum(v)
        @printf("    s/GiB spread %.2f× — %s\n", spread,
                spread < 1.3 ? "flat: the card is saturated at every shape, so cost is pure\n" *
                               "                       traffic and running points concurrently will not help" :
                               "NOT flat: the smaller shapes are leaving the card idle;\n" *
                               "                       there is headroom, run the suite's `share` step")
    end
end

# ---------------------------------------------------------------------- main --
banner()
println("mode=$MODE  points=$POINTS  extract=$EXTRACT  started $(Dates.now())")
if MODE == "accuracy"
    run_accuracy() || exit(1)
elseif MODE == "bench"
    run_bench()
else
    error("unknown --mode=$MODE (accuracy|bench)")
end
println("\ndone $(Dates.now())")
