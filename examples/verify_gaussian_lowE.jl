# =============================================================================
# A/B verification of the perf-branch stack against the running/completed
# tgfrog_gaussian_lowE_rtol8_gap1000um_1fs_40umUVFS reference scan
# (03_gaussian_control_lowE.jl, old code path).
#
# Recomputes selected delay points with the CURRENT code and compares every
# trace dataset against the collected file. Reference points that a running
# scan has not yet computed are skipped (reported as NaN).
#
# Usage:
#   julia [-t 8] --project=<env> verify_gaussian_lowE.jl \
#         <collected.h5> <idx1,idx2,...> [N] [fftw_threads]
#
#   N            transverse grid size (default 1024 = the reference grid;
#                pass 640 for the grid-change A/B)
#   fftw_threads default 5 = the reference scan's value. Keep it at 5 for the
#                strict same-grid comparison (FFT algorithm choice affects
#                round-off); -t / JULIA_NUM_THREADS does NOT affect results
#                and can be raised freely for speed.
#
# CLUSTER ISOLATION (avoid contaminating a running old-code scan): the running
# scan's job script resolves Luna/ModelPNPS through the dev paths recorded in
# the Manifest of the project it was submitted from (~/.julia/dev/...). Do NOT
# update those checkouts while it runs. Instead:
#
#   git clone -b perf <luna-fork>      ~/perfstack/Luna
#   git clone -b perf <modelpnps-repo> ~/perfstack/ModelPNPS
#   julia -e 'using Pkg; Pkg.activate("~/perfstack/env");
#             Pkg.develop(path="~/perfstack/Luna");
#             Pkg.develop(path="~/perfstack/ModelPNPS")'
#
# then run this script with --project=~/perfstack/env. Read the collected file
# from a COPY (`cp` it first) — a running scan writes to it between points.
#
# Example batch job (strict same-grid A/B on 3 completed points):
#   sbatch --ntasks=1 --cpus-per-task=8 --mem=40G --time=12:00:00 --wrap \
#     'cp .../tgfrog_gaussian_lowE..._collected.h5 ref.h5 && \
#      JULIA_NUM_THREADS=8 julia --project=$HOME/perfstack/env \
#        $HOME/perfstack/ModelPNPS/examples/verify_gaussian_lowE.jl ref.h5 10,55,100'
#
# Expected results:
#   N=1024, fftw_threads=5 : ≲1e-10 relative on every dataset (propagation is
#                            bit-identical; extraction and any wisdom drift
#                            contribute only rounding-level differences)
#   N=640                  : <1e-3 relative on Iω_win (grid-convergence check;
#                            acceptance for production use is <0.1%)
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna

length(ARGS) >= 2 || error("usage: verify_gaussian_lowE.jl <collected.h5> <idx1,idx2,...> [N] [fftw_threads]")
collected    = ARGS[1]
scanidcs     = parse.(Int, split(ARGS[2], ","))
N            = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1024
fftw_threads = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5

Luna.set_fftw_mode(:estimate)   # matches the reference scan
Luna.set_fftw_threads(fftw_threads)

# --- Setup arguments: EXACT copy of 03_gaussian_control_lowE.jl --------------
λ0           = 260e-9
τfwhm        = 1.0e-15
energy       = 3.0e-9
material     = :SiO2
thickness    = 40e-6
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = 1.0e-3
λlims        = (143e-9, 600e-9)

w0        = λ0 * f_foc / (π * mask_diam/2)
d_hole    = mask_spacing/2 + mask_diam/2
crossingθ = d_hole / f_foc
Δk        = 2π / λ0 * sin(crossingθ)

beam = TS.GaussianBeam(w0, f_foc)
windows = [
    TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/w0, pad=1.25),
    TS.PlanckOmegaWindow(xc=-d_hole, yc=-d_hole,
                          holediam=mask_diam/2, f_foc=f_foc, pad=1.25),
]

setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims,
                beam, window=windows,
                R=366.0e-6, N=N)

zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

# --- Run ----------------------------------------------------------------------
results = TS.verify_against_collected(setup_args, collected, scanidcs;
                                      zsave=zsave, init_dz=5e-7,
                                      rtol=1e-8, max_dz=2e-6)

println("\n==== SUMMARY (N=$N, fftw_threads=$fftw_threads, ",
        "JULIA_NUM_THREADS=$(Threads.nthreads())) ====")
for point in results
    println("scanidx $(point["scanidx"])  τ = $(point["τ"]*1e15) fs  ",
            "wall = $(round(point["wall_s"]; digits=1)) s  ",
            "maxrss = $(round(point["maxrss_GiB"]; digits=1)) GiB")
    for k in sort(collect(keys(point)))
        startswith(k, "Iω") && println("    $(rpad(k, 24)) max rel diff = $(point[k])")
    end
end
