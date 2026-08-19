# =============================================================================
# 11: the Gaussian-beam d/D series, on one H100/H200.
#
# Mirrors FROG_paper_new/11_gaussian_dd_series.jl. Every physical and solver
# parameter is copied from it; only the execution differs (one process, one
# card, `Scans.LocalExec`, resume). The Gaussian beam model and the two-window
# collection pattern both already run on the device — verified host-vs-device to
# ~2e-13 on all five trace datasets — so nothing new was needed to support them.
#
# TWO MODES
#   MODE=scan       run the delay scan (default)
#   MODE=gridcheck  one delay point at a ladder of (R, N), reporting convergence.
#                   Run this BEFORE trusting GRID=lean. ~1 min on an H200.
#
# TWO GRIDS
#   GRID=pinned  (default) R = 366 µm, N per the CONFIG table — identical to the
#                CPU script, so results are directly comparable with the HE11
#                d/D series at the same grid.
#   GRID=lean    R = 189 µm, N = 200/360/400. **6-10x less memory and work.**
#
#     Why lean is possible: `optimal_spatial_grid` sizes R as 5 Airy radii of the
#     1 mm mask hole at 600 nm — i.e. for APERTURE DIFFRACTION at the focus. The
#     Gaussian model has no aperture and therefore no Airy rings: the field is a
#     w0 = 16.55 µm Gaussian, and R = 366 µm holds 24 w0 where ~12 is ample.
#     R cannot shrink freely, because R also sets Δk = π/R(1+wf) and Δk must
#     resolve the k-space structure; 189 µm is where that binds (16 samples
#     across the Gaussian's 2/w0 k-width). N then follows from needing
#     1.5 x 3 x Δk(λmin) of k-space, with Δk from the beam CENTRE d_hole rather
#     than the aperture edge — which is itself a 1.33x saving over the HE11 rule.
#
#     Why to check it first: lean roughly HALVES the k-space resolution of the
#     collection window (the PlanckOmegaWindow hole radius goes from ~14 to ~7
#     samples at λmin). Window-edge sampling is the known weak point of this
#     model, so `Iω_win` is the dataset to watch, not `Iω_full`.
#
# RUN (see h200_production_05_100um.sh for the environment; same prerequisites):
#   source /workspace/env.sh
#   cd /workspace/runs && mkdir -p prod11 && cd prod11
#   tmux new -s prod11
#   MODE=gridcheck GAP_MM=2.0 julia --project=/workspace/code/dev \
#       /workspace/code/ModelPNPS.jl/examples/h200_gaussian_dd_series.jl
#   GAP_MM=2.0 GRID=lean julia --project=/workspace/code/dev \
#       /workspace/code/ModelPNPS.jl/examples/h200_gaussian_dd_series.jl 2>&1 | tee prod11.log
#
# Re-running in the same directory resumes.
# Env: GAP_MM (0.5|1.5|2.0), MODE, GRID, PROD_POINTS (default 200), PROD_NAME.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import CUDA
import Printf: @printf
import Dates

Luna.set_fftw_mode(:measure)
Luna.set_fftw_threads(2)

GAP_MM = parse(Float64, get(ENV, "GAP_MM", "2.0"))
MODE   = get(ENV, "MODE", "scan")
GRID   = get(ENV, "GRID", "pinned")
NPTS   = parse(Int, get(ENV, "PROD_POINTS", "200"))

# N per gap: the CPU script's CONFIG (pinned), and the Gaussian-sized alternative.
const CONFIG = Dict(0.5 => (Npin=640,  Nlean=200),
                    1.5 => (Npin=900,  Nlean=360),
                    2.0 => (Npin=1024, Nlean=400))
haskey(CONFIG, GAP_MM) || error("GAP_MM must be one of $(sort(collect(keys(CONFIG))))")
const CFG = CONFIG[GAP_MM]

const TWIN_SAVES_ONLY = 1_000_000_000

# ---------------------------------------------------------------- parameters --
λ0           = 260e-9
τfwhm        = 1.0e-15
energy       = 3.0e-9     # PERTURBATIVE: B ≈ 0.025 rad over 40 µm, as in 03 v2
material     = :SiO2
thickness    = 40e-6
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = GAP_MM * 1e-3
λlims        = (143e-9, 600e-9)

w0        = λ0 * f_foc / (π * mask_diam/2)          # ≈ 16.55 µm, ω-independent
d_hole    = mask_spacing/2 + mask_diam/2            # axis -> hole centre
crossingθ = d_hole / f_foc
Δk        = 2π / λ0 * sin(crossingθ)

beam = TS.GaussianBeam(w0, f_foc)
windows = [
    TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/w0, pad=1.25),
    TS.PlanckOmegaWindow(xc=-d_hole, yc=-d_hole,
                          holediam=mask_diam/2, f_foc=f_foc, pad=1.25),
]

grid_for(g) = g == "lean" ? (189.0e-6, CFG.Nlean) : (366.0e-6, CFG.Npin)

function args_for(g)
    R, N = grid_for(g)
    (; λ0, τfwhm, energy, thickness, material,
       mask_diam, mask_spacing, λlims,
       beam, window=windows,
       trange=110e-15, store_window=false,
       R=R, N=N, arraytype=CUDA.CuArray)
end

τ = collect(range(-25e-15, 25e-15, NPTS))
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6
const SOLVER = (; init_dz=5e-7, rtol=1e-7, max_dz=2e-6, twin_period=TWIN_SAVES_ONLY)

field_gib(Nω, N) = 16*Nω*N*N/2^30
dv = CUDA.device()
@printf("11 Gaussian d/D series — gap %.1f mm, w0 %.2f µm, crossing %.1f mrad\n",
        GAP_MM, w0*1e6, crossingθ*1e3)
@printf("device %s  %.1f GiB   mode=%s  grid=%s\n",
        CUDA.name(dv), CUDA.total_memory()/2^30, MODE, GRID)

# ============================================================================ #
if MODE == "gridcheck"
    # A convergence ladder, not a benchmark: does the lean grid change the
    # answer? Compare against the pinned grid at the SAME delay. Iω_win is the
    # dataset that matters — it is the one the collection window touches, and
    # window-edge sampling is what lean gives up.
    println("\n=== grid convergence: one delay point per (R, N) ===")
    τc = -6e-15   # off centre: a wing point is the harder case for the window
    ref = nothing
    println("R [µm]   N     field GiB   wall s   device GiB   max rel diff vs pinned")
    for g in ("pinned", "lean")
        R, N = grid_for(g)
        GC.gc(); Luna.device_reclaim()
        f0 = Luna.device_memory_status()[1]
        setup = TS.build_setup(; args_for(g)...)
        Nω = length(setup.grid.ω)
        t = @elapsed out = TS.simulate_delay_point(setup, τc; zsave=zsave, SOLVER...)
        dev = (f0 - Luna.device_memory_status()[1])/2^30
        # k-integrated datasets are in FFT-bin units scaling as N^4 at fixed R;
        # and R differs too, so compare only the N-invariant re-imaged spectra.
        d = isnothing(ref) ? 0.0 :
            maximum(abs.(out.Iω_win_reimaged .- ref)) / maximum(abs, ref)
        isnothing(ref) && (ref = out.Iω_win_reimaged)
        @printf("%-8.0f %-5d %-11.2f %-8.1f %-12.1f %s\n",
                R*1e6, N, field_gib(Nω, N), t, dev,
                g == "pinned" ? "(reference)" : @sprintf("%.3e", d))
        setup = nothing; GC.gc(); Luna.device_reclaim()
    end
    println("\nJudge on Iω_win_reimaged (N-invariant; the k-integrated datasets are in")
    println("FFT-bin units that scale as N^4 and are not directly comparable across grids).")
    println("The campaign acceptance elsewhere is <1e-3 relative. If lean exceeds that,")
    println("use GRID=pinned — the saving is not worth a grid-dependent trace.")
    exit(0)
end

# ============================================================================ #
NAME = get(ENV, "PROD_NAME",
           "tgfrog_gaussian_lowE_v2_rtol7_sw_gap$(Int(GAP_MM*1000))um_1fs_40umUVFS" *
           (GRID == "lean" ? "_lean" : ""))
R, N = grid_for(GRID)
@printf("grid R=%.0f µm N=%d, %d z-slices, %d delay points ±25 fs\n",
        R*1e6, N, length(zsave), NPTS)
println("scan name: $NAME")
println("started $(Dates.now())\n"); flush(stdout)

t0 = time()
TS.run_scan(args_for(GRID), τ; scan_name=NAME, exec=Scans.LocalExec(),
            zsave=zsave, SOLVER...,
            fftw_threads=2, fftw_mode=:measure,
            skip_existing=true)
wall = time() - t0

ndone = length(TS._completed_scanidcs(NAME))
@printf("\nscan wall %.1f s (%.2f h) — %d/%d points complete\n", wall, wall/3600, ndone, NPTS)
ndone < NPTS && println("INCOMPLETE — rerun in the same directory to resume")
println("collected: $(NAME)_collected.h5")
println("done $(Dates.now())")
