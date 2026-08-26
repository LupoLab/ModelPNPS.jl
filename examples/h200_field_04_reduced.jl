# =============================================================================
# 04, field-resolved, on one H200: 16 delays instead of 230.
#
# WHAT THIS IS
#   `04_production_gap1000_kerr_raman.jl` (the Kerr arm) run on Luna's REAL,
#   carrier-resolved field instead of the envelope. Every physical and solver
#   parameter is copied from that script — 1 fs, 0.1 µJ, 260 nm into 40 µm of SiO2
#   through the gap-1.0 mm boxcars mask, N = 768, R = 366 µm, λlims 143-600 nm,
#   trange 110 fs, rtol 1e-7, max_dz 2 µm, apodisation at the z-saves only, and the
#   identical 16-slice zsave ladder. Four things differ, all of them deliberate:
#
#     field_mode = true    Grid.RealGrid, so no envelope/carrier split, no dropped
#                          third-harmonic term, no negative-frequency wrap.
#     response = :nothg    (3/4)ε₀χ³|E_a|²E — the SAME physics content as the
#                          envelope's Kerr_env, so the envelope-versus-field
#                          difference isolates REPRESENTATION error and nothing
#                          else. `:thg` (plain ε₀χ³E³) is the other experiment; the
#                          difference between the two is what the envelope's
#                          RESPONSE omits, and confusing the two questions is the
#                          one thing this run exists to avoid.
#     arraytype = :cuda    one card, points in sequence (Scans.LocalExec).
#     16 delays            below. 230 would be ~5 h; this is ~25 min.
#
#   Raman is not available in field mode and 04's Kerr arm does not use it, so the
#   RAMAN toggle is simply absent here.
#
# THE DELAYS
#   Given explicitly, and they are production τ-grid points: the 04 axis is a
#   200-point ±25 fs core (0.251256 fs step) plus 1 fs wings to ±40 fs, and these 16
#   sit at core indices 115∓{0,5,10,15,20,25,30,40,50} — symmetric about the centre,
#   fine near zero and coarsening outwards. The values below are those points
#   rounded to 7 significant figures, so the script RECONSTRUCTS the production axis
#   and snaps to it, asserting first that every requested value is within 1e-19 s of
#   a grid point. That keeps each delay bit-identical to the reference file's, so
#   the comparison needs no interpolation in τ; and if the assumption is ever wrong
#   the assertion fails loudly instead of quietly moving a delay.
#
# COST AND FIT (measured on this card, 2026-08-26)
#   Device 96.9 GiB of 139.8 — the model in `TS.memory_budget` predicted 96.9, and
#   18.0 GiB of it (the analytic signal) is allocated on the FIRST RHS, not at setup.
#   Per point 67 s at τ = 0 and 104 s at τ = -8 fs (57 and 71 steps), i.e. ~6× the
#   envelope per step. These 16 delays reach ±12.7 fs, so expect ~25 minutes.
#
#   This is H200-only as configured. `:thg` or `ffac = 4` would each cut it to
#   ~70 GiB and fit an 80 GB card — but neither is the controlled comparison, and
#   `ffac = 4` additionally changes δω and the realised time window, so a ffac-4
#   trace is not directly comparable with the delivered envelope files.
#
# HOW TO RUN IT
#   bash /workspace/code/Luna.jl/test/manual/runpodcoldstart.sh
#   source /workspace/env.sh
#   julia /workspace/code/ModelPNPS.jl/examples/h200_field_04_reduced.jl 2>&1 | tee run.log
#
#   `source /workspace/env.sh` is not optional: it is what puts julia on PATH and
#   the depot on the volume. The script activates /workspace/code/dev itself if the
#   active project does not already have ModelPNPS, so no --project is needed.
#
#   PNPS_DRYRUN=1 validates the configuration, prints the grid, the memory budget
#   and the delay table, and stops before propagating. Do that first — it is
#   seconds, and it is the cheap way to find out that the shape does not fit.
#
#   Re-running in the same directory RESUMES: `scansave` fills the collected file
#   point by point and completed points are skipped.
#
# Environment: PNPS_DRYRUN, FIELD_RUNDIR (default /workspace/runs/field04),
# FIELD_ARRAYTYPE (cuda|cpu), FIELD_NAME.
# =============================================================================

import Pkg

# Run as `julia script.jl` with no --project: adopt the shared dev environment, but
# never override a project the caller chose deliberately.
const DEV = get(ENV, "LUNA_DEV", "/workspace/code/dev")
# Adopt the pod's shared environment when the caller gave no --project, and never
# override one they did give. "No --project" is exactly "the active project is the
# default vN.N environment" — testing instead for the presence of ModelPNPS would
# silently accept a stale copy sitting in someone's default environment, which is a
# confusing way to fail (it loads, then breaks inside Luna).
let dflt = try Base.load_path_expand("@v#.#") catch; nothing end
    if Base.active_project() == dflt
        isdir(DEV) || error(
            "no --project given and $DEV does not exist. Either run " *
            "runpodcoldstart.sh (which creates it), set LUNA_DEV, or pass " *
            "--project pointing at an environment with ModelPNPS.")
        Pkg.activate(DEV; io=devnull)
    end
end

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import Printf: @printf
import Dates

# :estimate, NOT 04's :measure. 04 depends on FFTW wisdom pre-generated for the
# ENVELOPE shape (Nω = 256); the field grid is Nω = 513 with a 2048-point fine grid,
# for which no wisdom exists, and :measure would spend a long time planning 3-D
# transforms that the device path barely uses anyway.
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(Threads.nthreads())

# Apodisation only at the z-saves. Per-step windowing makes the damping scale with
# the step count, so the integrator does not converge in its own rtol (04's header
# documents the defect; 90c_apodisation_test.jl measured it).
const TWIN_SAVES_ONLY = 1_000_000_000

# ---------------------------------------------------------------- 04 parameters --
# Copied from 04_production_gap1000_kerr_raman.jl. Do not change one without
# changing it there: the whole point is that the ONLY difference is the
# field-versus-envelope representation.
const GAP    = 1.0e-3
λ0           = 260e-9
τfwhm        = 1.0e-15
energy       = 0.1e-6
material     = :SiO2
thickness    = 40e-6
a            = 125e-6
f_coll       = 5.0
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = GAP
λlims        = (143e-9, 600e-9)
d            = mask_spacing/2 + mask_diam/2

const R_GRID = 366.0e-6
const N_GRID = 768          # = optimal_spatial_grid for this geometry; pinned

beam   = TS.HE11Beam(a, f_coll, f_foc)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                               zmask=f_foc, apod=:tanh)

ATYPE = get(ENV, "FIELD_ARRAYTYPE", "cuda")

# apod: input holes :supergauss(16), collection window :tanh — deliberate, see 04's
# header (the 1.0 mm input holes are resolved 2× better than the 0.5 mm window hole).
setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod=:supergauss, apod_param=16,
                trange=110e-15, store_window=false,
                R=R_GRID, N=N_GRID,
                field_mode=true, response=:nothg, ffac=6,
                arraytype = ATYPE == "cuda" ? :cuda : Array,
                # Two fewer resident device fields (9 GiB here), traded for one
                # host-to-device transfer per point — a fraction of a second against
                # a propagation of minutes.
                beamlets_on_host=true)

# ------------------------------------------------------------------- delays -----
# As requested. See the header: these are 04 τ-grid points to 7 s.f., and are snapped
# back onto the exact grid so each is bit-identical to the reference file's.
const Τ_REQUESTED = [-1.268844e-14, -1.017588e-14, -7.663317e-15, -5.150754e-15,
                     -3.894472e-15, -2.638191e-15, -1.381910e-15, -1.256281e-16,
                      1.256281e-16,  1.381910e-15,  2.638191e-15,  3.894472e-15,
                      5.150754e-15,  7.663317e-15,  1.017588e-14,  1.268844e-14]

# 04's axis, verbatim.
const Τ_CORE = collect(range(-25e-15, 25e-15, 200))
const Τ_WING = collect(26e-15:1e-15:40e-15)
const Τ_04   = sort(vcat(-reverse(Τ_WING), Τ_CORE, Τ_WING))   # 230 points

const SNAP_TOL = 1e-19      # ~1e-4 of the 0.251 fs core step
τ = map(Τ_REQUESTED) do t
    i = argmin(abs.(Τ_04 .- t))
    abs(Τ_04[i] - t) < SNAP_TOL || error(
        "requested τ = $t is $(abs(Τ_04[i]-t)) s from the nearest 04 grid point " *
        "($(Τ_04[i])); it is not a production delay, so snapping would silently " *
        "move it. Fix the value or drop the snap.")
    Τ_04[i]
end
issorted(τ) || error("delays must be ascending")

# 04's ladder, verbatim: dense at the L_D = 1.7 µm generation-zone scale, the
# experimental 9.5 µm included exactly, legacy 4-40 µm retained.
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

NAME = get(ENV, "FIELD_NAME",
           "tgfrog_field_nothg_rtol7_sw_gap1000um_tanh_1fs_40umUVFS_16pt")

# ---------------------------------------------------------------- run dir -------
RUNDIR = get(ENV, "FIELD_RUNDIR", "/workspace/runs/field04")
mkpath(RUNDIR)
cd(RUNDIR)

# ------------------------------------------------------------------- report -----
bu = TS.memory_budget(setup_args)

@printf("04 FIELD-RESOLVED, reduced — %s\n", NAME)
@printf("  %.1f fs, %.2f µJ, %d µm %s, gap %.1f mm, response :nothg, ffac 6\n",
        τfwhm*1e15, energy*1e6, round(Int, thickness*1e6), material, GAP*1e3)
@printf("  grid: Nω %d, Nt %d, Nto %d (%s), transverse %d×%d, R %.0f µm, trange %.0f fs\n",
        bu.Nω, bu.Nt, bu.Nto, bu.Nto == bu.Nt ? "not oversampled" : "$(bu.Nto ÷ bu.Nt)×",
        N_GRID, N_GRID, R_GRID*1e6, 110.0)
@printf("        (04's envelope grid for comparison is Nω = 256)\n")
@printf("  solver: rtol 1e-7, max_dz 2 µm, init_dz 0.5 µm, apodisation at the saves only\n")
@printf("  %d delays, %d z-slices, run dir %s\n", length(τ), length(zsave), RUNDIR)
@printf("  memory: state %.1f + Et_win %.1f + Eto %.1f + Eωo %.1f + Pto %.1f",
        bu.state, bu.et_win, bu.eto, bu.ewo, bu.pto)
@printf(" + analytic %.1f + window %.1f + input %.1f = %.1f GiB device\n",
        bu.analytic, bu.window, bu.input, bu.device)
@printf("          host peak in build_setup %.1f GiB; %.1f GiB of the device figure\n",
        bu.host, bu.analytic)
@printf("          (the analytic signal) appears on the FIRST RHS, not at setup\n")

println("\n  delays (snapped to the 04 axis):")
for (k, (treq, t)) in enumerate(zip(Τ_REQUESTED, τ))
    i = argmin(abs.(Τ_04 .- t))
    @printf("    %2d  requested %+12.6f fs   04 grid index %3d  %+18.12f fs   Δ %.1e fs\n",
            k, treq*1e15, i, t*1e15, abs(t-treq)*1e15)
end

@printf("\n  arraytype %s   julia %s   threads %d   host %s\n",
        ATYPE, VERSION, Threads.nthreads(), gethostname())
flush(stdout)

if ATYPE == "cuda"
    # Resolve (and load) the GPU package the way build_setup will, then ask the card
    # how much room there actually is.
    Luna.resolve_arraytype(:cuda)
    st = Luna.device_memory_status()
    if isnothing(st)
        println("  WARNING: could not read device memory")
    else
        free, total = st ./ 2^30
        @printf("  device: %.1f GiB free of %.1f GiB\n", free, total)
        if bu.device > 0.9*free
            @printf("\nREFUSING TO START: needs ~%.1f GiB, only %.1f GiB free.\n",
                    bu.device, free)
            println("  This shape is H200-only as configured. The ways down, in order:")
            println("    * response = :thg   — pointwise, so no Pto and no analytic buffer")
            println("    * ffac = 4          — halves the fine grid, but CHANGES δω and the")
            println("                          realised time window, so the result is not")
            println("                          directly comparable with the envelope files")
            println("    * a smaller N       — then the envelope reference must be recomputed")
            println("                          at the same N for the comparison to mean anything")
            exit(1)
        end
    end
    # 67 s at τ = 0 and 104 s at τ = -8 fs were measured on this card; these delays
    # reach ±12.7 fs, so quote the range rather than a single number.
    @printf("  estimate: %.0f-%.0f min for %d points (67-104 s each, measured)\n",
            length(τ)*67/60, length(τ)*104/60, length(τ))
end

if get(ENV, "PNPS_DRYRUN", "0") == "1"
    @info "dry run: configuration validated, NOT propagating" scan_name=NAME delays=length(τ)
    exit(0)
end

# --------------------------------------------------------------------- run ------
println("\nstarted $(Dates.now())")
flush(stdout)
t0 = time()
TS.run_scan(setup_args, τ; scan_name=NAME, exec=Scans.LocalExec(),
            zsave=zsave, init_dz=5e-7, rtol=1e-7, max_dz=2e-6,
            twin_period=TWIN_SAVES_ONLY,
            fftw_threads=Threads.nthreads(), fftw_mode=:estimate,
            skip_existing=true)
wall = time() - t0

ndone = length(TS._completed_scanidcs(NAME))
@printf("\nscan wall %.1f s (%.2f h) — %d/%d points complete, %.0f s/point\n",
        wall, wall/3600, ndone, length(τ), wall/max(ndone, 1))
ndone < length(τ) && println("INCOMPLETE — rerun in this directory to resume")
@printf("collected: %s\n", joinpath(RUNDIR, NAME * "_collected.h5"))
println("done $(Dates.now())")

# NEXT
#   The file carries /grid/field_mode = 1, /grid/response = "nothg" and /grid/ffac,
#   and its /grid/ω is a MONOTONIC rfft half-spectrum — not the envelope files'
#   FFT-ordered relative-frequency axis. `TS.load_simulated_scan` reads the marker
#   and skips the fftshift; anything else reading it must do the same.
#
#   `verify_against_collected` CANNOT compare this against the envelope 04 file:
#   Nω is 513 against 256 and the axes differ, so it will refuse on the size
#   mismatch. The comparison is between physical spectral densities on the common
#   band — |E|² divided by Δω², splined onto one axis — reported as rms difference
#   relative to trace peak at matched (τ, z), PER DEPTH. How any difference GROWS
#   with depth is the quantity of interest, because the unexplained retrieval
#   residual does exactly that. The recipe is in ModelPNPS's
#   "field mode: field-versus-envelope trace agreement" testset.
