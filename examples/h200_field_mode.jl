# =============================================================================
# Field-resolved (RealGrid) TG-FROG delay points on one H100/H200.
#
# THE QUESTION THIS ANSWERS
#   Retrieving the simulated 1 fs traces leaves a model error ~5e-3 of trace peak, ten
#   times what the same pipeline achieves at 2 fs, and eight candidate explanations have
#   been measured and eliminated. At 260 nm a 1 fs pulse is 1.15 optical cycles, so the
#   remaining suspect is the envelope/carrier split itself: the production envelope grid
#   (λlims = (143, 600) nm, trange = 110 fs) has its relative-frequency window's red edge
#   at exactly DC and uses 253.7 of its 256 samples.
#
#   A field-resolved run has no carrier/envelope split, no dropped third-harmonic term and
#   no negative-frequency wrap. If a field-mode trace matches the envelope one, the
#   envelope model is vindicated and the residual belongs to the retrieval code. If it
#   differs, the 1 fs simulation data is unsafe.
#
#   DO NOT TUNE ANYTHING TO MAKE THE TWO AGREE. A disagreement is a result.
#
# THE ARMS, set by the consts below.
#   CONFIG = :port2fs   2 fs, RESPONSE = :nothg.  The PORT CHECK. At 2.31 cycles the
#                       envelope is sound (croak's independent 1-D formulation reproduces
#                       that trace to 2.7e-4 of peak), so the two MUST agree. A
#                       disagreement here is a port bug, not physics — fix it before
#                       running anything else.
#   CONFIG = :expt1fs   1 fs, RESPONSE = :nothg.  The experiment. `:nothg` is
#                       (3/4)ε₀χ³|E_a|²E, the same physics content as the envelope's
#                       Kerr_env, so this isolates REPRESENTATION error with nothing else
#                       changed.
#   CONFIG = :expt1fs   1 fs, RESPONSE = :thg.    Plain ε₀χ³E³, which adds the
#                       third-harmonic and counter-rotating terms the envelope drops. The
#                       difference between the two 1 fs runs is exactly what the envelope
#                       omits. (At λlims = (143, 600) nm the third harmonic itself is
#                       discarded by the crop; the within-band counter-rotating terms are
#                       what survive. Propagating real THG would need λlims to ~λ0/3,
#                       which quadruples Nω.)
#
# WHAT IT COSTS, and why this is a GPU script rather than an HPC one.
#   Per delay point at N = 768, RESPONSE = :nothg, FFAC = 6:
#     state (9 RK45 registers)  40.6 GiB      Eto                       9.0 GiB
#     window scratch Et_win      4.5 GiB      Eωo (Pωo aliases it)      9.0 GiB
#     Pto                        9.0 GiB      analytic-signal buffer   18.0 GiB
#                                                              total  ~90 GiB
#   plus the extraction window (2.25 GiB) — call it ~95 GiB resident. With RESPONSE = :thg
#   the response is pointwise, so Pto aliases Eto and the analytic buffer disappears:
#   ~63 GiB. FFAC = 4 (see below) takes :nothg to ~68 GiB.
#
#   So: an A40 (44 GiB) cannot run this at all, an 80 GB H100 fits :thg and :nothg-at-
#   FFAC-4, and only an H200 (141 GiB) fits everything.
#
#   Time: measured 3.0x the envelope per delay point at N = 64 and 3.3x at N = 128, at
#   MATCHED step counts (9 vs 9, 8 vs 8) — so it is a clean per-step ratio, not a
#   step-count artefact. Combined with the memory, a full 200-delay field-mode scan is on
#   the order of ten times the envelope campaign's resource. Take a SUBSET of delays ON the
#   production τ grid points instead, so each one compares directly with no interpolation
#   in τ; the comparison does not need all 200.
#
# FFAC — read before changing it.
#   Luna's RealGrid samples the nonlinear grid at 6x fmax, which is what E³ needs. The
#   no-THG response reaches only 2ωmax - ωmin, for which 4x suffices, and at this grid
#   that removes the oversampling entirely: half the memory and half the RHS cost. It is
#   validated (Luna test/test_full_freespace.jl "ffac convergence for the no-THG response"
#   measures 4e-8, unchanged from z=0 to z=end). But it CHANGES THE GRID — δω, the realised
#   time window and in general Nω — so a FFAC = 4 trace is not bin-for-bin comparable with
#   a FFAC = 6 one. Leave it at 6 for anything that will be compared against the envelope
#   files; use 4 only for a self-contained scan whose companion runs also use 4.
#
# HOW TO RUN IT — use the launcher, which sources /workspace/env.sh and logs:
#
#   bash /workspace/code/Luna.jl/test/manual/runpodcoldstart.sh    # if not yet, or to pull
#   tmux new -s fieldmode
#   bash /workspace/code/ModelPNPS.jl/examples/h200_field_mode.sh
#   # detach with C-b d; reattach with `tmux attach -t fieldmode`
#
# PNPS_DRYRUN=1 builds the setup, prints the grid and the device-memory budget, and stops
# without propagating. Do that first: it is seconds, and it is the only cheap way to find
# out that this shape does not fit the card.
#
# Re-running in the same directory resumes (`skip_existing=true`).
#
# Environment: PNPS_DRYRUN, FIELD_POINTS (default 11), FIELD_ARRAYTYPE (cuda|cpu),
# FIELD_NAME (scan name override).
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import Printf: @printf
import Dates

# --- THE TOGGLES. Consts, not environment variables, deliberately: these scripts are
# shipped to a remote host and re-run there, and an env-var switch would read correctly
# here and then silently take its default on the far side — submitting the wrong arm under
# a scan_name that agrees with itself. Same reasoning as 15_convergence_spectral.jl.
const CONFIG   = :port2fs      # :port2fs (the port check) | :expt1fs (the experiment)
const RESPONSE = :nothg        # :nothg (envelope-matched physics) | :thg (E³)
const FFAC     = 6             # see the header before changing

const ARMS = Dict(
    :port2fs => (τfwhm = 2.0e-15, thickness = 40e-6, τmax = 25e-15, tag = "2fs"),
    :expt1fs => (τfwhm = 1.0e-15, thickness = 40e-6, τmax = 25e-15, tag = "1fs"),
)
haskey(ARMS, CONFIG) || error("CONFIG must be one of $(keys(ARMS)); got $CONFIG")
const ARM = ARMS[CONFIG]

NPTS  = parse(Int, get(ENV, "FIELD_POINTS", "11"))
ATYPE = get(ENV, "FIELD_ARRAYTYPE", "cuda")

# :estimate, NOT :measure. No FFTW wisdom exists for the field-grid shapes (Nt = 1024 /
# Nto = 2048 against the envelope's 256), and :measure would either stall for a long time
# planning 3-D transforms of that size or silently fall back.
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(4)

# Apodisation only at the z-saves. Per-step windowing makes the damping scale with the
# step count, so the integrator does not converge in its own rtol.
const TWIN_SAVES_ONLY = 1_000_000_000

# ---------------------------------------------------------------- parameters --
# Every physical and solver parameter matches 13_production_2fs_gdd_pair.jl /
# 04_production_gap1000_kerr_raman.jl, so the ONLY difference from the delivered envelope
# files is the field/envelope representation. Do not change one without changing those.
const GAP    = 1.0e-3
λ0           = 260e-9
τfwhm        = ARM.τfwhm
energy       = 0.1e-6
material     = :SiO2
thickness    = ARM.thickness
a            = 125e-6
f_coll       = 5.0
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = GAP
λlims        = (143e-9, 600e-9)
d            = mask_spacing/2 + mask_diam/2

const R_GRID = 366.0e-6
const N_GRID = 768

beam   = TS.HE11Beam(a, f_coll, f_foc)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                               zmask=f_foc, apod=:tanh)

setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod=:supergauss, apod_param=16,
                trange=110e-15, store_window=false,
                R=R_GRID, N=N_GRID,
                field_mode=true, response=RESPONSE, ffac=FFAC,
                arraytype = ATYPE == "cuda" ? :cuda : Array,
                # Two fewer resident device fields (9 GiB at this shape), traded for one
                # host-to-device transfer per delay point — a fraction of a second against
                # a propagation of minutes. Worth it here in a way it is not at envelope
                # sizes, because the field grid is what makes the card tight.
                beamlets_on_host=true)

# Delay subset taken ON the production grid points, so each one compares directly against
# the delivered envelope file with no interpolation in τ. A full 200-point scan is ~40x
# the envelope campaign's resource; the comparison does not need it.
τ_prod = collect(range(-ARM.τmax, ARM.τmax, 200))
τ = τ_prod[round.(Int, range(1, 200, NPTS))]

# The same 16-slice ladder as 04/13, so every depth has an envelope counterpart. The
# residual's DEPTH DEPENDENCE is the quantity of interest: the unexplained retrieval
# residual grows with propagation, so a representation error that does the same is the
# signature to look for.
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

NAME = get(ENV, "FIELD_NAME",
           "tgfrog_field_$(RESPONSE)_ffac$(FFAC)_rtol7_sw_gap1000um_tanh_" *
           "$(ARM.tag)_$(round(Int, thickness*1e6))umUVFS")

# ------------------------------------------------------------------- report ---
# Grid and memory budget, computed rather than quoted: at these sizes "it did not fit" is
# an hour of queue time and a dead process, and the numbers are cheap to derive.
grid = Luna.Grid.RealGrid(thickness, λ0, λlims, 110e-15; ffac=FFAC)
Nω, Nt, Nto = length(grid.ω), length(grid.t), length(grid.to)
Nωo = length(grid.ωo)
fld(n, sz) = n * N_GRID^2 * sz / 2^30
pointwise = RESPONSE === :thg      # E³ is pointwise; |E_a|²E is batched
need = 9*fld(Nω, 16) +             # RK45 state registers
       # Et_win, the window-application scratch — only when the grid is oversampled;
       # otherwise Eto already has the coarse shape and doubles as it.
       (Nto == Nt ? 0.0 : fld(Nt, 8)) +
       fld(Nto, 8) +               # Eto
       fld(Nωo, 16) +              # Eωo (Pωo aliases it)
       (pointwise ? 0.0 : fld(Nto, 8)) +   # Pto (aliases Eto when all responses pointwise)
       (pointwise ? 0.0 : fld(Nto, 16)) +  # analytic-signal buffer of the no-THG response
       fld(Nω, 8)                  # the extraction window, on the device

@printf("field-mode TG-FROG — %s\n", NAME)
@printf("arm %s: %.1f fs, %d µm, response :%s, ffac %d\n",
        CONFIG, τfwhm*1e15, round(Int, thickness*1e6), RESPONSE, FFAC)
@printf("grid: Nω %d, Nt %d, Nto %d, Nωo %d, oversampling %s; transverse %d×%d, R %.0f µm\n",
        Nω, Nt, Nto, Nωo, Nto == Nt ? "none" : "$(Nto ÷ Nt)×", N_GRID, N_GRID, R_GRID*1e6)
@printf("  (the envelope production grid for comparison is Nω = 256)\n")
@printf("%d delay points on the production τ grid, %d z-slices, rtol 1e-7, max_dz 2 µm\n",
        NPTS, length(zsave))
@printf("estimated resident field memory: %.1f GiB\n", need)
@printf("arraytype %s   julia %s   threads %d\n", ATYPE, VERSION, Threads.nthreads())
flush(stdout)

if ATYPE == "cuda"
    # Resolve (and load) the GPU package the same way build_setup will, then ask it how
    # much room there actually is.
    Luna.resolve_arraytype(:cuda)
    st = Luna.device_memory_status()
    if isnothing(st)
        println("WARNING: could not read device memory")
    else
        free, total = st ./ 2^30
        @printf("device: %.1f GiB free of %.1f GiB\n", free, total)
        if need > 0.9*free
            @printf("\nREFUSING TO START: this shape needs ~%.1f GiB and only %.1f GiB is free.\n",
                    need, free)
            println("Options, in order of preference:")
            println("  * run on a larger card (an H200 is 141 GiB; an 80 GB H100 fits")
            println("    RESPONSE = :thg, or :nothg with FFAC = 4)")
            println("  * RESPONSE = :thg          (pointwise: no Pto, no analytic buffer)")
            println("  * FFAC = 4                 (:nothg only — READ THE HEADER FIRST)")
            println("  * reduce N_GRID            (then the envelope reference must be")
            println("    recomputed at the same N: the comparison needs matched grids)")
            exit(1)
        end
    end
end

if get(ENV, "PNPS_DRYRUN", "0") == "1"
    @info "dry run: configuration validated, NOT propagating" scan_name=NAME delays=NPTS zsaves=length(zsave)
    exit(0)
end

# --------------------------------------------------------------------- run ----
println("\nstarted $(Dates.now())")
flush(stdout)
t0 = time()
TS.run_scan(setup_args, τ; scan_name=NAME, exec=Scans.LocalExec(),
            zsave=zsave, init_dz=5e-7, rtol=1e-7, max_dz=2e-6,
            twin_period=TWIN_SAVES_ONLY,
            fftw_threads=4, fftw_mode=:estimate,
            skip_existing=true)
wall = time() - t0

ndone = length(TS._completed_scanidcs(NAME))
@printf("\nscan wall %.1f s (%.2f h) — %d/%d points now complete\n",
        wall, wall/3600, ndone, NPTS)
ndone < NPTS && println("INCOMPLETE — rerun this script in the same directory to resume")
@printf("collected: %s_collected.h5\n", NAME)
println("done $(Dates.now())")

# NEXT
#   The collected file carries /grid/field_mode = 1, /grid/response and /grid/ffac, and its
#   /grid/ω is a MONOTONIC rfft half-spectrum — not the envelope files' FFT-ordered
#   relative-frequency axis. `TS.load_simulated_scan` reads the marker and skips the
#   fftshift accordingly; anything else reading these files must do the same.
#
#   `verify_against_collected` CANNOT compare this against an envelope file: Nω is 513
#   against 256 and the axes differ, so it will (correctly) refuse on the size mismatch.
#   The comparison is between physical spectral densities on the common band — |E|² divided
#   by Δω², splined onto one axis — reported as rms difference relative to trace peak at
#   matched (τ, z), PER DEPTH. How any difference GROWS with depth is the interesting
#   quantity, because the unexplained retrieval residual does exactly that. The recipe is
#   in ModelPNPS's "field mode: field-versus-envelope trace agreement" testset.
