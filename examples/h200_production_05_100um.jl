# =============================================================================
# 05: the 100 µm 3-D anchor scan, on one H100/H200.
#
# A production run, not a benchmark: this is the campaign described in
# FROG_paper_new/05_production_100um.jl, which has never been executed. Every
# physical and solver parameter below is copied from that script — do not change
# any of them without changing it there too.
#
#   λ0 260 nm, 1 fs, 0.1 µJ into 100 µm SiO2, boxcars gap 1.0 mm, R = 366 µm,
#   N = 768, trange = 220 fs (→ Nω = 512), τ = ±30 fs in 241 points,
#   10 z-slices, rtol = 1e-7, max_dz = 2 µm, apodisation at the saves only,
#   and the DEFAULT `weaknorm` — `signal_quadrant_norm` was written for this run,
#   measured against it, and the measurement argued against it (F39).
#
# WHAT DIFFERS FROM THE HPC VERSION, AND WHY
#   * `arraytype = CUDA.CuArray` and `Scans.LocalExec()`: one process, one card,
#     points in sequence. Running several points concurrently would not help —
#     cost per GiB of state is flat across shapes, i.e. one point already
#     saturates the memory system.
#   * H200 ONLY at this grid. Scaling the measured N=640 point (3.12 GiB field,
#     85 steps, 20.5 s, 32.8 GiB device) by the flat 0.0771 s/step/GiB gives, at
#     N=768: field 4.50 GiB, ~30 s/point, **~47 GiB resident** — which does NOT
#     fit an A40's 44.4 GiB. Host peak is ~20 GiB (four beamlet fields plus the
#     window, transiently, in build_setup).
#   * No FFTW wisdom pre-generation. The 3-D host plan the HPC script needs is
#     never built on the device path (`Luna.setup` skips it when the caller
#     supplies the state), so the mandatory pre-step in that script's header does
#     not apply here. Only the small beamlet transforms remain.
#   * `skip_existing=true`: `scansave` fills the collected file point by point, so
#     an interrupted run resumes where it stopped. On rented hardware that is the
#     difference between losing minutes and losing the whole scan.
#
# HOW TO RUN IT — use the launcher, which sources /workspace/env.sh, checks the
# prerequisites and logs. This takes ~1.5 h, so run it detached:
#
#   bash /workspace/code/Luna.jl/test/manual/runpodcoldstart.sh    # if not yet, or to pull
#   tmux new -s prod05
#   bash /workspace/code/ModelPNPS.jl/examples/h200_production_05_100um.sh
#   # detach with C-b d; reattach with `tmux attach -t prod05`
#
# By hand instead, note that `source /workspace/env.sh` is NOT optional: without
# it `julia` is not on PATH and JULIA_DEPOT_PATH points at container disk, so
# everything silently re-resolves into storage that is wiped on terminate.
#
#   source /workspace/env.sh
#   cd /workspace/runs/prod05
#   julia --project=/workspace/code/dev \
#         /workspace/code/ModelPNPS.jl/examples/h200_production_05_100um.jl 2>&1 | tee prod05.log
#
# Re-running in the same directory resumes. To start over, delete
# tgfrog_kerr_rtol7_sw_gap1000um_tanh_1fs_100umUVFS_collected.h5.
#
# Environment: PROD_POINTS (default 241 — lower it for a rehearsal),
# PROD_ARRAYTYPE (cuda|cpu, default cuda), PROD_NAME (scan name).
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import CUDA
import Printf: @printf
import Dates

# Matched to the production script. On the device path these barely matter — the HE11
# builder is Bessel functions and broadcasts, not FFTs, and `Luna.setup` skips the 3-D
# host plan entirely when the caller supplies the state, so only the 1-D length-Nω plan
# is affected. Set anyway so the two scripts differ in nothing but the execution.
# No wisdom pre-generation is needed here for the same reason (see the header).

Luna.set_fftw_mode(:measure)
Luna.set_fftw_threads(4)

NPTS  = parse(Int, get(ENV, "PROD_POINTS", "241"))
ATYPE = get(ENV, "PROD_ARRAYTYPE", "cuda")
NAME  = get(ENV, "PROD_NAME", "tgfrog_kerr_rtol7_sw_gap1000um_tanh_1fs_100umUVFS")

# Apodisation only at the z-saves. Per-step windowing makes the damping scale with
# the step count, so the integrator does not converge in its own rtol; any value
# above the step count is equivalent and `Output.willsave` forces a window before
# each save. See the 05 header (90c_apodisation_test.jl).
const TWIN_SAVES_ONLY = 1_000_000_000

# ---------------------------------------------------------------- parameters --
const GAP    = 1.0e-3
λ0           = 260e-9
τfwhm        = 1.0e-15
energy       = 0.1e-6
material     = :SiO2
thickness    = 100e-6
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

# apod: input holes :supergauss(16), collection window :tanh. The two differ
# because edge width scales with hole diameter, so the 1.0 mm input holes are
# resolved 2× better than the 0.5 mm window hole. Pinned rather than relying on
# the default.
setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod=:supergauss, apod_param=16,
                trange=220e-15, store_window=false,
                R=R_GRID, N=N_GRID,
                arraytype = ATYPE == "cuda" ? CUDA.CuArray : Array)

τ = collect(range(-30e-15, 30e-15, NPTS))

# Dense to the generation zone; 9.5 µm (experimental) and 40 µm (cross-anchor to
# the 40 µm series) included exactly.
zsave = [0.0, 2.0, 4.0, 8.0, 9.5, 16.0, 25.0, 40.0, 60.0, 100.0] .* 1e-6

# --------------------------------------------------------------------- run ----
@printf("05 100 µm anchor scan — %s\n", NAME)
@printf("grid (Nω, %d, %d), trange %.0f fs, %d z-slices, %d delay points ±%.0f fs\n",
        N_GRID, N_GRID, 220.0, length(zsave), NPTS, 30.0)
@printf("arraytype %s   julia %s   threads %d\n", ATYPE, VERSION, Threads.nthreads())
if ATYPE == "cuda"
    dv = CUDA.device()
    @printf("device %s   %.1f GiB total\n", CUDA.name(dv), CUDA.total_memory()/2^30)
    @printf("expect ~30 s and ~47 GiB per point at N=768 (extrapolated from the\n")
    @printf("measured N=640 point: 20.5 s, 32.8 GiB; cost is linear in field size)\n")
end
@printf("estimate: %.1f h for %d points (at ~30 s/point + ~1 min setup)\n",
        (NPTS*30.0 + 60)/3600, NPTS)
println("started $(Dates.now())\n")
flush(stdout)

t0 = time()
TS.run_scan(setup_args, τ; scan_name=NAME, exec=Scans.LocalExec(),
            zsave=zsave, init_dz=5e-7, rtol=1e-7, max_dz=2e-6,
            twin_period=TWIN_SAVES_ONLY,
            # Redundant with the top-level setters above for a single LocalExec
            # process — they exist so `procs` workers, which never run the script's
            # top level, still get them. Passed anyway so this differs from
            # 05_production_100um.jl in nothing but the execution.
            fftw_threads=4, fftw_mode=:measure,
            skip_existing=true)
wall = time() - t0

ndone = length(TS._completed_scanidcs(NAME))
@printf("\nscan wall %.1f s (%.2f h) — %d/%d points now complete\n",
        wall, wall/3600, ndone, NPTS)
ndone < NPTS && println("INCOMPLETE — rerun this script in the same directory to resume")
@printf("collected: %s_collected.h5\n", NAME)
println("done $(Dates.now())")

# NEXT: the accuracy companion. This file is the reference for a CPU run on the
# HPC — the parameters are the production ones, so recompute a few points there
# with verify_against_collected against it and read the SCAN-PEAK normalised
# column (examples/scan_peaks.jl explains why the own-peak column misleads in the
# delay wings).
