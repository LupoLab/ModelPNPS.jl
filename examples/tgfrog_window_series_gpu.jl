# =============================================================================
# tgfrog_window_series_gpu.jl — the collection-window series of
# `tgfrog_window_series.jl`, run on a single NVIDIA GPU with a chirped input
# pulse.
#
# THE INPUT. The `GDD` and `TOD` keywords chirp the analytic pulse: here
# +2 fs² of group-delay dispersion AND +2 fs³ of third-order phase at once, a
# structured test case in which the two orders must be disentangled by a
# retrieval. Set both to zero for the transform-limited pulse of the CPU
# example.
#
# THE GPU PATH. `arraytype = :cuda` moves the whole propagation and extraction
# onto the device (about 160× faster than two CPU cores at this shape — 42 s
# per delay point on an H200); `beamlets_on_host = true` keeps the three input
# beamlets in host memory and uploads the delayed sum once per point, saving
# about 9 GiB of device memory for a sub-second transfer. The `arraytype`
# symbol must travel INSIDE `setup_args`, so that CUDA.jl is loaded on the
# machine that runs the propagation — see the GPU manual page for this rule
# and the world-age reasoning behind it.
#
# The environment needs CUDA.jl alongside ModelPNPS; it is not a dependency of
# the package. Physics, grid, delays and z-ladder are otherwise identical to
# the CPU example, which documents them.
#
# HOW TO RUN. Dry run first — it prints the memory budget and checks the card
# without propagating:
#
#     PNPS_DRYRUN=1 julia --project=. -t auto tgfrog_window_series_gpu.jl
#
# then plain (typically under nohup: a full 200-delay scan is a few hours).
# Output lands in the working directory. On a multi-GPU machine, pin one scan
# per card with CUDA_VISIBLE_DEVICES.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import Printf: @printf
import Dates

Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(Threads.nthreads())

# Apodise only at the z-saves; see the CPU example and the accuracy manual page.
const TWIN_SAVES_ONLY = 1_000_000_000

λ0 = 260.0e-9
τfwhm = 1.0e-15
energy = 0.1e-6
material = :SiO2
thickness = 40.0e-6
a = 125.0e-6
f_coll = 5.0
f_foc = 0.1
mask_diam = 1.0e-3
mask_spacing = 1.0e-3
λlims = (143.0e-9, 600.0e-9)
d = mask_spacing / 2 + mask_diam / 2

beam = TS.HE11Beam(a, f_coll, f_foc)
const HOLE_DIAMS = [0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
window = [TS.PhysicalMaskWindow(holex = -d, holey = -d, holediam = hd,
                                zmask = f_foc, apod = :tanh) for hd in HOLE_DIAMS]

setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod = :supergauss, apod_param = 16, trange = 110.0e-15,
                store_window = false, R = 366.0e-6, N = 768,
                GDD = +2.0e-30, TOD = +2.0e-45,
                arraytype = :cuda, beamlets_on_host = true)

τ = collect(range(-25.0e-15, 25.0e-15, 200))
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

scan_name = "tgfrog_winseries_p2GDDp2TOD_1fs_40umUVFS"

# Refuse to start rather than dying an hour in: compare the measured budget
# against the free device memory before the first propagation.
bu = TS.memory_budget(setup_args)
@printf("GPU window series -> %s\n  memory %.1f GiB device / %.1f GiB host\n",
        scan_name, bu.device, bu.host)
Luna.resolve_arraytype(:cuda)
st = Luna.device_memory_status()
if !isnothing(st)
    free = st[1] / 2^30
    if bu.device > 0.9 * free
        println("REFUSING: needs $(bu.device) GiB, free $(round(free, digits = 1)) GiB")
        exit(1)
    end
end

if get(ENV, "PNPS_DRYRUN", "0") == "1"
    @info "dry run: configuration validated, NOT propagating" scan_name delays = length(τ)
    exit(0)
end

# `skip_existing = true` lets an interrupted scan resume where it stopped.
println("started $(Dates.now())")
flush(stdout)
TS.run_scan(setup_args, τ; scan_name, exec = Scans.LocalExec(), zsave,
            init_dz = 5.0e-7, rtol = 1.0e-7, max_dz = 2.0e-6,
            twin_period = TWIN_SAVES_ONLY,
            fftw_threads = Threads.nthreads(), fftw_mode = :estimate,
            skip_existing = true)
println("done $(Dates.now())")
