# =============================================================================
# tgfrog_window_series.jl — a production-scale TG-FROG delay scan on a SLURM
# cluster.
#
# THE MEASUREMENT. A transform-limited 1 fs pulse at 260 nm illuminates a
# four-hole boxcar mask (1.0 mm holes, 1.0 mm edge gap). The three transmitted
# beamlets are focused into 40 µm of UV fused silica, where the gate pair
# writes a transient refractive-index grating and the probe diffracts from it
# into the background-free fourth corner. The diffracted signal is collected
# through an aperture at the mask-conjugate plane and spectrally resolved,
# delay by delay, giving the TG-FROG trace I(ω, τ).
#
# THE COLLECTION-WINDOW SERIES. The signal field is the product of three
# aperture-filtered fields, so its k-space content is the three-fold
# convolution of the input aperture: full support of three aperture diameters,
# but an rms width of only 0.325 D. The fraction a collection hole passes
# therefore rises steeply with its diameter:
#
#     output hole   0.50   0.75   1.00   1.50   2.00   2.50   3.00 mm
#     collected      24%    47%    69%    93%  99.4%   100%   100%
#
# A small hole rejects pump leakage but transmits a chromatic, coherent subset
# of the signal; a large hole collects essentially everything but admits more
# background. For this layout the hole can open to 3D before it reaches a
# transmitted input beamlet (2(2d - D/2) = 3D exactly when gap = D). ModelPNPS
# accepts a VECTOR of windows, and every window is a reduction of the same
# propagated field — so one propagation yields the whole series (`Iω_win`,
# `Iω_win_2`, ..., plus `_reimaged` variants) at the cost of one window-sized
# array each in memory.
#
# THE THICKNESS LADDER. The propagation dynamics do not depend on z except
# through the accumulated field, so a field snapshot at an intermediate z is
# identical to a dedicated run of that thickness. The `zsave` vector below
# turns the single 40 µm scan into traces at sixteen substrate thicknesses.
#
# HOW TO RUN. First a dry run on the login node, which validates the
# configuration and prints the memory budget without submitting:
#
#     PNPS_DRYRUN=1 julia --project=. tgfrog_window_series.jl
#
# then plain to submit the SLURM array job. To submit from a workstation to a
# remote cluster instead, wrap the `SlurmExec` in `Scans.SSHExec(slurm, host,
# subdir)`.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans

# FFTW planning. `:estimate` costs nothing at startup; `:measure` gives faster
# transforms but takes tens of minutes to plan production-sized 3-D FFTs unless
# wisdom for this exact (grid, thread count) has been generated beforehand.
Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(2)

# Apply the apodisation windows only at the z-saves rather than after every
# accepted step, so the absorbed leakage does not scale with the step count and
# an `rtol` convergence study means something (see the accuracy manual page).
# Any value larger than the achievable step count has the same effect;
# `Output.willsave` forces a window immediately before every save regardless.
const TWIN_SAVES_ONLY = 1_000_000_000

# `:test` runs 16 delays — enough to check the configuration end to end.
# `:full` is the 200-delay production axis. Edit the const rather than reading
# an environment variable: the SLURM tasks re-run this file on the compute
# nodes without the submitting shell's environment.
const DELAY_SET = :test

λ0 = 260.0e-9
τfwhm = 1.0e-15
energy = 0.1e-6
material = :SiO2
thickness = 40.0e-6
a = 125.0e-6           # hollow-capillary core radius feeding the HE11 mode
f_coll = 5.0           # collimating focal length [m]
f_foc = 0.1            # focusing focal length [m]
mask_diam = 1.0e-3
mask_spacing = 1.0e-3  # edge-to-edge gap between adjacent holes
λlims = (143.0e-9, 600.0e-9)
d = mask_spacing / 2 + mask_diam / 2  # hole-centre offset from the optical axis

beam = TS.HE11Beam(a, f_coll, f_foc)

# The collection-aperture series. Centre and apodisation are fixed; only the
# diameter varies. The `:tanh` edge width is set by the k-grid (about 97 µm
# here), so it is the same ABSOLUTE edge on every hole — 19% of the 0.5 mm
# hole but 3.9% of the 2.5 mm one, i.e. the large holes are the more
# top-hat-like, which is the intended limit.
const HOLE_DIAMS = [0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
window = [TS.PhysicalMaskWindow(holex = -d, holey = -d, holediam = hd,
                                zmask = f_foc, apod = :tanh) for hd in HOLE_DIAMS]

# `trange` must hold the dispersively stretched pulse PLUS the full delay scan
# inside the apodisation; the delay enters as a spectral phase on a periodic
# grid, so a delay beyond half the window aliases onto its complement.
# `store_window = false` keeps the (Nω, Ny, Nx) window arrays out of the file.
setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod = :supergauss, apod_param = 16,
                trange = 110.0e-15, store_window = false,
                R = 366.0e-6, N = 768)

const Τ_FULL = collect(range(-25.0e-15, 25.0e-15, 200))
const Τ_TEST = collect(range(-24.0e-15, 24.0e-15, 16))

τ = DELAY_SET === :full ? Τ_FULL :
    DELAY_SET === :test ? Τ_TEST :
    throw(ArgumentError("DELAY_SET must be :test or :full, got $DELAY_SET"))

# Substrate thicknesses to save; the traces become (Nω, nz, Nτ) and
# `load_simulated_scan(...; z_thickness = ...)` picks a slice.
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

# Each array task holds `instances` delay points at once, so the allocation
# must cover `instances` × the per-point footprint that the dry run prints
# (about 36 GiB/point at this grid with six windows).
const NJOBS = DELAY_SET === :full ? 10 : 4
exec = Scans.SlurmExec(@__FILE__, NJOBS; nthreads = 2, instances = 4, cpus = 8,
                       memory = "160G", time = "1-00:00:00", arraymode = :queue)

scan_name = "tgfrog_winseries_" * string(DELAY_SET) * "_1fs_40umUVFS"

let bu = TS.memory_budget(setup_args)
    println("TG-FROG collection-window series — $scan_name")
    println("  holes (mm): ", join(HOLE_DIAMS .* 1e3, ", "))
    println("  delays $(length(τ)) ($(DELAY_SET)), z-slices $(length(zsave)), ",
            "jobs $NJOBS x $(exec.instances) instances x 2 threads")
    println("  memory: $(round(bu.device, digits = 1)) GiB propagation state, ",
            "$(round(bu.host, digits = 1)) GiB setup peak per point; ",
            "measured RSS at this shape is ~36 GiB/point")
    println("  expect Iω_win, Iω_win_2 ... Iω_win_6 plus _reimaged variants")
end

if get(ENV, "PNPS_DRYRUN", "0") == "1"
    @info "dry run: configuration validated, NOT submitting" scan_name delays = length(τ)
    exit(0)
end

TS.run_scan(setup_args, τ; scan_name, exec, zsave, init_dz = 5.0e-7,
            rtol = 1.0e-7, max_dz = 2.0e-6, twin_period = TWIN_SAVES_ONLY,
            fftw_threads = 2, fftw_mode = :estimate)
