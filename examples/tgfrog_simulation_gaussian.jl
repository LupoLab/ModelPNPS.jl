# =============================================================================
# Forward-model TG-FROG simulation: Gaussian-beam scheme.
#
# Reference ModelPNPS example.
#
#   * Three Gaussian beams placed directly at the boxcar k-space corners
#     (no fibre mode, no physical mask).
#   * TWO signal-extraction windows in a single run, both smooth Planck
#     tapers:
#       - PlanckWindow      (frequency-INDEPENDENT — no chromatic vignetting)
#       - PlanckOmegaWindow (frequency-DEPENDENT  — chromatic vignetting)
#     Comparing the two outputs isolates chromatic vignetting from the
#     mask-edge / mode-shape effects.
#
# All grid/scan parameters match the mask example for direct comparability:
# trange=40 fs, λlims=(160, 500) nm, mask_spacing=0.5 mm, 80 delays
# over ±10 fs.
#
# WARNING: this script requires a SLURM cluster — the full propagation is
# typically tens of minutes per delay point and is NOT runnable on a
# laptop. To launch a local test instead, swap `Scans.SlurmExec` for
# `Scans.LocalExec()` and reduce τ to a single value.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna.Scans
import Luna

Luna.set_fftw_mode(:estimate)

# --- Pulse / substrate parameters (identical to the mask example) ------------
λ0           = 260e-9
τfwhm        = 2.0e-15
energy       = 0.2e-6
material     = :SiO2
thickness    = 10e-6

# --- Optical / mask geometry (identical to the mask example) -----------------
a            = 125e-6
f_coll       = 5.0
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = 0.5e-3

# --- Gaussian beam waist & crossing geometry derived from mask parameters ---
# w0 = Airy disc radius from a single mask hole through the focusing lens:
#      w0 ≈ λ0·f_foc / (π · holediam/2)
w0       = λ0 * f_foc / (π * mask_diam/2)
d_hole   = mask_spacing/2 + mask_diam/2          # hole centre-to-axis [m]
crossingθ = d_hole / f_foc                        # crossing half-angle [rad]
Δk       = 2π / λ0 * sin(crossingθ)              # k-space tilt per beam [rad/m]

beam = TS.GaussianBeam(w0, f_foc)

# --- Two signal windows ------------------------------------------------------
# The signal beam emerges at the (-Δk, -Δk) corner of the boxcar. The Planck
# half-width 2.5/w0 captures the central Gaussian k-lobe with comfortable
# margin; pad=1.25 sets the outer roll-off to 1.25 × half-width.
windows = [
    TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/w0, pad=1.25),
    TS.PlanckOmegaWindow(xc=-d_hole, yc=-d_hole,
                          holediam=mask_diam/2, f_foc=f_foc, pad=1.25),
]

# --- Build the once-only setup ----------------------------------------------
setup = TS.build_setup(; λ0, τfwhm, energy, thickness, material,
                         mask_diam, mask_spacing,
                         beam, window=windows)

# --- FROG delay scan and SLURM launch ---------------------------------------
τ = collect(range(-10e-15, 10e-15, 80))
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="18G", arraymode=:batch)

scan_name = "tgfrog_260nm_2fs_FTL_gaussian_SiO2_10um_1.5mmCtC_1mmHole"

TS.run_scan(setup, τ; scan_name, exec, nz=2, init_dz=5e-7)
