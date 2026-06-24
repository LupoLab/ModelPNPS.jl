# =============================================================================
# Forward-model TG-FROG simulation: full physical (mask) scheme — 1 fs pulse.
#
# Reference ModelPNPS example.
#
#   * Hollow-fibre HE₁₁ mode, collimating + focusing lens pair, four-hole
#     boxcar mask, χ⁽³⁾ Kerr four-wave mixing in a thin SiO₂ slab.
#   * Signal extracted by a frequency-dependent apodised hole (chromatic
#     vignetting captured exactly).
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

# --- Pulse / substrate parameters --------------------------------------------
λ0           = 260e-9          # carrier wavelength [m] — deep UV
τfwhm        = 1.0e-15         # intensity FWHM duration [s]
energy       = 0.1e-6          # total pulse energy [J] — split across the 3 holes
material     = :SiO2           # UV fused silica
thickness    = 40e-6           # substrate thickness / propagation distance [m]

# --- Optical / mask geometry -------------------------------------------------
a            = 125e-6          # hollow capillary core radius [m]
f_coll       = 5.0             # collimating lens focal length [m]
f_foc        = 0.1             # focusing lens focal length [m]
mask_diam    = 1.0e-3          # mask hole diameter [m]
mask_spacing = 0.5e-3          # edge-to-edge gap between adjacent holes [m]

# Signal hole sits at the (-x, -y) corner of the boxcar with HALF the
# diameter — this is the master script's tighter spatial filter.
d            = mask_spacing/2 + mask_diam/2     # hole-centre distance from axis [m]

beam   = TS.HE11Beam(a, f_coll, f_foc)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d,
                                holediam=mask_diam/2,
                                zmask=f_foc,
                                apod=:supergauss, apod_param=16)

# --- Build the once-only setup ----------------------------------------------
setup = TS.build_setup(; λ0, τfwhm, energy, thickness, material,
                         mask_diam, mask_spacing, λlims=(143e-9, 600e-9),
                         beam, window)

# --- FROG delay scan and SLURM launch ---------------------------------------
τ = collect(range(-14e-15, 14e-15, 128))   # 128 delay points across ±14 fs
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="60G", arraymode=:batch)
#exec = Scans.LocalExec()

scan_name = "tgfrog_260nm_1fs_FTL_mask_SiO2_40um_1.5mmCtC_1mmHole"

# `zsave` saves the propagated field at multiple material thicknesses in this
# ONE 40 µm run. Because propagation is forward-marching with z-independent
# dynamics, the trace saved at e.g. 10 µm is identical to a dedicated 10 µm run —
# so 1/10/20/40 µm all come (almost) free from a single simulation. The trace
# datasets become (Nω, nz, Nτ) and the z positions are stored in /grid/zsave.
# `thickness` (=40 µm) is appended automatically if not already in the list.
TS.run_scan(setup, τ; scan_name, exec,
            zsave=[1e-6, 10e-6, 20e-6, 40e-6], init_dz=5e-7)
# Uniform alternative: zsave=21 saves 21 evenly-spaced slices over [0, 40 µm].

# Recover any thickness afterwards, e.g.:
#   d10 = TS.load_simulated_scan("$(scan_name)_collected.h5"; z_thickness=10e-6)
#   all = TS.load_simulated_scan("$(scan_name)_collected.h5"; z_index=:all)  # (Nω, nz, Nτ)
