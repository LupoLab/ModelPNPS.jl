"""
High-fidelity forward modelling of PNPS (Parametrized Nonlinear Process
Spectrum) pulse-characterisation traces by full spatially-resolved nonlinear
propagation, using [Luna.jl](https://github.com/LupoLab/Luna.jl).

`ModelPNPS` is a standalone package for generating synthetic
pulse-characterisation traces directly from the underlying experimental
physics — spatial beam overlap, mode shape, mask edges, chromatic vignetting,
material dispersion, phase-matching and full χ⁽ⁿ⁾ nonlinear propagation. Given
an analytic input pulse and an experimental geometry, it produces the trace a
real apparatus would record. These ground-truth traces are intended for
testing advanced retrieval algorithms and developing new characterisation
techniques.

The long-term ambition is a *complete* PNPS trace modeller spanning the full
Geib et al. (2019) taxonomy of nonlinear process × parametrization (FROG,
d-scan, time-domain ptychography, …). The **currently implemented process is
TG-FROG** (Transient-Grating FROG); see the documentation roadmap for the
planned methods.

# Physical model

The TG-FROG signal in the boxcar geometry is a degenerate four-wave mixing
process

```
    k_signal = k_g2 - k_g1 + k_test
```

Three input beams (gates `g1`, `g2` and a delayed test pulse `t`) are crossed
inside a thin nonlinear substrate (e.g. UV fused silica). Their interference
produces a transient grating; the test pulse diffracts off this grating into
the fourth corner of the boxcar. Scanning the test-pulse delay τ and
spectrally resolving the diffracted signal yields a 2-D `I(ω, τ)` spectrogram.

# Two beam models

Two `AbstractInputBeam` subtypes are provided:

- `HE11Beam` — the master experimental model. The HE₁₁ mode of a hollow
  capillary fibre is collimated by a long lens, clipped by a four-hole
  apodised mask in the collimated beam, then focused into the substrate.
  Each hole selects one of the four boxcar arms.
- `GaussianBeam` — a simplified Gaussian-beam model that places three
  Gaussian beams directly at the correct k-space angles (no mask, no fibre
  mode). Useful as a sanity-check baseline.

# Three signal-extraction window types

Three `AbstractSignalWindow` subtypes are provided:

- `PhysicalMaskWindow` — the master experimental signal extraction:
  a frequency-dependent apodised hole in the mask plane (chromatic vignetting
  is captured exactly).
- `PlanckWindow` — a smooth, frequency-*independent* radial Planck taper in
  k-space (no chromatic vignetting; baseline for the Gaussian model).
- `PlanckOmegaWindow` — a smooth, frequency-*dependent* Planck taper that
  mimics the chromatic vignetting of `PhysicalMaskWindow` while keeping
  the smooth-edge advantage. Used to isolate the two effects (smooth-edge
  vs ω-scaling) within the Gaussian model.

# Two field representations

By default the propagation is Luna's complex **envelope** (`Grid.EnvGrid`): an analytic
field about a carrier. `build_setup(field_mode=true)` instead propagates the real,
carrier-resolved field on a `Grid.RealGrid`, which has no envelope/carrier split, no
dropped third-harmonic term and no negative-frequency wrap. That matters when the pulse is
only a cycle or two long — at 260 nm a 1 fs pulse is 1.15 optical cycles — where the
envelope approximation is marginal by construction and the two representations can be
compared directly. It costs roughly twice the memory and three times the time per delay
point, so it is a diagnostic, not the production default. See `build_setup`'s
`field_mode`, `response` and `ffac` keywords.

# High-level usage

```julia
using ModelPNPS
import Luna.Scans

beam   = HE11Beam(125e-6, 5.0, 0.1)
window = PhysicalMaskWindow(
    holex=-0.75e-3, holey=-0.75e-3,
    holediam=0.5e-3, zmask=0.1)

setup = build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                      thickness=10e-6, material=:SiO2,
                      mask_diam=1.0e-3, mask_spacing=0.5e-3,
                      beam, window)

τ = collect(range(-10e-15, 10e-15, 80))
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="18G",
                       arraymode=:batch)
run_scan(setup, τ; scan_name="my_trace", exec)
```

The full simulation requires SLURM (it is hours of CPU per delay scan with
typical grid sizes); the unit tests exercise everything except the actual
`Luna.run` call by passing `skip_propagation=true` to
`simulate_delay_point`.
"""
module ModelPNPS

import Adapt
import FFTW
import FFTW: fft, ifft, irfft, plan_fft, plan_rfft
import HDF5
import LinearAlgebra: mul!
import Luna
import Luna: Capillary, Fields, Grid, LinearOps, Maths, Nonlinear, NonlinearRHS,
    Output, PhysData, Raman, Scans
import Luna.Capillary: besselj
import Statistics: median

export AbstractInputBeam, HE11Beam, GaussianBeam,
    AbstractSignalWindow, PhysicalMaskWindow, PlanckWindow, PlanckOmegaWindow,
    TGFROGSetup,
    InputPulseData, load_input_pulse, spectral_window!, center_pulse!,
    interp_input_pulse,
    optimal_spatial_grid,
    build_he11_kspace, build_gaussian_kspace,
    apply_tilt, apply_delay,
    makemask, build_window,
    build_setup, simulate_delay_point, run_scan, memory_budget,
    signal_quadrant_norm,
    extract_signal_spectra,
    load_simulated_scan

public delayed_input, verify_against_collected

include("spatial_grid.jl")
include("types.jl")
include("input_pulse.jl")
include("fields.jl")
include("windows.jl")
include("beamlets.jl")
include("raman.jl")
include("memory.jl")
include("grid_representations.jl")
include("setup.jl")
include("simulation.jl")
include("scans.jl")
include("scan_io.jl")

end # module
