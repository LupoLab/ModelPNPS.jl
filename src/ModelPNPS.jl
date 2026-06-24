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

# Two signal-extraction window types

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

import Luna
import Luna: Capillary, Fields, Grid, LinearOps, Maths, Nonlinear, NonlinearRHS,
             Output, PhysData, Scans
import Luna.Capillary: besselj
import FFTW
import FFTW: fft, ifft, plan_fft
import HDF5
import Statistics: mean

export AbstractInputBeam, HE11Beam, GaussianBeam,
       AbstractSignalWindow, PhysicalMaskWindow, PlanckWindow, PlanckOmegaWindow,
       TGFROGSetup,
       optimal_spatial_grid,
       build_he11_kspace, build_gaussian_kspace,
       apply_tilt, apply_delay,
       makemask, build_window,
       build_setup, simulate_delay_point, run_scan,
       extract_signal_spectra,
       load_simulated_scan

# ============================================================================
# Spatial-grid sizing
# ============================================================================

"""
    optimal_spatial_grid(f, mask_diam, mask_spacing, λmin, λmax;
                         n_airy=5, pts_per_lobe=10, safety=1.5)

Return `(R, N)` for a Luna `FreeGrid(R, N)` chosen so that the spatial grid

1. contains at least `n_airy` Airy diffraction patterns of the longest
   wavelength `λmax` from a mask hole of diameter `mask_diam` focused by a
   lens of focal length `f` (real-space containment), and
2. resolves the Airy pattern at the shortest wavelength `λmin` with at
   least `pts_per_lobe` points across the central lobe (real-space
   resolution), and
3. has a k-space half-extent that comfortably encloses the FWM nonlinear
   k-vectors generated at `λmin` from the outermost mask hole, with a
   `safety` headroom factor (k-space containment).

`N` is rounded up to the next power of 2 for FFT efficiency. Diagnostic
information is printed via `@info`.

# Arguments
- `f`: focal length of the focusing lens [m].
- `mask_diam`: diameter of each mask hole [m].
- `mask_spacing`: edge-to-edge spacing between adjacent mask holes [m].
- `λmin`, `λmax`: shortest and longest wavelengths the simulation must
  represent [m]. These should bracket the input spectrum *and* its FWM
  products.

# Keyword arguments
- `n_airy=5`: number of Airy patterns the grid should contain at `λmax`.
- `pts_per_lobe=10`: real-space samples across the central Airy lobe at
  `λmin`.
- `safety=1.5`: multiplier on the required nonlinear k-vector envelope to
  guard against aliasing.
"""
function optimal_spatial_grid(f, mask_diam, mask_spacing, λmin, λmax;
                              n_airy=5, pts_per_lobe=10, safety=1.5)
    x_max = mask_spacing/2 + mask_diam     # outermost mask edge from optical axis
    r_airy_max = 1.22 * λmax * f / mask_diam
    r_airy_min = 1.22 * λmin * f / mask_diam

    # Real-space containment: half-width R must hold n_airy Airy patterns at λmax.
    R_min = n_airy * r_airy_max

    # Real-space resolution: dx must resolve the Airy lobe at λmin.
    dx_max = r_airy_min / pts_per_lobe
    N_from_realspace = 2 * R_min / dx_max
    @info "N from real-space resolution" N_from_realspace

    # k-space containment: kmax must exceed the largest FWM nonlinear k-vector
    # (≈ 3 × outermost-hole k, with a safety factor).
    k_NL_max = safety * 3 * 2π * x_max / (λmin * f)
    N_from_kspace = 2 * R_min * k_NL_max / π
    @info "N from k-space containment" N_from_kspace

    N = nextpow(2, ceil(Int, max(N_from_realspace, N_from_kspace)))
    R = R_min

    dx = 2R / N
    dk = π / R
    kmax = π * N / (2R)
    k_hole_width_min = 2π * mask_diam / (λmax * f)   # narrowest hole in k-space
    n_hole = k_hole_width_min / dk
    @info "Spatial grid parameters" R_µm=R*1e6 N dx_µm=dx*1e6
    @info "Real-space check" airy_min_pts=r_airy_min/dx airy_max_pts=r_airy_max/dx
    @info "k-space check" k_NL_max kmax margin=kmax/k_NL_max pts_per_hole=n_hole

    return R, N
end

# ============================================================================
# Input-beam types
# ============================================================================

"""
    AbstractInputBeam

Abstract supertype for the input-beam models from which the three TG-FROG
beamlets are constructed. Concrete subtypes: [`HE11Beam`](@ref),
[`GaussianBeam`](@ref).
"""
abstract type AbstractInputBeam end

"""
    HE11Beam(a, f_coll, f_foc)

The HE₁₁ capillary mode imaged from the fibre output through a collimating
lens (`f_coll`) onto a beam mask, then focused (`f_foc`) into the substrate.
The Hankel transform of the mode has a closed form

```
    Ẽ(k_⊥) ∝ -a² u₁₁ J₁(u₁₁) J₀(a k_⊥) / (a² k_⊥² - u₁₁²)
```

where `u₁₁` is the first zero of `J₁`. The "image" of the fibre core at the
substrate has demagnified radius `a_scaled = a · f_foc / f_coll`.

# Fields
- `a::Float64`         — fibre core radius [m]
- `f_coll::Float64`    — collimating-lens focal length [m]
- `f_foc::Float64`     — focusing-lens focal length [m]
"""
struct HE11Beam <: AbstractInputBeam
    a::Float64
    f_coll::Float64
    f_foc::Float64
end
HE11Beam(; a, f_coll, f_foc) = HE11Beam(a, f_coll, f_foc)

"a_scaled = fibre core radius imaged onto the focal plane."
a_scaled(b::HE11Beam) = b.a * b.f_foc / b.f_coll

"""
    GaussianBeam(w0, f_foc)

A Gaussian beam with 1/e² intensity radius `w0` at the focus. `f_foc` is
retained only so the crossing angle (and `Δk`) can be derived from the same
mask geometry as the HE₁₁ model.

# Fields
- `w0::Float64`     — 1/e² intensity radius at the focus [m]
- `f_foc::Float64`  — focusing-lens focal length [m] (geometry only)
"""
struct GaussianBeam <: AbstractInputBeam
    w0::Float64
    f_foc::Float64
end
GaussianBeam(; w0, f_foc) = GaussianBeam(w0, f_foc)

# ============================================================================
# Signal-window types
# ============================================================================

"""
    AbstractSignalWindow

Abstract supertype for k-space windows used to extract the FWM signal beam
from the propagated field. Concrete subtypes: [`PhysicalMaskWindow`](@ref),
[`PlanckWindow`](@ref), [`PlanckOmegaWindow`](@ref).
"""
abstract type AbstractSignalWindow end

"""
    PhysicalMaskWindow(holex, holey, holediam, zmask;
                       apod=:supergauss, apod_param=nothing)

A frequency-dependent mask hole: physical position `(holex, holey)` and
diameter `holediam` in the mask plane, sitting `zmask` (= focal length)
upstream of the substrate. The mask plane ↔ k-space mapping is

```
    (x_mask, y_mask) = (k_x, k_y) · zmask · c / ω
```

so the *same* physical hole transmits a wavelength-dependent k-space
region (chromatic vignetting). Apodisation choices:

- `:hard` — binary (1 inside the hole, 0 outside).
- `:supergauss` — `exp(-(2 r/d)^n)` with `n = apod_param` (default 16).
- `:tanh` — smooth `0.5(1 - tanh((r - d/2)/Δ))` with `Δ = apod_param` in
  mask-plane metres (default `3 × Δx_mask` evaluated at the carrier
  wavelength).

# Fields
- `holex, holey`     — hole centre in the mask plane [m]
- `holediam`         — hole diameter [m]
- `zmask`            — focal length / mask-to-focus distance [m]
- `apod`             — apodisation type `:hard | :supergauss | :tanh`
- `apod_param`       — apodisation parameter (`nothing` → defaults)
"""
struct PhysicalMaskWindow <: AbstractSignalWindow
    holex::Float64
    holey::Float64
    holediam::Float64
    zmask::Float64
    apod::Symbol
    apod_param::Union{Nothing,Real}
end
PhysicalMaskWindow(; holex, holey, holediam, zmask,
                     apod=:supergauss, apod_param=nothing) =
    PhysicalMaskWindow(holex, holey, holediam, zmask, apod, apod_param)

"""
    PlanckWindow(kxc, kyc, kwidth, pad)

A radial Planck-taper window centred at `(kxc, kyc)` in k-space with flat
half-width `kwidth` and an outer roll-off radius `pad·kwidth`. The window is
**frequency-independent**: the same mask shape is applied to every spectral
component, so chromatic vignetting is removed.

# Fields
- `kxc, kyc`   — k-space centre of the window [rad/m]
- `kwidth`     — flat half-width of the window [rad/m]
- `pad`        — multiplier setting the outer roll-off (typically 1.25)
"""
struct PlanckWindow <: AbstractSignalWindow
    kxc::Float64
    kyc::Float64
    kwidth::Float64
    pad::Float64
end
PlanckWindow(; kxc, kyc, kwidth, pad=1.25) = PlanckWindow(kxc, kyc, kwidth, pad)

"""
    PlanckOmegaWindow(xc, yc, holediam, f_foc, pad)

A frequency-*dependent* Planck-taper window. The hole is specified in the
*mask plane* by its centre `(xc, yc)` and diameter `holediam`; at frequency
ω the window centre and half-width in k-space are

```
    k_c(ω)    = (ω/c) · (xc, yc) / f_foc
    k_hole(ω) = (ω/c) · (holediam/2) / f_foc
```

This restores the chromatic vignetting of [`PhysicalMaskWindow`](@ref) while
keeping the smooth-edge advantage of [`PlanckWindow`](@ref).

# Fields
- `xc, yc`     — hole centre in the mask plane [m]
- `holediam`   — hole diameter in the mask plane [m]
- `f_foc`      — focusing-lens focal length [m]
- `pad`        — outer roll-off multiplier (typically 1.25)
"""
struct PlanckOmegaWindow <: AbstractSignalWindow
    xc::Float64
    yc::Float64
    holediam::Float64
    f_foc::Float64
    pad::Float64
end
PlanckOmegaWindow(; xc, yc, holediam, f_foc, pad=1.25) =
    PlanckOmegaWindow(xc, yc, holediam, f_foc, pad)

# ============================================================================
# Setup container
# ============================================================================

"""
    TGFROGSetup

Container holding everything that is built once (independent of the FROG
delay τ): grids, propagation operators, FFT plan, the three pre-built
input beamlets, the signal window(s) and the metadata dictionary.

Use [`build_setup`](@ref) to construct one and [`simulate_delay_point`](@ref)
or [`run_scan`](@ref) to use it.

# Fields
The struct is a passive bundle; fields are not part of the public API and
may evolve. Use the constructors and methods provided.
"""
struct TGFROGSetup{LO,TR,FTT,WIN,WA}
    # Physical / numerical parameters echoed for output metadata
    λ0::Float64
    τfwhm::Float64
    energy::Float64
    thickness::Float64
    material::Symbol
    mask_diam::Float64
    mask_spacing::Float64

    # Luna grids
    grid::Grid.EnvGrid
    xygrid::Grid.FreeGrid

    # Pre-built propagation pieces
    linop::LO
    transform::TR
    FT::FTT
    energyfun_ω::Function

    # Pre-built input beamlets, all in k-space (Nω, Nky, Nkx)
    Eωk_g1::Array{ComplexF64,3}        # gate 1 (no delay)
    Eωk_g2::Array{ComplexF64,3}        # gate 2 (no delay)
    Eωk_t_base::Array{ComplexF64,3}    # test beam at τ=0

    # 1-D reference spectrum (Nω,)
    Eω::Vector{ComplexF64}

    # Signal window object(s) and precomputed array(s)
    window::WIN
    window_array::WA
    window_suffix::Vector{String}      # one per window in the multi-window case

    # Metadata dict, ready for Output.scansave
    combined_grid::Dict{String,Any}
end

# ============================================================================
# Field-construction primitives
# ============================================================================

"""
    build_he11_kspace(grid, xygrid, beam::HE11Beam, Eω) -> Array{ComplexF64,3}

Construct the 3-D field `E(ω, ky, kx)` for the HE₁₁ capillary mode imaged
onto the focal plane, multiplied by the 1-D spectral pulse `Eω`. Phase
ramps shift the beam from the FFTW corner to the centre of the spatial
grid.

The closed-form Hankel transform of the J₀ mode profile is used; the
`a²k² - u₁₁²` denominator is finite at all (kx, ky) sample points for
reasonable grid sizes (the singular ring is at radius `u₁₁/a`, well outside
typical Nyquist limits at the focal-plane scale).
"""
function build_he11_kspace(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
                            beam::HE11Beam, Eω::AbstractVector)
    a_s = a_scaled(beam)

    # phase ramps to shift the beam from FFTW corner (DC at index 1) to grid centre
    xshift = length(xygrid.x) * (xygrid.x[2] - xygrid.x[1]) / 2
    yshift = length(xygrid.y) * (xygrid.y[2] - xygrid.y[1]) / 2

    # HE₁₁ first transverse zero of J₁
    unm = Capillary.get_unm(1, 1, :HE)

    # |k⊥| on the (ky, kx) plane — Luna convention is (ω, ky, kx).
    k = sqrt.((xygrid.kx .^ 2)' .+ xygrid.ky .^ 2)        # (Nky, Nkx)
    k = reshape(k, (1, size(k)...))                       # (1, Nky, Nkx)

    Eωk = (-a_s^2 * unm * besselj(1, unm) .* besselj.(0, a_s .* k) ./
           (a_s^2 .* k .^ 2 .- unm^2)
           .* Eω
           .* exp.(-1im .* reshape(xygrid.ky, (1, length(xygrid.ky), 1)) .* yshift)
           .* exp.(-1im .* reshape(xygrid.kx, (1, 1, length(xygrid.kx))) .* xshift))
    Eωk
end

"""
    build_gaussian_kspace(grid, xygrid, beam::GaussianBeam,
                          λ0, τfwhm, energy) -> Array{ComplexF64,3}

Construct the 3-D field `E(ω, ky, kx)` for a Gaussian-Gaussian
spatio-temporal pulse: temporal Gaussian envelope (FWHM = `τfwhm`) at
carrier `λ0`, spatial Gaussian (1/e² radius = `beam.w0`) centred on the
grid, with total spectral energy normalised to `energy`. Internally uses
`Luna.Fields.GaussGaussField` and `Luna.setup` (with no nonlinearity) to
construct the field, then discards the throw-away transform/FT.
"""
function build_gaussian_kspace(grid::Grid.EnvGrid, xygrid::Grid.FreeGrid,
                                beam::GaussianBeam, λ0, τfwhm, energy)
    inputs = Fields.GaussGaussField(; λ0=λ0, τfwhm=τfwhm, energy=energy, w0=beam.w0)
    # Use a no-op nonlinearity setup just to get a populated Eωk array. We must
    # pass a non-empty `responses` tuple to disambiguate from the modal-setup
    # method (which matches an empty tuple as Vararg{Mode}).
    densityfun = z -> 1
    nfun_unit = (λ) -> 1.0
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun_unit)
    responses = (Nonlinear.Kerr_env(0.0),)   # χ3=0 — never evaluated
    Eωk, _, _ = Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)
    Eωk
end

"""
    apply_tilt(Eωxy, xygrid, Δkx, Δky) -> Array{ComplexF64,3}

Multiply a real-space field `E(ω, y, x)` by the phase ramp
`exp(i Δkx · x) · exp(i Δky · y)`, which shifts its centre by
`(Δky, Δkx)` in k-space (after FFT). `Δkx = Δky = 0` is the identity.
"""
function apply_tilt(Eωxy::AbstractArray{<:Complex,3}, xygrid::Grid.FreeGrid,
                    Δkx::Real, Δky::Real)
    return (Eωxy
            .* reshape(exp.(1im * Δky .* xygrid.y), (1, :, 1))
            .* reshape(exp.(1im * Δkx .* xygrid.x), (1, 1, :)))
end

"""
    apply_delay(Eωk, grid, τ) -> Array{ComplexF64,3}

Apply a time delay `τ` (seconds) to a frequency-domain field by multiplying
each spectral component by `exp(-i ω τ)`. `τ = 0` returns a copy equal to
the input.
"""
function apply_delay(Eωk::AbstractArray{<:Complex,3}, grid::Grid.EnvGrid, τ::Real)
    return Eωk .* reshape(exp.(-1im .* grid.ω .* τ), (length(grid.ω), 1, 1))
end

# ============================================================================
# Mask / window construction
# ============================================================================

"""
    makemask(holex, holey, holediam, grid, xygrid;
             zmask, apod=:supergauss, apod_param=nothing,
             λ0_for_default=nothing) -> Array{Float64,3}

Build a 3-D `(Nω, Nky, Nkx)` apodised-hole mask. For each `(ω, ky, kx)`
sample, the k-vector is mapped to the mask-plane position
`x = kx · zmask · c / ω` (and likewise for `y`), and a hole of diameter
`holediam` centred at `(holex, holey)` is evaluated.

`λ0_for_default` is only used when `apod=:tanh` and `apod_param===nothing`,
in which case the smoothing width is set to `3·Δx_mask` evaluated at the
carrier wavelength.
"""
function makemask(holex::Real, holey::Real, holediam::Real,
                  grid::Grid.EnvGrid, xygrid::Grid.FreeGrid;
                  zmask::Real,
                  apod::Symbol=:supergauss,
                  apod_param=nothing,
                  λ0_for_default=nothing)
    # Resolve default apod_param.
    if apod_param === nothing
        if apod === :supergauss
            apod_param = 16
        elseif apod === :tanh
            isnothing(λ0_for_default) && error(
                "λ0_for_default must be provided to derive a default tanh smoothing width")
            Δk = xygrid.kx[2] - xygrid.kx[1]
            ω0 = grid.ω[argmin(abs.(grid.ω .- 2π * PhysData.c / λ0_for_default))]
            Δx_mask = Δk * zmask * PhysData.c / ω0
            apod_param = 3 * Δx_mask
        end
    end

    mask = zeros(Float64, length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    @inbounds for ii in CartesianIndices(mask)
        ω  = grid.ω[ii[1]]
        ky = xygrid.ky[ii[2]]    # Luna convention: dim 2 = ky
        kx = xygrid.kx[ii[3]]    # Luna convention: dim 3 = kx
        # ω == 0 is the DC bin (often present in EnvGrid). The mapping diverges,
        # so leave the mask zero there — it won't carry any signal anyway.
        ω == 0 && continue
        x  = kx * zmask * PhysData.c / ω
        y  = ky * zmask * PhysData.c / ω
        rhole = hypot(x - holex, y - holey)
        if apod === :hard
            mask[ii] = rhole <= holediam/2 ? 1.0 : 0.0
        elseif apod === :supergauss
            mask[ii] = exp(-(2 * rhole / holediam)^apod_param)
        elseif apod === :tanh
            mask[ii] = 0.5 * (1 - tanh((rhole - holediam/2) / apod_param))
        else
            error("Unknown apod type: $apod")
        end
    end
    return mask
end

"""
    build_window(w::AbstractSignalWindow, grid, xygrid; λ0=nothing)
        -> Array{Float64, N}

Materialise the precomputed signal-extraction window. Returns a
`(Nky, Nkx)` 2-D array for [`PlanckWindow`](@ref) and a
`(Nω, Nky, Nkx)` 3-D array for [`PhysicalMaskWindow`](@ref) and
[`PlanckOmegaWindow`](@ref). `λ0` is forwarded to [`makemask`](@ref) for
default `:tanh` apodisation widths only.
"""
function build_window(w::PhysicalMaskWindow, grid::Grid.EnvGrid,
                       xygrid::Grid.FreeGrid; λ0=nothing)
    return makemask(w.holex, w.holey, w.holediam, grid, xygrid;
                    zmask=w.zmask, apod=w.apod, apod_param=w.apod_param,
                    λ0_for_default=λ0)
end

function build_window(w::PlanckWindow, grid::Grid.EnvGrid,
                       xygrid::Grid.FreeGrid; λ0=nothing)
    # Radial distance from window centre.
    κ = @. sqrt((xygrid.ky - w.kyc)^2 + (xygrid.kx' - w.kxc)^2)
    # Planck taper: flat in [0, kwidth], rolling off to zero by pad·kwidth.
    return Maths.planck_taper.(κ, -w.kwidth, -w.kwidth, w.kwidth, w.pad * w.kwidth)
end

function build_window(w::PlanckOmegaWindow, grid::Grid.EnvGrid,
                       xygrid::Grid.FreeGrid; λ0=nothing)
    win = zeros(Float64, length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    @inbounds for (iω, ω) in enumerate(grid.ω)
        ω == 0 && continue
        # Hole centre and half-width in k-space at this frequency.
        kxc   = ω / PhysData.c * w.xc / w.f_foc
        kyc   = ω / PhysData.c * w.yc / w.f_foc
        khole = ω / PhysData.c * (w.holediam/2) / w.f_foc
        for (ikx, kx) in enumerate(xygrid.kx)
            for (iky, ky) in enumerate(xygrid.ky)
                κi = sqrt((ky - kyc)^2 + (kx - kxc)^2)
                win[iω, iky, ikx] = Maths.planck_taper(
                    κi, -khole, -khole, khole, w.pad * khole)
            end
        end
    end
    return win
end

# ============================================================================
# Beamlet construction (dispatched on beam type)
# ============================================================================

"""
    build_beamlets(beam, grid, xygrid, geom, Eω, energy, energyfun_ω;
                   apod=:supergauss, apod_param=nothing)
        -> (Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_metadata::Dict)

Construct the three input beamlets `(g1, g2, t-base)` at the substrate, in
k-space. The geometry `geom` is a `NamedTuple(mask_diam, mask_spacing,
f_foc, λ0, τfwhm)` shared by both beam models.

For `HE11Beam`: builds the full HE₁₁ k-space field, rescales to the
requested energy, then applies three apodised hole masks (g1, g2, t).
Each beamlet sits at one of the boxcar corners. `Iω_beamlet` is the
spatially-integrated spectrum of `g1` (used as a chromatic-vignetting
diagnostic in the output file).

For `GaussianBeam`: builds a Gaussian-Gaussian field with energy
`energy/3` per beam, ifft's to real space, then applies real-space tilts
to position the three beams at the boxcar corners. `Iω_beamlet` here is
just the (unvignetted) input spectrum scaled to `energy/3`; it is returned
for uniformity with the HE₁₁ model so downstream code never special-cases
the beam type.
"""
function build_beamlets(beam::HE11Beam, grid::Grid.EnvGrid,
                         xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
                         energy::Real, energyfun_ω;
                         apod::Symbol=:supergauss, apod_param=nothing)
    # Full beam (no mask) in k-space, rescaled to the requested energy.
    Eωk0 = build_he11_kspace(grid, xygrid, beam, Eω)
    Eωk0 .*= sqrt(energy) / sqrt(energyfun_ω(Eωk0))

    # Hole centres at (±d, ±d), where d = mask_spacing/2 + mask_diam/2.
    d = geom.mask_spacing/2 + geom.mask_diam/2

    # Boxcar layout (looking along +z):
    #
    #   test (-x, +y)    | gate1 (+x, +y)
    #  ---------------------------------------
    #   signal (-x, -y)  | gate2 (+x, -y)
    mask_g1 = makemask( d,  d, geom.mask_diam, grid, xygrid;
                       zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                       λ0_for_default=geom.λ0)
    mask_g2 = makemask( d, -d, geom.mask_diam, grid, xygrid;
                       zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                       λ0_for_default=geom.λ0)
    mask_t  = makemask(-d,  d, geom.mask_diam, grid, xygrid;
                       zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                       λ0_for_default=geom.λ0)

    Eωk_g1     = Eωk0 .* mask_g1
    Eωk_g2     = Eωk0 .* mask_g2
    Eωk_t_base = Eωk0 .* mask_t

    # Spatially-integrated spectrum of the gate-1 beamlet — a useful diagnostic
    # showing the chromatic vignetting imprinted by the physical mask.
    Eωxy_g1 = ifft(Eωk_g1, (2, 3))
    Iω_beamlet = dropdims(sum(abs2.(Eωxy_g1); dims=(2, 3)); dims=(2, 3))

    beam_meta = Dict{String,Any}(
        "Iω_beamlet" => Iω_beamlet,
        "a"          => beam.a,
        "a_scaled"   => a_scaled(beam),
        "f_coll"     => beam.f_coll,
        "f_foc"      => beam.f_foc,
    )

    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end

function build_beamlets(beam::GaussianBeam, grid::Grid.EnvGrid,
                         xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
                         energy::Real, energyfun_ω;
                         apod::Symbol=:supergauss, apod_param=nothing)
    # In the Gaussian model each beamlet carries an equal third of the energy.
    energy_per_beam = energy / 3
    Eωk_base = build_gaussian_kspace(grid, xygrid, beam,
                                     geom.λ0, geom.τfwhm, energy_per_beam)
    Eωxy = ifft(Eωk_base, (2, 3))

    # Crossing geometry derived from the mask parameters.
    d_hole    = geom.mask_spacing/2 + geom.mask_diam/2
    crossingθ = d_hole / beam.f_foc
    Δk        = 2π / geom.λ0 * sin(crossingθ)

    # Boxcar tilts:
    #   gate 1: (+Δk_x, +Δk_y)
    #   gate 2: (+Δk_x, -Δk_y)
    #   test  : (-Δk_x, +Δk_y)
    Eωk_g1     = fft(apply_tilt(Eωxy, xygrid, +Δk, +Δk), (2, 3))
    Eωk_g2     = fft(apply_tilt(Eωxy, xygrid, +Δk, -Δk), (2, 3))
    Eωk_t_base = fft(apply_tilt(Eωxy, xygrid, -Δk, +Δk), (2, 3))

    # Spatially-integrated beamlet spectrum. The tilt is a pure phase ramp, so
    # every beamlet has the same integrated spectrum as the untilted base beam;
    # for the unmasked Gaussian model this carries no chromatic vignetting (it is
    # just the input spectrum scaled to energy/3) but is saved for uniformity
    # with the HE₁₁ model so downstream code never needs to special-case beams.
    Iω_beamlet = dropdims(sum(abs2.(Eωk_base); dims=(2, 3)); dims=(2, 3))

    beam_meta = Dict{String,Any}(
        "Iω_beamlet" => Iω_beamlet,
        "w0"         => beam.w0,
        "f_foc"      => beam.f_foc,
        "Δk"         => Δk,
        "crossingθ"  => crossingθ,
        "d_hole"     => d_hole,
    )
    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end

# ============================================================================
# Top-level constructor
# ============================================================================

"""
    build_setup(; λ0, τfwhm, energy, thickness, material,
                  mask_diam, mask_spacing,
                  beam, window,
                  trange = 40e-15,
                  λlims  = (160e-9, 500e-9),
                  R      = nothing,
                  N      = nothing,
                  apod   = :supergauss,
                  apod_param = nothing,
                  optimal_grid_kwargs = (;),
                  extra_grid_metadata = Dict{String,Any}()) -> TGFROGSetup

Build the once-per-simulation setup: temporal/spatial grids, propagation
operators, FFT plans, the three input beamlets and the signal window(s).
The defaults reproduce the master script
`context/tgfrog_DUV_mask_apod6.jl`.

# Required keyword arguments

- `λ0`, `τfwhm`, `energy`         — pulse carrier wavelength [m], intensity
                                    FWHM [s], total pulse energy [J]
- `thickness`, `material`         — substrate thickness [m] and Luna
                                    `PhysData` material symbol (e.g. `:SiO2`)
- `mask_diam`, `mask_spacing`     — mask hole diameter [m] and edge-to-edge
                                    gap [m]
- `beam::AbstractInputBeam`       — input-beam model
                                    ([`HE11Beam`](@ref) or
                                    [`GaussianBeam`](@ref))
- `window`                        — signal-extraction window: a single
                                    [`AbstractSignalWindow`](@ref) or a
                                    vector of them (the latter is used by the
                                    Gaussian example to save both the
                                    ω-independent and ω-dependent windows in
                                    one run)

# Optional keyword arguments

- `trange = 40e-15`               — temporal window [s]
- `λlims  = (160e-9, 500e-9)`     — wavelength window [m]
- `R, N`                          — spatial half-width [m] and grid size; if
                                    either is `nothing`, both are computed
                                    via [`optimal_spatial_grid`](@ref)
- `apod, apod_param`              — apodisation for the *input-beamlet*
                                    masks (only relevant for `HE11Beam`)
- `optimal_grid_kwargs`           — extra kwargs forwarded to
                                    `optimal_spatial_grid`
- `extra_grid_metadata`           — additional entries merged into the
                                    output `combined_grid` dict
"""
function build_setup(; λ0, τfwhm, energy, thickness, material,
                       mask_diam, mask_spacing,
                       beam::AbstractInputBeam,
                       window,
                       trange = 40e-15,
                       λlims  = (160e-9, 500e-9),
                       R = nothing, N = nothing,
                       apod::Symbol = :supergauss, apod_param = nothing,
                       optimal_grid_kwargs = (;),
                       extra_grid_metadata = Dict{String,Any}())

    # --- Resolve spatial grid ----------------------------------------------
    if R === nothing || N === nothing
        f_foc = beam.f_foc
        R, N = optimal_spatial_grid(f_foc, mask_diam, mask_spacing,
                                     λlims[1], λlims[2]; optimal_grid_kwargs...)
    end

    # --- Build Luna grids --------------------------------------------------
    grid = Grid.EnvGrid(thickness, λ0, λlims, trange)
    xygrid = Grid.FreeGrid(R, N)

    # --- Material dispersion + Kerr nonlinearity --------------------------
    χ3 = PhysData.χ3(material)
    responses = (Nonlinear.Kerr_env(χ3),)
    nfun = PhysData.ref_index_fun(material)
    nfunreal = (λ) -> real(nfun(λ))
    linop = LinearOps.make_const_linop(grid, xygrid, nfunreal)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfunreal)
    densityfun = z -> 1
    _, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses, ())
    _, energyfun_ω = Fields.energyfuncs(grid, xygrid)

    # --- 1-D reference spectrum (used by the HE₁₁ builder and as diagnostic)
    FT1d = plan_fft(copy(grid.t))
    Eω = Fields.GaussField(; λ0=λ0, τfwhm=τfwhm, energy=energy)(grid, FT1d)

    # --- Build three input beamlets ---------------------------------------
    geom = (; mask_diam, mask_spacing, f_foc=beam.f_foc, λ0, τfwhm)
    Eωk_g1, Eωk_g2, Eωk_t_base, _Iω_beamlet, beam_meta =
        build_beamlets(beam, grid, xygrid, geom, Eω, energy, energyfun_ω;
                       apod=apod, apod_param=apod_param)

    # --- Build signal window(s) -------------------------------------------
    window_array, window_suffix = _build_window_set(window, grid, xygrid; λ0=λ0)

    # --- Assemble combined_grid metadata ----------------------------------
    combined_grid = _combined_grid(grid, xygrid, beam_meta,
                                    window, window_array, window_suffix,
                                    Eω, λ0, τfwhm, material, thickness,
                                    extra_grid_metadata)

    return TGFROGSetup{typeof(linop),typeof(transform),typeof(FT),
                       typeof(window),typeof(window_array)}(
        λ0, τfwhm, energy, thickness, material, mask_diam, mask_spacing,
        grid, xygrid, linop, transform, FT, energyfun_ω,
        Eωk_g1, Eωk_g2, Eωk_t_base, Eω,
        window, window_array, window_suffix, combined_grid)
end

# ----- Window-set helper (single vs vector of windows) ---------------------

_build_window_set(w::AbstractSignalWindow, grid, xygrid; λ0) =
    (build_window(w, grid, xygrid; λ0=λ0), [""])

function _build_window_set(ws::AbstractVector{<:AbstractSignalWindow},
                            grid, xygrid; λ0)
    arrs = [build_window(w, grid, xygrid; λ0=λ0) for w in ws]
    suffixes = _default_suffixes(ws)
    return arrs, suffixes
end

# Default suffixes: first window gets "", subsequent windows get "_2", "_3", ...
# unless we recognise the gaussian (Planck + PlanckOmega) two-window pattern,
# in which case we emit ["", "_ωdep"] for compatibility with the master script.
function _default_suffixes(ws::AbstractVector{<:AbstractSignalWindow})
    if length(ws) == 2 && ws[1] isa PlanckWindow && ws[2] isa PlanckOmegaWindow
        return ["", "_ωdep"]
    end
    return [i == 1 ? "" : "_$i" for i in eachindex(ws)]
end

# ----- combined_grid metadata helper ---------------------------------------

function _combined_grid(grid, xygrid, beam_meta::Dict,
                         window, window_array, window_suffix::Vector{String},
                         Eω::AbstractVector, λ0, τfwhm, material, thickness,
                         extra::Dict)
    cg = Dict{String,Any}()
    for (k, v) in pairs(Grid.to_dict(grid))
        cg[string(k)] = v
    end
    for (k, v) in pairs(Grid.to_dict(xygrid))
        cg[string(k)] = v
    end

    # Always-present diagnostics.
    It = abs2.(ifft(Eω))
    Iω = abs2.(Eω)
    to, eo = Maths.oversample(grid.t, ifft(Eω); factor=8)
    Ito = abs2.(eo)
    cg["Iω"]        = Iω
    cg["It"]        = It
    cg["To"]        = to
    cg["Ito"]       = Ito
    cg["τfwhm"]     = τfwhm
    cg["material"]  = string(material)
    cg["thickness"] = thickness

    # Beam-specific metadata (Iω_beamlet for both models, w0/Δk/crossingθ for
    # Gaussian, a/f_coll/f_foc for HE₁₁).
    for (k, v) in beam_meta
        cg[k] = v
    end

    # Time-domain profile of the (input-vignetted) beamlet — "the pulse that is
    # actually retrieved" when a measured trace is inverted. The mask is a real
    # amplitude filter, so the beamlet's effective spectral phase is the input
    # pulse phase: we reconstruct the beamlet envelope by combining the saved
    # power spectrum Iω_beamlet with the input spectral phase, then transform to
    # time. (For an FTL input this is just the transform-limited profile of the
    # vignetted beamlet spectrum.)
    if haskey(cg, "Iω_beamlet")
        Iω_beamlet = cg["Iω_beamlet"]
        reg   = maximum(Iω) * 1e-12
        phase = [Iω[i] > reg ? Eω[i] / sqrt(Iω[i]) : zero(eltype(Eω))
                 for i in eachindex(Eω)]
        Eω_beamlet = sqrt.(max.(Iω_beamlet, 0)) .* phase
        et_beamlet = ifft(Eω_beamlet)
        cg["It_beamlet"] = abs2.(et_beamlet)
        _, eob_beamlet = Maths.oversample(grid.t, et_beamlet; factor=8)
        cg["Ito_beamlet"] = abs2.(eob_beamlet)   # shares the "To" grid above
    end

    # Window arrays under "window" (+ optional suffixes for multi-window).
    if window isa AbstractSignalWindow
        cg["window"] = window_array
    else
        for (suf, arr) in zip(window_suffix, window_array)
            cg["window" * suf] = arr
        end
    end

    # User-supplied extras override anything above on collision.
    for (k, v) in extra
        cg[k] = v
    end
    return cg
end

# ============================================================================
# Per-delay simulation
# ============================================================================

"""
    extract_signal_spectra(Eωk_out, window_array, xygrid)
        -> (Iω_integrated, Iω_reimaged)

Apply a precomputed signal window to the output field of `Luna.run`
(shape `(Nω, Nky, Nkx, Nz)`) and extract two spectral diagnostics:

1. `Iω_integrated` — `|E|²` summed over all (ky, kx); shape `(Nω, Nz)`.
   Models a spectrometer collecting **all** the signal light.
2. `Iω_reimaged`   — `|E|²` at the centre pixel of the IFFT'd field;
   shape `(Nω, Nz)`. Models a spectrometer fed only by the on-axis
   re-collimated signal.

`window_array` is broadcast over `ω` (if 2-D) or matched directly (if 3-D),
and over the `Nz` z-slices in either case.
"""
function extract_signal_spectra(Eωk_out::AbstractArray{<:Complex,4},
                                 window_array::AbstractArray{<:Real},
                                 xygrid::Grid.FreeGrid)
    if ndims(window_array) == 2
        Eωk_win = Eωk_out .* reshape(window_array, (1, size(window_array)..., 1))
    elseif ndims(window_array) == 3
        Eωk_win = Eωk_out .* reshape(window_array, (size(window_array)..., 1))
    else
        error("window_array must be 2-D or 3-D, got $(ndims(window_array))")
    end
    Eωxy_win = ifft(Eωk_win, (2, 3))
    Iω_reimaged = abs2.(Eωxy_win[:, length(xygrid.y) ÷ 2 + 1,
                                  length(xygrid.x) ÷ 2 + 1, :])
    Iω_integrated = dropdims(sum(abs2.(Eωk_win); dims=(2, 3)); dims=(2, 3))
    return Iω_integrated, Iω_reimaged
end

"""
    _resolve_zsave(zsave, zmax) -> Vector{Float64}

Resolve the `zsave` propagation-snapshot specification into a validated, sorted
vector of z positions [m] at which the field is saved during propagation.

- `zsave::Integer` — a uniform grid of `zsave` points over `[0, zmax]`
  (`range(0, zmax, zsave)`), reproducing the legacy `nz` behaviour exactly
  (including the entrance slice at `z=0` and the exit slice at `z=zmax`).
- `zsave::AbstractVector` — explicit material thicknesses [m]. Must be strictly
  increasing, all `>= 0`, and all `<= zmax`. `zmax` is appended if not already
  present (within `rtol=1e-12`) so the full-thickness ("`:end`") slice always
  exists.

Because the propagation is a forward-marching integrator with z-independent
dynamics, the field saved at an intermediate `z` is identical to a dedicated run
of thickness `z`, so a single `zmax` run yields every shorter thickness for free.

The function is idempotent: re-resolving an already-resolved vector (which the
integer path produces *with* an entrance slice at `z=0`) returns it unchanged,
so it is safe to call more than once on the same grid.
"""
function _resolve_zsave(zsave::Integer, zmax::Real)
    zsave >= 2 || throw(ArgumentError("integer zsave must be ≥ 2, got $zsave"))
    return collect(range(0.0, zmax, zsave))
end

function _resolve_zsave(zsave::AbstractVector, zmax::Real)
    v = collect(Float64, zsave)
    isempty(v) && throw(ArgumentError("zsave vector must be non-empty"))
    issorted(v) && allunique(v) ||
        throw(ArgumentError("zsave must be strictly increasing, got $v"))
    all(>=(0.0), v) ||
        throw(ArgumentError("all zsave positions must be ≥ 0, got $v"))
    vmax = maximum(v)
    vmax <= zmax || throw(ArgumentError(
        "zsave position $vmax exceeds the propagation distance zmax=$zmax"))
    if !isapprox(v[end], zmax; rtol=1e-12)
        push!(v, zmax)
    end
    return v
end

"""
    simulate_delay_point(setup::TGFROGSetup, τi;
                         nz=2, init_dz=5e-7, skip_propagation=false)
        -> NamedTuple

Run the full per-delay computation: apply delay `τi` to the test beam,
coherently superpose the three beamlets, propagate them through the
substrate via `Luna.run`, apply each signal window and extract two
spectra per window. The returned `NamedTuple` has, for a single window,
fields `(Iω_win, Iω_win_reimaged, Iω_full)`. For a vector of windows the
suffixes recorded in `setup.window_suffix` are appended (e.g. `Iω_win_ωdep`,
`Iω_win_ωdep_reimaged`), and the single `Iω_full` is shared. All extracted
arrays have shape `(Nω, nz)`. The returned NamedTuple also carries `zsave`, the
vector of realized z save positions [m] (length `nz`) — this is metadata, not a
per-delay trace, and is excluded from the `scansave` dataset splat by [`run_scan`](@ref).

`zsave` selects the propagation snapshots. Pass an `Integer` for a uniform grid
of that many points over `[0, zmax]` (default `nz`), or a `Vector` of explicit
material thicknesses [m] (e.g. `[1e-6, 10e-6, 20e-6, 40e-6]`); `zmax` is appended
to the vector if absent. Because the field at an intermediate `z` equals a
dedicated thickness-`z` run, every shorter thickness comes free from one `zmax`
run. Peak memory scales with `nz` (the in-memory 4-D field is held per slice).

`Iω_full` is the signal beam collected in full: `|E|²` integrated over the
signal's k-space quadrant only. The propagated field holds the three strong
pump beamlets (at the g1/g2/test boxcar corners) plus the weak FWM signal at
the fourth corner; integrating over *all* of k-space would be dominated by the
pumps, so we restrict to the quadrant the signal occupies (`kx<0, ky<0`),
which captures the whole signal lobe without aperture vignetting while
excluding the pumps. `Iω_win ./ Iω_full` is therefore the exact per-(ω, τ)
collection / chromatic-vignetting efficiency of the signal aperture, so the
trace can be corrected for collection vignetting exactly rather than via a
power-law approximation. (This assumes the boxcar beams are well separated, so
pump tails leaking into the signal quadrant are negligible vs. the signal.)

Setting `skip_propagation=true` substitutes the input field for the
Luna output, exercising every other code path. This is used by the unit
tests to keep the suite fast and deterministic.
"""
function simulate_delay_point(setup::TGFROGSetup, τi::Real;
                              nz::Int=2,
                              zsave::Union{Integer,AbstractVector}=nz,
                              init_dz::Float64=5e-7,
                              skip_propagation::Bool=false)
    # --- Resolve the propagation snapshot grid ---------------------------
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    nz_eff = length(zvec)

    # --- Build the delayed test beam and coherently superpose ------------
    Eωk_t = apply_delay(setup.Eωk_t_base, setup.grid, τi)
    Eωk_in = setup.Eωk_g1 .+ setup.Eωk_g2 .+ Eωk_t

    # --- Propagate (or fake the propagation for tests) -------------------
    if skip_propagation
        # Fake a (Nω, Nky, Nkx, nz) output by stacking the input nz times.
        Nω, Nky, Nkx = size(Eωk_in)
        Eωk_out = Array{ComplexF64}(undef, Nω, Nky, Nkx, nz_eff)
        @inbounds for iz in 1:nz_eff
            Eωk_out[:, :, :, iz] = Eωk_in
        end
        z_realized = copy(zvec)
    else
        output = Output.MemoryOutput(Output.GridCondition(zvec, nz_eff), "Eω", "z")
        Luna.run(Eωk_in, setup.grid, setup.linop, setup.transform, setup.FT,
                  output; init_dz=init_dz)
        Eωk_out = output["Eω"]
        z_realized = output["z"]
    end

    # --- Full signal-beam collection (no aperture crop) ------------------
    # The propagated field contains the three strong pump beamlets (at the
    # g1/g2/test boxcar corners) plus the weak FWM signal at the fourth,
    # signal corner. Integrating over ALL of k-space would be dominated by the
    # pumps, so instead we integrate |E|² over the signal's k-space quadrant
    # only: by the fixed boxcar layout (g1=(d,d), g2=(d,-d), test=(-d,d)) the
    # signal sits at (kx<0, ky<0) while all three pumps occupy the other
    # quadrants. This captures the whole signal lobe with no aperture
    # vignetting and excludes the pumps, so Iω_win ./ Iω_full is the exact
    # per-(ω, τ) collection / chromatic-vignetting efficiency of the signal
    # aperture — no power-law approximation. Window-independent, computed once.
    # Shape (Nω, nz). (Luna array dims are (ω, ky, kx, z).)
    sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
    Iω_full = dropdims(sum(abs2.(Eωk_out) .*
                           reshape(sig_quad, (1, size(sig_quad)..., 1));
                           dims=(2, 3)); dims=(2, 3))

    # --- Extract spectra per window --------------------------------------
    if setup.window isa AbstractSignalWindow
        Iω_w, Iω_r = extract_signal_spectra(Eωk_out, setup.window_array,
                                              setup.xygrid)
        return (; Iω_win=Iω_w, Iω_win_reimaged=Iω_r, Iω_full, zsave=z_realized)
    else
        pairs_kv = Pair{Symbol,Any}[]
        for (suf, arr) in zip(setup.window_suffix, setup.window_array)
            Iω_w, Iω_r = extract_signal_spectra(Eωk_out, arr, setup.xygrid)
            push!(pairs_kv, Symbol("Iω_win" * suf)         => Iω_w)
            push!(pairs_kv, Symbol("Iω_win" * suf * "_reimaged") => Iω_r)
        end
        push!(pairs_kv, :Iω_full => Iω_full)
        push!(pairs_kv, :zsave   => z_realized)
        return NamedTuple(pairs_kv)
    end
end

# ============================================================================
# High-level scan orchestrator
# ============================================================================

"""
    run_scan(setup, τs;
             scan_name, exec,
             nz=2, zsave=nz, init_dz=5e-7,
             extra_outputs=(out)->NamedTuple()) -> Nothing

Build a `Luna.Scans.Scan` over the delay array `τs` and run
[`simulate_delay_point`](@ref) at every τ, calling `Output.scansave` to
write each result into the collected HDF5 file at
`"<scan_name>_collected.h5"`. The metadata block (`combined_grid`) is
written once on the first scan point.

`exec` must be a `Luna.Scans.AbstractExec` instance (e.g.
`Scans.SlurmExec(...)` or `Scans.LocalExec()`).

`zsave` selects the propagation snapshots saved at every delay (see
[`simulate_delay_point`](@ref)): an `Integer` gives a uniform grid of that many
points over `[0, thickness]` (default `nz`), or a `Vector` of explicit material
thicknesses [m] (e.g. `[1e-6, 10e-6, 20e-6, 40e-6]`). `thickness` is appended to
the vector if absent so the final slice is always the full-propagation output.
The trace datasets become `(Nω, nz, Nτ)` and the realized z positions are stored
once in `/grid/zsave`. Because the field at an intermediate z equals a dedicated
thickness-z run, every shorter thickness comes free from one full-thickness run;
note that peak memory scales with the number of z points.

`extra_outputs(output_namedtuple)` — optional escape hatch returning extra
named tuples to splat into `scansave`. The default is empty.
"""
function run_scan(setup::TGFROGSetup, τs::AbstractVector;
                  scan_name::AbstractString,
                  exec,
                  nz::Int=2, zsave::Union{Integer,AbstractVector}=nz,
                  init_dz::Float64=5e-7,
                  extra_outputs::Function=(out)->NamedTuple())
    # Resolve the z grid up front and persist it once in the metadata. The
    # resolution is deterministic and saves land exactly on these points, so
    # /grid/zsave equals the per-point realized `out.zsave`. Use a shallow copy
    # so the shared `setup.combined_grid` is not mutated.
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    cg = copy(setup.combined_grid)
    cg["zsave"] = zvec

    scan = Scans.Scan(scan_name, exec; τ=τs)
    Luna.runscan(scan) do scanidx, τi
        out = simulate_delay_point(setup, τi; zsave=zvec, init_dz=init_dz)
        # `zsave` is metadata (stored in /grid/zsave), not a per-delay dataset.
        out_save = Base.structdiff(out, NamedTuple{(:zsave,)})
        Output.scansave(scan, scanidx; grid=cg, out_save...,
                         extra_outputs(out)...)
    end
    return nothing
end

# ============================================================================
# Loading and post-processing scan output files
# ============================================================================
#
# `run_scan` writes one HDF5 file per delay scan via `Output.scansave`. The
# file structure is:
#
#   /scanvariables/τ              the FROG delay axis (Nτ,) [s]
#   /grid/ω                       absolute angular frequency (FFT-ordered)
#   /grid/t, /grid/ω0, /grid/Iω, /grid/It, /grid/τfwhm, ...
#   /grid/Iω_beamlet              input-vignetted beamlet spectrum
#   /grid/It_beamlet, /grid/Ito_beamlet   beamlet temporal intensity (+ oversampled)
#   /grid/window, /grid/window_ωdep   precomputed signal mask(s)
#   /grid/zsave                   (nz,) realized propagation z positions [m]
#   /Iω_win                       (Nω, nz, Nτ) integrated FROG trace
#   /Iω_win_reimaged              (Nω, nz, Nτ) on-axis re-imaged trace
#   /Iω_full                      (Nω, nz, Nτ) full signal-beam collection (signal quadrant)
#   /Iω_win_ωdep, /Iω_win_ωdep_reimaged    Gaussian two-window extras
#
# `load_simulated_scan` extracts the chosen window/z-slice, fftshifts ω-
# dependent arrays into natural (centred) order, and returns a NamedTuple
# ready for inspection, plotting, or downstream processing.

"""
    load_simulated_scan(filename; window_key="Iω_win", z_index=:end,
                        z_thickness=nothing) -> NamedTuple

Read the raw HDF5 file produced by [`run_scan`](@ref) and return its
contents as a NamedTuple, with all ω-dependent arrays fftshifted into
natural (centred) order and the requested z slice(s) extracted from the
propagated trace.

# Arguments
- `filename`: path to the `<scan_name>_collected.h5` file.

# Keyword arguments
- `window_key="Iω_win"`: which scansave dataset to use as the FROG trace.
  Common choices:
    * `"Iω_win"` — full-beam k-space integrated spectrum
    * `"Iω_win_reimaged"` — on-axis re-imaged spectrum
    * `"Iω_win_ωdep"` — ω-dependent window (Gaussian two-window setup)
    * `"Iω_win_ωdep_reimaged"` — ω-dependent re-imaged
- `z_index=:end`: which propagation z slice to use; the default `:end`
  picks the final (full-propagation) slice. Pass an `Int` for a specific
  slice index, or `:all` to return *every* z slice as a `(Nω, nz, Nτ)`
  stack (the equivalent of the trace at every saved material thickness).
- `z_thickness=nothing`: select the slice whose saved z position [m] is
  nearest this material thickness. Requires `/grid/zsave` in the file
  (written by recent `run_scan` runs); takes precedence over `z_index`.

# Returned NamedTuple

| field         | shape          | description                                          |
|---------------|----------------|------------------------------------------------------|
| `ω`           | `(Nω,)`        | absolute angular frequency [rad/s], natural order    |
| `ω0`          | scalar         | carrier angular frequency [rad/s] (from `/grid/ω0`)  |
| `t`           | `(Nt,)`        | time grid [s]                                        |
| `τ`           | `(Nτ,)`        | scan-variable delay grid [s]                         |
| `trace`       | `(Nω, Nτ)` or `(Nω, nz, Nτ)` | FROG trace, natural ω order; 3-D when `z_index=:all` |
| `zsave`       | `(nz,)`        | realized propagation z positions [m] (when `/grid/zsave` present) |
| `Iω`          | `(Nω,)`        | reference pulse spectrum, natural ω order            |
| `It`          | `(Nt,)`        | reference pulse temporal intensity                   |
| `τfwhm`       | scalar         | input pulse FWHM [s]                                 |
| `Iω_beamlet`  | `(Nω,)`        | input-vignetted beamlet spectrum (the retrievable pulse) |
| `It_beamlet`  | `(Nt,)`        | beamlet temporal intensity (when `/grid/It_beamlet` present) |
| `Ito_beamlet` | `(Nto,)`       | 8× oversampled beamlet temporal intensity (shares `To`) |
| `To`          | `(Nto,)`       | 8× oversampled time grid [s] (when `/grid/To` present)  |
| `Ito`         | `(Nto,)`       | 8× oversampled temporal intensity (when `/grid/Ito` present) |

To inspect the full signal-beam collection (and hence the exact collection /
chromatic-vignetting efficiency `Iω_win ./ Iω_full`), load the signal-quadrant
reference with `window_key="Iω_full"`.
"""
function load_simulated_scan(filename::AbstractString;
                              window_key::AbstractString="Iω_win",
                              z_index=:end,
                              z_thickness::Union{Nothing,Real}=nothing)
    HDF5.h5open(filename, "r") do f
        # --- Grid block ---
        haskey(f, "grid") || error("$filename: missing /grid group (not a scansave file?)")
        g = f["grid"]
        ω_raw   = read(g["ω"])
        ω0      = read(g["ω0"])
        t       = read(g["t"])
        Iω_raw  = read(g["Iω"])
        It      = read(g["It"])
        τfwhm   = read(g["τfwhm"])
        Iω_beam_raw = haskey(g, "Iω_beamlet") ? read(g["Iω_beamlet"]) : nothing
        It_beam_raw = haskey(g, "It_beamlet")  ? read(g["It_beamlet"])  : nothing
        Ito_beam_raw= haskey(g, "Ito_beamlet") ? read(g["Ito_beamlet"]) : nothing
        To_raw      = haskey(g, "To")  ? read(g["To"])  : nothing
        Ito_raw     = haskey(g, "Ito") ? read(g["Ito"]) : nothing
        zsave       = haskey(g, "zsave") ? read(g["zsave"]) : nothing

        # --- Scan variable ---
        haskey(f, "scanvariables") && haskey(f["scanvariables"], "τ") ||
            error("$filename: missing /scanvariables/τ")
        τ = read(f["scanvariables"]["τ"])

        # --- Trace ---
        if !haskey(f, window_key)
            available = filter(k -> !(k in ("grid", "scanvariables", "scanorder")),
                                keys(f))
            error("$filename: window_key '$window_key' not found. " *
                  "Available top-level datasets: $available")
        end
        win_full = read(f[window_key])    # shape (Nω, nz, Nτ)
        nz = size(win_full, 2)

        # --- Select z slice(s): z_thickness > z_index ---
        if z_thickness !== nothing
            isnothing(zsave) && error("$filename: z_thickness requested but the " *
                "file has no /grid/zsave (run produced with an older run_scan)")
            z_idx = argmin(abs.(zsave .- z_thickness))
            win = win_full[:, z_idx, :]                  # (Nω, Nτ)
        elseif z_index === :all
            win = win_full                               # (Nω, nz, Nτ)
        else
            z_idx = z_index === :end ? nz : Int(z_index)
            (1 <= z_idx <= nz) ||
                error("$filename: z_index=$z_idx out of range (nz=$nz)")
            win = win_full[:, z_idx, :]                  # (Nω, Nτ)
        end

        # --- Apply fftshift along the ω axis (dim 1) ---
        ω           = FFTW.fftshift(ω_raw)
        Iω          = FFTW.fftshift(Iω_raw)
        trace       = FFTW.fftshift(win, 1)
        Iω_beamlet  = isnothing(Iω_beam_raw) ? nothing : FFTW.fftshift(Iω_beam_raw)

        nt = (; ω, ω0, t, τ, trace, Iω, It, τfwhm)
        nt = isnothing(zsave)       ? nt : merge(nt, (; zsave))
        nt = isnothing(Iω_beamlet)  ? nt : merge(nt, (; Iω_beamlet))
        nt = isnothing(It_beam_raw) ? nt : merge(nt, (; It_beamlet=It_beam_raw))
        nt = isnothing(Ito_beam_raw) ? nt : merge(nt, (; Ito_beamlet=Ito_beam_raw))
        isnothing(To_raw) ? nt : merge(nt, (; To=To_raw, Ito=Ito_raw))
    end
end

end # module
