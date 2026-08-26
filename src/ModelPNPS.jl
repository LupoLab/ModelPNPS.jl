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

# Two field representations

By default the propagation is Luna's complex **envelope** (`Grid.EnvGrid`): an analytic
field about a carrier. `build_setup(field_mode=true)` instead propagates the real,
carrier-resolved field on a `Grid.RealGrid`, which has no envelope/carrier split, no
dropped third-harmonic term and no negative-frequency wrap. That matters when the pulse is
only a cycle or two long — at 260 nm a 1 fs pulse is 1.15 optical cycles — where the
envelope approximation is marginal by construction and the two representations can be
compared directly. It costs roughly twice the memory and three times the time per delay point,
so it is a diagnostic, not the production default. See `build_setup`'s `field_mode`,
`response` and `ffac` keywords.

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
             Output, PhysData, Raman, Scans
import Luna.Capillary: besselj
import FFTW
import FFTW: fft, ifft, irfft, plan_fft, plan_rfft
import HDF5
import LinearAlgebra: mul!
import Adapt
import Statistics: mean

export AbstractInputBeam, HE11Beam, GaussianBeam,
       AbstractSignalWindow, PhysicalMaskWindow, PlanckWindow, PlanckOmegaWindow,
       TGFROGSetup,
       optimal_spatial_grid,
       build_he11_kspace, build_gaussian_kspace,
       apply_tilt, apply_delay,
       makemask, build_window,
       build_setup, simulate_delay_point, run_scan, memory_budget,
       signal_quadrant_norm,
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
- `margin=1.1`: multiplier on the resolved grid size before rounding up to the
  next even 2,3,5-smooth FFT size (guards the containment against grid
  quantisation).
"""
function optimal_spatial_grid(f, mask_diam, mask_spacing, λmin, λmax;
                              n_airy=5, pts_per_lobe=10, safety=1.5, margin=1.1,
                              geometry::Symbol=:tg)
    # Outermost extent of the nonlinear k-content, measured in the mask plane.
    #
    # :tg  four-hole boxcar. Holes at (+-d, +-d) with d = spacing/2 + diam/2, so
    #      the outermost edge is spacing/2 + diam, and chi(3) combinations reach
    #      three times the hole offset (the 3x below).
    # :sd  two holes on ONE axis at +-s/2, s = spacing + diam. Self-diffraction
    #      puts the signal at 2k_1 - k_2, i.e. at 3s/2 from the axis — one
    #      further slot out than the beams themselves — and that, plus a beam
    #      radius, is the true bound. It is quoted directly rather than as 3x
    #      an inner radius, so no extra factor of three is applied.
    x_max = geometry === :sd ? 1.5 * (mask_spacing + mask_diam) + mask_diam/2 :
                               mask_spacing/2 + mask_diam
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
    k_NL_max = safety * (geometry === :sd ? 1 : 3) * 2π * x_max / (λmin * f)
    N_from_kspace = 2 * R_min * k_NL_max / π
    @info "N from k-space containment" N_from_kspace

    # 2,3,5-smooth FFT sizes are as fast as powers of two for FFTW but track the
    # requirement much more closely: nextpow(2, ...) rounded 576 up to 1024 (3.2× the
    # memory and FFT work of 576), whereas nextprod with a 10% margin gives 640.
    # `margin` guards the containment against grid quantisation on top of `safety`.
    # N must be even: the grid layout (x = (n - N/2)δx) and the centre-pixel signal
    # extraction both assume it.
    Nmin = ceil(Int, margin * max(N_from_realspace, N_from_kspace))
    N = nextprod([2, 3, 5], Nmin)
    while isodd(N)
        N = nextprod([2, 3, 5], N + 1)
    end
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
struct TGFROGSetup{GT,LO,TR,FTT,WIN,WA,BT,PT}
    # Physical / numerical parameters echoed for output metadata
    λ0::Float64
    τfwhm::Float64
    energy::Float64
    thickness::Float64
    material::Symbol
    mask_diam::Float64
    mask_spacing::Float64

    # Luna grids. `grid` is an `EnvGrid` in the default (envelope) mode and a `RealGrid`
    # in field mode — see `build_setup`'s `field_mode`. It is a type parameter rather than
    # an abstractly-typed field so that every method reading `setup.grid` stays inferrable.
    grid::GT
    xygrid::Grid.FreeGrid

    # Pre-built propagation pieces
    linop::LO
    transform::TR
    FT::FTT
    energyfun_ω::Function

    # Pre-built input beamlets, all in k-space (Nω, Nky, Nkx). The gate pair
    # is stored pre-summed: the two gates are only ever used together
    # (delayed_input), and one array instead of two saves a full field copy
    # (2.15 GB at production size).
    # These follow the propagation's array type: on a GPU run they are device-resident
    # unless `beamlets_on_host=true`, which keeps them here and uploads the delayed sum
    # once per delay point instead (2 fewer device fields, one transfer per point).
    Eωk_g12::BT                        # gate pair g1 + g2 (no delay)
    Eωk_t_base::BT                     # test beam at τ=0

    # Delay phase ramp exp(-iωτ) support: the ω axis on the propagation's array type, so
    # `delayed_input` can build the ramp without a host/device mismatch
    ωd::PT

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
function build_he11_kspace(grid::Grid.TimeGrid, xygrid::Grid.FreeGrid,
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

    # The transverse amplitude depends only on (ky, kx): evaluate the Bessel factor on
    # the (1, Nky, Nkx) plane once instead of at all Nω×Nky×Nkx points (the subsequent
    # broadcast chain is unchanged, so the result is identical)
    A = (-a_s^2 * unm * besselj(1, unm) .* besselj.(0, a_s .* k) ./
         (a_s^2 .* k .^ 2 .- unm^2))
    Eωk = (A
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
function build_gaussian_kspace(grid::Grid.TimeGrid, xygrid::Grid.FreeGrid,
                                beam::GaussianBeam, λ0, τfwhm, energy)
    inputs = Fields.GaussGaussField(; λ0=λ0, τfwhm=τfwhm, energy=energy, w0=beam.w0)
    # Use a no-op nonlinearity setup just to get a populated Eωk array. We must
    # pass a non-empty `responses` tuple to disambiguate from the modal-setup
    # method (which matches an empty tuple as Vararg{Mode}).
    densityfun = z -> 1
    nfun_unit = (λ) -> 1.0
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun_unit)
    # χ3 = 0, so never evaluated — but it must MATCH the grid's field type, because
    # `Luna.setup` builds the transform's buffers from it.
    responses = (_is_field_mode(grid) ? Nonlinear.Kerr_field(0.0) :
                                        Nonlinear.Kerr_env(0.0),)
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
function apply_delay(Eωk::AbstractArray{<:Complex,3}, grid::Grid.TimeGrid, τ::Real)
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
                  grid::Grid.TimeGrid, xygrid::Grid.FreeGrid;
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
function build_window(w::PhysicalMaskWindow, grid::Grid.TimeGrid,
                       xygrid::Grid.FreeGrid; λ0=nothing)
    return makemask(w.holex, w.holey, w.holediam, grid, xygrid;
                    zmask=w.zmask, apod=w.apod, apod_param=w.apod_param,
                    λ0_for_default=λ0)
end

function build_window(w::PlanckWindow, grid::Grid.TimeGrid,
                       xygrid::Grid.FreeGrid; λ0=nothing)
    # Radial distance from window centre.
    κ = @. sqrt((xygrid.ky - w.kyc)^2 + (xygrid.kx' - w.kxc)^2)
    # Planck taper: flat in [0, kwidth], rolling off to zero by pad·kwidth.
    return Maths.planck_taper.(κ, -w.kwidth, -w.kwidth, w.kwidth, w.pad * w.kwidth)
end

function build_window(w::PlanckOmegaWindow, grid::Grid.TimeGrid,
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
# Beamlet focal profile (diagnostic)
# ============================================================================

"""
    _beamlet_profile(grid, xygrid, Eωk, holex, holey, zmask; nr, rmax, nθ=64)
        -> (r, Eωr, asym)

The spatially resolved complex focal field of one beamlet, reduced to a radial profile
`Eωr[ω, r]` about its own centre, with the radius axis `r` in metres and a per-ω measure
`asym` of how well the radial reduction describes it.

Diagnostic only: nothing here feeds the propagation.

# Where the beamlet actually is

**In this representation the beamlets do not sit at BOXCARS corners in real space — they
all cross at the focus, and the corners are in k-space.** `build_he11_kspace` builds the
field in `(ω, ky, kx)` with the transverse amplitude the Hankel transform of the HE₁₁ mode,
so k-space is the COLLIMATED (mask) plane — `makemask` maps `x = kx·zmask·c/ω` — and real
space, after `ifft` over dims 2 and 3, is the FOCAL plane. A hole at mask position
`(holex, holey)` therefore selects k around `k₀ = hole·ω/(c·zmask)`, and in the focal plane
that offset is a **tilt**, not a displacement: measured on the production geometry, the
gate beamlet peaks at the real-space grid centre to the pixel, and carries a phase slope of
2.402e5 rad/m against the predicted `k₀ = 2.417e5`.

So the centre of this profile is the grid centre. Centring it on the mask-hole position
mapped through the focus — 1 mm out, against a 26 µm spot — would sample nothing.

# Why the tilt is removed first

`k₀·r` reaches **37 rad** across the default sampling radius, so an azimuthal average of the
raw complex field would annihilate it. The field is demodulated by `exp(-i k₀(ω)·r)` before
sampling, leaving the beamlet's own envelope. `k₀ ∝ ω`, so the coefficient is a constant
`hole/(c·zmask)` in s/m; it is stored, and multiplying the profile by `exp(+iω(cₓx + c_yy))`
restores the full field. Note that the removed tilt is physical — a linear delay across the
beamlet, i.e. the pulse-front tilt of the crossing geometry — not an artefact.

# Accuracy

Sampled by bilinear interpolation on `nr × nθ` polar points (the focal spot is ~27 grid
cells across, so the interpolation is not the limiting error) and averaged azimuthally.
`asym` is the azimuthal RMS of `|E|` over its mean, restricted to radii carrying signal:
1.2–3.8 % on the production geometry, i.e. the radial reduction is a good description but
not an exact one, and a consumer can see how good.

Integrating `|Eωr|²` with the `2πr dr` Jacobian and dividing by the cell area reproduces
the stored `Iω_beamlet` to ~1.5 % at the default `nr = 64` — the shortfall is the Airy
wings beyond `rmax` plus that asymmetry, and it converges (0.980/0.987/0.988 at nr = 128
against 0.975/0.984/0.986 at 64, at 200/260/350 nm). The test suite asserts this closure,
which is the check that the centre, the Jacobian and the normalisation are all right.
"""
function _beamlet_profile(grid, xygrid, Eωk::AbstractArray{<:Complex,3},
                          holex::Real, holey::Real, zmask::Real;
                          nr::Int, rmax::Real, nθ::Int=64)
    Nω = length(grid.ω)
    Ny, Nx = length(xygrid.y), length(xygrid.x)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    # the beamlets cross at the focus, which the phase ramps in build_he11_kspace put at
    # the middle of the real-space grid
    cy, cx = Ny ÷ 2 + 1, Nx ÷ 2 + 1
    # The requested radius can exceed the transverse grid (a small R, or a large
    # rmax_units). Sampling past the edge would silently repeat the boundary value, so
    # clamp — `beamlet_r` then records what was actually covered, and
    # `beamlet_r_max_requested` what was asked for, which is how a reader tells.
    rgrid = min(maximum(xygrid.x), maximum(xygrid.y))
    rmax_eff = min(rmax, 0.98 * rgrid)
    rmax_eff < 0.5 * rmax && @warn(
        "beamlet profile truncated by the transverse grid: asked for $(round(rmax*1e6, digits=1)) µm, " *
        "the grid supports $(round(rmax_eff*1e6, digits=1)) µm. The radial closure against " *
        "Iω_beamlet will fall short by the energy outside.", maxlog=1)
    r = collect(range(0, rmax_eff, nr))
    θ = collect(range(0, 2π, nθ + 1)[1:nθ])
    # polar sample points in fractional grid indices, computed once
    fx = [rr*cos(tt)/δx + cx for rr in r, tt in θ]
    fy = [rr*sin(tt)/δy + cy for rr in r, tt in θ]

    Eωr  = zeros(ComplexF64, Nω, nr)
    asym = zeros(Float64, Nω)
    # k₀ = hole·ω/(c·zmask) per axis; the ω-independent part is the stored coefficient
    coefx = holex / (PhysData.c * zmask)
    coefy = holey / (PhysData.c * zmask)
    # ω is the FASTEST-varying axis, so a fixed-ω slice is strided and FFTW refuses to
    # apply a contiguous plan to it. Copy into the buffer and transform it in place.
    slice = Array{ComplexF64}(undef, Ny, Nx)
    IFT = FFTW.plan_ifft!(slice, (1, 2))
    ring = Vector{ComplexF64}(undef, nθ)
    relσ = Vector{Float64}(undef, nr)      # per-radius azimuthal RMS of |E| over its mean
    xv, yv = xygrid.x, xygrid.y
    for iω in 1:Nω
        copyto!(slice, view(Eωk, iω, :, :))
        IFT * slice                               # (ky, kx) -> (y, x), the focal plane
        ω = grid.ω[iω]
        k0x, k0y = coefx*ω, coefy*ω
        # Demodulate the WHOLE slice before sampling. Interpolating first and
        # demodulating after means interpolating ~6 cycles of tilt across the sampling
        # radius, and bilinear smoothing of that biases the amplitude low — measured, it
        # cost 20 % of the radial closure at the blue end.
        @inbounds for j in 1:Nx
            px = cis(-k0x*xv[j])
            for i in 1:Ny
                slice[i, j] *= px * cis(-k0y*yv[i])
            end
        end
        for ir in 1:nr
            @inbounds for it in 1:nθ
                gx, gy = fx[ir, it], fy[ir, it]
                j0 = clamp(floor(Int, gx), 1, Nx-1); tx = gx - j0
                i0 = clamp(floor(Int, gy), 1, Ny-1); ty = gy - i0
                ring[it] = ((1-ty)*(1-tx)*slice[i0,   j0]   + (1-ty)*tx*slice[i0,   j0+1] +
                                 ty*(1-tx)*slice[i0+1, j0]  +      ty*tx*slice[i0+1, j0+1])
            end
            Eωr[iω, ir] = sum(ring) / nθ
            # |E| is unchanged by the demodulation, so this is the beamlet's own
            # azimuthal asymmetry — how well a radial profile describes it here.
            μ = sum(abs, ring)/nθ
            relσ[ir] = μ > 0 ? sqrt(sum(x -> (abs(x) - μ)^2, ring)/nθ)/μ : 0.0
        end
        # Report it over the radii that actually carry signal; the far wings are noise
        # and would dominate an unweighted mean.
        pk = maximum(abs, view(Eωr, iω, :))
        if pk > 0
            keep = [ir for ir in 1:nr if abs(Eωr[iω, ir]) > 0.05pk]
            asym[iω] = isempty(keep) ? 0.0 : sum(relσ[keep])/length(keep)
        end
    end
    return r, Eωr, asym, (coefx, coefy), rmax
end

"""
    _profile_meta(r, Eωr, asym, coef, holex, holey, zmask, rmax_units, which) -> Dict

Package [`_beamlet_profile`](@ref)'s output for the output file. Complex data is split into
two real datasets, matching the `Eω_beamlet_re`/`_im` convention (h5py reads HDF5.jl's
native complex compound awkwardly), and enough geometry is recorded for the file to be
self-describing without the script that made it.
"""
function _profile_meta(r, Eωr, asym, coef, rmax_req, holex, holey, zmask,
                       rmax_units, which)
    Dict{String,Any}(
        "beamlet_r"            => r,                    # metres, from the beamlet centre
        "Eω_beamlet_r_re"      => real.(Eωr),           # (Nω, nr)
        "Eω_beamlet_r_im"      => imag.(Eωr),
        "beamlet_r_asym"       => asym,                 # azimuthal RMS/mean of |E| per ω
        "beamlet_r_which"      => which,                # which beamlet this profile is of
        "beamlet_r_holex"      => holex,                # its hole centre in the MASK plane
        "beamlet_r_holey"      => holey,
        "beamlet_r_zmask"      => zmask,
        # k₀(ω) = coef·ω is the geometric tilt REMOVED before the azimuthal average;
        # multiply by exp(+iω(coefx·x + coefy·y)) to restore the full focal field.
        "beamlet_r_tilt_coefx" => coef[1],
        "beamlet_r_tilt_coefy" => coef[2],
        "beamlet_r_max_units"  => float(rmax_units),
        # what was asked for; `beamlet_r[end]` is what the grid actually supported
        "beamlet_r_max_requested" => float(rmax_req),
        "beamlet_profile"      => 1,
    )
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
function build_beamlets(beam::HE11Beam, grid::Grid.TimeGrid,
                         xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
                         energy::Real, energyfun_ω;
                         apod::Symbol=:supergauss, apod_param=nothing,
                         ϕ=nothing, profile::Bool=true, profile_nr::Int=64,
                         profile_rmax_units::Real=6)
    # `ϕ` is accepted for a uniform interface and deliberately IGNORED: this
    # builder derives its beamlets from the 1-D reference `Eω`, which already
    # carries the Taylor spectral phase applied in `build_setup`. Applying ϕ
    # again here would double the chirp.
    # Full beam (no mask) in k-space, rescaled to the requested energy.
    Eωk0 = build_he11_kspace(grid, xygrid, beam, Eω)
    Eωk0 .*= sqrt(energy) / sqrt(energyfun_ω(Eωk0))

    # --- Self-diffraction: TWO collinear beams --------------------------
    #
    # SD is the same propagation and the same chi(3) response; only the input
    # differs. The signal is 2k_E - k_G, so with the probe E and gate G on one
    # axis at -+s/2 (s = spacing + diam, centre to centre) the signal appears at
    # -3s/2 — one further slot out, on the probe's side.
    #
    # The layout is SYMMETRIC about the axis (E at -s/2, G at +s/2) rather than
    # putting E on axis and centring the signal. The centred variant needs a
    # 27% narrower grid, but cuts the two beams from very different parts of the
    # HE11 profile, giving them unequal energy; SD's signal is E^2 G*, so that
    # asymmetry does not cancel.
    #
    # Which beam is delayed follows croak's SD kernel E^2 G*: the GATE appears
    # once, carries the conjugation, and takes the delay — the same convention
    # as TG, where the delay rides the singly-appearing conjugated arm.
    if get(geom, :geometry, :tg) === :sd
        s_cc = geom.mask_spacing + geom.mask_diam
        Eωk_E = Eωk0 .* makemask(-s_cc/2, 0.0, geom.mask_diam, grid, xygrid;
                                 zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                                 λ0_for_default=geom.λ0)
        Eωk_G = Eωk0 .* makemask( s_cc/2, 0.0, geom.mask_diam, grid, xygrid;
                                 zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                                 λ0_for_default=geom.λ0)
        NyNx_sd = length(xygrid.y) * length(xygrid.x)
        Iω_bl = dropdims(sum(abs2, Eωk_E; dims=(2, 3)); dims=(2, 3)) ./ NyNx_sd
        meta_sd = Dict{String,Any}("Iω_beamlet" => Iω_bl, "a" => beam.a,
                                   "geometry" => "sd",
                                   "sd_separation_cc" => s_cc,
                                   "sd_signal_x" => -1.5 * s_cc)
        # Profile the beamlet `Iω_beamlet` describes, so the two agree and the radial
        # closure check means something. In SD that is the probe E, at (-s_cc/2, 0).
        if profile
            rmax = profile_rmax_units * geom.λ0 * beam.f_foc / geom.mask_diam
            pr = _beamlet_profile(grid, xygrid, Eωk_E, -s_cc/2, 0.0, beam.f_foc;
                                  nr=profile_nr, rmax=rmax)
            merge!(meta_sd, _profile_meta(pr..., -s_cc/2, 0.0, beam.f_foc,
                                          profile_rmax_units, "sd_probe"))
        end
        # E is the undelayed pair-slot (it appears squared); G takes the delay.
        # `nothing` for the second gate rather than a zero array: at SD grid
        # size that array is ~0.5 GB of pure waste, and this campaign has been
        # OOM-killed by exactly this class of transient before.
        return Eωk_E, nothing, Eωk_G, Iω_bl, meta_sd
    end

    # Hole centres at (±d, ±d), where d = mask_spacing/2 + mask_diam/2.
    d = geom.mask_spacing/2 + geom.mask_diam/2

    # Boxcar layout (looking along +z):
    #
    #   test (-x, +y)    | gate1 (+x, +y)
    #  ---------------------------------------
    #   signal (-x, -y)  | gate2 (+x, -y)
    # Build and apply the masks one at a time: holding all three (plus their products)
    # simultaneously added ~2 field-sized arrays to the setup's peak memory.
    Eωk_g1     = Eωk0 .* makemask( d,  d, geom.mask_diam, grid, xygrid;
                                  zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                                  λ0_for_default=geom.λ0)
    Eωk_g2     = Eωk0 .* makemask( d, -d, geom.mask_diam, grid, xygrid;
                                  zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                                  λ0_for_default=geom.λ0)
    Eωk_t_base = Eωk0 .* makemask(-d,  d, geom.mask_diam, grid, xygrid;
                                  zmask=beam.f_foc, apod=apod, apod_param=apod_param,
                                  λ0_for_default=geom.λ0)

    # Spatially-integrated spectrum of the gate-1 beamlet — a useful diagnostic
    # showing the chromatic vignetting imprinted by the physical mask.
    # By Parseval (unitary up to 1/(Ny·Nx) for the ifft), the real-space sum equals the
    # k-space sum — no need to materialise the ifft'd field.
    NyNx = length(xygrid.y) * length(xygrid.x)
    Iω_beamlet = dropdims(sum(abs2, Eωk_g1; dims=(2, 3)); dims=(2, 3)) ./ NyNx

    beam_meta = Dict{String,Any}(
        "Iω_beamlet" => Iω_beamlet,
        "a"          => beam.a,
        "a_scaled"   => a_scaled(beam),
        "f_coll"     => beam.f_coll,
        "f_foc"      => beam.f_foc,
    )

    # Spatially resolved focal field of gate 1 — the beamlet `Iω_beamlet` describes, so
    # the two are consistent and the radial closure check is meaningful. Diagnostic only:
    # `Eωk_g1` is read, never modified. Computed ONCE here, not per delay point.
    if profile
        rmax = profile_rmax_units * geom.λ0 * beam.f_foc / geom.mask_diam
        pr = _beamlet_profile(grid, xygrid, Eωk_g1, d, d, beam.f_foc;
                              nr=profile_nr, rmax=rmax)
        merge!(beam_meta, _profile_meta(pr..., d, d, beam.f_foc,
                                        profile_rmax_units, "tg_gate1"))
    end

    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end

function build_beamlets(beam::GaussianBeam, grid::Grid.TimeGrid,
                         xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
                         energy::Real, energyfun_ω;
                         apod::Symbol=:supergauss, apod_param=nothing,
                         ϕ=nothing, profile::Bool=true, profile_nr::Int=64,
                         profile_rmax_units::Real=6)
    # In the Gaussian model each beamlet carries an equal third of the energy.
    energy_per_beam = energy / 3
    Eωk_base = build_gaussian_kspace(grid, xygrid, beam,
                                     geom.λ0, geom.τfwhm, energy_per_beam)
    # Apply the input spectral phase (GDD/TOD) HERE rather than inheriting it
    # from the 1-D reference `Eω`, which this builder does not use.
    # `Fields.GaussGaussField` takes a ϕ, but for a SpatioTemporalField that ϕ
    # is a SCALAR carrier-envelope phase (`exp(i(ϕ + Δω t))`), not the Taylor
    # coefficient VECTOR that the 1-D `GaussField` accepts — so it cannot carry
    # GDD at all, and a chirp requested through `build_setup` would otherwise be
    # silently dropped for this beam type.
    #
    # Applying it as a pure spectral phase is exact here: the Gaussian
    # transverse profile is ω-independent by construction (that is the whole
    # point of this beam), so the spectral phase commutes with the spatial
    # structure. It would NOT be exact for a chromatic beam, which is why the
    # HE11 builder instead inherits the already-chirped `Eω`.
    if ϕ !== nothing && any(!iszero, ϕ)
        Fields.prop_taylor!(Eωk_base, grid, ϕ, geom.λ0)
    end
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
    # As for HE₁₁, but this model has no mask: the beamlet is a tilted Gaussian, so the
    # natural radial scale is w0 rather than λf/D, and the tilt to remove is the applied
    # Δk. `Δk` here is ω-INDEPENDENT (it is built at λ0 in `apply_tilt`), unlike the
    # mask geometry's `hole·ω/(c·zmask)`, so the equivalent coefficient is Δk/ω0.
    if profile
        ω0 = 2π * PhysData.c / geom.λ0
        rmax = profile_rmax_units * beam.w0
        pr = _beamlet_profile(grid, xygrid, Eωk_g1, Δk*PhysData.c*beam.f_foc/ω0,
                              Δk*PhysData.c*beam.f_foc/ω0, beam.f_foc;
                              nr=profile_nr, rmax=rmax)
        merge!(beam_meta, _profile_meta(pr..., d_hole, d_hole, beam.f_foc,
                                        profile_rmax_units, "gaussian_gate1"))
    end
    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end

# ============================================================================
# Delayed (Raman) nonlinear response
# ============================================================================

"""
    FrozenRamanPolarEnv(t, r)

Envelope Raman polarisation response with a *frozen* response kernel.

Wraps `Luna.Nonlinear.RamanPolarEnv`, precomputing the frequency-domain
response function `hω` once at construction (at unit density). Luna's own
response recomputes the time-domain kernel and its FFT on *every* call, which
is negligible for modal simulations (one call per step) but dominant for
free-space grids, where the response runs once per transverse point — ~10⁶
calls per RK stage on a 1024² grid, each re-evaluating the 13-mode
Hollenbeck–Cantrell sum over the doubled time grid. Freezing is exact here:
the density is constant (`densityfun = z -> 1`) and the `:intermediate`
response ignores its density argument entirely.

The per-call convolution below reproduces `Luna.Nonlinear.(::RamanPolar)`
line for line, minus the kernel update and with preallocated plan
applications; the test suite verifies agreement with Luna's response to
machine precision, which guards against drift in Luna's internals.
"""
struct FrozenRamanPolarEnv{TR, TI}
    R::TR    # wrapped Luna response: buffers, FFT plan, frozen hω
    IFT::TI  # cached inverse plan, so no per-call allocation
end

function FrozenRamanPolarEnv(t::AbstractVector, r)
    R = Nonlinear.RamanPolarEnv(t, r)
    R.r(R.ht, 1.0)       # kernel at unit density — never changes
    R.hω .= R.FT * R.h
    return FrozenRamanPolarEnv(R, inv(R.FT))
end

function (F::FrozenRamanPolarEnv)(out, Et, ρ)
    R = F.R
    n = size(Et, 1)
    if ndims(Et) > 1
        size(Et, 2) == 1 || error("vector Raman not implemented")
        E = reshape(Et, n)
    else
        E = Et
    end
    Nonlinear.sqr!(R, E)               # R.E2v .= |E|²/2 (first half; rest 0)
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2 * R.dt      # convolution on the doubled grid
    mul!(R.P, F.IFT, R.Pω)
    @inbounds for i in 1:length(E)
        R.Pout[i] = ρ * E[i] * R.P[i]
    end
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        out .+= R.Pout
    end
    return nothing
end

# ============================================================================
# Memory budget
# ============================================================================

"""
    memory_budget(setup_args::NamedTuple) -> NamedTuple

Resident **device** memory one delay point of `setup_args` will need, and the **host** peak
[`build_setup`](@ref) will reach, broken down by buffer. `setup_args` is the same NamedTuple
[`run_scan`](@ref) and [`verify_against_collected`](@ref) take; only the grid-determining
entries are read, and building the 1-D time grid is the whole cost, so this is free to call.

This exists because guessing is expensive. The envelope path obeys a simple rule — 9 RK45
registers plus one transform buffer, i.e. 10× the field size, measured exactly on an A40 —
and **the field path does not**: its state is twice as long in ω, its nonlinear evaluation
runs on a grid twice as long again in time, and the no-THG response carries a complex
analytic-signal buffer on that grid. At the 40 µm production shape (N = 768) that is 92 GiB
against the envelope's 24. Finding this out by running is an hour of rented GPU and a dead
process.

!!! warning "The response's buffer appears on the first RHS, not at setup"
    `Nonlinear.KerrFieldNoTHG` allocates its analytic-signal buffer lazily, when it first
    sees a field. A card with room to spare after `build_setup` can therefore still die on
    the first step — 18 GiB later at the production shape. This function counts it; a
    measurement taken after `build_setup` alone will not.

Fields: `Nω`, `Nt`, `Nto`, `Nωo`, `field` (one state array), the per-buffer terms
`state`, `et_win`, `eto`, `ewo`, `pto`, `analytic`, `window`, `input`, and the totals
`device` and `host`. All in GiB.

`state` … `window` are what the transform and solver hold; `input` is the per-delay field
`delayed_input` produces. Only the first group is allocated by `build_setup`, so a
measurement taken across `build_setup` alone will fall short of `device` by `state`,
`analytic` and `input` — those appear when the first delay point runs.

The buffer set and its aliasing are `NonlinearRHS.TransFree`'s: `Pωo` always aliases `Eωo`
(the inverse transform consumes it), `Pto` aliases `Eto` when every response is pointwise
(the envelope Kerr, and field `:thg`, but not field `:nothg`), and `Et_win` exists only when
the grid is oversampled. `window` is the extraction window, which is device-resident when
save-time extraction is used — the default on a device.
"""
function memory_budget(setup_args::NamedTuple)
    a = setup_args
    N = a.N
    field = get(a, :field_mode, false)
    response = get(a, :response, :auto)
    ffac = get(a, :ffac, 6)
    grid = field ? Grid.RealGrid(a.thickness, a.λ0, a.λlims, a.trange; ffac=ffac) :
                   Grid.EnvGrid(a.thickness, a.λ0, a.λlims, a.trange;
                                fftsize=get(a, :fftsize, :pow2))
    nwin = a.window isa AbstractSignalWindow ? 1 : length(a.window)
    _budget(grid, N, field, response, nwin, get(a, :beamlets_on_host, false))
end

function _budget(grid, N, field, response, nwin, beamlets_on_host)
    Nω, Nt, Nto, Nωo = length(grid.ω), length(grid.t), length(grid.to), length(grid.ωo)
    b(n, sz) = n * N^2 * sz / 2^30
    tsz = field ? 8 : 16                 # the time-domain buffers are REAL in field mode
    # `:auto` resolves to `:nothg`, which is batched — not pointwise — so it needs its own
    # polarisation buffer as well as the analytic signal.
    pointwise = !field || response === :thg
    state    = 9 * b(Nω, 16)
    et_win   = Nto == Nt ? 0.0 : b(Nt, tsz)
    eto      = b(Nto, tsz)
    ewo      = field ? b(Nωo, 16) : 0.0  # the envelope fast path has no oversampled buffers
    pto      = pointwise ? 0.0 : b(Nto, tsz)
    analytic = (field && response !== :thg) ? b(Nto, 16) : 0.0
    window   = nwin * b(Nω, 8)
    # The per-delay input. With `beamlets_on_host` the gate pair and the test beam live on
    # the host and `delayed_input` uploads one field per point; otherwise both beamlets are
    # device-resident and their delayed sum is a third. Either way it is live for the whole
    # propagation — the solver adopts it as its first register — so it belongs in the
    # resident total, and leaving it out is what made the measured figure look 5 % over.
    input    = (beamlets_on_host ? 1 : 3) * b(Nω, 16)
    (; Nω, Nt, Nto, Nωo, field=b(Nω, 16),
       state, et_win, eto, ewo, pto, analytic, window, input,
       device = state + et_win + eto + ewo + pto + analytic + window + input,
       # build_setup holds the unmasked HE11 field, the three masked beamlets, their
       # pre-summed gate pair and the window, all at once.
       host   = 5*b(Nω, 16) + window)
end

# ============================================================================
# Grid-representation helpers
# ============================================================================
#
# The two grid types differ in exactly three places downstream of `build_setup`: how a
# spectrum is transformed back to time, what "intensity" means once it is there, and what
# the frequency axis looks like to a reader. Everything else — the masks, the windows, the
# delay ramp, the k-space builders, the whole extraction path — is written in terms of the
# ABSOLUTE frequency axis `grid.ω`, which both grid types carry, and needs no branching.

"""
    _to_time(grid, Eω)

The 1-D time-domain field for a spectrum on `grid`'s own frequency axis: an inverse FFT for
an `EnvGrid` (whose spectrum is FFT-ordered about the carrier) and an inverse *real* FFT for
a `RealGrid` (whose spectrum is the monotonic `rfft` half-spectrum of a real field).
"""
_to_time(grid::Grid.EnvGrid, Eω::AbstractVector) = ifft(Eω)
_to_time(grid::Grid.RealGrid, Eω::AbstractVector) = irfft(Eω, length(grid.t))

"""
    _envelope_intensity(grid, Et)

`|A(t)|²` — the ENVELOPE intensity — from whatever time-domain field `grid` produces. On an
`EnvGrid` that is `Et` itself; on a `RealGrid` the field is carrier-resolved and the
envelope is recovered through its analytic signal.

Both conventions coincide numerically: Luna builds a real-grid pulse as `√I·cos(ω₀t)` and an
envelope-grid pulse as `√I·exp(iΔωt)`, so `|A|² = I` either way. Keeping the *envelope*
intensity in the output metadata means a consumer of a field-mode file sees the same
physical quantity in `It`/`Ito` as in every envelope file, rather than a carrier-modulated
one it would have to demodulate.
"""
_envelope_intensity(::Grid.EnvGrid, Et) = abs2.(Et)
_envelope_intensity(::Grid.RealGrid, Et) = abs2.(Maths.hilbert(Et))

"""
    _plan_1d(grid)

Forward transform plan for the 1-D reference pulse: complex for an `EnvGrid`, real-to-complex
for a `RealGrid`. `Fields.GaussField` dispatches its time-domain shape on the grid type but
takes the plan from the caller, so the two have to be chosen together.
"""
_plan_1d(grid::Grid.EnvGrid) = plan_fft(copy(grid.t))
_plan_1d(grid::Grid.RealGrid) = plan_rfft(zeros(Float64, length(grid.t)))

"""Whether `grid` is field-resolved (real) rather than an envelope grid."""
_is_field_mode(::Grid.RealGrid) = true
_is_field_mode(::Grid.TimeGrid) = false

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
- `GDD = 0.0`, `TOD = 0.0`        — group-delay and third-order dispersion
                                    [s², s³] applied to the input pulse
- `raman = false`                 — include the delayed (Raman) part of the
                                    nonlinear response via
                                    [`FrozenRamanPolarEnv`](@ref); requires a
                                    material with an `:intermediate` Raman
                                    model in `Luna.PhysData.raman_parameters`
                                    (for `:SiO2` the multimode
                                    Hollenbeck–Cantrell response). The total
                                    polarisation is
                                    `(3/4)ε₀χ³[(1-f_R)|E|²E + f_R E(h_R⊛|E|²)]`
                                    — equal prefactors on both terms, the
                                    envelope-defined `f_R` convention of
                                    Luna's `prop_gnlse`, so the quasi-static
                                    limit reproduces the Kerr-only response
                                    exactly
- `raman_fraction = 0.18`         — envelope-defined nuclear fraction `f_R`
                                    of χ³ (the Blow–Wood silica value)
- `raman_impl = :batched`         — Raman implementation: `:batched` computes
                                    the convolution for all transverse points
                                    at once (two batched FFTs per RHS
                                    evaluation); `:frozen` is the legacy
                                    per-column [`FrozenRamanPolarEnv`](@ref).
                                    Results agree to rounding accuracy
- `field_mode = false`            — propagate the real, carrier-resolved field on a
                                    `Luna.Grid.RealGrid` instead of the complex envelope on
                                    an `EnvGrid`. There is then no carrier/envelope split,
                                    no dropped third-harmonic term and no negative-frequency
                                    wrap; the cost is roughly 2× the memory and 3×
                                    the time per delay point (measured 3.0× at N = 64 and
                                    3.3× at N = 128, at matched step counts). The envelope
                                    path is untouched and remains the default
- `response = :auto`              — field-mode nonlinearity: `:nothg` (= `:auto`) for
                                    `(3/4) ε₀ χ³ |E_a|² E`, the same physics content as the
                                    envelope `Kerr_env` and hence the response for an
                                    envelope-versus-field comparison; `:thg` for `ε₀ χ³ E³`,
                                    which adds what the envelope drops. Ignored unless
                                    `field_mode = true`
- `ffac = 6`                      — field-mode nonlinear-grid sampling factor, forwarded to
                                    `Grid.RealGrid`. 6 (the default) sizes the fine grid for
                                    `E³`; 4 is enough for `:nothg` alone and typically
                                    removes the oversampling entirely, halving memory and
                                    per-step cost. It changes the grid, so use it only with
                                    a convergence check against the default
- `raman`                         — not implemented in field mode (see the error message
                                    there for why)
- `beamlet_profile = true`        — store the gate beamlet's spatially resolved complex
                                    focal field as a radial profile `Eω_beamlet_r_re/_im`
                                    `(Nω, nr)` plus the radius axis `beamlet_r` in metres,
                                    so the pulse that actually drives the signal can be
                                    computed rather than assumed. Diagnostic only — no
                                    propagation result depends on it — costing one 2-D
                                    inverse transform per ω ONCE at setup and ~130 kB in
                                    the file. See [`_beamlet_profile`](@ref) for where the
                                    beamlet is (the focus, not a BOXCARS corner) and why
                                    the geometric tilt is removed first
- `beamlet_profile_nr = 64`       — radial samples. Measured radial closure against
                                    `Iω_beamlet` on the production geometry, at 200 /
                                    260 / 350 nm: 0.955/0.972/0.980 at nr = 32,
                                    0.975/0.984/0.986 at 64, 0.980/0.987/0.988 at 128.
                                    64 is where it has essentially converged, for 262 kB
                                    at `Nω = 256`; the residual ~1.5 % is truncation at
                                    `rmax` plus the beamlet's real azimuthal asymmetry
- `beamlet_profile_rmax_units = 6` — outer radius, in units of `λ0·f_foc/mask_diam`
                                    (`w0` for [`GaussianBeam`](@ref))
- `factored_linop = true`         — use Luna's lazy (factored) linear operator
                                    and normalisation, saving two field-sized
                                    arrays; bit-identical to the materialised
                                    versions
- `store_window = true`           — store the materialised window array(s) in
                                    the output metadata (≈1 GiB at production
                                    size); the window parameters (`window_def`)
                                    are always stored and reconstruct the
                                    array via [`build_window`](@ref)
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
                       geometry::Symbol = :tg,
                       GDD = 0.0, TOD = 0.0,
                       raman::Bool = false, raman_fraction::Float64 = 0.18,
                       raman_impl::Symbol = :batched,
                       field_mode::Bool = false,
                       response::Symbol = :auto,
                       ffac::Real = 6,
                       beamlet_profile::Bool = true,
                       beamlet_profile_nr::Int = 64,
                       beamlet_profile_rmax_units::Real = 6,
                       factored_linop::Bool = true,
                       store_window::Bool = true,
                       arraytype = Array,
                       beamlets_on_host::Bool = false,
                       fftsize::Symbol = :pow2,
                       optimal_grid_kwargs = (;),
                       extra_grid_metadata = Dict{String,Any}())

    # --- Resolve spatial grid ----------------------------------------------
    if R === nothing || N === nothing
        f_foc = beam.f_foc
        R, N = optimal_spatial_grid(f_foc, mask_diam, mask_spacing,
                                     λlims[1], λlims[2];
                                     geometry=geometry, optimal_grid_kwargs...)
    end

    # --- Build Luna grids --------------------------------------------------
    # `field_mode` swaps the envelope (analytic-field) grid for the field-resolved one.
    # `RealGrid` has no `fftsize` control — its propagated grid is always the next power of
    # two above 1.1 ωmax — so that keyword applies to the envelope path only.
    grid = field_mode ? Grid.RealGrid(thickness, λ0, λlims, trange; ffac=ffac) :
                        Grid.EnvGrid(thickness, λ0, λlims, trange; fftsize=fftsize)
    xygrid = Grid.FreeGrid(R, N)

    # --- Material dispersion + Kerr (+ optional Raman) nonlinearity -------
    χ3 = PhysData.χ3(material)
    if field_mode
        raman && error(
            "raman=true is not implemented in field mode. The delayed response itself is " *
            "available (Luna.Nonlinear.RamanPolarFieldBatched), but the nuclear fraction " *
            "f_R below is defined in the ENVELOPE convention — the 3/2 that reconciles " *
            "Kerr_env's internal 3/4 with the Raman kernel's 1/2 — and that factor does " *
            "not carry over to a carrier-resolved field unexamined. Deriving it needs its " *
            "own quasi-static consistency test, like the envelope one in the test suite.")
        responses = _field_responses(response, χ3)
    elseif raman
        # Split χ³ into an instantaneous electronic part and a delayed
        # nuclear part carrying the envelope-defined fraction
        # `raman_fraction` (the Blow–Wood f_R = 0.18 for silica), following
        # the convention of Luna's `prop_gnlse`: both terms must end with
        # the SAME prefactor, so that the quasi-static (long-pulse) limit of
        # the split response is identical to the Kerr-only response.
        # `Kerr_env` applies the envelope factor 3/4 internally while the
        # Raman response's `sqr!` applies 1/2, so the Raman `scale` carries
        # the compensating (3/4)/(1/2) = 3/2 (and the explicit ε₀, which
        # `Kerr_env` adds internally):
        #     P = (3/4) ε₀ χ³ [ (1-f_R) |E|²E + f_R E (h_R ⊛ |E|²) ].
        # (Luna's low-level silica envelope examples omit the 3/2 and thus
        # under-weight Raman by 2/3 relative to this convention.) The
        # equal-prefactor property is enforced by the quasi-static
        # consistency test in the test suite.
        rp = PhysData.raman_parameters(material)
        rp.kind == :intermediate ||
            error("raman=true supports materials with an :intermediate " *
                  "(condensed-phase) Raman model, e.g. :SiO2; " *
                  "$material has kind :$(rp.kind)")
        rr = Raman.raman_response(grid.to, material,
                                  1.5 * raman_fraction * PhysData.ε_0 * χ3)
        # :batched (default) computes the Raman convolution for all transverse points
        # at once with batched FFTs (two per RHS evaluation instead of two small serial
        # FFTs per transverse column); :frozen is the legacy per-column implementation,
        # kept for A/B comparison. Results agree to rounding accuracy (~1e-15 relative).
        Rresp = if raman_impl === :batched
            Nonlinear.RamanPolarEnvBatched(grid.to, rr)
        elseif raman_impl === :frozen
            FrozenRamanPolarEnv(grid.to, rr)
        else
            error("raman_impl must be :batched or :frozen, got :$raman_impl")
        end
        responses = (Nonlinear.Kerr_env((1 - raman_fraction) * χ3), Rresp)
    else
        responses = (Nonlinear.Kerr_env(χ3),)
    end
    nfun = PhysData.ref_index_fun(material, lookup=false)
    nfunreal = (λ) -> real(nfun(λ))
    # factored (lazy) operators compute their elements on demand instead of storing two
    # field-sized arrays; bit-identical to the materialised versions (Luna guarantees
    # and tests this)
    arraytype = Luna.resolve_arraytype(arraytype)
    linop = LinearOps.make_const_linop(grid, xygrid, nfunreal;
                                       factored=factored_linop, arraytype)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfunreal;
                                           factored=factored_linop, arraytype)
    densityfun = z -> 1
    _, transform, FT = Luna.setup(grid, xygrid, densityfun, normfun, responses, ();
                                  arraytype)
    _, energyfun_ω = Fields.energyfuncs(grid, xygrid)

    # --- 1-D reference spectrum (used by the HE₁₁ builder and as diagnostic)
    # Complex on an envelope grid, real-to-complex on a field-resolved one: `GaussField`
    # dispatches its time-domain shape on the grid but takes the plan from here, and
    # `prop_taylor!` applies ϕ on the ABSOLUTE ω axis, which both grids carry.
    FT1d = _plan_1d(grid)
    ϕ = [0.0, 0.0, GDD, TOD]  # up to 3rd order
    Eω = Fields.GaussField(; λ0=λ0, τfwhm=τfwhm, energy=energy, ϕ=ϕ)(grid, FT1d)

    # --- Build three input beamlets ---------------------------------------
    geom = (; mask_diam, mask_spacing, f_foc=beam.f_foc, λ0, τfwhm, geometry)
    Eωk_g1, Eωk_g2, Eωk_t_base, _Iω_beamlet, beam_meta =
        build_beamlets(beam, grid, xygrid, geom, Eω, energy, energyfun_ω;
                       apod=apod, apod_param=apod_param, ϕ=ϕ,
                       profile=beamlet_profile, profile_nr=beamlet_profile_nr,
                       profile_rmax_units=beamlet_profile_rmax_units)

    # --- Build signal window(s) -------------------------------------------
    window_array, window_suffix = _build_window_set(window, grid, xygrid; λ0=λ0)

    # --- Assemble combined_grid metadata ----------------------------------
    # Record the nonlinearity model alongside the grids (explicit entries win
    # over these defaults if the caller supplies their own).
    extra_grid_metadata = merge(Dict{String,Any}(
                                    "raman" => raman ? 1 : 0,
                                    "raman_fraction" => raman_fraction,
                                    # How the file was made. `field_mode` is also the
                                    # marker every reader needs: a field-mode /grid/ω is a
                                    # monotonic rfft half-spectrum, not the FFT-ordered
                                    # relative-frequency axis of an envelope file, so it
                                    # must NOT be fftshifted on load. Absent = envelope,
                                    # which is what every pre-existing file is.
                                    # 0 when the diagnostic focal profile is off, so a
                                    # file always says whether it should have one.
                                    "beamlet_profile" => beamlet_profile ? 1 : 0,
                                    "field_mode" => field_mode ? 1 : 0,
                                    "response" => _response_name(field_mode, response,
                                                                 raman),
                                    # `ffac` is the field grid's nonlinear-sampling factor
                                    # (6 = sized for E³, 4 = sized for |E_a|²E only). It is
                                    # not a field of Grid.RealGrid, so record it here.
                                    "ffac" => field_mode ? float(ffac) : 0.0,
                                    # The MASK GEOMETRY, recorded so the file is
                                    # self-describing. Retrieval codes need d/D
                                    # = (spacing + D)/2D to build the smearing
                                    # kernel, and until now it was carried by
                                    # hand from script to script: a sweep run
                                    # against a gap-1000 trace while still
                                    # holding the gap-500 default silently built
                                    # a kernel 25% too narrow, worth ~2% of
                                    # retrieved duration with NO signature in
                                    # the trace error. It cannot be recovered
                                    # reliably after the fact either — the
                                    # signal-window fields only pin it for some
                                    # geometries, and for SD the window sits at
                                    # 1.5s rather than d.
                                    "mask_diam" => mask_diam,
                                    "mask_spacing" => mask_spacing,
                                    "f_foc" => beam.f_foc,
                                    "geometry" => string(geometry)),
                                extra_grid_metadata)
    combined_grid = _combined_grid(grid, xygrid, beam_meta,
                                    window, window_array, window_suffix,
                                    Eω, λ0, τfwhm, material, thickness,
                                    extra_grid_metadata; store_window)

    # Beamlets are built on the host (masks, Bessel profiles and FFTs are host code).
    # Move them to the propagation's array type unless asked to keep them here: at the
    # largest campaign shapes those two fields are the difference between fitting a card
    # and not, and the per-point upload that replaces them is a fraction of a second.
    # `Eωk_g2 === nothing` is the self-diffraction geometry: two beams, not three, so
    # the "gate pair" is the single gate beamlet (see the :sd branch of build_beamlets).
    Eωk_g12 = isnothing(Eωk_g2) ? Eωk_g1 : Eωk_g1 .+ Eωk_g2
    if arraytype !== Array && !beamlets_on_host
        Eωk_g12 = Adapt.adapt(arraytype, Eωk_g12)
        Eωk_t_base = Adapt.adapt(arraytype, Eωk_t_base)
    end
    # The delay phase must live wherever the BEAMLETS do, not wherever the propagation
    # does: `delayed_input` multiplies them together in one broadcast, and mixing a host
    # array into a device broadcast is rejected outright by CUDA. With
    # `beamlets_on_host` the whole expression is evaluated on the host and the result
    # uploaded afterwards.
    ωd = Eωk_g12 isa Array ? grid.ω : Adapt.adapt(arraytype, collect(grid.ω))

    return TGFROGSetup{typeof(grid),typeof(linop),typeof(transform),typeof(FT),
                       typeof(window),typeof(window_array),
                       typeof(Eωk_g12),typeof(ωd)}(
        λ0, τfwhm, energy, thickness, material, mask_diam, mask_spacing,
        grid, xygrid, linop, transform, FT, energyfun_ω,
        Eωk_g12, Eωk_t_base, ωd, Eω,
        window, window_array, window_suffix, combined_grid)
end

# ----- Field-mode response selector ----------------------------------------

"""
    _field_responses(response, χ3) -> Tuple

The nonlinear response for a field-resolved (`RealGrid`) run.

- `:nothg` (and `:auto`) — `(3/4) ε₀ χ³ |E_a|² E` via
  `Luna.Nonlinear.Kerr_field_nothg`. This is the SAME physics content as the
  envelope `Kerr_env`, evaluated on a carrier-resolved field, so an envelope-versus-field
  comparison made with it isolates *representation* error with nothing else changed. It is
  the default for exactly that reason.
- `:thg` — `ε₀ χ³ E³` via `Luna.Nonlinear.Kerr_field`, which adds the third-harmonic
  and counter-rotating terms the envelope drops. The difference between the two runs is
  precisely what the envelope omits.

!!! note "What `:thg` does and does not propagate on a UV window"
    The third harmonic is generated on the fine grid and then discarded by the crop back to
    the propagated grid whenever it falls outside the window (at λlims = (143, 600) nm the
    3ω band of a 2 fs 260 nm pulse starts above ωmax). The within-band counter-rotating
    terms are retained, and those are the real difference from the envelope. Propagating
    the third harmonic itself needs λlims extended to ~λ0/3.
"""
function _field_responses(response::Symbol, χ3)
    if response === :auto || response === :nothg
        return (Nonlinear.Kerr_field_nothg(χ3),)
    elseif response === :thg
        return (Nonlinear.Kerr_field(χ3),)
    else
        error("field-mode `response` must be :auto, :nothg or :thg, got :$response")
    end
end

"""Canonical name of the response actually used, for the output metadata."""
_response_name(field_mode::Bool, response::Symbol, raman::Bool) =
    field_mode ? (response === :auto ? "nothg" : string(response)) :
                 (raman ? "kerr_env+raman" : "kerr_env")

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

# ----- Window parameter serialisation (reconstructible via build_window) ----

_window_def(w::PhysicalMaskWindow) = Dict{String,Any}(
    "type" => "PhysicalMaskWindow", "holex" => w.holex, "holey" => w.holey,
    "holediam" => w.holediam, "zmask" => w.zmask, "apod" => string(w.apod),
    "apod_param" => (w.apod_param === nothing ? "default" : w.apod_param))

_window_def(w::PlanckWindow) = Dict{String,Any}(
    "type" => "PlanckWindow", "kxc" => w.kxc, "kyc" => w.kyc,
    "kwidth" => w.kwidth, "pad" => w.pad)

_window_def(w::PlanckOmegaWindow) = Dict{String,Any}(
    "type" => "PlanckOmegaWindow", "xc" => w.xc, "yc" => w.yc,
    "holediam" => w.holediam, "f_foc" => w.f_foc, "pad" => w.pad)

function _combined_grid(grid, xygrid, beam_meta::Dict,
                         window, window_array, window_suffix::Vector{String},
                         Eω::AbstractVector, λ0, τfwhm, material, thickness,
                         extra::Dict; store_window::Bool=true)
    cg = Dict{String,Any}()
    for (k, v) in pairs(Grid.to_dict(grid))
        cg[string(k)] = v
    end
    for (k, v) in pairs(Grid.to_dict(xygrid))
        cg[string(k)] = v
    end

    # Always-present diagnostics.
    # `Eω` is the grid's own spectral representation: FFT-ordered about the carrier on an
    # EnvGrid, a monotonic rfft half-spectrum on a RealGrid. `_to_time` inverts whichever it
    # is, and `_envelope_intensity` reduces the result to |A|² either way, so `It`/`Ito` mean
    # the same physical thing in an envelope and a field-mode file.
    Et = _to_time(grid, Eω)
    It = _envelope_intensity(grid, Et)
    Iω = abs2.(Eω)
    # `Maths.oversample` has separate real and complex methods and picks the right one.
    to, eo = Maths.oversample(grid.t, Et; factor=8)
    Ito = _envelope_intensity(grid, eo)
    cg["Iω"]        = Iω
    cg["It"]        = It
    cg["To"]        = to
    cg["Ito"]       = Ito
    cg["τfwhm"]     = τfwhm
    cg["material"]  = string(material)
    cg["thickness"] = thickness
    # `Grid.to_dict` writes every field of the grid struct, and `RealGrid` has no `ω0` — it
    # propagates the field itself, not an envelope about a carrier. Readers key off
    # /grid/ω0 unconditionally, so write the carrier explicitly in that case.
    haskey(cg, "ω0") || (cg["ω0"] = 2π * PhysData.c / λ0)

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
        et_beamlet = _to_time(grid, Eω_beamlet)
        cg["It_beamlet"] = _envelope_intensity(grid, et_beamlet)
        _, eob_beamlet = Maths.oversample(grid.t, et_beamlet; factor=8)
        cg["Ito_beamlet"] = _envelope_intensity(grid, eob_beamlet)  # shares "To" above
        # The COMPLEX beamlet spectrum (amplitude and phase), stored as two
        # real datasets for cross-language portability (h5py reads HDF5.jl's
        # native complex compound awkwardly). This is the retrievable ground
        # truth WITH its spectral phase — it carries any input chirp (GDD/TOD)
        # exactly, since the mask is a real amplitude filter — enabling
        # complex-field retrieval-error metrics (Geib ε) and direct truth-GDD
        # measurements instead of intensity-only comparisons.
        cg["Eω_beamlet_re"] = real.(Eω_beamlet)
        cg["Eω_beamlet_im"] = imag.(Eω_beamlet)
    end
    # The complex INPUT (source) spectrum, same encoding: the pre-mask truth.
    cg["Eω_re"] = real.(Eω)
    cg["Eω_im"] = imag.(Eω)

    # Window parameters under "window_def*" (+ suffixes): always stored — they are a few
    # scalars and losslessly reconstruct the window via build_window. Flattened to
    # individual entries (scansave writes plain datasets, not nested groups).
    if window isa AbstractSignalWindow
        for (fk, fv) in _window_def(window)
            cg["window_def_" * fk] = fv
        end
    else
        for (suf, w) in zip(window_suffix, window)
            for (fk, fv) in _window_def(w)
                cg["window_def" * suf * "_" * fk] = fv
            end
        end
    end
    # Window arrays under "window" (+ optional suffixes for multi-window). At production
    # size a 3-D window is a ~1 GiB Float64 array in every collected file; pass
    # build_setup(store_window=false) to skip it and rely on window_def instead.
    if store_window
        if window isa AbstractSignalWindow
            cg["window"] = window_array
        else
            for (suf, arr) in zip(window_suffix, window_array)
                cg["window" * suf] = arr
            end
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
function extract_signal_spectra(Eωk_z::AbstractArray{<:Complex,3},
                                 window_array::AbstractArray{<:Real},
                                 xygrid::Grid.FreeGrid)
    Nω, Nky, Nkx = size(Eωk_z)
    if isodd(Nky) || isodd(Nkx)
        # the centre-pixel phase-sum identity below assumes even grid sizes
        return _extract_signal_spectra_fft(Eωk_z, window_array, xygrid)
    end
    w3 = ndims(window_array) == 3
    w3 || ndims(window_array) == 2 ||
        error("window_array must be 2-D or 3-D, got $(ndims(window_array))")
    # Single pass, no field-sized temporaries. The re-imaged spectrum needs only the
    # CENTRE pixel of ifft(Eωk_win, (2,3)): at (y, x) index (Nky÷2+1, Nkx÷2+1) the ifft
    # phase factor is exp(iπ(iky-1))·exp(iπ(ikx-1)) = (-1)^(iky+ikx-2), so the pixel is
    # the (-1)^…-weighted sum divided by Nky·Nkx — the previous full 2-D ifft (a
    # field-sized allocation and an N²logN transform per z-slice) computed exactly this.
    Iω_integrated = zeros(Float64, Nω)
    acc = zeros(ComplexF64, Nω)
    @inbounds for ikx in 1:Nkx, iky in 1:Nky
        sgn = isodd((iky - 1) + (ikx - 1)) ? -1.0 : 1.0
        for iω in 1:Nω
            w = w3 ? window_array[iω, iky, ikx] : window_array[iky, ikx]
            Ew = w * Eωk_z[iω, iky, ikx]
            Iω_integrated[iω] += abs2(Ew)
            acc[iω] += sgn * Ew
        end
    end
    Iω_reimaged = abs2.(acc ./ (Nky * Nkx))
    return Iω_integrated, Iω_reimaged
end

# Reference implementation via the full 2-D ifft; used as fallback for odd-sized grids
# and by the test suite to validate the one-pass version.
function _extract_signal_spectra_fft(Eωk_z::AbstractArray{<:Complex,3},
                                      window_array::AbstractArray{<:Real},
                                      xygrid::Grid.FreeGrid)
    if ndims(window_array) == 2
        Eωk_win = Eωk_z .* reshape(window_array, (1, size(window_array)...))
    elseif ndims(window_array) == 3
        Eωk_win = Eωk_z .* window_array
    else
        error("window_array must be 2-D or 3-D, got $(ndims(window_array))")
    end
    Eωxy_win = ifft(Eωk_win, (2, 3))
    Iω_reimaged = abs2.(Eωxy_win[:, length(xygrid.y) ÷ 2 + 1,
                                  length(xygrid.x) ÷ 2 + 1])
    Iω_integrated = dropdims(sum(abs2.(Eωk_win); dims=(2, 3)); dims=(2, 3))
    return Iω_integrated, Iω_reimaged
end

function extract_signal_spectra(Eωk_out::AbstractArray{<:Complex,4},
                                 window_array::AbstractArray{<:Real},
                                 xygrid::Grid.FreeGrid)
    # Slice-wise loop: peak memory is one windowed slice, not a second copy
    # of the whole 4-D stack (which at production size is tens of GB).
    nz = size(Eωk_out, 4)
    Nω = size(Eωk_out, 1)
    Iω_integrated = Array{Float64}(undef, Nω, nz)
    Iω_reimaged = Array{Float64}(undef, Nω, nz)
    for iz in 1:nz
        a, b = extract_signal_spectra(view(Eωk_out, :, :, :, iz),
                                      window_array, xygrid)
        Iω_integrated[:, iz] = a
        Iω_reimaged[:, iz] = b
    end
    return Iω_integrated, Iω_reimaged
end

"""
    _quadrant_spectrum!(out, Ez, quad)

Accumulate `|E|²` over the transverse points selected by the `(Nky, Nkx)` mask `quad`
into the length-`Nω` vector `out`, without materialising any field-sized temporary
(the previous broadcast allocated two per z-slice).
"""
function _quadrant_spectrum!(out, Ez::AbstractArray{<:Complex,3}, quad::AbstractMatrix)
    fill!(out, 0.0)
    @inbounds for ikx in axes(Ez, 3), iky in axes(Ez, 2)
        quad[iky, ikx] || continue
        for iω in axes(Ez, 1)
            out[iω] += abs2(Ez[iω, iky, ikx])
        end
    end
    return out
end

# ============================================================================
# Save-time extraction — reduce each z-slice as it is produced
# ============================================================================
#
# The default route saves every z-slice in full and reduces the stack afterwards. At
# production shapes that is 16 × 2.25 GiB written to a temp file and read straight back
# by the same process, to produce three (Nω, nz) arrays — 32 KB. On the CPU that cost was
# ~3 % of a delay point; after the GPU port made the propagation 12× faster it is 14-18 %,
# and 72 GiB of filesystem traffic per point across a many-instance campaign.
#
# `TraceExtractOutput` reduces each slice at the moment it is saved, so no slice is ever
# stored. On a device the reduction runs on the device array itself: `HostOutput` passes
# `y` through untouched when the handler does not need a host copy (`needs_host_y`, false
# by default), and because `simulate_delay_point` pins `step_on` to the save positions,
# `Luna.RK45.interpolate` snaps to the step endpoint and returns that same array. The
# guard below checks that rather than assuming it.
#
# NOTE this needs NO changes in Luna: the handler satisfies the existing `Output`
# interface and opts out of the device-statistics check with a `nostats_only` method on
# its own type.

"""
    _signed_window(w, arraytype) -> array

The signal window with the re-imaging sign pattern `(-1)^((iky-1)+(ikx-1))` folded in,
on `arraytype`.

Folding the sign into the window lets **one** array serve both reductions of
[`extract_signal_spectra`](@ref): the signed sum needs it, and the intensity sum is
unaffected because `|±w·E|² == |w·E|²`. Carrying a separate sign array instead would cost
a second field-sized device array, and could not be broadcast into the reduction anyway
(see the shape note in `_sqn_fused`).

A 2-D window is expanded to 3-D: the reduction takes `mapreduce` over two arrays, which
does not broadcast shapes.
"""
function _signed_window(w::AbstractArray{<:Real}, arraytype, Nω::Integer)
    nd = ndims(w)
    nd == 2 || nd == 3 || error("window_array must be 2-D or 3-D, got $nd")
    Nky, Nkx = nd == 3 ? size(w)[2:3] : size(w)
    sgn = [isodd((iky - 1) + (ikx - 1)) ? -1.0 : 1.0 for iky in 1:Nky, ikx in 1:Nkx]
    host = nd == 3 ? w .* reshape(sgn, 1, Nky, Nkx) :
                     repeat(reshape(w .* sgn, 1, Nky, Nkx), Nω, 1, 1)
    return arraytype === Array ? host : Adapt.adapt(arraytype, host)
end

"""
    _extract_slice_device!(Iint, Ireim, Ifull, Ez, wsgn, quadrng)

Reduce one `(Nω, Nky, Nkx)` **device** slice into the three per-ω spectra, writing into
column views of the host result arrays. Three reductions over dims `(2, 3)` and one small
copy back per slice; the field itself never moves.

Mathematically identical to [`extract_signal_spectra`](@ref) plus
[`_quadrant_spectrum!`](@ref), which is what a slice arriving on the host uses instead.
The sums are formed in a different order here, so results agree to rounding rather than
bitwise — the standard everywhere else on the device path.

Both operands of the two windowed reductions have the SAME shape: `mapreduce` over
several arrays does not broadcast (Base throws `DimensionMismatch`, and a GPU backend may
silently compute something else), which is why the re-imaging sign lives inside `wsgn`
rather than in a `(1, Nky, Nkx)` array of its own. The quadrant sum is a single-array
reduction over a strided view, so it reads only the quadrant instead of masking the whole
field.
"""
function _extract_slice_device!(Iint, Ireim, Ifull, Ez, wsgn, quadrng)
    ys, xs = quadrng
    Nky, Nkx = size(Ez, 2), size(Ez, 3)
    sfull = mapreduce(abs2, +, view(Ez, :, ys, xs); dims=(2, 3))
    sint = mapreduce(+, Ez, wsgn; dims=(2, 3)) do e, w
        abs2(w * e)
    end
    sacc = mapreduce(+, Ez, wsgn; dims=(2, 3)) do e, w
        w * e
    end
    copyto!(Ifull, dropdims(Array(sfull); dims=(2, 3)))
    copyto!(Iint, dropdims(Array(sint); dims=(2, 3)))
    Ireim .= abs2.(dropdims(Array(sacc); dims=(2, 3)) ./ (Nky * Nkx))
    return nothing
end

"""
    TraceExtractOutput(setup, zvec, arraytype)

A `Luna` output handler that reduces each saved z-slice to the trace spectra immediately
and keeps only the results, so the full field is never stored, streamed or transferred.

Satisfies the `Output` interface `Luna.run` uses (the save call, `willsave`, metadata
calls, and the generic `check_cache` fallback). Metadata is discarded: `ModelPNPS` builds
its own from `setup.combined_grid`, and the temp file this replaces was thrown away too.
"""
mutable struct TraceExtractOutput{S, W, HW, G}
    save_cond::S
    saved::Int
    nz::Int
    zs::Vector{Float64}
    wsgn::W               # signed windows on the state's array type; empty on the host
    hwin::HW              # the plain host windows — ALIASES `setup.window_array`, no copy
    sig_quad::BitMatrix
    quadrng::Tuple{UnitRange{Int}, UnitRange{Int}}
    xygrid::G
    Iω_w::Vector{Matrix{Float64}}
    Iω_r::Vector{Matrix{Float64}}
    Iω_full::Matrix{Float64}
end

function TraceExtractOutput(setup::TGFROGSetup, zvec::AbstractVector, arraytype)
    Nω = length(setup.grid.ω)
    nz = length(zvec)
    single = setup.window isa AbstractSignalWindow
    wins = single ? (setup.window_array,) : Tuple(setup.window_array)
    sig_quad = BitMatrix((setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)')
    # The signed window exists only to serve the device reduction; on the host the
    # original kernels take the plain window, so building it would be a field-sized
    # array for nothing.
    wsgn = arraytype === Array ? Any[] :
           Any[_signed_window(w, arraytype, Nω) for w in wins]
    TraceExtractOutput(Output.GridCondition(zvec, nz), 0, nz, Float64[],
                       wsgn, wins, sig_quad,
                       quadrant_ranges(sig_quad),   # asserts the quadrant is that rectangle
                       setup.xygrid,
                       [zeros(Float64, Nω, nz) for _ in wins],
                       [zeros(Float64, Nω, nz) for _ in wins],
                       zeros(Float64, Nω, nz))
end

# `Output.foreach_save` runs the body once per data point the save condition asks for and
# advances `o.saved`; the loop itself is Luna's, so this handler cannot drift from the
# built-in ones. `Ez` is the solver's own array — `Luna.needs_host_save` above declines
# the copy, and `RK45.interpolate` returns the stepped solution itself at a step boundary
# rather than evaluating the dense-output polynomial.
function (o::TraceExtractOutput)(y, t, dt, yfun)
    Output.foreach_save(o, y, t, dt, yfun) do iz, ts, Ez
        iz <= o.nz || error("TraceExtractOutput: more saves ($iz) than the $(o.nz) " *
                            "zsave positions it was built for")
        _reduce_slice!(o, Ez, iz)
        push!(o.zs, ts)
    end
end

"""
    _reduce_slice!(o::TraceExtractOutput, Ez, iz)

Reduce one saved slice into column `iz`, routing on where the slice actually is.

A device propagation delivers every slice on the device — including `z = 0`, which is the
step *start* rather than an endpoint and so comes through the interpolant, because
`Luna.needs_host_save` declines the copy `HostOutput` would otherwise make.

The host branch therefore serves a **host** propagation (`extract_on_save=true` on the
CPU), and any save that is genuinely interpolated. It goes through the original
[`extract_signal_spectra`](@ref)/[`_quadrant_spectrum!`](@ref) kernels against the plain
host window the setup already holds: no extra memory, no transfer, and the host result is
bit-identical to the save-the-stack route by construction rather than by a parallel
implementation that could drift.
"""
function _reduce_slice!(o::TraceExtractOutput, Ez, iz)
    if Luna.Utils.isdevice(Ez)
        for iw in eachindex(o.wsgn)
            _extract_slice_device!(view(o.Iω_w[iw], :, iz), view(o.Iω_r[iw], :, iz),
                                   view(o.Iω_full, :, iz), Ez, o.wsgn[iw], o.quadrng)
        end
    else
        _quadrant_spectrum!(view(o.Iω_full, :, iz), Ez, o.sig_quad)
        for iw in eachindex(o.hwin)
            a, b = extract_signal_spectra(Ez, o.hwin[iw], o.xygrid)
            o.Iω_w[iw][:, iz] .= a
            o.Iω_r[iw][:, iz] .= b
        end
    end
    return nothing
end

# Metadata calls from `Luna.run` (grid, simulation type). Discarded — see the docstring.
(o::TraceExtractOutput)(d; kwargs...) = nothing
(o::TraceExtractOutput)(key::AbstractString, val; kwargs...) = nothing

Output.willsave(o::TraceExtractOutput, y, t, dt) = first(o.save_cond(y, t, dt, o.saved))

# `Luna.run` refuses a device propagation with statistics, because a statistics function
# is called with the state on every step and would force a full transfer each time. This
# handler collects none, so it opts out on its own type.
Luna.nostats_only(::TraceExtractOutput) = true
# `HostOutput` must not copy `y` per step for us (already the generic default, but the
# whole design depends on it)...
Luna.needs_host_y(::TraceExtractOutput) = false
# ...nor allocate a host buffer and copy each SAVE into it: we reduce on the device and
# keep only (Nω,) results. Declining this hands over the solver's own array untouched,
# which also removes the one slice that used to come back on the host (`z = 0`, reached
# through the interpolant because it is the step start rather than an endpoint).
Luna.needs_host_save(::TraceExtractOutput) = false

"""
    _trace_results(setup, o::TraceExtractOutput) -> NamedTuple

The same `NamedTuple` the save-the-stack route returns, assembled from an extraction
handler. Kept next to that route's assembly block so the two cannot drift.
"""
function _trace_results(setup::TGFROGSetup, o::TraceExtractOutput)
    if setup.window isa AbstractSignalWindow
        return (; Iω_win=o.Iω_w[1], Iω_win_reimaged=o.Iω_r[1], Iω_full=o.Iω_full,
                  zsave=copy(o.zs))
    end
    pairs_kv = Pair{Symbol,Any}[]
    for (iw, suf) in enumerate(setup.window_suffix)
        push!(pairs_kv, Symbol("Iω_win" * suf)               => o.Iω_w[iw])
        push!(pairs_kv, Symbol("Iω_win" * suf * "_reimaged") => o.Iω_r[iw])
    end
    push!(pairs_kv, :Iω_full => o.Iω_full)
    push!(pairs_kv, :zsave   => copy(o.zs))
    return NamedTuple(pairs_kv)
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
    signal_quadrant_norm(setup::TGFROGSetup; floor_rel=1e-6)

Region-relative RK45 error norm for weak-signal accuracy at moderate `rtol`.

The default `Luna.RK45.weaknorm` measures the step error relative to the norm
of the WHOLE field, which the three pump beamlets dominate. The FWM signal is
orders of magnitude weaker, so the stepper's error budget — concentrated on
the fastest-evolving (signal) components — allows a per-step signal error of
order `rtol × ‖pump‖/‖signal‖` *relative to the signal*: with `rtol = 1e-6`
the collected signal carries measured solver errors of 0.1–1% mid-slab,
growing to ~10% at 40 µm (see `90_solver_accuracy_test.jl`). Brute-forcing
`rtol = 1e-8` fixes this at ~4× the step count.

This norm instead measures relative error separately in the signal k-space
quadrant (`kx < 0, ky < 0` — the same quadrant `Iω_full` integrates) and in
the rest of the field, and returns the larger: `rtol` then controls the
signal's OWN relative error directly, recovering weak-signal accuracy at
close to the default-`rtol` step count.

While the signal quadrant is still (nearly) empty its error is measured
against a floor of `floor_rel × ‖rest‖` (never below `atol`), so early steps
are not throttled by a 0/0 relative error; once the signal exceeds that
fraction of the pump field the relative control takes over.

Pass the result to [`simulate_delay_point`](@ref) / [`run_scan`](@ref) via
their `norm` keyword. Validate a new `(rtol, floor_rel)` choice against a
tight-`rtol` reference before production use (the pass criterion used here:
every z-slice of `Iω_win` within 1e-3 relative of an `rtol = 1e-8` run).
"""
mutable struct SignalQuadrantNorm
    sig_quad::BitMatrix
    floor_rel::Float64
    # 0/1 quadrant indicator of shape (1, Nky, Nkx) on the solver's array type, built on
    # first use by the device path and cached. See `_sqn_devmask!`.
    devmask::Any
end

SignalQuadrantNorm(sig_quad, floor_rel) = SignalQuadrantNorm(sig_quad, floor_rel, nothing)

function signal_quadrant_norm(setup::TGFROGSetup; floor_rel::Float64=1e-6)
    # Same quadrant the Iω_full collection integrates (see the comment there).
    sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
    return SignalQuadrantNorm(BitMatrix(sig_quad), floor_rel)
end

"""
    quadrant_ranges(sig_quad) -> (ys, xs)

The signal quadrant as a pair of index ranges. In FFT ordering the negative half of each
k axis is exactly the second half of its index range, so the mask is a dense rectangle —
which lets the device norm use strided views instead of a boolean mask (a `BitMatrix`
cannot enter a device kernel, and a masked reduction would need a gather).

Throws if the mask is not that rectangle, so a future change to the k-space layout
cannot silently corrupt the solver's error control.
"""
function quadrant_ranges(sig_quad::AbstractMatrix{Bool})
    Ny, Nx = size(sig_quad)
    (iseven(Ny) && iseven(Nx)) || error(
        "signal_quadrant_norm needs even transverse grid sizes, got ($Ny, $Nx)")
    ys = (Ny÷2 + 1):Ny
    xs = (Nx÷2 + 1):Nx
    (all(sig_quad[ys, xs]) && count(sig_quad) == length(ys)*length(xs)) || error(
        "the signal quadrant is not the expected index rectangle — the k-space " *
        "ordering must have changed. Refusing to guess: the error norm depends on it.")
    return ys, xs
end

function (n::SignalQuadrantNorm)(yerr, y, yn, rtol, atol)
    sig_quad = n.sig_quad
    floor_rel = n.floor_rel
    Ny, Nx = size(sig_quad)
    begin
        size(y, 2) == Ny && size(y, 3) == Nx || throw(DimensionMismatch(
            "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
            "the solver state is $(size(y))"))
        s_y_s = 0.0; s_yn_s = 0.0; s_e_s = 0.0
        s_y_r = 0.0; s_yn_r = 0.0; s_e_r = 0.0
        Nω = size(y, 1)
        @inbounds for ix in 1:Nx, iy in 1:Ny
            if sig_quad[iy, ix]
                for iw in 1:Nω
                    s_y_s += abs2(y[iw, iy, ix])
                    s_yn_s += abs2(yn[iw, iy, ix])
                    s_e_s += abs2(yerr[iw, iy, ix])
                end
            else
                for iw in 1:Nω
                    s_y_r += abs2(y[iw, iy, ix])
                    s_yn_r += abs2(yn[iw, iy, ix])
                    s_e_r += abs2(yerr[iw, iy, ix])
                end
            end
        end
        errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), atol)
        floor_s = max(atol, floor_rel * errwt_r)
        errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
        return max(sqrt(s_e_r) / (rtol * errwt_r),
                   sqrt(s_e_s) / (rtol * errwt_s))
    end
end

"""
Fused error metric for [`SignalQuadrantNorm`](@ref): the DP5 error estimate is computed
element-by-element on the fly from the stepper's stage arrays instead of materialising a
field-sized `yerr` array (Luna.RK45 allocates it lazily only for norms without a fused
version). Same per-element expression and accumulation order as the materialised path,
so the result is bit-identical.
"""
Luna.RK45.fused_errnorm(n::SignalQuadrantNorm) =
    s -> _sqn_fused(Luna.Utils.backend(s.y), n, s)

_sqn_fused(n::SignalQuadrantNorm, s) = _sqn_fused(Luna.Utils.backend(s.y), n, s)

"""
Device version of [`_sqn_fused`](@ref): the same six sums, computed as reductions along
ω into one partial per transverse point, which are then split by quadrant.

Every operand of a reduction here has the SAME shape. That is deliberate: `mapreduce`
over several arrays does not broadcast shapes (Base throws `DimensionMismatch`, and a
GPU backend may quietly compute something else instead), so the `(1, Nky, Nkx)` quadrant
mask cannot be folded into the reduction and is applied afterwards, to the small
per-transverse-point partials. Reducing over strided views of the quadrant would also
work in principle, but whole-array reductions are the shape the rest of Luna's device
code uses and the one best supported across backends.

Both halves are summed directly rather than one being `total - other`, so no
cancellation is involved. The error estimate is never materialised — it is formed inside
the reduction kernel.
"""
function _sqn_fused(::Luna.Utils.DeviceBackend, n::SignalQuadrantNorm, s)
    Ny, Nx = size(n.sig_quad)
    size(s.y, 2) == Ny && size(s.y, 3) == Nx || throw(DimensionMismatch(
        "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
        "the solver state is $(size(s.y))"))
    q = _sqn_devmask!(n, s.y)
    dt = s.dt
    e = Luna.RK45.errest
    e1 = e[1]; e3 = e[3]; e4 = e[4]; e5 = e[5]; e6 = e[6]; e7 = e[7]
    k1, _, k3, k4, k5, k6, k7 = s.ks

    # Reduce along ω first, giving one partial sum per transverse point. Every operand
    # here has the SAME shape: `mapreduce` over several arrays does not broadcast
    # shapes (Base throws DimensionMismatch; a GPU backend may silently compute
    # something else), so mixing the (1, Nky, Nkx) mask into the reduction is invalid.
    # The error estimate is still never materialised — it is formed inside the kernel.
    Sy = mapreduce(abs2, +, s.y; dims=1)
    Syn = mapreduce(abs2, +, s.yn; dims=1)
    Se = mapreduce(+, k1, k3, k4, k5, k6, k7; dims=1) do a1, a3, a4, a5, a6, a7
        abs2(0 + dt*a1*e1 + dt*a3*e3 + dt*a4*e4 + dt*a5*e5 + dt*a6*e6 + dt*a7*e7)
    end

    # Split by quadrant on the small (1, Nky, Nkx) partials. Both halves are summed
    # directly rather than one being `total - other`, so no cancellation is involved.
    r = 1 .- q
    s_y_s = sum(Sy .* q); s_yn_s = sum(Syn .* q); s_e_s = sum(Se .* q)
    s_y_r = sum(Sy .* r); s_yn_r = sum(Syn .* r); s_e_r = sum(Se .* r)

    errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), s.atol)
    floor_s = max(s.atol, n.floor_rel * errwt_r)
    errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
    return max(sqrt(s_e_r) / (s.rtol * errwt_r),
               sqrt(s_e_s) / (s.rtol * errwt_s))
end

"""
The quadrant indicator as a `(1, Nky, Nkx)` array on `y`'s array type, built once and
cached on the norm. Broadcasting it into the reduction costs one small array (8 MB even
at the largest campaign shape) and keeps the reduction over whole, contiguous arrays.

The mask is built from the same `BitMatrix` the host path uses, via
[`quadrant_ranges`](@ref) — so its rectangle assertion still guards the device path.
"""
function _sqn_devmask!(n::SignalQuadrantNorm, y)
    Ny, Nx = size(n.sig_quad)
    want = Base.typename(typeof(y)).wrapper
    if isnothing(n.devmask) || size(n.devmask) != (1, Ny, Nx) ||
       Base.typename(typeof(n.devmask)).wrapper !== want
        ys, xs = quadrant_ranges(n.sig_quad)   # asserts the mask is that rectangle
        host = zeros(Float64, 1, Ny, Nx)
        host[1, ys, xs] .= 1.0
        n.devmask = Adapt.adapt(want, host)
    end
    return n.devmask
end

function _sqn_fused(::Luna.Utils.CPUBackend, n::SignalQuadrantNorm, s)
    sig_quad = n.sig_quad
    floor_rel = n.floor_rel
    Ny, Nx = size(sig_quad)
    y = s.y
    yn = s.yn
    size(y, 2) == Ny && size(y, 3) == Nx || throw(DimensionMismatch(
        "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
        "the solver state is $(size(y))"))
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    dt = s.dt
    errest = Luna.RK45.errest
    e1 = errest[1]; e3 = errest[3]; e4 = errest[4] # errest[2] == 0, skipped as in Luna
    e5 = errest[5]; e6 = errest[6]; e7 = errest[7]
    s_y_s = 0.0; s_yn_s = 0.0; s_e_s = 0.0
    s_y_r = 0.0; s_yn_r = 0.0; s_e_r = 0.0
    Nω = size(y, 1)
    @inbounds for ix in 1:Nx, iy in 1:Ny
        if sig_quad[iy, ix]
            for iw in 1:Nω
                yerr = 0 + dt*k1[iw,iy,ix]*e1 + dt*k3[iw,iy,ix]*e3 +
                           dt*k4[iw,iy,ix]*e4 + dt*k5[iw,iy,ix]*e5 +
                           dt*k6[iw,iy,ix]*e6 + dt*k7[iw,iy,ix]*e7
                s_y_s += abs2(y[iw, iy, ix])
                s_yn_s += abs2(yn[iw, iy, ix])
                s_e_s += abs2(yerr)
            end
        else
            for iw in 1:Nω
                yerr = 0 + dt*k1[iw,iy,ix]*e1 + dt*k3[iw,iy,ix]*e3 +
                           dt*k4[iw,iy,ix]*e4 + dt*k5[iw,iy,ix]*e5 +
                           dt*k6[iw,iy,ix]*e6 + dt*k7[iw,iy,ix]*e7
                s_y_r += abs2(y[iw, iy, ix])
                s_yn_r += abs2(yn[iw, iy, ix])
                s_e_r += abs2(yerr)
            end
        end
    end
    errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), s.atol)
    floor_s = max(s.atol, floor_rel * errwt_r)
    errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
    return max(sqrt(s_e_r) / (s.rtol * errwt_r),
               sqrt(s_e_s) / (s.rtol * errwt_s))
end

"""Provenance label for the error norm used by a scan."""
_norm_name(n::SignalQuadrantNorm) = "signal_quadrant(floor_rel=$(n.floor_rel))"
_norm_name(f) = f === Luna.RK45.weaknorm ? "weaknorm" : "custom"

"""
    delayed_input(setup, τ) -> Array{ComplexF64,3}

Coherent input field for scan delay `τ`, in the paper's **gate-delay
convention**: the stored trace ``T(ω, τ)`` has the GATE pair delayed by `+τ`
relative to the probe. Physically the probe arm carries the delay stage, so
the probe is delayed by `-τ`, which equals gating at `+τ` up to a global time
shift that the time-integrating measurement cannot see. The same-τ gate pair
stays untouched, so the smearing structure (same-τ gate pair; paper
Appendix A.2) is preserved. Files written with this convention carry
`/grid/delay_convention = "gate"` and need NO delay-axis reversal on loading
(croak's `reverse_trace` auto-detects; legacy marker-less files are reversed
as before).
"""
function delayed_input(setup::TGFROGSetup, τ::Real)
    # Single fused broadcast: one field-sized allocation instead of two
    # (apply_delay's intermediate plus the sum) — at production size the
    # difference is a 2.15 GB transient per worker, which is exactly the
    # margin that OOM-killed procs=2 tasks at the 100G quota line.
    # `ωd` is the frequency axis on the beamlets' array type, so the phase ramp is built
    # where they live and no host vector is broadcast against device data.
    phase = reshape(exp.(1im .* (-setup.ωd .* -τ)), (:, 1, 1))
    Eωk = setup.Eωk_g12 .+ setup.Eωk_t_base .* phase
    # With `beamlets_on_host` the sum is formed on the host and uploaded once here,
    # trading two resident device fields for one transfer per delay point.
    return _match_arraytype(Eωk, setup.transform)
end

_match_arraytype(Eωk, transform) =
    Luna.Utils.isdevice(transform.Eto) && !Luna.Utils.isdevice(Eωk) ?
        Adapt.adapt(typeof(transform.Eto).name.wrapper, Eωk) : Eωk

"""Delay phase angle ``-ωτ`` on the grid's frequency axis."""
grid_delay_phase(grid::Grid.TimeGrid, τ::Real) = -grid.ω .* τ

"""
    simulate_delay_point(setup::TGFROGSetup, τi;
                         nz=2, init_dz=5e-7, rtol=1e-6, max_dz=0.0,
                         filename=nothing,
                         skip_propagation=false)
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

Pass `filename` to persist the propagation to disk: the `Luna.run` then writes
to an `Output.HDF5Output` at that path instead of an in-memory
`Output.MemoryOutput`. Downstream extraction is identical either way (both
outputs index as `output["Eω"]`/`output["z"]`); `filename` is ignored when
`skip_propagation=true`.

Setting `skip_propagation=true` substitutes the input field for the
Luna output, exercising every other code path. This is used by the unit
tests to keep the suite fast and deterministic.
"""
function simulate_delay_point(setup::TGFROGSetup, τi::Real;
                              nz::Int=2,
                              zsave::Union{Integer,AbstractVector}=nz,
                              init_dz::Float64=5e-7,
                              rtol::Float64=1e-6,
                              max_dz::Float64=0.0,
                              norm=Luna.RK45.weaknorm,
                              twin_period::Int=1,
                              filename::Union{Nothing,AbstractString}=nothing,
                              extract_on_save::Union{Nothing,Bool}=nothing,
                              skip_propagation::Bool=false)
    # --- Resolve the propagation snapshot grid ---------------------------
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    nz_eff = length(zvec)

    # --- Build the delayed test beam and coherently superpose ------------
    Eωk_in = delayed_input(setup, τi)

    # --- Save-time extraction --------------------------------------------
    # Reduce each z-slice as it is produced rather than saving the whole field and
    # reducing afterwards; `filename`/streaming then has nothing to stream. Defaults ON
    # for a device propagation, where the saved stack costs 14-18 % of the delay point in
    # temp-file traffic, and OFF on the host, whose current route is validated
    # bit-for-bit against the production references — pass `true` to use it there too
    # (it is bit-identical: the same host kernels on the same arrays, minus a lossless
    # HDF5 round trip).
    if something(extract_on_save, Luna.Utils.isdevice(Eωk_in)) && !skip_propagation
        arrwrap = Base.typename(typeof(Eωk_in)).wrapper
        o = TraceExtractOutput(setup, zvec, arrwrap)
        Luna.run(Eωk_in, setup.grid, setup.linop, setup.transform, setup.FT, o;
                 init_dz=init_dz, rtol=rtol, norm=norm,
                 max_dz=(max_dz > 0 ? max_dz : setup.grid.zmax/2),
                 step_on=zvec, preserve_input=false, twin_period=twin_period)
        o.saved == nz_eff || error(
            "save-time extraction got $(o.saved) of $nz_eff z-slices; the stepper did " *
            "not land on every save position")
        return _trace_results(setup, o)
    end

    # --- Propagate (or fake the propagation for tests) -------------------
    if skip_propagation
        # Fake a (Nω, Nky, Nkx, nz) output by stacking the input nz times.
        Nω, Nky, Nkx = size(Eωk_in)
        getslice = _ -> Eωk_in
        z_realized = copy(zvec)
    else
        save_cond = Output.GridCondition(zvec, nz_eff)
        # cache=false: the crash-resume cache would rewrite the full field-sized array
        # to the (throwaway) temp file on every save
        output = isnothing(filename) ?
            Output.MemoryOutput(save_cond, "Eω", "z") :
            Output.HDF5Output(filename, save_cond, "Eω", "z", Output.nostats, false,
                              nothing, false)
        # NOTE the solver tolerance: the RK45 error is controlled relative to
        # the FULL field, which the pump beamlets dominate. The FWM signal is
        # orders of magnitude weaker (signal/pump field ratio ∝ pulse energy),
        # so its RELATIVE accuracy is ~(pump/signal) × rtol and degrades with
        # propagation distance and with decreasing energy. At the defaults
        # (rtol = 1e-6, max_dz = zmax/2, 0.1 µJ) the collected signal spectrum
        # carries ~1% (mid-slab) to ~10% (40 µm) solver-dependent error —
        # measured in FROG_paper_new/90_solver_accuracy_test.jl. Tighten to
        # rtol ≤ 1e-8 and a µm-scale max_dz for quantitative small-effect
        # studies (Raman A/B, low-energy runs, residual-floor analyses).
        # step_on = the save positions: the stepper lands on each zsave point
        # exactly, so saved slices are step endpoints rather than dense-output
        # interpolations — the interpolant shares the error norm's weak-signal
        # blind spot and interpolated saves scatter at the percent level
        # between runs (FINDINGS F14.12 twin test).
        # preserve_input=false: the solver adopts Eωk_in as a working buffer instead of
        # copying it (one field-sized allocation less at peak); Eωk_in is not used again
        # on this path
        Luna.run(Eωk_in, setup.grid, setup.linop, setup.transform, setup.FT,
                  output; init_dz=init_dz, rtol=rtol, norm=norm,
                  max_dz=(max_dz > 0 ? max_dz : setup.grid.zmax/2),
                  step_on=zvec, preserve_input=false, twin_period=twin_period)
        # Slice access: streamed runs read one z-slice at a time back from
        # the HDF5 file (Output.getindex opens the file per read), so the
        # full (ω, ky, kx, z) stack — tens of GB at production size — never
        # exists in memory; in-memory runs use free views into it.
        if isnothing(filename)
            Eωk_mem = output["Eω"]
            getslice = iz -> view(Eωk_mem, :, :, :, iz)
        else
            getslice = iz -> output["Eω", :, :, :, iz]
        end
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
    Nω = length(setup.grid.ω)
    single_window = setup.window isa AbstractSignalWindow
    wins = single_window ? (setup.window_array,) : Tuple(setup.window_array)
    Iω_full = Array{Float64}(undef, Nω, nz_eff)
    Iω_w = [Array{Float64}(undef, Nω, nz_eff) for _ in wins]
    Iω_r = [Array{Float64}(undef, Nω, nz_eff) for _ in wins]
    for iz in 1:nz_eff
        Ez = getslice(iz)
        _quadrant_spectrum!(view(Iω_full, :, iz), Ez, sig_quad)
        for (iw, arr) in enumerate(wins)
            a, b = extract_signal_spectra(Ez, arr, setup.xygrid)
            Iω_w[iw][:, iz] = a
            Iω_r[iw][:, iz] = b
        end
    end

    # --- Assemble the per-window outputs ----------------------------------
    if single_window
        return (; Iω_win=Iω_w[1], Iω_win_reimaged=Iω_r[1], Iω_full,
                  zsave=z_realized)
    else
        pairs_kv = Pair{Symbol,Any}[]
        for (iw, suf) in enumerate(setup.window_suffix)
            push!(pairs_kv, Symbol("Iω_win" * suf)              => Iω_w[iw])
            push!(pairs_kv, Symbol("Iω_win" * suf * "_reimaged") => Iω_r[iw])
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
    _completed_scanidcs(scan_name) -> Set{Int}

Scan indices already present in `<scan_name>_collected.h5`, i.e. those whose trace data
is not all zero. An empty set if the file does not exist yet.

Reads one point at a time: the file may be large and this runs before any propagation.
"""
function _completed_scanidcs(scan_name::AbstractString)
    fn = scan_name * "_collected.h5"
    isfile(fn) || return Set{Int}()
    done = Set{Int}()
    HDF5.h5open(fn, "r") do f
        ks = filter(k -> startswith(k, "Iω"), collect(keys(f)))
        isempty(ks) && return
        d = f[first(ks)]
        for i in 1:size(d, 3)
            any(!iszero, d[:, :, i]) && push!(done, i)
        end
    end
    return done
end

"""
    run_scan(setup, τs;
             scan_name, exec,
             nz=2, zsave=nz, init_dz=5e-7, rtol=1e-6, max_dz=0.0,
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
function run_scan(setup_fn::Function, τs::AbstractVector;
                  scan_name::AbstractString,
                  exec,
                  nz::Int=2, zsave::Union{Integer,AbstractVector}=nz,
                  init_dz::Float64=5e-7,
                  rtol::Float64=1e-6,
                  max_dz::Float64=0.0,
                  norm=Luna.RK45.weaknorm,
                  twin_period::Int=1,
                  norm_builder::Union{Nothing,Function}=nothing,
                  fftw_threads::Int=0,
                  fftw_mode::Symbol=:estimate,
                  stream::Bool=true,
                  extract_on_save::Union{Nothing,Bool}=nothing,
                  skip_existing::Bool=false,
                  extra_outputs::Function=(out)->NamedTuple())
    scan = Scans.Scan(scan_name, exec; τ=τs)
    # Resume: `Output.scansave` allocates the full (Nω, nz, Nτ) datasets up front and
    # fills points in as they complete, so an all-zero slice IS the marker for "not yet
    # computed" — the same test `verify_against_collected` uses. Skipping those indices
    # lets an interrupted scan continue where it stopped, which matters when the machine
    # is rented by the hour.
    done_idcs = skip_existing ? _completed_scanidcs(scan_name) : Set{Int}()
    isempty(done_idcs) || @info "resuming: skipping $(length(done_idcs)) completed " *
                                "point(s) of $(length(τs)) in $(scan_name)_collected.h5"
    # LAZY SETUP: everything expensive — building the multi-GB beamlet arrays,
    # planning the FFTs, resolving metadata — happens inside the scan closure,
    # on the FIRST point this process executes. At submission time
    # (SlurmExec/SSHExec generate the job script and submit without running
    # any point), `setup_fn` is never called, so launching a scan from a
    # memory-limited login node costs seconds and megabytes. Each array task
    # re-executes the script and builds its own setup on its first point.
    setup = nothing
    zvec = nothing
    cg = nothing
    normx = nothing
    Luna.runscan(scan) do scanidx, τi
        # Before anything else, including building the setup: a resumed scan should pay
        # nothing at all for a point it already has.
        scanidx in done_idcs && return nothing
        if setup === nothing
            # Per-process FFTW threading, applied exactly where the plans are
            # created. In `procs` (multi-worker) scans the workers never
            # execute the script's top level, so a top-level
            # `set_fftw_threads` call does not reach them — this does. With
            # `procs` workers sharing `cpus` cores, pass
            # `fftw_threads = cpus ÷ procs`.
            fftw_threads > 0 && Luna.set_fftw_threads(fftw_threads)
            # Same worker-visibility problem as the threads: a top-level
            # `set_fftw_mode` in the script never reaches `procs` workers,
            # whose planning would then use Luna's default — MEASURE-class
            # planning of production-size 3D transforms takes tens of minutes
            # per worker. :estimate plans in seconds and is what every
            # ModelPNPS production script uses.
            Luna.set_fftw_mode(fftw_mode)
            setup = setup_fn()::TGFROGSetup
            zvec = _resolve_zsave(zsave, setup.grid.zmax)
            # `norm_builder` exists because a setup-derived norm (e.g.
            # `signal_quadrant_norm`) cannot be constructed before the setup:
            # pass `norm_builder = s -> signal_quadrant_norm(s)` instead of
            # `norm` and it is built here, lazily.
            normx = isnothing(norm_builder) ? norm : norm_builder(setup)
            # Metadata: shallow copy so the shared combined_grid is unmutated;
            # /grid/zsave equals the per-point realized out.zsave (the
            # resolution is deterministic). rtol/max_dz/norm are provenance.
            cg = copy(setup.combined_grid)
            cg["zsave"] = zvec
            cg["rtol"] = rtol
            cg["max_dz"] = max_dz > 0 ? max_dz : setup.grid.zmax/2
            # Delay-convention marker: traces are stored in the paper's
            # gate-delay frame; loaders must not reverse the axis.
            cg["delay_convention"] = "gate"
            cg["error_norm"] = _norm_name(normx)
            # Apodisation cadence. 1 = the spectral/temporal windows are applied
            # in place after EVERY accepted step, which makes the damping scale
            # with the step count and the scheme non-convergent in rtol; large
            # values apply them only at saves (Output.willsave), which with
            # step_on are at identical positions for any rtol.
            cg["twin_period"] = twin_period
        end
        # Stream the propagation slices to a node-local temp file instead of
        # holding the (ω, ky, kx, z) stack in memory (~2.15 GB × nz at
        # production size); only the extracted (Nω, nz) spectra survive.
        # Save-time extraction stores no slices, so there is nothing to stream: skip the
        # temp file rather than creating one that is written to zero times.
        onsave = something(extract_on_save, Luna.Utils.isdevice(setup.transform.Eto))
        fname = (stream && !onsave) ? tempname() * "_pnps.h5" : nothing
        # try/finally, not a plain call followed by rm: if the point throws, the
        # temp file is 36 GB of orphan at production size. run_scan CATCHES the
        # error per point and moves to the next one, so without this a transient
        # full disk becomes a permanent one -- each failing point leaks another
        # file and none of them are ever cleaned. That is what turned an
        # overflowing /tmp into three scans returning 2-24% of their delays on
        # 2026-08-24, with every SLURM job still reporting COMPLETED 0:0.
        #
        # invokelatest: with `arraytype=:cuda` the GPU package was loaded *inside* this
        # closure, so its methods are newer than the world this closure is running in
        # and would be invisible to every device kernel below. One dynamic dispatch per
        # delay point, against a propagation of minutes.
        out = try
            Base.invokelatest(simulate_delay_point, setup, τi;
                              zsave=zvec, init_dz=init_dz,
                              rtol=rtol, max_dz=max_dz, norm=normx,
                              twin_period=twin_period, filename=fname,
                              extract_on_save=onsave)
        finally
            stream && !isnothing(fname) && rm(fname; force=true)
        end
        # Return freed field-sized garbage (the point's input array, extraction
        # temporaries) to the allocator before the next point starts — with
        # two workers sharing a tight cgroup, un-collected garbage from one
        # worker coinciding with the other's peak is an OOM risk.
        GC.gc()
        # A GPU array library keeps its own memory pool, which garbage collection alone
        # does not return. Without this the next point can find the card full.
        Luna.device_reclaim()
        # `zsave` is metadata (stored in /grid/zsave), not a per-delay dataset.
        out_save = Base.structdiff(out, NamedTuple{(:zsave,)})
        Output.scansave(scan, scanidx; grid=cg, out_save...,
                         extra_outputs(out)...)
    end
    return nothing
end

"""
    _scan_peak(dset) -> Float64

Largest absolute value over every *computed* delay point of a collected trace dataset.
Read one point at a time rather than whole: this runs against a file a scan may still be
writing, and the datasets grow with the delay count.
"""
function _scan_peak(dset)
    pk = 0.0
    for i in 1:size(dset, 3)
        s = dset[:, :, i]
        any(!iszero, s) || continue # not yet computed
        pk = max(pk, maximum(abs, s))
    end
    return pk
end

"""
    verify_against_collected(setup_args, collected, scanidcs;
                             zsave, init_dz=5e-7, rtol=1e-6, max_dz=0.0,
                             norm=Luna.RK45.weaknorm, stream=true)
        -> Vector{Dict}

Recompute selected delay points of an existing scan and compare against the collected
HDF5 file — the A/B harness for validating a new code path (or a changed grid) against
reference data.

For each scan index in `scanidcs`, the delay `τ` is read from `/scanvariables/τ` in
`collected`, the point is recomputed via [`simulate_delay_point`](@ref) with the given
solver settings (pass the SAME settings the reference scan used, unless deliberately
testing a change), and every returned trace dataset (`Iω_win`, `Iω_full`, ...) present in
the file is compared. Reference points that are still all-zero (not yet computed by a
running scan) are reported as `NaN` and skipped.

Returns one `Dict` per point with the delay, wall time, `Sys.maxrss()` [GiB], and for
each dataset `ks` the global relative difference
`maximum(abs, new - ref)/maximum(abs, ref)`, plus three diagnostics:

| key | meaning |
|---|---|
| `ks` | max abs difference ÷ **this point's** reference peak |
| `ks*"\\|relscan"` | max abs difference ÷ the **scan-wide** reference peak |
| `ks*"\\|refpeak"` | this point's reference peak |
| `ks*"\\|scanpeak"` | the scan-wide reference peak |

Both normalisations matter. A delay-scan wing carries a signal orders of magnitude below
the τ≈0 signal, so a difference that is irrelevant in the assembled trace can still be a
large fraction of that point's own peak. `relscan` is what a FROG retrieval sees; the
own-peak number is the stricter statement about the code path.

To test a grid change (e.g. N=640 against an N=1024 reference), pass the changed `N`
inside `setup_args` — differences then reflect the grid, not the code.

!!! note
    Strict comparisons need matched FFT configuration: run with the same `fftw_threads`
    and `fftw_mode` as the reference scan (FFT algorithm choice affects round-off).
    Julia-level threading (`JULIA_NUM_THREADS`) does NOT affect results and can be used
    freely to speed up verification.
"""
function verify_against_collected(setup_args::NamedTuple, collected::AbstractString,
                                  scanidcs::AbstractVector{<:Integer};
                                  zsave::Union{Integer,AbstractVector},
                                  init_dz::Float64=5e-7,
                                  rtol::Float64=1e-6,
                                  max_dz::Float64=0.0,
                                  norm=Luna.RK45.weaknorm,
                                  twin_period::Int=1,
                                  stream::Bool=true,
                                  extract_on_save::Union{Nothing,Bool}=nothing)
    setup = _build_setup_resolved(setup_args)
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    results = Dict{String,Any}[]
    scanpeaks = Dict{String,Float64}()
    HDF5.h5open(collected, "r") do f
        τs = read(f["scanvariables"]["τ"])
        # k-space-integrated spectra (Iω_win, Iω_full, ...) are in FFT-bin units which
        # scale as N⁴ at fixed R (Parseval over the transverse FFT: Σₖ|Eₖ|² = N²Σₓ|Eₓ|²
        # and Σₓ|Eₓ|² ∝ N² at fixed physical energy). When comparing across grid sizes,
        # rescale the recomputed values to the reference grid's units. The re-imaged
        # (real-space pixel) spectra are N-invariant and are not rescaled.
        Nnew = length(setup.xygrid.x)
        Nref = haskey(f["grid"], "x") ? length(read(f["grid"]["x"])) : Nnew
        kscale = (Nref/Nnew)^4
        if Nref != Nnew
            xref_max = maximum(abs, read(f["grid"]["x"]))
            isapprox(xref_max, maximum(abs, setup.xygrid.x); rtol=0.05) ||
                @warn "reference and recomputed grids differ in physical extent; " *
                      "the N⁴ unit rescaling assumes fixed R and may be invalid"
            @info "grid size differs (N=$Nnew vs reference $Nref): " *
                  "k-integrated datasets rescaled by (Nref/N)⁴ = $kscale before comparison"
        end
        for idx in scanidcs
            τi = τs[idx]
            onsave = something(extract_on_save,
                               Luna.Utils.isdevice(setup.transform.Eto))
            fname = (stream && !onsave) ? tempname() * "_verify.h5" : nothing
            GC.gc()
            # On a device the array library's pool is not returned by GC alone, so
            # successive points would accumulate it until the card fills.
            Luna.device_reclaim()
            devfree0 = Luna.device_memory_status()
            t0 = time()
            # invokelatest for the same reason as in `run_scan`: the GPU package may
            # have been loaded by `_build_setup_resolved` above, i.e. during this call.
            out = try
                Base.invokelatest(simulate_delay_point, setup, τi;
                                  zsave=zvec, init_dz=init_dz,
                                  rtol=rtol, max_dz=max_dz, norm=norm,
                                  twin_period=twin_period, filename=fname,
                                  extract_on_save=onsave)
            finally
                # See the note at the streaming call above: a throwing point
                # must not leave its temp file behind.
                stream && !isnothing(fname) && rm(fname; force=true)
            end
            wall = time() - t0
            point = Dict{String,Any}("scanidx" => idx, "τ" => τi,
                                     "wall_s" => wall,
                                     "maxrss_GiB" => Sys.maxrss()/2^30)
            # `maxrss` is HOST memory; on a device that is only the input construction,
            # the save buffer and the runtime, so report the device side separately.
            devfree1 = Luna.device_memory_status()
            if !isnothing(devfree0) && !isnothing(devfree1)
                point["device_used_GiB"] = (devfree0[1] - devfree1[1])/2^30
                point["device_free_GiB"] = devfree1[1]/2^30
            end
            out_save = Base.structdiff(out, NamedTuple{(:zsave,)})
            for (k, v) in pairs(out_save)
                ks = string(k)
                haskey(f, ks) || continue
                ref = f[ks][:, :, idx]
                if !any(!iszero, ref)
                    point[ks] = NaN # reference point not (yet) computed
                    continue
                end
                size(ref) == size(v) || error(
                    "dataset $ks: recomputed size $(size(v)) does not match " *
                    "reference $(size(ref)) — grid mismatch? (compare N via /grid/x)")
                vn = endswith(ks, "_reimaged") ? v : v .* kscale
                absdiff = maximum(abs.(vn .- ref))
                refpeak = maximum(abs, ref)
                point[ks] = absdiff / refpeak
                # Normalising to the point's OWN peak makes a delay-scan wing — where
                # the signal beam is orders of magnitude below the τ≈0 signal — look
                # catastrophic for a difference that is negligible in the assembled
                # trace. Report the scan-wide normalisation alongside: that is the
                # quantity a FROG retrieval actually sees. Keep both — a wing point
                # disagreeing at its own scale is still worth knowing about.
                scanpeak = get!(scanpeaks, ks) do
                    _scan_peak(f[ks])
                end
                point[ks*"|relscan"] = scanpeak > 0 ? absdiff/scanpeak : NaN
                point[ks*"|refpeak"] = refpeak
                point[ks*"|scanpeak"] = scanpeak
            end
            @info "verified scan point" point["scanidx"] point["τ"] point["wall_s"] point["maxrss_GiB"]
            for k in sort(collect(keys(point)))
                (startswith(k, "Iω") && !occursin('|', k)) || continue
                @info "  $k: rel(own peak) = $(point[k])  " *
                      "rel(scan peak) = $(get(point, k*"|relscan", NaN))  " *
                      "peak = $(get(point, k*"|refpeak", NaN)) of " *
                      "$(get(point, k*"|scanpeak", NaN))"
            end
            push!(results, point)
        end
    end
    return results
end

"""Eager variant: wrap an already-built setup (costs nothing extra when the
setup exists anyway, e.g. in interactive use or LocalExec runs)."""
run_scan(setup::TGFROGSetup, τs::AbstractVector; kwargs...) =
    run_scan(() -> setup, τs; kwargs...)

"""
    run_scan(setup_args::NamedTuple, τs; kwargs...)

RECOMMENDED for scan scripts: pass the [`build_setup`](@ref) keyword
arguments as a NamedTuple, e.g.

    setup_args = (; λ0, τfwhm, energy, thickness, material,
                    mask_diam, mask_spacing, λlims, beam, window,
                    R=366.0e-6, N=1024)
    run_scan(setup_args, τ; ...)

The setup is then built lazily on each process that executes scan points.
This form is robust under EVERY execution mode, including multi-worker
(`procs > 0`) queue scans: a NamedTuple of parameters serialises to the
workers by value, whereas a NAMED function defined in a script
(`make_setup() = ...`) serialises by reference and fails to deserialise on
workers (Julia ships code only for anonymous closures). The wrapping closure
here is defined inside ModelPNPS, which Luna loads on the workers.
"""
run_scan(setup_args::NamedTuple, τs::AbstractVector; kwargs...) =
    run_scan(() -> _build_setup_resolved(setup_args), τs; kwargs...)

"""
    _build_setup_resolved(setup_args) -> TGFROGSetup

Build the setup, resolving `arraytype` FIRST and then calling [`build_setup`](@ref)
through `Base.invokelatest`.

`Luna.resolve_arraytype(:cuda)` loads the GPU package at run time, and methods defined
by a package loaded *during* a call are not visible to that same call — Julia rejects
them as "too new to be called from this world context". Resolving first and invoking
afterwards puts the construction in a world where the array type's constructors exist.

This is why a scan script should pass `arraytype=:cuda` inside `setup_args` and let this
happen on the compute node, rather than loading the GPU package itself.
"""
function _build_setup_resolved(setup_args::NamedTuple)
    args = if haskey(setup_args, :arraytype)
        merge(setup_args, (; arraytype=Luna.resolve_arraytype(setup_args.arraytype)))
    else
        setup_args
    end
    return Base.invokelatest(build_setup; args...)::TGFROGSetup
end

# NOTE on GPU runs: pass `arraytype=:cuda` (and optionally `beamlets_on_host=true`)
# inside `setup_args`, NOT as a `run_scan` keyword. The setup is built lazily inside the
# scan closure, which only ever executes on a compute node — so the GPU package is
# loaded there and never on the submitting host, which may have no GPU at all.

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
        # Field-mode files carry a monotonic rfft half-spectrum; envelope files carry the
        # FFT-ordered relative-frequency axis that needs shifting. Absent marker = envelope,
        # so every file written before field mode existed loads exactly as before.
        field_mode = haskey(g, "field_mode") && read(g["field_mode"]) != 0
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

        # --- Put the ω axis in ascending order (dim 1) ---
        shift(x) = field_mode ? x : FFTW.fftshift(x)
        shift(x, d) = field_mode ? x : FFTW.fftshift(x, d)
        ω           = shift(ω_raw)
        Iω          = shift(Iω_raw)
        trace       = shift(win, 1)
        Iω_beamlet  = isnothing(Iω_beam_raw) ? nothing : shift(Iω_beam_raw)

        nt = (; ω, ω0, t, τ, trace, Iω, It, τfwhm, field_mode)
        nt = isnothing(zsave)       ? nt : merge(nt, (; zsave))
        nt = isnothing(Iω_beamlet)  ? nt : merge(nt, (; Iω_beamlet))
        nt = isnothing(It_beam_raw) ? nt : merge(nt, (; It_beamlet=It_beam_raw))
        nt = isnothing(Ito_beam_raw) ? nt : merge(nt, (; Ito_beamlet=Ito_beam_raw))
        isnothing(To_raw) ? nt : merge(nt, (; To=To_raw, Ito=Ito_raw))
    end
end

end # module
