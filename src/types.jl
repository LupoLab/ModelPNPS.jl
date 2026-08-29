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
    HE11Beam(; a, f_coll, f_foc)

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
    GaussianBeam(; w0, f_foc)

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
    PhysicalMaskWindow(holex, holey, holediam, zmask, apod, apod_param)
    PhysicalMaskWindow(; holex, holey, holediam, zmask,
                       apod = :supergauss, apod_param = nothing)

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
struct PhysicalMaskWindow{T} <: AbstractSignalWindow
    holex::Float64
    holey::Float64
    holediam::Float64
    zmask::Float64
    apod::Symbol
    apod_param::T
end
PhysicalMaskWindow(;
    holex, holey, holediam, zmask,
    apod = :supergauss, apod_param = nothing
) =
    PhysicalMaskWindow(holex, holey, holediam, zmask, apod, apod_param)

"""
    PlanckWindow(kxc, kyc, kwidth, pad)
    PlanckWindow(; kxc, kyc, kwidth, pad = 1.25)

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
PlanckWindow(; kxc, kyc, kwidth, pad = 1.25) = PlanckWindow(kxc, kyc, kwidth, pad)

"""
    PlanckOmegaWindow(xc, yc, holediam, f_foc, pad)
    PlanckOmegaWindow(; xc, yc, holediam, f_foc, pad = 1.25)

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
PlanckOmegaWindow(; xc, yc, holediam, f_foc, pad = 1.25) =
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
struct TGFROGSetup{GT, LO, TR, FTT, EF, WIN, WA, BT, PT}
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
    energyfun_ω::EF

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
    combined_grid::Dict{String, Any}
end
