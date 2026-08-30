```@meta
CurrentModule = ModelPNPS
```

# Nonlinear Response

The TG-FROG signal is a χ⁽³⁾ four-wave-mixing product, so what
[`build_setup`](@ref) puts on the right-hand side of the propagation equation decides
what physics the trace contains. This page describes the two responses available on
the default envelope grid — the instantaneous electronic Kerr effect, and the
optional delayed nuclear (Raman) contribution — the convention that ties them
together, and when the delayed part is worth its cost.

The third option, propagating a real carrier-resolved field instead of an envelope,
changes the *representation* rather than the response and has its own page:
[Field-Resolved Mode](field_mode.md).

## The default: instantaneous Kerr

With no extra keywords the response is Luna's `Nonlinear.Kerr_env(χ3)`, the
instantaneous electronic nonlinearity of the substrate material, with χ⁽³⁾ taken
from `Luna.PhysData` for the `material` symbol you pass:

```math
P_{\text{NL}} = \tfrac{3}{4}\,\varepsilon_0\,\chi^{(3)}\,|E|^2 E
```

The 3/4 is the envelope convention: it is what remains of ``E^3`` once the
third-harmonic and counter-rotating terms are dropped, which is exactly the
approximation an envelope grid makes. `Kerr_env` applies it internally.

This is the right response for essentially every production run. Fused silica's
electronic response is close to instantaneous on a few-femtosecond timescale, and
the trace is dominated by it.

## The delayed nuclear response

Set `raman = true` to add the delayed nuclear contribution. χ⁽³⁾ is then split into
an instantaneous electronic part and a delayed part carrying the nuclear fraction
``f_R``:

```math
P_{\text{NL}} = \tfrac{3}{4}\,\varepsilon_0\,\chi^{(3)}
    \left[(1 - f_R)\,|E|^2 E + f_R\, E\,(h_R \circledast |E|^2)\right]
```

```julia
setup = build_setup(;
    λ0 = 260e-9, τfwhm = 2e-15, energy = 0.2e-6,
    thickness = 40e-6, material = :SiO2,
    mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
    beam, window,
    raman = true,                # default raman_fraction = 0.18
)
```

### The prefactor convention

Both terms end with the **same** prefactor. That is the property that matters, and it
is not automatic.

`Kerr_env` applies the envelope factor 3/4 internally, while Luna's Raman response
applies 1/2 in its `sqr!` step. The Raman scale therefore carries the compensating
``(3/4)/(1/2) = 3/2``, plus the explicit ``\varepsilon_0`` that `Kerr_env` adds for
itself. This is the convention of Luna's `prop_gnlse`, in which ``f_R`` is
envelope-defined, and it is what makes the quasi-static (long-pulse) limit of the
split response identical to the Kerr-only response.

!!! warning "Luna's low-level envelope examples differ"
    Luna's own low-level silica envelope examples omit the 3/2 and so under-weight
    Raman by a factor 2/3 relative to this convention. If you compare a ModelPNPS
    run against one of those, that factor is the difference. The equal-prefactor
    property here is pinned by a quasi-static consistency test in the test suite,
    which drives the split response with a long pulse and requires it to reproduce
    the Kerr-only result.

### `raman_fraction`

`raman_fraction = 0.18` is the Blow–Wood value for silica, in the envelope
definition above. Change it only with the convention in mind — a value taken from a
paper using a different split will not mean the same thing here.

### Which material

`raman = true` requires a material with an `:intermediate` condensed-phase Raman
model in `Luna.PhysData.raman_parameters`. For `:SiO2` that is the multimode
Hollenbeck–Cantrell response — a 13-mode sum. Any other kind throws an
`ArgumentError` naming the material and the kind it actually has, rather than
silently propagating something else.

### Implementations: `:batched` and `:frozen`

`raman_impl` selects how the convolution is evaluated. Both compute the same thing
and agree to rounding accuracy (~10⁻¹⁵ relative); the difference is entirely cost.

- `:batched` (default) — one pair of batched FFTs per right-hand-side
  evaluation, covering all transverse points at once.
- `:frozen` — the legacy per-column route, [`FrozenRamanPolarEnv`](@ref), kept
  for A/B comparison.

The distinction exists because a free-space grid is a hostile environment for a
response written for modal simulations. Luna's own `RamanPolarEnv` recomputes the
time-domain kernel and its FFT on *every* call — negligible at one call per step in
a modal run, dominant here, where the response runs once per transverse point. On a
1024² grid that is of order 10⁶ calls per Runge–Kutta stage, each re-evaluating the
13-mode Hollenbeck–Cantrell sum over the doubled time grid.

[`FrozenRamanPolarEnv`](@ref) was the first fix: it wraps Luna's response and
precomputes the frequency-domain kernel once at construction. Freezing is exact
here, because the density is constant and the `:intermediate` response ignores its
density argument entirely. `:batched` then goes further and removes the per-column
loop as well. Prefer the default; reach for `:frozen` only when you want to check
one against the other.

### Cost, and whether you need it

The delayed response roughly doubles the work in the nonlinear step and adds
buffers on the doubled time grid. For few-femtosecond DUV pulses in a thin
substrate the electronic response dominates and `raman = false` is the right
default. Turn it on when the pulse is long enough for the nuclear response to
follow it, or when you want to know how much of the trace it accounts for — in
which case run both and difference the traces.

That differencing has been done at production scale. The reference paper (see
[The paper](index.md#The-paper)) ran its 1 fs, 260 nm thickness series twice
through the full 3D instrument model, once with the instantaneous Kerr response
and once with the multimode Hollenbeck–Cantrell silica response at
``f_R = 0.18``, all else identical. The delayed response distorts the trace at
the ``10^{-4}``–``10^{-3}`` level over 4–40 µm of fused silica, concentrated in
the delay wings as a one-sided vibrational wake; retrieving the Raman-containing
traces with an instantaneous forward model shifts the retrieved duration by at
most 0.02 fs. At these thicknesses the delayed response is measurable in the
trace and negligible in the retrieved pulse.

## Not available in field mode

`raman = true` together with `field_mode = true` throws an `ArgumentError`. The
delayed response itself exists for real fields
(`Luna.Nonlinear.RamanPolarFieldBatched`), but ``f_R`` as used here is defined in
the *envelope* convention — including the 3/2 reconciling `Kerr_env`'s internal 3/4
with the Raman kernel's 1/2 — and that factor does not carry over to a
carrier-resolved field unexamined. Deriving it needs its own quasi-static
consistency test, matching the envelope one. Rather than guess, the combination is
refused.

## API

[`FrozenRamanPolarEnv`](@ref) is documented on the
[API Reference](interface.md) page; the `raman`, `raman_fraction` and `raman_impl`
keywords are documented under [`build_setup`](@ref).
