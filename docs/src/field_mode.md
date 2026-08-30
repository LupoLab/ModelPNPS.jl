```@meta
CurrentModule = ModelPNPS
```

# Field-Resolved Mode

By default ModelPNPS propagates a complex **envelope**: an analytic field about a
carrier, on Luna's `Grid.EnvGrid`. That is the standard choice and the right one for
almost every run. `build_setup(; field_mode = true)` instead propagates the real,
carrier-resolved field on a `Grid.RealGrid`.

This is a **diagnostic**, not a production default. It costs roughly twice the memory
and three times the time per delay point, and it exists to answer one question:
how much of a residual belongs to the envelope approximation rather than to the
retrieval.

## Why it exists

At 260 nm a 1 fs pulse is **1.15 optical cycles**. The envelope approximation is
marginal by construction at that duration, and the production envelope grid's
relative-frequency window has its red edge at exactly DC — so there is no headroom
left to argue with. When a retrieval leaves an unexplained residual at the 10⁻³
level, "the forward model is an envelope model" is a live hypothesis, and the only
way to test it is to run the same geometry in both representations and difference
the traces.

That test has been run. The reference paper (see [The paper](index.md#The-paper))
repeated 16 delay points of its 1 fs production scan in field-resolved mode and
measured an envelope-versus-field trace difference of ``1.2\times10^{-5}`` at
the 9.5 µm substrate — a factor of about 400 below the model residual under
investigation there, which exonerated the envelope representation. The mode
exists to make exactly this kind of statement possible.

Three things differ on a real grid:

- there is **no carrier/envelope split** — the field is the field;
- the **third-harmonic term is not dropped**, nor are the counter-rotating terms;
- there is **no negative-frequency wrap**, because the spectrum is the monotonic
  `rfft` half-spectrum of a real field rather than an FFT-ordered window about a
  carrier.

Everything else is unchanged. The masks, windows, delay ramp, k-space builders and
the whole extraction path are written against the **absolute** frequency axis
`grid.ω`, which both grid types carry, so none of them branch on the representation.

## Using it

```julia
setup = build_setup(;
    λ0 = 260e-9, τfwhm = 1e-15, energy = 0.1e-6,
    thickness = 40e-6, material = :SiO2,
    mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
    λlims = (143e-9, 600e-9),
    beam, window,
    field_mode = true,           # Grid.RealGrid
    response = :nothg,           # the default; see below
    ffac = 6,                    # the default; see below
)
```

Two further keywords are only meaningful when `field_mode = true`.

### `response` — which nonlinearity

- `:nothg` (also `:auto`, the default) — the polarisation
  ``\tfrac{3}{4}\varepsilon_0\chi^{(3)}|E_a|^2 E``, via
  `Luna.Nonlinear.Kerr_field_nothg`.
- `:thg` — the polarisation ``\varepsilon_0\chi^{(3)}E^3``, via
  `Luna.Nonlinear.Kerr_field`.

`:nothg` has the **same physics content** as the envelope `Kerr_env`, evaluated on a
carrier-resolved field. That is precisely what makes it the default: a
comparison made with it isolates *representation* error with nothing else changed.
`:thg` adds the third-harmonic and counter-rotating terms, so the difference between
the two runs is exactly what the envelope omits.

!!! note "What `:thg` does and does not propagate on a UV window"
    The third harmonic is generated on the fine grid and then discarded by the crop
    back to the propagated grid whenever it falls outside the window. At
    `λlims = (143, 600) nm` the 3ω band of a 2 fs 260 nm pulse starts above `ωmax`,
    so none of it survives. What is retained are the **within-band counter-rotating
    terms**, and those are the real difference from the envelope. To propagate the
    third harmonic itself, extend `λlims` down to about `λ0/3`.

### `ffac` — the nonlinear-grid sampling factor

`ffac` is forwarded to `Grid.RealGrid` and sets how finely the nonlinear grid is
sampled relative to the propagated one.

- `ffac = 6` (default) sizes the fine grid for ``E^3``.
- `ffac = 4` is enough for `:nothg` alone, and typically removes the oversampling
  entirely — halving both memory and per-step cost. Luna's *ffac convergence for the
  no-THG response* testset measures 4 × 10⁻⁸ agreement, unchanged from `z = 0` to
  `z = end`.

!!! warning "`ffac` changes the grid"
    It changes `δω`, the realised time window and in general `Nω`, so an `ffac = 4`
    trace is **not** bin-for-bit comparable with an `ffac = 6` one, nor with an
    envelope file. Use it for a self-contained scan, not for the comparison the mode
    exists for — and only with a convergence check against the default.

## Cost

Field mode is much heavier than the envelope, and in a way the old "ten times the
field size" rule of thumb does not capture: the state is twice as long in ω, the
nonlinear evaluation runs on a grid twice as long again in time, and the no-THG
response carries a complex analytic-signal buffer on that grid.

Measured device budgets at the production shape (`N = 768`, 40 µm substrate):

| Configuration | `Nω` | Device | Host peak |
|---|---|---|---|
| envelope | 256 | ~24 GiB | ~12 GiB |
| field `:nothg`, `ffac = 6` | 513 | ~92 GiB | ~25 GiB |
| field `:thg`, `ffac = 6` | 513 | ~65 GiB | |
| field `:nothg`, `ffac = 4` | 513 | ~65 GiB | |

So `:nothg` at `ffac = 6` is H200-only; `:thg` and `:nothg`-at-`ffac = 4` fit an
80 GB H100; an A40 fits none of them. At `N = 1024`, `:nothg` needs 164 GiB and fits
nothing currently available.

In time, field mode costs roughly 3× the envelope per delay point at matched step
counts (measured 3.0× at `N = 64` and 3.3× at `N = 128`).

Call [`memory_budget`](@ref) before launching — it accounts for the response's
analytic-signal buffer, which is allocated lazily on the *first* right-hand-side
evaluation rather than at setup, and so is invisible to a measurement taken after
`build_setup` alone. See [Running on a GPU](gpu.md#Budgeting-device-memory).

## Raman is refused

`raman = true` with `field_mode = true` throws an `ArgumentError`. The reason is
given in [Nonlinear Response](nonlinear_response.md#Not-available-in-field-mode):
the nuclear fraction ``f_R`` is defined in the envelope convention, and that
definition does not carry over to a carrier-resolved field without its own
derivation and consistency test.

## Reading field-mode output

A field-mode run writes a `field_mode` marker into the file's `/grid` group, and
`/grid/ω` is then the **monotonic `rfft` half-spectrum** rather than the
FFT-ordered relative-frequency axis an envelope run writes.

[`load_simulated_scan`](@ref) reads the marker and does the right thing: it
`fftshift`s an envelope file into natural order and leaves a field-mode file alone.
A file written before field mode existed carries no marker and loads exactly as it
always did.

The temporal diagnostics stay comparable across the two modes. `It` and `Ito` in
the metadata are the **envelope** intensity ``|A(t)|^2`` in both cases — recovered
through the analytic signal on a real grid — so a consumer of a field-mode file
sees the same physical quantity as in every envelope file, rather than a
carrier-modulated one it would have to demodulate. Luna builds a real-grid pulse as
``\sqrt{I}\cos(\omega_0 t)`` and an envelope-grid pulse as
``\sqrt{I}\exp(i\Delta\omega t)``, so ``|A|^2 = I`` either way.

## API

The `field_mode`, `response` and `ffac` keywords are documented under
[`build_setup`](@ref) on the [API Reference](interface.md) page.
