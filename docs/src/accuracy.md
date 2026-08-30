```@meta
CurrentModule = ModelPNPS
```

# Accuracy and Validation

A TG-FROG trace is a **weak** signal sitting next to three strong pumps. Most of
this page follows from that fact: the default adaptive stepper does not
automatically control the quantity you care about, the apodisation cadence
interacts with the step count, and a new code path is trusted by differencing it
against reference data. For the approximations the simulation itself makes —
representation, boundary, discretisation, interfaces — see the reference paper
([The paper](index.md#The-paper)), which enumerates and bounds each one for the
production instrument model.

## The weak-signal problem

Luna's default error norm, `RK45.weaknorm`, measures the step error relative to the
norm of the **whole** field. In this geometry the three pump beamlets dominate that
norm, and the four-wave-mixing signal is orders of magnitude weaker. The stepper's
error budget is concentrated on the fastest-evolving components — the signal — so
what it actually permits is a per-step signal error of order

```math
\text{rtol} \times \frac{\|\text{pump}\|}{\|\text{signal}\|}
```

*relative to the signal*. Measured consequence: at `rtol = 1e-6` the collected
signal carries solver errors of 0.1–1 % mid-slab, growing to about 10 % at 40 µm.

Brute force works — `rtol = 1e-8` fixes it — but costs roughly 4× the step count.

## The region-relative norm

[`signal_quadrant_norm`](@ref) is the cheaper answer. It measures relative error
**separately** in the signal's k-space quadrant (`kx < 0, ky < 0`, the same quadrant
`Iω_full` integrates) and in the rest of the field, and returns the larger of the
two. `rtol` then controls the signal's *own* relative error directly, recovering
weak-signal accuracy at close to the default-`rtol` step count.

```julia
setup = build_setup(; ...)
out = simulate_delay_point(setup, τ; rtol = 1e-6, norm = signal_quadrant_norm(setup))
```

While the signal quadrant is still nearly empty, its error would be a 0/0 relative
quantity and would throttle the early steps. It is therefore measured against a floor
of `floor_rel × ‖rest‖` (never below `atol`); once the signal exceeds that fraction
of the pump field, relative control takes over. `floor_rel` defaults to `1e-6`.

### Inside a scan

The norm needs the setup to exist, and in a scan the setup is built lazily on the
compute node. [`run_scan`](@ref) therefore takes `norm_builder`, a callable
`setup -> norm`, instead of a ready-made `norm`:

```julia
run_scan(setup_args, τ;
         scan_name = "my_run", exec = Scans.LocalExec(),
         rtol = 1e-7, norm_builder = signal_quadrant_norm)
```

The norm that was actually used is recorded in the output file's `/grid/error_norm`
as provenance, alongside `rtol` and `max_dz`.

!!! note "Validate a new `(rtol, floor_rel)` before production"
    The pass criterion used here is: every z-slice of `Iω_win` within 10⁻³ relative
    of an `rtol = 1e-8` run. Do that once for a new geometry rather than assuming a
    setting transfers.

## Apodisation cadence: `twin_period`

Luna applies spectral and temporal windows in place during propagation, to absorb
what reaches the edges of the computational domain. `twin_period` sets how many
accepted steps pass between applications.

- `twin_period = 1` (the default) applies them after **every** accepted step. The
  damping then scales with the step count, which makes the scheme non-convergent in
  `rtol`: halve the tolerance, take more steps, get more absorption.
- A large value applies them only immediately before a save (Luna always windows
  before a save). Combined with `step_on` — which [`simulate_delay_point`](@ref)
  passes automatically, so the stepper lands exactly on each `zsave` position — the
  windows are then applied at *identical* positions for any `rtol`, and a
  tolerance-convergence study means something.

The production campaigns use the second, passing a number larger than any achievable
step count:

```julia
const TWIN_SAVES_ONLY = 1_000_000_000
run_scan(setup_args, τ; ..., twin_period = TWIN_SAVES_ONLY)
```

The value is recorded in `/grid/twin_period`. Note that this changes results at the
apodisation-leakage level, so it is a choice to make once per campaign and record,
not to vary between runs you intend to compare.

## The exact collection efficiency

Every run also produces `Iω_full`: `|E|²` integrated over the signal's k-space
quadrant only. The propagated field holds the three strong pump beamlets plus the
weak signal in the fourth corner, so integrating over *all* of k-space would be
dominated by the pumps; restricting to the signal quadrant captures the whole signal
lobe without aperture vignetting while excluding them.

`Iω_win ./ Iω_full` is therefore the **exact** per-(ω, τ) collection and chromatic
vignetting efficiency of the signal aperture — so a trace can be corrected for
vignetting exactly, rather than through a power-law approximation. This assumes the
boxcar beams are well separated, so that pump tails leaking into the signal quadrant
are negligible against the signal.

To inspect it, load the same file with `window_key = "Iω_full"`:

```julia
nt_win  = load_simulated_scan("my_run_collected.h5")
nt_full = load_simulated_scan("my_run_collected.h5"; window_key = "Iω_full")
η = vec(sum(nt_win.trace, dims = 2) ./ sum(nt_full.trace, dims = 2))
```

## A/B validation against reference data

[`verify_against_collected`](@ref) is the harness for the question "did this change
alter the physics?". It recomputes selected delay points of an existing scan and
compares them against the stored file, dataset by dataset.

```julia
results = verify_against_collected(setup_args, "reference_collected.h5", 1:5;
                                   zsave = zvec, rtol = 1e-7, max_dz = 2e-6)
```

For each index the delay is read from `/scanvariables/τ` in the reference file, the
point is recomputed with the given solver settings — pass the **same** settings the
reference used, unless you are deliberately testing a change — and every trace
dataset present in the file is compared. Reference points that are still all-zero,
because a scan is running, are reported as `NaN` and skipped.

### Reading the result

One `Dict` per point, with the delay, wall time, `Sys.maxrss()` in GiB, and for each
dataset four numbers:

| Key | Meaning |
|---|---|
| `ks` | max absolute difference ÷ **this point's** reference peak |
| `ks*"\|relscan"` | max absolute difference ÷ the **scan-wide** reference peak |
| `ks*"\|refpeak"` | this point's reference peak |
| `ks*"\|scanpeak"` | the scan-wide reference peak |

Both normalisations matter, and they answer different questions. A delay-scan wing
carries a signal orders of magnitude below the τ ≈ 0 signal, so a difference that is
irrelevant in the assembled trace can still be a large fraction of that point's own
peak. `relscan` is what a FROG retrieval sees; the own-peak number is the stricter
statement about the code path.

### Comparing across grid sizes

To test a grid change — say `N = 640` against an `N = 1024` reference — pass the
changed `N` inside `setup_args`. Differences then reflect the grid, not the code.

The k-space-integrated spectra (`Iω_win`, `Iω_full`, …) are in FFT-bin units that
scale as `N⁴` at fixed `R`, by Parseval over the transverse FFT, so the recomputed
values are rescaled to the reference grid's units before comparison. The re-imaged
(real-space pixel) spectra are `N`-invariant and are not rescaled.

!!! note "Match the FFT configuration"
    Strict comparisons need the same `fftw_threads` and `fftw_mode` as the reference
    scan: FFT algorithm choice affects round-off. Julia-level threading
    (`JULIA_NUM_THREADS`) does **not** affect results and can be used freely to speed
    verification up.

## Free intermediate thicknesses

Because the field at an intermediate `z` equals a dedicated thickness-`z` run, every
shorter substrate thickness comes free from one full-thickness run. Pass a vector of
explicit thicknesses as `zsave` and the trace datasets become `(Nω, nz, Nτ)`, with
the realised positions stored once in `/grid/zsave`.

```julia
zsave = [1, 2, 4, 8, 12, 20, 40] .* 1e-6
run_scan(setup_args, τ; ..., zsave = zsave)
```

This is the cheapest convergence study available: thickness is the parameter the
trace is most sensitive to, and this gets a whole ladder of it for the price of one
scan. Peak memory scales with the number of z points, since the in-memory 4-D field
is held per slice.

## What the test suite guarantees

The test suite is deliberately laptop-fast and does not attempt to validate the
physics of a production run. It covers:

- every primitive in isolation — mask construction, window construction, k-space
  field builders, tilts, delays, extraction;
- `simulate_delay_point(...; skip_propagation = true)`, which exercises the whole
  per-delay path except `Luna.run` itself;
- one tiny end-to-end run on a 32×32 grid, which is the only place the integration
  boundary with Luna is proved;
- the full device path on `JLArrays`, so the GPU code paths are covered without a
  GPU;
- type stability of the public API through `@inferred`, and package, import and
  error-analysis hygiene through Aqua, ExplicitImports and JET.

Everything above that level — solver tolerances, apodisation cadence, grid
convergence — is what this page's tools are for.

## API

[`signal_quadrant_norm`](@ref) and [`verify_against_collected`](@ref) are documented
on the [API Reference](interface.md) page; `rtol`, `max_dz`, `twin_period`,
`norm` and `zsave` under [`simulate_delay_point`](@ref) and [`run_scan`](@ref).
