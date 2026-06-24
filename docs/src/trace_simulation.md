# Trace Simulation

ModelPNPS generates synthetic TG-FROG (Transient Grating FROG) spectrograms by
full spatially-resolved nonlinear propagation through a thin solid medium. Given
an analytic input pulse and an experimental geometry, it produces the
``I(\omega, \tau)`` trace a real apparatus would record — the ground truth for
benchmarking and developing retrieval algorithms.

The propagation is performed by
[Luna.jl](https://github.com/LupoLab/Luna.jl) using its angular-spectrum
free-space propagator and instantaneous Kerr χ⁽³⁾ response. A typical run
requires a SLURM cluster — full delay scans with realistic grid sizes
(`Nω` ≈ 4096, `N` ≈ 256–1024) take tens of minutes per delay point.

## Physical model

Three input beams (two gates `g1`, `g2` and a delayed test pulse `t`) are
crossed inside a thin nonlinear substrate (e.g. UV fused silica). Their
interference produces a transient refractive-index grating; the test pulse
diffracts off this grating into the fourth corner of the boxcar via degenerate
four-wave mixing,

```math
    \mathbf{k}_\text{signal} = \mathbf{k}_{g2} - \mathbf{k}_{g1} + \mathbf{k}_t.
```

The boxcar layout (looking along +z):

```
    test (-x, +y)    | gate 1 (+x, +y)
   --------------------------------------
    signal (-x, -y)  | gate 2 (+x, -y)
```

Scanning the test-pulse delay τ and spectrally resolving the diffracted signal
yields a 2-D `I(ω, τ)` spectrogram — the TG-FROG trace.

The simulation pipeline is:

1. Build a temporal/spectral grid (Luna `EnvGrid`) and a spatial grid
   (Luna `FreeGrid`).
2. Set up material dispersion (Sellmeier `n(λ)`) and Kerr χ⁽³⁾ via Luna.
3. Construct the three input beamlets in k-space at the substrate.
4. For each delay τᵢ:
   1. Apply `exp(-iωτᵢ)` to the test beam.
   2. Coherently superpose all three beamlets.
   3. Propagate through the substrate via `Luna.run` (adaptive RK4(5)
      split-step).
   4. Apply the signal-extraction window in k-space.
   5. Extract two spectral diagnostics: a re-imaged on-axis spectrum and a
      fully-integrated spectrum.
5. Save all outputs to a single HDF5 file via `Luna.Output.scansave`.

## Two beam models

Two [`AbstractInputBeam`](@ref) subtypes are provided:

- [`HE11Beam`](@ref) — the master experimental model. The HE₁₁ mode of a hollow
  capillary fibre is collimated by a long lens, clipped by a four-hole apodised
  mask in the collimated beam, then focused into the substrate. Each hole
  selects one of the four boxcar arms. Chromatic vignetting is captured exactly
  because the mask plane ↔ k-space mapping is wavelength-dependent.

- [`GaussianBeam`](@ref) — a simplified Gaussian-beam model that places three
  Gaussian beams directly at the correct k-space angles (no fibre mode, no
  physical mask). Useful as a sanity-check baseline.

## Three signal-extraction window types

- [`PhysicalMaskWindow`](@ref) — the master experimental signal extraction: a
  frequency-dependent apodised hole in the mask plane. Apodisation choices:
  `:hard`, `:supergauss` (default order 16), `:tanh`. Chromatic vignetting is
  captured exactly.

- [`PlanckWindow`](@ref) — a smooth, frequency-*independent* radial Planck taper
  in k-space. No chromatic vignetting; baseline for the Gaussian model.

- [`PlanckOmegaWindow`](@ref) — a smooth, frequency-*dependent* Planck taper that
  mimics the chromatic vignetting of the physical mask while keeping the
  smooth-edge advantage. Used to isolate the two effects (smooth-edge vs
  ω-scaling) within the Gaussian model.

When [`build_setup`](@ref) is given a *vector* of windows, every per-delay
output is computed for each window in turn and saved with a suffix. The standard
two-window pattern `[PlanckWindow, PlanckOmegaWindow]` produces output keys
`Iω_win`, `Iω_win_reimaged`, `Iω_win_ωdep`, `Iω_win_ωdep_reimaged`.

## Worked example: mask scheme

```julia
using ModelPNPS
import Luna.Scans

beam   = HE11Beam(125e-6, 5.0, 0.1)        # fibre radius, f_coll, f_foc
window = PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                            holediam=0.5e-3, zmask=0.1,
                            apod=:supergauss, apod_param=16)

setup = build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                      thickness=10e-6, material=:SiO2,
                      mask_diam=1.0e-3, mask_spacing=0.5e-3,
                      beam, window)

τ    = collect(range(-10e-15, 10e-15, 80))
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="18G", arraymode=:batch)
run_scan(setup, τ; scan_name="my_mask_run", exec)
```

See [`examples/tgfrog_simulation_mask_2fs.jl`](https://github.com/jtravs/ModelPNPS.jl/blob/main/examples/tgfrog_simulation_mask_2fs.jl)
(and the 1 fs variant) for the full annotated scripts.

## Worked example: Gaussian-beam scheme

```julia
using ModelPNPS
import Luna.Scans

f_foc, mask_diam, mask_spacing = 0.1, 1.0e-3, 0.5e-3
λ0       = 260e-9
w0       = λ0 * f_foc / (π * mask_diam/2)
d_hole   = mask_spacing/2 + mask_diam/2
Δk       = 2π/λ0 * sin(d_hole / f_foc)

beam    = GaussianBeam(w0, f_foc)
windows = [PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/w0, pad=1.25),
           PlanckOmegaWindow(xc=-d_hole, yc=-d_hole,
                             holediam=mask_diam/2, f_foc=f_foc, pad=1.25)]

setup = build_setup(; λ0, τfwhm=2e-15, energy=0.2e-6,
                      thickness=10e-6, material=:SiO2,
                      mask_diam, mask_spacing,
                      beam, window=windows)

τ    = collect(range(-10e-15, 10e-15, 80))
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="18G", arraymode=:batch)
run_scan(setup, τ; scan_name="my_gaussian_run", exec)
```

See [`examples/tgfrog_simulation_gaussian.jl`](https://github.com/jtravs/ModelPNPS.jl/blob/main/examples/tgfrog_simulation_gaussian.jl).

## Spatial grid sizing

[`optimal_spatial_grid`](@ref) computes a `(R, N)` pair such that the spatial
grid

1. **contains** at least `n_airy` Airy patterns from a single mask hole at the
   longest wavelength `λmax` (real-space containment),
2. **resolves** the Airy pattern at the shortest wavelength `λmin` with at least
   `pts_per_lobe` real-space samples across the central lobe,
3. has k-space half-extent **exceeding** the largest FWM nonlinear k-vector at
   `λmin`, with a `safety` headroom factor.

`N` is rounded up to the next power of 2 for FFT efficiency. For typical DUV
parameters (`f_foc=0.1 m`, `mask_diam=1 mm`, `mask_spacing=0.5 mm`,
`λmin=160 nm`, `λmax=500 nm`) this produces `R ≈ 3 mm` and `N` in the range
256–1024.

The default `safety=1.5` is conservative; reducing it produces a smaller grid
(and faster simulation) at the cost of potential FWM aliasing.

## Loading scan output

[`run_scan`](@ref) writes one HDF5 file per delay scan via
`Luna.Output.scansave`. [`load_simulated_scan`](@ref) reads that raw file,
extracts the chosen signal window and propagation z-slice, fftshifts the
ω-dependent arrays into natural (centred) order, and returns the trace and
reference spectra as a `NamedTuple` ready for inspection, plotting, or custom
post-processing:

```julia
using ModelPNPS

nt = load_simulated_scan("my_mask_run_collected.h5";
                         window_key="Iω_win", z_index=:end)
# nt.ω, nt.τ, nt.trace (Nω × Nτ), nt.Iω, nt.It, ...
```

## Diagnostics: the retrievable pulse and the efficiency curve

Two diagnostics are saved on every run to make the simulated trace a clean
retrieval benchmark.

### The retrievable pulse — `Iω_beamlet` / `It_beamlet`

`Iω_beamlet` is the spatially-integrated spectrum of one input beamlet *after*
the mask — i.e. the input pulse as chromatically vignetted by a single mask
hole. `It_beamlet` is its time-domain intensity (the mask is a real amplitude
filter, so the beamlet inherits the input pulse's spectral phase). This is the
"ground-truth" pulse a retrieval should recover, and is the right thing to plot
against a reconstructed `|E(t)|²` and `|E(ω)|²` — not the un-vignetted input
`It`/`Iω`. Both are saved under `/grid` and returned by
[`load_simulated_scan`](@ref); `Ito_beamlet` is the 8× oversampled version
sharing the `To` grid.

### The exact collection efficiency — `Iω_full`

The TG-FROG frequency marginal ``M(\omega) = \sum_\tau I(\omega,\tau)`` does not
match `Iω_beamlet`, even after the nonlinear ``\omega^n`` correction. It is
important to separate the two *very different* reasons why, because only one of
them is an efficiency that should be divided out:

1. **Nonlinear generation scaling** (an efficiency) — the χ⁽³⁾ polarization →
   radiated-field conversion carries explicit ω factors; the
   ``(\omega+\omega_0)^n`` power law is only an *approximation* to it.

2. **The marginal is a gate convolution, not the spectrum** (*not* an
   efficiency). Even for a perfect, unvignetted, single pulse, a TG-FROG
   marginal is intrinsically

   ```math
       M(\omega) = |\tilde{E}(\omega)|^2 \;\circledast\; |\tilde{g}(\omega)|^2,
       \qquad g(t) = |E(t)|^2,
   ```

   the pulse spectrum convolved with the transient-grating gate spectrum (a
   baseband function), which smears and broadens it. All three beamlets are the
   *same* vignetted pulse (the three holes are symmetric and the HE₁₁ mode is
   radial), so this is **fully determined by the single pulse `Iω_beamlet`** — it
   is real FROG physics that the retrieval forward model reproduces, and must
   **not** be divided out. Vignetting only sets *which* pulse (`Iω_beamlet`)
   enters the convolution.

3. **Collection vignetting** (an efficiency) — the signal beam is collected
   through the (chromatic) signal window, multiplying the marginal by a per-ω
   collection efficiency. Under good imaging (output mask conjugate to the input
   mask) this is small; `Iω_full` lets you confirm exactly how small.

For the Gaussian-beam scheme the efficiency factors (1) and (3) are mild (no
input mask; a broad, smooth Planck window), which is why its marginal lines up
with the spectrum after the ``\omega^2`` correction once the gate convolution is
accounted for. For the physical-mask scheme factor (3) and the gate convolution
of the vignetted spectrum are both significant.

To remove factor 3 **exactly** — with no power-law approximation — every run
also saves `Iω_full`: the signal beam collected *in full*. The propagated field
holds the three strong pump beamlets (at the g1/g2/test boxcar corners) plus the
weak FWM signal at the fourth corner, so integrating over *all* of k-space would
just be dominated by the pumps. Instead `Iω_full` integrates `|E|²` over the
signal's k-space quadrant only (the signal sits alone at `kx<0, ky<0` while the
three pumps occupy the other quadrants), capturing the whole signal lobe with no
aperture vignetting and excluding the pumps. The ratio

```math
    \eta_\text{collect}(\omega, \tau) = \frac{I_\text{win}(\omega,\tau)}
                                              {I_\text{full}(\omega,\tau)}
```

is then the exact per-(ω, τ) collection / chromatic-vignetting efficiency of the
signal aperture. Divide it out (or use `Iω_full` directly) to obtain a trace free
of collection vignetting; the per-ω efficiency curve is the delay-marginal ratio
``\sum_\tau I_\text{win} / \sum_\tau I_\text{full}``. (This assumes the boxcar
beams are well separated, so pump tails leaking into the signal quadrant are
negligible vs. the signal — true for any working TG-FROG geometry.) Load it
with:

```julia
nt_win  = load_simulated_scan("run_collected.h5"; window_key="Iω_win")
nt_full = load_simulated_scan("run_collected.h5"; window_key="Iω_full")
η_collect = vec(sum(nt_win.trace, dims=2) ./ sum(nt_full.trace, dims=2))
```

Together, `Iω_full` removes collection vignetting (factor 3) exactly, while
`Iω_beamlet`/`It_beamlet` give the actual vignetted pulse under test — so once
the nonlinear ``\omega^n`` scaling (factor 1) is divided out, the only remaining
"offset" is the gate convolution (factor 2), which is genuine FROG physics the
retrieval forward model reproduces rather than an artifact to correct.

## Computational cost

Per delay point, the propagation cost is dominated by the 3-D FFTs in the
split-step solver. Empirically, with `Nω ≈ 4096`, `N ≈ 256–1024`, substrate
thickness 10 µm, and Luna's default tolerances, one delay point takes a few tens
of seconds to a few minutes on a single SLURM core. A full 80-delay scan with
one task per delay completes in well under an hour given enough parallelism. Use
`Scans.SlurmExec` with `arraymode=:batch` to dispatch the full scan as one SLURM
array job.

## Testability

Every primitive ([`build_he11_kspace`](@ref), [`build_gaussian_kspace`](@ref),
[`apply_tilt`](@ref), [`apply_delay`](@ref), [`makemask`](@ref),
[`build_window`](@ref), [`extract_signal_spectra`](@ref)) is independently
unit-testable. The test file `test/tracesimulation_test.jl` exercises each in
isolation, plus a `simulate_delay_point(...; skip_propagation=true)` mode that
bypasses `Luna.run` to test signal extraction without paying the propagation
cost. A single tiny end-to-end smoke test runs `Luna.run` on a 32×32 grid in a
few seconds — this is the only place the test suite proves the integration
boundary works. The full pipeline is validated by the example scripts on SLURM,
not by the unit tests.

## API reference

Full docstrings for every type and function named above — grouped as beam types,
signal-window types, setup/simulation/scan, primitives and scan-loading — are
collected on the [API Reference](interface.md) page.
