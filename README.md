# ModelPNPS

[![Docs][docs-badge]][docs-url]
[![Build Status][ci-badge]][ci-workflow]
[![Coverage][codecov-badge]][codecov-url]
[![Aqua QA][aqua-badge]][aqua-url]
[![JET][jet-badge]][jet-url]
[![Runic][runic-badge]][runic-url]

**High-fidelity forward modelling of PNPS pulse-characterisation traces.**

ModelPNPS generates synthetic pulse-characterisation traces directly from the
underlying experimental physics — full spatially-resolved nonlinear propagation
through the measurement medium, using
[Luna.jl](https://github.com/LupoLab/Luna.jl). Given an analytic input pulse and
an experimental geometry, it produces the trace a real apparatus would record.
These ground-truth traces are intended for **testing advanced retrieval
algorithms** against a known input pulse and for **developing new
characterisation techniques**. ModelPNPS does the forward modelling only — it
does not perform retrieval.

## Ambition

ModelPNPS aspires to be a *complete* PNPS (Parametrized Nonlinear Process
Spectrum) trace-modelling package: full-3D, high-fidelity numerical models of
the major pulse-characterisation experiments, in which the simulated trace
faithfully reflects the real measurement physics —

- **spatial effects** (finite beam size, mode shape, beam overlap and crossing
  geometry, diffraction, apertures and mask edges),
- **phase-matching** (wavelength- and angle-dependent nonlinear efficiency),
- **dispersion** (material dispersion and pulse reshaping during propagation),
- **walkoff** (spatial/temporal walkoff between interacting beams),
- **chromatic vignetting** of the signal by the collection optics, and
- **real χ⁽ⁿ⁾ nonlinear efficiency** (not an idealised instantaneous
  thin-medium response).

The aim is faithful numerical experiments for benchmarking and developing
retrieval algorithms, especially in regimes (broadband DUV/VUV, thick media,
strong phase mismatch) where the usual analytic forward models break down.

## Status & roadmap

The currently implemented process is **TG-FROG** (Transient-Grating FROG). The
package is organised around the Geib et al. (2019) PNPS taxonomy, in which every
technique is a **(nonlinear process × parametrization)** pair:

| Technique | Process | Parametrization | Status |
|-----------|---------|-----------------|--------|
| **TG-FROG** | transient grating (four-wave mixing) | delay | ✅ implemented |
| X-TG-FROG | TG + reference | delay | 🔜 planned |
| SD-FROG | self-diffraction | delay | 🟡 input geometry implemented |
| SHG-FROG | second-harmonic generation | delay | ⏳ pending Luna SHG/SFG support |
| THG-FROG | third-harmonic generation | delay | ⏳ planned |
| X-FROG (SHG/SD/THG) | cross-correlation | delay | ⏳ planned |
| SHG-d-scan | second-harmonic generation | glass insertion | ⏳ blocked on Luna |
| SD-d-scan | self-diffraction | glass insertion | 🔜 planned |
| Time-domain ptychography | SHG/THG/SD | position | ⏳ planned |

The **self-diffraction** beam layout is built and grid-sized:
`build_setup(; geometry = :sd, ...)` places two collinear holes instead of the
four-hole boxcar and puts the `2k_E − k_G` signal one slot further out on the same
axis. Windowed extraction works there as it does for TG; the `Iω_full` diagnostic
and `signal_quadrant_norm` are still boxcar-specific. See the
[self-diffraction geometry][sd-docs] section of the manual.

## Installation

ModelPNPS requires Julia 1.12 or later and the **`modal-fixed` branch of
[Luna.jl](https://github.com/jtravs/Luna.jl)**. This is not optional and not
GPU-specific: the package is written against Luna APIs — `Output.willsave`,
`resolve_arraytype`, `HostOutput`, `Luna.run`'s `twin_period` and `step_on`, the
batched Raman and field-mode responses — that are not yet in a registered Luna
release, and it will not even load without them.

Add Luna first, so the resolver has the branch before it looks in the registry:

```julia
import Pkg
Pkg.add(; url = "https://github.com/jtravs/Luna.jl", rev = "modal-fixed")
Pkg.add(; url = "https://github.com/LupoLab/ModelPNPS.jl")
```

Working *inside* a clone of this repository needs no such step — `Project.toml`
carries a `[sources]` entry pointing at the branch, so `Pkg.instantiate()` gets it
automatically. That entry is only honoured for the active project, which is why a
downstream environment has to add the branch itself.

For GPU runs, add CUDA.jl as well:

```julia
Pkg.add("CUDA")
```

This all goes away when the Luna changes reach a registered release.

## Quick start

```julia
using ModelPNPS
import Luna.Scans

# Hollow-fibre HE11 mode through a four-hole boxcar mask, χ³ in a thin SiO2 slab.
beam   = HE11Beam(125e-6, 5.0, 0.1)          # fibre radius, f_coll, f_foc
window = PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                            holediam=0.5e-3, zmask=0.1,
                            apod=:supergauss, apod_param=16)

setup = build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                      thickness=10e-6, material=:SiO2,
                      mask_diam=1.0e-3, mask_spacing=0.5e-3,
                      beam, window)

# Full TG-FROG delay scan, dispatched as one SLURM array job.
τ    = collect(range(-10e-15, 10e-15, 80))
exec = Scans.SlurmExec(@__FILE__, length(τ); memory="18G", arraymode=:batch)
run_scan(setup, τ; scan_name="my_tgfrog_run", exec)
```

Load the result for inspection:

```julia
nt = load_simulated_scan("my_tgfrog_run_collected.h5")
# nt.ω, nt.τ, nt.trace (Nω × Nτ), nt.Iω, nt.It, ...
```

Runnable, annotated scripts live in [`examples/`](examples/): two mask-scheme
runs (1 fs and 2 fs) and the Gaussian-beam comparison.

## Beyond the analytic Gaussian

The forward model is not restricted to a transform-limited Gaussian in an
instantaneous-Kerr medium:

- **Measured or simulated input pulses.** Inject a real complex spectrum with
  `input_pulse`, after conditioning it with `load_input_pulse`,
  `spectral_window!` and `center_pulse!` — the way to test a retrieval against
  the light a particular laser actually produces.
- **Delayed nuclear response.** `raman = true` adds the Raman contribution
  alongside the electronic Kerr effect, in a convention whose quasi-static limit
  reproduces the Kerr-only response exactly.
- **Field-resolved propagation.** `field_mode = true` propagates the real,
  carrier-resolved field instead of an envelope — no carrier split, no dropped
  third harmonic — which is how the envelope approximation itself gets tested at
  single-cycle durations.
- **Free intermediate thicknesses.** One `zsave` vector gets a whole ladder of
  substrate thicknesses out of a single scan.

## Runs on a GPU — about 160× faster

The whole propagation and extraction path runs on an NVIDIA GPU by passing one
keyword:

```julia
setup_args = (; λ0=260e-9, τfwhm=1e-15, energy=0.1e-6,
                thickness=40e-6, material=:SiO2,
                mask_diam=1.0e-3, mask_spacing=1.0e-3,
                beam, window,
                arraytype = :cuda, beamlets_on_host = true)

run_scan(setup_args, τ; scan_name="my_run", exec=Scans.LocalExec())
```

Measured on the same geometry:

| Hardware | Per delay point | 200-point scan |
|---|---|---|
| NVIDIA H200 | **42 s** | ~2.3 h |
| 2 CPU cores | **1.9 h** | ~16 days |

`memory_budget(setup_args)` reports what a configuration will need on the device
and the host before you launch it — an envelope run at `N = 768` is about 24 GiB
of device memory at 25–30 s per delay point.

GPU support is **experimental but working**, and currently needs the
`modal-fixed` branch of Luna.jl (see [Installation](#installation)). The device
code paths are covered in CI on `JLArrays`, so they are tested without a GPU. See
the [Running on a GPU](https://lupolab.github.io/ModelPNPS.jl/dev/gpu/) manual
page for the memory budget, the world-age rule that decides where `arraytype`
must be passed, and the practical setup.

## Designed for HPC

The CPU path remains fully supported. A full delay scan at realistic grid sizes
(`Nω` ≈ 4096, `N` ≈ 256–1024) is CPU-hours of work per delay point and is
intended to run on a SLURM cluster via `Luna.Scans.SlurmExec`, one task per
delay. The test suite stays laptop-fast: it exercises every primitive without the
propagation step (plus one tiny end-to-end smoke run) and completes in seconds —

```julia
import Pkg; Pkg.test("ModelPNPS")
```

For a faster development loop, select one isolated group with `GROUP=Core`,
`GROUP=Physics`, `GROUP=Quality`, or `GROUP=Docs` before running `Pkg.test()`.

## Documentation

Full documentation is at
[lupolab.github.io/ModelPNPS.jl](https://lupolab.github.io/ModelPNPS.jl/dev/),
built with [Documenter.jl](https://documenter.juliadocs.org/) from [`docs/`](docs/):

- **Trace Simulation** — the physical model, beam and window types, worked
  examples, grid sizing, and the self-diffraction geometry.
- **Input Pulses** — injecting a measured or separately simulated pulse.
- **Nonlinear Response** — the Kerr and Raman responses and their conventions.
- **Field-Resolved Mode** — propagating a real field instead of an envelope.
- **Running on a GPU** — the device path, memory budgeting and practical setup.
- **Accuracy and Validation** — weak-signal error control, apodisation cadence
  and A/B verification against reference data.
- **PNPS Framework** — the taxonomy and roadmap.

## Credits
ModelPNPS is jointly developed by John Travers
([@jtravs](https://github.com/jtravs)) and Chris Brahms
([@chrisbrahms](https://github.com/chrisbrahms)).

[sd-docs]: https://lupolab.github.io/ModelPNPS.jl/dev/trace_simulation/#Self-diffraction-geometry
[docs-badge]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://lupolab.github.io/ModelPNPS.jl/dev/
[ci-badge]: https://github.com/LupoLab/ModelPNPS.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-workflow]: https://github.com/LupoLab/ModelPNPS.jl/actions/workflows/CI.yml?query=branch%3Amain
[codecov-badge]: https://codecov.io/gh/LupoLab/ModelPNPS.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/LupoLab/ModelPNPS.jl
[aqua-badge]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
[jet-badge]: https://img.shields.io/badge/tested%20with-JET.jl-233f9a.svg
[jet-url]: https://github.com/aviatesk/JET.jl
[runic-badge]: https://img.shields.io/badge/code%20style-Runic-2a6099.svg
[runic-url]: https://github.com/fredrikekre/Runic.jl
