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
| SD-FROG | self-diffraction | delay | 🔜 planned |
| SHG-FROG | second-harmonic generation | delay | ⏳ pending Luna SHG/SFG support |
| THG-FROG | third-harmonic generation | delay | ⏳ planned |
| X-FROG (SHG/SD/THG) | cross-correlation | delay | ⏳ planned |
| SHG-d-scan | second-harmonic generation | glass insertion | ⏳ blocked on Luna |
| SD-d-scan | self-diffraction | glass insertion | 🔜 planned |
| Time-domain ptychography | SHG/THG/SD | position | ⏳ planned |

## Installation

ModelPNPS depends on [Luna.jl](https://github.com/LupoLab/Luna.jl) (registered
in the General registry) and requires Julia 1.12 or later. From the Julia REPL:

```julia
import Pkg
Pkg.add(; url = "https://github.com/LupoLab/ModelPNPS.jl")
```

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

## Designed for HPC

A full delay scan at realistic grid sizes (`Nω` ≈ 4096, `N` ≈ 256–1024) is
CPU-hours of work and is intended to run on a SLURM cluster via
`Luna.Scans.SlurmExec`. The test suite stays laptop-fast: it exercises every
primitive without the propagation step (plus one tiny end-to-end smoke run) and
completes in seconds —

```julia
import Pkg; Pkg.test("ModelPNPS")
```

For a faster development loop, select one isolated group with `GROUP=Core`,
`GROUP=Physics`, `GROUP=Quality`, or `GROUP=Docs` before running `Pkg.test()`.

## Documentation

Full documentation — physical model, beam and window types, worked examples,
grid sizing, and the PNPS framework — is built with
[Documenter.jl](https://documenter.juliadocs.org/) under [`docs/`](docs/).

## Credits
ModelPNPS is jointly developed by John Travers
([@jtravs](https://github.com/jtravs)) and Chris Brahms
([@chrisbrahms](https://github.com/chrisbrahms)).

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
