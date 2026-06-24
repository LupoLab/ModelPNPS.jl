```@meta
CurrentModule = ModelPNPS
```

# ModelPNPS

**ModelPNPS** is a Julia package for *high-fidelity forward modelling* of PNPS
(Parametrized Nonlinear Process Spectrum) pulse-characterisation traces. Given
an analytic input pulse and an experimental geometry, it generates the trace a
real apparatus would record by full spatially-resolved nonlinear propagation
through the measurement medium, using
[Luna.jl](https://github.com/LupoLab/Luna.jl).

The goal is not a fast 1-D approximation but a **truth model**: a faithful
numerical experiment that captures the effects ordinarily neglected in the
analytic forward models used inside retrieval algorithms.

## Ambition

ModelPNPS aspires to be a *complete* PNPS trace-modelling package — full-3D,
high-fidelity numerical models of the major pulse-characterisation experiments,
built so that the simulated trace reflects the real physics of the measurement:

- **Spatial effects** — finite beam size, mode shape, beam overlap and crossing
  geometry, diffraction, apertures and mask edges.
- **Phase-matching** — the wavelength- and angle-dependent efficiency of the
  nonlinear process across the interaction volume.
- **Dispersion** — material dispersion of the nonlinear medium and the
  associated pulse reshaping during propagation.
- **Walkoff** — spatial and temporal walkoff between the interacting beams.
- **Chromatic vignetting** — the wavelength-dependent spatial filtering of the
  signal beam by the collection optics.
- **Real nonlinear efficiency** — the true χ⁽ⁿ⁾ conversion, not an idealised
  instantaneous-thin-medium response.

These ground-truth traces are intended for **testing advanced retrieval
algorithms** against a known input pulse, and for **developing new
characterisation techniques** where the analytic forward model is not yet known
or is known to be inadequate (e.g. broadband DUV/VUV pulses, thick media,
strongly phase-mismatched geometries).

## Current status

The currently implemented process is **TG-FROG** (Transient-Grating FROG): a
degenerate four-wave-mixing measurement in a thin solid substrate, modelled with
two beam schemes (hollow-fibre HE₁₁ mode through a four-hole boxcar mask, or a
simplified Gaussian-beam baseline) and a choice of signal-extraction windows.
See [Trace Simulation](trace_simulation.md) for the full description and worked
examples, and the [PNPS Framework](pnps.md) page for the broader taxonomy and
roadmap.

!!! note "Designed for HPC"
    A full delay scan at realistic grid sizes is CPU-hours of work and is
    intended to run on a SLURM cluster. The unit tests stay laptop-fast by
    exercising every primitive without the propagation step, plus one tiny
    end-to-end smoke run.

## Contents

```@contents
Pages = [
    "pnps.md",
    "trace_simulation.md",
    "interface.md",
]
Depth = 2
```

## API Index

```@index
```
