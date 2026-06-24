# The PNPS Framework

ModelPNPS is organised around the **PNPS** (Parametrized Nonlinear Process
Spectrum) formalism of Geib et al. (2019)[^geib], which provides a single
mathematical description for the whole family of self-referenced ultrashort-pulse
measurement techniques — FROG, d-scan, MIIPS, time-domain ptychography and their
many process variants.

[^geib]: N. C. Geib, M. Zilk, T. Pertsch, and F. Eilenberger,
    *"Common pulse retrieval algorithm: a fast and universal method to retrieve
    ultrashort pulses,"* Optica **6**, 495–505 (2019).

## The PNPS trace

A PNPS measurement records a two-dimensional **trace** that depends on the
pulse, the (angular) frequency ``\omega``, and a method-specific
**parametrization variable** ``\delta``:

```math
\tilde{T}(\delta, \omega; \tilde{E}) =
    \left| \mathcal{F}\!\left[ S_\delta[\tilde{E}](t) \right](\omega) \right|^2 .
```

Here ``\tilde{E}`` is the complex pulse, ``\mathcal{F}`` the Fourier transform,
and ``S_\delta`` the **parametrized nonlinear-process signal operator**. The
signal operator combines

- a **nonlinear process** ``N[\tilde{E}]`` that converts the pulse via a
  collinear nonlinearity, and
- a **parametrization filter** ``\mathcal{H}_\delta(\omega)`` that applies the
  scanned modification (a delay, a glass insertion, a phase-pattern shift, …).

A measurement technique is therefore specified by a **(process ×
parametrization)** pair, and its standard name follows the pattern
`[Process]-[Parametrization]` (e.g. SHG-FROG, SD-FROG, SHG-d-scan).

## Nonlinear processes

| Process | Signal ``N[\tilde{E}]`` | Notes |
|---------|-------------------------|-------|
| **SHG** | ``\tilde{E}^2``           | second-harmonic generation |
| **THG** | ``\tilde{E}^3``           | third-harmonic generation |
| **SD**  | ``\lvert\tilde{E}\rvert^2\tilde{E}`` | self-diffraction |
| **PG**  | ``\lvert\tilde{E}\rvert^2\tilde{E}`` | polarization gating |
| **TG**  | degenerate four-wave mixing | transient grating; two gate beams + test |
| **X-**  | ``\tilde{E}\,\tilde{E}_\text{ref}`` (cross term) | cross-correlation with a known reference |

## Parametrizations

| Parametrization | Variable ``\delta`` | Filter ``\mathcal{H}_\delta(\omega)`` | Technique family |
|-----------------|---------------------|----------------------------------------|------------------|
| **Delay**       | pulse delay ``\tau``    | ``e^{i(\omega+\Omega_0)\tau}``         | FROG |
| **Glass insertion** | glass thickness ``z`` | ``e^{i(\omega+\Omega_0)\,k(\omega)\,z}`` | d-scan |
| **Pattern shift** | phase-pattern shift | ``e^{\pm i(\omega+\Omega_0)\delta}``   | MIIPS |
| **Position**    | spatial / scan position | spatial filtering | (time-domain) ptychography |

## Roadmap

ModelPNPS currently implements **TG-FROG**. The table below places the planned
methods in the PNPS taxonomy and tracks their status. Several SHG/SFG-based
methods depend on second-order nonlinearity support arriving in Luna.

| Technique | Process | Parametrization | Status |
|-----------|---------|-----------------|--------|
| **TG-FROG** | TG (four-wave mixing) | delay | ✅ implemented |
| X-TG-FROG | TG + reference | delay | 🔜 planned |
| SD-FROG | SD | delay | 🔜 planned |
| SHG-FROG | SHG | delay | ⏳ pending Luna SHG/SFG support |
| THG-FROG | THG | delay | ⏳ planned |
| X-FROG (SHG/SD/THG) | cross-correlation | delay | ⏳ planned |
| SHG-d-scan | SHG | glass insertion | ⏳ pending Luna SHG/SFG support |
| SD-d-scan | SD | glass insertion | 🔜 planned |
| Time-domain ptychography | SHG/THG/SD | position | ⏳ planned |

Legend: ✅ available · 🔜 planned next · ⏳ planned (may depend on upstream
features).

The high-fidelity modelling goal is the same across every entry: capture the
full spatial, dispersive, phase-matching, walkoff and nonlinear-efficiency
physics of the real experiment, so that the simulated trace is a faithful
ground truth for retrieval-algorithm development and validation.
