# Changelog

All notable changes to ModelPNPS are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Manual pages for the features that had no documentation outside their docstrings:
  data-driven input pulses, the Kerr and Raman responses, field-resolved mode,
  running on a GPU, and accuracy/validation. The self-diffraction geometry is now
  described in the trace-simulation chapter.
- README badges for CI, documentation, coverage, Aqua, JET and Runic, and the CI
  jobs behind them: coverage upload to Codecov, a standalone `Quality` group job,
  and a Runic formatting check.
- `codecov.yml`, reporting coverage informationally rather than as a gate, since a
  suite that deliberately skips the propagation step cannot meet a line-coverage
  target that would mean anything.
- An explicit `slug` on the Codecov upload, which an organisation-wide upload token
  requires and a repository token ignores. This does not by itself enable coverage:
  the repository must also be activated on Codecov, without which uploads are
  rejected as "Repository not found".
- A `[sources]` entry resolving Luna from the `modal-fixed` branch. The package is
  written against Luna APIs that are not in a registered release and does not load
  without them, so CI had never been able to pass; it now resolves the branch the
  same way a clone of this repository does.
- `TODO.md`, recording deferred work — the `examples/` cleanup, the absence of
  executable manual examples and doctests, the boxcar-specific SD diagnostics, and
  the CI services still to be enabled.
- `build_setup` and `optimal_spatial_grid` now reject an unknown `geometry`, and
  `build_setup` rejects `geometry = :sd` for a beam model whose builder only ever
  places three beams. Both cases previously produced a boxcar run with no diagnostic.

### Fixed

- The documentation build, which died at `import ModelPNPS` on a cold runner.
  Luna loads PyPlot unconditionally, so the build needs matplotlib; PyCall was
  binding to the runner's system python, which has none. Both workflows now set
  `PYTHON = ""` so PyCall uses its own Conda python and PyPlot installs matplotlib
  itself. CI had been passing only because its depot cache happened to carry a
  working PyCall.
- Installation instructions. They presented the `modal-fixed` Luna branch as a GPU
  extra, but it is required for the package to load at all, and Luna has to be added
  before ModelPNPS so the resolver sees the branch first.

### Changed

- Raised the minimum supported Julia version to 1.12 and replaced the unbounded Luna
  compatibility entry with the tested 0.6.2 series.
- Applied Runic formatting across the source, tests, and documentation build script.
- Split the package implementation into domain-focused source files while preserving the
  public API.
- Reworked the test entry point into isolated `Core`, `Physics`, `Quality`, and `Docs`
  groups, with package-hygiene, import-hygiene, JET, and doctest coverage.
- Replaced generic `ErrorException` failures with specific argument, dimension, and
  invariant exceptions.
- Tightened setup field types so callable storage remains inferable.
- Pointed the badges, documentation URLs and Documenter deploy target at
  `LupoLab/ModelPNPS.jl`, the actual repository, instead of `jtravs/ModelPNPS.jl`.
- Documentation links now use the organisation's custom domain,
  `lupo-lab.com/ModelPNPS.jl`, which is what `lupolab.github.io` redirects to.
- CI now tests Julia 1.12 and current stable, matching `[compat] julia`; it
  previously included 1.10, which cannot resolve the project.
- Brought the hand-written docstring signature lines back in line with the code, and
  documented the keywords they had drifted away from (`geometry`, `fftsize`,
  `arraytype`, `beamlets_on_host`, and the `run_scan` and `simulate_delay_point`
  solver keywords).

[Unreleased]: https://github.com/LupoLab/ModelPNPS.jl/compare/v1.0.0...HEAD
