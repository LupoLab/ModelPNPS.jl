# Changelog

All notable changes to ModelPNPS are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `build_setup` and `optimal_spatial_grid` now reject an unknown `geometry`, and
  `build_setup` rejects `geometry = :sd` for a beam model whose builder only ever
  places three beams. Both cases previously produced a boxcar run with no diagnostic.

### Added

- README badges for CI, documentation, coverage, Aqua, JET and Runic, and the CI
  jobs behind them: coverage upload to Codecov, a standalone `Quality` group job,
  and a Runic formatting check.

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
- CI now tests Julia 1.12 and current stable, matching `[compat] julia`; it
  previously included 1.10, which cannot resolve the project.
- Brought the hand-written docstring signature lines back in line with the code, and
  documented the keywords they had drifted away from (`geometry`, `fftsize`,
  `arraytype`, `beamlets_on_host`, and the `run_scan` and `simulate_delay_point`
  solver keywords).

[Unreleased]: https://github.com/LupoLab/ModelPNPS.jl/compare/v1.0.0...HEAD
