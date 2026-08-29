# Changelog

All notable changes to ModelPNPS are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/jtravs/ModelPNPS.jl/compare/v1.0.0...HEAD
