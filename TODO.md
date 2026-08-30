# TODO

Known work that is deliberately deferred. Each entry says what is wrong now, what
"done" looks like, and why it was not done at the time it was found. Anything here
should either get done or be argued out of existence — it is not a wish list.

## Examples

- **Bring `examples/` up to the standards in `AGENTS.md`.** The example scripts
  predate the current conventions and were left untouched by the package refactor.
  They are outside the Runic scope (`src test docs`), so nothing checks them: 28
  lines exceed the 92-character limit, mostly `@printf` format strings, and the
  naming and comment conventions vary between scripts. Reflowing working operator
  scripts carries real risk of breaking a running campaign, so this needs doing
  deliberately, script by script, rather than by a global pass.

- **Decide which examples are the *documented* ones.** `README.md` and the manual
  point at three scripts (two mask-scheme runs and the Gaussian comparison), but the
  directory now holds the whole H200 campaign as well. Either curate a small set of
  canonical examples and move the campaign scripts somewhere that says what they are,
  or document the campaign scripts properly.

## Documentation

- **No executable examples.** `AGENTS.md` asks for Documenter `@example` blocks so
  that manual examples execute at build time and cannot silently rot. Every example
  in this manual is a plain `julia` fence instead, because each one needs either an
  external HDF5 file or a multi-gigabyte propagation. A small synthetic case that
  runs in a Documenter build — a tiny grid, `skip_propagation = true`, or the 32×32
  smoke configuration — would make at least the API-shape parts of the manual
  self-checking.

- **No doctests.** For the same reason, there are none. The `Docs` test group and
  the strict build run `doctest(ModelPNPS)` against an empty set. Candidates exist:
  `optimal_spatial_grid` returning a known `(R, N)`, `_resolve_zsave`'s validation,
  the window constructors. Follow the rules in `AGENTS.md` — print invariants, not
  full-precision floats.

## Self-diffraction

- **`Iω_full` and `signal_quadrant_norm` are boxcar-specific.** Both integrate the
  `kx < 0, ky < 0` quadrant, which is where the TG signal sits and is not where the
  SD signal sits (SD puts it on one axis at `-3s/2`). For an SD run they therefore
  report a region containing pump light rather than signal. Windowed extraction
  (`Iω_win`, `Iω_win_reimaged`) is correct for SD provided the window is placed at
  `sd_signal_x`. Generalising the signal region to follow `geometry` would fix both
  the collection-efficiency diagnostic and the error norm.

- **No physics validation of the SD path.** The test suite covers the layout, the
  metadata, the grid bound and the refusals, but nothing checks that an SD run
  produces the expected `2k_E - k_G` signal. That needs a small end-to-end run
  asserting where the signal lands in k-space.

## CI

- **Codecov needs enabling on the repository.** `CI.yml` uploads `lcov.info`,
  `codecov.yml` configures the reporting, and the README carries the badge — but
  none of it does anything until the Codecov GitHub App is installed on the LupoLab
  organisation and `CODECOV_TOKEN` is set as a repository secret. Tokenless upload
  covers only pull requests from forks into a public repository, not pushes to
  `main`. The upload is `fail_ci_if_error: false`, so CI stays green in the meantime
  and the badge shows "unknown".

- **The documentation deploy needs `DOCUMENTER_KEY`.** `Documentation.yml` will build
  on every push but cannot publish to `gh-pages` until the deploy key is set. Until
  then the docs badge points at a URL that does not exist yet.

- **No GPU coverage in CI.** The device code paths are exercised on `JLArrays`, which
  catches array-type and host/device mixing bugs but not CUDA-specific ones. A
  self-hosted runner with a GPU would close that gap; failing that, the campaign
  scripts are the only real coverage.

## Packaging

- **GPU support depends on an unregistered Luna branch.** The device path currently
  requires `jtravs/Luna.jl#modal-fixed` rather than the registered release, so it
  cannot be expressed as a `[compat]` bound and a plain `Pkg.add` of ModelPNPS does
  not get a working GPU setup. This resolves itself when those changes land in a
  registered Luna release; until then it is documented in `README.md` and in the
  manual's GPU page.
