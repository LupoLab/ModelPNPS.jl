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

## Upstream (Luna)

- **`Plotting.jl` forces matplotlib on every consumer.** `src/Luna.jl` includes
  `Plotting.jl` unconditionally, and that does `using PyPlot`, so merely importing
  Luna — and therefore ModelPNPS — requires a working matplotlib through PyCall.
  This broke the documentation build on a cold CI runner, and is worked around by
  forcing PyCall onto its own Conda python in the workflows. The real fix is
  upstream: make PyPlot a `[weakdeps]` package extension so plotting loads only when
  the user asks for it. That would also drop Conda, PyCall and matplotlib from every
  headless deployment, which matters most on a GPU pod.

## Packaging

- **The package depends on an unregistered Luna branch.** ModelPNPS does not load
  against a registered Luna release — `Output.willsave` and a dozen other APIs it is
  written against only exist on `jtravs/Luna.jl#modal-fixed`, which is 108 commits
  past `v0.6.2`. `Project.toml` carries a `[sources]` entry so that the active
  project and CI resolve the branch, but `[sources]` is ignored when ModelPNPS is
  used *as a dependency*, so a downstream environment must add the same branch
  itself. The `[compat] Luna = "0.6.2"` bound is satisfied but meaningless, since the
  branch still declares that version. All of this resolves when the changes reach a
  registered release; that is also when a General registration becomes possible,
  since a package with `[sources]` cannot be registered.

- **`[sources]` blocks `Pkg.develop` on Luna in this project.** With a URL source in
  `Project.toml`, `Pkg.develop(path = "~/.julia/dev/Luna")` fails with ``path` and
  `url` are conflicting specifications`. To test ModelPNPS against in-progress Luna
  edits, use a separate development environment that `develop`s both packages, the
  way the pod scripts do, rather than the ModelPNPS project environment.
