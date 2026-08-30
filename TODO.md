# TODO

Known work that is deliberately deferred. Each entry says what is wrong now, what
"done" looks like, and why it was not done at the time it was found. Anything here
should either get done or be argued out of existence — it is not a wish list.

## Examples

- **`examples/` is not checked by anything.** The directory now holds two curated
  scripts (`tgfrog_window_series.jl` and its GPU variant), but they are outside the
  Runic scope (`src test docs`) and outside the test suite, so nothing verifies that
  they stay formatted or even parse. Adding them to the Runic scope is cheap; a
  parse-only check (`include` with the scan call behind the existing dry-run gate)
  would need a CI environment that can load the package, which exists. Deferred with
  the docs `@example` work below, which has the same shape.

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

- **The repository is not activated on Codecov.** `CI.yml` uploads `lcov.info`,
  `codecov.yml` configures the reporting, `CODECOV_TOKEN` is set and the upload runs
  — and Codecov rejects it with "Repository not found". Its API explains why:
  `api.codecov.io/api/v2/github/LupoLab/repos/ModelPNPS.jl/` reports
  `active: false, activated: false`. Appearing in the organisation's repository list
  only means the GitHub App can see the repository; to the upload endpoint an
  unactivated repository does not exist. Activating it on Codecov and using the
  repository upload token it then issues should be the whole fix. The upload is
  `fail_ci_if_error: false`, so CI stays green in the meantime and the badge shows
  "unknown".

- **The documentation deploy needs `DOCUMENTER_KEY`.** `Documentation.yml` will build
  on every push but cannot publish to `gh-pages` until the deploy key is set. Until
  then the docs badge points at a URL that does not exist yet.

- **No GPU coverage in CI.** The device code paths are exercised on `JLArrays`, which
  catches array-type and host/device mixing bugs but not CUDA-specific ones. A
  self-hosted runner with a GPU would close that gap; failing that, real-hardware
  coverage only happens when someone runs a scan.

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
  edits, use a separate development environment that `develop`s both packages,
  rather than the ModelPNPS project environment.

## GPU

Deferred device-path work, in rough order of value. All of it is optimisation or
validation; the path itself works and is covered on `JLArrays` in CI.

- **Host peak during `build_setup`.** The beamlet build holds more intermediate
  fields than it needs to — of order 9 GiB (envelope) and 18 GiB (field mode) of the
  setup peak could go, by building beamlets in place. The host peak decides how many
  scan instances share a machine, so this is the highest-value item.

- **The envelope device budget is unvalidated on hardware.** `memory_budget`'s
  field-mode figure matches a real card to 0.1%, but the envelope path measured
  about 11 GiB above the model during a delay point. Nothing is at risk (the shapes
  fit with margin), but the number the drivers refuse shapes on deserves the same
  validation the field side has.

- **Reuse the delayed-input buffer across delay points.** `delayed_input` allocates
  a fresh device field per point, which the solver adopts and frees — an
  allocate/free cycle per point against a memory pool that then needs
  `device_reclaim()`. A setup-owned buffer would remove the churn. It is the `input`
  term in `memory_budget`.

- **Shrink the field-mode analytic-signal buffer.** The `:nothg` response allocates
  a complex `(nto, ny, nx)` buffer — 18 GiB at `N = 768`, the largest term in the
  field-mode budget. An `|E_a|² = E² + H[E]²` formulation would trade it for one
  real buffer at half the size plus one extra transform. Worth doing only if
  field mode has to run on an 80 GB card.

- **No field-versus-envelope comparison tool in the tree.**
  `verify_against_collected` refuses on the `Nω` mismatch between the two grids.
  The right comparison is between physical spectral densities splined onto a common
  band, reported per depth relative to the trace peak; the recipe exists as a
  testset but was deliberately not shipped as API.
