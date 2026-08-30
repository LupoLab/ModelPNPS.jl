# AGENTS.md

> Package-specific development instructions for `ModelPNPS`.

## Guiding principles

- There should be one obvious way to do each thing. Prefer the existing pattern over inventing a new one; if you find two ways to do the same thing, consolidate them.
- Optimise for the reader. Clear beats clever — code is read far more often than it is written.
- Small, single-purpose functions with explicit inputs and outputs. Prefer pure functions; push side effects (I/O, global state, mutation) to the edges.
- Generic code is preferred unless the code is known to be specific. Write against interfaces (`eachindex`, `similar`, broadcast) rather than against `Array` and one-based indexing, so that `Float32`, `BigFloat`, `ForwardDiff.Dual`, `OffsetArray` and GPU array types keep working.
- Every method of a function must mean the same thing. A new dispatch is an instantiation of the existing meaning, never a different meaning that happens to share a name.
- Make the implicit explicit: explicit arguments over hidden state, explicit public API over convention, explicit errors over silent fallbacks.
- Fail loudly and early with a specific exception type. Never swallow errors or return a sentinel where throwing is correct.

## Environment and tooling

- Julia is managed with **juliaup**. The minimum supported version is **1.12**; keep this consistent with `[compat] julia` in `Project.toml`. CI runs 1.12 and the current stable release.
- Dependencies are declared in `Project.toml`. This is a **library**: `Manifest.toml` is not committed and must not be added to the repository.
- Every dependency has a `[compat]` bound. Omit the default caret (`DataFrames = "0.17"`, not `"^0.17"`). Never use `>=` to avoid an upper bound. The lower bound is the last tested version. CompatHelper keeps these current.
- Adding a dependency is a design decision, not a fix. Reach for the standard library first. If a new dependency is genuinely needed, say so explicitly and wait — do not add one silently to make something work.
- Add dependencies with `Pkg.add` from within the project environment so `Project.toml` is updated correctly, then add the `[compat]` bound by hand. Never install into the default `@v1.x` environment, and never hand-edit the `[deps]` section.
- Run everything in the project environment: prefix commands with `julia --project=. --startup-file=no`. The `--startup-file=no` is not optional — without it you inherit whatever is in the user's `startup.jl`.
- Runic lives in its own shared environment (`@runic`) and is deliberately not a project dependency.
- `docs/` and `test/` have their own `Project.toml` files.

## Commands

- Instantiate: `julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'`
- Format: `julia --project=@runic -m Runic --inplace src test docs`
- Check formatting: `julia --project=@runic -m Runic --check --diff src test docs`
- Test (everything): `julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'`
- Test one group: `GROUP=Core julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'`
- Doctests only: `julia --project=docs --startup-file=no -e 'using Documenter, ModelPNPS; doctest(ModelPNPS)'`
- Build docs: `julia --project=docs --startup-file=no docs/make.jl`
- Type-instability report (advisory): `julia --project=. --startup-file=no -e 'using JET, ModelPNPS; JET.report_opt(ModelPNPS)'`

### Working quickly

Julia pays compilation cost on every fresh process, so a naive edit–test loop is far slower than the Python equivalent. Keep it tight:

- While iterating, run **one** test group with `GROUP=...`, not the whole suite. Run the full suite once, before declaring the work done.
- Run `Pkg.precompile()` once after any dependency change rather than absorbing it repeatedly.
- `doctest(ModelPNPS)` is much cheaper than a full `docs/make.jl`. Use it while editing docstrings; build the full docs only at the end.
- Do not launch interactive REPLs. Use `julia --project=. --startup-file=no -e '...'`.

## Definition of done

Before treating any change as complete, run these in order and ensure each passes with no new warnings:

1. `julia --project=@runic -m Runic --check --diff src test docs`
2. `julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'` — the full suite, all groups. This covers unit tests, Aqua, ExplicitImports, JET error analysis and doctests.
3. `julia --project=docs --startup-file=no docs/make.jl` — must build with `warnonly = false`.

Then, advisory but expected:

4. `JET.report_opt` on the code you touched. Fix type instabilities that are straightforward to fix (a concrete field type, an annotation, removing a closure). If a report is a false positive or the fix is invasive, say so explicitly rather than silently ignoring it.

Note that **there is no static type checker in this ecosystem** — nothing corresponds to mypy or pyright. Do not go looking for one, and do not add type annotations in an attempt to satisfy one. Type stability is the property that gets verified, via `@inferred` tests and `JET.report_opt`. Similarly, there is no single lint tool: the responsibilities are split across Runic (whitespace), Aqua (package hygiene), ExplicitImports (import hygiene), JET (correctness), and the written rules below, which nothing enforces mechanically.

Do not declare work finished while any of steps 1–3 fails. If a check genuinely cannot pass for a justified reason, say so explicitly rather than suppressing it.

Furthermore, ensure all added functionality is documented both in docstrings *and* in the manual under `docs/`.

## Code style

- Formatting is owned by **Runic.jl**, which has no configuration. Do not hand-format, do not fight the formatter, and do not add a formatter configuration file of any kind.
- `# runic: off` / `# runic: on` may be used in pairs, on their own lines at the same level of the syntax tree, only where manual alignment genuinely aids readability (a hand-aligned table of Butcher tableau coefficients, for example). Each use carries a comment saying why.
- Everything Runic does not enforce follows the **SciML Style Guide** (<https://docs.sciml.ai/SciMLStyle/stable/>), with the deviations recorded below. Where this file and SciMLStyle disagree, this file wins.
- Line length is **92 characters**. Runic does not wrap long lines, so this is on you — check it manually.
- Naming: `CamelCase` for types, structs and modules; `snake_case` for functions and variables; `SNAKE_CASE` for constants; abstract types begin with `Abstract`; type variables are a single capital letter related to what they type. Whole words beat abbreviations, except for well-established domain terms.
- **Deviation from SciMLStyle:** do not use the `__` prefix for internal names. Public and internal are distinguished by `export` and `public` (see *Functions and modules*), not by a naming convention.
- **Deviation from SciMLStyle:** Unicode is permitted, including in the public API, where it matches standard physical or mathematical notation — `γ`, `ω₀`, `β₂`, `ħ` are all fine as argument and field names. Do not invent Unicode for names that are not mathematical entities.
- Imports are explicit: `using Dates: Year, Month`, never a bare `using Foo` for a package whose names you then use unqualified. No wildcard imports. Keep `import` and `using` in separate blocks separated by a blank line, at the top of the file or immediately after the `module` declaration. Large sets of imports go on space-filling comma-separated lines.
- Floating-point literals always carry a leading and/or trailing zero: `0.1`, `2.0`, `3.0f0`.
- Prefer `Int` to `Int32`/`Int64` unless the bit size matters.
- `for` loops always use `in`, never `=` or `∈`. Runic converts these automatically.
- Ternary operators occupy a single line and are never chained. Use `if`/`elseif`/`else` or dispatch instead.
- At call sites, separate keyword arguments from positional ones with a semicolon.
- Avoid splatting (`...`). Prefer `collect`, `vcat`, `hcat`, `reduce`.
- Unix line endings (`\n`).

## Types, dispatch, and stability

The rules here are close to the opposite of the Python convention. Annotations exist for dispatch and correctness, not for documentation.

- **Function signatures: annotate as generally as possible.** `splicer(arr::AbstractArray, step::Integer)`, not `splicer(arr::Array{Int}, step::Int)`. Over-constraining a signature is a real bug: it breaks `ForwardDiff.Dual`, `Unitful` quantities, `BigFloat`, `Float32` and GPU arrays. If an annotation is not doing dispatch work or enforcing a genuine contract, leave it off.
- **Struct fields: concrete or parametric, never abstract.** An abstract field type makes access non-inferable and is a performance defect, not a style preference. Use a type parameter when generality is required. Untyped fields are annotated `::Any` explicitly.
- Prefer `struct` to `mutable struct`. Where a mutable-looking update is needed on an immutable struct, use Accessors.jl (`@set`) rather than making the type mutable.
- Group related parameters into a small immutable struct rather than passing many positional arguments. `Base.@kwdef` for defaults.
- Avoid elaborate `Union` types. `Vector{Union{Int,AbstractString,Tuple,Array}}` should probably be `Vector{Any}`. Keep unions to two or three types for branch splitting.
- Do not compare types with `===`; use `isa` or `<:`.
- Keep code type-grounded: well-typed containers, concrete types where they can be concrete, no untyped globals. Globals must be `const` and `SNAKE_CASE`, declared at the top of the file after the imports and exports.
- **Avoid closures.** They cause type instabilities that are tedious to track down, and their absence makes instability hunting much faster. Prefer `map(Base.Fix2(getindex, i), vs)` to `map(v -> v[i], vs)`. To update an outer variable, do it explicitly with a `Ref` or a purpose-built struct. Where a closure is unavoidable or overwhelmingly clearer in code that is not performance-critical, use one and note why.
- Assert type stability on the public API with `@inferred` in the tests.

## Mutation and allocation

- A function either treats its inputs as **immutable**, or is **non-allocating and reuses caches**. These are two different worlds and mixing them within one function is the worst outcome: you take on the non-locality and compiler opacity of mutation without gaining the allocation benefit.
- Out-of-place is the default. Mutation must earn its place by demonstrably removing heap allocations. For sufficiently large matrices `A * B` is as fast as `mul!(C, A, B)`, so write `A * B` — unless the enclosing function is carefully non-allocating throughout, in which case use `mul!` for consistency with its surroundings.
- Functions that mutate an argument end in `!`. The mutated argument comes early in the signature: after any function argument or `IO` stream, and before the values being written into it.
- Allocate generically with `similar(A)` rather than `Array{T}(undef, size(A))`, so the output type follows the input type.
- Default to constructs that initialise data (`zeros`, `fill`). Use `undef` only where there is a demonstrated performance impact, and then ensure every element is assigned within the same function that allocated it.
- Prefer broadcast (`@.`) to explicit indexing loops where it preserves genericity.
- `@inbounds` is an unsafe operation and since 1.9 frequently does not help performance. Avoid it. If it is used, apply it to the narrowest possible expression, use `eachindex` rather than `1:length(A)` inside it, and add a comment justifying that every index is in bounds by inspection.

## Docstrings

- Every exported name has a docstring. Internal helpers get a one-line docstring where the name is not fully self-explanatory. Document the *function*, not each method, unless a method's behaviour genuinely deviates.
- Document at the highest level that applies: an interface that all methods follow, or an abstract type, in preference to documenting every concrete instance. Instances then refer to the higher-level documentation.
- Docstrings are Markdown, hand-written, wrapped at 92 characters.
- **The signature line is written by hand and is not generated.** It is therefore the thing most likely to go stale. Whenever you change a signature — arguments, keywords, defaults, return type — update the docstring signature line in the same edit, and check it before declaring the work done.
- Structure: an indented signature line, a blank line, a one-sentence imperative summary, then as applicable `# Arguments`, `# Keywords`, `# Returns`, `# Throws`, `# Examples`, `# Extended help`. For types, `# Fields`.
- For every numerical quantity, state the **units, the valid range, and any sign or normalisation convention** in the parameter description. Where a Fourier-transform or phase-sign convention is implied, state it explicitly.
- Only public fields are documented. A documented field is part of the public API and renaming it is a breaking change. Prefer documenting accessor functions to documenting fields.
- Use `@doc doc""" """` whenever the docstring contains LaTeX.
- Where a signature has many arguments or keywords, abbreviate it as `args...` / `kwargs...` on the signature line and describe them in the sections below.

```julia
"""
    dispersion(fibre::HollowFibre, ω::Real; order::Integer = 2) -> Float64

Compute the propagation constant derivative of the given order at angular frequency `ω`.

# Arguments
- `fibre::HollowFibre`: the waveguide geometry and fill gas.
- `ω::Real`: angular frequency in rad/s, valid for `0 < ω < 4e16`.

# Keywords
- `order::Integer = 2`: derivative order of β with respect to ω. `order = 2` returns the
    group-velocity dispersion in s²/m.

# Returns
- `Float64`: the derivative dⁿβ/dωⁿ in units of sⁿ/m.

# Throws
- `DomainError`: if `ω` is outside the transmission window of the fill gas.
"""
```

## Doctests

Doctests are compared as text, with no tolerance. Any digit that shifts across architectures or Julia versions breaks the build. Write them accordingly:

- **Prefer output that cannot drift.** A doctest that prints `true` from `isapprox(energy(sol), e0; rtol = 1e-8)` documents the invariant *and* is immune to last-digit variation. The same applies to `size`, `length` and `eltype` checks.
- Where a number genuinely helps the reader, reduce the printed precision explicitly with `round(x; sigdigits = 3)` or `Printf.@printf`, choosing a precision comfortably inside the expected numerical variation rather than one digit inside it.
- Never paste full-precision floating-point output, and never paste array `repr` output — array display formatting has changed between Julia minor versions.
- `DocTestFilters` is available as an escape hatch, but a filter must be specific and carry a stated reason, in the same way as a lint suppression. Never a blanket filter.
- Anything computationally prohibitive for CI is not a doctest. Make it an `@example` block in the manual, or a test.
- Quantitative verification belongs in the test suite with explicit tolerances, not in docstrings. A doctest demonstrates usage.
- **Never run `doctest(...; fix = true)`.** It rewrites source files. Report what would change and let the user run it.

## Comments

**This section deliberately overrides SciMLStyle**, which minimises comments. Two categories of comment are being distinguished, and only one of them is noise.

- A comment that restates what the code does is noise — fix the names instead. `# fx applies the effects to a tree` should become `apply_effects(tree)`.
- A comment that records the **physics and mathematics the code implements** is required, and should be comprehensive. Cover: the governing equation and the form in which it has been discretised; the approximation being made and the regime in which it is valid; unit, normalisation, sign and Fourier-transform conventions; why a particular numerical scheme was chosen; non-obvious trade-offs; and references, with a DOI or arXiv number for a paper and a URL for an issue or PR.
- These comments are not refactoring debris. When code moves, the comment moves with it. Never delete one because it looks verbose.
- Keep each comment adjacent to what it describes and update it when the code changes.
- Quote code in comments with backticks.
- Comments go above the code they refer to; inline comments only where they fit inside the line limit.
- Mark deliberate follow-ups as `# TODO(context): ...` so they are greppable, and known-broken code as `# XXX: ...`.
- Delete commented-out code — version history is the archive.

## Functions and modules

- One function, one responsibility. If a function needs a paragraph to explain, or has many nested branches, split it.
- Keep parameter lists short. Group related parameters into a small struct rather than passing many positional arguments.
- Arguments without defaults should be positional. Give a default only where it is historically expected or right for more than about 95% of uses; a tolerance may have a sensible default, a learning rate does not.
- Prefer instances to types as arguments: `solve(prob, RK4())`, not `solve(prob, RK4)`.
- Use short-form definitions (`f(x) = ...`) only when they fit on one line.
- Avoid type piracy: do not add methods to functions you do not own on types you do not own.
- Prefer not to shadow names. Add a method to a `Base` function when your operation really is an instance of that function's meaning; otherwise choose a distinct name rather than creating `ModelPNPS.sort`.
- `src/ModelPNPS.jl` contains the `module` declaration, the imports, the includes and the public-API declarations, and nothing else. No code before or after the module block except a module docstring. Code inside the top-level module block is not indented.
- Include order matters — a type must be defined before the file that uses it is included.
- Declare the public surface explicitly: `export` for names that belong in the user's namespace, `public` for names that are API but should stay qualified. Anything neither exported nor declared `public` is internal and may change without a breaking release.
- Do not use non-public names from Base or from other packages (`Base.foobar`). Extending a function with a qualified method definition (`function Base.getindex(...)`) is fine and is not what this refers to.
- Optional dependencies use package extensions in `ext/` with `[weakdeps]`. Never Requires.jl.
- Organise modules by domain concept, not by a catch-all `utils`. Submodules should be rare — if something is separable enough to be a submodule, consider whether it should be a separate package.
- Avoid `eval` except for top-level code generation. Avoid the `unsafe_*` family and `ccall` unless genuinely necessary.

## Errors

- `error("string")` is avoided. Throw a specific exception type: `ArgumentError`, `DomainError`, `DimensionMismatch`, or a package-specific type `<: Exception` with a `Base.showerror` method.
- Catch problems as high as possible. Validate inputs at the top of the public entry point, not deep in an inner loop, so that the error message can be expressed in the user's terms.
- Error messages use user-facing terminology, not internal implementation details, and suggest a correction where one is obvious. "Pressure must be positive; got -0.5 bar" beats "DomainError with -0.5".
- `@assert` is for internal invariants only. It is not a contract, it may be disabled, and it must never be used to validate arguments.
- Avoid `try`/`catch`; use it as minimally as possible.
- Never swallow an error or return a sentinel where throwing is correct.

## Tests

- Tests use the `Test` stdlib with **SafeTestsets**, and live in `tests/` mirroring the package layout.
- `test/runtests.jl` only shuttles to other files. Every test file is included via a single-line safe testset:

```julia
@time @safetestset "Dispersion" include("physics/dispersion_tests.jl")
```

- A plain `@testset` does not fully enclose definitions made inside it, so values and functions leak between tests. Use `@safetestset`.
- Every test script must be fully reproducible in isolation — a reader should be able to copy it out and run it.
- Tests are grouped by category, selected with the `GROUP` environment variable, with `All` as the fallback so that `Pkg.test()` runs everything. Current groups: `Core`, `Physics`, `Quality`, `Docs`. Grouped tests live in the same folder.
- The `Quality` group runs `Aqua.test_all(ModelPNPS)`, the ExplicitImports checks (`check_no_implicit_imports`, `check_no_stale_explicit_imports`, and the other `check_*` entry points as appropriate), and `JET.test_package(ModelPNPS)` for error analysis. Failures here are blocking.
- Name testsets `test_<unit>_<behaviour>`.
- Every public function has tests for the normal case, the edge cases, and the error paths. Use `@test_throws` with the specific exception type. Add a regression test with every bug fix.
- Parametrise with a `for` on the testset line, never by copy-paste:

```julia
@testset "dz = $dz" for dz in (1e-3, 1e-4, 1e-5)
```

- For numerical code, assert with explicit tolerances (`@test a ≈ b rtol = 1e-10`). Test **invariants and conservation laws** — energy, photon number, norm, symmetry, known analytic limits — not only point values. A convergence test against the expected order of the scheme is worth more than a stored reference number.
- Cover input **types**, not just lines. Where applicable, test `Float64`, `Float32`, `ComplexF64`, `BigFloat` and `ForwardDiff.Dual`. Line coverage with a single type says very little.
- Use `@inferred` to assert type stability of the public API.
- Randomness must be reproducible across Julia versions: use **StableRNGs.jl** for anything whose expected values depend on the RNG. Julia's default random stream is not stable between releases.
- Tests are deterministic, isolated and fast. No network access, no hidden global state.
- Write the test alongside the code; a feature is not done until it is tested.
- CI runs 1.12 and the current stable release.

## Documentation

- Docs are built with **Documenter.jl** (stock HTML backend), and the API reference is generated from docstrings — so the docstring is the source of truth.
- `makedocs` runs with `warnonly = false`, `checkdocs = :all` and `doctest = true`. Link checking runs in CI.
- All functionality of this package must be described in the user manual under `docs/`. This includes background explanations and context, how the functionality works, the physical assumptions behind it, and worked examples.
- Tutorials come before reference material. There is a starting tutorial covering the 90% use case, showing a complete, opinionated workflow end to end. Variable names in tutorials matter — whatever you use will be copied.
- Examples in the manual use Documenter `@example` blocks so that they execute at build time and cannot silently rot. Advanced or performance-oriented material is separated from the introductory tutorial.
- Summarise before specifying: each page describes its contents before descending into API details.
- When adding functionality, always add documentation in the same change. When working on something that does not appear to be documented, check this and add appropriate documentation.
- Keep `README.md` and any usage guide current with behaviour changes.
- Record notable changes in `CHANGELOG.md` (Keep a Changelog style). Follow semantic versioning.

## Git and releases

- Make clean, logical git commits with descriptive but not overly verbose commit messages.
- Prefer more frequenct clean commits over large big ones.
- *Never* push.
- You can fetch, pull, branch when instructed to do so. If you want to do this, ask.
- Do not make releases or change release versions
