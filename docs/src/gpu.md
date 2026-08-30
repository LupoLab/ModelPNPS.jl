```@meta
CurrentModule = ModelPNPS
```

# Running on a GPU

A TG-FROG delay scan is a stack of 3-D nonlinear propagations, and the cost is
dominated by 3-D FFTs on a state of shape `(Nω, Nky, Nkx)`. That is exactly the work
a GPU is built for, and ModelPNPS runs the whole propagation and extraction path on
one by passing a single keyword.

The speedup is the difference between a scan being feasible and not:

| Hardware | Per delay point | 200-point scan |
|---|---|---|
| NVIDIA H200 | **42 s** | ~2.3 h |
| 2 CPU cores | **1.9 h** | ~16 days |

That is a factor of **≈ 160**, measured on the same geometry. A campaign that was a
cluster allocation becomes an afternoon on a single rented card.

!!! warning "Experimental, and it needs a Luna branch"
    GPU support is **working but experimental**. It currently requires the
    `modal-fixed` branch of Luna.jl rather than the registered release:

    ```julia
    import Pkg
    Pkg.add(; url = "https://github.com/jtravs/Luna.jl", rev = "modal-fixed")
    Pkg.add("CUDA")
    ```

    Only NVIDIA cards are supported, through CUDA.jl. Everything on this page is
    exercised by the campaign scripts and by the device test group, which runs the
    full device path on `JLArrays` so that CI covers it without a GPU — but the
    interface may still change.

## Turning it on

One keyword:

```julia
setup_args = (;
    λ0 = 260e-9, τfwhm = 1e-15, energy = 0.1e-6,
    thickness = 40e-6, material = :SiO2,
    mask_diam = 1.0e-3, mask_spacing = 1.0e-3,
    λlims = (143e-9, 600e-9),
    beam, window,
    arraytype = :cuda,               # <- the whole change
    beamlets_on_host = true,
)

run_scan(setup_args, τ; scan_name = "my_run", exec = Scans.LocalExec())
```

Everything that follows is about doing this on a real machine without running out of
memory or waiting for a plan that never comes.

## Pass `arraytype` inside `setup_args`, not as a `run_scan` keyword

This is the one structural rule, and it is not stylistic.

`arraytype = :cuda` is a **symbol**, resolved lazily by `Luna.resolve_arraytype`,
which loads CUDA.jl at the moment it is called. Two consequences follow.

**The GPU package must load where the GPU is.** [`run_scan`](@ref) builds the setup
lazily, inside the scan closure, on the first point each process executes. Put
`arraytype` in `setup_args` and CUDA.jl is loaded on the compute node and nowhere
else. A cluster login node typically has no GPU at all — and has been observed to
fault outright while precompiling CUDA.jl — yet it is where the script is parsed to
generate and submit the batch job.

**World age.** Methods defined by a package loaded *during* a call are invisible to
that same call; Julia rejects them as "too new to be called from this world
context". ModelPNPS handles this internally by resolving the array type first and
then re-entering [`build_setup`](@ref) through `Base.invokelatest`, which is what
`_build_setup_resolved` exists for. You get this for free by passing the symbol
through `setup_args`.

For the same reason, a **cluster** script must not `import CUDA` at the top level. A
rented pod with the GPU directly attached is different: there the simple thing is
also the correct one, and a top-level import is fine.

## `beamlets_on_host`

The three input beamlets are built on the host — masks, Bessel profiles and FFTs are
host code — and then moved to the propagation's array type. `beamlets_on_host = true`
keeps them in host memory instead and uploads the delayed sum once per delay point.

The trade is two fewer resident device fields (about 9 GiB at the production shape)
against one host-to-device transfer per point, which is a fraction of a second
against a propagation of minutes. Turn it on whenever the card is the constraint,
which on a large grid it usually is.

## Budgeting device memory

Guessing is expensive: finding out that a shape does not fit by running it is an
hour of rented GPU and a dead process. [`memory_budget`](@ref) takes the same
`setup_args` NamedTuple that [`run_scan`](@ref) does, reads only the grid-determining
entries, and returns a per-buffer breakdown in GiB. Building the 1-D time grid is
the entire cost, so it is free to call.

```julia
bu = memory_budget(setup_args)
@info "budget" device = bu.device host = bu.host field = bu.field
```

The returned NamedTuple carries the grid sizes (`Nω`, `Nt`, `Nto`, `Nωo`), the size
of one state array (`field`), the per-buffer terms (`state`, `et_win`, `eto`, `ewo`,
`pto`, `analytic`, `window`, `input`) and the totals `device` and `host`.

The envelope path obeys a simple rule — nine RK45 registers plus one transform
buffer, i.e. ten times the field size, measured exactly on an A40. The field path
does **not**; see [Field-Resolved Mode](field_mode.md#Cost).

!!! warning "The response's buffer appears on the first step, not at setup"
    `Nonlinear.KerrFieldNoTHG` allocates its analytic-signal buffer lazily, when it
    first sees a field. A card with room to spare after `build_setup` can therefore
    still die on the first step — 18 GiB later at the production shape.
    `memory_budget` counts it; a measurement taken after `build_setup` alone will
    not.

### Guard the launch

The campaign scripts refuse to start rather than dying an hour in, and the pattern is
worth copying:

```julia
bu = memory_budget(setup_args)
Luna.resolve_arraytype(:cuda)                 # its own top-level statement
st = Luna.device_memory_status()              # (free, total) in bytes, or nothing
if !isnothing(st)
    free, total = st ./ 2^30
    @info "device" free_GiB = free total_GiB = total
    bu.device > 0.9 * free && error("needs ~$(bu.device) GiB, only $free GiB free")
end
```

### Measured shapes

Envelope mode, 40 µm fused silica, six collection windows, on an H200 (141 GB):

| Transverse `N` | Device | Per point |
|---|---|---|
| 640 | 21.9 GiB | 17–21 s |
| 768 | ~24 GiB | 25–30 s |
| 900 | 43.3 GiB | 34–41 s |
| 1024 | 56.0 GiB | 44–53 s |

Host peak is around 32 GiB per process at the largest of these — budget system RAM
when running several scans concurrently. All of these fit an H200 with better than
2× headroom; `N = 1024` also passes on an 80 GB card with roughly 24 GiB of margin,
while a 48 GB card cannot run `N ≥ 900`.

A longer time grid scales this directly: a 305-point scan at `N = 900` with `Nt` 400
runs at ~68 GiB and 55–65 s per point, so about five hours end to end.

## What the GPU path does differently

Two behaviours switch on automatically for a device run.

**Save-time extraction.** Each z-slice is reduced to its spectra as it is produced,
so the field is never stored, streamed or transferred. This defaults to `true` on a
device, where the saved stack costs 14–18 % of the delay point in temp-file traffic,
and to `false` on the host. The two routes are bit-identical — the same kernels on
the same arrays, minus a lossless HDF5 round trip — so `extract_on_save = true` is
safe on the host too. See [`simulate_delay_point`](@ref).

**Streaming is skipped.** `stream = true` writes propagation slices to a node-local
temp file instead of holding the whole `(ω, ky, kx, z)` stack in memory. With
save-time extraction there are no slices to stream, so no temp file is created.

Between delay points [`run_scan`](@ref) runs a garbage collection and calls
`Luna.device_reclaim()`, because a GPU array library keeps its own memory pool that
collection alone does not return. Without it the next point can find the card full.

## Practical setup for a single-GPU machine

```julia
using ModelPNPS
import Luna
import Luna.Scans

Luna.set_fftw_mode(:estimate)
Luna.set_fftw_threads(Threads.nthreads())
```

`:estimate` planning matters more than it looks. MEASURE-class planning of
production-size 3-D transforms takes tens of minutes per worker, and every ModelPNPS
production script uses `:estimate`. Set both through [`run_scan`](@ref)'s
`fftw_threads` and `fftw_mode` keywords as well: they are applied exactly where the
plans are created, which is the only place that reaches multi-worker `procs` scans —
a top-level call does not.

Use `Scans.LocalExec()` on a directly attached card. To put one scan on each GPU of a
multi-GPU machine, pin them with the environment and run one process per card:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=. my_scan.jl  # one configuration
CUDA_VISIBLE_DEVICES=1 julia --project=. my_scan.jl  # another, in parallel
```

Pass `skip_existing = true` so an interrupted scan resumes where it stopped rather
than recomputing points already in the collected file — which matters when the
machine is rented by the hour.

## A dry-run gate

Every campaign script supports a dry run: build the configuration, print the budget,
check the card, and exit before propagating anything. It costs seconds and catches
the mistakes that otherwise surface hours in.

```julia
if get(ENV, "PNPS_DRYRUN", "0") == "1"
    @info "dry run: configuration validated, NOT propagating" delays = length(τ)
    exit(0)
end
```

## API

[`memory_budget`](@ref) is documented on the [API Reference](interface.md) page; the
`arraytype` and `beamlets_on_host` keywords are documented under
[`build_setup`](@ref), and `extract_on_save`, `stream`, `fftw_threads`, `fftw_mode`
and `skip_existing` under [`run_scan`](@ref).
