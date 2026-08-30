# ============================================================================
# High-level scan orchestrator
# ============================================================================

"Return no additional values for `Output.scansave`."
_empty_extra_outputs(_) = NamedTuple()

"Callable wrapper for an already-built setup."
struct ExistingSetup{S}
    setup::S
end

(builder::ExistingSetup)() = builder.setup

"Callable wrapper that resolves and builds setup arguments lazily."
struct SetupArguments{A}
    arguments::A
end

(builder::SetupArguments)() = _build_setup_resolved(builder.arguments)

"""
    _completed_scanidcs(scan_name) -> Set{Int}

Scan indices already present in `<scan_name>_collected.h5`, i.e. those whose trace data
is not all zero. An empty set if the file does not exist yet.

Reads one point at a time: the file may be large and this runs before any propagation.
"""
function _completed_scanidcs(scan_name::AbstractString)
    fn = scan_name * "_collected.h5"
    isfile(fn) || return Set{Int}()
    done = Set{Int}()
    HDF5.h5open(fn, "r") do f
        ks = filter(Base.Fix2(startswith, "Iω"), collect(keys(f)))
        isempty(ks) && return
        d = _hdf5_dataset(f, first(ks))
        for i in 1:size(d, 3)
            any(!iszero, d[:, :, i]) && push!(done, i)
        end
    end
    return done
end

"""
    run_scan(setup, τs; scan_name, exec, kwargs...) -> Nothing

Build a `Luna.Scans.Scan` over the delay array `τs` and run
[`simulate_delay_point`](@ref) at every τ, calling `Output.scansave` to
write each result into the collected HDF5 file at
`"<scan_name>_collected.h5"`. The metadata block (`combined_grid`) is
written once on the first scan point.

`exec` must be a `Luna.Scans.AbstractExec` instance (e.g.
`Scans.SlurmExec(...)` or `Scans.LocalExec()`).

`zsave` selects the propagation snapshots saved at every delay (see
[`simulate_delay_point`](@ref)): an `Integer` gives a uniform grid of that many
points over `[0, thickness]` (default `nz`), or a `Vector` of explicit material
thicknesses [m] (e.g. `[1e-6, 10e-6, 20e-6, 40e-6]`). `thickness` is appended to
the vector if absent so the final slice is always the full-propagation output.
The trace datasets become `(Nω, nz, Nτ)` and the realized z positions are stored
once in `/grid/zsave`. Because the field at an intermediate z equals a dedicated
thickness-z run, every shorter thickness comes free from one full-thickness run;
note that peak memory scales with the number of z points.

`extra_outputs(output_namedtuple)` is an optional callable returning extra named tuples to
splat into `scansave`. The default is empty.

# Keywords

- `scan_name`: base name of the collected file, `"<scan_name>_collected.h5"`.
- `exec`: the `Luna.Scans.AbstractExec` instance described above.
- `nz = 2`, `zsave = nz`: propagation snapshots, as above.
- `init_dz = 5e-7`, `rtol = 1e-6`, `max_dz = 0.0`: solver settings forwarded to
  [`simulate_delay_point`](@ref); `max_dz = 0.0` means `thickness/2`. They are
  recorded in the file's `/grid` block as provenance.
- `norm = Luna.RK45.weaknorm`: RK45 error norm.
- `norm_builder = nothing`: a callable `setup -> norm`, used instead of `norm`, for a
  norm that cannot exist before the setup does. Pass
  `norm_builder = signal_quadrant_norm` to get [`signal_quadrant_norm`](@ref) built
  lazily on the compute node.
- `twin_period = 1`: accepted steps between applications of the spectral/temporal
  windows. `1` applies them after every step, which makes the apodisation damping
  scale with the step count; a large value applies them only at saves, which with
  `step_on` sit at identical positions for any `rtol`.
- `fftw_threads = 0`: FFTW threads per process, set where the plans are created so
  that it reaches `procs` workers (a top-level `Luna.set_fftw_threads` does not).
  With `procs` workers sharing `cpus` cores, pass `cpus ÷ procs`. `0` leaves it alone.
- `fftw_mode = :estimate`: FFTW planning effort, set on the same path and for the same
  reason. MEASURE-class planning of production-size 3-D transforms costs tens of
  minutes per worker.
- `stream = true`: write the propagation slices to a node-local temp file rather than
  holding the whole `(ω, ky, kx, z)` stack in memory (~2.15 GB per slice at production
  size). Ignored when save-time extraction is active, which stores no slices at all.
- `extract_on_save = nothing`: reduce each slice as it is produced; see
  [`simulate_delay_point`](@ref). `nothing` picks the per-device default.
- `skip_existing = false`: resume an interrupted scan by skipping delay points already
  present in the collected file. An all-zero slice is the "not yet computed" marker,
  the same test [`verify_against_collected`](@ref) uses.
"""
function run_scan(
        setup_fn, τs::AbstractVector;
        scan_name::AbstractString,
        exec,
        nz::Int = 2, zsave::Union{Integer, AbstractVector} = nz,
        init_dz::Float64 = 5.0e-7,
        rtol::Float64 = 1.0e-6,
        max_dz::Float64 = 0.0,
        norm = Luna.RK45.weaknorm,
        twin_period::Int = 1,
        norm_builder = nothing,
        fftw_threads::Int = 0,
        fftw_mode::Symbol = :estimate,
        stream::Bool = true,
        extract_on_save::Union{Nothing, Bool} = nothing,
        skip_existing::Bool = false,
        extra_outputs = _empty_extra_outputs
    )
    scan = Scans.Scan(scan_name, exec; τ = τs)
    # Resume: `Output.scansave` allocates the full (Nω, nz, Nτ) datasets up front and
    # fills points in as they complete, so an all-zero slice IS the marker for "not yet
    # computed" — the same test `verify_against_collected` uses. Skipping those indices
    # lets an interrupted scan continue where it stopped, which matters when the machine
    # is rented by the hour.
    done_idcs = skip_existing ? _completed_scanidcs(scan_name) : Set{Int}()
    isempty(done_idcs) || @info "resuming: skipping $(length(done_idcs)) completed " *
        "point(s) of $(length(τs)) in $(scan_name)_collected.h5"
    # LAZY SETUP: everything expensive — building the multi-GB beamlet arrays,
    # planning the FFTs, resolving metadata — happens inside the scan closure,
    # on the FIRST point this process executes. At submission time
    # (SlurmExec/SSHExec generate the job script and submit without running
    # any point), `setup_fn` is never called, so launching a scan from a
    # memory-limited login node costs seconds and megabytes. Each array task
    # re-executes the script and builds its own setup on its first point.
    setup = nothing
    zvec = nothing
    cg = nothing
    normx = nothing
    Luna.runscan(scan) do scanidx, τi
        # Before anything else, including building the setup: a resumed scan should pay
        # nothing at all for a point it already has.
        scanidx in done_idcs && return nothing
        if setup === nothing
            # Per-process FFTW threading, applied exactly where the plans are
            # created. In `procs` (multi-worker) scans the workers never
            # execute the script's top level, so a top-level
            # `set_fftw_threads` call does not reach them — this does. With
            # `procs` workers sharing `cpus` cores, pass
            # `fftw_threads = cpus ÷ procs`.
            fftw_threads > 0 && Luna.set_fftw_threads(fftw_threads)
            # Same worker-visibility problem as the threads: a top-level
            # `set_fftw_mode` in the script never reaches `procs` workers,
            # whose planning would then use Luna's default — MEASURE-class
            # planning of production-size 3D transforms takes tens of minutes
            # per worker. :estimate plans in seconds and is what every
            # ModelPNPS production script uses.
            Luna.set_fftw_mode(fftw_mode)
            setup = setup_fn()::TGFROGSetup
            zvec = _resolve_zsave(zsave, setup.grid.zmax)
            # `norm_builder` exists because a setup-derived norm (e.g.
            # `signal_quadrant_norm`) cannot be constructed before the setup:
            # pass `norm_builder = signal_quadrant_norm` instead of `norm` and it is
            # built here, lazily.
            normx = isnothing(norm_builder) ? norm : norm_builder(setup)
            # Metadata: shallow copy so the shared combined_grid is unmutated;
            # /grid/zsave equals the per-point realized out.zsave (the
            # resolution is deterministic). rtol/max_dz/norm are provenance.
            cg = copy(setup.combined_grid)
            cg["zsave"] = zvec
            cg["rtol"] = rtol
            cg["max_dz"] = max_dz > 0 ? max_dz : setup.grid.zmax / 2
            # Delay-convention marker: traces are stored in the gate-delay
            # frame (see `delayed_input`); loaders must not reverse the axis.
            cg["delay_convention"] = "gate"
            cg["error_norm"] = _norm_name(normx)
            # Apodisation cadence. 1 = the spectral/temporal windows are applied
            # in place after EVERY accepted step, which makes the damping scale
            # with the step count and the scheme non-convergent in rtol; large
            # values apply them only at saves (Output.willsave), which with
            # step_on are at identical positions for any rtol.
            cg["twin_period"] = twin_period
        end
        # Stream the propagation slices to a node-local temp file instead of
        # holding the (ω, ky, kx, z) stack in memory (~2.15 GB × nz at
        # production size); only the extracted (Nω, nz) spectra survive.
        # Save-time extraction stores no slices, so there is nothing to stream: skip the
        # temp file rather than creating one that is written to zero times.
        onsave = something(extract_on_save, Luna.Utils.isdevice(setup.transform.Eto))
        fname = (stream && !onsave) ? tempname() * "_pnps.h5" : nothing
        # try/finally, not a plain call followed by rm: if the point throws, the
        # temp file is 36 GB of orphan at production size. run_scan CATCHES the
        # error per point and moves to the next one, so without this a transient
        # full disk becomes a permanent one -- each failing point leaks another
        # file and none of them are ever cleaned. That is what turned an
        # overflowing /tmp into three scans returning 2-24% of their delays on
        # 2026-08-24, with every SLURM job still reporting COMPLETED 0:0.
        #
        # invokelatest: with `arraytype=:cuda` the GPU package was loaded *inside* this
        # closure, so its methods are newer than the world this closure is running in
        # and would be invisible to every device kernel below. One dynamic dispatch per
        # delay point, against a propagation of minutes.
        out = try
            Base.invokelatest(
                simulate_delay_point, setup, τi;
                zsave = zvec, init_dz = init_dz,
                rtol = rtol, max_dz = max_dz, norm = normx,
                twin_period = twin_period, filename = fname,
                extract_on_save = onsave
            )
        finally
            stream && !isnothing(fname) && rm(fname; force = true)
        end
        # Return freed field-sized garbage (the point's input array, extraction
        # temporaries) to the allocator before the next point starts — with
        # two workers sharing a tight cgroup, un-collected garbage from one
        # worker coinciding with the other's peak is an OOM risk.
        GC.gc()
        # A GPU array library keeps its own memory pool, which garbage collection alone
        # does not return. Without this the next point can find the card full.
        Luna.device_reclaim()
        # `zsave` is metadata (stored in /grid/zsave), not a per-delay dataset.
        out_save = Base.structdiff(out, NamedTuple{(:zsave,)})
        Output.scansave(
            scan, scanidx; grid = cg, out_save...,
            extra_outputs(out)...
        )
    end
    return nothing
end

"""
    _scan_peak(dset) -> Float64

Largest absolute value over every *computed* delay point of a collected trace dataset.
Read one point at a time rather than whole: this runs against a file a scan may still be
writing, and the datasets grow with the delay count.
"""
function _scan_peak(dset)
    pk = 0.0
    for i in 1:size(dset, 3)
        s = dset[:, :, i]
        any(!iszero, s) || continue # not yet computed
        pk = max(pk, maximum(abs, s))
    end
    return pk
end

"""
    verify_against_collected(setup_args, collected, scanidcs;
                             zsave, init_dz=5e-7, rtol=1e-6, max_dz=0.0,
                             norm=Luna.RK45.weaknorm, twin_period=1,
                             stream=true, extract_on_save=nothing)
        -> Vector{Dict}

Recompute selected delay points of an existing scan and compare against the collected
HDF5 file — the A/B harness for validating a new code path (or a changed grid) against
reference data.

For each scan index in `scanidcs`, the delay `τ` is read from `/scanvariables/τ` in
`collected`, the point is recomputed via [`simulate_delay_point`](@ref) with the given
solver settings (pass the SAME settings the reference scan used, unless deliberately
testing a change), and every returned trace dataset (`Iω_win`, `Iω_full`, ...) present in
the file is compared. Reference points that are still all-zero (not yet computed by a
running scan) are reported as `NaN` and skipped.

Returns one `Dict` per point with the delay, wall time, `Sys.maxrss()` [GiB], and for
each dataset `ks` the global relative difference
`maximum(abs, new - ref)/maximum(abs, ref)`, plus three diagnostics:

| key | meaning |
|---|---|
| `ks` | max abs difference ÷ **this point's** reference peak |
| `ks*"\\|relscan"` | max abs difference ÷ the **scan-wide** reference peak |
| `ks*"\\|refpeak"` | this point's reference peak |
| `ks*"\\|scanpeak"` | the scan-wide reference peak |

Both normalisations matter. A delay-scan wing carries a signal orders of magnitude below
the τ≈0 signal, so a difference that is irrelevant in the assembled trace can still be a
large fraction of that point's own peak. `relscan` is what a FROG retrieval sees; the
own-peak number is the stricter statement about the code path.

To test a grid change (e.g. N=640 against an N=1024 reference), pass the changed `N`
inside `setup_args` — differences then reflect the grid, not the code.

!!! note
    Strict comparisons need matched FFT configuration: run with the same `fftw_threads`
    and `fftw_mode` as the reference scan (FFT algorithm choice affects round-off).
    Julia-level threading (`JULIA_NUM_THREADS`) does NOT affect results and can be used
    freely to speed up verification.
"""
function verify_against_collected(
        setup_args::NamedTuple, collected::AbstractString,
        scanidcs::AbstractVector{<:Integer};
        zsave::Union{Integer, AbstractVector},
        init_dz::Float64 = 5.0e-7,
        rtol::Float64 = 1.0e-6,
        max_dz::Float64 = 0.0,
        norm = Luna.RK45.weaknorm,
        twin_period::Int = 1,
        stream::Bool = true,
        extract_on_save::Union{Nothing, Bool} = nothing
    )
    setup = _build_setup_resolved(setup_args)
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    results = Dict{String, Any}[]
    scanpeaks = Dict{String, Float64}()
    HDF5.h5open(collected, "r") do f
        scanvariables = _hdf5_group(f, "scanvariables")
        τs = read(_hdf5_dataset(scanvariables, "τ"))
        # k-space-integrated spectra (Iω_win, Iω_full, ...) are in FFT-bin units which
        # scale as N⁴ at fixed R (Parseval over the transverse FFT:
        # Σₖ|Eₖ|² = N²Σₓ|Eₓ|² and Σₓ|Eₓ|² ∝ N² at fixed physical
        # energy).
        # When comparing across grid sizes,
        # rescale the recomputed values to the reference grid's units. The re-imaged
        # (real-space pixel) spectra are N-invariant and are not rescaled.
        Nnew = length(setup.xygrid.x)
        file_grid = _hdf5_group(f, "grid")
        Nref = haskey(file_grid, "x") ? length(read(_hdf5_dataset(file_grid, "x"))) : Nnew
        kscale = (Nref / Nnew)^4
        if Nref != Nnew
            xref_max = maximum(abs, read(_hdf5_dataset(file_grid, "x")))
            isapprox(xref_max, maximum(abs, setup.xygrid.x); rtol = 0.05) ||
                @warn "reference and recomputed grids differ in physical extent; " *
                "the N⁴ unit rescaling assumes fixed R and may be invalid"
            @info "grid size differs (N=$Nnew vs reference $Nref): " *
                "k-integrated datasets rescaled by (Nref/N)⁴ = $kscale before comparison"
        end
        for idx in scanidcs
            τi = τs[idx]
            onsave = something(
                extract_on_save,
                Luna.Utils.isdevice(setup.transform.Eto)
            )
            fname = (stream && !onsave) ? tempname() * "_verify.h5" : nothing
            GC.gc()
            # On a device the array library's pool is not returned by GC alone, so
            # successive points would accumulate it until the card fills.
            Luna.device_reclaim()
            devfree0 = Luna.device_memory_status()
            t0 = time()
            # invokelatest for the same reason as in `run_scan`: the GPU package may
            # have been loaded by `_build_setup_resolved` above, i.e. during this call.
            out = try
                Base.invokelatest(
                    simulate_delay_point, setup, τi;
                    zsave = zvec, init_dz = init_dz,
                    rtol = rtol, max_dz = max_dz, norm = norm,
                    twin_period = twin_period, filename = fname,
                    extract_on_save = onsave
                )
            finally
                # See the note at the streaming call above: a throwing point
                # must not leave its temp file behind.
                stream && !isnothing(fname) && rm(fname; force = true)
            end
            wall = time() - t0
            point = Dict{String, Any}(
                "scanidx" => idx, "τ" => τi,
                "wall_s" => wall,
                "maxrss_GiB" => Sys.maxrss() / 2^30
            )
            # `maxrss` is HOST memory; on a device that is only the input construction,
            # the save buffer and the runtime, so report the device side separately.
            devfree1 = Luna.device_memory_status()
            if !isnothing(devfree0) && !isnothing(devfree1)
                point["device_used_GiB"] = (devfree0[1] - devfree1[1]) / 2^30
                point["device_free_GiB"] = devfree1[1] / 2^30
            end
            out_save = Base.structdiff(out, NamedTuple{(:zsave,)})
            for (k, v) in pairs(out_save)
                ks = string(k)
                haskey(f, ks) || continue
                ref = _hdf5_dataset(f, ks)[:, :, idx]
                if !any(!iszero, ref)
                    point[ks] = NaN # reference point not (yet) computed
                    continue
                end
                size(ref) == size(v) || throw(
                    DimensionMismatch(
                        "dataset $ks has recomputed size $(size(v)) but reference size " *
                            "$(size(ref)); compare the transverse grid via /grid/x"
                    )
                )
                vn = endswith(ks, "_reimaged") ? v : v .* kscale
                absdiff = maximum(abs.(vn .- ref))
                refpeak = maximum(abs, ref)
                point[ks] = absdiff / refpeak
                # Normalising to the point's OWN peak makes a delay-scan wing — where
                # the signal beam is orders of magnitude below the τ≈0 signal — look
                # catastrophic for a difference that is negligible in the assembled
                # trace. Report the scan-wide normalisation alongside: that is the
                # quantity a FROG retrieval actually sees. Keep both — a wing point
                # disagreeing at its own scale is still worth knowing about.
                scanpeak = get!(scanpeaks, ks) do
                    _scan_peak(_hdf5_dataset(f, ks))
                end
                point[ks * "|relscan"] = scanpeak > 0 ? absdiff / scanpeak : NaN
                point[ks * "|refpeak"] = refpeak
                point[ks * "|scanpeak"] = scanpeak
            end
            scanidx = point["scanidx"]
            delay = point["τ"]
            wall_s = point["wall_s"]
            maxrss_GiB = point["maxrss_GiB"]
            @info "verified scan point" scanidx delay wall_s maxrss_GiB
            for k in sort(collect(keys(point)))
                (startswith(k, "Iω") && !occursin('|', k)) || continue
                @info "  $k: rel(own peak) = $(point[k])  " *
                    "rel(scan peak) = $(get(point, k * "|relscan", NaN))  " *
                    "peak = $(get(point, k * "|refpeak", NaN)) of " *
                    "$(get(point, k * "|scanpeak", NaN))"
            end
            push!(results, point)
        end
    end
    return results
end

"""Eager variant: wrap an already-built setup (costs nothing extra when the
setup exists anyway, e.g. in interactive use or LocalExec runs)."""
run_scan(setup::TGFROGSetup, τs::AbstractVector; kwargs...) =
    run_scan(ExistingSetup(setup), τs; kwargs...)

"""
    run_scan(setup_args::NamedTuple, τs; kwargs...)

RECOMMENDED for scan scripts: pass the [`build_setup`](@ref) keyword
arguments as a NamedTuple, e.g.

    setup_args = (; λ0, τfwhm, energy, thickness, material,
                    mask_diam, mask_spacing, λlims, beam, window,
                    R=366.0e-6, N=1024)
    run_scan(setup_args, τ; ...)

The setup is then built lazily on each process that executes scan points.
This form is robust under EVERY execution mode, including multi-worker
(`procs > 0`) queue scans: a NamedTuple of parameters serialises to the
workers by value, whereas a NAMED function defined in a script
(`make_setup() = ...`) serialises by reference and fails to deserialise on
workers (Julia ships code only for anonymous closures). The wrapping closure
here is defined inside ModelPNPS, which Luna loads on the workers.
"""
run_scan(setup_args::NamedTuple, τs::AbstractVector; kwargs...) =
    run_scan(SetupArguments(setup_args), τs; kwargs...)

"""
    _build_setup_resolved(setup_args) -> TGFROGSetup

Build the setup, resolving `arraytype` FIRST and then calling [`build_setup`](@ref)
through `Base.invokelatest`.

`Luna.resolve_arraytype(:cuda)` loads the GPU package at run time, and methods defined
by a package loaded *during* a call are not visible to that same call — Julia rejects
them as "too new to be called from this world context". Resolving first and invoking
afterwards puts the construction in a world where the array type's constructors exist.

This is why a scan script should pass `arraytype=:cuda` inside `setup_args` and let this
happen on the compute node, rather than loading the GPU package itself.
"""
function _build_setup_resolved(setup_args::NamedTuple)
    args = if haskey(setup_args, :arraytype)
        merge(setup_args, (; arraytype = Luna.resolve_arraytype(setup_args.arraytype)))
    else
        setup_args
    end
    return Base.invokelatest(build_setup; args...)::TGFROGSetup
end

# NOTE on GPU runs: pass `arraytype=:cuda` (and optionally `beamlets_on_host=true`)
# inside `setup_args`, NOT as a `run_scan` keyword. The setup is built lazily inside the
# scan closure, which only ever executes on a compute node — so the GPU package is
# loaded there and never on the submitting host, which may have no GPU at all.
