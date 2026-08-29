# ============================================================================
# Per-delay simulation
# ============================================================================

"""
    extract_signal_spectra(Eωk, window_array, xygrid)
        -> (Iω_integrated, Iω_reimaged)

Apply a precomputed signal window to a propagated field and extract two
spectral diagnostics:

1. `Iω_integrated` — `|E|²` summed over all (ky, kx).
   Models a spectrometer collecting **all** the signal light.
2. `Iω_reimaged`   — `|E|²` at the centre pixel of the IFFT'd field.
   Models a spectrometer fed only by the on-axis re-collimated signal.

`Eωk` is either a single `(Nω, Nky, Nkx)` slice, for which both spectra are
length-`Nω` vectors, or the `(Nω, Nky, Nkx, Nz)` stack `Luna.run` produces, for
which both are `(Nω, Nz)`. The 4-D method loops over z and calls the 3-D one, so
its peak extra memory is one windowed slice rather than a second copy of the
whole stack (tens of GB at production size).

`window_array` is broadcast over `ω` (if 2-D) or matched directly (if 3-D),
and over the `Nz` z-slices in either case.
"""
function extract_signal_spectra(
        Eωk_z::AbstractArray{<:Complex, 3},
        window_array::AbstractArray{<:Real},
        xygrid::Grid.FreeGrid
    )
    Nω, Nky, Nkx = size(Eωk_z)
    if isodd(Nky) || isodd(Nkx)
        # the centre-pixel phase-sum identity below assumes even grid sizes
        return _extract_signal_spectra_fft(Eωk_z, window_array, xygrid)
    end
    w3 = ndims(window_array) == 3
    w3 || ndims(window_array) == 2 || throw(
        DimensionMismatch("window_array must be 2-D or 3-D; got $(ndims(window_array))-D")
    )
    # Single pass, no field-sized temporaries. The re-imaged spectrum needs only the
    # CENTRE pixel of ifft(Eωk_win, (2,3)): at (y, x) index (Nky÷2+1, Nkx÷2+1) the ifft
    # phase factor is exp(iπ(iky-1))·exp(iπ(ikx-1)) = (-1)^(iky+ikx-2), so the pixel is
    # the (-1)^…-weighted sum divided by Nky·Nkx — the previous full 2-D ifft (a
    # field-sized allocation and an N²logN transform per z-slice) computed exactly this.
    Iω_integrated = zeros(Float64, Nω)
    acc = zeros(ComplexF64, Nω)
    for ikx in 1:Nkx, iky in 1:Nky
        sgn = isodd((iky - 1) + (ikx - 1)) ? -1.0 : 1.0
        for iω in 1:Nω
            w = w3 ? window_array[iω, iky, ikx] : window_array[iky, ikx]
            Ew = w * Eωk_z[iω, iky, ikx]
            Iω_integrated[iω] += abs2(Ew)
            acc[iω] += sgn * Ew
        end
    end
    Iω_reimaged = abs2.(acc ./ (Nky * Nkx))
    return Iω_integrated, Iω_reimaged
end

# Reference implementation via the full 2-D ifft; used as fallback for odd-sized grids
# and by the test suite to validate the one-pass version.
function _extract_signal_spectra_fft(
        Eωk_z::AbstractArray{<:Complex, 3},
        window_array::AbstractArray{<:Real},
        xygrid::Grid.FreeGrid
    )
    if ndims(window_array) == 2
        Eωk_win = Eωk_z .* reshape(window_array, (1, size(window_array)...))
    elseif ndims(window_array) == 3
        Eωk_win = Eωk_z .* window_array
    else
        throw(
            DimensionMismatch(
                "window_array must be 2-D or 3-D; got $(ndims(window_array))-D"
            )
        )
    end
    Eωxy_win = ifft(Eωk_win, (2, 3))
    Iω_reimaged = abs2.(
        Eωxy_win[
            :, length(xygrid.y) ÷ 2 + 1,
            length(xygrid.x) ÷ 2 + 1,
        ]
    )
    Iω_integrated = dropdims(sum(abs2.(Eωk_win); dims = (2, 3)); dims = (2, 3))
    return Iω_integrated, Iω_reimaged
end

function extract_signal_spectra(
        Eωk_out::AbstractArray{<:Complex, 4},
        window_array::AbstractArray{<:Real},
        xygrid::Grid.FreeGrid
    )
    # Slice-wise loop: peak memory is one windowed slice, not a second copy
    # of the whole 4-D stack (which at production size is tens of GB).
    nz = size(Eωk_out, 4)
    Nω = size(Eωk_out, 1)
    Iω_integrated = Array{Float64}(undef, Nω, nz)
    Iω_reimaged = Array{Float64}(undef, Nω, nz)
    for iz in 1:nz
        a, b = extract_signal_spectra(
            view(Eωk_out, :, :, :, iz),
            window_array, xygrid
        )
        Iω_integrated[:, iz] = a
        Iω_reimaged[:, iz] = b
    end
    return Iω_integrated, Iω_reimaged
end

"""
    _quadrant_spectrum!(out, Ez, quad)

Accumulate `|E|²` over the transverse points selected by the `(Nky, Nkx)` mask `quad`
into the length-`Nω` vector `out`, without materialising any field-sized temporary
(the previous broadcast allocated two per z-slice).
"""
function _quadrant_spectrum!(out, Ez::AbstractArray{<:Complex, 3}, quad::AbstractMatrix)
    fill!(out, 0.0)
    for ikx in axes(Ez, 3), iky in axes(Ez, 2)
        quad[iky, ikx] || continue
        for iω in axes(Ez, 1)
            out[iω] += abs2(Ez[iω, iky, ikx])
        end
    end
    return out
end

# ============================================================================
# Save-time extraction — reduce each z-slice as it is produced
# ============================================================================
#
# The default route saves every z-slice in full and reduces the stack afterwards. At
# production shapes that is 16 × 2.25 GiB written to a temp file and read straight back
# by the same process, to produce three (Nω, nz) arrays — 32 KB. On the CPU that cost was
# ~3 % of a delay point; after the GPU port made the propagation 12× faster it is 14-18 %,
# and 72 GiB of filesystem traffic per point across a many-instance campaign.
#
# `TraceExtractOutput` reduces each slice at the moment it is saved, so no slice is ever
# stored. On a device the reduction runs on the device array itself: `HostOutput` passes
# `y` through untouched when the handler does not need a host copy (`needs_host_y`, false
# by default), and because `simulate_delay_point` pins `step_on` to the save positions,
# `Luna.RK45.interpolate` snaps to the step endpoint and returns that same array. The
# guard below checks that rather than assuming it.
#
# NOTE this needs NO changes in Luna: the handler satisfies the existing `Output`
# interface and opts out of the device-statistics check with a `nostats_only` method on
# its own type.

"""
    _signed_window(w, arraytype, Nω) -> array

The signal window with the re-imaging sign pattern `(-1)^((iky-1)+(ikx-1))` folded in,
on `arraytype`.

Folding the sign into the window lets **one** array serve both reductions of
[`extract_signal_spectra`](@ref): the signed sum needs it, and the intensity sum is
unaffected because `|±w·E|² == |w·E|²`. Carrying a separate sign array instead would
cost a second field-sized device array, and could not be broadcast into the reduction anyway
(see the shape note in `_sqn_fused`).

A 2-D window is expanded to 3-D: the reduction takes `mapreduce` over two arrays, which
does not broadcast shapes.
"""
function _signed_window(w::AbstractArray{<:Real}, arraytype, Nω::Integer)
    nd = ndims(w)
    nd == 2 || nd == 3 || throw(
        DimensionMismatch("window_array must be 2-D or 3-D; got $nd-D")
    )
    Nky, Nkx = nd == 3 ? size(w)[2:3] : size(w)
    sgn = [isodd((iky - 1) + (ikx - 1)) ? -1.0 : 1.0 for iky in 1:Nky, ikx in 1:Nkx]
    host = if nd == 3
        w .* reshape(sgn, 1, Nky, Nkx)
    else
        repeat(reshape(w .* sgn, 1, Nky, Nkx), Nω, 1, 1)
    end
    return arraytype <: Array ? host : Adapt.adapt(arraytype, host)
end

"""
    _extract_slice_device!(Iint, Ireim, Ifull, Ez, wsgn, quadrng)

Reduce one `(Nω, Nky, Nkx)` **device** slice into the three per-ω spectra, writing into
column views of the host result arrays. Three reductions over dims `(2, 3)` and one small
copy back per slice; the field itself never moves.

Mathematically identical to [`extract_signal_spectra`](@ref) plus
[`_quadrant_spectrum!`](@ref), which is what a slice arriving on the host uses instead.
The sums are formed in a different order here, so results agree to rounding rather than
bitwise — the standard everywhere else on the device path.

Both operands of the two windowed reductions have the SAME shape: `mapreduce` over
several arrays does not broadcast (Base throws `DimensionMismatch`, and a GPU backend may
silently compute something else), which is why the re-imaging sign lives inside `wsgn`
rather than in a `(1, Nky, Nkx)` array of its own. The quadrant sum is a single-array
reduction over a strided view, so it reads only the quadrant instead of masking the whole
field.
"""
function _extract_slice_device!(Iint, Ireim, Ifull, Ez, wsgn, quadrng)
    ys, xs = quadrng
    Nky, Nkx = size(Ez, 2), size(Ez, 3)
    sfull = mapreduce(abs2, +, view(Ez, :, ys, xs); dims = (2, 3))
    sint = mapreduce(+, Ez, wsgn; dims = (2, 3)) do e, w
        abs2(w * e)
    end
    sacc = mapreduce(+, Ez, wsgn; dims = (2, 3)) do e, w
        w * e
    end
    copyto!(Ifull, dropdims(Array(sfull); dims = (2, 3)))
    copyto!(Iint, dropdims(Array(sint); dims = (2, 3)))
    Ireim .= abs2.(dropdims(Array(sacc); dims = (2, 3)) ./ (Nky * Nkx))
    return nothing
end

"""
    TraceExtractOutput(setup, zvec, arraytype)

A `Luna` output handler that reduces each saved z-slice to the trace spectra immediately
and keeps only the results, so the full field is never stored, streamed or transferred.

Satisfies the `Output` interface `Luna.run` uses (the save call, `willsave`, metadata
calls, and the generic `check_cache` fallback). Metadata is discarded: `ModelPNPS` builds
its own from `setup.combined_grid`, and the temp file this replaces was thrown away too.
"""
mutable struct TraceExtractOutput{S, W, HW, G}
    save_cond::S
    saved::Int
    nz::Int
    zs::Vector{Float64}
    wsgn::W               # signed windows on the state's array type; empty on the host
    hwin::HW              # the plain host windows — ALIASES `setup.window_array`, no copy
    sig_quad::BitMatrix
    quadrng::Tuple{UnitRange{Int}, UnitRange{Int}}
    xygrid::G
    Iω_w::Vector{Matrix{Float64}}
    Iω_r::Vector{Matrix{Float64}}
    Iω_full::Matrix{Float64}
end

function TraceExtractOutput(setup::TGFROGSetup, zvec::AbstractVector, arraytype)
    Nω = length(setup.grid.ω)
    nz = length(zvec)
    single = setup.window isa AbstractSignalWindow
    wins = single ? (setup.window_array,) : Tuple(setup.window_array)
    sig_quad = BitMatrix((setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)')
    # The signed window exists only to serve the device reduction; on the host the
    # original kernels take the plain window, so building it would be a field-sized
    # array for nothing.
    wsgn = if arraytype <: Array
        Any[]
    else
        Any[_signed_window(w, arraytype, Nω) for w in wins]
    end
    return TraceExtractOutput(
        Output.GridCondition(zvec, nz), 0, nz, Float64[],
        wsgn, wins, sig_quad,
        quadrant_ranges(sig_quad),   # asserts the quadrant is that rectangle
        setup.xygrid,
        [zeros(Float64, Nω, nz) for _ in wins],
        [zeros(Float64, Nω, nz) for _ in wins],
        zeros(Float64, Nω, nz)
    )
end

# `Output.foreach_save` runs the body once per data point the save condition asks for and
# advances `o.saved`; the loop itself is Luna's, so this handler cannot drift from the
# built-in ones. `Ez` is the solver's own array — `Luna.needs_host_save` above declines
# the copy, and `RK45.interpolate` returns the stepped solution itself at a step boundary
# rather than evaluating the dense-output polynomial.
function (o::TraceExtractOutput)(y, t, dt, yfun)
    return Output.foreach_save(o, y, t, dt, yfun) do iz, ts, Ez
        iz <= o.nz || throw(
            AssertionError(
                "TraceExtractOutput received save $iz but was built for $(o.nz) saves"
            )
        )
        _reduce_slice!(o, Ez, iz)
        push!(o.zs, ts)
    end
end

"""
    _reduce_slice!(o::TraceExtractOutput, Ez, iz)

Reduce one saved slice into column `iz`, routing on where the slice actually is.

A device propagation delivers every slice on the device — including `z = 0`, which is the
step *start* rather than an endpoint and so comes through the interpolant, because
`Luna.needs_host_save` declines the copy `HostOutput` would otherwise make.

The host branch therefore serves a **host** propagation (`extract_on_save=true` on the
CPU), and any save that is genuinely interpolated. It goes through the original
[`extract_signal_spectra`](@ref)/[`_quadrant_spectrum!`](@ref) kernels against the plain
host window the setup already holds: no extra memory, no transfer, and the host result is
bit-identical to the save-the-stack route by construction rather than by a parallel
implementation that could drift.
"""
function _reduce_slice!(o::TraceExtractOutput, Ez, iz)
    if Luna.Utils.isdevice(Ez)
        for iw in eachindex(o.wsgn)
            _extract_slice_device!(
                view(o.Iω_w[iw], :, iz), view(o.Iω_r[iw], :, iz),
                view(o.Iω_full, :, iz), Ez, o.wsgn[iw], o.quadrng
            )
        end
    else
        _quadrant_spectrum!(view(o.Iω_full, :, iz), Ez, o.sig_quad)
        for iw in eachindex(o.hwin)
            a, b = extract_signal_spectra(Ez, o.hwin[iw], o.xygrid)
            o.Iω_w[iw][:, iz] .= a
            o.Iω_r[iw][:, iz] .= b
        end
    end
    return nothing
end

# Metadata calls from `Luna.run` (grid, simulation type). Discarded — see the docstring.
(o::TraceExtractOutput)(d; kwargs...) = nothing
(o::TraceExtractOutput)(key::AbstractString, val; kwargs...) = nothing

Output.willsave(o::TraceExtractOutput, y, t, dt) = first(o.save_cond(y, t, dt, o.saved))

# `Luna.run` refuses a device propagation with statistics, because a statistics function
# is called with the state on every step and would force a full transfer each time. This
# handler collects none, so it opts out on its own type.
Luna.nostats_only(::TraceExtractOutput) = true
# `HostOutput` must not copy `y` per step for us (already the generic default, but the
# whole design depends on it)...
Luna.needs_host_y(::TraceExtractOutput) = false
# ...nor allocate a host buffer and copy each SAVE into it: we reduce on the device and
# keep only (Nω,) results. Declining this hands over the solver's own array untouched,
# which also removes the one slice that used to come back on the host (`z = 0`, reached
# through the interpolant because it is the step start rather than an endpoint).
Luna.needs_host_save(::TraceExtractOutput) = false

"""
    _trace_results(setup, o::TraceExtractOutput) -> NamedTuple

The same `NamedTuple` the save-the-stack route returns, assembled from an extraction
handler. Kept next to that route's assembly block so the two cannot drift.
"""
function _trace_results(setup::TGFROGSetup, o::TraceExtractOutput)
    if setup.window isa AbstractSignalWindow
        return (;
            Iω_win = o.Iω_w[1], Iω_win_reimaged = o.Iω_r[1], Iω_full = o.Iω_full,
            zsave = copy(o.zs),
        )
    end
    pairs_kv = Pair{Symbol, Any}[]
    for (iw, suf) in enumerate(setup.window_suffix)
        push!(pairs_kv, Symbol("Iω_win" * suf) => o.Iω_w[iw])
        push!(pairs_kv, Symbol("Iω_win" * suf * "_reimaged") => o.Iω_r[iw])
    end
    push!(pairs_kv, :Iω_full => o.Iω_full)
    push!(pairs_kv, :zsave => copy(o.zs))
    return NamedTuple(pairs_kv)
end

"""
    _resolve_zsave(zsave, zmax) -> Vector{Float64}

Resolve the `zsave` propagation-snapshot specification into a validated, sorted
vector of z positions [m] at which the field is saved during propagation.

- `zsave::Integer` — a uniform grid of `zsave` points over `[0, zmax]`
  (`range(0, zmax, zsave)`), reproducing the legacy `nz` behaviour exactly
  (including the entrance slice at `z=0` and the exit slice at `z=zmax`).
- `zsave::AbstractVector` — explicit material thicknesses [m]. Must be strictly
  increasing, all `>= 0`, and all `<= zmax`. `zmax` is appended if not already
  present (within `rtol=1e-12`) so the full-thickness ("`:end`") slice always
  exists.

Because the propagation is a forward-marching integrator with z-independent
dynamics, the field saved at an intermediate `z` is identical to a dedicated run
of thickness `z`, so a single `zmax` run yields every shorter thickness for free.

The function is idempotent: re-resolving an already-resolved vector (which the
integer path produces *with* an entrance slice at `z=0`) returns it unchanged,
so it is safe to call more than once on the same grid.
"""
function _resolve_zsave(zsave::Integer, zmax::Real)
    zsave >= 2 || throw(ArgumentError("integer zsave must be ≥ 2, got $zsave"))
    return collect(range(0.0, zmax, zsave))
end

function _resolve_zsave(zsave::AbstractVector, zmax::Real)
    v = collect(Float64, zsave)
    isempty(v) && throw(ArgumentError("zsave vector must be non-empty"))
    issorted(v) && allunique(v) ||
        throw(ArgumentError("zsave must be strictly increasing, got $v"))
    all(>=(0.0), v) ||
        throw(ArgumentError("all zsave positions must be ≥ 0, got $v"))
    vmax = maximum(v)
    vmax <= zmax || throw(
        ArgumentError(
            "zsave position $vmax exceeds the propagation distance zmax=$zmax"
        )
    )
    if !isapprox(v[end], zmax; rtol = 1.0e-12)
        push!(v, zmax)
    end
    return v
end

"""
Callable RK45 error norm built by [`signal_quadrant_norm`](@ref).

Holds the signal-quadrant mask, the relative floor, and — on a device — a cached
0/1 indicator of the quadrant on the solver's own array type.
"""
mutable struct SignalQuadrantNorm
    sig_quad::BitMatrix
    floor_rel::Float64
    # 0/1 quadrant indicator of shape (1, Nky, Nkx) on the solver's array type, built on
    # first use by the device path and cached. See `_sqn_devmask!`.
    devmask::Any
end

SignalQuadrantNorm(sig_quad, floor_rel) = SignalQuadrantNorm(sig_quad, floor_rel, nothing)

"""
    signal_quadrant_norm(setup::TGFROGSetup; floor_rel=1e-6)

Region-relative RK45 error norm for weak-signal accuracy at moderate `rtol`.

The default `Luna.RK45.weaknorm` measures the step error relative to the norm
of the WHOLE field, which the three pump beamlets dominate. The FWM signal is
orders of magnitude weaker, so the stepper's error budget — concentrated on
the fastest-evolving (signal) components — allows a per-step signal error of
order `rtol × ‖pump‖/‖signal‖` *relative to the signal*: with `rtol = 1e-6`
the collected signal carries measured solver errors of 0.1–1% mid-slab,
growing to ~10% at 40 µm (see `90_solver_accuracy_test.jl`). Brute-forcing
`rtol = 1e-8` fixes this at ~4× the step count.

This norm instead measures relative error separately in the signal k-space
quadrant (`kx < 0, ky < 0` — the same quadrant `Iω_full` integrates) and in
the rest of the field, and returns the larger: `rtol` then controls the
signal's OWN relative error directly, recovering weak-signal accuracy at
close to the default-`rtol` step count.

While the signal quadrant is still (nearly) empty its error is measured
against a floor of `floor_rel × ‖rest‖` (never below `atol`), so early steps
are not throttled by a 0/0 relative error; once the signal exceeds that
fraction of the pump field the relative control takes over.

Pass the result to [`simulate_delay_point`](@ref) / [`run_scan`](@ref) via
their `norm` keyword. Validate a new `(rtol, floor_rel)` choice against a
tight-`rtol` reference before production use (the pass criterion used here:
every z-slice of `Iω_win` within 1e-3 relative of an `rtol = 1e-8` run).
"""
function signal_quadrant_norm(setup::TGFROGSetup; floor_rel::Float64 = 1.0e-6)
    # Same quadrant the Iω_full collection integrates (see the comment there).
    sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
    return SignalQuadrantNorm(BitMatrix(sig_quad), floor_rel)
end

"""
    quadrant_ranges(sig_quad) -> (ys, xs)

The signal quadrant as a pair of index ranges. In FFT ordering the negative half of each
k axis is exactly the second half of its index range, so the mask is a dense rectangle —
which lets the device norm use strided views instead of a boolean mask (a `BitMatrix`
cannot enter a device kernel, and a masked reduction would need a gather).

Throws if the mask is not that rectangle, so a future change to the k-space layout
cannot silently corrupt the solver's error control.
"""
function quadrant_ranges(sig_quad::AbstractMatrix{Bool})
    Ny, Nx = size(sig_quad)
    (iseven(Ny) && iseven(Nx)) || throw(
        ArgumentError("signal_quadrant_norm needs even grid sizes; got ($Ny, $Nx)")
    )
    ys = (Ny ÷ 2 + 1):Ny
    xs = (Nx ÷ 2 + 1):Nx
    (all(sig_quad[ys, xs]) && count(sig_quad) == length(ys) * length(xs)) || throw(
        ArgumentError(
            "the signal quadrant is not the expected index rectangle; the k-space " *
                "ordering may have changed"
        )
    )
    return ys, xs
end

function (n::SignalQuadrantNorm)(yerr, y, yn, rtol, atol)
    sig_quad = n.sig_quad
    floor_rel = n.floor_rel
    Ny, Nx = size(sig_quad)
    begin
        size(y, 2) == Ny && size(y, 3) == Nx || throw(
            DimensionMismatch(
                "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
                    "the solver state is $(size(y))"
            )
        )
        s_y_s = 0.0; s_yn_s = 0.0; s_e_s = 0.0
        s_y_r = 0.0; s_yn_r = 0.0; s_e_r = 0.0
        Nω = size(y, 1)
        for ix in 1:Nx, iy in 1:Ny
            if sig_quad[iy, ix]
                for iw in 1:Nω
                    s_y_s += abs2(y[iw, iy, ix])
                    s_yn_s += abs2(yn[iw, iy, ix])
                    s_e_s += abs2(yerr[iw, iy, ix])
                end
            else
                for iw in 1:Nω
                    s_y_r += abs2(y[iw, iy, ix])
                    s_yn_r += abs2(yn[iw, iy, ix])
                    s_e_r += abs2(yerr[iw, iy, ix])
                end
            end
        end
        errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), atol)
        floor_s = max(atol, floor_rel * errwt_r)
        errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
        return max(
            sqrt(s_e_r) / (rtol * errwt_r),
            sqrt(s_e_s) / (rtol * errwt_s)
        )
    end
end

"""
Fused error metric for [`SignalQuadrantNorm`](@ref): the DP5 error estimate is computed
element-by-element on the fly from the stepper's stage arrays instead of materialising a
field-sized `yerr` array (Luna.RK45 allocates it lazily only for norms without a fused
version). Same per-element expression and accumulation order as the materialised path,
so the result is bit-identical.
"""
struct FusedSignalQuadrantNorm{N}
    norm::N
end

(f::FusedSignalQuadrantNorm)(stepper) =
    _sqn_fused(Luna.Utils.backend(stepper.y), f.norm, stepper)

Luna.RK45.fused_errnorm(n::SignalQuadrantNorm) = FusedSignalQuadrantNorm(n)

_sqn_fused(n::SignalQuadrantNorm, s) = _sqn_fused(Luna.Utils.backend(s.y), n, s)

"""
Device version of [`_sqn_fused`](@ref): the same six sums, computed as reductions along
ω into one partial per transverse point, which are then split by quadrant.

Every operand of a reduction here has the SAME shape. That is deliberate: `mapreduce`
over several arrays does not broadcast shapes (Base throws `DimensionMismatch`, and a
GPU backend may quietly compute something else instead), so the `(1, Nky, Nkx)` quadrant
mask cannot be folded into the reduction and is applied afterwards, to the small
per-transverse-point partials. Reducing over strided views of the quadrant would also
work in principle, but whole-array reductions are the shape the rest of Luna's device
code uses and the one best supported across backends.

Both halves are summed directly rather than one being `total - other`, so no
cancellation is involved. The error estimate is never materialised — it is formed inside
the reduction kernel.
"""
function _sqn_fused(::Luna.Utils.DeviceBackend, n::SignalQuadrantNorm, s)
    Ny, Nx = size(n.sig_quad)
    size(s.y, 2) == Ny && size(s.y, 3) == Nx || throw(
        DimensionMismatch(
            "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
                "the solver state is $(size(s.y))"
        )
    )
    q = _sqn_devmask!(n, s.y)
    dt = s.dt
    e = Luna.RK45.errest
    e1 = e[1]; e3 = e[3]; e4 = e[4]; e5 = e[5]; e6 = e[6]; e7 = e[7]
    k1, _, k3, k4, k5, k6, k7 = s.ks

    # Reduce along ω first, giving one partial sum per transverse point. Every operand
    # here has the SAME shape: `mapreduce` over several arrays does not broadcast
    # shapes (Base throws DimensionMismatch; a GPU backend may silently compute
    # something else), so mixing the (1, Nky, Nkx) mask into the reduction is invalid.
    # The error estimate is still never materialised — it is formed inside the kernel.
    Sy = mapreduce(abs2, +, s.y; dims = 1)
    Syn = mapreduce(abs2, +, s.yn; dims = 1)
    Se = mapreduce(+, k1, k3, k4, k5, k6, k7; dims = 1) do a1, a3, a4, a5, a6, a7
        err = dt * (a1 * e1 + a3 * e3 + a4 * e4 + a5 * e5 + a6 * e6 + a7 * e7)
        abs2(err)
    end

    # Split by quadrant on the small (1, Nky, Nkx) partials. Both halves are summed
    # directly rather than one being `total - other`, so no cancellation is involved.
    r = 1 .- q
    s_y_s = sum(Sy .* q); s_yn_s = sum(Syn .* q); s_e_s = sum(Se .* q)
    s_y_r = sum(Sy .* r); s_yn_r = sum(Syn .* r); s_e_r = sum(Se .* r)

    errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), s.atol)
    floor_s = max(s.atol, n.floor_rel * errwt_r)
    errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
    return max(
        sqrt(s_e_r) / (s.rtol * errwt_r),
        sqrt(s_e_s) / (s.rtol * errwt_s)
    )
end

"""
The quadrant indicator as a `(1, Nky, Nkx)` array on `y`'s array type, built once and
cached on the norm. Broadcasting it into the reduction costs one small array (8 MB even
at the largest campaign shape) and keeps the reduction over whole, contiguous arrays.

The mask is built from the same `BitMatrix` the host path uses, via
[`quadrant_ranges`](@ref) — so its rectangle assertion still guards the device path.
"""
function _sqn_devmask!(n::SignalQuadrantNorm, y)
    Ny, Nx = size(n.sig_quad)
    want = Base.typename(typeof(y)).wrapper
    if isnothing(n.devmask) || size(n.devmask) != (1, Ny, Nx) ||
            Base.typename(typeof(n.devmask)).wrapper !== want
        ys, xs = quadrant_ranges(n.sig_quad)   # asserts the mask is that rectangle
        host = zeros(Float64, 1, Ny, Nx)
        host[1, ys, xs] .= 1.0
        n.devmask = Adapt.adapt(want, host)
    end
    return n.devmask
end

function _sqn_fused(::Luna.Utils.CPUBackend, n::SignalQuadrantNorm, s)
    sig_quad = n.sig_quad
    floor_rel = n.floor_rel
    Ny, Nx = size(sig_quad)
    y = s.y
    yn = s.yn
    size(y, 2) == Ny && size(y, 3) == Nx || throw(
        DimensionMismatch(
            "signal_quadrant_norm built for a $(Ny)×$(Nx) transverse grid; " *
                "the solver state is $(size(y))"
        )
    )
    k1, k2, k3, k4, k5, k6, k7 = s.ks
    dt = s.dt
    errest = Luna.RK45.errest
    e1 = errest[1]; e3 = errest[3]; e4 = errest[4] # errest[2] == 0, skipped as in Luna
    e5 = errest[5]; e6 = errest[6]; e7 = errest[7]
    s_y_s = 0.0; s_yn_s = 0.0; s_e_s = 0.0
    s_y_r = 0.0; s_yn_r = 0.0; s_e_r = 0.0
    Nω = size(y, 1)
    for ix in 1:Nx, iy in 1:Ny
        if sig_quad[iy, ix]
            for iw in 1:Nω
                yerr = 0 + dt * k1[iw, iy, ix] * e1 + dt * k3[iw, iy, ix] * e3 +
                    dt * k4[iw, iy, ix] * e4 + dt * k5[iw, iy, ix] * e5 +
                    dt * k6[iw, iy, ix] * e6 + dt * k7[iw, iy, ix] * e7
                s_y_s += abs2(y[iw, iy, ix])
                s_yn_s += abs2(yn[iw, iy, ix])
                s_e_s += abs2(yerr)
            end
        else
            for iw in 1:Nω
                yerr = 0 + dt * k1[iw, iy, ix] * e1 + dt * k3[iw, iy, ix] * e3 +
                    dt * k4[iw, iy, ix] * e4 + dt * k5[iw, iy, ix] * e5 +
                    dt * k6[iw, iy, ix] * e6 + dt * k7[iw, iy, ix] * e7
                s_y_r += abs2(y[iw, iy, ix])
                s_yn_r += abs2(yn[iw, iy, ix])
                s_e_r += abs2(yerr)
            end
        end
    end
    errwt_r = max(max(sqrt(s_y_r), sqrt(s_yn_r)), s.atol)
    floor_s = max(s.atol, floor_rel * errwt_r)
    errwt_s = max(max(sqrt(s_y_s), sqrt(s_yn_s)), floor_s)
    return max(
        sqrt(s_e_r) / (s.rtol * errwt_r),
        sqrt(s_e_s) / (s.rtol * errwt_s)
    )
end

"""Provenance label for the error norm used by a scan."""
_norm_name(n::SignalQuadrantNorm) = "signal_quadrant(floor_rel=$(n.floor_rel))"
_norm_name(f) = f === Luna.RK45.weaknorm ? "weaknorm" : "custom"

"""
    delayed_input(setup, τ) -> Array{ComplexF64,3}

Coherent input field for scan delay `τ`, in the paper's **gate-delay
convention**: the stored trace ``T(ω, τ)`` has the GATE pair delayed by `+τ`
relative to the probe. Physically the probe arm carries the delay stage, so
the probe is delayed by `-τ`, which equals gating at `+τ` up to a global time
shift that the time-integrating measurement cannot see. The same-τ gate pair
stays untouched, so the smearing structure (same-τ gate pair; paper
Appendix A.2) is preserved. Files written with this convention carry
`/grid/delay_convention = "gate"` and need NO delay-axis reversal on loading
(croak's `reverse_trace` auto-detects; legacy marker-less files are reversed
as before).
"""
function delayed_input(setup::TGFROGSetup, τ::Real)
    # Single fused broadcast: one field-sized allocation instead of two
    # (apply_delay's intermediate plus the sum) — at production size the
    # difference is a 2.15 GB transient per worker, which is exactly the
    # margin that OOM-killed procs=2 tasks at the 100G quota line.
    # `ωd` is the frequency axis on the beamlets' array type, so the phase ramp is built
    # where they live and no host vector is broadcast against device data.
    phase = reshape(exp.(1im .* (-setup.ωd .* -τ)), (:, 1, 1))
    Eωk = setup.Eωk_g12 .+ setup.Eωk_t_base .* phase
    # With `beamlets_on_host` the sum is formed on the host and uploaded once here,
    # trading two resident device fields for one transfer per delay point.
    return _match_arraytype(Eωk, setup.transform)
end

function _match_arraytype(Eωk, transform)
    if Luna.Utils.isdevice(transform.Eto) && !Luna.Utils.isdevice(Eωk)
        return Adapt.adapt(typeof(transform.Eto).name.wrapper, Eωk)
    end
    return Eωk
end

"Return the same array for every requested propagation slice."
struct ConstantSlice{A}
    array::A
end

(slice::ConstantSlice)(_) = slice.array

"Return a view of a propagation slice held in memory."
struct MemorySlice{A}
    array::A
end

(slice::MemorySlice)(index) = view(slice.array, :, :, :, index)

"Read one propagation slice from an output backend."
struct OutputSlice{O}
    output::O
end

(slice::OutputSlice)(index) = slice.output["Eω", :, :, :, index]

"""
    simulate_delay_point(setup::TGFROGSetup, τi;
                         nz=2, zsave=nz, init_dz=5e-7, rtol=1e-6, max_dz=0.0,
                         norm=Luna.RK45.weaknorm, twin_period=1,
                         filename=nothing, extract_on_save=nothing,
                         skip_propagation=false)
        -> NamedTuple

Run the full per-delay computation: apply delay `τi` to the test beam,
coherently superpose the three beamlets, propagate them through the
substrate via `Luna.run`, apply each signal window and extract two
spectra per window. The returned `NamedTuple` has, for a single window,
fields `(Iω_win, Iω_win_reimaged, Iω_full)`. For a vector of windows the
suffixes recorded in `setup.window_suffix` are appended (e.g. `Iω_win_ωdep`,
`Iω_win_ωdep_reimaged`), and the single `Iω_full` is shared. All extracted
arrays have shape `(Nω, nz)`. The returned NamedTuple also carries `zsave`, the
vector of realized z save positions [m] (length `nz`) — this is metadata, not a
per-delay trace, and is excluded from the `scansave` dataset splat by [`run_scan`](@ref).

`zsave` selects the propagation snapshots. Pass an `Integer` for a uniform grid
of that many points over `[0, zmax]` (default `nz`), or a `Vector` of explicit
material thicknesses [m] (e.g. `[1e-6, 10e-6, 20e-6, 40e-6]`); `zmax` is appended
to the vector if absent. Because the field at an intermediate `z` equals a
dedicated thickness-`z` run, every shorter thickness comes free from one `zmax`
run. Peak memory scales with `nz` (the in-memory 4-D field is held per slice).

`Iω_full` is the signal beam collected in full: `|E|²` integrated over the
signal's k-space quadrant only. The propagated field holds the three strong
pump beamlets (at the g1/g2/test boxcar corners) plus the weak FWM signal at
the fourth corner; integrating over *all* of k-space would be dominated by the
pumps, so we restrict to the quadrant the signal occupies (`kx<0, ky<0`),
which captures the whole signal lobe without aperture vignetting while
excluding the pumps. `Iω_win ./ Iω_full` is therefore the exact per-(ω, τ)
collection / chromatic-vignetting efficiency of the signal aperture, so the
trace can be corrected for collection vignetting exactly rather than via a
power-law approximation. (This assumes the boxcar beams are well separated, so
pump tails leaking into the signal quadrant are negligible vs. the signal.)

The solver keywords go straight to `Luna.run`: `init_dz` [m] is the first step,
`rtol` the RK45 relative tolerance, `max_dz` [m] the step ceiling (`0.0` means
`zmax/2`), and `norm` the error norm — pass [`signal_quadrant_norm`](@ref) to
control the weak signal's OWN relative error rather than the pump-dominated whole
field's. `twin_period` is the number of accepted steps between applications of the
spectral/temporal windows; the default `1` applies them on every step, and larger
values change the result at the apodisation-leakage level.

`extract_on_save` reduces each z-slice to its spectra as it is produced, so the
field is never stored, streamed or transferred. It defaults to `true` for a device
propagation, where the saved stack costs 14–18 % of the delay point in temp-file
traffic, and to `false` on the host; the two routes are bit-identical, so passing
`true` on the host is safe.

Pass `filename` to persist the propagation to disk: the `Luna.run` then writes
to an `Output.HDF5Output` at that path instead of an in-memory
`Output.MemoryOutput`. Downstream extraction is identical either way (both
outputs index as `output["Eω"]`/`output["z"]`); `filename` is ignored when
`skip_propagation=true`.

Setting `skip_propagation=true` substitutes the input field for the
Luna output, exercising every other code path. This is used by the unit
tests to keep the suite fast and deterministic.
"""
function simulate_delay_point(
        setup::TGFROGSetup, τi::Real;
        nz::Int = 2,
        zsave::Union{Integer, AbstractVector} = nz,
        init_dz::Float64 = 5.0e-7,
        rtol::Float64 = 1.0e-6,
        max_dz::Float64 = 0.0,
        norm = Luna.RK45.weaknorm,
        twin_period::Int = 1,
        filename::Union{Nothing, AbstractString} = nothing,
        extract_on_save::Union{Nothing, Bool} = nothing,
        skip_propagation::Bool = false
    )
    # --- Resolve the propagation snapshot grid ---------------------------
    zvec = _resolve_zsave(zsave, setup.grid.zmax)
    nz_eff = length(zvec)

    # --- Build the delayed test beam and coherently superpose ------------
    Eωk_in = delayed_input(setup, τi)

    # --- Save-time extraction --------------------------------------------
    # Reduce each z-slice as it is produced rather than saving the whole field and
    # reducing afterwards; `filename`/streaming then has nothing to stream. Defaults ON
    # for a device propagation, where the saved stack costs 14-18 % of the delay point in
    # temp-file traffic, and OFF on the host, whose current route is validated
    # bit-for-bit against the production references — pass `true` to use it there too
    # (it is bit-identical: the same host kernels on the same arrays, minus a lossless
    # HDF5 round trip).
    if something(extract_on_save, Luna.Utils.isdevice(Eωk_in)) && !skip_propagation
        arrwrap = Base.typename(typeof(Eωk_in)).wrapper
        o = TraceExtractOutput(setup, zvec, arrwrap)
        Luna.run(
            Eωk_in, setup.grid, setup.linop, setup.transform, setup.FT, o;
            init_dz = init_dz, rtol = rtol, norm = norm,
            max_dz = (max_dz > 0 ? max_dz : setup.grid.zmax / 2),
            step_on = zvec, preserve_input = false, twin_period = twin_period
        )
        o.saved == nz_eff || throw(
            AssertionError(
                "save-time extraction got $(o.saved) of $nz_eff z-slices; the stepper " *
                    "did not land on every save position"
            )
        )
        return _trace_results(setup, o)
    end

    # --- Propagate (or fake the propagation for tests) -------------------
    if skip_propagation
        # Fake a (Nω, Nky, Nkx, nz) output by stacking the input nz times.
        Nω, Nky, Nkx = size(Eωk_in)
        getslice = ConstantSlice(Eωk_in)
        z_realized = copy(zvec)
    else
        save_cond = Output.GridCondition(zvec, nz_eff)
        # cache=false: the crash-resume cache would rewrite the full field-sized array
        # to the (throwaway) temp file on every save
        output = if isnothing(filename)
            Output.MemoryOutput(save_cond, "Eω", "z")
        else
            Output.HDF5Output(
                filename, save_cond, "Eω", "z", Output.nostats, false,
                nothing, false
            )
        end
        # NOTE the solver tolerance: the RK45 error is controlled relative to
        # the FULL field, which the pump beamlets dominate. The FWM signal is
        # orders of magnitude weaker (signal/pump field ratio ∝ pulse energy),
        # so its RELATIVE accuracy is ~(pump/signal) × rtol and degrades with
        # propagation distance and with decreasing energy. At the defaults
        # (rtol = 1e-6, max_dz = zmax/2, 0.1 µJ) the collected signal spectrum
        # carries ~1% (mid-slab) to ~10% (40 µm) solver-dependent error —
        # measured in FROG_paper_new/90_solver_accuracy_test.jl. Tighten to
        # rtol ≤ 1e-8 and a µm-scale max_dz for quantitative small-effect
        # studies (Raman A/B, low-energy runs, residual-floor analyses).
        # step_on = the save positions: the stepper lands on each zsave point
        # exactly, so saved slices are step endpoints rather than dense-output
        # interpolations — the interpolant shares the error norm's weak-signal
        # blind spot and interpolated saves scatter at the percent level
        # between runs (FINDINGS F14.12 twin test).
        # preserve_input=false: the solver adopts Eωk_in as a working buffer instead of
        # copying it (one field-sized allocation less at peak); Eωk_in is not used again
        # on this path
        Luna.run(
            Eωk_in, setup.grid, setup.linop, setup.transform, setup.FT,
            output; init_dz = init_dz, rtol = rtol, norm = norm,
            max_dz = (max_dz > 0 ? max_dz : setup.grid.zmax / 2),
            step_on = zvec, preserve_input = false, twin_period = twin_period
        )
        # Slice access: streamed runs read one z-slice at a time back from
        # the HDF5 file (Output.getindex opens the file per read), so the
        # full (ω, ky, kx, z) stack — tens of GB at production size — never
        # exists in memory; in-memory runs use free views into it.
        if isnothing(filename)
            Eωk_mem = output["Eω"]
            getslice = MemorySlice(Eωk_mem)
        else
            getslice = OutputSlice(output)
        end
        z_realized = output["z"]
    end

    # --- Full signal-beam collection (no aperture crop) ------------------
    # The propagated field contains the three strong pump beamlets (at the
    # g1/g2/test boxcar corners) plus the weak FWM signal at the fourth,
    # signal corner. Integrating over ALL of k-space would be dominated by the
    # pumps, so instead we integrate |E|² over the signal's k-space quadrant
    # only: by the fixed boxcar layout (g1=(d,d), g2=(d,-d), test=(-d,d)) the
    # signal sits at (kx<0, ky<0) while all three pumps occupy the other
    # quadrants. This captures the whole signal lobe with no aperture
    # vignetting and excludes the pumps, so Iω_win ./ Iω_full is the exact
    # per-(ω, τ) collection / chromatic-vignetting efficiency of the signal
    # aperture — no power-law approximation. Window-independent, computed once.
    # Shape (Nω, nz). (Luna array dims are (ω, ky, kx, z).)
    sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
    Nω = length(setup.grid.ω)
    single_window = setup.window isa AbstractSignalWindow
    wins = single_window ? (setup.window_array,) : Tuple(setup.window_array)
    Iω_full = Array{Float64}(undef, Nω, nz_eff)
    Iω_w = [Array{Float64}(undef, Nω, nz_eff) for _ in wins]
    Iω_r = [Array{Float64}(undef, Nω, nz_eff) for _ in wins]
    for iz in 1:nz_eff
        Ez = getslice(iz)
        _quadrant_spectrum!(view(Iω_full, :, iz), Ez, sig_quad)
        for (iw, arr) in enumerate(wins)
            a, b = extract_signal_spectra(Ez, arr, setup.xygrid)
            Iω_w[iw][:, iz] = a
            Iω_r[iw][:, iz] = b
        end
    end

    # --- Assemble the per-window outputs ----------------------------------
    if single_window
        return (;
            Iω_win = Iω_w[1], Iω_win_reimaged = Iω_r[1], Iω_full,
            zsave = z_realized,
        )
    else
        pairs_kv = Pair{Symbol, Any}[]
        for (iw, suf) in enumerate(setup.window_suffix)
            push!(pairs_kv, Symbol("Iω_win" * suf) => Iω_w[iw])
            push!(pairs_kv, Symbol("Iω_win" * suf * "_reimaged") => Iω_r[iw])
        end
        push!(pairs_kv, :Iω_full => Iω_full)
        push!(pairs_kv, :zsave => z_realized)
        return NamedTuple(pairs_kv)
    end
end
