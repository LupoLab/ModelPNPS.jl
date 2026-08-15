# =============================================================================
# Print the per-delay-point peak of each trace dataset in a collected scan file.
#
# This is the context needed to read a `verify_against_collected` result. That
# harness normalises each point's difference to THAT POINT's peak, which is the
# strict statement about the code path — but a delay-scan wing carries a signal
# orders of magnitude below the τ≈0 signal, so a difference that is invisible in
# the assembled FROG trace can still be a large fraction of a wing point's own
# peak. This script shows which regime a given scan index is in.
#
# Reads one point at a time, so it is safe against a scan that is still writing
# and cheap regardless of file size (seconds, no GPU, no propagation).
#
# Usage:
#   julia --project=<env> scan_peaks.jl <collected.h5> [dataset] [idx1,idx2,...]
#
#   dataset   trace dataset to report (default Iω_win)
#   idx...    if given, ONLY these indices are printed in full detail; the
#             scan-wide statistics still cover every computed point.
# =============================================================================

import HDF5

length(ARGS) >= 1 || error("usage: scan_peaks.jl <collected.h5> [dataset] [idx1,idx2,...]")
collected = ARGS[1]
dsname    = length(ARGS) >= 2 ? ARGS[2] : "Iω_win"
highlight = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : Int[]

HDF5.h5open(collected, "r") do f
    if !haskey(f, dsname)
        avail = join(keys(f), ", ")
        error("no dataset `$dsname` in $collected; available: $avail")
    end
    τs = read(f["scanvariables"]["τ"])
    dset = f[dsname]
    npts = size(dset, 3)

    peaks = fill(NaN, npts)
    for i in 1:npts
        s = dset[:, :, i]
        any(!iszero, s) || continue # not yet computed by a running scan
        peaks[i] = maximum(abs, s)
    end

    done = findall(!isnan, peaks)
    isempty(done) && error("no computed points in $dsname")
    scanpeak = maximum(peaks[done])
    imax = done[argmax(peaks[done])]

    println("$dsname: $(length(done))/$npts points computed")
    println("scan-wide peak = $scanpeak at idx $imax (τ = ",
            round(τs[imax]*1e15; digits=3), " fs)\n")

    println(rpad("idx", 6), rpad("τ [fs]", 12), rpad("peak", 14), "peak/scanpeak")
    show = isempty(highlight) ? done : highlight
    for i in show
        isnan(peaks[i]) && (println(rpad(i, 6), rpad(round(τs[i]*1e15; digits=3), 12),
                                    "not computed"); continue)
        println(rpad(i, 6), rpad(round(τs[i]*1e15; digits=3), 12),
                rpad(round(peaks[i]; sigdigits=4), 14),
                round(peaks[i]/scanpeak; sigdigits=4))
    end
end
