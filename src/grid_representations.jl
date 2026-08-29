# ============================================================================
# Grid-representation helpers
# ============================================================================
#
# The two grid types differ in exactly three places downstream of `build_setup`: how a
# spectrum is transformed back to time, what "intensity" means once it is there, and what
# the frequency axis looks like to a reader. Everything else — the masks, the windows, the
# delay ramp, the k-space builders, the whole extraction path — is written in terms of the
# ABSOLUTE frequency axis `grid.ω`, which both grid types carry, and needs no branching.

"""
    _to_time(grid, Eω)

The 1-D time-domain field for a spectrum on `grid`'s own frequency axis: an inverse FFT for
an `EnvGrid` (whose spectrum is FFT-ordered about the carrier) and an inverse *real* FFT for
a `RealGrid` (whose spectrum is the monotonic `rfft` half-spectrum of a real field).
"""
_to_time(grid::Grid.EnvGrid, Eω::AbstractVector) = ifft(Eω)
_to_time(grid::Grid.RealGrid, Eω::AbstractVector) = irfft(Eω, length(grid.t))

"""
    _envelope_intensity(grid, Et)

`|A(t)|²` — the ENVELOPE intensity — from whatever time-domain field `grid` produces.
On an `EnvGrid` that is `Et` itself; on a `RealGrid` the field is carrier-resolved and
the envelope is recovered through its analytic signal.

Both conventions coincide numerically: Luna builds a real-grid pulse as
`√I·cos(ω₀t)` and an envelope-grid pulse as `√I·exp(iΔωt)`, so `|A|² = I` either
way. Keeping the *envelope*
intensity in the output metadata means a consumer of a field-mode file sees the same
physical quantity in `It`/`Ito` as in every envelope file, rather than a carrier-modulated
one it would have to demodulate.
"""
_envelope_intensity(::Grid.EnvGrid, Et) = abs2.(Et)
_envelope_intensity(::Grid.RealGrid, Et) = abs2.(Maths.hilbert(Et))

"""
    _plan_1d(grid)

Forward transform plan for the 1-D reference pulse: complex for an `EnvGrid`,
real-to-complex for a `RealGrid`. `Fields.GaussField` dispatches its time-domain shape
on the grid type but
takes the plan from the caller, so the two have to be chosen together.
"""
_plan_1d(grid::Grid.EnvGrid) = plan_fft(copy(grid.t))
_plan_1d(grid::Grid.RealGrid) = plan_rfft(zeros(Float64, length(grid.t)))

"""Whether `grid` is field-resolved (real) rather than an envelope grid."""
_is_field_mode(::Grid.RealGrid) = true
_is_field_mode(::Grid.TimeGrid) = false
