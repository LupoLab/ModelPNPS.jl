# ============================================================================
# Delayed (Raman) nonlinear response
# ============================================================================

"""
    FrozenRamanPolarEnv(t, r)

Envelope Raman polarisation response with a *frozen* response kernel.

Wraps `Luna.Nonlinear.RamanPolarEnv`, precomputing the frequency-domain
response function `hω` once at construction (at unit density). Luna's own
response recomputes the time-domain kernel and its FFT on *every* call, which
is negligible for modal simulations (one call per step) but dominant for
free-space grids, where the response runs once per transverse point — ~10⁶
calls per RK stage on a 1024² grid, each re-evaluating the 13-mode
Hollenbeck–Cantrell sum over the doubled time grid. Freezing is exact here:
the density is constant (`densityfun = z -> 1`) and the `:intermediate`
response ignores its density argument entirely.

The per-call convolution below reproduces `Luna.Nonlinear.(::RamanPolar)`
line for line, minus the kernel update and with preallocated plan
applications; the test suite verifies agreement with Luna's response to
machine precision, which guards against drift in Luna's internals.
"""
struct FrozenRamanPolarEnv{TR, TI}
    R::TR    # wrapped Luna response: buffers, FFT plan, frozen hω
    IFT::TI  # cached inverse plan, so no per-call allocation
end

function FrozenRamanPolarEnv(t::AbstractVector, r)
    R = Nonlinear.RamanPolarEnv(t, r)
    R.r(R.ht, 1.0)       # kernel at unit density — never changes
    R.hω .= R.FT * R.h
    return FrozenRamanPolarEnv(R, inv(R.FT))
end

function (F::FrozenRamanPolarEnv)(out, Et, ρ)
    R = F.R
    n = size(Et, 1)
    if ndims(Et) > 1
        size(Et, 2) == 1 || throw(
            ArgumentError("FrozenRamanPolarEnv supports scalar fields only")
        )
        E = reshape(Et, n)
    else
        E = Et
    end
    Nonlinear.sqr!(R, E)               # R.E2v .= |E|²/2 (first half; rest 0)
    mul!(R.Eω2, R.FT, R.E2)
    @. R.Pω = R.hω * R.Eω2 * R.dt      # convolution on the doubled grid
    mul!(R.P, F.IFT, R.Pω)
    for i in eachindex(E)
        R.Pout[i] = ρ * E[i] * R.P[i]
    end
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        out .+= R.Pout
    end
    return nothing
end
