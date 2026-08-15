# =============================================================================
# Does the RK45 error norm agree between host and CUDA device?
#
# WHY: the production A/B showed the GPU taking systematically fewer steps than
# the CPU for the same delay point (73 vs 88 at scanidx 10, 77 vs 86 at 60).
# The step-size controller sets dt from the error estimate, so a systematically
# smaller estimate on device means the GPU runs at an effectively looser rtol
# than requested. Luna compares `weaknorm_fused` host-vs-device to rtol=1e-12,
# but only under JLArrays (CPU-backed): the real CUDA reduction is never checked
# against the host, and the hardware end-to-end test cannot catch a norm scale
# error because both runs would still be accurate, just at different step counts.
#
# This measures the norms directly, rather than inferring a ratio from step
# counts — unreliable here because 16 forced `step_on` saves keep resetting the
# controller.
#
# THE TEST MUST REPRODUCE THE CANCELLATION. Dormand-Prince's `errest` coefficients
# sum to zero (errest = b - b*, and both weight sets sum to 1), so with identical
# stages yerr vanishes exactly: the error estimate is built entirely from the
# tiny differences BETWEEN stages. Independent random ks would give a yerr with
# no cancellation at all and would agree between backends to rounding no matter
# what the real runs do. So the stages here are a common base field plus a
# relative perturbation `delta`, which is swept: any backend disagreement driven
# by cancellation (FMA contraction in the device kernel, or the sequential host
# sum versus the device tree reduction) grows as delta shrinks.
#
# The base field also carries a ~10-decade dynamic range, as the real state does
# between the pump beamlets and the FWM signal.
#
# Run on a GPU node (~2.5 GB host and ~2.5 GB device at the default size):
#   julia --project=$HOME/sharedscratch/cuda/env check_device_norm.jl [Nw] [N]
#
# INTERPRETING THE RESULT
#   ratios ~1 to 1e-12 at every delta
#       -> the norm is fine; the step-count difference is two valid trajectories.
#   ratio drifting from 1 as delta shrinks
#       -> cancellation-driven, i.e. real. Step count scales as ratio^(1/5), so
#          the observed ~15% step deficit corresponds to a ratio near 0.45.
#   host fused vs host materialised also disagreeing
#       -> the issue is not device-specific.
# =============================================================================

import Luna
import Luna: RK45, Utils
import CUDA
import Random
import Printf: @printf

Nw = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
N  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 256
sz = (Nw, N, N)

# The stepper fields the norm reads. Mirrors the stand-in in Luna/test/test_device.jl.
mutable struct FakeStepper{T, N2}
    y::T
    yn::T
    ks::NTuple{7, T}
    yerr::Union{Nothing, T}
    dt::Float64
    rtol::Float64
    atol::Float64
    norm::N2
end

const DT   = 1.7e-4
const RTOL = 1e-7   # production 04
const ATOL = 1e-10

rng = Random.Xoshiro(20260815)

# Pump-to-signal dynamic range across the transverse axis, as the real field has.
dynrange = reshape(10.0 .^ range(0, -10; length=N), 1, :, 1)
base = randn(rng, ComplexF64, sz) .* dynrange

println("size = $sz  ($(prod(sz)) elements)   rtol = $RTOL")
println("stages = base .* (1 .+ delta .* noise)  [delta -> 0 reproduces the ",
        "cancellation in yerr]\n")
@printf("%-10s %-14s %-14s %-14s %-12s %-11s %-11s\n",
        "delta", "host fused", "host materl", "device", "dev/host", "reldiff", "cancel")
println("-"^92)

for delta in (1e-2, 1e-4, 1e-6, 1e-8)
    ks = ntuple(_ -> base .* (1 .+ delta .* randn(rng, ComplexF64, sz)), 7)
    y  = base
    yn = base .* (1 .+ delta .* randn(rng, ComplexF64, sz))

    host = FakeStepper(y, yn, ks, nothing, DT, RTOL, ATOL, RK45.weaknorm)
    dev  = FakeStepper(CUDA.CuArray(y), CUDA.CuArray(yn), map(CUDA.CuArray, ks),
                       nothing, DT, RTOL, ATOL, RK45.weaknorm)
    @assert Utils.backend(dev.y) isa Utils.DeviceBackend

    eh = RK45.weaknorm_fused(host)
    ed = RK45.weaknorm_fused(dev)

    # The materialised reference: what a custom (non-fused) norm would receive.
    # Same expression as Luna's own test.
    yerr = @. DT*(ks[1]*RK45.errest[1] + ks[3]*RK45.errest[3] +
                  ks[4]*RK45.errest[4] + ks[5]*RK45.errest[5] +
                  ks[6]*RK45.errest[6] + ks[7]*RK45.errest[7])
    em = RK45.weaknorm(yerr, y, yn, RTOL, ATOL)

    # How much cancellation yerr actually underwent: sum of term magnitudes over
    # the magnitude of their sum. 1 = none; large = severe.
    terms = @. DT*(abs(ks[1]*RK45.errest[1]) + abs(ks[3]*RK45.errest[3]) +
                   abs(ks[4]*RK45.errest[4]) + abs(ks[5]*RK45.errest[5]) +
                   abs(ks[6]*RK45.errest[6]) + abs(ks[7]*RK45.errest[7]))
    cancel = sum(terms)/sum(abs, yerr)

    # reldiff as well as the ratio: a 1e-12 agreement is indistinguishable from
    # 1.000000000 in the ratio column, and the two mean very different things.
    @printf("%-10.0e %-14.6e %-14.6e %-14.6e %-12.9f %-11.2e %-11.2e\n",
            delta, eh, em, ed, ed/eh, abs(ed-eh)/abs(eh), cancel)

    ks = nothing; yerr = nothing; terms = nothing; dev = nothing
    GC.gc(); Luna.device_reclaim()
end

println("\nStep count scales as (dev/host)^(1/5); the production A/B showed ~0.85,")
println("which would need a norm ratio near 0.45.")
