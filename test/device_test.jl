# =============================================================================
# Tests for ModelPNPS's device (GPU) support.
#
# As in Luna's test_device.jl, the device code paths are exercised with
# JLArrays.JLArray — a CPU-backed AbstractGPUArray which enforces the same
# no-scalar-indexing contract as CuArray — so no GPU hardware is needed. Skipped when
# JLArrays is unavailable.
#
# What this cannot check (and hardware tests must): CUDA code generation, cuFFT
# semantics, and the actual device memory footprint.
# =============================================================================
using Test
using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna: Grid, PhysData, Utils, RK45
import Random
import Adapt
import FFTW
import AbstractFFTs
import LinearAlgebra

have_jlarrays = try
    @eval import JLArrays
    true
catch
    false
end

@testset "Device support" begin

@testset "quadrant_ranges" begin
    # The signal quadrant must be exactly the second half of each k axis: that is what
    # lets the device norm use strided views instead of a mask.
    xygrid = Grid.FreeGrid(400e-6, 16)
    sig_quad = BitMatrix((xygrid.ky .< 0) .& (xygrid.kx .< 0)')
    ys, xs = TS.quadrant_ranges(sig_quad)
    @test ys == 9:16
    @test xs == 9:16
    @test all(sig_quad[ys, xs])
    @test count(sig_quad) == length(ys)*length(xs)

    # A mask that is not that rectangle must be refused, not silently mis-handled: the
    # solver's error control depends on which points count as "signal".
    bad = copy(sig_quad)
    bad[1, 1] = true
    @test_throws ErrorException TS.quadrant_ranges(bad)
    @test_throws ErrorException TS.quadrant_ranges(trues(5, 6))  # odd size
end

if have_jlarrays
    JLArray = JLArrays.JLArray

    # Minimal AbstractFFTs plans for JLArray (JLArrays provides no FFTs). Deliberately
    # no ldiv!, so the generic inv -> ScaledPlan route is taken, as on a real device.
    mutable struct JLPlan{T, N, P} <: AbstractFFTs.Plan{T}
        hp::P
        sz::NTuple{N, Int}
        dims::Any
        pinv::AbstractFFTs.ScaledPlan
        JLPlan{T, N, P}(hp, sz, dims) where {T, N, P} = new{T, N, P}(hp, sz, dims)
    end
    JLPlan(hp, sz::NTuple{N, Int}, dims) where {N} =
        JLPlan{ComplexF64, N, typeof(hp)}(hp, sz, dims)
    Base.size(p::JLPlan) = p.sz
    Base.eltype(::JLPlan{T}) where {T} = T
    AbstractFFTs.plan_fft(x::JLArrays.JLArray, dims) =
        JLPlan(FFTW.plan_fft(Array(x), dims), size(x), dims)
    AbstractFFTs.plan_inv(p::JLPlan) =
        AbstractFFTs.ScaledPlan(JLPlan(inv(p.hp).p, p.sz, p.dims),
                                AbstractFFTs.normalization(Float64, p.sz, p.dims))
    Base.:*(p::JLPlan, x::JLArrays.JLArray) = JLArrays.JLArray(p.hp * Array(x))
    LinearAlgebra.mul!(y::JLArrays.JLArray, p::JLPlan, x::JLArrays.JLArray) =
        (copyto!(y, p.hp * Array(x)); y)

    mutable struct JLPlanInplace{T, N, P} <: AbstractFFTs.Plan{T}
        hp::P
        sz::NTuple{N, Int}
        dims::Any
        pinv::AbstractFFTs.ScaledPlan
        JLPlanInplace{T, N, P}(hp, sz, dims) where {T, N, P} = new{T, N, P}(hp, sz, dims)
    end
    JLPlanInplace(hp, sz::NTuple{N, Int}, dims) where {N} =
        JLPlanInplace{ComplexF64, N, typeof(hp)}(hp, sz, dims)
    Base.size(p::JLPlanInplace) = p.sz
    Base.eltype(::JLPlanInplace{T}) where {T} = T
    AbstractFFTs.plan_fft!(x::JLArrays.JLArray, dims) =
        JLPlanInplace(FFTW.plan_fft(Array(x), dims), size(x), dims)
    AbstractFFTs.plan_ifft!(x::JLArrays.JLArray, dims) =
        JLPlanInplace(FFTW.plan_ifft(Array(x), dims), size(x), dims)
    Base.:*(p::JLPlanInplace, x::JLArrays.JLArray) = (copyto!(x, p.hp * Array(x)); x)

    # Shared small setup, matching the CPU smoke test's grid
    beam = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    base_kwargs = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                     thickness=1e-6, material=:SiO2,
                     mask_diam=1.0e-3, mask_spacing=0.5e-3,
                     beam, window,
                     trange=20e-15, λlims=(200e-9, 400e-9),
                     R=40e-6, N=16)

    @testset "build_setup on device" begin
        sh = TS.build_setup(; base_kwargs...)
        sd = TS.build_setup(; base_kwargs..., arraytype=JLArray)

        # the propagating buffers and beamlets follow the array type
        @test Utils.backend(sd.transform.Eto) isa Utils.DeviceBackend
        @test Utils.backend(sd.Eωk_g12) isa Utils.DeviceBackend
        @test Utils.backend(sd.Eωk_t_base) isa Utils.DeviceBackend
        @test Utils.backend(sh.Eωk_g12) isa Utils.CPUBackend

        # ...while the host-side metadata and windows stay on the host
        @test sd.window_array isa Array
        @test sd.Eω isa Vector{ComplexF64}

        # beamlets_on_host trades two resident device fields for one upload per point
        sb = TS.build_setup(; base_kwargs..., arraytype=JLArray, beamlets_on_host=true)
        @test Utils.backend(sb.Eωk_g12) isa Utils.CPUBackend
        @test Utils.backend(sb.transform.Eto) isa Utils.DeviceBackend

        # delayed_input must produce a device array in every case, since the solver
        # adopts it as a working buffer (preserve_input=false)
        for s in (sd, sb)
            Eωk = TS.delayed_input(s, 0.7e-15)
            @test Utils.backend(Eωk) isa Utils.DeviceBackend
        end
        # and agree with the host version
        @test isapprox(Array(TS.delayed_input(sd, 0.7e-15)),
                       TS.delayed_input(sh, 0.7e-15); rtol=1e-12)
        @test isapprox(Array(TS.delayed_input(sb, 0.7e-15)),
                       TS.delayed_input(sh, 0.7e-15); rtol=1e-12)
    end

    @testset "SignalQuadrantNorm on device" begin
        sh = TS.build_setup(; base_kwargs...)
        qn = TS.signal_quadrant_norm(sh)

        # Stand-in for the stepper: the norm only reads these fields.
        mutable struct FakeStepper{T, N}
            y::T
            yn::T
            ks::NTuple{7, T}
            yerr::Union{Nothing, T}
            dt::Float64
            rtol::Float64
            atol::Float64
            norm::N
        end

        rng = Random.Xoshiro(31415)
        sz = (length(sh.grid.ω), 16, 16)
        mk() = randn(rng, ComplexF64, sz)
        y, yn = mk(), mk()
        ks = ntuple(_ -> mk(), 7)
        host = FakeStepper(y, yn, ks, nothing, 1.3e-7, 1e-8, 1e-10, qn)
        dev = FakeStepper(JLArray(y), JLArray(yn), map(JLArray, ks), nothing,
                          host.dt, host.rtol, host.atol, qn)

        f = RK45.fused_errnorm(qn)
        eh = f(host)
        ed = f(dev)
        # Not bitwise: a parallel reduction sums in a different order
        @test isapprox(eh, ed; rtol=1e-11)
        @test dev.yerr === nothing      # the error estimate is never materialised

        # ...and both agree with the materialised reference the norm was written against
        yerr = @. host.dt*(ks[1]*RK45.errest[1] + ks[3]*RK45.errest[3] +
                           ks[4]*RK45.errest[4] + ks[5]*RK45.errest[5] +
                           ks[6]*RK45.errest[6] + ks[7]*RK45.errest[7])
        @test isapprox(eh, qn(yerr, y, yn, host.rtol, host.atol); rtol=1e-11)

        # the dimension guard still fires on a device
        bad = FakeStepper(JLArray(randn(rng, ComplexF64, sz[1], 8, 8)),
                          JLArray(randn(rng, ComplexF64, sz[1], 8, 8)),
                          ntuple(_ -> JLArray(randn(rng, ComplexF64, sz[1], 8, 8)), 7),
                          nothing, host.dt, host.rtol, host.atol, qn)
        @test_throws DimensionMismatch f(bad)
    end

    @testset "simulate_delay_point on device" begin
        # The acceptance check for the ModelPNPS side: a whole delay point, propagation
        # and extraction, on a device array versus the host.
        sh = TS.build_setup(; base_kwargs...)
        sd = TS.build_setup(; base_kwargs..., arraytype=JLArray)
        outh = TS.simulate_delay_point(sh, 0.0; nz=2, init_dz=5e-7, rtol=1e-8)
        outd = TS.simulate_delay_point(sd, 0.0; nz=2, init_dz=5e-7, rtol=1e-8)
        for k in (:Iω_win, :Iω_win_reimaged, :Iω_full)
            @test isapprox(getfield(outd, k), getfield(outh, k); rtol=1e-8)
            @test getfield(outd, k) isa Array   # extraction results are host-side
        end

        # with the Raman arm as well
        shr = TS.build_setup(; base_kwargs..., raman=true)
        sdr = TS.build_setup(; base_kwargs..., raman=true, arraytype=JLArray)
        ohr = TS.simulate_delay_point(shr, 0.0; nz=2, init_dz=5e-7, rtol=1e-8)
        odr = TS.simulate_delay_point(sdr, 0.0; nz=2, init_dz=5e-7, rtol=1e-8)
        @test isapprox(odr.Iω_win, ohr.Iω_win; rtol=1e-8)

        # the streamed (HDF5) path works too — that is what run_scan uses
        mktempdir() do dir
            fn = joinpath(dir, "dev.h5")
            outs = TS.simulate_delay_point(sd, 0.0; nz=2, init_dz=5e-7, rtol=1e-8,
                                           filename=fn)
            @test isapprox(outs.Iω_win, outh.Iω_win; rtol=1e-8)
        end
    end
else
    @info "JLArrays not available — skipping ModelPNPS device tests"
end

end
