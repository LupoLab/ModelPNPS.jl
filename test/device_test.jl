# =============================================================================
# Tests for ModelPNPS's device (GPU) support.
#
# As in Luna's test_device.jl, the device code paths are exercised with
# JLArrays.JLArray — a CPU-backed AbstractGPUArray which enforces the same
# no-scalar-indexing contract as CuArray — so no GPU hardware is needed. Skipped when
# JLArrays is unavailable.
#
# What this cannot check (and hardware tests must): CUDA code generation, cuFFT
# semantics, and the actual device memory footprint. Two specific blind spots that have
# already bitten, both because JLArray is CPU-backed and so tolerates what CUDA rejects:
#   * a broadcast mixing host and device operands just works under JLArrays;
#   * so does putting a non-native AbstractArray (e.g. a lazy operator) in a device
#     broadcast.
# Where those matter, assert them structurally (on types) rather than by running.
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

        # The delay phase must live wherever the beamlets do: `delayed_input` combines
        # them in one broadcast, and CUDA rejects a broadcast mixing host and device
        # operands. JLArrays does NOT — it is CPU-backed, so such a broadcast simply
        # works — which is why this is asserted structurally rather than by running it.
        wrapper(x) = Base.typename(typeof(x)).wrapper
        for s in (sh, sd, sb)
            @test wrapper(s.ωd) === wrapper(s.Eωk_g12)
        end

        # delayed_input must produce a device array in every case, since the solver
        # adopts it as a working buffer (preserve_input=false)
        for s in (sd, sb)
            Eωk = TS.delayed_input(s, 0.7e-15)
            @test Utils.backend(Eωk) isa Utils.DeviceBackend
        end

        # ...and beamlets_on_host must give the same delay point, not merely run
        ob = TS.simulate_delay_point(sb, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
        oh = TS.simulate_delay_point(sh, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
        @test isapprox(ob.Iω_win, oh.Iω_win; rtol=1e-8)
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

        # the cached device mask is the quadrant indicator, on the right array type
        mask = TS._sqn_devmask!(qn, dev.y)
        @test Utils.backend(mask) isa Utils.DeviceBackend
        @test size(mask) == (1, 16, 16)
        @test Array(mask)[1, :, :] == Float64.(qn.sig_quad)
        @test TS._sqn_devmask!(qn, dev.y) === mask     # cached, not rebuilt
    end

    @testset "delay point with the quadrant norm on device" begin
        # The norm has to work through a whole propagation, not just when called
        # directly: an error metric that is merely wrong (rather than throwing) shows up
        # only as the stepper rejecting every step.
        #
        # NB rtol=1e-6, not the production 1e-8. On a grid this small the signal
        # quadrant holds numerical noise rather than a real FWM signal, and at 1e-8 the
        # default floor_rel=1e-6 lets the norm chase that noise's relative error until
        # the stepper gives up — on the HOST as well as on a device. That is a property
        # of the tolerance/floor pair on a toy grid, not of the device path.
        sh = TS.build_setup(; base_kwargs...)
        sd = TS.build_setup(; base_kwargs..., arraytype=JLArray)
        qh = TS.signal_quadrant_norm(sh)
        qd = TS.signal_quadrant_norm(sd)
        oh = TS.simulate_delay_point(sh, 0.0; nz=2, init_dz=5e-7, rtol=1e-6, norm=qh)
        od = TS.simulate_delay_point(sd, 0.0; nz=2, init_dz=5e-7, rtol=1e-6, norm=qd)
        @test isapprox(od.Iω_win, oh.Iω_win; rtol=1e-8)
        @test all(isfinite, od.Iω_win)
    end

    @testset "save-time extraction" begin
        # The whole point is that no z-slice is ever stored, streamed or transferred:
        # each is reduced where it is produced. On a device that means the reduction
        # must run on the DEVICE array — `HostOutput` hands `y` through untouched and
        # `step_on` makes the interpolant return that same array.
        sh = TS.build_setup(; base_kwargs...)
        sd = TS.build_setup(; base_kwargs..., arraytype=JLArray)
        zv = [0.0, 0.5e-6, 1e-6]
        kwp = (; zsave=zv, init_dz=5e-7, rtol=1e-8)

        # On the host the new route reuses the same kernels on the same arrays, so it
        # must be BIT-identical to saving the stack and reducing afterwards. This is the
        # guarantee that lets it be switched on for a validated CPU campaign.
        a = TS.simulate_delay_point(sh, 0.5e-15; kwp...)
        b = TS.simulate_delay_point(sh, 0.5e-15; kwp..., extract_on_save=true)
        for k in keys(a)
            @test a[k] == b[k]
        end

        # ...and on a device it agrees to rounding (the sums are formed in a different
        # order), which is the standard everywhere else on the device path.
        d = TS.simulate_delay_point(sd, 0.5e-15; kwp...)   # defaults ON for a device
        @test isapprox(d.Iω_win, a.Iω_win; rtol=1e-10)
        @test isapprox(d.Iω_win_reimaged, a.Iω_win_reimaged; rtol=1e-10)
        @test isapprox(d.Iω_full, a.Iω_full; rtol=1e-10)
        @test d.zsave ≈ a.zsave
        # explicitly disabling it must fall back to the save-the-stack route
        d0 = TS.simulate_delay_point(sd, 0.5e-15; kwp..., extract_on_save=false)
        @test isapprox(d0.Iω_win, d.Iω_win; rtol=1e-10)

        # The handler must hold no field-sized data: that is the memory claim.
        o = TS.TraceExtractOutput(sd, zv, JLArray)
        @test Utils.backend(o.wsgn[1]) isa Utils.DeviceBackend
        @test size(o.Iω_full) == (length(sd.grid.ω), length(zv))
        # Luna must neither refuse the run for statistics nor copy `y` every step.
        @test TS.Luna.nostats_only(o)
        @test !TS.Luna.needs_host_y(o)

        # Folding the sign into the window is what lets ONE array serve both windowed
        # reductions; squaring it must give back the unsigned window.
        w = sd.window_array
        @test Array(o.wsgn[1]).^2 ≈ (ndims(w) == 3 ? w : reshape(w, 1, size(w)...)).^2
    end

    @testset "save-time extraction with several windows" begin
        # The multi-window path assembles a different NamedTuple; check the two routes
        # agree key for key rather than just on the first window.
        wins = [TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3, holediam=0.25e-3,
                                      zmask=0.1, apod=:supergauss, apod_param=16),
                TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3, holediam=0.35e-3,
                                      zmask=0.1, apod=:supergauss, apod_param=16)]
        mw = merge(base_kwargs, (; window=wins))
        sh = TS.build_setup(; mw...)
        zv = [0.0, 1e-6]
        a = TS.simulate_delay_point(sh, 0.3e-15; zsave=zv, init_dz=5e-7, rtol=1e-8)
        b = TS.simulate_delay_point(sh, 0.3e-15; zsave=zv, init_dz=5e-7, rtol=1e-8,
                                    extract_on_save=true)
        @test keys(a) == keys(b)
        for k in keys(a)
            @test a[k] == b[k]
        end
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

# --- Real GPU: gated on LUNA_TEST_CUDA=1, skipped everywhere else -------------
# The JLArrays tests above validate the device LOGIC. This runs the same thing on
# actual hardware, which additionally exercises CUDA code generation, cuFFT, and the
# `arraytype=:cuda` resolution path a production scan uses.
if get(ENV, "LUNA_TEST_CUDA", "") == "1"
    cuda_ok = try
        @eval import CUDA
        Base.invokelatest(CUDA.functional)
    catch
        false
    end
    if !cuda_ok
        @warn "LUNA_TEST_CUDA=1 but CUDA is not functional; skipping"
    else
        @testset "CUDA delay point" begin
            Base.invokelatest(CUDA.allowscalar, false)
            beam = TS.HE11Beam(125e-6, 5.0, 0.1)
            window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                            holediam=0.25e-3, zmask=0.1,
                                            apod=:supergauss, apod_param=16)
            kw = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                    thickness=1e-6, material=:SiO2,
                    mask_diam=1.0e-3, mask_spacing=0.5e-3,
                    beam, window,
                    trange=20e-15, λlims=(200e-9, 400e-9),
                    R=40e-6, N=32)

            # `:cuda` must resolve to a device array type — this is the path a scan
            # script takes, where the symbol is only resolved on the compute node.
            sd = TS.build_setup(; kw..., arraytype=:cuda)
            @test Utils.backend(sd.transform.Eto) isa Utils.DeviceBackend

            sh = TS.build_setup(; kw...)
            for raman in (false, true)
                sh_r = raman ? TS.build_setup(; kw..., raman=true) : sh
                sd_r = raman ? TS.build_setup(; kw..., raman=true, arraytype=:cuda) : sd
                oh = TS.simulate_delay_point(sh_r, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
                od = TS.simulate_delay_point(sd_r, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
                rel = maximum(abs.(od.Iω_win .- oh.Iω_win)) / maximum(abs, oh.Iω_win)
                @info "ModelPNPS CUDA vs host" raman rel_Iω_win=rel
                @test rel < 1e-8
                @test od.Iω_win isa Array          # extraction lands on the host
            end

            # The setup-derived error norm must work on the device too: it is what
            # production scans use, and without a fused method the solver would refuse.
            # rtol=1e-6 here for the reason given in the JLArray version of this test.
            qn = TS.signal_quadrant_norm(sd)
            qh2 = TS.signal_quadrant_norm(sh)
            @test TS.Luna.RK45.fused_errnorm(qn) !== nothing
            oq = TS.simulate_delay_point(sd, 0.0; nz=2, init_dz=5e-7, rtol=1e-6, norm=qn)
            oqh = TS.simulate_delay_point(sh, 0.0; nz=2, init_dz=5e-7, rtol=1e-6, norm=qh2)
            @test all(isfinite, oq.Iω_win)
            @test isapprox(oq.Iω_win, oqh.Iω_win; rtol=1e-8)

            # beamlets_on_host is the memory lever for the largest campaigns; check it
            # produces the same answer, not just that it runs
            sb = TS.build_setup(; kw..., arraytype=:cuda, beamlets_on_host=true)
            ob = TS.simulate_delay_point(sb, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
            oh0 = TS.simulate_delay_point(sh, 0.5e-15; nz=2, init_dz=5e-7, rtol=1e-8)
            @test isapprox(ob.Iω_win, oh0.Iω_win; rtol=1e-8)
        end
    end
else
    @info "ModelPNPS CUDA tests skipped (set LUNA_TEST_CUDA=1 on a GPU machine)"
end

end
