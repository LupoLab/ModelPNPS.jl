# =============================================================================
# Tests for the data-driven input pulse (InputPulseData + build_setup injection).
#
# The equivalence test is the load-bearing one: a displaced Gaussian pulse fed
# through the complete production data path (fine-grid samples -> center ->
# InputPulseData -> interp -> build_setup) must reproduce the analytic path's
# beamlet field and centred temporal diagnostics. That closes the loop on grid
# conventions (absolute-ω, FFT ordering and centred time axes), interpolation,
# and the exact time-origin bug which originally put data pulses at -T/2.
#
# Run standalone with:
#     julia --project=. test/input_pulse_test.jl
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end
using Test
using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna: Fields, PhysData
import FFTW

@testset "Input pulse" begin

    # -----------------------------------------------------------------------------
    @testset "InputPulseData validation" begin
        ω = collect(range(1.0e15, 2.0e15, length = 64))
        E = ones(ComplexF64, 64)
        @test TS.InputPulseData(ω, E) isa TS.InputPulseData
        @test_throws ArgumentError TS.InputPulseData(ω, E[1:32])
        @test_throws ArgumentError TS.InputPulseData(reverse(ω), E)
        @test_throws ArgumentError TS.InputPulseData(ω[1:4], E[1:4])
    end

    # -----------------------------------------------------------------------------
    @testset "spectral_window!" begin
        ω = collect(range(0.1e15, 2.0e16, length = 4096))
        p = TS.InputPulseData(ω, ones(ComplexF64, length(ω)))
        TS.spectral_window!(p, 200.0e-9, 800.0e-9)
        λ = 2π * PhysData.c ./ ω
        deep_in = (λ .> 300.0e-9) .& (λ .< 600.0e-9)
        deep_out = (λ .> 1500.0e-9) .| (λ .< 150.0e-9)
        @test all(abs.(p.Eω[deep_in]) .> 0.99)
        @test all(abs.(p.Eω[deep_out]) .< 1.0e-2)
    end

    # -----------------------------------------------------------------------------
    @testset "center_pulse!" begin
        # Gaussian spectrum with a pure linear phase = a time-shifted pulse.
        ω0 = 2π * PhysData.c / 260.0e-9
        ω = collect(range(0.2ω0, 2.0ω0, length = 8192))
        τg = 2.0e-15 / (2 * sqrt(2 * log(2)))     # 2 fs FWHM intensity
        t0 = 7.3e-15
        G = exp.(-(ω .- ω0) .^ 2 .* τg^2)
        p = TS.InputPulseData(ω, G .* exp.(-1im .* ω .* t0))
        p, tshift = TS.center_pulse!(p)
        dt_resolvable = 2π / (last(ω) - first(ω))
        @test abs(abs(tshift) - t0) < 2 * dt_resolvable
        # Idempotent: a centred pulse re-centres by ~0.
        _, tshift2 = TS.center_pulse!(p)
        @test abs(tshift2) < 2 * dt_resolvable
        # Non-uniform grid is rejected.
        ωnu = copy(ω); ωnu[7] += 0.3 * (ω[2] - ω[1])
        @test_throws ArgumentError TS.center_pulse!(TS.InputPulseData(ωnu, G))
    end

    # -----------------------------------------------------------------------------
    @testset "interp_input_pulse anchors Luna time grids at t=0" begin
        ω0 = 2π * PhysData.c / 260.0e-9
        ω = collect(range(0.2ω0, 2.0ω0, length = 8192))
        τg = 2.0e-15 / (2 * sqrt(2 * log(2)))
        p = TS.InputPulseData(ω, exp.(-(ω .- ω0) .^ 2 .* τg^2))

        for grid in (
                Luna.Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15),
                Luna.Grid.RealGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15),
            )
            Eω = TS.interp_input_pulse(grid, p)
            It = TS._envelope_intensity(grid, TS._to_time(grid, Eω))
            ipk = argmax(It)
            @test abs(grid.t[ipk]) <= grid.t[2] - grid.t[1]
            @test ipk != 1
            @test ipk != length(grid.t)
        end
    end

    # -----------------------------------------------------------------------------
    @testset "build_setup equivalence: data path vs analytic Gaussian" begin
        λ0, τfwhm, energy = 260.0e-9, 2.0e-15, 0.2e-6
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        kwargs = (;
            λ0, τfwhm, energy, thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3, beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9), R = 40.0e-6, N = 32,
        )

        setup1 = TS.build_setup(; kwargs...)

        # The analytic 1-D reference, rebuilt exactly as build_setup does. Remove
        # the target grid's middle-sample phase before resampling it onto a fine
        # ASCENDING absolute-ω data grid: measured/simulated source data can have a
        # completely different time window, and center_pulse! must remove that
        # source-grid phase before interp_input_pulse supplies the target one.
        grid = setup1.grid
        Eref = Fields.GaussField(;
            λ0 = λ0, τfwhm = τfwhm, energy = energy,
            ϕ = [0.0, 0.0, 0.0, 0.0]
        )(grid, TS._plan_1d(grid))
        τgrid = length(grid.t) * (grid.t[2] - grid.t[1]) / 2
        Esrc = Eref .* exp.(1im .* grid.ω .* τgrid)
        ord = sortperm(grid.ω)
        ωs, Es = grid.ω[ord], Esrc[ord]
        keep = ωs .> 0
        ωs, Es = ωs[keep], Es[keep]
        ωfine = collect(range(first(ωs), last(ωs), length = 8 * length(ωs)))
        spl_r = Luna.Maths.BSpline(ωs, real(Es))
        spl_i = Luna.Maths.BSpline(ωs, imag(Es))
        # Add a source-grid delay as real data would carry, then run the same
        # centring call as the production RDW driver.
        tdata = 7.3e-15
        p = TS.InputPulseData(
            ωfine,
            complex.(spl_r.(ωfine), spl_i.(ωfine)) .* exp.(-1im .* ωfine .* tdata)
        )
        p, tshift = TS.center_pulse!(p)
        dt_resolvable = 2π / (last(ωfine) - first(ωfine))
        @test abs(tshift - tdata) < 2 * dt_resolvable

        setup2 = TS.build_setup(; kwargs..., input_pulse = p)

        # Beamlet spectra agree (shape): the whole chromatic-vignetting chain saw
        # the same pulse.
        b1 = setup1.combined_grid["Iω_beamlet"]
        b2 = setup2.combined_grid["Iω_beamlet"]
        @test sum(abs, b2 ./ maximum(b2) .- b1 ./ maximum(b1)) /
            sum(abs, b1 ./ maximum(b1)) < 1.0e-3

        # Complex 1-D fields agree up to their irrelevant amplitude scale and a
        # constant phase. This catches a missing/duplicated half-window phase even
        # though the spectral-intensity comparison above cannot see one.
        e1 = setup1.Eω ./ sqrt(sum(abs2, setup1.Eω))
        e2 = setup2.Eω ./ sqrt(sum(abs2, setup2.Eω))
        ph = sum(conj.(e1) .* e2)
        ph /= abs(ph)
        # center_pulse! locates the peak on a finite oversampled time grid, so a
        # small residual linear phase remains; the missing T/2 phase this guards
        # against would instead give an order-unity disagreement.
        @test sqrt(sum(abs2, e2 .- ph .* e1)) < 3.0e-2

        # The input and beamlet truth diagnostics must be single pulses centred in
        # the Luna time window, not periodic halves at both endpoints. This is the
        # failure which made croak's truth.fwhm return NaN for the RDW file.
        for (tk, Ik) in (
                ("t", "It"), ("To", "Ito"),
                ("t", "It_beamlet"), ("To", "Ito_beamlet"),
            )
            tdiag = setup2.combined_grid[tk]
            Idiag = setup2.combined_grid[Ik]
            @test abs(tdiag[argmax(Idiag)]) <= tdiag[2] - tdiag[1]
            above = Idiag .>= 0.5maximum(Idiag)
            @test !above[1]
            @test !above[end]
            ii = findall(above)
            @test all(diff(ii) .== 1)
        end

        # Provenance metadata.
        @test setup1.combined_grid["input_pulse"] == 0
        @test setup2.combined_grid["input_pulse"] == 1
        @test length(setup2.combined_grid["Iω_input"]) == length(grid.ω)

        # The data path still runs the delay-point machinery. (skip_propagation
        # generates no FWM, and the input beamlets underflow to exactly zero in
        # the signal quadrant, so like the analytic-path tests this asserts
        # structure and non-negativity, not signal.)
        out = TS.simulate_delay_point(setup2, 0.0; skip_propagation = true, nz = 2)
        @test all(isfinite, out.Iω_win)
        @test all(out.Iω_win .>= 0)
        @test size(out.Iω_win) == (length(grid.ω), 2)

        # Guards: Gaussian beam model and non-overlapping band are rejected.
        gbeam = TS.GaussianBeam(2.5e-3, 0.1)
        @test_throws ArgumentError TS.build_setup(;
            kwargs..., beam = gbeam,
            input_pulse = p
        )
        pfar = TS.InputPulseData(
            collect(range(1.0e14, 2.0e14, length = 64)),
            ones(ComplexF64, 64)
        )
        @test_throws ArgumentError TS.build_setup(; kwargs..., input_pulse = pfar)
    end

end
