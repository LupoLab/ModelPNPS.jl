# =============================================================================
# Tests for ModelPNPS TG-FROG trace simulation.
#
# Most tests run quickly because they exercise individual primitives (grid
# sizing, mode construction, mask construction, window construction, beam
# tilts, time delays, signal extraction) without invoking Luna.run. The
# `simulate_delay_point` integration tests use `skip_propagation=true` to
# fake the propagation step. There is also one tiny end-to-end smoke test
# that DOES call Luna.run on a minimal grid in a few seconds.
#
# Run standalone with:
#     julia --project=. test/tracesimulation_test.jl
# Or as part of the suite via test/runtests.jl.
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end
using Test
using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna: Grid, PhysData
import FFTW
import FFTW: fft, ifft
import HDF5
import Random: MersenneTwister, Xoshiro

@testset "Trace simulation" begin

    # -----------------------------------------------------------------------------
    @testset "optimal_spatial_grid" begin
        f = 0.1
        mask_diam = 1.0e-3
        mask_spc = 0.5e-3
        λmin, λmax = 160.0e-9, 500.0e-9

        R, N = TS.optimal_spatial_grid(f, mask_diam, mask_spc, λmin, λmax)

        # N is an even, 2,3,5-smooth FFT size (fast for FFTW without the up-to-2×
        # overshoot of nextpow(2, ...)).
        smooth(n) = (
            for p in (2, 3, 5)
                while n % p == 0
                    n ÷= p
                end
            end; n == 1
        )
        @test smooth(N)
        @test iseven(N)
        @test N > 0

        # The margin keyword scales the resolved size.
        _, N_bigmargin = TS.optimal_spatial_grid(
            f, mask_diam, mask_spc, λmin, λmax;
            margin = 2.0
        )
        @test N_bigmargin > N

        # Real-space resolution: dx ≤ Airy(λmin) / pts_per_lobe (default 10).
        dx = 2R / N
        r_airy_min = 1.22 * λmin * f / mask_diam
        @test dx <= r_airy_min / 10 + 1.0e-12

        # k-space containment requires `kmax ≥ safety·3·2π·x_max/(λmin·f)`.
        kmax = π * N / (2R)
        x_max = mask_spc / 2 + mask_diam
        k_NL_max = 1.5 * 3 * 2π * x_max / (λmin * f)
        @test kmax >= k_NL_max * (1 - 1.0e-12)

        # Larger safety produces ≥ as large N.
        R2, N2 = TS.optimal_spatial_grid(
            f, mask_diam, mask_spc, λmin, λmax;
            safety = 3.0
        )
        @test N2 >= N
    end

    # -----------------------------------------------------------------------------
    @testset "HE11Beam k-space construction" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        @test TS.a_scaled(beam) ≈ 2.5e-6

        # Tiny grid so the test is cheap.
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)

        # 1-D reference spectrum (using a Luna GaussField).
        FT1d = FFTW.plan_fft(copy(grid.t))
        field = Luna.Fields.GaussField(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 1.0e-9
        )
        Eω = field(grid, FT1d)

        Eωk0 = TS.build_he11_kspace(grid, xygrid, beam, Eω)
        @test size(Eωk0) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
        @test all(isfinite, Eωk0)

        # After IFFT to (y, x), the beam should peak near the centre pixel
        # (phase ramps in build_he11_kspace shift it from the FFTW corner).
        Eωxy0 = ifft(Eωk0, (2, 3))
        iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260.0e-9))
        Ixy = abs2.(Eωxy0[iω0, :, :])
        cy = length(xygrid.y) ÷ 2 + 1
        cx = length(xygrid.x) ÷ 2 + 1
        py, px = Tuple(argmax(Ixy))
        @test abs(py - cy) <= 2
        @test abs(px - cx) <= 2

        # Energy rescaling round-trip: rescale → energyfun_ω returns the target.
        _, energyfun_ω = Luna.Fields.energyfuncs(grid, xygrid)
        target_E = 1.5e-9
        Eωk_rescaled = Eωk0 .* (sqrt(target_E) / sqrt(energyfun_ω(Eωk0)))
        @test energyfun_ω(Eωk_rescaled) ≈ target_E rtol = 1.0e-10
    end

    # -----------------------------------------------------------------------------
    @testset "GaussianBeam k-space construction" begin
        beam = TS.GaussianBeam(8.3e-6, 0.1)
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)

        target_E = 0.2e-6 / 3
        Eωk = TS.build_gaussian_kspace(grid, xygrid, beam, 260.0e-9, 2.0e-15, target_E)
        @test size(Eωk) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
        @test all(isfinite, Eωk)

        # Total spectral energy via Parseval-based energyfun_ω.
        _, energyfun_ω = Luna.Fields.energyfuncs(grid, xygrid)
        @test energyfun_ω(Eωk) ≈ target_E rtol = 5.0e-3
    end

    # -----------------------------------------------------------------------------
    @testset "apply_tilt — k-space shift identity" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)
        beam = TS.GaussianBeam(8.3e-6, 0.1)
        Eωk0 = TS.build_gaussian_kspace(grid, xygrid, beam, 260.0e-9, 2.0e-15, 1.0e-9)
        Eωxy0 = ifft(Eωk0, (2, 3))

        # Δkx = Δky = 0 ⇒ identity (modulo numerical noise).
        Eωxy_id = TS.apply_tilt(Eωxy0, xygrid, 0.0, 0.0)
        @test maximum(abs.(Eωxy_id .- Eωxy0)) <= 1.0e-12 * maximum(abs.(Eωxy0))

        # Apply a tilt corresponding to a single k-space sample step in each axis.
        dkx = xygrid.kx[2] - xygrid.kx[1]
        dky = xygrid.ky[2] - xygrid.ky[1]
        Eωxy_t = TS.apply_tilt(Eωxy0, xygrid, dkx, dky)
        Eωk_t = fft(Eωxy_t, (2, 3))

        # The peak in the (ky, kx) slice at the carrier ω should shift by exactly
        # one bin in each direction (centroid of |E|² before and after).
        iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260.0e-9))
        Ixy0 = abs2.(Eωk0[iω0, :, :])
        Ixy_t = abs2.(Eωk_t[iω0, :, :])
        py0, px0 = Tuple(argmax(Ixy0))
        pyt, pxt = Tuple(argmax(Ixy_t))
        # FFTW shift wraps around the grid; check modulo grid size.
        @test mod(pyt - py0, length(xygrid.ky)) in (1, length(xygrid.ky) - 1, 0)
        @test mod(pxt - px0, length(xygrid.kx)) in (1, length(xygrid.kx) - 1, 0)
    end

    # -----------------------------------------------------------------------------
    @testset "apply_delay — phase ramp" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 16)
        Eωk = randn(
            ComplexF64, length(grid.ω), length(xygrid.ky),
            length(xygrid.kx)
        )

        # τ=0 ⇒ identity.
        Eωk0 = TS.apply_delay(Eωk, grid, 0.0)
        @test maximum(abs.(Eωk0 .- Eωk)) <= 1.0e-12

        # τ ≠ 0: phase ramp matches -ω·τ at every (ω, ky, kx) where |E| > 0.
        τ = 1.5e-15
        Eωk_d = TS.apply_delay(Eωk, grid, τ)
        @test all(isfinite, Eωk_d)
        # Pick a few non-trivial ω indices and verify ratio.
        for iω in (3, length(grid.ω) ÷ 4, length(grid.ω) ÷ 2)
            ratio = Eωk_d[iω, 1, 1] / Eωk[iω, 1, 1]
            expected = exp(-1im * grid.ω[iω] * τ)
            @test ratio ≈ expected rtol = 1.0e-10
        end
    end

    # -----------------------------------------------------------------------------
    @testset "makemask — apodisation behaviour" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)

        # Hard mask ⇒ binary.
        m_hard = TS.makemask(
            0.0, 0.0, 0.5e-3, grid, xygrid;
            zmask = 0.1, apod = :hard
        )
        @test all(v -> v == 0.0 || v == 1.0, m_hard)

        # supergauss ⇒ values in [0, 1], peak = 1 at hole centre at carrier ω.
        m_sg = TS.makemask(
            0.0, 0.0, 0.5e-3, grid, xygrid;
            zmask = 0.1, apod = :supergauss
        )
        @test all(0.0 .<= m_sg .<= 1.0)
        iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260.0e-9))
        @test maximum(m_sg[iω0, :, :]) <= 1.0 + 1.0e-12

        # Chromatic vignetting: at twice the frequency, the hole's k-space radius
        # halves (because k_extent = (ω/c) · holediam/2 / zmask scales linearly
        # with ω, so larger ω ⇒ wider hole). Check that the count of mask points
        # above 0.5 increases when ω doubles.
        iω1 = argmin(abs.(grid.ω .- 2 * 2π * PhysData.c / 260.0e-9))
        if iω1 > 0 && iω1 != iω0
            n0 = count(>(0.5), m_hard[iω0, :, :])
            n1 = count(>(0.5), m_hard[iω1, :, :])
            @test n1 > n0
        end
    end

    # -----------------------------------------------------------------------------
    @testset "PlanckWindow (ω-independent)" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)

        Δk = 2π / 260.0e-9 * sin(0.015)   # 15 mrad crossing
        w = TS.PlanckWindow(kxc = -Δk, kyc = -Δk, kwidth = 2.5 / 8.3e-6, pad = 1.25)
        arr = TS.build_window(w, grid, xygrid)
        @test size(arr) == (length(xygrid.ky), length(xygrid.kx))
        @test all(0.0 .<= arr .<= 1.0)
        @test maximum(arr) ≈ 1.0 atol = 1.0e-12
    end

    # -----------------------------------------------------------------------------
    @testset "PlanckOmegaWindow (ω-dependent)" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)

        w = TS.PlanckOmegaWindow(
            xc = -0.75e-3, yc = -0.75e-3,
            holediam = 0.5e-3, f_foc = 0.1, pad = 1.25
        )
        arr = TS.build_window(w, grid, xygrid)
        @test size(arr) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
        @test all(0.0 .<= arr .<= 1.0)

        # The hole's k-space half-width khole(ω) = (ω/c)·(holediam/2)/f_foc grows
        # linearly with ω. Count of pixels above 0.5 in the per-ω slice should
        # increase with ω.
        iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 400.0e-9))   # low freq
        iω1 = argmin(abs.(grid.ω .- 2π * PhysData.c / 200.0e-9))   # high freq
        if iω0 != iω1
            n0 = count(>(0.5), arr[iω0, :, :])
            n1 = count(>(0.5), arr[iω1, :, :])
            @test n1 >= n0
        end
    end

    # -----------------------------------------------------------------------------
    @testset "extract_signal_spectra (skip_propagation)" begin
        # Build a small HE11+PhysicalMask setup.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )

        out = TS.simulate_delay_point(setup, 0.0; skip_propagation = true, nz = 2)
        @test haskey(out, :Iω_win)
        @test haskey(out, :Iω_win_reimaged)
        @test haskey(out, :Iω_full)
        @test size(out.Iω_win) == (length(setup.grid.ω), 2)
        @test size(out.Iω_win_reimaged) == (length(setup.grid.ω), 2)
        @test size(out.Iω_full) == (length(setup.grid.ω), 2)
        @test all(out.Iω_win .>= 0)
        @test all(out.Iω_win_reimaged .>= 0)
        @test all(out.Iω_full .>= 0)
        @test all(isfinite, out.Iω_full)

        # Same with a non-zero delay — should still produce finite, non-negative
        # spectra of identical shape.
        out2 = TS.simulate_delay_point(setup, 1.0e-15; skip_propagation = true, nz = 2)
        @test size(out2.Iω_win) == size(out.Iω_win)
        @test all(isfinite, out2.Iω_win)
    end

    # -----------------------------------------------------------------------------
    @testset "extract_signal_spectra: one-pass equals FFT reference" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        Ez = TS.delayed_input(setup, 0.7e-15) # non-trivial complex field
        for warr in (
                setup.window_array,                        # 3-D window
                abs.(randn(Xoshiro(7), 32, 32)),
            )           # 2-D window
            a1, b1 = TS.extract_signal_spectra(Ez, warr, setup.xygrid)
            a2, b2 = TS._extract_signal_spectra_fft(Ez, warr, setup.xygrid)
            @test isapprox(a1, a2, rtol = 1.0e-12)
            @test isapprox(b1, b2, rtol = 1.0e-10, atol = 1.0e-12 * maximum(b2))
        end
        # quadrant spectrum matches the broadcast it replaced
        sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
        sq3 = reshape(sig_quad, (1, size(sig_quad)...))
        ref = dropdims(sum(abs2.(Ez) .* sq3; dims = (2, 3)); dims = (2, 3))
        out = zeros(length(setup.grid.ω))
        TS._quadrant_spectrum!(out, Ez, sig_quad)
        @test isapprox(out, ref, rtol = 1.0e-12)
    end

    # -----------------------------------------------------------------------------
    @testset "window storage options" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        kwargs = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32,
        )
        # default: window array stored, parameters always stored (flattened scalars, so
        # scansave can write them as plain HDF5 datasets)
        s1 = TS.build_setup(; kwargs...)
        @test haskey(s1.combined_grid, "window")
        @test s1.combined_grid["window_def_type"] == "PhysicalMaskWindow"
        @test s1.combined_grid["window_def_holediam"] == 0.25e-3
        # store_window=false: parameters only (the array is ~1 GiB at production size)
        s2 = TS.build_setup(; kwargs..., store_window = false)
        @test !haskey(s2.combined_grid, "window")
        @test haskey(s2.combined_grid, "window_def_type")
        # the in-memory window array is unaffected
        @test isequal(s2.window_array, s1.window_array)
    end

    # -----------------------------------------------------------------------------
    @testset "_resolve_zsave" begin
        # Integer: uniform grid over [0, zmax], reproduces legacy nz behaviour.
        @test TS._resolve_zsave(2, 10.0e-6) == [0.0, 10.0e-6]
        @test TS._resolve_zsave(3, 10.0e-6) ≈ [0.0, 5.0e-6, 10.0e-6]
        @test_throws ArgumentError TS._resolve_zsave(1, 10.0e-6)        # need ≥ 2

        # Vector: zmax appended when absent, not duplicated when present.
        @test TS._resolve_zsave([2.0e-6, 6.0e-6], 10.0e-6) == [2.0e-6, 6.0e-6, 10.0e-6]
        @test TS._resolve_zsave([2.0e-6, 10.0e-6], 10.0e-6) == [2.0e-6, 10.0e-6]
        @test TS._resolve_zsave([1.0e-6, 10.0e-6, 20.0e-6, 40.0e-6], 40.0e-6) ==
            [1.0e-6, 10.0e-6, 20.0e-6, 40.0e-6]

        # Validation failures.
        @test_throws ArgumentError TS._resolve_zsave([6.0e-6, 2.0e-6], 10.0e-6)   # unsorted
        # Repeated positions are ambiguous save requests.
        @test_throws ArgumentError TS._resolve_zsave([2.0e-6, 2.0e-6], 10.0e-6)
        @test_throws ArgumentError TS._resolve_zsave([20.0e-6], 10.0e-6)        # > zmax
        # Propagation positions cannot precede the medium entrance.
        @test_throws ArgumentError TS._resolve_zsave([-1.0e-6, 5.0e-6], 10.0e-6)

        # Idempotent: re-resolving an already-resolved grid (which the integer path
        # produces with an entrance slice at z=0) returns it unchanged. This is the
        # path run_scan exercises when it forwards the resolved vector per delay.
        @test TS._resolve_zsave(TS._resolve_zsave(11, 40.0e-6), 40.0e-6) ==
            TS._resolve_zsave(11, 40.0e-6)
        @test TS._resolve_zsave([0.0, 5.0e-6, 10.0e-6], 10.0e-6) == [0.0, 5.0e-6, 10.0e-6]
    end

    # -----------------------------------------------------------------------------
    @testset "simulate_delay_point — zsave snapshots (skip_propagation)" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        Nω = length(setup.grid.ω)

        # Explicit thickness list (already ending at zmax): 3 slices.
        out = TS.simulate_delay_point(
            setup, 0.0; skip_propagation = true,
            zsave = [2.0e-6, 6.0e-6, 10.0e-6]
        )
        @test out.zsave == [2.0e-6, 6.0e-6, 10.0e-6]
        @test size(out.Iω_win) == (Nω, 3)
        @test size(out.Iω_full) == (Nω, 3)

        # zmax appended automatically when absent.
        out2 = TS.simulate_delay_point(
            setup, 0.0; skip_propagation = true,
            zsave = [2.0e-6, 6.0e-6]
        )
        @test out2.zsave == [2.0e-6, 6.0e-6, 10.0e-6]
        @test size(out2.Iω_win) == (Nω, 3)

        # Integer zsave and the default both reproduce the legacy nz=2 path.
        out3 = TS.simulate_delay_point(setup, 0.0; skip_propagation = true, zsave = 2)
        @test out3.zsave == [0.0, 10.0e-6]
        @test size(out3.Iω_win) == (Nω, 2)
        out4 = TS.simulate_delay_point(setup, 0.0; skip_propagation = true)
        @test size(out4.Iω_win) == (Nω, 2)
    end

    # -----------------------------------------------------------------------------
    @testset "extract_signal_spectra — multi-window" begin
        # Build a small Gaussian + two-window setup.
        beam = TS.GaussianBeam(8.3e-6, 0.1)
        Δk = 2π / 260.0e-9 * sin((0.5e-3 / 2 + 1.0e-3 / 2) / 0.1)
        windows = [
            TS.PlanckWindow(kxc = -Δk, kyc = -Δk, kwidth = 2.5 / 8.3e-6, pad = 1.25),
            TS.PlanckOmegaWindow(
                xc = -0.75e-3, yc = -0.75e-3,
                holediam = 0.5e-3, f_foc = 0.1, pad = 1.25
            ),
        ]
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window = windows,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )

        out = TS.simulate_delay_point(setup, 0.0; skip_propagation = true, nz = 2)
        @test haskey(out, :Iω_win)
        @test haskey(out, :Iω_win_reimaged)
        @test haskey(out, Symbol("Iω_win_ωdep"))
        @test haskey(out, Symbol("Iω_win_ωdep_reimaged"))
        # Single shared full signal-collection reference across both windows.
        @test haskey(out, :Iω_full)
        @test all(out.Iω_full .>= 0)
    end

    # -----------------------------------------------------------------------------
    @testset "build_setup metadata" begin
        # HE11 + PhysicalMask path: combined_grid contains beam_meta entries.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        cg = setup.combined_grid
        for key in (
                "Iω", "It", "To", "Ito", "τfwhm", "material", "thickness",
                "Iω_beamlet", "It_beamlet", "Ito_beamlet",
                "a", "a_scaled", "f_coll", "f_foc", "window",
            )
            @test haskey(cg, key)
        end
        # It_beamlet is a real, non-negative temporal intensity on grid.t; the
        # oversampled Ito_beamlet shares the To grid.
        @test size(cg["It_beamlet"]) == size(setup.grid.t)
        @test all(cg["It_beamlet"] .>= 0)
        @test length(cg["Ito_beamlet"]) == length(cg["To"])
        @test all(cg["Ito_beamlet"] .>= 0)

        # Gaussian + two-window path: combined_grid contains both windows.
        beamg = TS.GaussianBeam(8.3e-6, 0.1)
        Δk = 2π / 260.0e-9 * sin((0.5e-3 / 2 + 1.0e-3 / 2) / 0.1)
        windows = [
            TS.PlanckWindow(kxc = -Δk, kyc = -Δk, kwidth = 2.5 / 8.3e-6, pad = 1.25),
            TS.PlanckOmegaWindow(
                xc = -0.75e-3, yc = -0.75e-3,
                holediam = 0.5e-3, f_foc = 0.1, pad = 1.25
            ),
        ]
        setupg = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam = beamg, window = windows,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        cgg = setupg.combined_grid
        for key in (
                "w0", "Δk", "crossingθ", "window", "window_ωdep",
                "Iω_beamlet", "It_beamlet", "Ito_beamlet",
            )
            @test haskey(cgg, key)
        end
        # Gaussian beamlets are unvignetted, so the beamlet spectrum has the same
        # spectral *shape* as the input pulse (no chromatic clipping). Compare the
        # unit-normalised spectra (robust to the differing FFT normalisations of the
        # 1-D input vector and the spatially-integrated beamlet array).
        @test length(cgg["Iω_beamlet"]) == length(cgg["Iω"])
        @test all(cgg["Iω_beamlet"] .>= 0)
        nb = cgg["Iω_beamlet"] ./ maximum(cgg["Iω_beamlet"])
        ni = cgg["Iω"] ./ maximum(cgg["Iω"])
        @test maximum(abs.(nb .- ni)) < 5.0e-2
    end

    # -----------------------------------------------------------------------------
    @testset "Smoke test: tiny end-to-end Luna.run" begin
        # The smallest grid that still exercises the whole pipeline. Runs in a
        # few seconds on a laptop. If Luna or its FFT plans fail to set up,
        # the test fails noisily — that's intentional, this is the only place
        # the suite proves the integration boundary works.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )

        out = TS.simulate_delay_point(setup, 0.0; nz = 2, init_dz = 5.0e-7)
        @test haskey(out, :Iω_win)
        @test haskey(out, :Iω_win_reimaged)
        @test haskey(out, :Iω_full)

        # factored (lazy) linop/norm vs materialised: bit-identical end to end
        setup_mat = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32, factored_linop = false
        )
        out_mat = TS.simulate_delay_point(setup_mat, 0.0; nz = 2, init_dz = 5.0e-7)
        @test isequal(out_mat.Iω_win, out.Iω_win)
        @test isequal(out_mat.Iω_win_reimaged, out.Iω_win_reimaged)
        @test isequal(out_mat.Iω_full, out.Iω_full)

        # batched vs frozen (legacy per-column) Raman: agreement to rounding accuracy
        # through a real propagation
        ram_kwargs = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32, raman = true,
        )
        out_bat = TS.simulate_delay_point(
            TS.build_setup(; ram_kwargs...), 0.0;
            nz = 2, init_dz = 5.0e-7
        )
        out_frz = TS.simulate_delay_point(
            TS.build_setup(;
                ram_kwargs...,
                raman_impl = :frozen
            ), 0.0;
            nz = 2, init_dz = 5.0e-7
        )
        @test isapprox(out_bat.Iω_win, out_frz.Iω_win, rtol = 1.0e-8)
        @test isapprox(out_bat.Iω_full, out_frz.Iω_full, rtol = 1.0e-8)
        @test size(out.Iω_win) == (length(setup.grid.ω), 2)
        @test size(out.Iω_full) == (length(setup.grid.ω), 2)
        @test all(isfinite, out.Iω_win)
        @test all(out.Iω_win .>= 0)
        @test all(out.Iω_win_reimaged .>= 0)
        @test all(isfinite, out.Iω_full)
        @test all(out.Iω_full .>= 0)
        # Default nz=2 saves the entrance and exit (full-thickness) slices.
        @test out.zsave ≈ [0.0, setup.grid.zmax]

        # Multi-z snapshots from ONE run: realized z lands exactly on the requested
        # grid and distinct z give distinct fields (dense-output interpolation, not
        # stacked copies). This is the core "free intermediate thicknesses" claim.
        outz = TS.simulate_delay_point(
            setup, 0.0; zsave = [0.5e-6, 1.0e-6], init_dz = 5.0e-7
        )
        @test outz.zsave ≈ [0.5e-6, 1.0e-6] atol = 1.0e-15
        @test size(outz.Iω_win) == (length(setup.grid.ω), 2)
        @test any(outz.Iω_win[:, 1] .!= outz.Iω_win[:, 2])

        # `filename` swaps the in-memory MemoryOutput for an HDF5Output. The
        # extracted spectra must be identical to the in-memory run, and the file
        # must hold the raw Eω/z propagation datasets.
        mktempdir() do dir
            fpath = joinpath(dir, "delay.h5")
            outf = TS.simulate_delay_point(
                setup, 0.0; nz = 2, init_dz = 5.0e-7,
                filename = fpath
            )
            @test isfile(fpath)
            @test outf.Iω_win == out.Iω_win
            @test outf.Iω_full == out.Iω_full
            @test outf.zsave == out.zsave
            HDF5.h5open(fpath, "r") do f
                @test haskey(f, "Eω")
                @test haskey(f, "z")
                @test read(f["z"]) ≈ out.zsave
                @test size(read(f["Eω"]))[end] == length(out.zsave)
            end
        end
    end

    # -----------------------------------------------------------------------------
    # Synthetic-file test for load_simulated_scan. The real scansave files are
    # 10s-100s of MB; we write a tiny mock file with the same structure to
    # exercise the loader logic.
    # -----------------------------------------------------------------------------

    function _write_mock_scan_file(
            path; Nω = 64, Nt = 64, Nτ = 8, nz = 2,
            with_beamlet = true,
            with_omega_dep = false,
            zsave = nothing,
            ω0 = 2π * 2.99792458e8 / 260.0e-9,
            dω = 1.0e13, dt = 1.0e-15
        )
        # Build an FFT-ordered frequency vector centred on zero.
        halfN = Nω ÷ 2
        # Frequencies are relative to the carrier `ω0`.
        ω_fft = [0:(halfN - 1); -halfN:-1] .* dω
        ω_abs = ω_fft .+ ω0                              # absolute frequency
        Iω_fft = abs2.(exp.(-(ω_fft ./ (5dω)) .^ 2))       # Gaussian centred at DC bin
        t = collect(((-Nt ÷ 2):(Nt ÷ 2 - 1))) .* dt
        It = abs2.(exp.(-(t ./ (5dt)) .^ 2))
        τ = collect(((-Nτ ÷ 2):(Nτ ÷ 2 - 1))) .* (2 * dt)

        # FROG trace: random non-negative, FFT-ordered along ω axis
        rng_seed = 1234
        rand_arr = rand(MersenneTwister(rng_seed), Nω, nz, Nτ)

        HDF5.h5open(path, "w") do f
            g = HDF5.create_group(f, "grid")
            # `scansave` writes absolute frequencies in FFT order.
            g["ω"] = ω_abs
            g["ω0"] = ω0
            g["t"] = t
            g["Iω"] = Iω_fft               # in same FFT order as ω
            g["It"] = It
            g["τfwhm"] = 2.0e-15
            if with_beamlet
                g["Iω_beamlet"] = Iω_fft .* 0.7   # smaller (vignetted)
                g["It_beamlet"] = It .* 0.7        # beamlet temporal intensity
            end
            if !isnothing(zsave)
                g["zsave"] = collect(Float64, zsave)
            end
            sv = HDF5.create_group(f, "scanvariables")
            sv["τ"] = τ
            f["Iω_win"] = rand_arr
            f["Iω_win_reimaged"] = rand_arr .* 0.5
            # The full collection reference contains at least the windowed signal.
            f["Iω_full"] = rand_arr .* 2.0
            if with_omega_dep
                f["Iω_win_ωdep"] = rand_arr .* 0.8
                f["Iω_win_ωdep_reimaged"] = rand_arr .* 0.4
            end
        end
        return ω_abs, ω_fft, Iω_fft, t, It, τ
    end

    @testset "load_simulated_scan — basic round-trip" begin
        mktempdir() do tmpdir
            path = joinpath(tmpdir, "mock_scan.h5")
            ω_abs, ω_fft, Iω_fft, t, It, τ =
                _write_mock_scan_file(path; Nω = 32, Nτ = 4, nz = 2, with_beamlet = true)

            nt = TS.load_simulated_scan(path)
            # Shapes
            @test length(nt.ω) == 32
            @test length(nt.t) == length(t)
            @test length(nt.τ) == 4
            @test size(nt.trace) == (32, 4)
            @test length(nt.Iω) == 32
            @test haskey(nt, :Iω_beamlet)
            @test length(nt.Iω_beamlet) == 32
            # Beamlet temporal intensity round-trips (time-domain, not fftshifted).
            @test haskey(nt, :It_beamlet)
            @test nt.It_beamlet ≈ It .* 0.7

            # fftshift: nt.ω == fftshift(ω_abs)
            @test nt.ω ≈ FFTW.fftshift(ω_abs)
            @test nt.Iω ≈ FFTW.fftshift(Iω_fft)
            @test nt.Iω_beamlet ≈ FFTW.fftshift(Iω_fft .* 0.7)

            # ω0 came back as the value we wrote
            @test nt.ω0 ≈ 2π * 2.99792458e8 / 260.0e-9

            # z_index defaults to :end (= last z slice)
            @test all(nt.trace .>= 0)

            # Selecting a different window
            nt_re = TS.load_simulated_scan(path; window_key = "Iω_win_reimaged")
            @test nt_re.trace ≈ nt.trace .* 0.5

            # The full signal-collection reference loads via window_key and is ≥
            # the windowed trace everywhere (collection efficiency ≤ 1).
            nt_full = TS.load_simulated_scan(path; window_key = "Iω_full")
            @test nt_full.trace ≈ nt.trace .* 2.0
            @test all(nt.trace .<= nt_full.trace .+ 1.0e-20)

            # z_index=1 picks the first slice
            nt_z1 = TS.load_simulated_scan(path; z_index = 1)
            @test size(nt_z1.trace) == (32, 4)

            # Bad window key raises
            @test_throws Exception TS.load_simulated_scan(path; window_key = "not_a_key")

            # Back-compat: a file with no /grid/zsave loads fine and omits :zsave.
            @test !haskey(nt, :zsave)
        end
    end

    # -----------------------------------------------------------------------------
    @testset "load_simulated_scan — multi-z (zsave, :all, z_thickness)" begin
        mktempdir() do tmpdir
            path = joinpath(tmpdir, "mock_scan_z.h5")
            zvec = [1.0e-6, 10.0e-6, 20.0e-6, 40.0e-6]
            _write_mock_scan_file(path; Nω = 32, Nτ = 4, nz = 4, zsave = zvec)

            # zsave round-trips; default :end still returns a single 2-D slice.
            nt = TS.load_simulated_scan(path)
            @test haskey(nt, :zsave)
            @test nt.zsave == zvec
            @test size(nt.trace) == (32, 4)

            # :all returns the full (Nω, nz, Nτ) stack; fftshift only along ω, so
            # the last slice equals the default :end load.
            nt_all = TS.load_simulated_scan(path; z_index = :all)
            @test size(nt_all.trace) == (32, 4, 4)
            @test nt_all.zsave == zvec
            @test nt_all.trace[:, end, :] ≈ nt.trace

            # z_thickness selects the nearest saved slice (11 µm → 10 µm = index 2).
            nt_t = TS.load_simulated_scan(path; z_thickness = 11.0e-6)
            @test nt_t.trace ≈ nt_all.trace[:, 2, :]

            # z_thickness on a file without /grid/zsave errors.
            path2 = joinpath(tmpdir, "mock_no_z.h5")
            _write_mock_scan_file(path2; Nω = 32, Nτ = 4, nz = 2)
            @test_throws Exception TS.load_simulated_scan(path2; z_thickness = 10.0e-6)
        end
    end

    # -----------------------------------------------------------------------------
    @testset "FrozenRamanPolarEnv matches Luna's RamanPolarEnv" begin
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
        scale = 0.18 * PhysData.ε_0 * PhysData.χ3(:SiO2)
        Rluna = Luna.Nonlinear.RamanPolarEnv(
            grid.to, Luna.Raman.raman_response(grid.to, :SiO2, scale)
        )
        Rfroz = TS.FrozenRamanPolarEnv(
            grid.to, Luna.Raman.raman_response(grid.to, :SiO2, scale)
        )

        # Realistic field amplitude (V/m): with O(1) fields the polarisation
        # ~ ε₀χ³|E|²E ≈ 1e-34 sits below the ulp of any O(1) accumulation buffer
        # and the additivity check would be vacuous.
        rng = MersenneTwister(7)
        Et = 1.0e12 .* randn(rng, ComplexF64, length(grid.to))

        P1 = zeros(ComplexF64, length(grid.to))
        P2 = zeros(ComplexF64, length(grid.to))
        Rluna(P1, Et, 1.0)
        Rfroz(P2, Et, 1.0)
        @test maximum(abs, P1) > 0          # response not trivially zero
        @test P1 ≈ P2 rtol = 1.0e-13

        # Accumulation semantics: the response must ADD to a nonzero buffer.
        seed = maximum(abs, P1)
        outa = fill(complex(seed), length(grid.to))
        Rfroz(outa, Et, 1.0)
        @test outa .- seed ≈ P2 rtol = 1.0e-8   # looser: cancellation in the subtraction

        # Column-matrix branch (N, 1), as used by scalar modal paths.
        Etm = reshape(Et, :, 1)
        out1m = zeros(ComplexF64, size(Etm))
        out2m = zeros(ComplexF64, size(Etm))
        Rluna(out1m, Etm, 1.0)
        Rfroz(out2m, Etm, 1.0)
        @test out1m ≈ out2m rtol = 1.0e-13

        # Second call with fresh data: buffers are reused, kernel stays frozen.
        Et2 = 1.0e12 .* randn(rng, ComplexF64, length(grid.to))
        fill!(P1, 0); fill!(P2, 0)
        Rluna(P1, Et2, 1.0)
        Rfroz(P2, Et2, 1.0)
        @test P1 ≈ P2 rtol = 1.0e-13
    end

    # -----------------------------------------------------------------------------
    @testset "build_setup raman keyword" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        kwargs = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32,
        )

        # Default: Raman off, recorded as such in the metadata.
        setup0 = TS.build_setup(; kwargs...)
        @test setup0.combined_grid["raman"] == 0

        # Raman on: builds cleanly and records the model in the metadata.
        setup1 = TS.build_setup(; kwargs..., raman = true, raman_fraction = 0.2)
        @test setup1.combined_grid["raman"] == 1
        @test setup1.combined_grid["raman_fraction"] == 0.2

        # Fake-propagation signal extraction still works with the Raman setup —
        # a smoke check that the two-response pipeline is wired through.
        out = TS.simulate_delay_point(setup1, 0.0; skip_propagation = true, nz = 2)
        @test all(isfinite, out.Iω_win)

        # Materials without a condensed-phase (:intermediate) Raman model raise.
        @test_throws ArgumentError TS.build_setup(;
            kwargs..., material = :N2,
            raman = true
        )
    end

    # -----------------------------------------------------------------------------
    @testset "Raman split: quasi-static limit recovers the Kerr-only response" begin
        # The defining property of the envelope-defined f_R (prop_gnlse
        # convention): for a pulse much longer than the Raman memory,
        # h_R ⊛ |E|² → |E|², so Kerr((1-f_R)χ³) + Raman must equal Kerr(χ³).
        # This pins the relative Kerr/Raman normalisation — with the (3/2)-less
        # scale of Luna's low-level examples the totals differ by ~6%, well
        # outside the tolerance below.
        grid = Grid.EnvGrid(10.0e-6, 260.0e-9, (255.0e-9, 265.0e-9), 20.0e-12)
        fr = 0.18
        χ3 = PhysData.χ3(:SiO2)

        K_full = Luna.Nonlinear.Kerr_env(χ3)
        K_part = Luna.Nonlinear.Kerr_env((1 - fr) * χ3)
        R_part = TS.FrozenRamanPolarEnv(
            grid.to, Luna.Raman.raman_response(
                grid.to, :SiO2,
                1.5 * fr * PhysData.ε_0 * χ3
            )
        )

        # 5 ps FWHM Gaussian envelope: ~50x the Raman memory.
        Et = ComplexF64.(1.0e12 .* exp.(-2 * log(2) .* (grid.to ./ 5.0e-12) .^ 2))

        P_full = zeros(ComplexF64, length(grid.to))
        K_full(P_full, Et, 1.0)
        P_split = zeros(ComplexF64, length(grid.to))
        K_part(P_split, Et, 1.0)
        R_part(P_split, Et, 1.0)

        @test P_split ≈ P_full rtol = 2.0e-2
        @test isapprox(maximum(abs, P_split), maximum(abs, P_full); rtol = 1.0e-2)
    end

    # -----------------------------------------------------------------------------
    @testset "delay convention: gate frame (probe delayed by -tau)" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        τ = 1.5e-15
        # The scan input must carry the probe at -τ (gate-delay/paper convention).
        got = TS.delayed_input(setup, τ)
        @test got ≈ setup.Eωk_g12 .+
            TS.apply_delay(setup.Eωk_t_base, setup.grid, -τ)
        @test !(
            got ≈ setup.Eωk_g12 .+
                TS.apply_delay(setup.Eωk_t_base, setup.grid, τ)
        )
        # τ = 0 is the identity for both conventions.
        @test TS.delayed_input(setup, 0.0) ≈
            setup.Eωk_g12 .+ setup.Eωk_t_base
    end

    # -----------------------------------------------------------------------------
    @testset "complex beamlet spectrum stored with phase" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        kwargs = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32,
        )

        # Transform-limited input: flat phase, so the beamlet spectrum is real.
        s0 = TS.build_setup(; kwargs...)
        cg = s0.combined_grid
        @test haskey(cg, "Eω_beamlet_re") && haskey(cg, "Eω_beamlet_im")
        E0 = cg["Eω_beamlet_re"] .+ im .* cg["Eω_beamlet_im"]
        @test abs2.(E0) ≈ cg["Iω_beamlet"] rtol = 1.0e-10
        @test maximum(abs, cg["Eω_beamlet_im"]) < 1.0e-8 * maximum(abs, E0)
        # source spectrum stored too, consistent with Iω
        @test abs2.(cg["Eω_re"] .+ im .* cg["Eω_im"]) ≈ cg["Iω"] rtol = 1.0e-10

        # Chirped input: the phase must survive into the stored beamlet.
        s2 = TS.build_setup(; kwargs..., GDD = 2.0e-30)
        cg2 = s2.combined_grid
        E2 = cg2["Eω_beamlet_re"] .+ im .* cg2["Eω_beamlet_im"]
        @test abs2.(E2) ≈ cg2["Iω_beamlet"] rtol = 1.0e-10   # amplitude unchanged
        @test maximum(abs, cg2["Eω_beamlet_im"]) > 1.0e-3 * maximum(abs, E2)
    end

    # -----------------------------------------------------------------------------
    @testset "signal_quadrant_norm" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        qnorm = TS.signal_quadrant_norm(setup; floor_rel = 1.0e-6)
        wknrm = Luna.RK45.weaknorm

        Nω = length(setup.grid.ω)
        ky = setup.xygrid.ky; kx = setup.xygrid.kx
        sigmask = (ky .< 0) .& (kx .< 0)'
        # helper: field with given amplitudes in the signal quadrant / rest
        function field(a_sig, a_rest)
            f = fill(ComplexF64(a_rest), (Nω, 32, 32))
            for ix in 1:32, iy in 1:32
                sigmask[iy, ix] && (f[:, iy, ix] .= a_sig)
            end
            f
        end

        rtol, atol = 1.0e-6, 1.0e-10
        # Pump-dominated field with a weak signal; error concentrated in the
        # signal quadrant. The quadrant norm must flag what weaknorm misses.
        y = field(1.0e-3, 1.0)
        err_sig = field(1.0e-6, 0.0)          # error only in the signal quadrant
        @test qnorm(err_sig, y, y, rtol, atol) > 100 * wknrm(err_sig, y, y, rtol, atol)
        # ... because the signal-relative error is ~1e-3/rtol.

        # Error in the pump region: both norms agree to within the region split.
        err_pump = field(0.0, 1.0e-6)
        r = qnorm(err_pump, y, y, rtol, atol) / wknrm(err_pump, y, y, rtol, atol)
        @test 0.5 < r < 2.0

        # Empty signal quadrant: the floor prevents 0/0 blow-up; finite result.
        y0 = field(0.0, 1.0)
        e0 = field(1.0e-9, 0.0)
        v = qnorm(e0, y0, y0, rtol, atol)
        @test isfinite(v) && v > 0
        # The floor bounds the value: err ≤ ||err_sig|| / (rtol · floor_rel·||rest||)
        n_sig = sqrt(count(sigmask) * Nω) * 1.0e-9
        n_rest = sqrt(count(.!sigmask) * Nω) * 1.0
        @test v ≤ n_sig / (rtol * 1.0e-6 * n_rest) * (1 + 1.0e-9)

        # Wrong grid size errors loudly.
        bad = zeros(ComplexF64, (Nω, 16, 16))
        @test_throws DimensionMismatch qnorm(bad, bad, bad, rtol, atol)

        # Provenance labels.
        @test occursin("signal_quadrant", TS._norm_name(qnorm))
        @test TS._norm_name(Luna.RK45.weaknorm) == "weaknorm"
    end

    # -----------------------------------------------------------------------------
    @testset "verify_against_collected round-trip" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup_args = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32,
        )
        mktempdir() do tmpdir
            cd(tmpdir) do
                τs = [-1.0e-15, 1.0e-15]
                TS.run_scan(
                    setup_args, τs; scan_name = "verify_selftest",
                    exec = Luna.Scans.LocalExec(), zsave = 2,
                    init_dz = 5.0e-7, rtol = 1.0e-6
                )
                collected = "verify_selftest_collected.h5"
                @test isfile(collected)
                res = TS.verify_against_collected(
                    setup_args, collected, [1, 2];
                    zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-6
                )
                # The `|`-suffixed keys are the normalisation diagnostics, not differences.
                isdiff(k) = startswith(k, "Iω") && !occursin('|', k)
                @test length(res) == 2
                for point in res
                    @test point["wall_s"] > 0
                    ndatasets = 0
                    for (k, v) in point
                        isdiff(k) || continue
                        @test v < 1.0e-12 # same code, same settings: expect ~0
                        ndatasets += 1
                        # Both normalisations must be reported and self-consistent: the
                        # scan peak is a maximum over every point, so it is never below
                        # this point's own peak, and the scan-normalised difference is
                        # therefore never the larger of the two.
                        @test haskey(point, k * "|relscan")
                        @test point[k * "|refpeak"] > 0
                        @test point[k * "|scanpeak"] >= point[k * "|refpeak"]
                        @test point[k * "|relscan"] <= v + eps()
                    end
                    @test ndatasets == 3 # Iω_win, Iω_win_reimaged, Iω_full
                end
                # The scan peak is a property of the file, so every point must report the
                # same value for it — this is what makes the numbers comparable BETWEEN
                # points, which is the whole reason for reporting it.
                for k in filter(isdiff, collect(keys(res[1])))
                    @test res[1][k * "|scanpeak"] == res[2][k * "|scanpeak"]
                end
                # ...and at least one verified point must BE the scan peak here, since
                # both points of this two-point scan were verified.
                @test any(
                    isapprox(p[k * "|refpeak"], p[k * "|scanpeak"])
                        for p in res, k in filter(isdiff, collect(keys(res[1])))
                )

                # a deliberate grid change is detected as a (finite, nonzero) difference
                res640 = TS.verify_against_collected(
                    merge(setup_args, (; N = 48)),
                    collected, [1];
                    zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-6
                )
                @test all(
                    isfinite(v) && v > 1.0e-12
                        for (k, v) in res640[1] if isdiff(k)
                )
            end
        end
    end

    @testset "run_scan skip_existing (resume)" begin
        # scansave allocates the full (Nω, nz, Nτ) up front and fills points in, so an
        # all-zero slice means "not yet computed". Resume depends on that, and on
        # BatchExec preserving the ORIGINAL scanidx (RangeExec renumbers, which would
        # write points into the wrong slots).
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3, holediam = 0.25e-3,
            zmask = 0.1, apod = :supergauss, apod_param = 16
        )
        sa = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3, beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9), R = 40.0e-6, N = 32,
        )
        mktempdir() do tmpdir
            cd(tmpdir) do
                τ = [-1.0e-15, 0.0, 1.0e-15]
                @test isempty(TS._completed_scanidcs("res"))      # no file yet
                TS.run_scan(
                    sa, τ; scan_name = "res", exec = Luna.Scans.BatchExec(2, 1),
                    zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-6
                )
                partial = HDF5.h5open(f -> read(f["Iω_win"]), "res_collected.h5", "r")
                done = TS._completed_scanidcs("res")
                @test done == Set([1, 3])                          # batch 1 of 2
                @test all(any(!iszero, partial[:, :, i]) for i in done)
                @test !any(!iszero, partial[:, :, 2])

                TS.run_scan(
                    sa, τ; scan_name = "res", exec = Luna.Scans.LocalExec(),
                    zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-6, skip_existing = true
                )
                full = HDF5.h5open(f -> read(f["Iω_win"]), "res_collected.h5", "r")
                @test TS._completed_scanidcs("res") == Set([1, 2, 3])
                # The skipped points must be left exactly as they were, not recomputed.
                for i in done
                    @test partial[:, :, i] == full[:, :, i]
                end
                # ...and without skip_existing the same call recomputes everything.
                @test length(TS._completed_scanidcs("res")) == 3
            end
        end
    end

    @testset "streamed output equals in-memory output" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup = TS.build_setup(;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 10.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32
        )
        τ = 1.0e-15
        out_mem = TS.simulate_delay_point(setup, τ; nz = 3)
        fn = tempname() * "_pnps_test.h5"
        out_str = TS.simulate_delay_point(setup, τ; nz = 3, filename = fn)
        # identical stepping either way — the file only changes where bytes live
        @test out_str.Iω_win ≈ out_mem.Iω_win rtol = 1.0e-12
        @test out_str.Iω_win_reimaged ≈ out_mem.Iω_win_reimaged rtol = 1.0e-12
        @test out_str.Iω_full ≈ out_mem.Iω_full rtol = 1.0e-12
        @test out_str.zsave ≈ out_mem.zsave
        @test isfile(fn)   # user-supplied filename persists (run_scan cleans its own)
        rm(fn)
    end

    # -----------------------------------------------------------------------------
    # Regression: the input chirp must reach the GAUSSIAN beamlets themselves.
    #
    # `build_beamlets(::GaussianBeam, ...)` does not use the 1-D reference `Eω` that
    # `build_setup` applies the Taylor phase to — it builds its own field via
    # `Fields.GaussGaussField`, whose `ϕ` is a SCALAR carrier-envelope phase and so
    # cannot express GDD. Before the fix, `GDD=` was silently dropped for this beam
    # type: the *simulation* ran transform-limited.
    #
    # This is invisible in the obvious place. The `It_beamlet`/`Ito_beamlet` truth
    # diagnostics are reconstructed as sqrt(Iω_beamlet) x (phase of the 1-D `Eω`),
    # so they carried the chirp all along — a beamlet-FWHM test passes with or
    # without the fix, and pre-fix the recorded truth was chirped while the
    # simulated field was not. The assertion therefore has to look at the beamlet
    # FIELD, and it checks the applied spectral phase directly, which is exact and
    # grid-independent (an FWHM check is at the mercy of the test grid's coarse
    # time sampling).
    # -----------------------------------------------------------------------------
    @testset "GaussianBeam carries the input GDD" begin
        λ0, τfwhm, energy = 260.0e-9, 2.0e-15, 0.2e-6
        GDD = 2.0e-30
        grid = Grid.EnvGrid(10.0e-6, λ0, (200.0e-9, 400.0e-9), 40.0e-15)
        xygrid = Grid.FreeGrid(40.0e-6, 32)
        _, energyfun_ω = Luna.Fields.energyfuncs(grid, xygrid)
        beam = TS.GaussianBeam(8.3e-6, 0.1)
        geom = (; mask_diam = 1.0e-3, mask_spacing = 0.5e-3, f_foc = 0.1, λ0, τfwhm)
        FT1d = FFTW.plan_fft(copy(grid.t))

        function gate1(ϕ)
            Eω = Luna.Fields.GaussField(; λ0, τfwhm, energy, ϕ = ϕ)(grid, FT1d)
            first(
                TS.build_beamlets(
                    beam, grid, xygrid, geom, Eω, energy,
                    energyfun_ω; ϕ = ϕ
                )
            )
        end
        g0 = gate1([0.0, 0.0, 0.0, 0.0])
        g2 = gate1([0.0, 0.0, GDD, 0.0])

        # The transverse profile is omega-independent and the tilt is fixed, so one
        # fixed k-space sample tracks the beamlet at every frequency. Pick the
        # brightest.
        iω = argmax(dropdims(sum(abs2, g0; dims = (2, 3)); dims = (2, 3)))
        pk = argmax(abs2.(g0[iω, :, :]))
        iy, ix = pk[1], pk[2]
        E0 = g0[:, iy, ix]
        E2 = g2[:, iy, ix]

        # The chirp must actually reach the field.
        @test !isapprox(abs.(angle.(E2)), abs.(angle.(E0)); rtol = 1.0e-6)

        # `prop_taylor!` applies exp(-i phi) with phi = sum Δω^(n-1)/(n-1)! phi_n,
        # so the n=3 term contributes exp(-i GDD Δω^2 / 2): the phase difference is
        # a pure quadratic with coefficient -GDD/2. Restrict to samples that carry
        # signal and whose quadratic phase has not wrapped.
        Δω = grid.ω .- PhysData.wlfreq(λ0)
        significant = abs2.(E0) .> 1.0e-6 * maximum(abs2, E0)
        unwrapped = abs.(0.5 * GDD .* Δω .^ 2) .< 2.5
        ok = significant .& unwrapped
        @test count(ok) > 20
        dφ = angle.(E2[ok] ./ E0[ok])
        # Fit the least-squares quadratic through the origin.
        c = sum(dφ .* Δω[ok] .^ 2) / sum(Δω[ok] .^ 4)
        @test isapprox(c, -GDD / 2; rtol = 1.0e-3)
        # ...and the residual really is quadratic, not merely quadratic-ish.
        @test maximum(abs.(dφ .- c .* Δω[ok] .^ 2)) < 1.0e-6

        # Sign is carried through.
        gm = gate1([0.0, 0.0, -GDD, 0.0])
        Em = gm[:, iy, ix]
        dφm = angle.(Em[ok] ./ E0[ok])
        c_m = sum(dφm .* Δω[ok] .^ 2) / sum(Δω[ok] .^ 4)
        @test isapprox(c_m, +GDD / 2; rtol = 1.0e-3)

        # The HE11 builder inherits the chirp through `Eω` and must NOT double it:
        # passing phi as well changes nothing there.
        beamh = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        chirped_field = Luna.Fields.GaussField(;
            λ0, τfwhm, energy, ϕ = [0.0, 0.0, GDD, 0.0]
        )
        Eωc = chirped_field(grid, FT1d)
        h_noϕ = first(
            TS.build_beamlets(
                beamh, grid, xygrid, geom, Eωc, energy,
                energyfun_ω
            )
        )
        h_ϕ = first(
            TS.build_beamlets(
                beamh, grid, xygrid, geom, Eωc, energy,
                energyfun_ω; ϕ = [0.0, 0.0, GDD, 0.0]
            )
        )
        @test h_noϕ ≈ h_ϕ
    end

    # =============================================================================
    # Field mode: propagate the real, carrier-resolved field on a Luna RealGrid
    # =============================================================================

    @testset "field mode: grid, responses and metadata" begin
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        base = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3, beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9), R = 40.0e-6, N = 32,
        )

        se = TS.build_setup(; base...)
        sf = TS.build_setup(; base..., field_mode = true)

        @test se.grid isa Grid.EnvGrid
        @test sf.grid isa Grid.RealGrid
        @test sf.grid.ω[1] == 0                       # rfft half-spectrum starts at DC
        # A real-grid frequency axis is monotonic, unlike an envelope-grid axis.
        @test issorted(sf.grid.ω)
        @test !issorted(se.grid.ω)
        # every field-sized array follows the grid's ω axis
        @test size(sf.Eωk_g12, 1) == length(sf.grid.ω)
        @test size(sf.window_array, 1) == length(sf.grid.ω)
        @test length(sf.Eω) == length(sf.grid.ω)
        # An `rfft` output remains complex-valued.
        @test sf.Eω isa Vector{ComplexF64}

        # --- metadata contract ---
        @test se.combined_grid["field_mode"] == 0
        @test sf.combined_grid["field_mode"] == 1
        @test se.combined_grid["response"] == "kerr_env"
        @test sf.combined_grid["response"] == "nothg"
        @test TS.build_setup(;
            base..., field_mode = true,
            response = :thg
        ).combined_grid["response"] == "thg"
        @test sf.combined_grid["ffac"] == 6.0
        @test se.combined_grid["ffac"] == 0.0
        # RealGrid has no ω0 field, but readers key off /grid/ω0 unconditionally
        @test sf.combined_grid["ω0"] ≈ 2π * PhysData.c / 260.0e-9
        @test se.combined_grid["ω0"] ≈ 2π * PhysData.c / 260.0e-9

        # --- the pulse is the same pulse in either representation ---
        # Luna builds `√I cos(ω₀t)` on a real grid and `√I exp(iΔωt)` on an envelope
        # grid, so the envelope intensity `|A|²` agrees in both metadata records.
        @test maximum(sf.combined_grid["It"]) ≈
            maximum(se.combined_grid["It"]) rtol = 1.0e-4
        @test Luna.Maths.fwhm(sf.grid.t, sf.combined_grid["It"]) ≈
            Luna.Maths.fwhm(se.grid.t, se.combined_grid["It"]) rtol = 1.0e-2
        @test length(sf.combined_grid["It"]) == length(sf.grid.t)
        @test length(sf.combined_grid["Ito"]) == length(sf.combined_grid["To"])
        # ...and so does the energy actually put into each beamlet
        @test sf.energyfun_ω(sf.Eωk_t_base) ≈
            se.energyfun_ω(se.Eωk_t_base) rtol = 1.0e-3
        @test sf.energyfun_ω(sf.Eωk_g12) ≈ se.energyfun_ω(se.Eωk_g12) rtol = 1.0e-3

        # --- ffac ---
        s4 = TS.build_setup(; base..., field_mode = true, ffac = 4)
        @test length(s4.grid.to) == length(s4.grid.t)   # no oversampling at ffac = 4
        @test length(sf.grid.to) > length(sf.grid.t)    # ...but there is at the default 6
        @test s4.combined_grid["ffac"] == 4.0

        # --- refusals ---
        @test_throws ArgumentError TS.build_setup(;
            base..., field_mode = true, raman = true
        )
        @test_throws ArgumentError TS.build_setup(;
            base..., field_mode = true, response = :bogus
        )
        # the envelope path is unaffected by the new keywords
        @test_throws ArgumentError TS.build_setup(; base..., field_mode = true, ffac = 1)
    end

    @testset "field mode: :nothg and :thg agree when 3ω is outside the window" begin
        # Algebraically, the fundamental-band content of `E³` is exactly
        # `(3/4)Re(|E_a|²E_a) = (3/4)|E_a|²E`.
        # On a window that excludes the third harmonic entirely, here
        # 200-400 nm with a 260 nm carrier, whose 3ω band starts near 87 nm — the two
        # responses are the same operator and the propagations must agree to rounding.
        # Where the bands overlap, for a 1 fs pulse on the production window, they differ;
        # that difference is the physics the field mode exists to measure.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        base = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 2.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3, beam, window,
            trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 32, field_mode = true,
        )
        args = (; zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-10, twin_period = 1_000_000_000)
        setup_nothg = TS.build_setup(; base..., response = :nothg)
        setup_thg = TS.build_setup(; base..., response = :thg)
        on = TS.simulate_delay_point(setup_nothg, 0.0; args...)
        ot = TS.simulate_delay_point(setup_thg, 0.0; args...)
        for k in (:Iω_win, :Iω_win_reimaged, :Iω_full)
            a, b = getfield(on, k), getfield(ot, k)
            @test maximum(abs, a .- b) / maximum(a) < 1.0e-8
        end
    end

    @testset "field mode: field-versus-envelope trace agreement" begin
        # The port check, scaled down: the same geometry run both ways must give the same
        # trace. The two grids have different frequency axes and bin normalisations, so the
        # comparison uses physical spectral densities on the common band: `|E|²/Δω²`.
        # This is the frequency part of `Fields.energyfuncs`; the transverse part is common
        # to both and cancels), splined onto one axis.
        #
        # This configuration gives 3.2e-4 of trace peak, flat in depth, and unchanged to
        # three digits when pulse energy drops by a factor of 100. Depth-independent
        # and energy-independent means it is grid bookkeeping and spline interpolation, not
        # nonlinear physics — which is what "the port is wired correctly" looks like. The
        # interpolation part is confirmed by it GROWING for longer pulses (6.4e-4 at 4 fs,
        # 1.6e-3 at 8 fs), whose narrower spectra are sampled by fewer points.
        #
        # It is much larger on a 200-400 nm window, because the envelope grid's
        # relative-frequency window clips a 2 fs 260 nm spectrum. That is a property of the
        # toy window, not of the port, and it is exactly the kind of thing the field mode
        # exists to expose.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        base = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6,
            thickness = 5.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 0.5e-3, beam, window,
            trange = 120.0e-15, λlims = (143.0e-9, 600.0e-9), R = 40.0e-6, N = 32,
        )
        zs = [1.0e-6, 2.0e-6, 5.0e-6]
        args = (; zsave = zs, init_dz = 2.5e-7, rtol = 1.0e-9, twin_period = 1_000_000_000)
        oe = TS.simulate_delay_point(TS.build_setup(; base...), 0.0; args...)
        sf = TS.build_setup(; base..., field_mode = true)
        of = TS.simulate_delay_point(sf, 0.0; args...)

        # ascending ω axis and Δω-normalised spectral density, per grid type
        function density(grid, I)
            if grid isa Grid.RealGrid
                grid.ω, I ./ grid.ω[end]^2
            else
                (
                    FFTW.fftshift(grid.ω),
                    FFTW.fftshift(I, 1) ./ (length(grid.ω) * (grid.ω[2] - grid.ω[1]))^2,
                )
            end
        end
        se = TS.build_setup(; base...)
        rs = map(1:length(zs)) do iz
            ωe, De = density(se.grid, oe.Iω_win[:, iz])
            ωf, Df = density(sf.grid, of.Iω_win[:, iz])
            ωc = collect(
                range(
                    max(minimum(ωe[ωe .> 0]), ωf[2]),
                    min(maximum(ωe), ωf[end]), 600
                )
            )
            A = Luna.Maths.CSpline(ωe, De).(ωc)
            B = Luna.Maths.CSpline(ωf, Df).(ωc)
            m = A .> 1.0e-3 * maximum(A)
            sqrt(sum(abs2, (A .- B)[m]) / sum(abs2, A[m]))
        end
        @test all(<(2.0e-3), rs)
        # ...and, the point of reporting it per depth: it must not GROW with propagation.
        @test last(rs) < 2 * first(rs)
    end

    @testset "beamlet focal profile" begin
        # The stored truth `Eω_beamlet` is a hybrid: spatially integrated amplitude with
        # the one-dimensional input phase. Neither half necessarily describes the
        # three-field product driving the signal. This diagnostic stores the beamlet's
        # spatially resolved focal field so that the effective pulse can
        # be computed instead of assumed.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        win = TS.PhysicalMaskWindow(
            holex = -1.0e-3, holey = -1.0e-3, holediam = 0.5e-3,
            zmask = 0.1, apod = :tanh
        )
        # The production geometry, at reduced N and a short trange: mask_diam and R must be
        # physically consistent or the profile is of a spot that does not fit its own grid.
        base = (;
            λ0 = 260.0e-9, τfwhm = 1.0e-15, energy = 0.1e-6,
            thickness = 1.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 1.0e-3, beam, window = win,
            apod = :supergauss, apod_param = 16, trange = 40.0e-15,
            λlims = (143.0e-9, 600.0e-9), R = 366.0e-6, N = 192, store_window = false,
        )
        s = TS.build_setup(; base...)
        cg = s.combined_grid
        g, xy = s.grid, s.xygrid

        @test cg["beamlet_profile"] == 1
        @test cg["beamlet_r_which"] == "tg_gate1"
        r = cg["beamlet_r"]
        Er = cg["Eω_beamlet_r_re"] .+ 1im .* cg["Eω_beamlet_r_im"]
        @test length(r) == 64
        @test size(Er) == (length(g.ω), 64)
        @test r[1] == 0 && issorted(r)
        # the hole centre and the tilt coefficient make the file self-describing
        d = 1.0e-3 / 2 + 1.0e-3 / 2
        @test cg["beamlet_r_holex"] ≈ d && cg["beamlet_r_holey"] ≈ d
        @test cg["beamlet_r_tilt_coefx"] ≈ d / (PhysData.c * 0.1)
        @test length(cg["beamlet_r_asym"]) == length(g.ω)

        # --- 1. radial closure -------------------------------------------------------
        # Known a priori: integrating |E(ω,r)|² with the r dr Jacobian must reproduce the
        # stored `Iω_beamlet`. A mismatch identifies the centre, Jacobian, or normalisation
        # wrong — which is exactly how this went wrong once already (interpolating the raw
        # field, ~6 cycles of geometric tilt across the sampling radius, cost 20 % here).
        δx = xy.x[2] - xy.x[1]; δy = xy.y[2] - xy.y[1]
        function trapz(y, x)
            total = zero(eltype(y))
            for i in firstindex(y):(lastindex(y) - 1)
                total += (y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i])
            end
            return total
        end
        Iωb = cg["Iω_beamlet"]
        for λ in (200.0e-9, 260.0e-9, 350.0e-9)
            iω = argmin(abs.(g.ω .- 2π * PhysData.c / λ))
            rad = 2π * trapz(abs2.(Er[iω, :]) .* r, r) / (δx * δy)
            # ~1.5 % short at nr = 64: the Airy wings beyond rmax plus the real azimuthal
            # asymmetry. Bounded on BOTH sides — too much is as wrong as too little.
            @test 0.93 < rad / Iωb[iω] < 1.02
        end

        # --- 2. the width scales as λ ------------------------------------------------
        # The focal-spot area scales as `λ²`, so the on-axis spectrum
        # is bluer than the integrated one. If this scaling is absent the profile is not
        # describing an aperture-diffraction pattern and nothing downstream is meaningful.
        fwhm_of(iω) = (
            P = abs2.(Er[iω, :]); ih = findfirst(<(maximum(P) / 2), P);
            isnothing(ih) ? NaN : 2r[ih]
        )
        λs = (200.0e-9, 260.0e-9, 350.0e-9)
        ws = [fwhm_of(argmin(abs.(g.ω .- 2π * PhysData.c / λ))) for λ in λs]
        @test all(isfinite, ws)
        @test issorted(ws)                                  # wider at longer λ
        scaled = [w / (λ * 0.1 / 1.0e-3) for (w, λ) in zip(ws, λs)]   # w / (λf/D)
        @test maximum(scaled) / minimum(scaled) < 1.15        # constant to ~10 %

        # --- 3. azimuthal symmetry is good enough for a radial reduction -------------
        iω0 = argmin(abs.(g.ω .- 2π * PhysData.c / 260.0e-9))
        @test 0 < cg["beamlet_r_asym"][iω0] < 0.1

        # --- 4. switchable, and recorded either way ----------------------------------
        soff = TS.build_setup(; base..., beamlet_profile = false)
        @test soff.combined_grid["beamlet_profile"] == 0
        @test !haskey(soff.combined_grid, "beamlet_r")
        @test !haskey(soff.combined_grid, "Eω_beamlet_r_re")
        # ...and it does not disturb what was already there
        for k in ("Iω_beamlet", "Eω_beamlet_re", "Eω_beamlet_im", "It_beamlet")
            @test isequal(cg[k], soff.combined_grid[k])
        end

        # --- 5. DIAGNOSTIC ONLY: the propagation cannot see it -----------------------
        args = (; zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-8)
        setup_on = TS.build_setup(; base..., thickness = 1.0e-6)
        on = TS.simulate_delay_point(setup_on, 0.5e-15; args...)
        off = TS.simulate_delay_point(soff, 0.5e-15; args...)
        for k in (:Iω_win, :Iω_win_reimaged, :Iω_full)
            @test isequal(getfield(on, k), getfield(off, k))   # bit-identical, not approx
        end

        # --- 6. the Gaussian beam model carries it too -------------------------------
        Δk = 2π / 260.0e-9 * sin((1.0e-3 / 2 + 1.0e-3 / 2) / 0.1)
        gb = TS.build_setup(;
            base..., beam = TS.GaussianBeam(8.3e-6, 0.1),
            window = TS.PlanckWindow(
                kxc = -Δk, kyc = -Δk, kwidth = 2.5 / 8.3e-6,
                pad = 1.25
            )
        )
        @test gb.combined_grid["beamlet_r_which"] == "gaussian_gate1"
        @test size(gb.combined_grid["Eω_beamlet_r_re"]) == (length(gb.grid.ω), 64)
    end

    @testset "memory_budget matches what is actually allocated" begin
        # `memory_budget` tells the GPU drivers and benchmarks whether a
        # shape fits the card, and being wrong there costs an hour of rented GPU and a dead
        # process. So check the arithmetic against the buffers a real setup holds, in every
        # combination whose buffer set differs: the envelope fast path and field general
        # path with a pointwise response, a batched one, and no oversampling.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.5e-3, zmask = 0.1, apod = :tanh
        )
        N = 24
        base = (;
            λ0 = 260.0e-9, τfwhm = 1.0e-15, energy = 0.1e-6,
            thickness = 4.0e-6, material = :SiO2,
            mask_diam = 1.0e-3, mask_spacing = 1.0e-3, beam, window,
            apod = :supergauss, apod_param = 16, trange = 110.0e-15,
            λlims = (143.0e-9, 600.0e-9), R = 366.0e-6, N = N, store_window = false,
        )
        nbytes(x) = isnothing(x) ? 0 : length(x) * sizeof(eltype(x))
        for kw in (
                (;),
                (; field_mode = true, response = :nothg),
                (; field_mode = true, response = :thg),
                (; field_mode = true, response = :nothg, ffac = 4),
            )
            args = (; base..., kw...)
            b = TS.memory_budget(args)
            s = TS.build_setup(; args...)
            # The no-THG response allocates its analytic-signal buffer on its first call,
            # at setup — a card can look fine after build_setup and still die on the first
            # step. Propagate once so the count includes it; that is what the budget claims.
            TS.simulate_delay_point(s, 0.0; zsave = 2, init_dz = 2.0e-6, rtol = 1.0e-4)
            t = s.transform
            resp_buf = [r isa Luna.Nonlinear.KerrFieldNoTHG ? r.C : nothing for r in t.resp]
            seen = IdDict()
            actual = 0
            for buf in (t.Pto, t.Eto, t.Eωo, t.Pωo, t.Et_win, resp_buf...)
                isnothing(buf) && continue
                get!(seen, buf, false) && continue      # count an alias once
                seen[buf] = true
                actual += nbytes(buf)
            end
            Nω = length(s.grid.ω)
            actual += 9 * Nω * N * N * 16          # the RK45 registers
            actual += Nω * N * N * 8             # the device-resident extraction window
            @test b.Nω == Nω
            # `input` is the per-delay field from `delayed_input`, which is not a buffer the
            # setup holds, so account for it separately: here the beamlets are device-side
            # (three fields), and it is checked against the arrays the setup does hold.
            @test b.input ≈ 3 * Nω * N * N * 16 / 2^30
            @test (b.device - b.input) * 2^30 ≈ actual
            # ...and the structural facts the arithmetic rests on. The two paths encode "the
            # polarisation needs no buffer of its own" differently. The fast path leaves it
            # `nothing`, while the general path aliases it to the field, so test the
            # one of its two spellings.
            @test (isnothing(t.Pωo) ? isnothing(t.Eωo) : t.Pωo === t.Eωo)
            @test (isnothing(t.Pto) || t.Pto === t.Eto) == (b.pto == 0)
            @test isnothing(t.Et_win) == (length(s.grid.to) == length(s.grid.t))
        end
    end

    @testset "field mode: run_scan / load_simulated_scan round trip" begin
        # The whole file contract end to end. A field-mode file's `/grid/ω` is a monotonic
        # `rfft` half-spectrum with no `ω0` field on the grid, so a metadata slip here
        # would show up as a silently scrambled trace rather than an error.
        beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
        window = TS.PhysicalMaskWindow(
            holex = -0.75e-3, holey = -0.75e-3,
            holediam = 0.25e-3, zmask = 0.1,
            apod = :supergauss, apod_param = 16
        )
        setup_args = (;
            λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.2e-6, thickness = 1.0e-6,
            material = :SiO2, mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
            beam, window, trange = 20.0e-15, λlims = (200.0e-9, 400.0e-9),
            R = 40.0e-6, N = 16, field_mode = true,
        )
        mktempdir() do tmpdir
            cd(tmpdir) do
                TS.run_scan(
                    setup_args, [-1.0e-15, 1.0e-15]; scan_name = "fieldselftest",
                    exec = Luna.Scans.LocalExec(), zsave = 2,
                    init_dz = 5.0e-7, rtol = 1.0e-6
                )
                collected = "fieldselftest_collected.h5"
                @test isfile(collected)
                HDF5.h5open(collected) do f
                    @test read(f["grid"]["field_mode"]) == 1
                    @test read(f["grid"]["response"]) == "nothg"
                    @test read(f["grid"]["ffac"]) == 6.0
                    @test read(f["grid"]["ω0"]) ≈ 2π * PhysData.c / 260.0e-9
                    @test read(f["grid"]["delay_convention"]) == "gate"
                    @test issorted(read(f["grid"]["ω"]))
                end
                nt = TS.load_simulated_scan(collected)
                @test nt.field_mode
                @test issorted(nt.ω)
                @test nt.ω[1] == 0                       # untouched: no fftshift applied
                @test size(nt.trace) == (length(nt.ω), 2)
                @test all(isfinite, nt.trace)
                # ...and recomputing the same points must reproduce the file exactly
                res = TS.verify_against_collected(
                    setup_args, collected, [1, 2];
                    zsave = 2, init_dz = 5.0e-7, rtol = 1.0e-6
                )
                for point in res, (k, v) in point
                    startswith(k, "Iω") && !occursin('|', k) && @test v < 1.0e-12
                end
            end
        end
    end

    @testset "field mode: load_simulated_scan does not fftshift a monotonic axis" begin
        # A field-mode /grid/ω is already ascending; shifting it would scramble the trace
        # against its own axis. Files without a marker are envelope files and must shift;
        # is what every file written before field mode existed relies on.
        function writefile(fn, field_mode)
            Nω, nz, Nτ = 8, 1, 3
            ω = field_mode ? collect(1.0:Nω) : FFTW.fftshift(collect(1.0:Nω))
            HDF5.h5open(fn, "w") do f
                g = HDF5.create_group(f, "grid")
                g["ω"] = ω
                g["ω0"] = 4.5
                g["t"] = collect(1.0:Nω)
                g["Iω"] = collect(1.0:Nω)
                g["It"] = collect(1.0:Nω)
                g["τfwhm"] = 2.0e-15
                field_mode && (g["field_mode"] = 1)
                sv = HDF5.create_group(f, "scanvariables")
                sv["τ"] = collect(1.0:Nτ)
                f["Iω_win"] = reshape(repeat(collect(1.0:Nω), nz * Nτ), Nω, nz, Nτ)
            end
        end
        mktempdir() do dir
            fnf = joinpath(dir, "field.h5"); writefile(fnf, true)
            fne = joinpath(dir, "env.h5");   writefile(fne, false)
            nf = TS.load_simulated_scan(fnf)
            ne = TS.load_simulated_scan(fne)
            @test nf.field_mode
            @test !ne.field_mode
            @test issorted(nf.ω) && issorted(ne.ω)
            @test nf.ω == collect(1.0:8)                    # untouched
            @test nf.trace[:, 1] == collect(1.0:8)          # ...and so is the trace
            @test ne.trace[:, 1] == FFTW.fftshift(collect(1.0:8))
        end
    end

end # @testset "Trace simulation"
