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
    f         = 0.1
    mask_diam = 1e-3
    mask_spc  = 0.5e-3
    λmin, λmax = 160e-9, 500e-9

    R, N = TS.optimal_spatial_grid(f, mask_diam, mask_spc, λmin, λmax)

    # N is an even, 2,3,5-smooth FFT size (fast for FFTW without the up-to-2×
    # overshoot of nextpow(2, ...)).
    smooth(n) = (for p in (2, 3, 5); while n % p == 0; n ÷= p; end; end; n == 1)
    @test smooth(N)
    @test iseven(N)
    @test N > 0

    # The margin keyword scales the resolved size.
    _, N_bigmargin = TS.optimal_spatial_grid(f, mask_diam, mask_spc, λmin, λmax;
                                             margin=2.0)
    @test N_bigmargin > N

    # Real-space resolution: dx ≤ Airy(λmin) / pts_per_lobe (default 10).
    dx = 2R / N
    r_airy_min = 1.22 * λmin * f / mask_diam
    @test dx <= r_airy_min / 10 + 1e-12

    # k-space containment: kmax ≥ safety·3·2π·x_max/(λmin·f), default safety=1.5.
    kmax = π * N / (2R)
    x_max = mask_spc/2 + mask_diam
    k_NL_max = 1.5 * 3 * 2π * x_max / (λmin * f)
    @test kmax >= k_NL_max * (1 - 1e-12)

    # Larger safety produces ≥ as large N.
    R2, N2 = TS.optimal_spatial_grid(f, mask_diam, mask_spc, λmin, λmax;
                                      safety=3.0)
    @test N2 >= N
end

# -----------------------------------------------------------------------------
@testset "HE11Beam k-space construction" begin
    beam = TS.HE11Beam(125e-6, 5.0, 0.1)
    @test TS.a_scaled(beam) ≈ 2.5e-6

    # Tiny grid so the test is cheap.
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)

    # 1-D reference spectrum (using a Luna GaussField).
    FT1d = FFTW.plan_fft(copy(grid.t))
    Eω = Luna.Fields.GaussField(; λ0=260e-9, τfwhm=2e-15, energy=1e-9)(grid, FT1d)

    Eωk0 = TS.build_he11_kspace(grid, xygrid, beam, Eω)
    @test size(Eωk0) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    @test all(isfinite, Eωk0)

    # After IFFT to (y, x), the beam should peak near the centre pixel
    # (phase ramps in build_he11_kspace shift it from the FFTW corner).
    Eωxy0 = ifft(Eωk0, (2, 3))
    iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260e-9))
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
    @test energyfun_ω(Eωk_rescaled) ≈ target_E rtol=1e-10
end

# -----------------------------------------------------------------------------
@testset "GaussianBeam k-space construction" begin
    beam = TS.GaussianBeam(8.3e-6, 0.1)
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)

    target_E = 0.2e-6 / 3
    Eωk = TS.build_gaussian_kspace(grid, xygrid, beam, 260e-9, 2e-15, target_E)
    @test size(Eωk) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    @test all(isfinite, Eωk)

    # Total spectral energy via Parseval-based energyfun_ω.
    _, energyfun_ω = Luna.Fields.energyfuncs(grid, xygrid)
    @test energyfun_ω(Eωk) ≈ target_E rtol=5e-3
end

# -----------------------------------------------------------------------------
@testset "apply_tilt — k-space shift identity" begin
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)
    beam   = TS.GaussianBeam(8.3e-6, 0.1)
    Eωk0   = TS.build_gaussian_kspace(grid, xygrid, beam, 260e-9, 2e-15, 1e-9)
    Eωxy0  = ifft(Eωk0, (2, 3))

    # Δkx = Δky = 0 ⇒ identity (modulo numerical noise).
    Eωxy_id = TS.apply_tilt(Eωxy0, xygrid, 0.0, 0.0)
    @test maximum(abs.(Eωxy_id .- Eωxy0)) <= 1e-12 * maximum(abs.(Eωxy0))

    # Apply a tilt corresponding to a single k-space sample step in each axis.
    dkx = xygrid.kx[2] - xygrid.kx[1]
    dky = xygrid.ky[2] - xygrid.ky[1]
    Eωxy_t = TS.apply_tilt(Eωxy0, xygrid, dkx, dky)
    Eωk_t  = fft(Eωxy_t, (2, 3))

    # The peak in the (ky, kx) slice at the carrier ω should shift by exactly
    # one bin in each direction (centroid of |E|² before and after).
    iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260e-9))
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
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 16)
    Eωk    = randn(ComplexF64, length(grid.ω), length(xygrid.ky),
                              length(xygrid.kx))

    # τ=0 ⇒ identity.
    Eωk0 = TS.apply_delay(Eωk, grid, 0.0)
    @test maximum(abs.(Eωk0 .- Eωk)) <= 1e-12

    # τ ≠ 0: phase ramp matches -ω·τ at every (ω, ky, kx) where |E| > 0.
    τ = 1.5e-15
    Eωk_d = TS.apply_delay(Eωk, grid, τ)
    @test all(isfinite, Eωk_d)
    # Pick a few non-trivial ω indices and verify ratio.
    for iω in (3, length(grid.ω) ÷ 4, length(grid.ω) ÷ 2)
        ratio = Eωk_d[iω, 1, 1] / Eωk[iω, 1, 1]
        expected = exp(-1im * grid.ω[iω] * τ)
        @test ratio ≈ expected rtol=1e-10
    end
end

# -----------------------------------------------------------------------------
@testset "makemask — apodisation behaviour" begin
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)

    # Hard mask ⇒ binary.
    m_hard = TS.makemask(0.0, 0.0, 0.5e-3, grid, xygrid;
                          zmask=0.1, apod=:hard)
    @test all(v -> v == 0.0 || v == 1.0, m_hard)

    # supergauss ⇒ values in [0, 1], peak = 1 at hole centre at carrier ω.
    m_sg = TS.makemask(0.0, 0.0, 0.5e-3, grid, xygrid;
                        zmask=0.1, apod=:supergauss)
    @test all(0.0 .<= m_sg .<= 1.0)
    iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 260e-9))
    @test maximum(m_sg[iω0, :, :]) <= 1.0 + 1e-12

    # Chromatic vignetting: at twice the frequency, the hole's k-space radius
    # halves (because k_extent = (ω/c) · holediam/2 / zmask scales linearly
    # with ω, so larger ω ⇒ wider hole). Check that the count of mask points
    # above 0.5 increases when ω doubles.
    iω1 = argmin(abs.(grid.ω .- 2 * 2π * PhysData.c / 260e-9))
    if iω1 > 0 && iω1 != iω0
        n0 = count(>(0.5), m_hard[iω0, :, :])
        n1 = count(>(0.5), m_hard[iω1, :, :])
        @test n1 > n0
    end
end

# -----------------------------------------------------------------------------
@testset "PlanckWindow (ω-independent)" begin
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)

    Δk = 2π/260e-9 * sin(0.015)   # 15 mrad crossing
    w  = TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/8.3e-6, pad=1.25)
    arr = TS.build_window(w, grid, xygrid)
    @test size(arr) == (length(xygrid.ky), length(xygrid.kx))
    @test all(0.0 .<= arr .<= 1.0)
    @test maximum(arr) ≈ 1.0 atol=1e-12
end

# -----------------------------------------------------------------------------
@testset "PlanckOmegaWindow (ω-dependent)" begin
    grid   = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    xygrid = Grid.FreeGrid(40e-6, 32)

    w   = TS.PlanckOmegaWindow(xc=-0.75e-3, yc=-0.75e-3,
                                holediam=0.5e-3, f_foc=0.1, pad=1.25)
    arr = TS.build_window(w, grid, xygrid)
    @test size(arr) == (length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    @test all(0.0 .<= arr .<= 1.0)

    # The hole's k-space half-width khole(ω) = (ω/c)·(holediam/2)/f_foc grows
    # linearly with ω. Count of pixels above 0.5 in the per-ω slice should
    # increase with ω.
    iω0 = argmin(abs.(grid.ω .- 2π * PhysData.c / 400e-9))   # low freq
    iω1 = argmin(abs.(grid.ω .- 2π * PhysData.c / 200e-9))   # high freq
    if iω0 != iω1
        n0 = count(>(0.5), arr[iω0, :, :])
        n1 = count(>(0.5), arr[iω1, :, :])
        @test n1 >= n0
    end
end

# -----------------------------------------------------------------------------
@testset "extract_signal_spectra (skip_propagation)" begin
    # Build a small HE11+PhysicalMask setup.
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)

    out = TS.simulate_delay_point(setup, 0.0; skip_propagation=true, nz=2)
    @test haskey(out, :Iω_win)
    @test haskey(out, :Iω_win_reimaged)
    @test haskey(out, :Iω_full)
    @test size(out.Iω_win)        == (length(setup.grid.ω), 2)
    @test size(out.Iω_win_reimaged) == (length(setup.grid.ω), 2)
    @test size(out.Iω_full)       == (length(setup.grid.ω), 2)
    @test all(out.Iω_win .>= 0)
    @test all(out.Iω_win_reimaged .>= 0)
    @test all(out.Iω_full .>= 0)
    @test all(isfinite, out.Iω_full)

    # Same with a non-zero delay — should still produce finite, non-negative
    # spectra of identical shape.
    out2 = TS.simulate_delay_point(setup, 1.0e-15; skip_propagation=true, nz=2)
    @test size(out2.Iω_win) == size(out.Iω_win)
    @test all(isfinite, out2.Iω_win)
end

# -----------------------------------------------------------------------------
@testset "extract_signal_spectra: one-pass equals FFT reference" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    Ez = TS.delayed_input(setup, 0.7e-15) # non-trivial complex field
    for warr in (setup.window_array,                        # 3-D window
                 abs.(randn(Xoshiro(7), 32, 32)))           # 2-D window
        a1, b1 = TS.extract_signal_spectra(Ez, warr, setup.xygrid)
        a2, b2 = TS._extract_signal_spectra_fft(Ez, warr, setup.xygrid)
        @test isapprox(a1, a2, rtol=1e-12)
        @test isapprox(b1, b2, rtol=1e-10, atol=1e-12*maximum(b2))
    end
    # quadrant spectrum matches the broadcast it replaced
    sig_quad = (setup.xygrid.ky .< 0) .& (setup.xygrid.kx .< 0)'
    sq3 = reshape(sig_quad, (1, size(sig_quad)...))
    ref = dropdims(sum(abs2.(Ez) .* sq3; dims=(2, 3)); dims=(2, 3))
    out = zeros(length(setup.grid.ω))
    TS._quadrant_spectrum!(out, Ez, sig_quad)
    @test isapprox(out, ref, rtol=1e-12)
end

# -----------------------------------------------------------------------------
@testset "window storage options" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    kwargs = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                thickness=10e-6, material=:SiO2,
                mask_diam=1.0e-3, mask_spacing=0.5e-3,
                beam, window,
                trange=20e-15, λlims=(200e-9, 400e-9),
                R=40e-6, N=32)
    # default: window array stored, parameters always stored (flattened scalars, so
    # scansave can write them as plain HDF5 datasets)
    s1 = TS.build_setup(; kwargs...)
    @test haskey(s1.combined_grid, "window")
    @test s1.combined_grid["window_def_type"] == "PhysicalMaskWindow"
    @test s1.combined_grid["window_def_holediam"] == 0.25e-3
    # store_window=false: parameters only (the array is ~1 GiB at production size)
    s2 = TS.build_setup(; kwargs..., store_window=false)
    @test !haskey(s2.combined_grid, "window")
    @test haskey(s2.combined_grid, "window_def_type")
    # the in-memory window array is unaffected
    @test isequal(s2.window_array, s1.window_array)
end

# -----------------------------------------------------------------------------
@testset "_resolve_zsave" begin
    # Integer: uniform grid over [0, zmax], reproduces legacy nz behaviour.
    @test TS._resolve_zsave(2, 10e-6) == [0.0, 10e-6]
    @test TS._resolve_zsave(3, 10e-6) ≈ [0.0, 5e-6, 10e-6]
    @test_throws ArgumentError TS._resolve_zsave(1, 10e-6)        # need ≥ 2

    # Vector: zmax appended when absent, not duplicated when present.
    @test TS._resolve_zsave([2e-6, 6e-6], 10e-6) == [2e-6, 6e-6, 10e-6]
    @test TS._resolve_zsave([2e-6, 10e-6], 10e-6) == [2e-6, 10e-6]
    @test TS._resolve_zsave([1e-6, 10e-6, 20e-6, 40e-6], 40e-6) ==
          [1e-6, 10e-6, 20e-6, 40e-6]

    # Validation failures.
    @test_throws ArgumentError TS._resolve_zsave([6e-6, 2e-6], 10e-6)   # unsorted
    @test_throws ArgumentError TS._resolve_zsave([2e-6, 2e-6], 10e-6)   # duplicate
    @test_throws ArgumentError TS._resolve_zsave([20e-6], 10e-6)        # > zmax
    @test_throws ArgumentError TS._resolve_zsave([-1e-6, 5e-6], 10e-6)  # negative z

    # Idempotent: re-resolving an already-resolved grid (which the integer path
    # produces with an entrance slice at z=0) returns it unchanged. This is the
    # path run_scan exercises when it forwards the resolved vector per delay.
    @test TS._resolve_zsave(TS._resolve_zsave(11, 40e-6), 40e-6) ==
          TS._resolve_zsave(11, 40e-6)
    @test TS._resolve_zsave([0.0, 5e-6, 10e-6], 10e-6) == [0.0, 5e-6, 10e-6]
end

# -----------------------------------------------------------------------------
@testset "simulate_delay_point — zsave snapshots (skip_propagation)" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    Nω = length(setup.grid.ω)

    # Explicit thickness list (already ending at zmax): 3 slices.
    out = TS.simulate_delay_point(setup, 0.0; skip_propagation=true,
                                   zsave=[2e-6, 6e-6, 10e-6])
    @test out.zsave == [2e-6, 6e-6, 10e-6]
    @test size(out.Iω_win) == (Nω, 3)
    @test size(out.Iω_full) == (Nω, 3)

    # zmax appended automatically when absent.
    out2 = TS.simulate_delay_point(setup, 0.0; skip_propagation=true,
                                    zsave=[2e-6, 6e-6])
    @test out2.zsave == [2e-6, 6e-6, 10e-6]
    @test size(out2.Iω_win) == (Nω, 3)

    # Integer zsave and the default both reproduce the legacy nz=2 path.
    out3 = TS.simulate_delay_point(setup, 0.0; skip_propagation=true, zsave=2)
    @test out3.zsave == [0.0, 10e-6]
    @test size(out3.Iω_win) == (Nω, 2)
    out4 = TS.simulate_delay_point(setup, 0.0; skip_propagation=true)
    @test size(out4.Iω_win) == (Nω, 2)
end

# -----------------------------------------------------------------------------
@testset "extract_signal_spectra — multi-window" begin
    # Build a small Gaussian + two-window setup.
    beam = TS.GaussianBeam(8.3e-6, 0.1)
    Δk   = 2π/260e-9 * sin((0.5e-3/2 + 1.0e-3/2)/0.1)
    windows = [TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/8.3e-6, pad=1.25),
               TS.PlanckOmegaWindow(xc=-0.75e-3, yc=-0.75e-3,
                                     holediam=0.5e-3, f_foc=0.1, pad=1.25)]
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window=windows,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)

    out = TS.simulate_delay_point(setup, 0.0; skip_propagation=true, nz=2)
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
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    cg = setup.combined_grid
    for key in ("Iω", "It", "To", "Ito", "τfwhm", "material", "thickness",
                "Iω_beamlet", "It_beamlet", "Ito_beamlet",
                "a", "a_scaled", "f_coll", "f_foc", "window")
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
    Δk = 2π/260e-9 * sin((0.5e-3/2 + 1.0e-3/2)/0.1)
    windows = [TS.PlanckWindow(kxc=-Δk, kyc=-Δk, kwidth=2.5/8.3e-6, pad=1.25),
               TS.PlanckOmegaWindow(xc=-0.75e-3, yc=-0.75e-3,
                                     holediam=0.5e-3, f_foc=0.1, pad=1.25)]
    setupg = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                              thickness=10e-6, material=:SiO2,
                              mask_diam=1.0e-3, mask_spacing=0.5e-3,
                              beam=beamg, window=windows,
                              trange=20e-15, λlims=(200e-9, 400e-9),
                              R=40e-6, N=32)
    cgg = setupg.combined_grid
    for key in ("w0", "Δk", "crossingθ", "window", "window_ωdep",
                "Iω_beamlet", "It_beamlet", "Ito_beamlet")
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
    @test maximum(abs.(nb .- ni)) < 5e-2
end

# -----------------------------------------------------------------------------
@testset "Smoke test: tiny end-to-end Luna.run" begin
    # The smallest grid that still exercises the whole pipeline. Runs in a
    # few seconds on a laptop. If Luna or its FFT plans fail to set up,
    # the test fails noisily — that's intentional, this is the only place
    # the suite proves the integration boundary works.
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=1e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)

    out = TS.simulate_delay_point(setup, 0.0; nz=2, init_dz=5e-7)
    @test haskey(out, :Iω_win)
    @test haskey(out, :Iω_win_reimaged)
    @test haskey(out, :Iω_full)

    # factored (lazy) linop/norm vs materialised: bit-identical end to end
    setup_mat = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                                 thickness=1e-6, material=:SiO2,
                                 mask_diam=1.0e-3, mask_spacing=0.5e-3,
                                 beam, window,
                                 trange=20e-15, λlims=(200e-9, 400e-9),
                                 R=40e-6, N=32, factored_linop=false)
    out_mat = TS.simulate_delay_point(setup_mat, 0.0; nz=2, init_dz=5e-7)
    @test isequal(out_mat.Iω_win, out.Iω_win)
    @test isequal(out_mat.Iω_win_reimaged, out.Iω_win_reimaged)
    @test isequal(out_mat.Iω_full, out.Iω_full)

    # batched vs frozen (legacy per-column) Raman: agreement to rounding accuracy
    # through a real propagation
    ram_kwargs = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                    thickness=1e-6, material=:SiO2,
                    mask_diam=1.0e-3, mask_spacing=0.5e-3,
                    beam, window,
                    trange=20e-15, λlims=(200e-9, 400e-9),
                    R=40e-6, N=32, raman=true)
    out_bat = TS.simulate_delay_point(TS.build_setup(; ram_kwargs...), 0.0;
                                      nz=2, init_dz=5e-7)
    out_frz = TS.simulate_delay_point(TS.build_setup(; ram_kwargs...,
                                                       raman_impl=:frozen), 0.0;
                                      nz=2, init_dz=5e-7)
    @test isapprox(out_bat.Iω_win, out_frz.Iω_win, rtol=1e-8)
    @test isapprox(out_bat.Iω_full, out_frz.Iω_full, rtol=1e-8)
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
    outz = TS.simulate_delay_point(setup, 0.0; zsave=[0.5e-6, 1e-6], init_dz=5e-7)
    @test outz.zsave ≈ [0.5e-6, 1e-6] atol=1e-15
    @test size(outz.Iω_win) == (length(setup.grid.ω), 2)
    @test any(outz.Iω_win[:, 1] .!= outz.Iω_win[:, 2])

    # `filename` swaps the in-memory MemoryOutput for an HDF5Output. The
    # extracted spectra must be identical to the in-memory run, and the file
    # must hold the raw Eω/z propagation datasets.
    mktempdir() do dir
        fpath = joinpath(dir, "delay.h5")
        outf = TS.simulate_delay_point(setup, 0.0; nz=2, init_dz=5e-7,
                                        filename=fpath)
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

function _write_mock_scan_file(path; Nω=64, Nt=64, Nτ=8, nz=2,
                                 with_beamlet=true,
                                 with_omega_dep=false,
                                 zsave=nothing,
                                 ω0=2π * 2.99792458e8 / 260e-9,
                                 dω=1e13, dt=1e-15)
    # Build an FFT-ordered ω vector centred on 0: [0, dω, ..., (N/2-1)dω, -N/2 dω, ..., -dω]
    halfN = Nω ÷ 2
    ω_fft = [0:halfN-1; -halfN:-1] .* dω             # FFT-ordered (relative to ω0)
    ω_abs = ω_fft .+ ω0                              # absolute frequency
    Iω_fft = abs2.(exp.(-(ω_fft ./ (5dω)).^2))       # Gaussian centred at DC bin
    t = collect(((-Nt÷2):(Nt÷2 - 1))) .* dt
    It = abs2.(exp.(-(t ./ (5dt)).^2))
    τ = collect(((-Nτ÷2):(Nτ÷2 - 1))) .* (2 * dt)

    # FROG trace: random non-negative, FFT-ordered along ω axis
    rng_seed = 1234
    rand_arr = rand(MersenneTwister(rng_seed), Nω, nz, Nτ)

    HDF5.h5open(path, "w") do f
        g = HDF5.create_group(f, "grid")
        g["ω"]      = ω_abs                # NB: scansave saves the *absolute* ω in FFT order
        g["ω0"]     = ω0
        g["t"]      = t
        g["Iω"]     = Iω_fft               # in same FFT order as ω
        g["It"]     = It
        g["τfwhm"]  = 2.0e-15
        if with_beamlet
            g["Iω_beamlet"]  = Iω_fft .* 0.7   # smaller (vignetted)
            g["It_beamlet"]  = It .* 0.7        # beamlet temporal intensity
        end
        if !isnothing(zsave)
            g["zsave"] = collect(Float64, zsave)
        end
        sv = HDF5.create_group(f, "scanvariables")
        sv["τ"] = τ
        f["Iω_win"] = rand_arr
        f["Iω_win_reimaged"] = rand_arr .* 0.5
        f["Iω_full"] = rand_arr .* 2.0           # full signal-collection reference (≥ windowed)
        if with_omega_dep
            f["Iω_win_ωdep"]          = rand_arr .* 0.8
            f["Iω_win_ωdep_reimaged"] = rand_arr .* 0.4
        end
    end
    return ω_abs, ω_fft, Iω_fft, t, It, τ
end

@testset "load_simulated_scan — basic round-trip" begin
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "mock_scan.h5")
        ω_abs, ω_fft, Iω_fft, t, It, τ =
            _write_mock_scan_file(path; Nω=32, Nτ=4, nz=2, with_beamlet=true)

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
        @test nt.ω0 ≈ 2π * 2.99792458e8 / 260e-9

        # z_index defaults to :end (= last z slice)
        @test all(nt.trace .>= 0)

        # Selecting a different window
        nt_re = TS.load_simulated_scan(path; window_key="Iω_win_reimaged")
        @test nt_re.trace ≈ nt.trace .* 0.5

        # The full signal-collection reference loads via window_key and is ≥
        # the windowed trace everywhere (collection efficiency ≤ 1).
        nt_full = TS.load_simulated_scan(path; window_key="Iω_full")
        @test nt_full.trace ≈ nt.trace .* 2.0
        @test all(nt.trace .<= nt_full.trace .+ 1e-20)

        # z_index=1 picks the first slice
        nt_z1 = TS.load_simulated_scan(path; z_index=1)
        @test size(nt_z1.trace) == (32, 4)

        # Bad window key raises
        @test_throws Exception TS.load_simulated_scan(path; window_key="not_a_key")

        # Back-compat: a file with no /grid/zsave loads fine and omits :zsave.
        @test !haskey(nt, :zsave)
    end
end

# -----------------------------------------------------------------------------
@testset "load_simulated_scan — multi-z (zsave, :all, z_thickness)" begin
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "mock_scan_z.h5")
        zvec = [1e-6, 10e-6, 20e-6, 40e-6]
        _write_mock_scan_file(path; Nω=32, Nτ=4, nz=4, zsave=zvec)

        # zsave round-trips; default :end still returns a single 2-D slice.
        nt = TS.load_simulated_scan(path)
        @test haskey(nt, :zsave)
        @test nt.zsave == zvec
        @test size(nt.trace) == (32, 4)

        # :all returns the full (Nω, nz, Nτ) stack; fftshift only along ω, so
        # the last slice equals the default :end load.
        nt_all = TS.load_simulated_scan(path; z_index=:all)
        @test size(nt_all.trace) == (32, 4, 4)
        @test nt_all.zsave == zvec
        @test nt_all.trace[:, end, :] ≈ nt.trace

        # z_thickness selects the nearest saved slice (11 µm → 10 µm = index 2).
        nt_t = TS.load_simulated_scan(path; z_thickness=11e-6)
        @test nt_t.trace ≈ nt_all.trace[:, 2, :]

        # z_thickness on a file without /grid/zsave errors.
        path2 = joinpath(tmpdir, "mock_no_z.h5")
        _write_mock_scan_file(path2; Nω=32, Nτ=4, nz=2)
        @test_throws Exception TS.load_simulated_scan(path2; z_thickness=10e-6)
    end
end

# -----------------------------------------------------------------------------
@testset "FrozenRamanPolarEnv matches Luna's RamanPolarEnv" begin
    grid  = Grid.EnvGrid(10e-6, 260e-9, (200e-9, 400e-9), 20e-15)
    scale = 0.18 * PhysData.ε_0 * PhysData.χ3(:SiO2)
    Rluna = Luna.Nonlinear.RamanPolarEnv(
        grid.to, Luna.Raman.raman_response(grid.to, :SiO2, scale))
    Rfroz = TS.FrozenRamanPolarEnv(
        grid.to, Luna.Raman.raman_response(grid.to, :SiO2, scale))

    # Realistic field amplitude (V/m): with O(1) fields the polarisation
    # ~ ε₀χ³|E|²E ≈ 1e-34 sits below the ulp of any O(1) accumulation buffer
    # and the additivity check would be vacuous.
    rng = MersenneTwister(7)
    Et  = 1e12 .* randn(rng, ComplexF64, length(grid.to))

    P1 = zeros(ComplexF64, length(grid.to))
    P2 = zeros(ComplexF64, length(grid.to))
    Rluna(P1, Et, 1.0)
    Rfroz(P2, Et, 1.0)
    @test maximum(abs, P1) > 0          # response not trivially zero
    @test P1 ≈ P2 rtol=1e-13

    # Accumulation semantics: the response must ADD to a nonzero buffer.
    seed = maximum(abs, P1)
    outa = fill(complex(seed), length(grid.to))
    Rfroz(outa, Et, 1.0)
    @test outa .- seed ≈ P2 rtol=1e-8   # looser: cancellation in the subtraction

    # Column-matrix branch (N, 1), as used by scalar modal paths.
    Etm   = reshape(Et, :, 1)
    out1m = zeros(ComplexF64, size(Etm))
    out2m = zeros(ComplexF64, size(Etm))
    Rluna(out1m, Etm, 1.0)
    Rfroz(out2m, Etm, 1.0)
    @test out1m ≈ out2m rtol=1e-13

    # Second call with fresh data: buffers are reused, kernel stays frozen.
    Et2 = 1e12 .* randn(rng, ComplexF64, length(grid.to))
    fill!(P1, 0); fill!(P2, 0)
    Rluna(P1, Et2, 1.0)
    Rfroz(P2, Et2, 1.0)
    @test P1 ≈ P2 rtol=1e-13
end

# -----------------------------------------------------------------------------
@testset "build_setup raman keyword" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    kwargs = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                thickness=10e-6, material=:SiO2,
                mask_diam=1.0e-3, mask_spacing=0.5e-3,
                beam, window,
                trange=20e-15, λlims=(200e-9, 400e-9),
                R=40e-6, N=32)

    # Default: Raman off, recorded as such in the metadata.
    setup0 = TS.build_setup(; kwargs...)
    @test setup0.combined_grid["raman"] == 0

    # Raman on: builds cleanly and records the model in the metadata.
    setup1 = TS.build_setup(; kwargs..., raman=true, raman_fraction=0.2)
    @test setup1.combined_grid["raman"] == 1
    @test setup1.combined_grid["raman_fraction"] == 0.2

    # Fake-propagation signal extraction still works with the Raman setup —
    # a smoke check that the two-response pipeline is wired through.
    out = TS.simulate_delay_point(setup1, 0.0; skip_propagation=true, nz=2)
    @test all(isfinite, out.Iω_win)

    # Materials without a condensed-phase (:intermediate) Raman model raise.
    @test_throws ErrorException TS.build_setup(; kwargs..., material=:N2,
                                                 raman=true)
end

# -----------------------------------------------------------------------------
@testset "Raman split: quasi-static limit recovers the Kerr-only response" begin
    # The defining property of the envelope-defined f_R (prop_gnlse
    # convention): for a pulse much longer than the Raman memory,
    # h_R ⊛ |E|² → |E|², so Kerr((1-f_R)χ³) + Raman must equal Kerr(χ³).
    # This pins the relative Kerr/Raman normalisation — with the (3/2)-less
    # scale of Luna's low-level examples the totals differ by ~6%, well
    # outside the tolerance below.
    grid = Grid.EnvGrid(10e-6, 260e-9, (255e-9, 265e-9), 20e-12)
    fr   = 0.18
    χ3   = PhysData.χ3(:SiO2)

    K_full  = Luna.Nonlinear.Kerr_env(χ3)
    K_part  = Luna.Nonlinear.Kerr_env((1 - fr) * χ3)
    R_part  = TS.FrozenRamanPolarEnv(
        grid.to, Luna.Raman.raman_response(grid.to, :SiO2,
                                           1.5 * fr * PhysData.ε_0 * χ3))

    # 5 ps FWHM Gaussian envelope: ~50x the Raman memory.
    Et = ComplexF64.(1e12 .* exp.(-2 * log(2) .* (grid.to ./ 5e-12) .^ 2))

    P_full = zeros(ComplexF64, length(grid.to))
    K_full(P_full, Et, 1.0)
    P_split = zeros(ComplexF64, length(grid.to))
    K_part(P_split, Et, 1.0)
    R_part(P_split, Et, 1.0)

    @test P_split ≈ P_full rtol=2e-2
    @test isapprox(maximum(abs, P_split), maximum(abs, P_full); rtol=1e-2)
end

# -----------------------------------------------------------------------------
@testset "delay convention: gate frame (probe delayed by -tau)" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    τ = 1.5e-15
    # The scan input must carry the probe at -τ (gate-delay/paper convention).
    got = TS.delayed_input(setup, τ)
    @test got ≈ setup.Eωk_g12 .+
                TS.apply_delay(setup.Eωk_t_base, setup.grid, -τ)
    @test !(got ≈ setup.Eωk_g12 .+
                  TS.apply_delay(setup.Eωk_t_base, setup.grid, τ))
    # τ = 0 is the identity for both conventions.
    @test TS.delayed_input(setup, 0.0) ≈
          setup.Eωk_g12 .+ setup.Eωk_t_base
end

# -----------------------------------------------------------------------------
@testset "complex beamlet spectrum stored with phase" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    kwargs = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                thickness=10e-6, material=:SiO2,
                mask_diam=1.0e-3, mask_spacing=0.5e-3,
                beam, window,
                trange=20e-15, λlims=(200e-9, 400e-9),
                R=40e-6, N=32)

    # Transform-limited input: flat phase, so the beamlet spectrum is real.
    s0 = TS.build_setup(; kwargs...)
    cg = s0.combined_grid
    @test haskey(cg, "Eω_beamlet_re") && haskey(cg, "Eω_beamlet_im")
    E0 = cg["Eω_beamlet_re"] .+ im .* cg["Eω_beamlet_im"]
    @test abs2.(E0) ≈ cg["Iω_beamlet"] rtol=1e-10
    @test maximum(abs, cg["Eω_beamlet_im"]) < 1e-8 * maximum(abs, E0)
    # source spectrum stored too, consistent with Iω
    @test abs2.(cg["Eω_re"] .+ im .* cg["Eω_im"]) ≈ cg["Iω"] rtol=1e-10

    # Chirped input: the phase must survive into the stored beamlet.
    s2 = TS.build_setup(; kwargs..., GDD=2e-30)
    cg2 = s2.combined_grid
    E2 = cg2["Eω_beamlet_re"] .+ im .* cg2["Eω_beamlet_im"]
    @test abs2.(E2) ≈ cg2["Iω_beamlet"] rtol=1e-10   # amplitude unchanged
    @test maximum(abs, cg2["Eω_beamlet_im"]) > 1e-3 * maximum(abs, E2)
end

# -----------------------------------------------------------------------------
@testset "signal_quadrant_norm" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    qnorm = TS.signal_quadrant_norm(setup; floor_rel=1e-6)
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

    rtol, atol = 1e-6, 1e-10
    # Pump-dominated field with a weak signal; error concentrated in the
    # signal quadrant. The quadrant norm must flag what weaknorm misses.
    y = field(1e-3, 1.0)
    err_sig = field(1e-6, 0.0)          # error only in the signal quadrant
    @test qnorm(err_sig, y, y, rtol, atol) > 100 * wknrm(err_sig, y, y, rtol, atol)
    # ... because the signal-relative error is ~1e-3/rtol.

    # Error in the pump region: both norms agree to within the region split.
    err_pump = field(0.0, 1e-6)
    r = qnorm(err_pump, y, y, rtol, atol) / wknrm(err_pump, y, y, rtol, atol)
    @test 0.5 < r < 2.0

    # Empty signal quadrant: the floor prevents 0/0 blow-up; finite result.
    y0 = field(0.0, 1.0)
    e0 = field(1e-9, 0.0)
    v = qnorm(e0, y0, y0, rtol, atol)
    @test isfinite(v) && v > 0
    # The floor bounds the value: err ≤ ||err_sig|| / (rtol · floor_rel·||rest||)
    n_sig = sqrt(count(sigmask) * Nω) * 1e-9
    n_rest = sqrt(count(.!sigmask) * Nω) * 1.0
    @test v ≤ n_sig / (rtol * 1e-6 * n_rest) * (1 + 1e-9)

    # Wrong grid size errors loudly.
    bad = zeros(ComplexF64, (Nω, 16, 16))
    @test_throws DimensionMismatch qnorm(bad, bad, bad, rtol, atol)

    # Provenance labels.
    @test occursin("signal_quadrant", TS._norm_name(qnorm))
    @test TS._norm_name(Luna.RK45.weaknorm) == "weaknorm"
end

# -----------------------------------------------------------------------------
@testset "verify_against_collected round-trip" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup_args = (; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                    thickness=1e-6, material=:SiO2,
                    mask_diam=1.0e-3, mask_spacing=0.5e-3,
                    beam, window,
                    trange=20e-15, λlims=(200e-9, 400e-9),
                    R=40e-6, N=32)
    mktempdir() do tmpdir
        cd(tmpdir) do
            τs = [-1e-15, 1e-15]
            TS.run_scan(setup_args, τs; scan_name="verify_selftest",
                        exec=Luna.Scans.LocalExec(), zsave=2,
                        init_dz=5e-7, rtol=1e-6)
            collected = "verify_selftest_collected.h5"
            @test isfile(collected)
            res = TS.verify_against_collected(setup_args, collected, [1, 2];
                                              zsave=2, init_dz=5e-7, rtol=1e-6)
            # The `|`-suffixed keys are the normalisation diagnostics, not differences.
            isdiff(k) = startswith(k, "Iω") && !occursin('|', k)
            @test length(res) == 2
            for point in res
                @test point["wall_s"] > 0
                ndatasets = 0
                for (k, v) in point
                    isdiff(k) || continue
                    @test v < 1e-12 # same code, same settings: expect ~0
                    ndatasets += 1
                    # Both normalisations must be reported and self-consistent: the
                    # scan peak is a maximum over every point, so it is never below
                    # this point's own peak, and the scan-normalised difference is
                    # therefore never the larger of the two.
                    @test haskey(point, k*"|relscan")
                    @test point[k*"|refpeak"] > 0
                    @test point[k*"|scanpeak"] >= point[k*"|refpeak"]
                    @test point[k*"|relscan"] <= v + eps()
                end
                @test ndatasets == 3 # Iω_win, Iω_win_reimaged, Iω_full
            end
            # The scan peak is a property of the file, so every point must report the
            # same value for it — this is what makes the numbers comparable BETWEEN
            # points, which is the whole reason for reporting it.
            for k in filter(isdiff, collect(keys(res[1])))
                @test res[1][k*"|scanpeak"] == res[2][k*"|scanpeak"]
            end
            # ...and at least one verified point must BE the scan peak here, since
            # both points of this two-point scan were verified.
            @test any(isapprox(p[k*"|refpeak"], p[k*"|scanpeak"])
                      for p in res, k in filter(isdiff, collect(keys(res[1]))))

            # a deliberate grid change is detected as a (finite, nonzero) difference
            res640 = TS.verify_against_collected(merge(setup_args, (; N=48)),
                                                 collected, [1];
                                                 zsave=2, init_dz=5e-7, rtol=1e-6)
            @test all(isfinite(v) && v > 1e-12
                      for (k, v) in res640[1] if isdiff(k))
        end
    end
end

@testset "streamed output equals in-memory output" begin
    beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(holex=-0.75e-3, holey=-0.75e-3,
                                    holediam=0.25e-3, zmask=0.1,
                                    apod=:supergauss, apod_param=16)
    setup = TS.build_setup(; λ0=260e-9, τfwhm=2e-15, energy=0.2e-6,
                             thickness=10e-6, material=:SiO2,
                             mask_diam=1.0e-3, mask_spacing=0.5e-3,
                             beam, window,
                             trange=20e-15, λlims=(200e-9, 400e-9),
                             R=40e-6, N=32)
    τ = 1.0e-15
    out_mem = TS.simulate_delay_point(setup, τ; nz=3)
    fn = tempname() * "_pnps_test.h5"
    out_str = TS.simulate_delay_point(setup, τ; nz=3, filename=fn)
    # identical stepping either way — the file only changes where bytes live
    @test out_str.Iω_win ≈ out_mem.Iω_win rtol=1e-12
    @test out_str.Iω_win_reimaged ≈ out_mem.Iω_win_reimaged rtol=1e-12
    @test out_str.Iω_full ≈ out_mem.Iω_full rtol=1e-12
    @test out_str.zsave ≈ out_mem.zsave
    @test isfile(fn)   # user-supplied filename persists (run_scan cleans its own)
    rm(fn)
end

end # @testset "Trace simulation"
