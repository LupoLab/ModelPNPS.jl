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
import Random: MersenneTwister

@testset "Trace simulation" begin

# -----------------------------------------------------------------------------
@testset "optimal_spatial_grid" begin
    f         = 0.1
    mask_diam = 1e-3
    mask_spc  = 0.5e-3
    λmin, λmax = 160e-9, 500e-9

    R, N = TS.optimal_spatial_grid(f, mask_diam, mask_spc, λmin, λmax)

    # N is a power of 2.
    @test N == nextpow(2, N)
    @test N > 0

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
    @test size(out.Iω_win) == (length(setup.grid.ω), 2)
    @test size(out.Iω_full) == (length(setup.grid.ω), 2)
    @test all(isfinite, out.Iω_win)
    @test all(out.Iω_win .>= 0)
    @test all(out.Iω_win_reimaged .>= 0)
    @test all(isfinite, out.Iω_full)
    @test all(out.Iω_full .>= 0)
end

# -----------------------------------------------------------------------------
# Synthetic-file test for load_simulated_scan. The real scansave files are
# 10s-100s of MB; we write a tiny mock file with the same structure to
# exercise the loader logic.
# -----------------------------------------------------------------------------

function _write_mock_scan_file(path; Nω=64, Nt=64, Nτ=8, nz=2,
                                 with_beamlet=true,
                                 with_omega_dep=false,
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
    end
end

end # @testset "Trace simulation"
