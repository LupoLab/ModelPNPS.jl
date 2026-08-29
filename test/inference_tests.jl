using Test: @inferred, @test, @testset

import Luna: Grid
import ModelPNPS

const MP = ModelPNPS

@testset "test_public_api_type_stability" begin
    he11 = @inferred MP.HE11Beam(125.0e-6, 5.0, 0.1)
    gaussian = @inferred MP.GaussianBeam(2.5e-6, 0.1)
    physical = @inferred MP.PhysicalMaskWindow(
        -0.75e-3, -0.75e-3, 0.25e-3, 0.1, :supergauss, 16
    )
    planck = @inferred MP.PlanckWindow(-2.0e5, -2.0e5, 4.0e4, 1.25)
    planck_ω = @inferred MP.PlanckOmegaWindow(
        -0.75e-3, -0.75e-3, 0.25e-3, 0.1, 1.25
    )

    @test he11 isa MP.AbstractInputBeam
    @test gaussian isa MP.AbstractInputBeam
    @test physical isa MP.AbstractSignalWindow
    @test planck isa MP.AbstractSignalWindow
    @test planck_ω isa MP.AbstractSignalWindow

    R, N = @inferred MP.optimal_spatial_grid(
        0.1, 0.25e-3, 0.5e-3, 200.0e-9, 400.0e-9;
        n_airy = 2, pts_per_lobe = 4
    )
    @test R > 0.0
    @test iseven(N)

    grid = Grid.EnvGrid(1.0e-6, 260.0e-9, (200.0e-9, 400.0e-9), 20.0e-15)
    xygrid = Grid.FreeGrid(R, 8)
    field = ones(ComplexF64, length(grid.ω), 8, 8)

    delayed = @inferred MP.apply_delay(field, grid, 0.5e-15)
    tilted = @inferred MP.apply_tilt(field, xygrid, 1.0e4, -1.0e4)
    window = @inferred MP.build_window(planck, grid, xygrid)
    integrated, reimaged = @inferred MP.extract_signal_spectra(field, window, xygrid)

    @test size(delayed) == size(field)
    @test size(tilted) == size(field)
    @test size(window) == (8, 8)
    @test length(integrated) == length(grid.ω)
    @test length(reimaged) == length(grid.ω)

    ω = collect(range(1.0e15, 2.0e15; length = 16))
    pulse = @inferred MP.InputPulseData(ω, ones(ComplexF64, length(ω)))
    @test pulse isa MP.InputPulseData

    setup_args = (;
        thickness = 1.0e-6,
        λ0 = 260.0e-9,
        λlims = (200.0e-9, 400.0e-9),
        trange = 20.0e-15,
        N = 8,
        window = planck,
    )
    budget = @inferred MP.memory_budget(setup_args)
    @test budget.device > 0.0
end
