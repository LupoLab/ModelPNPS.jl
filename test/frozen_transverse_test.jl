# =============================================================================
# Tests for the frozen-transverse ablation (`build_setup(frozen_transverse=true)`).
#
# The ablation replaces k_z(ω, k⊥) by k_z(ω, 0) in the linear operator: every
# k⊥ component gets the same ω-dependent phase, so the transverse field pattern
# is frozen exactly at its entrance form while temporal dispersion runs
# unchanged. Three guarantees are pinned here:
#
#   (i)   flag off (the default) is bit-identical to the pre-existing operator,
#         and the flag touches ONLY the linop — beamlets, windows and metadata
#         are shared with the unfrozen setup;
#   (ii)  freeze self-test: linear-only propagation of one beamlet over the
#         full substrate leaves the transverse fluence profile equal to the
#         entrance profile to machine precision while the temporal envelope
#         disperses (SiO2 GDD over 40 µm reshapes a 2 fs pulse several-fold);
#   (iii) the tilt survives: the three-beam crossing-interference pattern —
#         which carries the pulse-front tilts in its ω–k⊥ coupling — is also
#         frozen exactly, and the same propagation under the UNFROZEN operator
#         demonstrably moves it (the negative control that shows the test can
#         tell the difference).
#
# Run standalone with:
#     julia --project=. test/frozen_transverse_test.jl
# Or as part of the suite via test/runtests.jl.
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end
using Test
using ModelPNPS
import ModelPNPS as TS
import FFTW: ifft

@testset "Frozen-transverse ablation" begin

    # Production optics on a tiny grid (the device-test smoke configuration,
    # but with the full 40 µm substrate so dispersion has something to do).
    beam = TS.HE11Beam(125.0e-6, 5.0, 0.1)
    window = TS.PhysicalMaskWindow(
        holex = -0.75e-3, holey = -0.75e-3,
        holediam = 0.25e-3, zmask = 0.1, apod = :tanh
    )
    base_kwargs = (;
        λ0 = 260.0e-9, τfwhm = 2.0e-15, energy = 0.1e-6,
        thickness = 40.0e-6, material = :SiO2,
        mask_diam = 1.0e-3, mask_spacing = 0.5e-3,
        beam, window,
        trange = 30.0e-15, λlims = (200.0e-9, 400.0e-9),
        R = 40.0e-6, N = 16,
    )

    s0 = TS.build_setup(; base_kwargs...)
    sfrz = TS.build_setup(; base_kwargs..., frozen_transverse = true)

    # k-space field -> transverse fluence (x, y). Any fixed unitary-up-to-scale
    # transform works here since only z-vs-entrance ratios are compared;
    # Parseval makes Σ_ω |E(ω, x, y)|² the time-integrated intensity.
    fluence(Ek) = dropdims(sum(abs2, ifft(Ek, (2, 3)); dims = 1); dims = 1)
    # Linear-only propagation over the full substrate: the constant linear
    # operator applied directly, no stepper and no nonlinearity.
    linprop(setup, Ek) = Ek .* exp.(collect(setup.linop) .* setup.thickness)

    # ------------------------------------------------------------------ (i) --
    @testset "flag off is bit-identical; flag touches the linop only" begin
        soff = TS.build_setup(; base_kwargs..., frozen_transverse = false)
        @test isequal(collect(soff.linop), collect(s0.linop))
        @test s0.combined_grid["frozen_transverse"] == 0
        @test sfrz.combined_grid["frozen_transverse"] == 1

        # the factored/materialised bit-identity survives the flag
        sfrzmat = TS.build_setup(;
            base_kwargs..., frozen_transverse = true, factored_linop = false
        )
        @test all(iszero, sfrz.linop.kperp2)
        @test isequal(collect(sfrz.linop), sfrzmat.linop)

        # everything except the linop is untouched by the flag
        @test isequal(sfrz.Eωk_g12, s0.Eωk_g12)
        @test isequal(sfrz.Eωk_t_base, s0.Eωk_t_base)
        @test isequal(sfrz.window_array, s0.window_array)
        @test isequal(sfrz.Eω, s0.Eω)

        # ...and the frozen operator really is transverse-uniform
        lf = collect(sfrz.linop)
        @test all(lf .== lf[:, 1:1, 1:1])
    end

    # ----------------------------------------------------------------- (ii) --
    @testset "freeze self-test: one beamlet, linear-only, z = 40 µm" begin
        Ek0 = collect(sfrz.Eωk_t_base)
        Ekz = linprop(sfrz, Ek0)
        flu0 = fluence(Ek0)
        fluz = fluence(Ekz)
        # transverse profile frozen to machine precision
        @test maximum(abs, fluz .- flu0) <= 1.0e-12 * maximum(flu0)
        # ...while the temporal envelope disperses: ~8 fs² of SiO2 GDD broadens
        # the 2 fs pulse several-fold, so the peak envelope intensity at the
        # fluence maximum must drop far below the entrance value
        ii = argmax(flu0)
        et0 = ifft(ifft(Ek0, (2, 3))[:, ii])
        etz = ifft(ifft(Ekz, (2, 3))[:, ii])
        @test maximum(abs2, etz) < 0.7 * maximum(abs2, et0)
    end

    # ---------------------------------------------------------------- (iii) --
    @testset "the tilt survives: three-beam interference at z = 40 µm" begin
        Ek0 = collect(sfrz.Eωk_g12) .+ collect(sfrz.Eωk_t_base)  # τ = 0 input
        flu0 = fluence(Ek0)
        # the pattern is genuinely interferometric: coherent cross terms are a
        # visible fraction of the incoherent sum (otherwise this test is vacuous)
        fluinc = fluence(collect(sfrz.Eωk_g12)) .+ fluence(collect(sfrz.Eωk_t_base))
        @test maximum(abs, flu0 .- fluinc) > 0.05 * maximum(flu0)

        # frozen: the full crossing pattern at the exit equals the entrance
        fluz = fluence(linprop(sfrz, Ek0))
        @test maximum(abs, fluz .- flu0) <= 1.0e-12 * maximum(flu0)

        # negative control: the UNFROZEN operator moves the pattern over the
        # same distance (crossing-phase walk ~5e-2 rad at this geometry), so
        # the machine-precision equality above is not test insensitivity
        fluz_full = fluence(linprop(s0, Ek0))
        @test maximum(abs, fluz_full .- flu0) > 1.0e-6 * maximum(flu0)
    end
end
