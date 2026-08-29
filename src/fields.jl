# ============================================================================
# Field-construction primitives
# ============================================================================

"""
    build_he11_kspace(grid, xygrid, beam::HE11Beam, Eω) -> Array{ComplexF64,3}

Construct the 3-D field `E(ω, ky, kx)` for the HE₁₁ capillary mode imaged
onto the focal plane, multiplied by the 1-D spectral pulse `Eω`. Phase
ramps shift the beam from the FFTW corner to the centre of the spatial
grid.

The closed-form Hankel transform of the J₀ mode profile is used; the
`a²k² - u₁₁²` denominator is finite at all (kx, ky) sample points for
reasonable grid sizes (the singular ring is at radius `u₁₁/a`, well outside
typical Nyquist limits at the focal-plane scale).
"""
function build_he11_kspace(
        grid::Grid.TimeGrid, xygrid::Grid.FreeGrid,
        beam::HE11Beam, Eω::AbstractVector
    )
    a_s = a_scaled(beam)

    # phase ramps to shift the beam from FFTW corner (DC at index 1) to grid centre
    xshift = length(xygrid.x) * (xygrid.x[2] - xygrid.x[1]) / 2
    yshift = length(xygrid.y) * (xygrid.y[2] - xygrid.y[1]) / 2

    # HE₁₁ first transverse zero of J₁
    unm = Capillary.get_unm(1, 1, :HE)

    # |k⊥| on the (ky, kx) plane — Luna convention is (ω, ky, kx).
    k = sqrt.((xygrid.kx .^ 2)' .+ xygrid.ky .^ 2)        # (Nky, Nkx)
    k = reshape(k, (1, size(k)...))                       # (1, Nky, Nkx)

    # The transverse amplitude depends only on (ky, kx): evaluate the Bessel factor on
    # the (1, Nky, Nkx) plane once instead of at all Nω×Nky×Nkx points (the subsequent
    # broadcast chain is unchanged, so the result is identical)
    A = (
        -a_s^2 * unm * besselj(1, unm) .* besselj.(0, a_s .* k) ./
            (a_s^2 .* k .^ 2 .- unm^2)
    )
    Eωk = (
        A
            .* Eω
            .* exp.(-1im .* reshape(xygrid.ky, (1, length(xygrid.ky), 1)) .* yshift)
            .* exp.(-1im .* reshape(xygrid.kx, (1, 1, length(xygrid.kx))) .* xshift)
    )
    return Eωk
end

"""
    build_gaussian_kspace(grid, xygrid, beam::GaussianBeam,
                          λ0, τfwhm, energy) -> Array{ComplexF64,3}

Construct the 3-D field `E(ω, ky, kx)` for a Gaussian-Gaussian
spatio-temporal pulse: temporal Gaussian envelope (FWHM = `τfwhm`) at
carrier `λ0`, spatial Gaussian (1/e² radius = `beam.w0`) centred on the
grid, with total spectral energy normalised to `energy`. Internally uses
`Luna.Fields.GaussGaussField` and `Luna.setup` (with no nonlinearity) to
construct the field, then discards the throw-away transform/FT.
"""
function build_gaussian_kspace(
        grid::Grid.TimeGrid, xygrid::Grid.FreeGrid,
        beam::GaussianBeam, λ0, τfwhm, energy
    )
    inputs = Fields.GaussGaussField(;
        λ0 = λ0, τfwhm = τfwhm, energy = energy, w0 = beam.w0
    )
    # Use a no-op nonlinearity setup just to get a populated Eωk array. We must
    # pass a non-empty `responses` tuple to disambiguate from the modal-setup
    # method (which matches an empty tuple as Vararg{Mode}).
    densityfun = Returns(1)
    nfun_unit = Returns(1.0)
    normfun = NonlinearRHS.const_norm_free(grid, xygrid, nfun_unit)
    # χ3 = 0, so never evaluated — but it must MATCH the grid's field type, because
    # `Luna.setup` builds the transform's buffers from it.
    response = if _is_field_mode(grid)
        Nonlinear.Kerr_field(0.0)
    else
        Nonlinear.Kerr_env(0.0)
    end
    responses = (response,)
    Eωk, _, _ = Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)
    return Eωk
end

"""
    apply_tilt(Eωxy, xygrid, Δkx, Δky) -> Array{ComplexF64,3}

Multiply a real-space field `E(ω, y, x)` by the phase ramp
`exp(i Δkx · x) · exp(i Δky · y)`, which shifts its centre by
`(Δky, Δkx)` in k-space (after FFT). `Δkx = Δky = 0` is the identity.
"""
function apply_tilt(
        Eωxy::AbstractArray{<:Complex, 3}, xygrid::Grid.FreeGrid,
        Δkx::Real, Δky::Real
    )
    return (
        Eωxy
            .* reshape(exp.(1im * Δky .* xygrid.y), (1, :, 1))
            .* reshape(exp.(1im * Δkx .* xygrid.x), (1, 1, :))
    )
end

"""
    apply_delay(Eωk, grid, τ) -> Array{ComplexF64,3}

Apply a time delay `τ` (seconds) to a frequency-domain field by multiplying
each spectral component by `exp(-i ω τ)`. `τ = 0` returns a copy equal to
the input.
"""
function apply_delay(Eωk::AbstractArray{<:Complex, 3}, grid::Grid.TimeGrid, τ::Real)
    return Eωk .* reshape(exp.(-1im .* grid.ω .* τ), (length(grid.ω), 1, 1))
end
