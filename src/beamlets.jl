# ============================================================================
# Beamlet focal profile (diagnostic)
# ============================================================================

"""
    _beamlet_profile(grid, xygrid, Eωk, holex, holey, zmask; nr, rmax, nθ=64)
        -> (r, Eωr, asym)

The spatially resolved complex focal field of one beamlet, reduced to a radial profile
`Eωr[ω, r]` about its own centre, with the radius axis `r` in metres and a per-ω measure
`asym` of how well the radial reduction describes it.

Diagnostic only: nothing here feeds the propagation.

# Where the beamlet actually is

**In this representation the beamlets do not sit at BOXCARS corners in real space — they
all cross at the focus, and the corners are in k-space.** `build_he11_kspace` builds the
field in `(ω, ky, kx)` with the transverse amplitude the Hankel transform of the HE₁₁
mode, so k-space is the COLLIMATED (mask) plane — `makemask` maps
`x = kx·zmask·c/ω` — and real space, after `ifft` over dims 2 and 3, is the FOCAL
plane.
A hole at mask position `(holex, holey)` therefore selects k around
`k₀ = hole·ω/(c·zmask)`, and in the focal plane that offset is a **tilt**, not a
displacement: measured on the production geometry, the
gate beamlet peaks at the real-space grid centre to the pixel, and carries a phase slope of
2.402e5 rad/m against the predicted `k₀ = 2.417e5`.

So the centre of this profile is the grid centre. Centring it on the mask-hole position
mapped through the focus — 1 mm out, against a 26 µm spot — would sample nothing.

# Why the tilt is removed first

`k₀·r` reaches **37 rad** across the default sampling radius, so an azimuthal average
of the raw complex field would annihilate it. The field is demodulated by
`exp(-i k₀(ω)·r)` before sampling, leaving the beamlet's own envelope. `k₀ ∝ ω`, so
the coefficient is a constant `hole/(c·zmask)` in s/m; it is stored, and multiplying
the profile by `exp(+iω(cₓx + c_yy))` restores the full field. Note that the removed
tilt is physical — a linear delay across the beamlet, i.e. the pulse-front tilt of the
crossing geometry — not an artefact.

# Accuracy

Sampled by bilinear interpolation on `nr × nθ` polar points (the focal spot is ~27 grid
cells across, so the interpolation is not the limiting error) and averaged azimuthally.
`asym` is the azimuthal RMS of `|E|` over its mean, restricted to radii carrying signal:
1.2–3.8 % on the production geometry, i.e. the radial reduction is a good description but
not an exact one, and a consumer can see how good.

Integrating `|Eωr|²` with the `2πr dr` Jacobian and dividing by the cell area reproduces
the stored `Iω_beamlet` to ~1.5 % at the default `nr = 64` — the shortfall is the Airy
wings beyond `rmax` plus that asymmetry, and it converges (0.980/0.987/0.988 at nr = 128
against 0.975/0.984/0.986 at 64, at 200/260/350 nm). The test suite asserts this closure,
which is the check that the centre, the Jacobian and the normalisation are all right.
"""
function _beamlet_profile(
        grid, xygrid, Eωk::AbstractArray{<:Complex, 3},
        holex::Real, holey::Real, zmask::Real;
        nr::Int, rmax::Real, nθ::Int = 64
    )
    Nω = length(grid.ω)
    Ny, Nx = length(xygrid.y), length(xygrid.x)
    δx = xygrid.x[2] - xygrid.x[1]
    δy = xygrid.y[2] - xygrid.y[1]
    # the beamlets cross at the focus, which the phase ramps in build_he11_kspace put at
    # the middle of the real-space grid
    cy, cx = Ny ÷ 2 + 1, Nx ÷ 2 + 1
    # The requested radius can exceed the transverse grid (a small R, or a large
    # rmax_units). Sampling past the edge would silently repeat the boundary value, so
    # clamp — `beamlet_r` then records what was actually covered, and
    # `beamlet_r_max_requested` what was asked for, which is how a reader tells.
    rgrid = min(maximum(xygrid.x), maximum(xygrid.y))
    rmax_eff = min(rmax, 0.98 * rgrid)
    rmax_eff < 0.5 * rmax && @warn(
        "beamlet profile truncated by the transverse grid: asked for " *
            "$(round(rmax * 1.0e6, digits = 1)) µm, the grid supports " *
            "$(round(rmax_eff * 1.0e6, digits = 1)) µm. The radial closure against " *
            "Iω_beamlet will fall short by the energy outside.", maxlog = 1
    )
    r = collect(range(0, rmax_eff, nr))
    θ = collect(range(0, 2π, nθ + 1)[1:nθ])
    # polar sample points in fractional grid indices, computed once
    fx = [rr * cos(tt) / δx + cx for rr in r, tt in θ]
    fy = [rr * sin(tt) / δy + cy for rr in r, tt in θ]

    Eωr = zeros(ComplexF64, Nω, nr)
    asym = zeros(Float64, Nω)
    # k₀ = hole·ω/(c·zmask) per axis; the ω-independent part is the stored coefficient
    coefx = holex / (PhysData.c * zmask)
    coefy = holey / (PhysData.c * zmask)
    # ω is the FASTEST-varying axis, so a fixed-ω slice is strided and FFTW refuses to
    # apply a contiguous plan to it. Copy into the buffer and transform it in place.
    slice = Array{ComplexF64}(undef, Ny, Nx)
    IFT = FFTW.plan_ifft!(slice, (1, 2))
    ring = Vector{ComplexF64}(undef, nθ)
    relσ = Vector{Float64}(undef, nr)      # per-radius azimuthal RMS of |E| over its mean
    xv, yv = xygrid.x, xygrid.y
    for iω in 1:Nω
        copyto!(slice, view(Eωk, iω, :, :))
        IFT * slice                               # (ky, kx) -> (y, x), the focal plane
        ω = grid.ω[iω]
        k0x, k0y = coefx * ω, coefy * ω
        # Demodulate the WHOLE slice before sampling. Interpolating first and
        # demodulating after means interpolating ~6 cycles of tilt across the sampling
        # radius, and bilinear smoothing of that biases the amplitude low — measured, it
        # cost 20 % of the radial closure at the blue end.
        for j in 1:Nx
            px = cis(-k0x * xv[j])
            for i in 1:Ny
                slice[i, j] *= px * cis(-k0y * yv[i])
            end
        end
        for ir in 1:nr
            for it in 1:nθ
                gx, gy = fx[ir, it], fy[ir, it]
                j0 = clamp(floor(Int, gx), 1, Nx - 1); tx = gx - j0
                i0 = clamp(floor(Int, gy), 1, Ny - 1); ty = gy - i0
                ring[it] = (
                    (1 - ty) * (1 - tx) * slice[i0, j0] +
                        (1 - ty) * tx * slice[i0, j0 + 1] +
                        ty * (1 - tx) * slice[i0 + 1, j0] + ty * tx * slice[i0 + 1, j0 + 1]
                )
            end
            Eωr[iω, ir] = sum(ring) / nθ
            # |E| is unchanged by the demodulation, so this is the beamlet's own
            # azimuthal asymmetry — how well a radial profile describes it here.
            μ = sum(abs, ring) / nθ
            deviation = Base.Fix2(_abs_deviation_squared, μ)
            relσ[ir] = μ > 0 ? sqrt(sum(deviation, ring) / nθ) / μ : 0.0
        end
        # Report it over the radii that actually carry signal; the far wings are noise
        # and would dominate an unweighted mean.
        pk = maximum(abs, view(Eωr, iω, :))
        if pk > 0
            keep = [ir for ir in 1:nr if abs(Eωr[iω, ir]) > 0.05pk]
            asym[iω] = isempty(keep) ? 0.0 : sum(relσ[keep]) / length(keep)
        end
    end
    return r, Eωr, asym, (coefx, coefy), rmax
end

"Return the squared deviation of `abs(value)` from `mean_amplitude`."
_abs_deviation_squared(value, mean_amplitude) = (abs(value) - mean_amplitude)^2

"""
    _profile_meta(r, Eωr, asym, coef, rmax_req, holex, holey, zmask,
                  rmax_units, which) -> Dict

Package [`_beamlet_profile`](@ref)'s output for the output file. Complex data is split into
two real datasets, matching the `Eω_beamlet_re`/`_im` convention (h5py reads HDF5.jl's
native complex compound awkwardly), and enough geometry is recorded for the file to be
self-describing without the script that made it.
"""
function _profile_meta(
        r, Eωr, asym, coef, rmax_req, holex, holey, zmask,
        rmax_units, which
    )
    return Dict{String, Any}(
        "beamlet_r" => r,                    # metres, from the beamlet centre
        "Eω_beamlet_r_re" => real.(Eωr),           # (Nω, nr)
        "Eω_beamlet_r_im" => imag.(Eωr),
        "beamlet_r_asym" => asym,                 # azimuthal RMS/mean of |E| per ω
        "beamlet_r_which" => which,                # which beamlet this profile is of
        "beamlet_r_holex" => holex,                # its hole centre in the MASK plane
        "beamlet_r_holey" => holey,
        "beamlet_r_zmask" => zmask,
        # k₀(ω) = coef·ω is the geometric tilt REMOVED before the azimuthal average;
        # multiply by exp(+iω(coefx·x + coefy·y)) to restore the full focal field.
        "beamlet_r_tilt_coefx" => coef[1],
        "beamlet_r_tilt_coefy" => coef[2],
        "beamlet_r_max_units" => float(rmax_units),
        # what was asked for; `beamlet_r[end]` is what the grid actually supported
        "beamlet_r_max_requested" => float(rmax_req),
        "beamlet_profile" => 1,
    )
end

# ============================================================================
# Beamlet construction (dispatched on beam type)
# ============================================================================

"""
    build_beamlets(beam, grid, xygrid, geom, Eω, energy, energyfun_ω;
                   apod=:supergauss, apod_param=nothing, ϕ=nothing,
                   profile=true, profile_nr=64, profile_rmax_units=6)
        -> (Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_metadata::Dict)

Construct the input beamlets at the substrate, in k-space. The geometry `geom` is
a `NamedTuple(mask_diam, mask_spacing, f_foc, λ0, τfwhm, geometry)` shared by both
beam models. `geom.geometry` is `:tg` for the three-beam boxcar layout
`(g1, g2, t-base)`, or `:sd` for the two-beam self-diffraction layout, which is
built by the `HE11Beam` method only and returns `nothing` in place of `Eωk_g2`
(a zero array of that size is half a gigabyte of pure waste), putting the probe
in `Eωk_g1` and the delayed gate in `Eωk_t_base`.

`ϕ` is accepted for a uniform interface across beam models and ignored by both:
the spectral phase is already carried by the 1-D reference `Eω` that
[`build_setup`](@ref) passes in. `profile`, `profile_nr` and
`profile_rmax_units` control the diagnostic radial focal profile added to
`beam_metadata`; see [`_beamlet_profile`](@ref).

For `HE11Beam`: builds the full HE₁₁ k-space field, rescales to the
requested energy, then applies three apodised hole masks (g1, g2, t).
Each beamlet sits at one of the boxcar corners. `Iω_beamlet` is the
spatially-integrated spectrum of `g1` (used as a chromatic-vignetting
diagnostic in the output file).

For `GaussianBeam`: builds a Gaussian-Gaussian field with energy
`energy/3` per beam, ifft's to real space, then applies real-space tilts
to position the three beams at the boxcar corners. `Iω_beamlet` here is
just the (unvignetted) input spectrum scaled to `energy/3`; it is returned
for uniformity with the HE₁₁ model so downstream code never special-cases
the beam type.
"""
function build_beamlets(
        beam::HE11Beam, grid::Grid.TimeGrid,
        xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
        energy::Real, energyfun_ω;
        apod::Symbol = :supergauss, apod_param = nothing,
        ϕ = nothing, profile::Bool = true, profile_nr::Int = 64,
        profile_rmax_units::Real = 6
    )
    # `ϕ` is accepted for a uniform interface and deliberately IGNORED: this
    # builder derives its beamlets from the 1-D reference `Eω`, which already
    # carries the Taylor spectral phase applied in `build_setup`. Applying ϕ
    # again here would double the chirp.
    # Full beam (no mask) in k-space, rescaled to the requested energy.
    Eωk0 = build_he11_kspace(grid, xygrid, beam, Eω)
    Eωk0 .*= sqrt(energy) / sqrt(energyfun_ω(Eωk0))

    # --- Self-diffraction: TWO collinear beams --------------------------
    #
    # SD is the same propagation and the same chi(3) response; only the input
    # differs. The signal is 2k_E - k_G, so with the probe E and gate G on one
    # axis at -+s/2 (s = spacing + diam, centre to centre) the signal appears at
    # -3s/2 — one further slot out, on the probe's side.
    #
    # The layout is SYMMETRIC about the axis (E at -s/2, G at +s/2) rather than
    # putting E on axis and centring the signal. The centred variant needs a
    # 27% narrower grid, but cuts the two beams from very different parts of the
    # HE11 profile, giving them unequal energy; SD's signal is E^2 G*, so that
    # asymmetry does not cancel.
    #
    # Which beam is delayed follows croak's SD kernel E^2 G*: the GATE appears
    # once, carries the conjugation, and takes the delay — the same convention
    # as TG, where the delay rides the singly-appearing conjugated arm.
    if get(geom, :geometry, :tg) === :sd
        s_cc = geom.mask_spacing + geom.mask_diam
        Eωk_E = Eωk0 .* makemask(
            -s_cc / 2, 0.0, geom.mask_diam, grid, xygrid;
            zmask = beam.f_foc, apod = apod, apod_param = apod_param,
            λ0_for_default = geom.λ0
        )
        Eωk_G = Eωk0 .* makemask(
            s_cc / 2, 0.0, geom.mask_diam, grid, xygrid;
            zmask = beam.f_foc, apod = apod, apod_param = apod_param,
            λ0_for_default = geom.λ0
        )
        NyNx_sd = length(xygrid.y) * length(xygrid.x)
        Iω_bl = dropdims(sum(abs2, Eωk_E; dims = (2, 3)); dims = (2, 3)) ./ NyNx_sd
        meta_sd = Dict{String, Any}(
            "Iω_beamlet" => Iω_bl, "a" => beam.a,
            "geometry" => "sd",
            "sd_separation_cc" => s_cc,
            "sd_signal_x" => -1.5 * s_cc
        )
        # Profile the beamlet `Iω_beamlet` describes, so the two agree and the radial
        # closure check means something. In SD that is the probe E, at (-s_cc/2, 0).
        if profile
            rmax = profile_rmax_units * geom.λ0 * beam.f_foc / geom.mask_diam
            pr = _beamlet_profile(
                grid, xygrid, Eωk_E, -s_cc / 2, 0.0, beam.f_foc;
                nr = profile_nr, rmax = rmax
            )
            merge!(
                meta_sd, _profile_meta(
                    pr..., -s_cc / 2, 0.0, beam.f_foc,
                    profile_rmax_units, "sd_probe"
                )
            )
        end
        # E is the undelayed pair-slot (it appears squared); G takes the delay.
        # `nothing` for the second gate rather than a zero array: at SD grid
        # size that array is ~0.5 GB of pure waste, and this campaign has been
        # OOM-killed by exactly this class of transient before.
        return Eωk_E, nothing, Eωk_G, Iω_bl, meta_sd
    end

    # Hole centres at (±d, ±d), where d = mask_spacing/2 + mask_diam/2.
    d = geom.mask_spacing / 2 + geom.mask_diam / 2

    # Boxcar layout (looking along +z):
    #
    #   test (-x, +y)    | gate1 (+x, +y)
    #  ---------------------------------------
    #   signal (-x, -y)  | gate2 (+x, -y)
    # Build and apply the masks one at a time: holding all three (plus their products)
    # simultaneously added ~2 field-sized arrays to the setup's peak memory.
    Eωk_g1 = Eωk0 .* makemask(
        d, d, geom.mask_diam, grid, xygrid;
        zmask = beam.f_foc, apod = apod, apod_param = apod_param,
        λ0_for_default = geom.λ0
    )
    Eωk_g2 = Eωk0 .* makemask(
        d, -d, geom.mask_diam, grid, xygrid;
        zmask = beam.f_foc, apod = apod, apod_param = apod_param,
        λ0_for_default = geom.λ0
    )
    Eωk_t_base = Eωk0 .* makemask(
        -d, d, geom.mask_diam, grid, xygrid;
        zmask = beam.f_foc, apod = apod, apod_param = apod_param,
        λ0_for_default = geom.λ0
    )

    # Spatially-integrated spectrum of the gate-1 beamlet — a useful diagnostic
    # showing the chromatic vignetting imprinted by the physical mask.
    # By Parseval (unitary up to 1/(Ny·Nx) for the ifft), the real-space sum equals the
    # k-space sum — no need to materialise the ifft'd field.
    NyNx = length(xygrid.y) * length(xygrid.x)
    Iω_beamlet = dropdims(sum(abs2, Eωk_g1; dims = (2, 3)); dims = (2, 3)) ./ NyNx

    beam_meta = Dict{String, Any}(
        "Iω_beamlet" => Iω_beamlet,
        "a" => beam.a,
        "a_scaled" => a_scaled(beam),
        "f_coll" => beam.f_coll,
        "f_foc" => beam.f_foc,
    )

    # Spatially resolved focal field of gate 1 — the beamlet `Iω_beamlet` describes, so
    # the two are consistent and the radial closure check is meaningful. Diagnostic only:
    # `Eωk_g1` is read, never modified. Computed ONCE here, not per delay point.
    if profile
        rmax = profile_rmax_units * geom.λ0 * beam.f_foc / geom.mask_diam
        pr = _beamlet_profile(
            grid, xygrid, Eωk_g1, d, d, beam.f_foc;
            nr = profile_nr, rmax = rmax
        )
        merge!(
            beam_meta, _profile_meta(
                pr..., d, d, beam.f_foc,
                profile_rmax_units, "tg_gate1"
            )
        )
    end

    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end

function build_beamlets(
        beam::GaussianBeam, grid::Grid.TimeGrid,
        xygrid::Grid.FreeGrid, geom, Eω::AbstractVector,
        energy::Real, energyfun_ω;
        apod::Symbol = :supergauss, apod_param = nothing,
        ϕ = nothing, profile::Bool = true, profile_nr::Int = 64,
        profile_rmax_units::Real = 6
    )
    # In the Gaussian model each beamlet carries an equal third of the energy.
    energy_per_beam = energy / 3
    Eωk_base = build_gaussian_kspace(
        grid, xygrid, beam,
        geom.λ0, geom.τfwhm, energy_per_beam
    )
    # Apply the input spectral phase (GDD/TOD) HERE rather than inheriting it
    # from the 1-D reference `Eω`, which this builder does not use.
    # `Fields.GaussGaussField` takes a ϕ, but for a SpatioTemporalField that ϕ
    # is a SCALAR carrier-envelope phase (`exp(i(ϕ + Δω t))`), not the Taylor
    # coefficient VECTOR that the 1-D `GaussField` accepts — so it cannot carry
    # GDD at all, and a chirp requested through `build_setup` would otherwise be
    # silently dropped for this beam type.
    #
    # Applying it as a pure spectral phase is exact here: the Gaussian
    # transverse profile is ω-independent by construction (that is the whole
    # point of this beam), so the spectral phase commutes with the spatial
    # structure. It would NOT be exact for a chromatic beam, which is why the
    # HE11 builder instead inherits the already-chirped `Eω`.
    if ϕ !== nothing && any(!iszero, ϕ)
        Fields.prop_taylor!(Eωk_base, grid, ϕ, geom.λ0)
    end
    Eωxy = ifft(Eωk_base, (2, 3))

    # Crossing geometry derived from the mask parameters.
    d_hole = geom.mask_spacing / 2 + geom.mask_diam / 2
    crossingθ = d_hole / beam.f_foc
    Δk = 2π / geom.λ0 * sin(crossingθ)

    # Boxcar tilts:
    #   gate 1: (+Δk_x, +Δk_y)
    #   gate 2: (+Δk_x, -Δk_y)
    #   test  : (-Δk_x, +Δk_y)
    Eωk_g1 = fft(apply_tilt(Eωxy, xygrid, +Δk, +Δk), (2, 3))
    Eωk_g2 = fft(apply_tilt(Eωxy, xygrid, +Δk, -Δk), (2, 3))
    Eωk_t_base = fft(apply_tilt(Eωxy, xygrid, -Δk, +Δk), (2, 3))

    # Spatially-integrated beamlet spectrum. The tilt is a pure phase ramp, so
    # every beamlet has the same integrated spectrum as the untilted base beam;
    # for the unmasked Gaussian model this carries no chromatic vignetting (it is
    # just the input spectrum scaled to energy/3) but is saved for uniformity
    # with the HE₁₁ model so downstream code never needs to special-case beams.
    Iω_beamlet = dropdims(sum(abs2.(Eωk_base); dims = (2, 3)); dims = (2, 3))

    beam_meta = Dict{String, Any}(
        "Iω_beamlet" => Iω_beamlet,
        "w0" => beam.w0,
        "f_foc" => beam.f_foc,
        "Δk" => Δk,
        "crossingθ" => crossingθ,
        "d_hole" => d_hole,
    )
    # As for HE₁₁, but this model has no mask: the beamlet is a tilted Gaussian, so the
    # natural radial scale is w0 rather than λf/D, and the tilt to remove is the applied
    # Δk. `Δk` here is ω-INDEPENDENT (it is built at λ0 in `apply_tilt`), unlike the
    # mask geometry's `hole·ω/(c·zmask)`, so the equivalent coefficient is Δk/ω0.
    if profile
        ω0 = 2π * PhysData.c / geom.λ0
        rmax = profile_rmax_units * beam.w0
        pr = _beamlet_profile(
            grid, xygrid, Eωk_g1, Δk * PhysData.c * beam.f_foc / ω0,
            Δk * PhysData.c * beam.f_foc / ω0, beam.f_foc;
            nr = profile_nr, rmax = rmax
        )
        merge!(
            beam_meta, _profile_meta(
                pr..., d_hole, d_hole, beam.f_foc,
                profile_rmax_units, "gaussian_gate1"
            )
        )
    end
    return Eωk_g1, Eωk_g2, Eωk_t_base, Iω_beamlet, beam_meta
end
