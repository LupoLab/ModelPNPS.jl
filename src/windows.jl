# ============================================================================
# Mask / window construction
# ============================================================================

"""
    makemask(holex, holey, holediam, grid, xygrid;
             zmask, apod=:supergauss, apod_param=nothing,
             λ0_for_default=nothing) -> Array{Float64,3}

Build a 3-D `(Nω, Nky, Nkx)` apodised-hole mask. For each `(ω, ky, kx)`
sample, the k-vector is mapped to the mask-plane position
`x = kx · zmask · c / ω` (and likewise for `y`), and a hole of diameter
`holediam` centred at `(holex, holey)` is evaluated.

`λ0_for_default` is only used when `apod=:tanh` and `apod_param===nothing`,
in which case the smoothing width is set to `3·Δx_mask` evaluated at the
carrier wavelength.
"""
function makemask(
        holex::Real, holey::Real, holediam::Real,
        grid::Grid.TimeGrid, xygrid::Grid.FreeGrid;
        zmask::Real,
        apod::Symbol = :supergauss,
        apod_param = nothing,
        λ0_for_default = nothing
    )
    # Resolve default apod_param.
    if apod_param === nothing
        if apod === :supergauss
            apod_param = 16
        elseif apod === :tanh
            isnothing(λ0_for_default) && throw(
                ArgumentError(
                    "λ0_for_default is required when apod=:tanh and " *
                        "apod_param is omitted"
                )
            )
            Δk = xygrid.kx[2] - xygrid.kx[1]
            ω0 = grid.ω[argmin(abs.(grid.ω .- 2π * PhysData.c / λ0_for_default))]
            Δx_mask = Δk * zmask * PhysData.c / ω0
            apod_param = 3 * Δx_mask
        end
    end

    mask = zeros(Float64, length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    for ii in CartesianIndices(mask)
        ω = grid.ω[ii[1]]
        ky = xygrid.ky[ii[2]]    # Luna convention: dim 2 = ky
        kx = xygrid.kx[ii[3]]    # Luna convention: dim 3 = kx
        # ω == 0 is the DC bin (often present in EnvGrid). The mapping diverges,
        # so leave the mask zero there — it won't carry any signal anyway.
        ω == 0 && continue
        x = kx * zmask * PhysData.c / ω
        y = ky * zmask * PhysData.c / ω
        rhole = hypot(x - holex, y - holey)
        if apod === :hard
            mask[ii] = rhole <= holediam / 2 ? 1.0 : 0.0
        elseif apod === :supergauss
            mask[ii] = exp(-(2 * rhole / holediam)^apod_param)
        elseif apod === :tanh
            mask[ii] = 0.5 * (1 - tanh((rhole - holediam / 2) / apod_param))
        else
            throw(
                ArgumentError(
                    "apod must be :hard, :supergauss, or :tanh; got :$apod"
                )
            )
        end
    end
    return mask
end

"""
    build_window(w::AbstractSignalWindow, grid, xygrid; λ0=nothing)
        -> Array{Float64, N}

Materialise the precomputed signal-extraction window. Returns a
`(Nky, Nkx)` 2-D array for [`PlanckWindow`](@ref) and a
`(Nω, Nky, Nkx)` 3-D array for [`PhysicalMaskWindow`](@ref) and
[`PlanckOmegaWindow`](@ref). `λ0` is forwarded to [`makemask`](@ref) for
default `:tanh` apodisation widths only.
"""
function build_window(
        w::PhysicalMaskWindow, grid::Grid.TimeGrid,
        xygrid::Grid.FreeGrid; λ0 = nothing
    )
    return makemask(
        w.holex, w.holey, w.holediam, grid, xygrid;
        zmask = w.zmask, apod = w.apod, apod_param = w.apod_param,
        λ0_for_default = λ0
    )
end

function build_window(
        w::PlanckWindow, grid::Grid.TimeGrid,
        xygrid::Grid.FreeGrid; λ0 = nothing
    )
    # Radial distance from window centre.
    κ = @. sqrt((xygrid.ky - w.kyc)^2 + (xygrid.kx' - w.kxc)^2)
    # Planck taper: flat in [0, kwidth], rolling off to zero by pad·kwidth.
    return Maths.planck_taper.(κ, -w.kwidth, -w.kwidth, w.kwidth, w.pad * w.kwidth)
end

function build_window(
        w::PlanckOmegaWindow, grid::Grid.TimeGrid,
        xygrid::Grid.FreeGrid; λ0 = nothing
    )
    win = zeros(Float64, length(grid.ω), length(xygrid.ky), length(xygrid.kx))
    for (iω, ω) in enumerate(grid.ω)
        ω == 0 && continue
        # Hole centre and half-width in k-space at this frequency.
        kxc = ω / PhysData.c * w.xc / w.f_foc
        kyc = ω / PhysData.c * w.yc / w.f_foc
        khole = ω / PhysData.c * (w.holediam / 2) / w.f_foc
        for (ikx, kx) in enumerate(xygrid.kx)
            for (iky, ky) in enumerate(xygrid.ky)
                κi = sqrt((ky - kyc)^2 + (kx - kxc)^2)
                win[iω, iky, ikx] = Maths.planck_taper(
                    κi, -khole, -khole, khole, w.pad * khole
                )
            end
        end
    end
    return win
end
