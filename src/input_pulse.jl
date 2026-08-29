# ============================================================================
# Data-driven input pulse
# ============================================================================

"Return the HDF5 dataset stored at `key`, or throw for a malformed file."
function _hdf5_dataset(parent, key)
    object = parent[key]
    object isa HDF5.Dataset || throw(
        ArgumentError("expected '$key' to be an HDF5 dataset; got $(typeof(object))")
    )
    return object
end

"Return the HDF5 group stored at `key`, or throw for a malformed file."
function _hdf5_group(parent, key)
    object = parent[key]
    object isa HDF5.Group || throw(
        ArgumentError("expected '$key' to be an HDF5 group; got $(typeof(object))")
    )
    return object
end

"""
    InputPulseData(ω, Eω)

A measured or simulated input pulse: a complex spectrum `Eω` on an ABSOLUTE,
ascending, (approximately) uniform angular-frequency axis `ω` [rad/s]. Inject
it through [`build_setup`](@ref)'s `input_pulse` keyword — HE₁₁ beam model
only, because there the 1-D reference spectrum IS the pulse and the chromatic
mask vignetting is applied to it downstream, so an arbitrary field composes
exactly (see `build_beamlets`). Amplitude units are irrelevant: the beamlet
builder rescales the assembled beam to the requested `energy`.

Companion utilities, typically chained in this order:
[`load_input_pulse`](@ref) → [`spectral_window!`](@ref) →
[`center_pulse!`](@ref) → `build_setup(...; input_pulse=p)`.
"""
struct InputPulseData
    ω::Vector{Float64}
    Eω::Vector{ComplexF64}
    function InputPulseData(ω, Eω)
        length(ω) == length(Eω) ||
            throw(ArgumentError("ω and Eω have different lengths"))
        length(ω) >= 8 || throw(ArgumentError("input pulse needs ≥ 8 samples"))
        issorted(ω) || throw(ArgumentError("ω must be ascending"))
        return new(collect(Float64, ω), collect(ComplexF64, Eω))
    end
end

"""
    load_input_pulse(path; ω_key="ω", Eω_key="Eω") -> InputPulseData

Read an [`InputPulseData`](@ref) from an HDF5 file: an absolute angular
frequency axis under `ω_key` and a complex spectrum under `Eω_key` (a native
complex dataset, e.g. as written by HDF5.jl or h5py).
"""
function load_input_pulse(path::AbstractString; ω_key = "ω", Eω_key = "Eω")
    return HDF5.h5open(path, "r") do f
        InputPulseData(
            read(_hdf5_dataset(f, ω_key)),
            read(_hdf5_dataset(f, Eω_key))
        )
    end
end

"""
    spectral_window!(p::InputPulseData, λmin, λmax;
                     wfrac_blue=0.05, wfrac_red=0.03) -> p

Smooth tanh band-pass in place: unity well inside (`λmin`, `λmax`) [m], rolling
off with tanh edges of width `wfrac_red · ω(λmax)` on the red side and
`wfrac_blue · ω(λmin)` on the blue side. Use before injection to remove
content the simulation band does not (or should not) carry — e.g. a residual
driver remnant that survived an imperfect spectral filter, which would
otherwise dominate the χ³ interaction. The windowed field is the ground truth
the retrieval is compared against, so keep the applied window with the run's
provenance.
"""
function spectral_window!(
        p::InputPulseData, λmin::Real, λmax::Real;
        wfrac_blue::Real = 0.05, wfrac_red::Real = 0.03
    )
    λmin < λmax || throw(ArgumentError("need λmin < λmax"))
    ωred = 2π * PhysData.c / λmax
    ωblue = 2π * PhysData.c / λmin
    wred = wfrac_red * ωred
    wblue = wfrac_blue * ωblue
    @. p.Eω *= 0.25 * (1 + tanh((p.ω - ωred) / wred)) *
        (1 + tanh((ωblue - p.ω) / wblue))
    return p
end

"""
    center_pulse!(p::InputPulseData; oversample=8) -> (p, tshift)

Remove the linear spectral-phase component so the temporal intensity envelope
peaks at the data FFT's natural origin (array index 1), returning the applied
shift `tshift` [s] (positive = the pulse arrived late and was advanced). A pure
linear phase is physically irrelevant; numerically, centring minimises the
`trange` the simulation needs to hold the pulse plus the delay scan, and — more
importantly — it is what makes the spectrum interpolatable: a pulse far from
its grid's natural time origin has a spectral phase rotating by up to π per
sample, which no Re/Im interpolation can resample
([`interp_input_pulse`](@ref) warns if it sees this). `interp_input_pulse` then
re-anchors the interpolated field at `t = 0`, the middle sample of Luna's
centred target time grid. Requires an (approximately) uniform ω grid. The
returned shift is reported modulo the data grid's time period (the on-grid
phase is identical for any branch).
"""
function center_pulse!(p::InputPulseData; oversample::Int = 8)
    dωs = diff(p.ω)
    dω = sum(dωs) / length(dωs)
    maximum(abs.(dωs .- dω)) < 1.0e-6 * dω ||
        throw(ArgumentError("center_pulse! requires a uniform ω grid"))
    n = nextpow(2, oversample * length(p.ω))
    buf = zeros(ComplexF64, n)
    buf[eachindex(p.ω)] .= p.Eω
    et = FFTW.ifft(buf)
    dt = 2π / (n * dω)
    ipk = argmax(abs2.(et))
    tpk = (ipk - 1) * dt
    tpk > n * dt / 2 && (tpk -= n * dt)
    @. p.Eω *= exp(1im * p.ω * tpk)
    return p, tpk
end

"""
    interp_input_pulse(grid, p::InputPulseData) -> Vector{ComplexF64}

The pulse's complex spectrum on `grid.ω` (the grid's ABSOLUTE frequency axis),
zero outside the data's range. Real and imaginary parts are interpolated
separately with cubic B-splines, which is accurate when the data grid is finer
than the simulation grid — a warning is emitted if it is not (then spectral
detail is being invented between samples; supply denser data instead). The
input is expected to have been moved to its data FFT's natural origin with
[`center_pulse!`](@ref). After interpolation, the field is shifted to the
middle sample of Luna's centred target time grid, matching Luna's native
`Fields.DataField` convention.
"""
function interp_input_pulse(grid::Grid.TimeGrid, p::InputPulseData)
    dω_data = (last(p.ω) - first(p.ω)) / (length(p.ω) - 1)
    dω_grid = length(grid.ω) > 1 ? abs(grid.ω[2] - grid.ω[1]) : Inf
    dω_data > dω_grid &&
        @warn "input pulse is coarser than the simulation grid" dω_data dω_grid
    # A pulse far from its grid's natural t = 0 carries a near-Nyquist phase
    # rotation that Re/Im interpolation cannot resample — centre it first.
    let a = abs.(p.Eω), thr = 0.01 * maximum(a)
        rot = [
            abs(angle(p.Eω[i + 1] * conj(p.Eω[i])))
                for i in 1:(length(p.Eω) - 1) if a[i] > thr && a[i + 1] > thr
        ]
        if !isempty(rot) && median(rot) > 1.0
            @warn "input pulse spectral phase rotates " *
                "$(round(median(rot), digits = 2)) rad per sample " *
                "— the pulse sits far from t = 0 and interpolation onto " *
                "the simulation grid will alias. Run center_pulse! first."
        end
    end
    spl_r = Maths.BSpline(p.ω, real(p.Eω))
    spl_i = Maths.BSpline(p.ω, imag(p.Eω))
    lo, hi = first(p.ω), last(p.ω)
    Eω = zeros(ComplexF64, length(grid.ω))
    for (i, ω) in enumerate(grid.ω)
        lo <= ω <= hi || continue
        Eω[i] = complex(spl_r(ω), spl_i(ω))
    end
    # `center_pulse!` deliberately removes the source grid's time-origin phase,
    # leaving the pulse at FFT array index 1 so its Re/Im spectrum is smooth
    # enough to interpolate. Luna time grids label that index as -T/2 and put
    # physical t=0 at the middle sample. Re-apply the target grid's half-window
    # delay after interpolation, exactly as Luna.Fields.DataField does, so both
    # the propagated beamlets and the stored temporal diagnostics are centred.
    τgrid = length(grid.t) * (grid.t[2] - grid.t[1]) / 2
    @. Eω *= exp(-1im * grid.ω * τgrid)
    return Eω
end
