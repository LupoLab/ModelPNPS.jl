# ============================================================================
# Loading and post-processing scan output files
# ============================================================================
#
# `run_scan` writes one HDF5 file per delay scan via `Output.scansave`. The
# file structure is:
#
#   /scanvariables/τ              the FROG delay axis (Nτ,) [s]
#   /grid/ω                       absolute angular frequency (FFT-ordered)
#   /grid/t, /grid/ω0, /grid/Iω, /grid/It, /grid/τfwhm, ...
#   /grid/Iω_beamlet              input-vignetted beamlet spectrum
#   /grid/It_beamlet, /grid/Ito_beamlet   beamlet temporal intensity (+ oversampled)
#   /grid/window, /grid/window_ωdep   precomputed signal mask(s)
#   /grid/zsave                   (nz,) realized propagation z positions [m]
#   /Iω_win                       (Nω, nz, Nτ) integrated FROG trace
#   /Iω_win_reimaged              (Nω, nz, Nτ) on-axis re-imaged trace
#   /Iω_full                      (Nω, nz, Nτ) full signal-quadrant collection
#   /Iω_win_ωdep, /Iω_win_ωdep_reimaged    Gaussian two-window extras
#
# `load_simulated_scan` extracts the chosen window/z-slice, fftshifts ω-
# dependent arrays into natural (centred) order, and returns a NamedTuple
# ready for inspection, plotting, or downstream processing.

"Return whether an HDF5 key names a scan result dataset."
_is_scan_dataset(key) = !(key in ("grid", "scanvariables", "scanorder"))

"Read `key` when it exists, otherwise return `nothing`."
_read_optional_dataset(group, key) =
    haskey(group, key) ? read(_hdf5_dataset(group, key)) : nothing

"Shift an envelope spectrum into natural order; field spectra are already ordered."
_shift_spectrum(array, field_mode::Bool) =
    field_mode ? array : FFTW.fftshift(array)

"Shift selected envelope-spectrum dimensions into natural order."
_shift_spectrum(array, dimensions, field_mode::Bool) =
    field_mode ? array : FFTW.fftshift(array, dimensions)

"""
    load_simulated_scan(filename; window_key="Iω_win", z_index=:end,
                        z_thickness=nothing) -> NamedTuple

Read the raw HDF5 file produced by [`run_scan`](@ref) and return its
contents as a NamedTuple, with all ω-dependent arrays fftshifted into
natural (centred) order and the requested z slice(s) extracted from the
propagated trace.

# Arguments
- `filename`: path to the `<scan_name>_collected.h5` file.

# Keyword arguments
- `window_key="Iω_win"`: which scansave dataset to use as the FROG trace.
  Common choices:
    * `"Iω_win"` — full-beam k-space integrated spectrum
    * `"Iω_win_reimaged"` — on-axis re-imaged spectrum
    * `"Iω_win_ωdep"` — ω-dependent window (Gaussian two-window setup)
    * `"Iω_win_ωdep_reimaged"` — ω-dependent re-imaged
- `z_index=:end`: which propagation z slice to use; the default `:end`
  picks the final (full-propagation) slice. Pass an `Int` for a specific
  slice index, or `:all` to return *every* z slice as a `(Nω, nz, Nτ)`
  stack (the equivalent of the trace at every saved material thickness).
- `z_thickness=nothing`: select the slice whose saved z position [m] is
  nearest this material thickness. Requires `/grid/zsave` in the file
  (written by recent `run_scan` runs); takes precedence over `z_index`.

# Returned NamedTuple

| field         | shape          | description                                          |
|---------------|----------------|------------------------------------------------------|
| `ω`           | `(Nω,)`        | absolute angular frequency [rad/s], natural order    |
| `ω0`          | scalar         | carrier angular frequency [rad/s] (from `/grid/ω0`)  |
| `t`           | `(Nt,)`        | time grid [s]                                        |
| `τ`           | `(Nτ,)`        | scan-variable delay grid [s]                         |
| `trace`       | 2-D or 3-D      | FROG trace; natural ω order; 3-D for `:all`           |
| `zsave`       | `(nz,)`        | realized propagation z positions [m]                  |
| `Iω`          | `(Nω,)`        | reference pulse spectrum, natural ω order            |
| `It`          | `(Nt,)`        | reference pulse temporal intensity                   |
| `τfwhm`       | scalar         | input pulse FWHM [s]                                 |
| `Iω_beamlet`  | `(Nω,)`        | input-vignetted beamlet spectrum                      |
| `It_beamlet`  | `(Nt,)`        | beamlet temporal intensity                            |
| `Ito_beamlet` | `(Nto,)`       | 8× oversampled beamlet intensity; shares `To`         |
| `To`          | `(Nto,)`       | 8× oversampled time grid [s]                          |
| `Ito`         | `(Nto,)`       | 8× oversampled temporal intensity                     |

The optional `zsave`, `It_beamlet`, `Ito_beamlet`, `To`, and `Ito` fields are returned
only when their corresponding datasets are present.

To inspect the full signal-beam collection (and hence the exact collection /
chromatic-vignetting efficiency `Iω_win ./ Iω_full`), load the signal-quadrant
reference with `window_key="Iω_full"`.
"""
function load_simulated_scan(
        filename::AbstractString;
        window_key::AbstractString = "Iω_win",
        z_index = :end,
        z_thickness::Union{Nothing, Real} = nothing
    )
    return HDF5.h5open(filename, "r") do f
        # --- Grid block ---
        haskey(f, "grid") || throw(
            ArgumentError("$filename is not a scansave file: missing /grid group")
        )
        g = _hdf5_group(f, "grid")
        ω_raw = read(_hdf5_dataset(g, "ω"))
        ω0 = read(_hdf5_dataset(g, "ω0"))
        # Field-mode files carry a monotonic rfft half-spectrum; envelope files carry the
        # FFT-ordered relative-frequency axis that needs shifting. Absent marker = envelope,
        # so every file written before field mode existed loads exactly as before.
        field_mode = haskey(g, "field_mode") &&
            read(_hdf5_dataset(g, "field_mode")) != 0
        t = read(_hdf5_dataset(g, "t"))
        Iω_raw = read(_hdf5_dataset(g, "Iω"))
        It = read(_hdf5_dataset(g, "It"))
        τfwhm = read(_hdf5_dataset(g, "τfwhm"))
        Iω_beam_raw = _read_optional_dataset(g, "Iω_beamlet")
        It_beam_raw = _read_optional_dataset(g, "It_beamlet")
        Ito_beam_raw = _read_optional_dataset(g, "Ito_beamlet")
        To_raw = _read_optional_dataset(g, "To")
        Ito_raw = _read_optional_dataset(g, "Ito")
        zsave = _read_optional_dataset(g, "zsave")

        # --- Scan variable ---
        haskey(f, "scanvariables") || throw(
            ArgumentError("$filename is not a valid delay scan: missing /scanvariables")
        )
        scanvariables = _hdf5_group(f, "scanvariables")
        haskey(scanvariables, "τ") || throw(
            ArgumentError("$filename is not a valid delay scan: missing /scanvariables/τ")
        )
        τ = read(_hdf5_dataset(scanvariables, "τ"))

        # --- Trace ---
        if !haskey(f, window_key)
            available = filter(_is_scan_dataset, keys(f))
            throw(
                ArgumentError(
                    "$filename has no window dataset '$window_key'; available top-level " *
                        "datasets: $available"
                )
            )
        end
        win_full = read(_hdf5_dataset(f, window_key))    # shape (Nω, nz, Nτ)
        nz = size(win_full, 2)

        # --- Select z slice(s): z_thickness > z_index ---
        if z_thickness !== nothing
            isnothing(zsave) && throw(
                ArgumentError(
                    "$filename has no /grid/zsave, so z_thickness cannot be selected"
                )
            )
            z_idx = argmin(abs.(zsave .- z_thickness))
            win = win_full[:, z_idx, :]                  # (Nω, Nτ)
        elseif z_index === :all
            win = win_full                               # (Nω, nz, Nτ)
        else
            z_idx = z_index === :end ? nz : Int(z_index)
            (1 <= z_idx <= nz) || throw(
                ArgumentError("z_index must be between 1 and $nz; got $z_idx")
            )
            win = win_full[:, z_idx, :]                  # (Nω, Nτ)
        end

        # --- Put the ω axis in ascending order (dim 1) ---
        ω = _shift_spectrum(ω_raw, field_mode)
        Iω = _shift_spectrum(Iω_raw, field_mode)
        trace = _shift_spectrum(win, 1, field_mode)
        Iω_beamlet = if isnothing(Iω_beam_raw)
            nothing
        else
            _shift_spectrum(Iω_beam_raw, field_mode)
        end

        nt = (; ω, ω0, t, τ, trace, Iω, It, τfwhm, field_mode)
        nt = isnothing(zsave) ? nt : merge(nt, (; zsave))
        nt = isnothing(Iω_beamlet) ? nt : merge(nt, (; Iω_beamlet))
        nt = isnothing(It_beam_raw) ? nt : merge(nt, (; It_beamlet = It_beam_raw))
        nt = isnothing(Ito_beam_raw) ? nt : merge(nt, (; Ito_beamlet = Ito_beam_raw))
        isnothing(To_raw) ? nt : merge(nt, (; To = To_raw, Ito = Ito_raw))
    end
end
