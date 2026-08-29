# ============================================================================
# Top-level constructor
# ============================================================================

"""
    build_setup(; λ0, τfwhm, energy, thickness, material,
                  mask_diam, mask_spacing, beam, window,
                  kwargs...) -> TGFROGSetup

Build the once-per-simulation setup: temporal/spatial grids, propagation
operators, FFT plans, the three input beamlets and the signal window(s).
The defaults reproduce the master script
`context/tgfrog_DUV_mask_apod6.jl`.

# Required keyword arguments

- `λ0`, `τfwhm`, `energy`         — pulse carrier wavelength [m], intensity
                                    FWHM [s], total pulse energy [J]
- `thickness`, `material`         — substrate thickness [m] and Luna
                                    `PhysData` material symbol (e.g. `:SiO2`)
- `mask_diam`, `mask_spacing`     — mask hole diameter [m] and edge-to-edge
                                    gap [m]
- `beam::AbstractInputBeam`       — input-beam model
                                    ([`HE11Beam`](@ref) or
                                    [`GaussianBeam`](@ref))
- `window`                        — signal-extraction window: a single
                                    [`AbstractSignalWindow`](@ref) or a
                                    vector of them (the latter is used by the
                                    Gaussian example to save both the
                                    ω-independent and ω-dependent windows in
                                    one run)

# Optional keyword arguments

- `trange = 40e-15`               — temporal window [s]
- `λlims  = (160e-9, 500e-9)`     — wavelength window [m]
- `R, N`                          — spatial half-width [m] and grid size; if
                                    either is `nothing`, both are computed
                                    via [`optimal_spatial_grid`](@ref)
- `apod, apod_param`              — apodisation for the *input-beamlet*
                                    masks (only relevant for `HE11Beam`)
- `geometry = :tg`                — beam layout. `:tg` is the four-hole boxcar
                                    TG-FROG geometry (three inputs, signal in the
                                    fourth corner); `:sd` places two collinear
                                    holes for self-diffraction, whose `2k_E - k_G`
                                    signal sits one slot further out on the same
                                    axis. It selects both the beamlet layout and
                                    the k-space bound used by
                                    [`optimal_spatial_grid`](@ref). `:sd` is
                                    implemented for `HE11Beam` only; any other
                                    value throws an `ArgumentError`
- `fftsize = :pow2`               — how the temporal sample count is rounded up:
                                    `:pow2` to the next power of two, `:smooth` to
                                    the next even 2,3,5-smooth size (a smaller grid
                                    for the same resolution). Envelope mode only —
                                    `Grid.RealGrid` has no such control
- `GDD = 0.0`, `TOD = 0.0`        — group-delay and third-order dispersion
                                    [s², s³] applied to the input pulse
- `input_pulse = nothing`         — an [`InputPulseData`](@ref): use this
                                    measured/simulated complex spectrum as the
                                    source instead of the analytic Gaussian
                                    (`HE11Beam` only). `λ0`/`τfwhm` then serve
                                    only as nominal values (mask apodisation
                                    defaults, diagnostics, metadata); `energy`
                                    still sets the beam energy (the data's
                                    amplitude scale is irrelevant); GDD/TOD
                                    compose on top if nonzero. See
                                    [`load_input_pulse`](@ref),
                                    [`spectral_window!`](@ref),
                                    [`center_pulse!`](@ref)
- `raman = false`                 — include the delayed (Raman) part of the
                                    nonlinear response via
                                    [`FrozenRamanPolarEnv`](@ref); requires a
                                    material with an `:intermediate` Raman
                                    model in `Luna.PhysData.raman_parameters`
                                    (for `:SiO2` the multimode
                                    Hollenbeck–Cantrell response). The total
                                    polarisation is
                                    `(3/4)ε₀χ³[(1-f_R)|E|²E + f_R E(h_R⊛|E|²)]`
                                    — equal prefactors on both terms, the
                                    envelope-defined `f_R` convention of
                                    Luna's `prop_gnlse`, so the quasi-static
                                    limit reproduces the Kerr-only response
                                    exactly
- `raman_fraction = 0.18`         — envelope-defined nuclear fraction `f_R`
                                    of χ³ (the Blow–Wood silica value)
- `raman_impl = :batched`         — Raman implementation: `:batched` computes
                                    the convolution for all transverse points
                                    at once (two batched FFTs per RHS
                                    evaluation); `:frozen` is the legacy
                                    per-column [`FrozenRamanPolarEnv`](@ref).
                                    Results agree to rounding accuracy
- `field_mode = false`            — propagate the real, carrier-resolved field on a
                                    `Luna.Grid.RealGrid` instead of the complex envelope on
                                    an `EnvGrid`. There is then no carrier/envelope split,
                                    no dropped third-harmonic term and no negative-frequency
                                    wrap; the cost is roughly 2× the memory and 3×
                                    the time per delay point (measured 3.0× at N = 64 and
                                    3.3× at N = 128, at matched step counts). The envelope
                                    path is untouched and remains the default
- `response = :auto`              — field-mode nonlinearity: `:nothg` (= `:auto`)
                                    for `(3/4) ε₀ χ³ |E_a|² E`, the same physics
                                    content as the envelope `Kerr_env` and hence the
                                    response for an envelope-versus-field comparison;
                                    `:thg` for `ε₀ χ³ E³`,
                                    which adds what the envelope drops. Ignored unless
                                    `field_mode = true`
- `ffac = 6`                      — field-mode nonlinear-grid sampling factor,
                                    forwarded to
                                    `Grid.RealGrid`. 6 (the default) sizes the fine grid for
                                    `E³`; 4 is enough for `:nothg` alone and typically
                                    removes the oversampling entirely, halving memory and
                                    per-step cost. It changes the grid, so use it only with
                                    a convergence check against the default
- `raman`                         — not implemented in field mode (see the error message
                                    there for why)
- `beamlet_profile = true`        — store the gate beamlet's spatially resolved complex
                                    focal field as a radial profile `Eω_beamlet_r_re/_im`
                                    `(Nω, nr)` plus the radius axis `beamlet_r` in metres,
                                    so the pulse that actually drives the signal can be
                                    computed rather than assumed. Diagnostic only — no
                                    propagation result depends on it — costing one 2-D
                                    inverse transform per ω ONCE at setup and ~130 kB in
                                    the file. See [`_beamlet_profile`](@ref) for where the
                                    beamlet is (the focus, not a BOXCARS corner) and why
                                    the geometric tilt is removed first
- `beamlet_profile_nr = 64`       — radial samples. Measured radial closure against
                                    `Iω_beamlet` on the production geometry, at 200 /
                                    260 / 350 nm: 0.955/0.972/0.980 at nr = 32,
                                    0.975/0.984/0.986 at 64, 0.980/0.987/0.988 at 128.
                                    64 is where it has essentially converged, for 262 kB
                                    at `Nω = 256`; the residual ~1.5 % is truncation at
                                    `rmax` plus the beamlet's real azimuthal asymmetry
- `beamlet_profile_rmax_units = 6` — outer radius, in units of `λ0·f_foc/mask_diam`
                                    (`w0` for [`GaussianBeam`](@ref))
- `factored_linop = true`         — use Luna's lazy (factored) linear operator
                                    and normalisation, saving two field-sized
                                    arrays; bit-identical to the materialised
                                    versions
- `store_window = true`           — store the materialised window array(s) in
                                    the output metadata (≈1 GiB at production
                                    size); the window parameters (`window_def`)
                                    are always stored and reconstruct the
                                    array via [`build_window`](@ref)
- `arraytype = Array`             — array type the propagation runs on. Pass
                                    `:cuda` to build the beamlets, operators and
                                    window on the GPU; it is resolved lazily, so a
                                    scan script passes it inside `setup_args`
                                    rather than as a [`run_scan`](@ref) keyword and
                                    the GPU package is then loaded on the compute
                                    node, never on the submitting host
- `beamlets_on_host = false`      — on a device run, keep the pre-built beamlets in
                                    host memory and upload the delayed sum once per
                                    delay point — two fewer resident device fields
                                    in exchange for one transfer per point. Use it
                                    when the card is memory-bound; see
                                    [`memory_budget`](@ref)
- `optimal_grid_kwargs`           — extra kwargs forwarded to
                                    `optimal_spatial_grid`
- `extra_grid_metadata`           — additional entries merged into the
                                    output `combined_grid` dict
"""
function build_setup(;
        λ0, τfwhm, energy, thickness, material,
        mask_diam, mask_spacing,
        beam::AbstractInputBeam,
        window,
        trange = 40.0e-15,
        λlims = (160.0e-9, 500.0e-9),
        R = nothing, N = nothing,
        apod::Symbol = :supergauss, apod_param = nothing,
        geometry::Symbol = :tg,
        input_pulse::Union{Nothing, InputPulseData} = nothing,
        GDD = 0.0, TOD = 0.0,
        raman::Bool = false, raman_fraction::Float64 = 0.18,
        raman_impl::Symbol = :batched,
        field_mode::Bool = false,
        response::Symbol = :auto,
        ffac::Real = 6,
        beamlet_profile::Bool = true,
        beamlet_profile_nr::Int = 64,
        beamlet_profile_rmax_units::Real = 6,
        factored_linop::Bool = true,
        store_window::Bool = true,
        arraytype = Array,
        beamlets_on_host::Bool = false,
        fftsize::Symbol = :pow2,
        optimal_grid_kwargs = (;),
        extra_grid_metadata = Dict{String, Any}()
    )
    # --- Validate the beam layout ------------------------------------------
    # Checked here rather than where it is consumed: `geometry` reaches two
    # independent consumers (the grid sizing below and `build_beamlets`), each of
    # which treats anything that is not `:sd` as `:tg`, so an unknown symbol would
    # otherwise produce a TG run with no diagnostic at all.
    geometry in (:tg, :sd) || throw(
        ArgumentError("geometry must be :tg or :sd; got :$geometry")
    )
    # Only the HE₁₁ builder implements the self-diffraction layout; the Gaussian
    # builder always places three beams at the boxcar corners, so an :sd request
    # there would silently give a TG field on an SD-sized grid.
    geometry === :tg || beam isa HE11Beam || throw(
        ArgumentError(
            "geometry = :sd is only supported with HE11Beam; got $(typeof(beam))"
        )
    )

    # --- Resolve spatial grid ----------------------------------------------
    if R === nothing || N === nothing
        f_foc = beam.f_foc
        R, N = optimal_spatial_grid(
            f_foc, mask_diam, mask_spacing,
            λlims[1], λlims[2];
            geometry = geometry, optimal_grid_kwargs...
        )
    end

    # --- Build Luna grids --------------------------------------------------
    # `field_mode` swaps the envelope (analytic-field) grid for the field-resolved one.
    # `RealGrid` has no `fftsize` control — its propagated grid is always the next
    # power of two above 1.1 ωmax — so that keyword applies to the envelope path only.
    grid = if field_mode
        Grid.RealGrid(thickness, λ0, λlims, trange; ffac = ffac)
    else
        Grid.EnvGrid(thickness, λ0, λlims, trange; fftsize = fftsize)
    end
    xygrid = Grid.FreeGrid(R, N)

    # --- Material dispersion + Kerr (+ optional Raman) nonlinearity -------
    χ3 = PhysData.χ3(material)
    if field_mode
        raman && throw(
            ArgumentError(
                "raman=true is not implemented in field mode. The delayed response " *
                    "itself is available (Luna.Nonlinear.RamanPolarFieldBatched), " *
                    "but the nuclear fraction f_R below is defined in the ENVELOPE " *
                    "convention — the 3/2 that reconciles Kerr_env's internal 3/4 " *
                    "with the Raman kernel's 1/2 — " *
                    "and that factor does not carry over to a carrier-resolved field " *
                    "unexamined. Deriving it needs its own quasi-static consistency " *
                    "test, " *
                    "like the envelope one in the test suite."
            )
        )
        responses = _field_responses(response, χ3)
    elseif raman
        # Split χ³ into an instantaneous electronic part and a delayed
        # nuclear part carrying the envelope-defined fraction
        # `raman_fraction` (the Blow–Wood f_R = 0.18 for silica), following
        # the convention of Luna's `prop_gnlse`: both terms must end with
        # the SAME prefactor, so that the quasi-static (long-pulse) limit of
        # the split response is identical to the Kerr-only response.
        # `Kerr_env` applies the envelope factor 3/4 internally while the
        # Raman response's `sqr!` applies 1/2, so the Raman `scale` carries
        # the compensating (3/4)/(1/2) = 3/2 (and the explicit ε₀, which
        # `Kerr_env` adds internally):
        #     P = (3/4) ε₀ χ³ [ (1-f_R) |E|²E + f_R E (h_R ⊛ |E|²) ].
        # (Luna's low-level silica envelope examples omit the 3/2 and thus
        # under-weight Raman by 2/3 relative to this convention.) The
        # equal-prefactor property is enforced by the quasi-static
        # consistency test in the test suite.
        rp = PhysData.raman_parameters(material)
        rp.kind == :intermediate || throw(
            ArgumentError(
                "raman=true requires an :intermediate condensed-phase Raman model, " *
                    "such as :SiO2; $material has kind :$(rp.kind)"
            )
        )
        rr = Raman.raman_response(
            grid.to, material,
            1.5 * raman_fraction * PhysData.ε_0 * χ3
        )
        # :batched (default) computes the Raman convolution for all transverse points
        # at once with batched FFTs (two per RHS evaluation instead of two small serial
        # FFTs per transverse column); :frozen is the legacy per-column implementation,
        # kept for A/B comparison. Results agree to rounding accuracy (~1e-15 relative).
        Rresp = if raman_impl === :batched
            Nonlinear.RamanPolarEnvBatched(grid.to, rr)
        elseif raman_impl === :frozen
            FrozenRamanPolarEnv(grid.to, rr)
        else
            throw(
                ArgumentError("raman_impl must be :batched or :frozen; got :$raman_impl")
            )
        end
        responses = (Nonlinear.Kerr_env((1 - raman_fraction) * χ3), Rresp)
    else
        responses = (Nonlinear.Kerr_env(χ3),)
    end
    nfun = PhysData.ref_index_fun(material, lookup = false)
    nfunreal = real ∘ nfun
    # factored (lazy) operators compute their elements on demand instead of storing two
    # field-sized arrays; bit-identical to the materialised versions (Luna guarantees
    # and tests this)
    arraytype = Luna.resolve_arraytype(arraytype)
    linop = LinearOps.make_const_linop(
        grid, xygrid, nfunreal;
        factored = factored_linop, arraytype
    )
    normfun = NonlinearRHS.const_norm_free(
        grid, xygrid, nfunreal;
        factored = factored_linop, arraytype
    )
    densityfun = Returns(1)
    _, transform, FT = Luna.setup(
        grid, xygrid, densityfun, normfun, responses, ();
        arraytype
    )
    _, energyfun_ω = Fields.energyfuncs(grid, xygrid)

    # --- 1-D reference spectrum (used by the HE₁₁ builder and as diagnostic)
    # Complex on an envelope grid, real-to-complex on a field-resolved one: `GaussField`
    # dispatches its time-domain shape on the grid but takes the plan from here, and
    # `prop_taylor!` applies ϕ on the ABSOLUTE ω axis, which both grids carry.
    FT1d = _plan_1d(grid)
    ϕ = [0.0, 0.0, GDD, TOD]  # up to 3rd order
    if input_pulse === nothing
        pulse = Fields.GaussField(; λ0 = λ0, τfwhm = τfwhm, energy = energy, ϕ = ϕ)
        Eω = pulse(grid, FT1d)
    else
        # Data-driven pulse. Valid only for the HE₁₁ model, where this 1-D
        # reference IS the pulse (the chromatic vignetting composes with it
        # exactly downstream); the Gaussian builder constructs its own field
        # from (λ0, τfwhm) and would silently ignore the data.
        beam isa HE11Beam || throw(
            ArgumentError(
                "input_pulse is only supported with HE11Beam"
            )
        )
        Eω = interp_input_pulse(grid, input_pulse)
        maximum(abs, Eω) > 0 || throw(
            ArgumentError(
                "input_pulse has no overlap with the simulation band λlims"
            )
        )
        # Taylor phase composes on top of the data (usually leave GDD=TOD=0;
        # λ0 is then only the expansion point).
        any(!iszero, ϕ) && Fields.prop_taylor!(Eω, grid, ϕ, λ0)
    end

    # --- Build three input beamlets ---------------------------------------
    geom = (; mask_diam, mask_spacing, f_foc = beam.f_foc, λ0, τfwhm, geometry)
    Eωk_g1, Eωk_g2, Eωk_t_base, _Iω_beamlet, beam_meta =
        build_beamlets(
        beam, grid, xygrid, geom, Eω, energy, energyfun_ω;
        apod = apod, apod_param = apod_param, ϕ = ϕ,
        profile = beamlet_profile, profile_nr = beamlet_profile_nr,
        profile_rmax_units = beamlet_profile_rmax_units
    )

    # --- Build signal window(s) -------------------------------------------
    window_array, window_suffix = _build_window_set(window, grid, xygrid; λ0 = λ0)

    # --- Assemble combined_grid metadata ----------------------------------
    # Record the nonlinearity model alongside the grids (explicit entries win
    # over these defaults if the caller supplies their own).
    extra_grid_metadata = merge(
        Dict{String, Any}(
            "raman" => raman ? 1 : 0,
            "raman_fraction" => raman_fraction,
            # Data-driven source marker + the 1-D input
            # spectrum actually injected (post-interp),
            # so a reader can reconstruct what was run
            # without the original data file.
            "input_pulse" => input_pulse === nothing ? 0 : 1,
            "Iω_input" => abs2.(Eω),
            # How the file was made. `field_mode` is also the
            # marker every reader needs: a field-mode /grid/ω is a
            # monotonic rfft half-spectrum, not the FFT-ordered
            # relative-frequency axis of an envelope file, so it
            # must NOT be fftshifted on load. Absent = envelope,
            # which is what every pre-existing file is.
            # 0 when the diagnostic focal profile is off, so a
            # file always says whether it should have one.
            "beamlet_profile" => beamlet_profile ? 1 : 0,
            "field_mode" => field_mode ? 1 : 0,
            "response" => _response_name(
                field_mode, response,
                raman
            ),
            # `ffac` is the field grid's nonlinear-sampling factor
            # (6 = sized for E³, 4 = sized for |E_a|²E only). It is
            # not a field of Grid.RealGrid, so record it here.
            "ffac" => field_mode ? float(ffac) : 0.0,
            # The MASK GEOMETRY, recorded so the file is
            # self-describing. Retrieval codes need d/D
            # = (spacing + D)/2D to build the smearing
            # kernel, and until now it was carried by
            # hand from script to script: a sweep run
            # against a gap-1000 trace while still
            # holding the gap-500 default silently built
            # a kernel 25% too narrow, worth ~2% of
            # retrieved duration with NO signature in
            # the trace error. It cannot be recovered
            # reliably after the fact either — the
            # signal-window fields only pin it for some
            # geometries, and for SD the window sits at
            # 1.5s rather than d.
            "mask_diam" => mask_diam,
            "mask_spacing" => mask_spacing,
            "f_foc" => beam.f_foc,
            "geometry" => string(geometry)
        ),
        extra_grid_metadata
    )
    combined_grid = _combined_grid(
        grid, xygrid, beam_meta,
        window, window_array, window_suffix,
        Eω, λ0, τfwhm, material, thickness,
        extra_grid_metadata; store_window
    )

    # Beamlets are built on the host (masks, Bessel profiles and FFTs are host code).
    # Move them to the propagation's array type unless asked to keep them here: at the
    # largest campaign shapes those two fields are the difference between fitting a card
    # and not, and the per-point upload that replaces them is a fraction of a second.
    # `Eωk_g2 === nothing` is the self-diffraction geometry: two beams, not three, so
    # the "gate pair" is the single gate beamlet (see the :sd branch of build_beamlets).
    Eωk_g12 = isnothing(Eωk_g2) ? Eωk_g1 : Eωk_g1 .+ Eωk_g2
    if arraytype !== Array && !beamlets_on_host
        Eωk_g12 = Adapt.adapt(arraytype, Eωk_g12)
        Eωk_t_base = Adapt.adapt(arraytype, Eωk_t_base)
    end
    # The delay phase must live wherever the BEAMLETS do, not wherever the propagation
    # does: `delayed_input` multiplies them together in one broadcast, and mixing a host
    # array into a device broadcast is rejected outright by CUDA. With
    # `beamlets_on_host` the whole expression is evaluated on the host and the result
    # uploaded afterwards.
    ωd = Eωk_g12 isa Array ? grid.ω : Adapt.adapt(arraytype, collect(grid.ω))

    return TGFROGSetup(
        λ0, τfwhm, energy, thickness, material, mask_diam, mask_spacing,
        grid, xygrid, linop, transform, FT, energyfun_ω,
        Eωk_g12, Eωk_t_base, ωd, Eω,
        window, window_array, window_suffix, combined_grid
    )
end

# ----- Field-mode response selector ----------------------------------------

"""
    _field_responses(response, χ3) -> Tuple

The nonlinear response for a field-resolved (`RealGrid`) run.

- `:nothg` (and `:auto`) — `(3/4) ε₀ χ³ |E_a|² E` via
  `Luna.Nonlinear.Kerr_field_nothg`. This is the SAME physics content as the
  envelope `Kerr_env`, evaluated on a carrier-resolved field, so an envelope-versus-field
  comparison made with it isolates *representation* error with nothing else changed. It is
  the default for exactly that reason.
- `:thg` — `ε₀ χ³ E³` via `Luna.Nonlinear.Kerr_field`, which adds the third-harmonic
  and counter-rotating terms the envelope drops. The difference between the two runs is
  precisely what the envelope omits.

!!! note "What `:thg` does and does not propagate on a UV window"
    The third harmonic is generated on the fine grid and then discarded by the crop back to
    the propagated grid whenever it falls outside the window (at λlims = (143, 600) nm the
    3ω band of a 2 fs 260 nm pulse starts above ωmax). The within-band counter-rotating
    terms are retained, and those are the real difference from the envelope. Propagating
    the third harmonic itself needs λlims extended to ~λ0/3.
"""
function _field_responses(response::Symbol, χ3)
    if response === :auto || response === :nothg
        return (Nonlinear.Kerr_field_nothg(χ3),)
    elseif response === :thg
        return (Nonlinear.Kerr_field(χ3),)
    else
        throw(
            ArgumentError(
                "field-mode response must be :auto, :nothg, or :thg; got :$response"
            )
        )
    end
end

"""Canonical name of the response actually used, for the output metadata."""
function _response_name(field_mode::Bool, response::Symbol, raman::Bool)
    if field_mode
        return response === :auto ? "nothg" : string(response)
    end
    return raman ? "kerr_env+raman" : "kerr_env"
end

# ----- Window-set helper (single vs vector of windows) ---------------------

_build_window_set(w::AbstractSignalWindow, grid, xygrid; λ0) =
    (build_window(w, grid, xygrid; λ0 = λ0), [""])

function _build_window_set(
        ws::AbstractVector{<:AbstractSignalWindow},
        grid, xygrid; λ0
    )
    arrs = [build_window(w, grid, xygrid; λ0 = λ0) for w in ws]
    suffixes = _default_suffixes(ws)
    return arrs, suffixes
end

# Default suffixes: first window gets "", subsequent windows get "_2", "_3", ...
# unless we recognise the gaussian (Planck + PlanckOmega) two-window pattern,
# in which case we emit ["", "_ωdep"] for compatibility with the master script.
function _default_suffixes(ws::AbstractVector{<:AbstractSignalWindow})
    if length(ws) == 2 && ws[1] isa PlanckWindow && ws[2] isa PlanckOmegaWindow
        return ["", "_ωdep"]
    end
    return [i == 1 ? "" : "_$i" for i in eachindex(ws)]
end

# ----- combined_grid metadata helper ---------------------------------------

# ----- Window parameter serialisation (reconstructible via build_window) ----

_window_def(w::PhysicalMaskWindow) = Dict{String, Any}(
    "type" => "PhysicalMaskWindow", "holex" => w.holex, "holey" => w.holey,
    "holediam" => w.holediam, "zmask" => w.zmask, "apod" => string(w.apod),
    "apod_param" => (w.apod_param === nothing ? "default" : w.apod_param)
)

_window_def(w::PlanckWindow) = Dict{String, Any}(
    "type" => "PlanckWindow", "kxc" => w.kxc, "kyc" => w.kyc,
    "kwidth" => w.kwidth, "pad" => w.pad
)

_window_def(w::PlanckOmegaWindow) = Dict{String, Any}(
    "type" => "PlanckOmegaWindow", "xc" => w.xc, "yc" => w.yc,
    "holediam" => w.holediam, "f_foc" => w.f_foc, "pad" => w.pad
)

function _combined_grid(
        grid, xygrid, beam_meta::Dict,
        window, window_array, window_suffix::Vector{String},
        Eω::AbstractVector, λ0, τfwhm, material, thickness,
        extra::Dict; store_window::Bool = true
    )
    cg = Dict{String, Any}()
    for (k, v) in pairs(Grid.to_dict(grid))
        cg[string(k)] = v
    end
    for (k, v) in pairs(Grid.to_dict(xygrid))
        cg[string(k)] = v
    end

    # Always-present diagnostics.
    # `Eω` is the grid's own spectral representation: FFT-ordered about the carrier on an
    # EnvGrid, a monotonic rfft half-spectrum on a RealGrid. `_to_time` inverts whichever it
    # is, and `_envelope_intensity` reduces the result to |A|² either way, so
    # `It`/`Ito` mean the same physical thing in an envelope and a field-mode file.
    Et = _to_time(grid, Eω)
    It = _envelope_intensity(grid, Et)
    Iω = abs2.(Eω)
    # `Maths.oversample` has separate real and complex methods and picks the right one.
    to, eo = Maths.oversample(grid.t, Et; factor = 8)
    Ito = _envelope_intensity(grid, eo)
    cg["Iω"] = Iω
    cg["It"] = It
    cg["To"] = to
    cg["Ito"] = Ito
    cg["τfwhm"] = τfwhm
    cg["material"] = string(material)
    cg["thickness"] = thickness
    # `Grid.to_dict` writes every field of the grid struct, and `RealGrid` has no
    # `ω0` — it propagates the field itself, not an envelope about a carrier. Readers
    # key off /grid/ω0 unconditionally, so write the carrier explicitly in that case.
    haskey(cg, "ω0") || (cg["ω0"] = 2π * PhysData.c / λ0)

    # Beam-specific metadata (Iω_beamlet for both models, w0/Δk/crossingθ for
    # Gaussian, a/f_coll/f_foc for HE₁₁).
    for (k, v) in beam_meta
        cg[k] = v
    end

    # Time-domain profile of the (input-vignetted) beamlet — "the pulse that is
    # actually retrieved" when a measured trace is inverted. The mask is a real
    # amplitude filter, so the beamlet's effective spectral phase is the input
    # pulse phase: we reconstruct the beamlet envelope by combining the saved
    # power spectrum Iω_beamlet with the input spectral phase, then transform to
    # time. (For an FTL input this is just the transform-limited profile of the
    # vignetted beamlet spectrum.)
    if haskey(cg, "Iω_beamlet")
        Iω_beamlet = cg["Iω_beamlet"]
        reg = maximum(Iω) * 1.0e-12
        phase = [
            Iω[i] > reg ? Eω[i] / sqrt(Iω[i]) : zero(eltype(Eω))
                for i in eachindex(Eω)
        ]
        Eω_beamlet = sqrt.(max.(Iω_beamlet, 0)) .* phase
        et_beamlet = _to_time(grid, Eω_beamlet)
        cg["It_beamlet"] = _envelope_intensity(grid, et_beamlet)
        _, eob_beamlet = Maths.oversample(grid.t, et_beamlet; factor = 8)
        cg["Ito_beamlet"] = _envelope_intensity(grid, eob_beamlet)  # shares "To" above
        # The COMPLEX beamlet spectrum (amplitude and phase), stored as two
        # real datasets for cross-language portability (h5py reads HDF5.jl's
        # native complex compound awkwardly). This is the retrievable ground
        # truth WITH its spectral phase — it carries any input chirp (GDD/TOD)
        # exactly, since the mask is a real amplitude filter — enabling
        # complex-field retrieval-error metrics (Geib ε) and direct truth-GDD
        # measurements instead of intensity-only comparisons.
        cg["Eω_beamlet_re"] = real.(Eω_beamlet)
        cg["Eω_beamlet_im"] = imag.(Eω_beamlet)
    end
    # The complex INPUT (source) spectrum, same encoding: the pre-mask truth.
    cg["Eω_re"] = real.(Eω)
    cg["Eω_im"] = imag.(Eω)

    # Window parameters under "window_def*" (+ suffixes): always stored — they are a few
    # scalars and losslessly reconstruct the window via build_window. Flattened to
    # individual entries (scansave writes plain datasets, not nested groups).
    if window isa AbstractSignalWindow
        for (fk, fv) in _window_def(window)
            cg["window_def_" * fk] = fv
        end
    else
        for (suf, w) in zip(window_suffix, window)
            for (fk, fv) in _window_def(w)
                cg["window_def" * suf * "_" * fk] = fv
            end
        end
    end
    # Window arrays under "window" (+ optional suffixes for multi-window). At production
    # size a 3-D window is a ~1 GiB Float64 array in every collected file; pass
    # build_setup(store_window=false) to skip it and rely on window_def instead.
    if store_window
        if window isa AbstractSignalWindow
            cg["window"] = window_array
        else
            for (suf, arr) in zip(window_suffix, window_array)
                cg["window" * suf] = arr
            end
        end
    end

    # User-supplied extras override anything above on collision.
    for (k, v) in extra
        cg[k] = v
    end
    return cg
end
