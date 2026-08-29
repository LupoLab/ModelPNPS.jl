# ============================================================================
# Memory budget
# ============================================================================

"""
    memory_budget(setup_args::NamedTuple) -> NamedTuple

Resident **device** memory one delay point of `setup_args` will need, and the **host** peak
[`build_setup`](@ref) will reach, broken down by buffer. `setup_args` is the same NamedTuple
[`run_scan`](@ref) and [`verify_against_collected`](@ref) take; only the grid-determining
entries are read, and building the 1-D time grid is the whole cost, so this is free to call.

This exists because guessing is expensive. The envelope path obeys a simple rule — 9 RK45
registers plus one transform buffer, i.e. 10× the field size, measured exactly on an
A40 — and **the field path does not**: its state is twice as long in ω, its nonlinear
evaluation runs on a grid twice as long again in time, and the no-THG response carries a
complex analytic-signal buffer on that grid. At the 40 µm production shape (N = 768)
that is 92 GiB against the envelope's 24. Finding this out by running is an hour of
rented GPU and a dead process.

!!! warning "The response's buffer appears on the first RHS, not at setup"
    `Nonlinear.KerrFieldNoTHG` allocates its analytic-signal buffer lazily, when it first
    sees a field. A card with room to spare after `build_setup` can therefore still die on
    the first step — 18 GiB later at the production shape. This function counts it; a
    measurement taken after `build_setup` alone will not.

Fields: `Nω`, `Nt`, `Nto`, `Nωo`, `field` (one state array), the per-buffer terms
`state`, `et_win`, `eto`, `ewo`, `pto`, `analytic`, `window`, `input`, and the totals
`device` and `host`. All in GiB.

`state` … `window` are what the transform and solver hold; `input` is the per-delay field
`delayed_input` produces. Only the first group is allocated by `build_setup`, so a
measurement taken across `build_setup` alone will fall short of `device` by `state`,
`analytic` and `input` — those appear when the first delay point runs.

The buffer set and its aliasing are `NonlinearRHS.TransFree`'s: `Pωo` always aliases `Eωo`
(the inverse transform consumes it), `Pto` aliases `Eto` when every response is pointwise
(the envelope Kerr, and field `:thg`, but not field `:nothg`), and `Et_win` exists only when
the grid is oversampled. `window` is the extraction window, which is device-resident when
save-time extraction is used — the default on a device.
"""
function memory_budget(setup_args::NamedTuple)
    a = setup_args
    N = a.N
    field = get(a, :field_mode, false)
    response = get(a, :response, :auto)
    ffac = get(a, :ffac, 6)
    grid = if field
        Grid.RealGrid(a.thickness, a.λ0, a.λlims, a.trange; ffac = ffac)
    else
        Grid.EnvGrid(
            a.thickness, a.λ0, a.λlims, a.trange;
            fftsize = get(a, :fftsize, :pow2)
        )
    end
    nwin = a.window isa AbstractSignalWindow ? 1 : length(a.window)
    return _budget(grid, N, field, response, nwin, get(a, :beamlets_on_host, false))
end

function _budget(grid, N, field, response, nwin, beamlets_on_host)
    Nω, Nt, Nto, Nωo = length(grid.ω), length(grid.t), length(grid.to), length(grid.ωo)
    b(n, sz) = n * N^2 * sz / 2^30
    tsz = field ? 8 : 16                 # the time-domain buffers are REAL in field mode
    # `:auto` resolves to `:nothg`, which is batched rather than pointwise, so it
    # needs its own polarisation buffer as well as the analytic signal.
    pointwise = !field || response === :thg
    state = 9 * b(Nω, 16)
    et_win = Nto == Nt ? 0.0 : b(Nt, tsz)
    eto = b(Nto, tsz)
    ewo = field ? b(Nωo, 16) : 0.0  # the envelope fast path has no oversampled buffers
    pto = pointwise ? 0.0 : b(Nto, tsz)
    analytic = (field && response !== :thg) ? b(Nto, 16) : 0.0
    window = nwin * b(Nω, 8)
    # The per-delay input. With `beamlets_on_host` the gate pair and the test beam live on
    # the host and `delayed_input` uploads one field per point; otherwise both beamlets are
    # device-resident and their delayed sum is a third. Either way it is live for the whole
    # propagation — the solver adopts it as its first register — so it belongs in the
    # resident total, and leaving it out is what made the measured figure look 5 % over.
    input = (beamlets_on_host ? 1 : 3) * b(Nω, 16)
    return (;
        Nω, Nt, Nto, Nωo, field = b(Nω, 16),
        state, et_win, eto, ewo, pto, analytic, window, input,
        device = state + et_win + eto + ewo + pto + analytic + window + input,
        # build_setup holds the unmasked HE11 field, the three masked beamlets, their
        # pre-summed gate pair and the window, all at once.
        host = 5 * b(Nω, 16) + window,
    )
end
