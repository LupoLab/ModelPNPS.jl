# =============================================================================
# ModelPNPS TG-FROG delay points on an H100/H200: accuracy A/B, then timing.
#
# Companion to Luna's test/manual/hpc_gpu_bench.jl (which benchmarks the MODAL
# transform). This one exercises the 3-D free-space path — one `(Nω, N, N)` state on
# `Grid.FreeGrid` — at the shapes the production campaigns actually use, on the envelope
# grid (`Grid.EnvGrid`, the default) or the field-resolved one (`Grid.RealGrid`,
# `--field=on`; see FIELD MODE below).
#
# Run through examples/h200_modelpnps_suite.sh, or directly:
#   julia --project=/workspace/code/dev h200_bench.jl --mode=accuracy
#   julia --project=/workspace/code/dev h200_bench.jl --mode=bench --cases=dd05,04
#   julia --project=/workspace/code/dev h200_bench.jl --mode=bench --field=on --cases=04
#   julia --project=/workspace/code/dev h200_bench.jl --mode=accuracy --field=on --case=04
#
# Options
#   --mode=accuracy|bench   accuracy: GPU-only self-consistency + rtol convergence
#                           bench:    GPU delay points at production shapes
#   --case=NAME             accuracy case (default dd05)
#   --cpu=1                 add a host reference to the accuracy run (slow; use the HPC)
#   --cases=a,b,c           bench cases, see CASES below (default dd05,04,dd20,100um)
#   --points=N              delay points per case (default 1)
#   --out=FILE              append a CSV row per point
#   --extract=on|off|both   save-time extraction (default on; `both` A/Bs it)
#   --field=on|off          FIELD-RESOLVED mode (Grid.RealGrid) instead of the envelope
#   --response=nothg|thg    field-mode nonlinearity (default nothg — see below)
#   --ffac=6|4              field-mode nonlinear-grid factor (default 6)
#   --ratio=on|off          in a field bench, also time the ENVELOPE point at the same
#                           shape and report the ratio (default on when --field=on)
#
# FIELD MODE
#   `--field=on` is the benchmark for the 1 fs question: whether the unexplained ~5e-3
#   retrieval residual belongs to the retrieval model or to the envelope approximation.
#   At 260 nm a 1 fs pulse is 1.15 optical cycles, and the production envelope grid's
#   relative-frequency window has its red edge at exactly DC — so the same geometry is run
#   on a real, carrier-resolved field and the two traces compared. Everything below is the
#   PRODUCTION 1 fs configuration already (τfwhm = 1 fs, 0.1 µJ, 260 nm, λlims 143-600 nm);
#   `--field=on` changes only the representation.
#
#   It is much heavier than the envelope, and in a way the old `10 × field size` rule of
#   thumb does not capture, so this script now computes the budget per case and REFUSES a
#   case that will not fit rather than letting it die an hour in. At the `04` production
#   shape (N = 768, 40 µm):
#
#       envelope                        Nω  256   ~24 GiB device,  ~12 GiB host peak
#       field :nothg  ffac 6            Nω  513   ~92 GiB device,  ~25 GiB host peak
#       field :thg    ffac 6            Nω  513   ~65 GiB device
#       field :nothg  ffac 4            Nω  513   ~65 GiB device
#
#   so :nothg at ffac 6 is H200-only; :thg and :nothg-at-ffac-4 fit an 80 GB H100; an A40
#   fits none of them. `dd20` (N = 1024) needs 164 GiB in :nothg and does not fit anything.
#
#   :nothg is (3/4)ε₀χ³|E_a|²E — the SAME physics content as the envelope's Kerr_env, so
#   an envelope-versus-field comparison made with it isolates representation error with
#   nothing else changed. That is what the campaign wants, and it is also the expensive
#   one (a batched analytic signal, hence the extra buffers). :thg is plain ε₀χ³E³.
#
#   ffac 4 halves the nonlinear grid. It is validated for :nothg (Luna's
#   "ffac convergence for the no-THG response" testset measures 4e-8, unchanged from z=0
#   to z=end) but it CHANGES δω, the realised time window and in general Nω — so a
#   ffac-4 trace is not bin-for-bit comparable with a ffac-6 one or with the delivered
#   envelope files. Use it for a self-contained scan, not for the comparison.
#
# CUDA is imported at the top level here deliberately. The cluster scripts must
# NOT do that (their login node faults precompiling CUDA.jl, hence the lazy
# `arraytype=:cuda` symbol and its world-age dance); a rented pod has the GPU
# directly attached, so the simple thing is also the correct one.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna
import CUDA
import Printf: @printf, @sprintf
import Dates

# ------------------------------------------------------------------ options --
opt(name, default) = begin
    p = "--$name="
    i = findfirst(a -> startswith(a, p), ARGS)
    isnothing(i) ? default : ARGS[i][length(p)+1:end]
end
MODE     = opt("mode", "accuracy")
POINTS   = parse(Int, opt("points", "1"))
OUT      = opt("out", "")
EXTRACT  = opt("extract", "on")
FIELD    = opt("field", "off") == "on"
RESPONSE = Symbol(opt("response", "nothg"))
FFAC     = parse(Int, opt("ffac", "6"))
RATIO    = opt("ratio", FIELD ? "on" : "off") == "on"
FIELD && RESPONSE in (:nothg, :thg) ||
    FIELD && error("--response must be nothg or thg, got $RESPONSE")

# --------------------------------------------------------- shared physics ----
# Identical to the production drivers (04 / 05 / 07): 1 fs, 0.1 µJ, 260 nm into
# 40 µm (or 100 µm) of SiO2 through the boxcars mask. Only the GRID differs
# between cases, which is the whole point — grid shape is what sets cost.
const TWIN_SAVES_ONLY = 1_000_000_000

function setup_args_for(; gap_mm, N, trange, thickness, field=FIELD)
    gap  = gap_mm * 1e-3
    d    = gap/2 + 1.0e-3/2
    beam = TS.HE11Beam(125e-6, 5.0, 0.1)
    win  = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                                 zmask=0.1, apod=:tanh)
    base = (; λ0=260e-9, τfwhm=1.0e-15, energy=0.1e-6,
              material=:SiO2, thickness=thickness,
              mask_diam=1.0e-3, mask_spacing=gap, λlims=(143e-9, 600e-9),
              beam=beam, window=win, apod=:supergauss, apod_param=16,
              trange=trange, store_window=false, R=366.0e-6, N=N)
    field || return base
    # `beamlets_on_host`: two fewer resident device fields (9 GiB at the 04 shape), traded
    # for one host-to-device transfer per delay point — a fraction of a second against a
    # propagation of minutes. Worth it in field mode in a way it is not at envelope sizes,
    # because the field grid is what makes the card tight in the first place.
    return (; base..., field_mode=true, response=RESPONSE, ffac=FFAC,
              beamlets_on_host=true)
end

const ZSAVE_40  = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
                   12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6
const ZSAVE_100 = [0.0, 2.0, 4.0, 8.0, 9.5, 16.0, 25.0, 40.0, 60.0, 100.0] .* 1e-6

# Production shapes. `field_gib` is one ComplexF64 (Nω, N, N) array; the solver
# holds 9 of them plus one transform buffer, so device memory ≈ 10 × that
# (measured exactly on an A40 at the `04` shape).
const CASES = Dict(
  # name      gap  N     trange   thickness  zsave      what it is
  "dd05"  => (gap_mm=0.5, N=640,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="07 d/D series, gap 0.5 mm"),
  "04"    => (gap_mm=1.0, N=768,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="04 Kerr production — the A40 reference shape"),
  "dd20"  => (gap_mm=2.0, N=1024, trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40,
              note="07 d/D series, gap 2.0 mm — marginal on a 48 GB card"),
  "100um" => (gap_mm=1.0, N=640,  trange=220e-15, thickness=100e-6, zsave=ZSAVE_100,
              note="05 anchor: Nω=512, 2.5× the propagation distance"),
)
CASE_NAMES = split(opt("cases", "dd05,04,dd20,100um"), ",")

# The production solver settings. These are not tuning knobs: rtol=1e-7 and
# saves-only windowing are what the campaigns are validated at, and changing
# either changes the answer by far more than anything measured here.
const SOLVER = (; init_dz=5e-7, rtol=1e-7, max_dz=2e-6, twin_period=TWIN_SAVES_ONLY)

field_gib(Nω, N) = 16 * Nω * N * N / 2^30
devfree() = (s = Luna.device_memory_status(); isnothing(s) ? NaN : s[1]/2^30)
gib(x) = round(x; digits=2)

# The budget model lives in the package (`TS.memory_budget`), so the driver script, this
# benchmark and the test that pins it against real allocations cannot drift apart.
device_budget(c) = TS.memory_budget(setup_args_for(gap_mm=c.gap_mm, N=c.N,
                                                   trange=c.trange, thickness=c.thickness))

"Print the budget and say whether it fits, with the ways out if it does not."
function report_budget(name, c; margin=0.92)
    bu = device_budget(c)
    free = devfree()
    @printf("    grid (%d, %d, %d)", bu.Nω, c.N, c.N)
    FIELD && @printf("  fine Nto %d (%s)", bu.Nto,
                     bu.Nto == bu.Nt ? "not oversampled" : "$(bu.Nto ÷ bu.Nt)× oversampled")
    @printf("   one field %.2f GiB\n", bu.field)
    @printf("    budget: state %.1f + Et_win %.1f + Eto %.1f + Eωo %.1f + Pto %.1f",
            bu.state, bu.et_win, bu.eto, bu.ewo, bu.pto)
    @printf(" + analytic %.1f + window %.1f = %.1f GiB device", bu.analytic, bu.window, bu.device)
    @printf("   (host peak %.1f GiB)\n", bu.host)
    if FIELD && RESPONSE !== :thg
        @printf("    NB %.1f GiB of that (the analytic signal) is allocated on the FIRST\n",
                bu.analytic)
        println("       RHS, not at setup — a card can look fine after build_setup and")
        println("       still die on the first step.")
    end
    # An unreadable free-memory figure (no device) gives NaN, and NaN comparisons are
    # false, so the run proceeds rather than being blocked by a missing measurement.
    fits = !(bu.device > margin*free)
    if !fits
        @printf("    SKIPPING %s: needs ~%.1f GiB, only %.1f GiB free\n", name, bu.device, free)
        if FIELD && RESPONSE === :nothg
            println("      --response=thg  or  --ffac=4  each cut this to about two thirds")
            println("      (ffac=4 changes the grid — see the header before using it)")
        end
        println("      otherwise: a larger card, or a smaller N (but then the envelope")
        println("      reference has to be recomputed at the same N to compare)")
    end
    return (bu, fits)
end

function banner()
    d = CUDA.device()
    @printf("host %s   julia %s   threads %d\n", gethostname(), VERSION, Threads.nthreads())
    @printf("device %s   %.1f GiB   driver CUDA %s   runtime %s\n",
            CUDA.name(d), CUDA.total_memory()/2^30,
            CUDA.driver_version(), CUDA.runtime_version())
    if FIELD
        @printf("FIELD-RESOLVED mode: Grid.RealGrid, response :%s, ffac %d\n",
                RESPONSE, FFAC)
        println("  (the envelope path is the default and is unaffected by any of this)")
    end
end

csv_row(f, row) = isempty(OUT) || open(OUT, "a") do io; println(io, join(row, ",")); end

# ============================================================================ #
#  ACCURACY — GPU only
# ============================================================================ #
#
# No CPU reference is computed here. A CPU delay point at a production shape is
# ~50 min on 8 threads, this is rented GPU time, and the host-vs-device comparison
# is already covered where it belongs: ModelPNPS's hardware-gated CUDA testset
# (small grids, seconds) and, at production scale, `verify_against_collected`
# against a bit-identical CPU reference on the HPC.
#
# What is left is what only this machine can answer, and both parts are pure GPU:
#
#   1. Save-time extraction vs saving the stack, same backend, same grid. These
#      reduce the same slices by different routes; a difference is the extraction,
#      not the propagation. Must agree to rounding.
#   2. The solver's own convergence at this shape: production `rtol` against a
#      decade tighter. This is the number that actually bounds the campaign, and
#      it is NOT a backend question — it is worth measuring at any grid the
#      campaigns have not already validated, which is most of them.
#
# Pass --cpu=1 to add a host reference anyway (slow; prefer the HPC).
function run_accuracy()
    name = get(CASES, opt("case", "dd05"), CASES["dd05"])
    c = name
    println("\n=== accuracy (GPU only): $(opt("case", "dd05")) — $(c.note) ===")
    sa = setup_args_for(gap_mm=c.gap_mm, N=c.N, trange=c.trange, thickness=c.thickness)
    bu, fits = report_budget(opt("case", "dd05"), c)
    fits || return false
    @printf("    %d z-slices\n", length(c.zsave))
    setup = TS.build_setup(; sa..., arraytype=CUDA.CuArray)
    Nω = length(setup.grid.ω)
    τ = -8e-15   # off centre: a wing point exercises the weak-signal regime

    ok = true
    println("\n1. save-time extraction vs saving the stack (same GPU, same grid)")
    t1 = @elapsed o1 = TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                               extract_on_save=true)
    t2 = @elapsed o2 = TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                               extract_on_save=false)
    # NB `@printf` reads its format at MACRO-EXPANSION time, so it must be a single
    # string literal — a `"a" * "b"` concatenation is an expression and fails at include
    # time, taking the whole script with it. Split long messages into separate calls.
    @printf("   on-save %.1f s   save-the-stack %.1f s\n", t1, t2)
    @printf("   (the difference is the %d × %.2f GiB of temp-file traffic the on-save\n",
            length(c.zsave), field_gib(Nω, c.N))
    println("    route does not do)")
    for k in keys(o1)
        k === :zsave && continue
        r = maximum(abs.(o1[k] .- o2[k])) / maximum(abs, o2[k])
        @printf("   %-22s %10.3e\n", k, r)
        ok &= r < 1e-10
    end

    println("\n2. solver convergence at this shape: rtol $(SOLVER.rtol) vs 1e-8")
    t3 = @elapsed o3 = TS.simulate_delay_point(setup, τ; zsave=c.zsave,
                                               init_dz=SOLVER.init_dz, rtol=1e-8,
                                               max_dz=SOLVER.max_dz,
                                               twin_period=SOLVER.twin_period)
    @printf("   rtol 1e-8 took %.1f s (%.2f× the production run)\n", t3, t3/t1)
    for k in keys(o1)
        k === :zsave && continue
        r = maximum(abs.(o1[k] .- o3[k])) / maximum(abs, o3[k])
        rz = maximum(abs.(o1[k][:, end] .- o3[k][:, end])) / maximum(abs, o3[k][:, end])
        @printf("   %-22s %10.3e   (final z-slice %10.3e)\n", k, r, rz)
    end
    println("   The campaign's own pass criterion is every z-slice of Iω_win within")
    println("   1e-3 relative of an rtol=1e-8 run. A wing point is the hard case:")
    println("   `weaknorm` measures error against the pump-dominated whole field, so")
    println("   the weak signal's own relative error is ~rtol × ‖pump‖/‖signal‖.")

    if opt("cpu", "0") == "1"
        println("\n3. host reference (requested with --cpu=1; slow)")
        sh = TS.build_setup(; sa...)
        th = @elapsed oh = TS.simulate_delay_point(sh, τ; zsave=c.zsave, SOLVER...)
        @printf("   cpu %.1f s on %d threads (%.1f× the GPU)\n", th, Threads.nthreads(), th/t1)
        for k in keys(oh)
            k === :zsave && continue
            @printf("   %-22s %10.3e\n", k,
                    maximum(abs.(o1[k] .- oh[k])) / maximum(abs, oh[k]))
        end
    end

    @printf("\n   %s\n", ok ? "PASS (extraction routes agree to < 1e-10)" :
                              "CHECK — the two extraction routes disagree")
    return ok
end

# ============================================================================ #
#  BENCH — GPU delay points at production shapes
# ============================================================================ #
const SUMMARY = Tuple{String,Float64,Float64}[]
dev_ratio(a, b) = (isfinite(a) && isfinite(b) && b > 0) ? a/b : NaN

function run_bench()
    println("\n=== bench: GPU delay points at production shapes ===")
    isempty(OUT) || csv_row(OUT, ["case","N","Nomega","field_GiB","point","wall_s",
                                  "device_GiB","host_GiB","extract_on_save",
                                  "mode","ffac"])
    # Warm up FIRST. CUDA kernels compile per TYPE, not per size, so a tiny propagation
    # compiles everything the real cases need for a couple of seconds. Without this the
    # first case carries ~11 s of compilation and reads as 2-3x slower than it is —
    # which is exactly what happened on the first H200 run.
    print("    warm-up (compiles the device kernels): ")
    tw = @elapsed for fld in (RATIO ? (FIELD, false) : (FIELD,))
        wa = setup_args_for(gap_mm=1.0, N=32, trange=110e-15, thickness=40e-6, field=fld)
        ws = TS.build_setup(; wa..., arraytype=CUDA.CuArray)
        TS.simulate_delay_point(ws, 0.0; zsave=[0.0, 40e-6], SOLVER...)
    end
    @printf("%.1f s\n", tw)
    GC.gc(); Luna.device_reclaim()
    for name in CASE_NAMES
        haskey(CASES, name) || (println("unknown case $name, skipping"); continue)
        c = CASES[name]
        sa = setup_args_for(gap_mm=c.gap_mm, N=c.N, trange=c.trange, thickness=c.thickness)
        println("\n--- $name: $(c.note)")
        GC.gc(); Luna.device_reclaim()
        bu, fits = report_budget(name, c)
        fits || continue
        free0 = devfree()
        tsetup = @elapsed setup = TS.build_setup(; sa..., arraytype=CUDA.CuArray)
        Nω = length(setup.grid.ω)
        @printf("    setup %.1f s   device after setup %.1f GiB\n",
                tsetup, free0 - devfree())

        tf0 = NaN; dev0 = NaN     # the τ = 0 point, for the envelope ratio below
        for ip in 1:POINTS
            # A spread of delays: the far wing and near zero differ in step count,
            # and the campaign cost is the average over the scan, not the best case.
            τ = (ip == 1) ? 0.0 : (-25e-15 + 50e-15 * (ip - 1) / max(POINTS - 1, 1))
            GC.gc(); Luna.device_reclaim()
            f0 = devfree()
            t = @elapsed TS.simulate_delay_point(setup, τ; zsave=c.zsave, SOLVER...,
                                                 extract_on_save=(EXTRACT != "off"))
            dev = f0 - devfree()
            host = Sys.maxrss()/2^30
            # Cost per GiB of state is the saturation signal: a delay point's work is
            # proportional to the field size at fixed step count, so if the card is
            # saturated this is roughly CONSTANT across cases whose fields differ by
            # 2.6×. If it falls as the field grows, the smaller cases are not filling
            # the card and there is headroom to exploit (see the suite's `share` step).
            @printf("    point %d  τ %+7.2f fs   wall %7.1f s   device %5.1f GiB   host %5.1f GiB   %6.2f s/GiB\n",
                    ip, τ*1e15, t, dev, host, t/field_gib(Nω, c.N))
            @printf("      predicted %.1f GiB device, measured %.1f — %s\n", bu.device, dev,
                    abs(dev - bu.device) < 0.1*bu.device ? "as budgeted" :
                    "OFF BY MORE THAN 10 %, the cuFFT workspace is not negligible here")
            ip == 1 && (tf0 = t; dev0 = dev)
            push!(SUMMARY, (name, field_gib(Nω, c.N), t))
            csv_row(OUT, [name, c.N, Nω, gib(field_gib(Nω, c.N)), ip, round(t; digits=2),
                          gib(dev), gib(host), EXTRACT != "off",
                          FIELD ? String(RESPONSE) : "envelope", FIELD ? FFAC : 0])
        end
        setup = nothing; GC.gc(); Luna.device_reclaim()

        # The headline number for a field-mode bench is the RATIO to the envelope at the
        # same shape, and it has to be measured on this card rather than carried over: it
        # is bandwidth-bound, and the two paths move different amounts through different
        # buffers. Measured on a laptop it is 3.0-3.3× at matched step counts; if this
        # card gives something far from that, the step counts are the first thing to check
        # (they are printed by Luna's own propagation log).
        if RATIO && FIELD
            ea = setup_args_for(gap_mm=c.gap_mm, N=c.N, trange=c.trange,
                                thickness=c.thickness, field=false)
            GC.gc(); Luna.device_reclaim()
            f0 = devfree()
            esetup = TS.build_setup(; ea..., arraytype=CUDA.CuArray)
            # τ = 0 to match `tf0`/`dev0`: the step count varies with delay, so a ratio
            # taken between different delays measures the delays, not the two paths.
            te = @elapsed TS.simulate_delay_point(esetup, 0.0; zsave=c.zsave, SOLVER...,
                                                  extract_on_save=(EXTRACT != "off"))
            edev = f0 - devfree()
            @printf("    envelope companion at the same shape, τ = 0: %.1f s, %.1f GiB\n",
                    te, edev)
            @printf("    FIELD/ENVELOPE at τ = 0:  time %.2f×   device %.2f×\n",
                    tf0/te, dev_ratio(dev0, edev))
            @printf("    a 200-delay scan would be %.1f h field vs %.1f h envelope\n",
                    200tf0/3600, 200te/3600)
            println("    (a laptop gives 3.0-3.3× on time at matched step counts; if this")
            println("     card differs a lot, compare the step counts in Luna's own log)")
            esetup = nothing; GC.gc(); Luna.device_reclaim()
        end
    end
    println("\n  Device memory should be ≈ 10 × the field size (9 solver registers + 1")
    println("  transform buffer); more means the cuFFT workspace is not negligible at")
    println("  that shape, which changes how many points fit on the card at once.")
    if length(SUMMARY) > 1
        println("\n  saturation check — cost per GiB of state across cases:")
        println("    case       field GiB    wall s    s/GiB     scan (200 pts)")
        for (nm, f, t) in SUMMARY
            @printf("    %-10s %8.2f %9.1f %8.2f     %5.1f min\n", nm, f, t, t/f, 200t/60)
        end
        v = [t/f for (_, f, t) in SUMMARY]
        spread = maximum(v) / minimum(v)
        @printf("    s/GiB spread %.2f× — %s\n", spread,
                spread < 1.3 ? "flat: the card is saturated at every shape, so cost is pure\n" *
                               "                       traffic and running points concurrently will not help" :
                               "NOT flat: the smaller shapes are leaving the card idle;\n" *
                               "                       there is headroom, run the suite's `share` step")
    end
end

# ---------------------------------------------------------------------- main --
banner()
println("mode=$MODE  points=$POINTS  extract=$EXTRACT  started $(Dates.now())")
if MODE == "accuracy"
    run_accuracy() || exit(1)
elseif MODE == "bench"
    run_bench()
else
    error("unknown --mode=$MODE (accuracy|bench)")
end
println("\ndone $(Dates.now())")
