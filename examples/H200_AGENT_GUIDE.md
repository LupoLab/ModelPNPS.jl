# H200 pod: triage guide for the ModelPNPS GPU suite

You are on a **rented GPU pod, billed by the hour**. Your job is to get
`examples/h200_modelpnps_suite.sh` to a clean, complete set of numbers with as little
wall-clock as possible, and to bring the numbers home. Optimise for *fewest GPU-minutes
to a decisive answer*, not for elegance.

Read this whole file before running anything.

- Code: `/workspace/code/ModelPNPS.jl` (branch `gpu`), `/workspace/code/Luna.jl` (branch
  `modal-fixed`). Project: `/workspace/code/dev`. Results: `/workspace/runs/<stamp>/`.
- The suite: `examples/h200_modelpnps_suite.sh`; drivers `examples/h200_bench.jl` and
  `examples/h200_scan_rehearsal.jl`.
- Luna's **modal** suite (`Luna.jl/test/manual/h200_gpu_suite.sh`) may be running at the
  same time. It is a different transform and a different person's problem — but both are
  expected to be memory-bandwidth-bound, so **timings taken while it runs are not clean**.

---

## 1. What this is measuring

3-D free-space TG-FROG delay points: state is one `(Nω, N, N)` `ComplexF64` array; the
solver holds 9 of them plus 1 transform buffer. Four production shapes:

| case | grid | field GiB | expect device | expect wall/point |
|---|---|---|---|---|
| `dd05` | 256×640² | 1.56 | ~15.6 GiB | ~7 s |
| `04` | 256×768² | 2.25 | ~22.5 GiB | ~10 s |
| `dd20` | 256×1024² | 4.00 | ~40 GiB | ~18 s |
| `100um` | 512×640² | 3.13 | ~31 GiB | ~28 s |

**The wall times are estimates from a traffic model that has never been validated against
a card — a 2× miss in either direction is not a bug.** The device memory figures are
*not* estimates: an A40 measured exactly 10.00 fields at the `04` shape, so a large
deviation there is a real finding (cuFFT workspace) and worth reporting.

Reference point: the same `04` shape took **276 s/point and exactly 22.5 GiB on an A40**.
The A40 is FP64-bound (it ran ~6× slower than its bandwidth allows); the H200 has 58× its
FP64 and 6.9× its bandwidth, so the expected regime here is bandwidth-bound.

---

## 2. Run order, and how to iterate cheaply

Default `STEPS=pkgs,tests,accuracy,bench,scan`. **Do not re-run the whole suite to test a
fix.** Re-run one step:

```bash
STEPS=tests    bash examples/h200_modelpnps_suite.sh      # ~2 min
STEPS=accuracy bash examples/h200_modelpnps_suite.sh      # ~3-8 min
STEPS=bench CASES=04 POINTS=1 bash examples/h200_modelpnps_suite.sh   # ~2 min
```

Or drive the Julia directly, which skips the suite's logging setup:

```bash
julia --project=/workspace/code/dev examples/h200_bench.jl --mode=bench --cases=04 --points=1
```

**Start with `CASES=04 POINTS=1`.** It is the shape with an A40 reference, so it is the
only one where you can tell "slow" from "as expected". Only widen once it looks right.

If you are debugging a crash rather than a number, reproduce it at a **small grid first**
— `--mode=accuracy --case=dd05` still uses a production grid, so for pure debugging drop
into the REPL and build a setup at `N=64`. That costs seconds, not minutes.

---

## 3. Triage

### Suite won't start / packages

`ERROR: ArgumentError: Package X not found` — the scripts are run as scripts, so every
package they `import` must be a **direct dependency of `/workspace/code/dev`**, not merely
a dependency of Luna. Fix: `STEPS=pkgs bash examples/h200_modelpnps_suite.sh`, or
`julia --project=/workspace/code/dev -e 'using Pkg; Pkg.add("X")'`.

`CUDA.jl` precompile failure or "precompiled for CUDA 13.x but driver is 13.y" — do **not**
pin a runtime version. `julia --project=/workspace/code/dev -e 'using CUDA;
CUDA.reset_runtime_version!()'` then re-precompile. See the note in `runpodcoldstart.sh`.

### `tests` fails

Run the failing testset alone. The ModelPNPS suite is `test/runtests.jl`; the CUDA part is
gated on `LUNA_TEST_CUDA=1`.

- **`Scalar indexing is disallowed`** — a real device bug, never a test artefact. The
  stack trace names the function. Most likely culprits, in order:
  `_reduce_slice!` (`src/ModelPNPS.jl:1422`) routing a host array into a device kernel or
  vice versa; `_extract_slice_device!` (`:1331`); a host vector reaching a device
  broadcast. **Check operand residency first** — `Luna.Utils.isdevice(x)` on each argument.
  This exact class of bug has bitten three times; it is almost always a host/device mix,
  not a kernel error.
- **A device-vs-host comparison fails by a huge factor** (not 1e-8, but 1e+6 or 1e-6× off)
  — suspect a normalisation, not a physics error. `_ift_unscaled`
  (`Luna/src/NonlinearRHS.jl:1309`) splits the inverse FFT plan into raw transform +
  `1/N`; if the scale went missing the answer is off by `N` per transform. The test
  `"unscaled inverse plan and the fused pointwise RHS"` in `Luna/test/test_device.jl`
  isolates this.

### `accuracy` — the two extraction routes disagree

Bar is **< 1e-10**; they reduce the same slices by different routes, so anything larger is
a bug in the extraction, not the propagation. Look at `_reduce_slice!`
(`src/ModelPNPS.jl:1422`) and the signed-window trick in `_signed_window` (`:1302`) — the
re-imaging sign is folded into the window so one array serves both reductions, and
`|±w·E|² == |w·E|²` is what makes that valid. If only `Iω_win_reimaged` disagrees, the
sign fold is the suspect.

### `accuracy` — the rtol comparison looks large

**This is a physics result, not a failure.** These runs use `weaknorm`, which measures
error against the pump-dominated whole field, so the weak FWM signal's own relative error
is `~rtol × ‖pump‖/‖signal‖`. A wing point is the hard case by design. Record the number;
the campaign criterion is every z-slice of `Iω_win` within 1e-3 of an `rtol=1e-8` run, and
it has **never been measured at the dd05/dd20/100um shapes**. Do not "fix" it.

### `bench` — device memory is much more than 10× the field

That is the one genuinely new finding available here. It means the cuFFT plan workspace is
not negligible at that shape (it was on an A40 at `04`), which changes how many points fit
on a card. Confirm by printing `Luna.device_memory_status()` around `TS.build_setup` alone
versus after one point. Report the number; do not work around it.

### `bench` — per-point wall is much slower than the table

Check in this order, cheapest first:

1. **Is the modal suite running?** `nvidia-smi`. If so, your numbers are contended. Stop
   and re-time, or note it.
2. **Is the card power- or clock-capped?** `nvidia-smi --query-gpu=power.limit,
   power.max_limit,clocks.sm,clocks.max.sm --format=csv`. Rented cards are sometimes
   capped well below nominal. The cold-start script prints a copy-bandwidth number —
   compare against ~4800 GB/s peak for an H200.
3. **Step count.** The Luna log line `Propagation finished in …, N steps` — `04` at 40 µm
   took 73–77 steps on the A40. Far more steps means the solver is struggling, which is a
   different problem from the card being slow.
4. Only then suspect the code. `CUDA.@profile` on a single RHS evaluation will show
   whether the FFTs or the elementwise passes dominate — **this is the §7.4 "measure
   first" item from the design doc and is the single most valuable thing you can bring
   home**, because it decides whether an A100/H200 is worth its price over an A40.

### `bench` — the `s/GiB` column is not flat

Flat across cases (spread < 1.3×) means the card is saturated at every shape, so cost is
pure traffic and running points concurrently cannot help. Falling with field size means
the small shapes are leaving the card idle — then, and only then, `STEPS=share` is worth
the minutes.

### Out of memory

`dd20` needs ~40 GiB and the modal suite may hold its own. `Luna.device_reclaim()` between
points already runs; if memory climbs across points anyway, that is a leak and is worth
reporting. For `share`, the requirement is `NPROC × 10 × field`.

Host RAM: peak is set by `build_setup` (four beamlet fields plus the window), ~18 GiB at
`N=1024`. If the pod has less than ~32 GB this is the binding constraint, not the GPU.

### World-age `MethodError … too new`

Should not happen: these scripts `import CUDA` at the top level and pass `CUDA.CuArray`
directly. If you see it, something is passing `arraytype=:cuda` (the *symbol*) instead —
that form exists for cluster login nodes that cannot load CUDA.jl and requires
`Base.invokelatest`. Grep for `:cuda` in whatever you changed.

---

## 4. What to bring home

Copy out of `/workspace/runs/<stamp>/`: `suite.log`, `bench.csv`, and the scan's
`*_collected.h5`. Then, in a short summary:

1. Wall per point for each case, and `s/GiB` (flat or not).
2. Device GiB per case as a multiple of the field size.
3. Step counts from the Luna log lines.
4. The `rtol=1e-7` vs `1e-8` differences per dataset — new physics, not previously measured.
5. Anything from `CUDA.@profile` about the FFT-vs-elementwise split.
6. `nvidia-smi` power/clock limits, and whether the modal suite was running.

The collected HDF5 is the accuracy artefact: it uses production parameters so a CPU run on
the HPC can be compared against it with `verify_against_collected` (read the **scan-peak**
normalised column, not the own-peak one — see `examples/scan_peaks.jl`).

---

## 5. Do not

- **Do not run CPU delay points.** ~50 min each at production shapes. CPU work belongs on
  the HPC. (`--cpu=1` exists on the accuracy step; do not use it.)
- **Do not re-run the whole suite to test one fix.** Use `STEPS=`.
- **Do not tune `rtol`, `max_dz` or `twin_period`.** They are what the campaigns are
  validated at; changing them changes the answer by more than anything being measured.
- **Do not "fix" a large rtol difference** — see above, it is a measurement.
- **Do not commit speculative changes to `gpu` or `modal-fixed`.** Work on a branch and
  report; the shared branches are being merged elsewhere.
- **Do not leave the pod idle.** If you are blocked on a decision, say so immediately
  rather than exploring — and if you are waiting on a long run, say what you are waiting
  for rather than starting another GPU job alongside it.
