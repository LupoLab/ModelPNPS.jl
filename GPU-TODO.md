# ModelPNPS GPU — deferred work for the 3-D TG-FROG campaigns

Branch: `gpu`. The port is **validated and in production condition**.

The Luna half of this list moved: `Luna/GPU-TODO.md` is now a pointer, and the design,
measurements and open items live in `Luna/docs/src/developer/modal_fixed_design.md` — §8
"Roadmap and open items" for the list, §3.7 for device execution and §3.9 for the general
(`RealGrid`) `TransFree` path. References below that used to read "`Luna/GPU-TODO.md` §7"
point there instead.

Legend — **Effort**: S ≲ ½ day / < 50 lines, M ≈ 1–2 days / 50–300 lines, L ≳ 3 days.
Priority: P0 = do first, P1 = next, P2 = worthwhile, P3 = when needed.

---

## Where the time and memory actually go

Measured on one A40 against the 04 Kerr campaign (`Nω=256`, `N=768`, `rtol=1e-7`,
16 z-saves; one field = **2.25 GiB**), jobs `verify04gpu-442443` / `verify04cpu-442326`,
`-442518`:

| | GPU (1×A40) | CPU (8 threads) |
|---|---|---|
| wall / delay point | 265–281 s | 2873–3150 s |
| of which propagation | 227–230 s | 2779–3048 s |
| **non-propagation overhead** | **38–50 s (14–18 %)** | 94–102 s (3.2 %) |
| device resident | 22.5 GiB (exactly 10.0 fields) | — |
| host peak RSS | 14.2 GiB | 32.8–33.3 GiB |

Two things follow, and they set the whole priority order:

**1. Per-point overhead went from negligible to material.** The same ~40–100 s of
setup/extraction that was 3 % of a CPU point is 14–18 % of a GPU point. Amdahl: the port
made the propagation 12× faster and left this untouched.

**2. Every delay point moves 72 GiB through a temp file to produce 32 KB of results.**
`simulate_delay_point` streams to `Output.HDF5Output` (`src/ModelPNPS.jl:1655`), so all
**16 z-slices × 2.25 GiB = 36 GiB are written** during the run and **read back** during
extraction — and `extract_trace_data` reduces each slice to three `(Nω,)` vectors. At
production that is 32 KB of output per point from 72 GiB of I/O. Reading it back accounts
for most of the measured 38–50 s overhead (36 GiB / ~45 s ≈ 800 MB/s, a plausible rate);
the write half is inside the propagation figure and is not separately measured.

Scale matters here too: at 20 concurrent delay points that is ~1.4 TB of temp-file traffic
per round on the shared filesystem, and the CPU campaign at 100 concurrent points is ~7 TB.

The device side, by contrast, is already tight: 10.0 fields is exactly 9 RK45 registers +
1 transform buffer, with negligible cuFFT workspace. **There is little device memory to
win and most of the remaining headroom is host-side and I/O.**

### H200, 2026-08-26 — and field mode

Measured on one H200 (139.8 GiB, CUDA 13.3) at the same `04` shape, after §1 landed, so
these are the on-save extraction route with no temp file at all:

| | envelope | field, `:nothg`, ffac 6 |
|---|---|---|
| grid | `Nω = 256` | `Nω = 513`, fine `Nto = 2048` |
| wall / delay point, τ = 0 | 12.1 s | 67.0 s |
| propagation | 10.3 s, 57 steps | 61.1 s, **the same 57 steps** |
| device, budget | 25.9 GiB | 96.9 GiB |
| host peak in `build_setup` | 12.4 GiB | 24.8 GiB |

**Field mode costs ~6× the envelope per step**, not the ~3× a laptop suggests — matched
step counts, so it is a clean per-step ratio; on a laptop the fixed extraction and input
costs dilute it. The extra is the no-THG response's batched analytic signal: an 18 GiB
complex buffer and two 1-D FFT passes over it per RHS.

**The `10 × field` rule of thumb does not hold in field mode** and should not be used
there. `ModelPNPS.memory_budget(setup_args)` computes the resident total per buffer
instead; it agrees with the card to **0.1 %** (96.9 predicted, 96.9 measured, decomposing
as state 40.6 + analytic 18.0 + input 4.5 + window 2.25 during the point, and
Eto 9.0 + Eωo 9.0 + Et_win 4.5 + Pto 9.0 held by setup). A test pins it against real
allocations. Note 18.0 GiB of it — the analytic signal — is allocated on the **first RHS**,
not at setup, so a card with room after `build_setup` can still die on the first step.

Accuracy at that shape, also on the H200: the two extraction routes agree to 6e-15, and
`rtol` 1e-7 against 1e-8 differs by 2.9e-7 on `Iω_win` at a wing point — 3400× inside the
campaign's 1e-3 criterion, so the field grid is solver-converged at the production
tolerance.

---

## 1. ~~P0 — Extract on device at save time; no temp file at all~~ — **DONE**

Landed on this branch: `f8f720c` (reduce each z-slice as it is saved) and `02fd0bf` (use
Luna's `needs_host_save`/`foreach_save`). `TraceExtractOutput` reduces each slice on the
device at the moment it is saved; `simulate_delay_point` defaults `extract_on_save` ON for
a device propagation and OFF on the host, where the streamed route stays bit-validated
against the production references. The rest of this section is kept as the record of why.

**Effort M** (was blocked on the Luna-side output-handler change, since done). The single
highest-value item.

Instead of saving 16 full fields and reducing them afterwards, reduce each slice **on the
device, at the moment it is saved**, and keep only the results. The reductions are already
in exactly the right shape:

- `_quadrant_spectrum!` (`src/ModelPNPS.jl:1255`) — `sum(abs2, E)` over the signal
  quadrant per ω. Same pattern as `SignalQuadrantNorm`'s device path (`_sqn_fused`), which
  already exists and is tested.
- `extract_signal_spectra` (`src/ModelPNPS.jl:1180`) — one pass producing two per-ω
  reductions over `(ky, kx)`: `Σ|w·E|²` and the signed sum `Σ(−1)^(iky+ikx)·w·E`. The perf
  work already replaced the 2-D inverse FFT with that centre-pixel identity, so **no
  transform is needed** — these are two `sum(…; dims=(2,3))` calls over a broadcast.

Since ω is dimension 1 and contiguous, reducing over dims 2–3 coalesces well.

Requires the signal window on device (`window_array` is `(Nω, Nky, Nkx)` `Float64` =
1.125 GiB — comfortable in the 21.9 GiB of A40 headroom). Keep the host methods untouched
so CPU results stay bit-identical, and A/B the device path against them per z-slice.

**Payoff:** removes the temp file entirely — 72 GiB/point of I/O, most of the measured
14–18 % overhead, the 2.25 GiB `HostOutput` save buffer and the 2.25 GiB extraction slice.
Host memory during the run drops to essentially nothing, and the filesystem load of a
whole campaign disappears. It also removes `tempname()` from the hot path, which is a
robustness win on a shared cluster.

---

## 2. P1 — Host memory during `build_setup`

The 14.2 GiB host peak is set by **setup**, not by the run; §1 removed the run's share
almost entirely. Fixing setup is what makes a small host node viable and lets more
instances share one.

**Field mode doubles this**: `build_setup` peaks at **24.8 GiB** at `N = 768` because every
field is `Nω = 513` rather than 256 (`memory_budget(...).host`). The rented H200 pod used
for the field campaign had 251 GB, so this was not binding there — but it is the number to
check on any host with less than ~32 GB, and it is what the cold-start script now warns
about below that.

- [ ] **P1 (Effort S) — Do not materialise a fourth beamlet field.**
  `build_setup` computes `Eωk_g12 = isnothing(Eωk_g2) ? Eωk_g1 : Eωk_g1 .+ Eωk_g2` while
  `Eωk_g1`, `Eωk_g2` and `Eωk_t_base` are all still live: **four full host fields = 9 GiB
  simultaneously** at production, 18 GiB in field mode. Accumulating in place
  (`Eωk_g1 .+= Eωk_g2`) drops the peak to three fields for a one-line change.
  **Two callers now read `Eωk_g1`, and the ordering is what makes this safe:** the focal
  profile (`_beamlet_profile`) reads it inside `build_beamlets`, i.e. before the
  accumulation, and the `:sd` geometry returns `Eωk_g2 === nothing`. Keep the profile
  before the accumulation and handle the `nothing` branch.

- [ ] **P1 (Effort S–M) — Upload each beamlet as it is built.** Better than the above:
  have `build_beamlets` take the `arraytype` and hand back device arrays, uploading and
  freeing each host beamlet in turn. Peak host contribution → one field (2.25 GiB). The
  builders themselves (Bessel profiles, masks, FFTs) stay host code.

- [ ] **P2 (Effort M) — Build beamlets and the window natively on device.** Removes the
  host field entirely rather than staging it. Needs a device Bessel evaluation and a
  device transverse FFT; worth it only if the two items above leave host memory as the
  binding constraint. Note Luna's `setup` already skips the host prototype and plan
  entirely on a device run with no input fields (which is how ModelPNPS calls it), so the
  4.5 GiB this used to reference is already gone.

- [ ] **P3 (Effort S) — Revisit the `beamlets_on_host` default** once the above land. It
  currently trades two resident device fields for a per-point upload; if host memory is no
  longer the constraint, the device-resident default is unambiguously right.

---

## 3. P2 — Per-point overhead outside extraction

- [ ] **P2 (Effort S) — Reuse the delayed-input buffer across delay points.**
  `delayed_input` allocates a fresh device field per point — 2.25 GiB envelope, 4.51 GiB
  field mode — which the solver then adopts (`preserve_input=false`) and frees. That is an
  allocate/free cycle per point against a pool that then needs `device_reclaim()`. Writing
  into a setup-owned buffer removes the churn. It is the `input` term in `memory_budget`,
  so its size is now explicit rather than inferred.

- [x] **P2 (Effort S) — Measure the per-point breakdown.** Partly answered by §1 landing:
  on the H200 the on-save route costs 103.8 s against 148.1 s for save-the-stack at the
  same shape and delay, i.e. the stack route's extra 44 s per point IS the read-back, as
  the estimate assumed.

- [ ] **P2 (Effort S) — Decompose the ENVELOPE device figure on the H200.** The field-mode
  budget agrees with the card to 0.1 %, but the envelope point measured 34.9 GiB during the
  point against a modelled 23.6 (state 20.25 + input 2.25 + window 1.125) — ~11 GiB, five
  fields, unaccounted. The A40 measurement found the cuFFT workspace negligible for the
  same shape, so either the workspace differs on CUDA 13.3 / Hopper, or the pool is holding
  something the reclaim did not return. It does not threaten anything (the card has room),
  but it means the envelope side of `memory_budget` is unvalidated on hardware while the
  field side is.

---

## 4. P1 — Hardware: is an A100 worth renting?

**No code.** Written when the A40 was the only card available; **largely overtaken** — the
campaign now runs on rented H100/H200s, where the question is answered by having run there.
Kept because the FP64-versus-bandwidth reasoning is what decides any future card.

The A40 (GA102) runs FP64 at 1/64 rate — ~0.58 TFLOPS — against the A100's 1/2 rate at
~9.7 TFLOPS: **16.7×**. Memory bandwidth differs by only 2.2× (0.70 vs 1.55 TB/s). So:

- if the step is **FFT/FP64-bound**, an A100 is worth far more than its bandwidth ratio,
  plausibly 3–5×, and dwarfs every kernel item in both TODO files combined;
- if it is **bandwidth-bound**, expect ~2.2× and the kernel work matters relatively more.

The H200 evidence now points at **bandwidth**, not FP64 throughput: field mode moves far
more data per step than the envelope (an extra 18 GiB analytic buffer and two passes over
it) and costs ~6× the time at identical step counts, which is what a traffic-bound kernel
looks like. So expect a new card to buy roughly its bandwidth ratio, and treat FP64 specs
as the weaker predictor.

An 80 GB card fits two envelope delay points (2 × 25.9 = 52 GiB) but only one field-mode
point (96.9 GiB, H200-only) — and the H200 rehearsal found no gain from sharing a card
between processes anyway, since one point already saturates it.

---

## 5. Field mode (`Grid.RealGrid`) — what is deferred there

Field mode landed after most of this document was written. It propagates the real,
carrier-resolved field instead of the envelope, to settle whether the 1 fs retrieval
residual is the retrieval model or the envelope approximation. The Luna-side design is in
`modal_fixed_design.md` §3.9; what is deferred on this side:

- [ ] **P2 (Effort M) — Avoid the analytic-signal buffer, or shrink it.** The `:nothg`
  response allocates a complex `(nto, ny, nx)` array — **18 GiB at `N = 768`**, the single
  largest term in the field-mode budget and what makes that arm H200-only. Two routes were
  considered during the port and neither was taken: computing the analytic signal from
  `Eωo` (already the half-spectrum) still needs a complex array of the same size; the
  `|E_a|² = E² + H[E]²` formulation needs one REAL buffer plus a complex spectrum, and the
  spectrum could alias `Eωo`, which the inverse transform has already consumed. That would
  trade 18 GiB for 9 and one extra transform. Worth it only if the arm has to run on an
  80 GB card.

- [ ] **P3 (Effort S) — Consider `ffac = 4` for `:nothg` production.** Validated (Luna's
  "ffac convergence for the no-THG response" measures 4e-8, unchanged from z=0 to z=end)
  and it halves both the fine grid and the memory: 96.9 → 69.9 GiB. NOT used for anything
  compared against the delivered envelope files, because it changes δω and the realised
  time window. Fine for a self-contained scan whose companions also use it.

- [ ] **P3 — Raman in field mode is unimplemented** and errors out. The response exists
  (`Nonlinear.RamanPolarFieldBatched`) but the nuclear fraction `f_R` here is the envelope
  convention — the 3/2 reconciling `Kerr_env`'s internal 3/4 with the Raman kernel's 1/2 —
  and that factor does not carry to a carrier-resolved field unexamined. Needs its own
  quasi-static consistency test, mirroring the envelope one in the suite.

- [ ] **P2 — There is no field-versus-envelope comparison tool in the tree.**
  `verify_against_collected` cannot bridge them (`Nω` 513 against 256; it refuses on the
  size mismatch). The comparison is between physical spectral densities `|E|²/Δω²` splined
  onto a common band, reported as rms difference relative to trace peak at matched (τ, z),
  **per depth** — the growth with depth is the quantity of interest. The working recipe is
  the `"field mode: field-versus-envelope trace agreement"` testset; it was deliberately
  left out of the shipped code.

---

## 6. Ranking for the 3-D TG-FROG campaigns

| # | Item | Effort | Expected gain | Basis |
|---|---|---|---|---|
| — | ~~§1 on-device extraction~~ | M | **DONE**: 44 s/point and 72 GiB/point of I/O removed | measured on the H200 |
| 1 | §2 host memory (both S items) | S+S | 9 → 2.25 GiB envelope, 18 → 4.5 GiB field, of the setup peak | arithmetic from field sizes |
| 2 | §3 decompose the envelope device figure | S | none directly — it validates the budget the drivers refuse shapes on | 11 GiB unexplained on the H200 |
| 3 | Luna: fuse the ScaledPlan + towin passes on the general path | S–M | ~10–15 % of a field-mode step, *estimated* | traffic model only; the fast path already has it |
| 4 | §3 delayed-input buffer reuse | S | small; removes per-point pool churn | — |
| 5 | §4 card choice | none (code) | ~bandwidth ratio | H200 evidence, above |

**If only one thing gets done: §2's two S items.** With §1 landed, the setup peak is what
is left, it is the constraint that decides how many instances share a host, and field mode
doubled it.

The kernel work (item 3) is still estimated from a traffic model rather than measured. The
one place it would show up is the field-mode general path, whose extra passes are now the
dominant cost — so measure that path's stage breakdown before committing to it.

---

## 7. Considered and rejected

- **Reducing the resident field count to fit two delay points on one A40** — two points
  need 45 GiB against 44.4 available, and on a bandwidth- or FFT-bound kernel running two
  concurrently would only share the same bottleneck. Revisit only on an 80 GB card, and
  only if the breakdown shows one point leaves the card idle.
- **FP32 anywhere in the propagation** — invalidates the `rtol=1e-7` accuracy campaign.
  (FP32 for the *extraction* reductions alone would be safe but saves nothing once §1
  removes the transfer.)
- **`Output.MemoryOutput` instead of streaming** — avoids the temp file but holds all 16
  slices in host RAM: 36 GiB envelope, **72 GiB in field mode**. §1 is the right fix, and
  is now the default on a device. The one place the stack is still materialised is
  `h200_bench.jl --mode=accuracy`, which compares the two extraction routes against each
  other; it reads the cgroup limit and refuses, naming an `--accz` that fits, rather than
  invoking the OOM killer.
- **Batching several delay points as an extra array dimension** — multiplies device memory
  by the batch size for no arithmetic gain; the points are already independent and
  parallelised across jobs.
