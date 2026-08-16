# ModelPNPS GPU — deferred work for the 3-D TG-FROG campaigns

Branch: `gpu`. The port is **validated and in production condition** (see
`Luna/GPU-TODO.md` §7 for the shared half of this list and for the Luna-side changes each
item depends on).

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

---

## 1. P0 — Extract on device at save time; no temp file at all

**Effort M** (blocked on `Luna/GPU-TODO.md` §7 "let an output handler consume the device
state"). The single highest-value item.

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

The 14.2 GiB host peak is set by **setup**, not by the run; the run itself needs only the
save buffer + one extraction slice + the window (≈5.6 GiB, and §1 removes most of that).
Fixing setup is what makes a 16 GB host node (e.g. H200 partitions) viable, and lets more
instances share a node.

- [ ] **P1 (Effort S) — Do not materialise a fourth beamlet field.**
  `src/ModelPNPS.jl:1004` computes `Eωk_g12 = Eωk_g1 .+ Eωk_g2` while `Eωk_g1`, `Eωk_g2`
  and `Eωk_t_base` are all still live: **four full host fields = 9 GiB simultaneously** at
  production. Accumulating in place (`Eωk_g1 .+= Eωk_g2`) drops the peak to three fields
  (6.75 GiB) for a one-line change. Check no caller needs `Eωk_g1` afterwards.

- [ ] **P1 (Effort S–M) — Upload each beamlet as it is built.** Better than the above:
  have `build_beamlets` take the `arraytype` and hand back device arrays, uploading and
  freeing each host beamlet in turn. Peak host contribution → one field (2.25 GiB). The
  builders themselves (Bessel profiles, masks, FFTs) stay host code.

- [ ] **P2 (Effort M) — Build beamlets and the window natively on device.** Removes the
  host field entirely rather than staging it. Needs a device Bessel evaluation and a
  device transverse FFT; worth it only if §1 and the two items above leave host memory as
  the binding constraint. Note `Luna/GPU-TODO.md` §7 removes a further 4.5 GiB from
  `Luna.setup` independently.

- [ ] **P3 (Effort S) — Revisit the `beamlets_on_host` default** once the above land. It
  currently trades two resident device fields for a per-point upload; if host memory is no
  longer the constraint, the device-resident default is unambiguously right.

---

## 3. P2 — Per-point overhead outside extraction

- [ ] **P2 (Effort S) — Reuse the delayed-input buffer across delay points.**
  `delayed_input` allocates a fresh 2.25 GiB device field per point, which the solver then
  adopts (`preserve_input=false`) and frees. That is an allocate/free cycle per point
  against a pool that then needs `device_reclaim()`. Writing into a setup-owned buffer
  removes the churn. Small, but it is pure overhead in a 4.6-minute point.

- [ ] **P2 (Effort S) — Measure the per-point breakdown** before doing more here: split
  wall time into `delayed_input`, `Luna.run`, and extraction, and within extraction split
  I/O from compute. The 38–50 s figure is a subtraction, not a measurement, and §1's
  payoff estimate rests on the assumption that most of it is the read-back.

---

## 4. P1 — Hardware: is an A100 worth renting?

**No code.** Potentially the largest single speedup available, and it hinges on one
measurement listed as P0 in `Luna/GPU-TODO.md` §7.

The A40 (GA102) runs FP64 at 1/64 rate — ~0.58 TFLOPS — against the A100's 1/2 rate at
~9.7 TFLOPS: **16.7×**. Memory bandwidth differs by only 2.2× (0.70 vs 1.55 TB/s). So:

- if the step is **FFT/FP64-bound**, an A100 is worth far more than its bandwidth ratio,
  plausibly 3–5×, and dwarfs every kernel item in both TODO files combined;
- if it is **bandwidth-bound**, expect ~2.2× and the kernel work matters relatively more.

Until the §7 P0 breakdown exists, this is genuinely undetermined — do not assume either.
An 80 GB A100 would also fit two delay points per card (2 × 22.5 = 45 GiB), though on a
bandwidth- or FFT-bound kernel concurrency buys throughput only if a single point leaves
the card idle, which the breakdown would also show.

---

## 5. Ranking for the 3-D TG-FROG campaigns

| # | Item | Effort | Expected gain | Basis |
|---|---|---|---|---|
| 1 | §1 on-device extraction, no temp file | M | **14–18 % wall, +72 GiB/point I/O removed, ~4.5 GiB host** | overhead measured; split estimated |
| 2 | §4 A100 evaluation | none (rental) | **2.2× to ~5×** | FP64/bandwidth specs; needs the §7 P0 breakdown |
| 3 | §2 host memory (both S items) | S+S | 9 GiB → 2.25 GiB of the host peak | arithmetic from field sizes |
| 4 | Luna §7 P1 skip host FFT prototype | S | further 4.5 GiB host | arithmetic |
| 5 | Luna §7 P2 ScaledPlan + towin fusion | S–M | ~10–15 % of step, *estimated* | traffic model only |
| 6 | §3 buffer reuse / overhead measurement | S | small | — |

**If only one thing gets done: §1.** It is the only item whose payoff is anchored in a
measured number rather than a model, it removes an entire failure mode (temp-file I/O on a
shared filesystem under a many-instance campaign) rather than just some time, and it is
the reason the host-memory items become nearly free afterwards.

**Do not start the kernel work (item 5) before the `Luna/GPU-TODO.md` §7 P0 breakdown.**
Those percentages come from a memory-traffic model that has not been checked against the
card, and the same measurement decides item 2, which is worth more than item 5 either way.

---

## 6. Considered and rejected

- **Reducing the resident field count to fit two delay points on one A40** — two points
  need 45 GiB against 44.4 available, and on a bandwidth- or FFT-bound kernel running two
  concurrently would only share the same bottleneck. Revisit only on an 80 GB card, and
  only if the breakdown shows one point leaves the card idle.
- **FP32 anywhere in the propagation** — invalidates the `rtol=1e-7` accuracy campaign.
  (FP32 for the *extraction* reductions alone would be safe but saves nothing once §1
  removes the transfer.)
- **`Output.MemoryOutput` instead of streaming** — avoids the temp file but holds all 16
  slices in host RAM (36 GiB). §1 is the right fix.
- **Batching several delay points as an extra array dimension** — multiplies device memory
  by the batch size for no arithmetic gain; the points are already independent and
  parallelised across jobs.
