#!/usr/bin/env bash
# =============================================================================
# ModelPNPS TG-FROG GPU suite on a Runpod H100/H200 pod (paid time).
#
# The 3-D free-space counterpart of Luna's test/manual/h200_gpu_suite.sh, which
# covers the MODAL transform. The two are independent and can run at the same
# time on one card — see CONCURRENCY below before you do.
#
# Follows on from test/manual/runpodcoldstart.sh, which has already put the Luna and
# ModelPNPS checkouts, the dev project (/workspace/code/dev) and /workspace/env.sh on
# the network volume. Everything lands in /workspace/runs/<timestamp>/.
#
# STEPS (in order; each is useful on its own and the order is deliberate — do not
# benchmark a backend you have not first shown to be correct):
#   pkgs      guard-add what the scripts import directly (no-op after the first time)
#   tests     ModelPNPS test suite incl. its hardware-gated CUDA testset   ~2 min
#   accuracy  GPU-only: the two extraction routes against each other, and
#             production rtol against a decade tighter                     ~3-8 min
#   bench     GPU delay points at the production shapes                    ~5-20 min
#   scan      a real short delay scan through run_scan, one process        ~5-15 min
#   share     OPT-IN saturation diagnostic: the same points split across processes
#             sharing the card. NOT in the default steps — see below.       ~5-10 min
#
# USAGE (on the pod):
#   bash /workspace/code/ModelPNPS.jl/examples/h200_modelpnps_suite.sh
#   STEPS=accuracy bash .../h200_modelpnps_suite.sh
#   CASES=04 POINTS=3 bash .../h200_modelpnps_suite.sh
#   NPROC=4 bash .../h200_modelpnps_suite.sh          # processes sharing the GPU
# Variables: STEPS, CASES (default dd05,04,dd20,100um), POINTS (per case, default 1),
# SCAN_CASE (default dd05), SCAN_POINTS (default 4), NPROC (default 3), ACC_CASE
# (accuracy case, default dd05), LUNA_THREADS, RUNDIR.
#
# NO CPU BENCHMARKS RUN HERE. This is rented GPU time; CPU numbers come from the HPC.
# For a full-accuracy A/B, take the `scan` step's collected HDF5 back to the HPC and
# run verify_against_collected against it there — the scan uses production parameters
# precisely so that comparison is direct. See the header of h200_scan_rehearsal.jl.
# The only host work is Julia startup, setup and the tiny grids in the test suite.
# `accuracy` needs no CPU reference — host-vs-device is covered by the hardware-gated
# CUDA testset (small grids) and, at production scale, by verify_against_collected
# against a bit-identical CPU reference on the HPC.
#
# ONE PROCESS, OR SEVERAL SHARING THE CARD?
#   One process running points sequentially is the right model here, and it is what
#   `run_scan` with `Scans.LocalExec` already does. The `share` step exists to CHECK
#   that, not because it is the recommended way to run a campaign.
#
#   Luna's modal suite does share a card across processes, and should: its kernels are
#   ~0.5M elements (nt 8192 × 65 nodes), so the GPU is idle between launches and a
#   second process fills real gaps. A 3-D free-space delay point is ~151M elements per
#   kernel — 300× larger — so one point already saturates the memory system, and a
#   second process only splits the same bandwidth while adding a second CUDA context
#   (~300-500 MB, its own cuFFT plans, its own Julia compilation).
#
#   Note also that without the MPS daemon (nvidia-cuda-mps-control) separate processes
#   are TIME-SLICED, not co-scheduled: the driver switches contexts rather than running
#   their kernels together. So `share` measures interleaving plus overhead. If it does
#   show a real gain, the right response is NOT more processes but per-task CUDA streams
#   inside one process (CUDA.jl gives each Julia task its own stream) — which would need
#   one setup and one set of buffers per task, so k × 10 field copies, and is new work.
#
# CONCURRENCY WITH THE MODAL SUITE
#   Both fit: the modal cases are ≲ 8 GiB, these are 16-40 GiB, against 141 GiB on an
#   H200. But they will contend for MEMORY BANDWIDTH, which is what both are expected
#   to be limited by, so timings taken while the other suite runs are not clean
#   numbers. Run the `bench` step alone if the per-point figure matters; `tests`,
#   `accuracy` and `scan` are fine to overlap.
#
# WHAT TO LOOK FOR
#   * tests: every testset passes.
#   * accuracy: the two extraction routes must agree to < 1e-10 (they reduce the same
#     slices by different routes, so anything larger is a bug in the extraction). The
#     rtol comparison is a PHYSICS result, not a pass/fail: the campaign criterion is
#     every z-slice of Iω_win within 1e-3 of an rtol=1e-8 run, and it has not been
#     measured at the dd05/dd20/100um shapes.
#   * bench: wall per point, and "device GiB" ≈ 10 × the field size. More than that
#     means the cuFFT workspace is NOT negligible at that shape, which changes how
#     many points fit on the card at once. Compare against the A40 reference:
#     the `04` shape measured 4.6 min/point and exactly 22.5 GiB there.
#   * scan: s/point INCLUDING setup and scansave — that is the number to multiply by
#     the delay count when sizing a campaign, not the benchmark's.
#   * share (if run): TOTAL THROUGHPUT, not per-point time. The expected answer is
#     "no meaningful gain" — see below. A gain would be the surprise.
# =============================================================================
set -uo pipefail

[ -r /workspace/env.sh ] && source /workspace/env.sh
PNPS="${PNPS:-/workspace/code/ModelPNPS.jl}"
LUNA="${LUNA:-/workspace/code/Luna.jl}"
DEV="${DEV:-/workspace/code/dev}"
STAMP=$(date +%Y%m%d-%H%M%S)
RUNDIR="${RUNDIR:-/workspace/runs/h200-pnps-$STAMP}"
STEPS="${STEPS:-pkgs,tests,accuracy,bench,scan}"   # `share` is opt-in, see header
CASES="${CASES:-dd05,04,dd20,100um}"
POINTS="${POINTS:-1}"
SCAN_CASE="${SCAN_CASE:-dd05}"
SCAN_POINTS="${SCAN_POINTS:-4}"
NPROC="${NPROC:-3}"
ACC_CASE="${ACC_CASE:-dd05}"

mkdir -p "$RUNDIR" /workspace/logs
LOG="$RUNDIR/suite.log"
exec > >(tee -a "$LOG") 2>&1

export LUNA_TEST_CUDA=1
export JULIA_NUM_THREADS="${LUNA_THREADS:-${JULIA_NUM_THREADS:-8}}"
export OPENBLAS_NUM_THREADS="$JULIA_NUM_THREADS"

t0=$SECONDS
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
step() { printf '\n\033[1;36m==> %s\033[0m  \033[2m(%s, t+%ds)\033[0m\n' "$1" "$(date +%H:%M:%S)" "$((SECONDS-t0))"; }

step "ModelPNPS H100/H200 suite → $RUNDIR"
echo "host $(hostname)  threads $JULIA_NUM_THREADS  cpu target ${JULIA_CPU_TARGET:-native}"
echo "Luna      $(git -C "$LUNA" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$LUNA" rev-parse --short HEAD 2>/dev/null)"
echo "ModelPNPS $(git -C "$PNPS" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$PNPS" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,power.limit --format=csv || true
julia --version
ln -sfn "$RUNDIR" /workspace/runs/latest-pnps

if has pkgs; then
    step "packages the scripts import directly (idempotent)"
    julia --project="$DEV" -e '
        using Pkg
        need = ["CUDA", "HDF5", "FFTW", "AbstractFFTs", "Adapt", "GPUArraysCore",
                "JLArrays", "LinearAlgebra", "Random", "Statistics", "Dates", "Printf", "Test"]
        have = keys(Pkg.project().dependencies)
        miss = filter(p -> !(p in have), need)
        isempty(miss) ? println("    all present") : (println("    adding ", miss); Pkg.add(miss))
        Pkg.precompile()'
fi

if has tests; then
    step "ModelPNPS test suite (incl. hardware-gated CUDA testset)"
    time julia --project="$DEV" "$PNPS/test/runtests.jl"
fi

if has accuracy; then
    step "accuracy (GPU only): extraction routes + rtol convergence, case $ACC_CASE"
    time julia --project="$DEV" "$PNPS/examples/h200_bench.jl" \
        --mode=accuracy --case="$ACC_CASE"
fi

if has bench; then
    step "bench: GPU delay points at production shapes ($CASES, $POINTS point(s) each)"
    time julia --project="$DEV" "$PNPS/examples/h200_bench.jl" \
        --mode=bench --cases="$CASES" --points="$POINTS" --out="$RUNDIR/bench.csv"
    echo; echo "bench.csv:"; cat "$RUNDIR/bench.csv" 2>/dev/null || true
fi

if has scan; then
    step "scan rehearsal: $SCAN_POINTS points of $SCAN_CASE, one process"
    time SCAN_CASE="$SCAN_CASE" SCAN_POINTS="$SCAN_POINTS" \
         SCAN_OUT="$RUNDIR/scan_1proc" \
         julia --project="$DEV" "$PNPS/examples/h200_scan_rehearsal.jl"

fi

if has share; then
    step "saturation check: the same points split across $NPROC processes sharing the GPU"
    # Each process pays its own compilation, so compare the per-point lines rather
    # than the wall clock. Device memory is the constraint: NPROC × 10 × field size
    # must fit, and nvidia-smi below is the check that it did.
    T0=$SECONDS
    for i in $(seq 1 "$NPROC"); do
        SCAN_CASE="$SCAN_CASE" SCAN_POINTS="$SCAN_POINTS" \
        SCAN_OUT="$RUNDIR/scan_${NPROC}proc_$i" \
            julia --project="$DEV" "$PNPS/examples/h200_scan_rehearsal.jl" --batch "$NPROC,$i" \
            > "$RUNDIR/scan_${NPROC}proc_$i.log" 2>&1 &
    done
    wait
    echo "$NPROC processes: wall $((SECONDS-T0)) s for the same $SCAN_POINTS points"
    echo "Compare against the one-process scan wall above: if the card was already"
    echo "saturated these are about equal (and the per-point times below have simply"
    echo "stretched by ~$NPROC×). Each process also paid its own compilation."
    grep -h "scan wall\|per point" "$RUNDIR"/scan_${NPROC}proc_*.log || true
    echo "GPU memory now:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
fi

step "done in $((SECONDS-t0)) s — results in $RUNDIR (also /workspace/runs/latest-pnps)"
cp -f "$LOG" "/workspace/logs/h200-pnps-$STAMP.log" 2>/dev/null || true
