#!/usr/bin/env bash
# =============================================================================
# Launch a field-resolved (RealGrid) TG-FROG scan on one H100/H200.
#
# The arm is a `const` INSIDE h200_field_mode.jl (CONFIG, RESPONSE, FFAC), not an
# environment variable — read that file's header before running anything. In order:
#
#   1. CONFIG = :port2fs, RESPONSE = :nothg   the port check. Must agree with the
#      delivered 2 fs envelope trace; a disagreement here is a bug, not physics.
#   2. CONFIG = :expt1fs, RESPONSE = :nothg   the experiment (representation error alone).
#   3. CONFIG = :expt1fs, RESPONSE = :thg     what the envelope additionally omits.
#
# PREREQUISITE: runpodcoldstart.sh must have been run on this pod at least once. It is
# idempotent (~2 min on a warm volume) and is also how the pod picks up new commits:
#
#   bash /workspace/code/Luna.jl/test/manual/runpodcoldstart.sh
#
# ALWAYS DRY-RUN FIRST. The field grid is 2x the envelope's in Nω and its nonlinear
# evaluation needs buffers the envelope path does not have at all; at N = 768 the :nothg
# arm is ~95 GiB resident, which does not fit an 80 GB card. The dry run computes the
# budget against the card actually present and refuses rather than dying an hour in:
#
#   PNPS_DRYRUN=1 bash /workspace/code/ModelPNPS.jl/examples/h200_field_mode.sh
#
# Then, from tmux so an SSH drop does not kill the run:
#
#   tmux new -s fieldmode
#   bash /workspace/code/ModelPNPS.jl/examples/h200_field_mode.sh
#   # detach C-b d, reattach: tmux attach -t fieldmode
#
# Re-running resumes: the collected file records which points are done.
# Variables: FIELD_POINTS (default 11), RUNDIR (default /workspace/runs/fieldmode),
# FIELD_NAME, FIELD_ARRAYTYPE, LUNA_THREADS, PNPS_DRYRUN.
# =============================================================================
set -uo pipefail

# env.sh is what puts julia on PATH and the depot on the volume. A tmux shell gets it via
# .bashrc; a non-interactive one does not, so do it here regardless.
if [ -r /workspace/env.sh ]; then
    # shellcheck source=/dev/null
    source /workspace/env.sh
else
    echo "ERROR: /workspace/env.sh not found — run runpodcoldstart.sh first."
    exit 1
fi

PNPS="${PNPS:-/workspace/code/ModelPNPS.jl}"
LUNA="${LUNA:-/workspace/code/Luna.jl}"
DEV="${DEV:-/workspace/code/dev}"
RUNDIR="${RUNDIR:-/workspace/runs/fieldmode}"
export FIELD_POINTS="${FIELD_POINTS:-11}"
export PNPS_DRYRUN="${PNPS_DRYRUN:-0}"
export JULIA_NUM_THREADS="${LUNA_THREADS:-${JULIA_NUM_THREADS:-8}}"
export OPENBLAS_NUM_THREADS="$JULIA_NUM_THREADS"

for d in "$PNPS" "$LUNA" "$DEV"; do
    [ -d "$d" ] || { echo "ERROR: $d missing — run runpodcoldstart.sh first."; exit 1; }
done
[ -f "$DEV/Manifest.toml" ] || { echo "ERROR: $DEV not instantiated — run runpodcoldstart.sh."; exit 1; }

mkdir -p "$RUNDIR"
cd "$RUNDIR" || exit 1
LOG="$RUNDIR/fieldmode-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== field-resolved TG-FROG scan ==="
echo "host $(hostname)   threads $JULIA_NUM_THREADS   points $FIELD_POINTS   dryrun $PNPS_DRYRUN"
for r in "$LUNA" "$PNPS"; do
    b=$(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null)
    h=$(git -C "$r" rev-parse --short HEAD 2>/dev/null)
    dirty=$([ -n "$(git -C "$r" status --porcelain 2>/dev/null)" ] && echo " (DIRTY)" || echo "")
    # Behind origin usually means coldstart has not been rerun since you pushed.
    behind=$(git -C "$r" rev-list --count "HEAD..origin/$b" 2>/dev/null || echo "?")
    [ "$behind" != "0" ] && [ "$behind" != "?" ] && behind=" — $behind commit(s) BEHIND origin/$b, rerun runpodcoldstart.sh" || behind=""
    printf '%-14s %s @ %s%s%s\n' "$(basename "$r")" "$b" "$h" "$dirty" "$behind"
done
# The arm is compiled into the .jl file, so echo it here: a log that does not say which
# arm produced it is a log you cannot use.
grep -E '^const (CONFIG|RESPONSE|FFAC) ' "$PNPS/examples/h200_field_mode.jl" || true
nvidia-smi --query-gpu=name,memory.total,memory.free,power.limit --format=csv || {
    echo "ERROR: no GPU visible"; exit 1; }
echo "run dir  $RUNDIR"
echo "log      $LOG"
echo

julia --project="$DEV" "$PNPS/examples/h200_field_mode.jl"
rc=$?

echo
if [ $rc -eq 0 ]; then
    if [ "$PNPS_DRYRUN" = "1" ]; then
        echo "=== dry run only: nothing was propagated ==="
    else
        echo "=== finished. Collected file in $RUNDIR: ==="
        ls -lh "$RUNDIR"/*_collected.h5 2>/dev/null
        echo "Copy it off the pod. NOTE its /grid/ω is a monotonic rfft half-spectrum and"
        echo "the file carries /grid/field_mode = 1; readers must not fftshift it."
    fi
else
    echo "=== exited with status $rc — rerun this script in the same directory to resume ==="
fi
exit $rc
