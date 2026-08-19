#!/usr/bin/env bash
# =============================================================================
# Launch the 05 100 µm production scan on one H100/H200.
#
# PREREQUISITE: runpodcoldstart.sh must have been run on this pod at least once.
# It is idempotent (~2 min on a warm volume) and does three things this needs:
# puts Julia and the depot on the persistent volume, creates the dev project with
# both packages `Pkg.develop`ed, and FAST-FORWARDS both checkouts to origin — so
# it is also how the pod picks up new commits. Run it again now if you have
# pushed anything since the pod started:
#
#   bash /workspace/code/Luna.jl/test/manual/runpodcoldstart.sh
#
# Then, from a tmux session so an SSH drop does not kill a 1.5 h run:
#
#   tmux new -s prod05
#   bash /workspace/code/ModelPNPS.jl/examples/h200_production_05_100um.sh
#   # detach C-b d, reattach: tmux attach -t prod05
#
# Re-running resumes: the collected file records which points are done.
# Variables: PROD_POINTS (default 241; try 8 for a ~4 min rehearsal on the real
# grid), RUNDIR (default /workspace/runs/prod05), PROD_NAME, LUNA_THREADS.
# =============================================================================
set -uo pipefail

# env.sh is what puts julia on PATH and the depot on the volume. A tmux shell
# gets it via .bashrc; a non-interactive one does not, so do it here regardless.
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
RUNDIR="${RUNDIR:-/workspace/runs/prod05}"
export PROD_POINTS="${PROD_POINTS:-241}"
export JULIA_NUM_THREADS="${LUNA_THREADS:-${JULIA_NUM_THREADS:-8}}"
export OPENBLAS_NUM_THREADS="$JULIA_NUM_THREADS"

for d in "$PNPS" "$LUNA" "$DEV"; do
    [ -d "$d" ] || { echo "ERROR: $d missing — run runpodcoldstart.sh first."; exit 1; }
done
[ -f "$DEV/Manifest.toml" ] || { echo "ERROR: $DEV not instantiated — run runpodcoldstart.sh."; exit 1; }

mkdir -p "$RUNDIR"
cd "$RUNDIR" || exit 1
LOG="$RUNDIR/prod05-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 05 100 µm production scan ==="
echo "host $(hostname)   threads $JULIA_NUM_THREADS   points $PROD_POINTS"
for r in "$LUNA" "$PNPS"; do
    b=$(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null)
    h=$(git -C "$r" rev-parse --short HEAD 2>/dev/null)
    dirty=$([ -n "$(git -C "$r" status --porcelain 2>/dev/null)" ] && echo " (DIRTY)" || echo "")
    # Behind origin usually means coldstart has not been rerun since you pushed.
    behind=$(git -C "$r" rev-list --count "HEAD..origin/$b" 2>/dev/null || echo "?")
    [ "$behind" != "0" ] && [ "$behind" != "?" ] && behind=" — $behind commit(s) BEHIND origin/$b, rerun runpodcoldstart.sh" || behind=""
    printf '%-14s %s @ %s%s%s\n' "$(basename "$r")" "$b" "$h" "$dirty" "$behind"
done
nvidia-smi --query-gpu=name,memory.total,memory.free,power.limit --format=csv || {
    echo "ERROR: no GPU visible"; exit 1; }
echo "run dir  $RUNDIR"
echo "log      $LOG"
echo

# One point of this shape measured at ~21 s / 32.8 GiB on an H200, so 241 points
# is ~1.5 h. The script prints its own estimate and resumes if interrupted.
julia --project="$DEV" "$PNPS/examples/h200_production_05_100um.jl"
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "=== finished. Collected file in $RUNDIR: ==="
    ls -lh "$RUNDIR"/*_collected.h5 2>/dev/null
    echo "Copy it off the pod — the accuracy companion is a CPU rerun of a few"
    echo "points on the HPC with verify_against_collected against this file."
else
    echo "=== exited with status $rc — rerun this script in the same directory to resume ==="
fi
exit $rc
