#!/bin/bash -l
# =============================================================================
# A/B verify selected 04-campaign delay points on one GPU, against the collected
# CPU reference. See examples/verify_production_04.jl for the parameter set
# (which mirrors 04_production_gap1000_kerr_raman.jl exactly).
#
# Submit:  sbatch verify04gpu.sbatch
#   other points:  sbatch --export=ALL,IDCS=10,60,110 verify04gpu.sbatch
#   CPU control:   sbatch --export=ALL,ARRAYTYPE=cpu verify04gpu.sbatch   (see NOTE)
#
# Prerequisites:
#   * ~/ref04.h5 — a COPY of the collected file (a running scan writes to the
#     original between points):
#       cp ~/sharedscratch/lunascans/tgfrog/tgfrog_kerr_rtol7_sw_gap1000um_tanh_1fs_40umUVFS_collected.h5 ~/ref04.h5
#   * ~/sharedscratch/cuda/env instantiated and precompiled by an earlier GPU
#     job (gputest.sbatch does this). Nothing here precompiles, so the login
#     node never loads CUDA.jl.
#
# WHAT CHANGED SINCE verify04gpu-442443
#   Delay points now reduce each z-slice as it is saved (TraceExtractOutput), so
#   the 16 x 2.25 GiB of slices are never written or read back: no temp file, and
#   no ~36 GiB round trip through the filesystem per point. Two consequences for
#   this script: local scratch is no longer needed, and the wall time should drop
#   by most of the 38-50 s/point of non-propagation overhead measured before.
#
#   The device reductions sum in a different order from the host loop, so the
#   relative differences will NOT reproduce -442443 exactly. Expect the same
#   order of magnitude; the acceptance bar is unchanged at < 1e-3 of the trace
#   maximum (which -442443 met at 7.4e-4 worst case).
# =============================================================================

#SBATCH --export=ALL
#SBATCH -o verify04gpu-%j.output
#SBATCH -e verify04gpu-%j.error
#SBATCH -J verify04gpu
#SBATCH --time=02:00:00
## Host memory. The 14.2 GiB peak measured on -442443 is set by build_setup, not
## by the run (four beamlet fields at 2.25 GiB each, plus the host FFT prototype
## Luna.setup allocates and the device path never uses). Save-time extraction
## removes the run-side host cost, not that peak — see GPU-TODO.md sections 2 and
## Luna section 7, which between them would take this to ~5 GiB.
#SBATCH --mem=24G
#SBATCH --ntasks=1
## The propagation is on the GPU; these serve host-side setup and the one z=0
## slice that still comes back through the interpolant.
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH -p gpu

set -uo pipefail

CUDAROOT="$HOME/sharedscratch/cuda"

# 4 host threads, matching -442443 so the wall times stay comparable.
export JULIA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

REF="${REF:-$HOME/ref04.h5}"
IDCS="${IDCS:-10,60,110}"
ARRAYTYPE="${ARRAYTYPE:-cuda}"
NGRID="${NGRID:-768}"
# fftw_threads=2 / :measure match the production run. On the GPU path these only
# affect the (unused) host plan, but keeping them means the cached wisdom is hit
# rather than re-measured, and it keeps a --export=ALL,ARRAYTYPE=cpu control run
# strictly comparable to the reference.
FFTW_THREADS="${FFTW_THREADS:-2}"
FFTW_MODE="${FFTW_MODE:-measure}"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
echo "ref=$REF  idcs=$IDCS  arraytype=$ARRAYTYPE  N=$NGRID"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
echo

[ -f "$REF" ] || { echo "ERROR: no reference file at $REF"; exit 1; }

julia --project="$CUDAROOT/env" \
      "$CUDAROOT/ModelPNPS/examples/verify_production_04.jl" \
      "$REF" "$IDCS" "$ARRAYTYPE" "$NGRID" "$FFTW_THREADS" "$FFTW_MODE"

echo "=== done ==="

# NOTE on the CPU control: ARRAYTYPE=cpu runs the save-the-stack route (save-time
# extraction defaults off on the host, so the CPU path stays bit-for-bit identical
# to the validated production reference). It therefore still needs the temp-file
# space and ~34 GiB of host memory, and takes ~50 min per point rather than ~4 —
# raise --mem to 40G and --time to 08:00:00, and drop --gres, if you run one.
