#!/bin/bash -l
# =============================================================================
# Minimal GPU job: compare the RK45 error norm on host vs CUDA device.
# See examples/check_device_norm.jl for what this measures and why.
#
# Submit:  sbatch check_device_norm.sbatch
#   larger grid:  sbatch --export=ALL,SZ="256 512" check_device_norm.sbatch
#
# Assumes the environment at ~/sharedscratch/cuda/env is already instantiated
# and precompiled by an earlier GPU job (gputest.sbatch does that); nothing
# here precompiles, so the login node never touches CUDA.jl.
# =============================================================================

#SBATCH --export=ALL
#SBATCH -o checknorm-%j.output
#SBATCH -e checknorm-%j.error
#SBATCH -J luna-checknorm
#SBATCH --time=00:20:00
## Host memory: 9 field-sized host arrays at the default (256,256,256) is ~2.4 GB,
## plus the stage temporaries and the Julia + CUDA runtime.
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres gpu:1
#SBATCH -p gpu

set -uo pipefail

CUDAROOT="$HOME/sharedscratch/cuda"

# 8 threads to match the CPU verification runs whose step counts prompted this.
# NB the host norm kernel is a sequential loop, so this does not change it --
# it only keeps the rest of the process consistent with those runs.
export JULIA_NUM_THREADS=8

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
echo

julia --project="$CUDAROOT/env" "$CUDAROOT/ModelPNPS/examples/check_device_norm.jl" ${SZ:-}

echo "=== done ==="
