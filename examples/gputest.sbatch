#!/bin/bash -l
# =============================================================================
# Hardware validation of the Luna/ModelPNPS GPU backend (plan Stage 11).
#
# Runs, on a GPU compute node:
#   1. Luna's hardware-gated CUDA tests (LUNA_TEST_CUDA=1) — cuFFT semantics,
#      CUDA code generation, device-vs-host agreement, and the resident memory
#      footprint including the cuFFT plan workspace.
#   2. ModelPNPS's device tests — a whole delay point on the device.
#   3. Optionally the standalone kernel benchmark (BENCH=1).
#
# ONE-TIME SETUP (see the block at the bottom of this file) must be done first:
# the environment is installed on the login node but PRECOMPILED on a GPU node,
# because this cluster's login node faults while precompiling CUDA.jl.
#
# Submit:   sbatch gputest.sbatch
#   two GPUs:   sbatch --gres gpu:2 gputest.sbatch
#   with bench: BENCH=1 sbatch --export=ALL,BENCH=1 gputest.sbatch
# =============================================================================

## Environment variables
#SBATCH --export=ALL
## Output and error files
#SBATCH -o gputest-%j.output
#SBATCH -e gputest-%j.error
## Job name
#SBATCH -J luna-gputest
## Run time
#SBATCH --time=02:00:00
## Host memory (TOTAL, not per-cpu). GPU memory is NOT counted here — the field
## arrays live on the card. The host only holds the input field built before
## upload, the device-to-host save buffer, and the Julia + CUDA runtime.
#SBATCH --mem=8G
## CPUs: the propagation runs on the GPU, so these only serve host-side setup,
## HDF5 and extraction. More would not help.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
## GPU requirements
#SBATCH --gres gpu:1
## Partition
#SBATCH -p gpu

# NB: no `set -e`. A failing test suite should not stop the later steps — the
# whole point of the job is to collect every result in one submission.
set -uo pipefail

CUDAROOT="$HOME/sharedscratch/cuda"
ENVDIR="$CUDAROOT/env"

# Host-side threads only (the device does the propagation). Keep this at or
# below --cpus-per-task so concurrent jobs do not oversubscribe the node.
export JULIA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
# Enables the hardware-gated tests in Luna/test/test_cuda.jl; without it they
# are skipped and the job would pass vacuously.
export LUNA_TEST_CUDA=1

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
echo

# Precompilation happens here, on the GPU node: the login node faults (illegal
# instruction) while precompiling CUDA.jl's BFloat16s dependency. Cached after
# the first run, so subsequent jobs skip it.
echo "=== instantiate/precompile ==="
julia --project="$ENVDIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo

echo "=== Luna CUDA tests ==="
julia --project="$ENVDIR" "$CUDAROOT/Luna/test/runtests.jl" test_cuda
echo

echo "=== ModelPNPS device tests ==="
julia --project="$ENVDIR" "$CUDAROOT/ModelPNPS/test/runtests.jl"
echo

if [ "${BENCH:-0}" = "1" ]; then
    echo "=== kernel benchmark ==="
    # ARGS[1] is the RK step count used for the per-point estimates; take it
    # from a CPU job log for the configuration you care about.
    julia --project="$ENVDIR" "$CUDAROOT/ModelPNPS/examples/gpu_bench.jl" 144
fi

echo "=== done ==="

# =============================================================================
# ONE-TIME SETUP
#
# On the login node (installs only — the precompile failure there is expected
# and harmless, the job above redoes it on a GPU node):
#
#   mkdir -p ~/sharedscratch/cuda && cd ~/sharedscratch/cuda
#   git clone -b gpu <luna-remote>      Luna
#   git clone -b gpu <modelpnps-remote> ModelPNPS
#   julia -e 'using Pkg; Pkg.activate(ENV["HOME"]*"/sharedscratch/cuda/env");
#             Pkg.develop(path=ENV["HOME"]*"/sharedscratch/cuda/Luna");
#             Pkg.develop(path=ENV["HOME"]*"/sharedscratch/cuda/ModelPNPS");
#             Pkg.add(["CUDA", "JLArrays", "AbstractFFTs", "Adapt", "FFTW", "HDF5",
#                      "LinearAlgebra", "Test", "Random"])'
#
# The test files are run DIRECTLY (not through Pkg.test), so every package they
# import must be a direct dependency of this environment — including ones that
# are dependencies of Luna/ModelPNPS themselves (FFTW, HDF5, ...), which are not
# transitively importable, and the test-only extras (JLArrays, AbstractFFTs).
#
# If a node ever faults loading CUDA.jl, add `-C x86-64-v3` to every julia
# invocation above: that caps the instruction set at an AVX2 baseline every
# node here can execute, at no cost since the work is on the GPU.
#
# WHAT TO LOOK FOR IN THE OUTPUT
#   * "CUDA vs host  relL2=..." — must be < 1e-8. This is the acceptance check.
#   * "device memory after setup / after run  ...fields" — the resident field
#     count. The budget assumes 9 solver registers + 1 transform buffer plus at
#     most ~1 field of cuFFT workspace; a much larger number means the workspace
#     estimate was wrong and the campaign memory table needs revising.
#   * Any scalar-indexing error is a real bug in the device path, not a test
#     artefact: the file sets CUDA.allowscalar(false) deliberately.
# =============================================================================
