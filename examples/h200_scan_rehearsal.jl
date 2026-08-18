# =============================================================================
# A real (short) TG-FROG delay scan on the GPU, through the production path.
#
# The benchmark measures delay points in isolation; this measures what a campaign
# actually costs, including everything the benchmark skips: `run_scan`, the
# per-point setup, `scansave` into the collected HDF5, and the accumulation of
# device memory across points that `Luna.device_reclaim()` exists to prevent.
#
# Bare machine, so `Scans.LocalExec` runs the points in this process. Several
# processes can share the GPU with Luna's `--batch` convention, which is how a
# campaign would use a card bigger than one point needs:
#
#   julia --project=$DEV h200_scan_rehearsal.jl
#   for i in 1 2 3; do julia --project=$DEV h200_scan_rehearsal.jl --batch 3,$i & done; wait
#
# Environment: SCAN_CASE (dd05|04|dd20|100um, default dd05), SCAN_POINTS (default 4),
# SCAN_OUT (default ./h200_scan_<case>), SCAN_ARRAYTYPE (cuda|cpu, default cuda).
#
# ACCURACY: use the collected file this writes as the reference for a CPU run on the
# HPC, rather than spending rented GPU time on a host reference. The parameters below
# are the production ones, so the two are directly comparable:
#
#   # on the HPC, against the file this produced:
#   verify_against_collected(setup_args, "h200_rehearsal_<case>_collected.h5",
#                            1:SCAN_POINTS; zsave, init_dz=5e-7, rtol=1e-7, max_dz=2e-6,
#                            twin_period=TWIN_SAVES_ONLY)
#
# That reports each dataset's difference normalised BOTH to the point's own peak and to
# the scan-wide peak — read the second one (see examples/scan_peaks.jl for why: a delay
# wing carries a signal orders of magnitude below τ≈0, so the own-peak number looks
# alarming for a difference that is invisible in the assembled trace).
# =============================================================================
using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.Scans
import CUDA
import Printf: @printf
import Dates

CASE   = get(ENV, "SCAN_CASE", "dd05")
NPTS   = parse(Int, get(ENV, "SCAN_POINTS", "4"))
OUTDIR = get(ENV, "SCAN_OUT", "./h200_scan_$CASE")
ATYPE  = get(ENV, "SCAN_ARRAYTYPE", "cuda")

const TWIN_SAVES_ONLY = 1_000_000_000
const ZSAVE_40  = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
                   12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6
const ZSAVE_100 = [0.0, 2.0, 4.0, 8.0, 9.5, 16.0, 25.0, 40.0, 60.0, 100.0] .* 1e-6
const CASES = Dict(
  "dd05"  => (gap_mm=0.5, N=640,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40),
  "04"    => (gap_mm=1.0, N=768,  trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40),
  "dd20"  => (gap_mm=2.0, N=1024, trange=110e-15, thickness=40e-6,  zsave=ZSAVE_40),
  "100um" => (gap_mm=1.0, N=640,  trange=220e-15, thickness=100e-6, zsave=ZSAVE_100),
)
haskey(CASES, CASE) || error("SCAN_CASE must be one of $(sort(collect(keys(CASES))))")
c = CASES[CASE]

gap = c.gap_mm * 1e-3
d   = gap/2 + 1.0e-3/2
setup_args = (; λ0=260e-9, τfwhm=1.0e-15, energy=0.1e-6,
                material=:SiO2, thickness=c.thickness,
                mask_diam=1.0e-3, mask_spacing=gap, λlims=(143e-9, 600e-9),
                beam=TS.HE11Beam(125e-6, 5.0, 0.1),
                window=TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                                             zmask=0.1, apod=:tanh),
                apod=:supergauss, apod_param=16,
                trange=c.trange, store_window=false, R=366.0e-6, N=c.N,
                # A resolved type, not the `:cuda` symbol: the symbol form exists for
                # cluster login nodes that cannot load CUDA.jl, and costs a world-age
                # dance this machine has no reason to pay.
                arraytype = ATYPE == "cuda" ? CUDA.CuArray : Array)

# Delays spanning wing to centre: step count and therefore cost vary across a
# scan, so a rehearsal that only ran τ=0 would under-report the campaign.
τ = collect(range(-25e-15, 0.0, NPTS))

# `--batch N,i` → this process takes every N-th point; anything else runs them all.
function exec_from_args()
    i = findfirst(==("--batch"), ARGS)
    isnothing(i) && return Scans.LocalExec()
    nb, b = parse.(Int, split(ARGS[i+1], ","))
    @printf("batch %d of %d\n", b, nb)
    return Scans.BatchExec(nb, b)
end

mkpath(OUTDIR)
cd(OUTDIR)
@printf("case %s: (Nω, %d, %d), %d z-slices, %d delay points, arraytype %s\n",
        CASE, c.N, c.N, length(c.zsave), NPTS, ATYPE)
ATYPE == "cuda" && @printf("device %s, %.1f GiB\n",
                           CUDA.name(CUDA.device()), CUDA.total_memory()/2^30)
println("started $(Dates.now())")

t0 = time()
TS.run_scan(setup_args, τ; scan_name="h200_rehearsal_$CASE",
            exec=exec_from_args(), zsave=c.zsave,
            init_dz=5e-7, rtol=1e-7, max_dz=2e-6, twin_period=TWIN_SAVES_ONLY)
wall = time() - t0

# BatchExec runs a subset, so report against what this process actually did.
npts_here = length(τ)
i = findfirst(==("--batch"), ARGS)
if !isnothing(i)
    nb, b = parse.(Int, split(ARGS[i+1], ","))
    npts_here = length(b:nb:length(τ))
end
@printf("\nscan wall %.1f s for %d point(s) → %.1f s per point (includes setup and scansave)\n",
        wall, npts_here, wall/npts_here)
@printf("collected file: %s\n", joinpath(OUTDIR, "h200_rehearsal_$(CASE)_collected.h5"))
println("done $(Dates.now())")
