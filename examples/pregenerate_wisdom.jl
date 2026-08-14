# =============================================================================
# Pre-generate FFTW wisdom for the production grid shapes.
#
# Production scan jobs use fftw_mode=:estimate because MEASURE-class planning
# of the 3-D transforms takes minutes to tens of minutes — but Luna caches
# FFTW wisdom per (shape, thread count) in a scratch directory shared via the
# home filesystem, so planning done ONCE here is reused by every subsequent
# job: they can then run with fftw_mode=:measure and get measured-plan FFT
# speed (typically 1.3–2x) at :estimate startup cost.
#
# Run once per (grid shape, FFTW thread count) combination, on a compute node
# of the same architecture as the production jobs. Parameters are positional
# command-line arguments — `pregenerate_wisdom.jl [fftw_threads] [N] [trange_fs]`:
#
#   # 04 production pair (N = 1024, fftw_threads = 2, default trange):
#   JULIA_NUM_THREADS=2 julia --project pregenerate_wisdom.jl 2 1024
#
#   # 05 100 µm run (N = 640, fftw_threads = 4, trange = 220 fs -> Nω = 512):
#   JULIA_NUM_THREADS=4 julia --project pregenerate_wisdom.jl 4 640 220
#
# IMPORTANT: the wisdom cache is keyed by the FFTW thread count AND the
# transform shape, so BOTH must match the production run exactly — the shape
# comes from (λlims, trange) via Nω and from the transverse N. Wisdom for the
# wrong (threads, shape) is silently unusable: every scan process then falls
# back to full MEASURE planning, which at 50 concurrent one-shot processes is
# minutes-to-tens-of-minutes each, repeated for every scan point.
#
# Verify afterwards that the cache file for the right thread count exists:
#   ls ~/.julia/scratchspaces/*/lunacache/FFTWcache_*threads
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna

# Positional args with the previous hardcoded values as defaults.
fftw_threads = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
N_grid       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 640
trange       = length(ARGS) >= 3 ? parse(Float64, ARGS[3])*1e-15 : 40e-15

@info "Pregenerating wisdom for fftw_threads=$fftw_threads, N=$N_grid, " *
      "trange=$(trange*1e15) fs (Julia threads: $(Threads.nthreads()))"

Luna.set_fftw_threads(fftw_threads)
Luna.set_fftw_mode(:measure)   # or :patient for another few % at much longer planning

# --- Match the production build_setup arguments exactly (grid shapes are what
# matters: Nω from (λlims, trange), transverse N, and raman on/off) -----------
const GAP = 1.0e-3
d = GAP/2 + 1.0e-3/2
beam   = TS.HE11Beam(125e-6, 5.0, 0.1)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                               zmask=0.1, apod=:tanh)

setup = TS.build_setup(; λ0=260e-9, τfwhm=1.0e-15, energy=0.1e-6,
                         thickness=40e-6, material=:SiO2,
                         mask_diam=1.0e-3, mask_spacing=GAP,
                         λlims=(143e-9, 600e-9), trange=trange,
                         beam, window,
                         raman=true,          # also plans the batched-Raman FFTs below
                         apod=:supergauss, apod_param=16, store_window=false,
                         R=366.0e-6, N=N_grid)

# One transform evaluation forces every lazily-created plan (in particular the
# batched-Raman in-place FFTs on the doubled time grid), so all wisdom is saved.
Eωk = TS.delayed_input(setup, 0.0)
nl = similar(Eωk)
setup.transform(nl, Eωk, 0.0)

@info "FFTW wisdom generated and cached for $(fftw_threads) threads."
