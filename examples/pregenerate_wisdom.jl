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
# of the same architecture as the production jobs:
#
#   JULIA_NUM_THREADS=1 julia --project pregenerate_wisdom.jl
#
# IMPORTANT: the wisdom cache is keyed by the FFTW thread count — set
# `fftw_threads` below to exactly the value the production scan passes to
# `run_scan(...; fftw_threads=...)`.
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna

fftw_threads = 8          # must match the production fftw_threads
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
                         λlims=(143e-9, 600e-9),
                         beam, window,
                         raman=true,          # also plans the batched-Raman FFTs below
                         R=366.0e-6, N=640)

# One transform evaluation forces every lazily-created plan (in particular the
# batched-Raman in-place FFTs on the doubled time grid), so all wisdom is saved.
Eωk = TS.delayed_input(setup, 0.0)
nl = similar(Eωk)
setup.transform(nl, Eωk, 0.0)

@info "FFTW wisdom generated and cached for $(fftw_threads) threads."
