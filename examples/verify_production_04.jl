# =============================================================================
# A/B verification of selected delay points against the running/completed
# 04 production scan (tgfrog_kerr_rtol8_gap1000um_tanh_1fs_40umUVFS).
#
# Recomputes chosen scan indices with the current code — on the host or on a GPU —
# and compares every trace dataset against the collected file, reporting the
# relative differences plus wall time, host RSS and device memory per point.
# Points the running scan has not yet written are reported as NaN and skipped.
#
# Usage:
#   julia --project=<env> verify_production_04.jl <collected.h5> <idx1,idx2,...> \
#         [arraytype] [N] [fftw_threads]
#
#   arraytype     cpu (default) or cuda
#   N             transverse grid size (default 1024 = the reference grid)
#   fftw_threads  default 2, matching the reference scan. Only affects the CPU
#                 path and the host-side setup; irrelevant on a GPU.
#
# Read the collected file from a COPY: a running scan writes to it between points.
#
#   cp ~/sharedscratch/lunascans/tgfrog/tgfrog_kerr_rtol8_gap1000um_tanh_1fs_40umUVFS_collected.h5 ~/ref04.h5
#
# ACCEPTANCE
#   cpu,  N=1024: the propagation is unchanged, so expect ~1e-12 or below.
#   cuda, N=1024: different FFT library and a parallel error-norm reduction, so
#                 not bitwise. Bar: < 1e-3 relative on every dataset, the same
#                 criterion used for the CPU grid change; in practice the small
#                 test cases agree to ~1e-11.
#
# The k-space-integrated datasets are rescaled automatically if N differs from
# the reference (their FFT-bin units scale as N^4 at fixed R).
# =============================================================================

using ModelPNPS
import ModelPNPS as TS
import Luna

length(ARGS) >= 2 || error("usage: verify_production_04.jl <collected.h5> " *
                           "<idx1,idx2,...> [arraytype] [N] [fftw_threads]")
collected    = ARGS[1]
scanidcs     = parse.(Int, split(ARGS[2], ","))
arraytype    = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :cpu
N            = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1024
fftw_threads = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2

Luna.set_fftw_mode(:estimate)   # matches the reference scan
Luna.set_fftw_threads(fftw_threads)

# --- Setup arguments: EXACT copy of 04_production_gap1000_kerr_raman.jl ------
const GAP = 1.0e-3
λ0           = 260e-9
τfwhm        = 1.0e-15
energy       = 0.1e-6
material     = :SiO2
thickness    = 40e-6
a            = 125e-6
f_coll       = 5.0
f_foc        = 0.1
mask_diam    = 1.0e-3
mask_spacing = GAP
λlims        = (143e-9, 600e-9)
d            = mask_spacing/2 + mask_diam/2

beam   = TS.HE11Beam(a, f_coll, f_foc)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                               zmask=f_foc, apod=:tanh)

setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                raman=false,                 # the Kerr arm of the 04 pair
                R=366.0e-6, N=N,
                arraytype=arraytype)

# The 16-slice ladder from the production script
zsave = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.5,
         12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0] .* 1e-6

@info "verifying" collected scanidcs arraytype N fftw_threads
results = TS.verify_against_collected(setup_args, collected, scanidcs;
                                      zsave=zsave, init_dz=5e-7,
                                      rtol=1e-8, max_dz=2e-6)

println("\n==== SUMMARY (arraytype=$arraytype, N=$N) ====")
for p in results
    print("scanidx $(p["scanidx"])  τ = $(round(p["τ"]*1e15; digits=3)) fs  ",
          "wall = $(round(p["wall_s"]/60; digits=1)) min  ",
          "host = $(round(p["maxrss_GiB"]; digits=1)) GiB")
    haskey(p, "device_used_GiB") &&
        print("  device = $(round(p["device_used_GiB"]; digits=1)) GiB")
    println()
    for k in sort(collect(keys(p)))
        startswith(k, "Iω") && println("    $(rpad(k, 24)) max rel diff = $(p[k])")
    end
end
