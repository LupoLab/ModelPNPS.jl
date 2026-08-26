# =============================================================================
# The 04 gate beamlet's spatially resolved focal field — setup only, no scan.
#
# Builds the setup for `04_production_gap1000_kerr_raman.jl` (the Kerr arm) and writes the
# beamlet diagnostics to one small HDF5. It does NOT propagate: `build_setup` constructs
# the grids, the beamlets and the focal profile, and the script stops there. A few tens of
# seconds and ~19 GiB of host memory at N = 768, against ~8 h for a delay point.
#
# WHY
#   The stored truth `Eω_beamlet` is a hybrid — spatially INTEGRATED amplitude with the
#   1-D input pulse's phase. Generation is a three-field product, so it weights the intense
#   centre of the focal spot rather than total energy, and for an Airy beamlet the spot
#   area goes as λ², making the on-axis spectrum bluer than the integrated one. This file
#   carries the actual E(ω, r) so the effective spectrum can be computed instead of assumed
#   — and carries `Iω_beamlet` and `Eω_beamlet` alongside it, so the check is one file.
#
# WHAT IS IN IT
#   /Eω_beamlet_r_re, _im  (Nω, nr)  complex focal field of gate 1, radial profile
#   /beamlet_r             (nr,)     radius from the beamlet centre, METRES
#   /beamlet_r_asym        (Nω,)     azimuthal RMS of |E| over its mean: how well a
#                                    radial profile describes the beamlet at each ω
#   /omega, /lambda        (Nω,)     the frequency and wavelength axes
#   /Iomega_beamlet        (Nω,)     the existing integrated truth, for comparison
#   /Eomega_beamlet_re,_im (Nω,)     the existing hybrid complex truth
#   /Iomega, /Eomega_re,_im          the 1-D input pulse
#   plus the geometry (hole centre, zmask, tilt coefficients, mask, R, N, λ0 …) and
#   provenance (git commit, date, the source script).
#
#   NOTE the radial profile has the geometric tilt REMOVED — k₀ = hole·ω/(c·zmask), which
#   reaches ~37 rad across the sampling radius and would otherwise annihilate the azimuthal
#   average. Multiply by exp(+iω(coefx·x + coefy·y)) to restore the full focal field. The
#   beamlets cross AT THE FOCUS; the BOXCARS corners are in the mask plane, i.e. k-space.
#
# RUN
#   julia --project=<env with ModelPNPS> beamlet_profile_04.jl
#   BEAMLET_OUT=/path/to/file.h5 julia ... beamlet_profile_04.jl
# =============================================================================

import Pkg
let dflt = try Base.load_path_expand("@v#.#") catch; nothing end
    if Base.active_project() == dflt
        dev = get(ENV, "LUNA_DEV", "/workspace/code/dev")
        isdir(dev) || error("no --project given and $dev does not exist; pass --project " *
                            "pointing at an environment with ModelPNPS")
        Pkg.activate(dev; io=devnull)
    end
end

using ModelPNPS
import ModelPNPS as TS
import Luna
import Luna.PhysData
import HDF5
import Printf: @printf
import Dates

Luna.set_fftw_mode(:estimate)

# ---------------------------------------------------------------- 04 parameters --
# Copied from 04_production_gap1000_kerr_raman.jl (RAMAN = false, the Kerr arm). The
# beamlet does not depend on the nonlinearity, but the arguments are kept identical so
# this is unambiguously the production beamlet.
const GAP    = 1.0e-3
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

const R_GRID = 366.0e-6
const N_GRID = parse(Int, get(ENV, "BEAMLET_N", "768"))   # 04's designed grid

beam   = TS.HE11Beam(a, f_coll, f_foc)
window = TS.PhysicalMaskWindow(holex=-d, holey=-d, holediam=0.5e-3,
                               zmask=f_foc, apod=:tanh)

setup_args = (; λ0, τfwhm, energy, thickness, material,
                mask_diam, mask_spacing, λlims, beam, window,
                apod=:supergauss, apod_param=16,
                trange=110e-15, raman=false, store_window=false,
                R=R_GRID, N=N_GRID,
                beamlet_profile=true,
                beamlet_profile_nr=parse(Int, get(ENV, "BEAMLET_NR", "64")),
                beamlet_profile_rmax_units=parse(Float64, get(ENV, "BEAMLET_RMAX", "6")))

OUT = get(ENV, "BEAMLET_OUT",
          "beamlet_profile_04_gap1000um_1fs_40umUVFS_N$(N_GRID).h5")

@printf("04 gate-beamlet focal profile — setup only, no propagation\n")
@printf("  %.1f fs, %.2f µJ, %d nm into %d µm %s; gap %.1f mm, holes %.1f mm, f %.0f mm\n",
        τfwhm*1e15, energy*1e6, round(Int, λ0*1e9), round(Int, thickness*1e6), material,
        GAP*1e3, mask_diam*1e3, f_foc*1e3)
@printf("  transverse %d×%d, R %.0f µm; λlims %.0f–%.0f nm, trange %.0f fs\n",
        N_GRID, N_GRID, R_GRID*1e6, λlims[1]*1e9, λlims[2]*1e9, 110.0)
@printf("  gate 1 hole at (%.2f, %.2f) mm; λf/D at λ0 = %.1f µm\n",
        d*1e3, d*1e3, λ0*f_foc/mask_diam*1e6)
flush(stdout)

t0 = time()
setup = TS.build_setup(; setup_args...)
@printf("  build_setup: %.1f s, host peak %.1f GiB\n", time()-t0, Sys.maxrss()/2^30)

cg = setup.combined_grid
haskey(cg, "beamlet_r") || error("no beamlet profile in the setup — is beamlet_profile on?")
g = setup.grid
r  = cg["beamlet_r"]
Er = cg["Eω_beamlet_r_re"] .+ 1im .* cg["Eω_beamlet_r_im"]

# --- the a-priori check, reported rather than assumed --------------------------
# Integrating |E(ω,r)|² with the r dr Jacobian must reproduce the stored Iω_beamlet.
δx = setup.xygrid.x[2] - setup.xygrid.x[1]
δy = setup.xygrid.y[2] - setup.xygrid.y[1]
trapz(y, x) = sum((y[i]+y[i+1])/2*(x[i+1]-x[i]) for i in 1:length(x)-1)
println("\n  radial closure and λ-scaling (the two properties known a priori):")
println("    λ[nm]   ∫|E|²r dr / Iω_beamlet    FWHM[µm]   FWHM/(λf/D)   azim. asym.")
for λ in (180e-9, 200e-9, 230e-9, 260e-9, 300e-9, 350e-9, 400e-9)
    iω = argmin(abs.(g.ω .- 2π*PhysData.c/λ))
    cg["Iω_beamlet"][iω] > 1e-12*maximum(cg["Iω_beamlet"]) || continue
    closure = 2π*trapz(abs2.(Er[iω, :]) .* r, r)/(δx*δy) / cg["Iω_beamlet"][iω]
    # linear interpolation of the half-maximum crossing: at dr = rmax/(nr-1) the raw
    # index would quantise the width to ~5 µm, which is a fifth of the spot and would
    # hide the λ-scaling this table exists to show
    P = abs2.(Er[iω, :]); half = maximum(P)/2
    ih = findfirst(<(half), P)
    fw = if isnothing(ih) || ih == 1
        NaN
    else
        2*(r[ih-1] + (r[ih]-r[ih-1])*(P[ih-1]-half)/(P[ih-1]-P[ih]))
    end
    @printf("    %5.0f   %8.4f                  %6.1f      %6.3f        %7.4f\n",
            λ*1e9, closure, fw*1e6, fw/(λ*f_foc/mask_diam), cg["beamlet_r_asym"][iω])
end

# --------------------------------------------------------------------- write ----
gitrev(dir) = try
    strip(read(`git -C $dir rev-parse --short HEAD`, String)) catch; "unknown" end

HDF5.h5open(OUT, "w") do f
    # the profile and its geometry
    for k in ("beamlet_r", "Eω_beamlet_r_re", "Eω_beamlet_r_im", "beamlet_r_asym",
              "beamlet_r_which", "beamlet_r_holex", "beamlet_r_holey", "beamlet_r_zmask",
              "beamlet_r_tilt_coefx", "beamlet_r_tilt_coefy",
              "beamlet_r_max_units", "beamlet_r_max_requested")
        f[replace(k, "ω" => "omega")] = cg[k]
    end
    # the axes, and the existing truth this exists to check
    f["omega"]  = collect(g.ω)
    f["lambda"] = [ω > 0 ? 2π*PhysData.c/ω : Inf for ω in g.ω]
    for k in ("Iω_beamlet", "Eω_beamlet_re", "Eω_beamlet_im", "Iω", "Eω_re", "Eω_im",
              "It_beamlet", "It", "t")
        haskey(cg, k) && (f[replace(k, "ω" => "omega")] = cg[k])
    end
    # geometry and provenance, so the file needs no script to interpret
    gg = HDF5.create_group(f, "geometry")
    for (k, v) in ("lambda0" => λ0, "tau_fwhm" => τfwhm, "energy" => energy,
                   "material" => string(material), "thickness" => thickness,
                   "mask_diam" => mask_diam, "mask_spacing" => mask_spacing,
                   "hole_offset_d" => d, "f_foc" => f_foc, "f_coll" => f_coll,
                   "capillary_a" => a, "R" => R_GRID, "N" => N_GRID,
                   "lambda_min" => λlims[1], "lambda_max" => λlims[2],
                   "trange" => 110e-15, "apod" => "supergauss", "apod_param" => 16,
                   "lambda_f_over_D" => λ0*f_foc/mask_diam)
        gg[k] = v
    end
    pg = HDF5.create_group(f, "provenance")
    pg["source"]    = "ModelPNPS/examples/beamlet_profile_04.jl"
    pg["mimics"]    = "FROG_paper_new/04_production_gap1000_kerr_raman.jl (Kerr arm)"
    pg["modelpnps"] = gitrev(dirname(dirname(pathof(ModelPNPS))))
    pg["luna"]      = gitrev(dirname(dirname(pathof(Luna))))
    pg["julia"]     = string(VERSION)
    pg["date"]      = string(Dates.now())
    pg["note"]      = "Radial profile of the GATE-1 beamlet about its centre, which is " *
                      "the FOCUS (the BOXCARS corners are in the mask plane = k-space). " *
                      "The geometric tilt k0 = hole*omega/(c*zmask) has been removed; " *
                      "multiply by exp(+i*omega*(coefx*x + coefy*y)) to restore it."
end

@printf("\n  wrote %s (%.0f kB)\n", OUT, filesize(OUT)/1024)
println("  done $(Dates.now())")
