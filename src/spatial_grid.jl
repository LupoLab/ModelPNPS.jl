# ============================================================================
# Spatial-grid sizing
# ============================================================================

"""
    optimal_spatial_grid(f, mask_diam, mask_spacing, λmin, λmax;
                         n_airy=5, pts_per_lobe=10, safety=1.5,
                         margin=1.1, geometry=:tg) -> (R, N)

Return `(R, N)` for a Luna `FreeGrid(R, N)` chosen so that the spatial grid

1. contains at least `n_airy` Airy diffraction patterns of the longest
   wavelength `λmax` from a mask hole of diameter `mask_diam` focused by a
   lens of focal length `f` (real-space containment), and
2. resolves the Airy pattern at the shortest wavelength `λmin` with at
   least `pts_per_lobe` points across the central lobe (real-space
   resolution), and
3. has a k-space half-extent that comfortably encloses the FWM nonlinear
   k-vectors generated at `λmin` from the outermost mask hole, with a
   `safety` headroom factor (k-space containment).

`N` is rounded up to the next power of 2 for FFT efficiency. Diagnostic
information is printed via `@info`.

# Arguments
- `f`: focal length of the focusing lens [m].
- `mask_diam`: diameter of each mask hole [m].
- `mask_spacing`: edge-to-edge spacing between adjacent mask holes [m].
- `λmin`, `λmax`: shortest and longest wavelengths the simulation must
  represent [m]. These should bracket the input spectrum *and* its FWM
  products.

# Keyword arguments
- `n_airy=5`: number of Airy patterns the grid should contain at `λmax`.
- `pts_per_lobe=10`: real-space samples across the central Airy lobe at
  `λmin`.
- `safety=1.5`: multiplier on the required nonlinear k-vector envelope to
  guard against aliasing.
- `margin=1.1`: multiplier on the resolved grid size before rounding up to the
  next even 2,3,5-smooth FFT size (guards the containment against grid
  quantisation).
- `geometry=:tg`: the beam layout the k-space bound (3.) is computed for. `:tg`
  is the four-hole boxcar, whose χ⁽³⁾ combinations reach three times the hole
  offset; `:sd` is the two-hole self-diffraction layout, whose `2k₁ - k₂` signal
  sits one further slot out along the same axis. See the comment on `x_max` in
  the implementation for the two bounds.
"""
function optimal_spatial_grid(
        f, mask_diam, mask_spacing, λmin, λmax;
        n_airy = 5, pts_per_lobe = 10, safety = 1.5, margin = 1.1,
        geometry::Symbol = :tg
    )
    geometry in (:tg, :sd) || throw(
        ArgumentError("geometry must be :tg or :sd; got :$geometry")
    )
    # Outermost extent of the nonlinear k-content, measured in the mask plane.
    #
    # :tg  four-hole boxcar. Holes at (+-d, +-d) with d = spacing/2 + diam/2, so
    #      the outermost edge is spacing/2 + diam, and chi(3) combinations reach
    #      three times the hole offset (the 3x below).
    # :sd  two holes on ONE axis at +-s/2, s = spacing + diam. Self-diffraction
    #      puts the signal at 2k_1 - k_2, i.e. at 3s/2 from the axis — one
    #      further slot out than the beams themselves — and that, plus a beam
    #      radius, is the true bound. It is quoted directly rather than as 3x
    #      an inner radius, so no extra factor of three is applied.
    x_max = if geometry === :sd
        1.5 * (mask_spacing + mask_diam) + mask_diam / 2
    else
        mask_spacing / 2 + mask_diam
    end
    r_airy_max = 1.22 * λmax * f / mask_diam
    r_airy_min = 1.22 * λmin * f / mask_diam

    # Real-space containment: half-width R must hold n_airy Airy patterns at λmax.
    R_min = n_airy * r_airy_max

    # Real-space resolution: dx must resolve the Airy lobe at λmin.
    dx_max = r_airy_min / pts_per_lobe
    N_from_realspace = 2 * R_min / dx_max
    @info "N from real-space resolution" N_from_realspace

    # k-space containment: kmax must exceed the largest FWM nonlinear k-vector
    # (≈ 3 × outermost-hole k, with a safety factor).
    k_NL_max = safety * (geometry === :sd ? 1 : 3) * 2π * x_max / (λmin * f)
    N_from_kspace = 2 * R_min * k_NL_max / π
    @info "N from k-space containment" N_from_kspace

    # 2,3,5-smooth FFT sizes are as fast as powers of two for FFTW but track the
    # requirement much more closely: nextpow(2, ...) rounded 576 up to 1024 (3.2× the
    # memory and FFT work of 576), whereas nextprod with a 10% margin gives 640.
    # `margin` guards the containment against grid quantisation on top of `safety`.
    # N must be even: the grid layout (x = (n - N/2)δx) and the centre-pixel signal
    # extraction both assume it.
    Nmin = ceil(Int, margin * max(N_from_realspace, N_from_kspace))
    N = nextprod([2, 3, 5], Nmin)
    while isodd(N)
        N = nextprod([2, 3, 5], N + 1)
    end
    R = R_min

    dx = 2R / N
    dk = π / R
    kmax = π * N / (2R)
    k_hole_width_min = 2π * mask_diam / (λmax * f)   # narrowest hole in k-space
    n_hole = k_hole_width_min / dk
    @info "Spatial grid parameters" R_µm = R * 1.0e6 N dx_µm = dx * 1.0e6
    @info "Real-space check" airy_min_pts = r_airy_min / dx airy_max_pts = r_airy_max / dx
    @info "k-space check" k_NL_max kmax margin = kmax / k_NL_max pts_per_hole = n_hole

    return R, N
end
