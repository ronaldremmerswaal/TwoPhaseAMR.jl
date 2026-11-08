# Code check 01: the acute/obtuse contact curvature (4.19) against the package.
#
# Label:  eq:interface:contact-acute
# Claim:  04_interface.tex:473-499 (eq. 4.19) and the checks paragraph 04_interface.tex:524-534.
# Code:   src/interface/height_functions.jl:733-804 (_contact_curvature, acute branch 780-799),
#         _wall_height 733-746.
# Method: (A) pipeline: sessile drops at random acute/obtuse angles, radii, offsets, with isotropic
#             and anisotropic cells; the curvature stored in every contact cell equals the
#             independent evaluation of (4.19) (rationalised) and of the thesis' root A^- written
#             without rationalisation (BigFloat), clipped as in (4.17);
#         (B) the branch itself with random admissible column heights (anisotropic cell widths,
#             both phases s = +-1): `_contact_curvature` on a modified height array equals (4.19);
#             inadmissible heights (h2 <= h1) are rejected.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-acute/scripts/code_01_acute_vs_package.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random
const AMR = TwoPhaseAMR
setprecision(BigFloat, 256)

# --- the formulas of the source, independent of the package --------------------------------

"""(4.19) as printed: returns (a2, a1, kappa) for heights h1, h2 (physical), width w, wedge slope T = |tan theta|, orientation s."""
function acute_doc(h1, h2, w, T, s)
    a2 = ((h2 - h1)^2 - w^2 * T^2) / (w^2 * (h1 + h2 + sqrt(4h1 * h2 + w^2 * T^2)))
    a1 = (h2 - h1) / w - a2 * w
    kappa = -2s * a2 / (1 + (a1 - 2a2 * w)^2)^(3 / 2)
    return a2, a1, kappa
end

"""The thesis' root A^- written without rationalisation, in 256-bit arithmetic, slope evaluated at x = -w."""
function acute_thesis_big(h1, h2, w, T, s)
    p, q, ww, TT = BigFloat(h1), BigFloat(h2), BigFloat(w), BigFloat(T)
    D = 4p * q + ww^2 * TT^2
    A = (p + q - sqrt(D)) / ww^2
    B = (-2p + sqrt(D)) / ww                                   # B^- of the thesis (positive root)
    return Float64(-2s * A / (1 + (B - 2A * ww)^2)^(3 / 2))
end

clip_curvature(k, hx, hy) = clamp(k, -1 / (2max(hx, hy)), 1 / (2max(hx, hy)))

# --- (A) pipeline on sessile drops ----------------------------------------------------------

function sessile(theta, R, cx, hx, hy; nx = 16, ny = 8)
    mesh = AMR.BlockTreeMesh(;
        base_mesh = (nx, ny), block_size = (nx, ny), extent = (nx * hx, ny * hy), periodic = (false, false),
    )
    cy = -R * cos(theta)
    field = AMR.InterfaceField(
        mesh, (x, y) -> hypot(x - cx, y - cy) - R; contact_angle = (nothing, nothing, theta, nothing),
    )
    return field, cy
end

# sum of the phase s over the four cells above the wall in column ii (non-increasing, last empty)
function wall_column(V, ii, s)
    (1 <= ii <= size(V, 1)) || return nothing
    total = 0.0; previous = 1.0
    for k in 1:4
        v = s > 0 ? V[ii, k] : 1 - V[ii, k]
        v <= previous + 64eps() || return nothing
        total += v; previous = v
    end
    previous <= 64eps() || return nothing
    return total
end

@testset "(A) pipeline: contact-cell curvature = (4.19) = thesis root, anisotropic cells included" begin
    rng = MersenneTwister(4711)
    ncells = 0
    worst_doc = worst_big = 0.0
    for trial in 1:60
        acute = rand(rng) < 0.5
        theta = acute ? deg2rad(8 + 36 * rand(rng)) : deg2rad(136 + 36 * rand(rng))      # |theta - pi/2| > pi/4
        hx, hy = trial % 3 == 0 ? (1.0, 1.0) : (0.6 + rand(rng), 0.6 + rand(rng))
        R = (4.5 + 2rand(rng)) * min(hx, hy)
        cx = (5.2 + 5.6 * rand(rng)) * hx
        field, cy = sessile(theta, R, cx, hx, hy)
        V = [AMR._get(field.conditional_fraction, AMR.CellRef(1, i, j)) for i in 1:16, j in 1:8]
        s = cos(theta) > 0 ? 1 : -1
        for i in 1:16
            0 < V[i, 1] < 1 || continue
            AMR._plic_touches_wall(field, AMR.CellRef(1, i, 1), 3) || continue
            side = (i - 0.5) * hx > cx ? 1 : -1                  # right / left contact point
            d = -s * side                                       # away from the contact point (geometry)
            W0 = wall_column(V, i + d, s); W1 = wall_column(V, i + 2d, s)
            (W0 === nothing || W1 === nothing || !(W1 > W0 > 0)) && continue
            h1, h2 = W0 * hy, W1 * hy
            _, _, kdoc = acute_doc(h1, h2, hx, abs(tan(theta)), s)
            kbig = acute_thesis_big(h1, h2, hx, abs(tan(theta)), s)
            kcode = AMR._get(field.curvature, AMR.CellRef(1, i, 1))
            kclip = clip_curvature(kdoc, hx, hy)
            worst_doc = max(worst_doc, abs(kcode - kclip) / max(abs(kclip), 1.0e-3))
            worst_big = max(worst_big, abs(clip_curvature(kbig, hx, hy) - kcode) / max(abs(kclip), 1.0e-3))
            @test isapprox(kcode, kclip; rtol = 1.0e-10, atol = 1.0e-12)
            @test isapprox(kcode, clip_curvature(kbig, hx, hy); rtol = 1.0e-9, atol = 1.0e-12)
            ncells += 1
        end
    end
    println("  (A) $ncells contact cells; max relative deviation from (4.19): $(worst_doc); from the thesis root: $(worst_big)")
    @test ncells >= 80
end

# --- (B) the branch with injected admissible heights ----------------------------------------

function fill_column!(values, ii, j0, H, orientation)
    n = floor(Int, H)
    v = zeros(4)
    v[1:n] .= 1.0
    n < 4 && (v[n + 1] = H - n)
    for k in 1:4
        values[ii, j0 + k - 1] = orientation > 0 ? v[k] : 1 - v[k]
    end
    return nothing
end

@testset "(B) injected column heights: _contact_curvature = (4.19)" begin
    rng = MersenneTwister(99)
    checked = rejected = 0
    for trial in 1:30
        acute = trial % 2 == 0
        theta = acute ? deg2rad(10 + 30 * rand(rng)) : deg2rad(140 + 30 * rand(rng))
        hx, hy = 0.5 + 1.5rand(rng), 0.5 + 1.5rand(rng)
        R = 5.0 * min(hx, hy)
        cx = 8.3 * hx
        field, _ = sessile(theta, R, cx, hx, hy)
        base = AMR._fill_height_values!(field)[1]
        s = cos(theta) > 0 ? 1 : -1
        for i in 1:16
            ref = AMR.CellRef(1, i, 1)
            0 < AMR._get(field.conditional_fraction, ref) < 1 || continue
            AMR._plic_touches_wall(field, ref, 3) || continue
            side = (i - 0.5) * hx > cx ? 1 : -1
            d = -s * side
            for _ in 1:20
                H1 = 0.05 + 1.5rand(rng)
                H2 = rand(rng) < 0.85 ? H1 + 0.02 + 1.4rand(rng) : H1 * rand(rng)           # 15 %: h2 <= h1
                H2 < 2.95 || continue
                values = copy(base)
                fill_column!(values, i + 4 + d, 5, H1, s)
                fill_column!(values, i + 4 + 2d, 5, H2, s)
                kcode = AMR._contact_curvature(field, values, 1, i, 1)
                if H2 > H1
                    _, _, k = acute_doc(H1 * hy, H2 * hy, hx, abs(tan(theta)), s)
                    @test kcode !== nothing && isapprox(kcode, k; rtol = 1.0e-11, atol = 1.0e-13)
                    @test isapprox(kcode, acute_thesis_big(H1 * hy, H2 * hy, hx, abs(tan(theta)), s); rtol = 1.0e-9, atol = 1.0e-12)
                    checked += 1
                else
                    # the branch requires h2 > h1 > 0; the loop then tries the other walls, none has an angle
                    @test kcode === nothing
                    rejected += 1
                end
            end
        end
    end
    println("  (B) $checked admissible and $rejected inadmissible height pairs")
    @test checked > 300 && rejected > 30
end
