# Code check 01: the package's `_plane_fraction` against exact polygon clipping.
#
# Label:  eq:interface:area
# Claim:  04_interface.tex:136-156 (eq. 4.8): the closed-form area fraction of a rectangle
#         prod_k [r_lo,k, r_hi,k] below the line n.xi <= a, incl. the reflected branch and the
#         fall-back clamp(z/w_>, 0, 1) for w_< <= eps*w_>.
# Code:   src/interface/interface_field.jl:229-243 (_plane_fraction).
# Method: reference = Sutherland--Hodgman clipping of the rectangle in 256-bit BigFloat, divided by
#         the rectangle area (independent of the closed form); random normals of every sign pattern,
#         random rectangles (not only sub-rectangles of the reference cell), plane constants from
#         below the lowest to above the highest corner, anisotropic and near-axis-aligned normals,
#         exact branch points; plus a literal transcription of (4.8) from the .tex.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-area/scripts/code_01_area_vs_reference.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random
const AMR = TwoPhaseAMR
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

setprecision(BigFloat, 256)

# Literal transcription of eq. (4.8) with the definitions of the surrounding text.
function area_tex(n1, n2, a, xlo, xhi, ylo, yhi)
    w1, w2 = abs(n1) * (xhi - xlo), abs(n2) * (yhi - ylo)
    wl, wg = minmax(w1, w2)
    z = a - (min(n1 * xlo, n1 * xhi) + min(n2 * ylo, n2 * yhi))
    C(zz) = zz < wl ? zz^2 / (2wl * wg) : (zz - wl / 2) / wg
    wl <= eps() * wg && return clamp(z / wg, 0.0, 1.0)      # the stated fall-back
    z <= 0 && return 0.0
    z <= (w1 + w2) / 2 && return C(z)
    z < w1 + w2 && return 1 - C(w1 + w2 - z)
    return 1.0
end

reference(n1, n2, a, xlo, xhi, ylo, yhi) = Float64(
    plane_area_reference(BigFloat(n1), BigFloat(n2), BigFloat(a), BigFloat(xlo), BigFloat(xhi), BigFloat(ylo), BigFloat(yhi)),
)

@testset "random draws: closed form = polygon area (rounding bound 8 eps (1 + max(|a|,|shift|)/w))" begin
    rng = MersenneTwister(20261108)
    worst_code = worst_tex = worst_text_vs_code = worst_scaled = 0.0
    count = 0
    for draw in 1:40_000
        # normal: random direction and scale, with a share of near-axis-aligned and 45-degree normals
        kind = draw % 8
        angle = 2pi * rand(rng)
        n1, n2 = cos(angle), sin(angle)
        kind == 0 && (n2 = 1.0e-3 * rand(rng) * 10.0^(-6 * rand(rng)))          # near axis aligned
        kind == 1 && (n1 = sign(n1) * abs(n2))                                  # 45 degrees (w1 = w2 for squares)
        kind == 2 && (n1 = 0.0)                                                 # exactly axis aligned
        s = 10.0^(2 * randn(rng))
        n1, n2 = s * n1, s * n2
        xlo = randn(rng); xhi = xlo + 10.0^(randn(rng))
        ylo = randn(rng); yhi = ylo + 10.0^(randn(rng))
        if draw % 5 == 0                                                        # the reference cell and halves
            xlo, xhi, ylo, yhi = -0.5, 0.5, -0.5, 0.5
            draw % 10 == 0 && (xlo = 0.0; ylo = -0.25)
        end
        shift = min(n1 * xlo, n1 * xhi) + min(n2 * ylo, n2 * yhi)
        w = abs(n1) * (xhi - xlo) + abs(n2) * (yhi - ylo)
        a = shift + w * (1.2 * rand(rng) - 0.1)                                  # from below to above the cell
        ref = reference(n1, n2, a, xlo, xhi, ylo, yhi)
        code = AMR._plane_fraction(n1, n2, a, xlo, xhi, ylo, yhi)
        tex = area_tex(n1, n2, a, xlo, xhi, ylo, yhi)
        # forming z = a - shift from the Float64 inputs loses eps*max(|a|,|shift|) absolutely
        scale = 1 + max(abs(a), abs(shift)) / w
        worst_code = max(worst_code, abs(code - ref))
        worst_tex = max(worst_tex, abs(tex - ref))
        worst_text_vs_code = max(worst_text_vs_code, abs(tex - code))
        worst_scaled = max(worst_scaled, abs(code - ref) / (eps() * scale))
        @test abs(code - ref) <= 8eps() * scale
        @test abs(tex - ref) <= 8eps() * scale
        count += 1
    end
    println(
        "  $count draws: max |code - reference| = $(worst_code), max |(4.8) as printed - reference| = ",
        "$(worst_tex), max |(4.8) - code| = $(worst_text_vs_code); max error / (eps*scale) = $(worst_scaled)"
    )
    @test worst_text_vs_code <= 1.0e-12   # both evaluate the same expressions up to branch ordering
end

@testset "branch points, symmetry and monotonicity" begin
    rng = MersenneTwister(7)
    for _ in 1:2000
        n1, n2 = randn(rng, 2)
        xlo, xhi, ylo, yhi = -0.5, 0.5, -0.5, 0.5
        w1, w2 = abs(n1), abs(n2)
        wl, wg = minmax(w1, w2)
        shift = -(w1 + w2) / 2
        # exact breakpoints z = w_<, z = (w1+w2)/2, z = w_>  (a = z + shift)
        for z in (wl, (w1 + w2) / 2, wg)
            a = z + shift
            ref = reference(n1, n2, a, xlo, xhi, ylo, yhi)
            @test abs(AMR._plane_fraction(n1, n2, a) - ref) <= 2.0e-16 * 4
        end
        # the centre line passes through the centre: F = 1/2 (exactly symmetric)
        @test abs(AMR._plane_fraction(n1, n2, 0.0) - 0.5) <= 4.0e-16
        # symmetry F(a) + F(-a) = 1 (reflection about the centre of the reference cell)
        for a in (0.1, 0.3, 0.77) .* ((w1 + w2) / 2)
            @test abs(AMR._plane_fraction(n1, n2, a) + AMR._plane_fraction(n1, n2, -a) - 1) <= 4.0e-16
        end
        # non-decreasing in a on a fine grid
        values = [AMR._plane_fraction(n1, n2, a) for a in range(-(w1 + w2) / 2 - 0.1, (w1 + w2) / 2 + 0.1; length = 401)]
        @test all(diff(values) .>= -1.0e-15)
        @test values[1] == 0 && values[end] == 1
    end
end

@testset "degenerate normals: w_< <= eps w_> uses clamp(z/w_>, 0, 1)" begin
    for tiny in (0.0, 1.0e-300, 1.0e-20, 0.5 * eps(), eps())
        for a in range(-0.7, 0.7; length = 29)
            code = AMR._plane_fraction(1.0, tiny, a)
            @test abs(code - clamp(a + 0.5 + 0.5 * tiny, 0.0, 1.0)) <= 2.0e-16     # the stated fall-back
            ref = reference(1.0, tiny, a, -0.5, 0.5, -0.5, 0.5)
            @test abs(code - ref) <= eps()      # neglected: w_</(2 w_>) <= eps/2
        end
    end
end
