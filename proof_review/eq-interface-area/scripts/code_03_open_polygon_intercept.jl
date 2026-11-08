# Code check 03: the plane constant in cut cells (second case of eq. 4.9).
#
# Label:  eq:interface:area (eq. 4.9, cut cells)
# Claim:  04_interface.tex:158-168: in a cut cell |L_c| = C |O_c| (O_c the open polygon of (7.3)),
#         found by 54 bisection steps on the range of n.xi over the vertices of O_c, the area being
#         that of the Sutherland--Hodgman clipped polygon.
# Code:   src/embedded/geometry.jl:84-106 (_open_plane_intercept), geometry_utils.jl (clipping).
# Method: cylinder fixture of the technical reference (8x8 unit cells, centre (4.3,3.6), radius 2.6);
#         for every cut cell and 200 random (C, n): the area of {open polygon, n.xi <= a(C)} computed
#         independently (own clipping, own shoelace, polygon rebuilt from the stored solid plane)
#         equals C*|O_c|; a(C) lies in the projection range of the polygon; pure limits C = 0, 1.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-area/scripts/code_03_open_polygon_intercept.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random
const AMR = TwoPhaseAMR
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

const SQUARE = rect_poly(-0.5, 0.5, -0.5, 0.5)

function cylinder_geometry()
    n = 8
    mesh = AMR.BlockTreeMesh(;
        base_mesh = (n, n), block_size = (n, n), extent = (Float64(n), Float64(n)),
        periodic = (false, false),
    )
    return AMR.GeometryField(mesh, (x, y) -> hypot(x - 4.3, y - 3.6) - 2.6)
end

@testset "cut-cell plane constant: clipped liquid area = C |open polygon|" begin
    geometry = cylinder_geometry()
    rng = MersenneTwister(2026)
    cut_cells = checked = 0
    worst = 0.0
    for j in 1:8, i in 1:8
        ref = AMR.CellRef(1, i, j)
        alpha = AMR._get(geometry.solid.fraction, ref)
        0 < alpha < 1 || continue
        cut_cells += 1
        sn = (AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref))
        sa = AMR._get(geometry.solid.intercept, ref)
        open_poly = clip_halfplane(SQUARE, sn, sa; keep_less = false)
        open_area = shoelace(open_poly)
        @test abs(open_area - AMR._get(geometry.open_fraction, ref)) <= 1.0e-14
        projections(n) = [n[1] * p[1] + n[2] * p[2] for p in open_poly]
        for _ in 1:200
            angle = 2pi * rand(rng)
            scale = 10.0^(randn(rng))
            n = scale .* (cos(angle), sin(angle))
            c = rand(rng)
            a = AMR._open_plane_intercept(geometry, ref, c, n[1], n[2])
            liquid = shoelace(clip_halfplane(open_poly, n, a))
            worst = max(worst, abs(liquid - c * open_area))
            lo, hi = extrema(projections(n))
            @test lo - 1.0e-14 * scale <= a <= hi + 1.0e-14 * scale
            checked += 1
        end
        # pure limits return the ends of the projection range
        n = (0.6, -0.8)
        lo, hi = extrema(projections(n))
        @test AMR._open_plane_intercept(geometry, ref, 0.0, n[1], n[2]) == lo
        @test AMR._open_plane_intercept(geometry, ref, 1.0, n[1], n[2]) == hi
    end
    println("  $cut_cells cut cells, $checked random (C, n): max | |L_c| - C |O_c| | = $worst")
    @test cut_cells >= 20
    @test worst <= 2.0e-15
end
