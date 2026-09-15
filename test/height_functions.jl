using Test, TwoPhaseAMR
function circle_fraction(xlo, xhi, ylo, yhi, r)
    max(abs(xlo), abs(xhi))^2 + max(abs(ylo), abs(yhi))^2 <= r * r && return 1.0
    minx = xlo > 0 ? xlo : xhi < 0 ? -xhi : 0.0
    miny = ylo > 0 ? ylo : yhi < 0 ? -yhi : 0.0
    minx^2 + miny^2 >= r * r && return 0.0
    lo, hi = max(xlo, -r), min(xhi, r)
    lo >= hi && return 0.0
    cuts = [lo, hi]
    for y in (ylo, yhi)
        if abs(y) < r
            root = sqrt(r * r - y * y)
            for x in (-root, root)
                lo < x < hi && push!(cuts, x)
            end
        end
    end
    sort!(cuts)
    primitive(x) = (x * sqrt(max(0.0, r * r - x * x)) + r * r * asin(clamp(x / r, -1, 1))) / 2
    area = 0.0
    for (a, b) in zip(cuts[1:(end - 1)], cuts[2:end])
        mid = (a + b) / 2; h = sqrt(max(0.0, r * r - mid * mid))
        min(yhi, h) > max(ylo, -h) || continue
        upper = yhi < h ? yhi * (b - a) : primitive(b) - primitive(a)
        lower = ylo > -h ? ylo * (b - a) : -primitive(b) + primitive(a)
        area += upper - lower
    end
    return clamp(area / ((xhi - xlo) * (yhi - ylo)), 0.0, 1.0)
end
@testset "WY fallback exactly reconstructs linear interfaces" begin
    for angle in range(0.02, 2pi - 0.02; length = 60),c in (0.03, 0.4, 0.6, 0.97)
        n = (cos(angle), sin(angle));a = TwoPhaseAMR._plane_intercept(c, n...)
        v = [TwoPhaseAMR._plane_fraction(n..., a, i - 0.5, i + 0.5, j - 0.5, j + 0.5) for i in -1:1,j in -1:1]
        mx, my = TwoPhaseAMR._wy_normal(v, 2, 2)
        norm = hypot(mx, my)
        @test [mx / norm, my / norm] ≈ collect(n) atol = 2.0e-13
    end
end
@testset "LHF planar normals and curvature" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (16, 16), max_level = 2, periodic = (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh)
        if kind == 3
            amr = AMRWorkspace(mesh, f)
            adapt!(amr, [:refine]; step = 1, force = true)
        end
        for angle in (0.1, 0.4, 0.7, 1.2, 2.0, 3.0, 4.0, 5.0)
            n = (cos(angle), sin(angle)); initialize!(f, n, sum(n) * 0.5)
            for r in TwoPhaseAMR._cell_references(mesh)
                c = TwoPhaseAMR._get(f.fraction, r)
                1.0e-10 < c < 1 - 1.0e-10 || continue
                x, y = cellcenter(mesh, r.block, r.i, r.j)
                0.2 < x < 0.8 && 0.2 < y < 0.8 || continue
                hx, hy = mesh.blocks[r.block].spacing
                mx = TwoPhaseAMR._get(f.normal_x, r) / hx;my = TwoPhaseAMR._get(f.normal_y, r) / hy
                norm = hypot(mx, my)
                @test [mx / norm, my / norm] ≈ collect(n) atol = 1.0e-10
                @test TwoPhaseAMR._get(f.reconstruction, r) == 1
                @test abs(TwoPhaseAMR._get(f.curvature, r)) < 1.0e-8
            end
        end
    end
end
@testset "Circular LHF curvature converges" begin
    errors = Float64[]
    for n in (32, 64, 128)
        mesh = BlockTreeMesh(block_size = (n, n), periodic = (false, false))
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2; lipschitz = 1)
        e = Float64[]
        for r in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(f.fraction, r)
            1.0e-8 < c < 1 - 1.0e-8 || continue
            method = TwoPhaseAMR._get(f.reconstruction, r)
            @test method in (1, 2)
            k = TwoPhaseAMR._get(f.curvature, r)
            @test isfinite(k)
            method == 1 && push!(e, abs(k - 5.0))
        end
        push!(errors, maximum(e))
    end
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(rates .>= 2 - 0.3)
end
