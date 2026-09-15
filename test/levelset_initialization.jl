using Test, TwoPhaseAMR
ls_values(f) = reduce(vcat, [vec(collect(interior(f, b))) for b in 1:nblocks(f.mesh)])
@testset "Level-set cell fractions on uniform, mixed and AMR meshes" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (12, 12), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh)
        if kind == 3
            amr = AMRWorkspace(mesh, f)
            adapt!(amr, [:refine]; step = 1, force = true)
            adapt!(amr, [:refine, :keep, :keep, :keep]; step = 2, force = true)
        end
        for normal in ((1.0, 0.0), (0.3, 0.7), (-0.8, 0.2))
            intercept = sum(normal) / 2 + 0.037
            reference = InterfaceField(mesh);initialize!(reference, normal, intercept)
            phi = (x, y) -> normal[1] * x + normal[2] * y - intercept
            initialize!(f, phi)
            @test ls_values(f) ≈ ls_values(reference) atol = 2.0e-11
            scaled = InterfaceField(mesh, (x, y) -> 1.0e6phi(x, y))
            @test ls_values(scaled) ≈ ls_values(f) atol = 2.0e-11
        end
        circle = (x, y) -> hypot(x - 0.47, y - 0.51) - 0.21
        initialize!(f, circle; lipschitz = 1)
        @test integrate(f) ≈ pi * 0.21^2 atol = 2.0e-11
        complement = InterfaceField(mesh, (x, y) -> -circle(x, y); lipschitz = 1)
        @test ls_values(f) + ls_values(complement) ≈ ones(ncells(mesh)) atol = 2.0e-11
        ellipse = InterfaceField(mesh, (x, y) -> ((x - 0.5) / 0.27)^2 + ((y - 0.5) / 0.17)^2 - 1)
        @test integrate(ellipse) ≈ pi * 0.27 * 0.17 atol = 2.0e-11
        # Smooth graph with a known exact integral, touching physical domain edges.
        wave = InterfaceField(mesh, (x, y) -> y - 0.5 - 0.1cospi(2x))
        @test integrate(wave) ≈ 0.5 atol = 2.0e-11
        @test all(0 .<= ls_values(f) .<= 1)
    end
end
@testset "Interior components and initialization failures" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    # This droplet misses all corner, edge-midpoint and centre probes of its cell.
    phi = (x, y) -> hypot(x - 0.037, y - 0.061) - 0.009
    f = InterfaceField(mesh, phi; lipschitz = 1, maxdepth = 20)
    @test integrate(f) ≈ pi * 0.009^2 atol = 2.0e-12
    g = InterfaceField(mesh, phi; min_depth = 4)
    @test integrate(g) ≈ integrate(f) atol = 2.0e-12
    initial = ls_values(f)
    @test_throws ArgumentError initialize!(f, (x, y) -> NaN)
    @test ls_values(f) == initial
    @test_throws ArgumentError InterfaceField(mesh, (x, y) -> x < 0.5)
    @test_throws ArgumentError InterfaceField(mesh, phi; atol = 0)
    @test_throws ArgumentError InterfaceField(mesh, phi; lipschitz = -1)
    @test_throws ArgumentError InterfaceField(mesh, phi; min_depth = 3, maxdepth = 2)
    @test_throws ErrorException initialize!(f, phi; lipschitz = 1, maxdepth = 0)
    @test ls_values(f) == initial
    @test integrate(InterfaceField(mesh, (x, y) -> -1.0)) == 1.0
    @test integrate(InterfaceField(mesh, (x, y) -> 1.0)) == 0.0
end
