using Test, TwoPhaseAMR
ls_values(f) = reduce(vcat, [vec(collect(interior(f, b))) for b in 1:nblocks(f.mesh)])

@testset "Default right-angle wall contact" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (false, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.43; lipschitz = 1,
    )
    @test interface.contact_angles == ntuple(_ -> pi / 2, 4)
    @test isnothing(interface.embedded_contact_angle)
    @test any(
        TwoPhaseAMR._get(interface.reconstruction, ref) == 5
            for ref in TwoPhaseAMR._cell_references(mesh)
    )

    unconstrained = InterfaceField(
        mesh, (x, y) -> x - 0.43; lipschitz = 1,
        contact_angle = nothing, embedded_contact_angle = nothing,
    )
    @test all(isnothing, unconstrained.contact_angles)
    @test isnothing(unconstrained.embedded_contact_angle)
end

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
        # A signed-distance plane passing exactly through mesh vertices used
        # to recurse to maxdepth because roundoff made the touching corner
        # appear strictly positive.
        normal = (0.6, -0.8)
        intercept = -0.1
        touching = InterfaceField(
            mesh,
            (x, y) -> normal[1] * x + normal[2] * y - intercept;
            lipschitz = 1
        )
        reference = InterfaceField(mesh)
        initialize!(reference, normal, intercept)
        @test ls_values(touching) ≈ ls_values(reference) atol = 2.0e-11
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

@testset "Level set tangent to a grid line at a Lipschitz bound" begin
    # A circle that touches a grid line at a point none of the nine samples hits
    # is not certified pure by the Lipschitz test at any depth and is not split
    # into monotone pieces. The acceptance that spends the remaining tolerance
    # used to compare the box area with a tolerance quartered together with it,
    # so it never fired and the cell recursed to `maxdepth` and threw.
    mesh = BlockTreeMesh(block_size = (16, 16))
    circle = (x, y) -> hypot(x - 0.52, y - 0.47) - 0.27  # touches x = 0.25
    reference = InterfaceField(mesh, circle)
    tangent = InterfaceField(mesh, circle; lipschitz = 1)
    @test integrate(tangent) ≈ pi * 0.27^2 atol = 2.0e-12
    @test ls_values(tangent) ≈ ls_values(reference) atol = 2.0e-12
    @test all(0 .<= ls_values(tangent) .<= 1)
    # The contact is accepted by the tolerance, not by the depth limit.
    @test_throws ErrorException InterfaceField(mesh, circle; lipschitz = 1, maxdepth = 14)
    @test_throws ErrorException InterfaceField(mesh, circle; lipschitz = 1, atol = 1.0e-13)

    # The bound the acceptance is based on is the exact disk area inside a box.
    disk_area(lo, hi, sx, sy, r) = TwoPhaseAMR._levelset_disk_area(
        lo[1], hi[1], lo[2], hi[2], sx, sy, r,
    )
    @test disk_area((0.0, 0.0), (1.0, 1.0), 0.0, 0.0, 1.0) ≈ pi / 4 atol = 1.0e-15
    @test disk_area((0.0, 0.0), (1.0, 1.0), 0.0, 0.0, 0.5) ≈ pi / 16 atol = 1.0e-15
    @test disk_area((0.0, 0.0), (2.0, 2.0), 1.0, 0.0, 1.0) ≈ pi / 2 atol = 1.0e-15
    @test disk_area((0.0, 0.0), (2.0, 2.0), 1.0, 1.0, 0.5) ≈ pi / 4 atol = 1.0e-15
    @test disk_area((0.0, 0.0), (2.0, 2.0), 1.0, 1.0, 1.0e3) ≈ 4.0 atol = 1.0e-15
    # Independent high-precision evaluation of the arc cutting two box sides.
    @test disk_area((0.0, 0.0), (1.0, 1.0), 0.25, 0.6, 0.7) ≈
        0.8743536742329306 atol = 1.0e-13
end

@testset "Polygon indicators use exact geometric clipping" begin
    vertices = [
        (0.1, 0.1), (0.8, 0.1), (0.8, 0.3),
        (0.4, 0.3), (0.4, 0.8), (0.1, 0.8),
    ]
    polygon = PolygonIndicator([vertices; first(vertices)])
    @test polygon.vertices == vertices
    @test eltype(polygon) === Float64

    meshes = (
        BlockTreeMesh(block_size = (8, 8)),
        BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8)),
        BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    for mesh in meshes
        interface = InterfaceField(mesh, polygon)
        reversed = InterfaceField(mesh, PolygonIndicator(reverse(vertices)))
        @test integrate(interface) ≈ 0.29 atol = 128eps()
        @test ls_values(interface) ≈ ls_values(reversed) atol = 128eps()
        @test all(0 .<= ls_values(interface) .<= 1)
    end

    mesh = BlockTreeMesh(block_size = (16, 16))
    geometry = GeometryField(mesh, (x, y) -> y - 0.2; lipschitz = 1)
    rectangle = PolygonIndicator(
        [(0.2, 0.1), (0.8, 0.1), (0.8, 0.6), (0.2, 0.6)],
    )
    cut = InterfaceField(mesh, rectangle; geometry)
    @test integrate(cut) ≈ 0.24 atol = 256eps()

    mesh32 = BlockTreeMesh(
        block_size = (8, 8), origin = (0.0f0, 0.0f0),
        extent = (1.0f0, 1.0f0),
    )
    polygon32 = PolygonIndicator(
        [(0.1f0, 0.1f0), (0.8f0, 0.1f0), (0.4f0, 0.8f0)],
    )
    @test eltype(polygon32) === Float32
    @test eltype(InterfaceField(mesh32, polygon32)) === Float32

    @test_throws ArgumentError PolygonIndicator([(0.0, 0.0), (1.0, 0.0)])
    @test_throws ArgumentError PolygonIndicator(
        [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)],
    )
end

@testset "Level-set pure cells are exactly pure" begin
    # The non-binary domain height makes area/volume roundoff visible without
    # endpoint snapping. Pure cells must not become spurious PLIC cells.
    mesh = BlockTreeMesh(
        block_size = (24, 64), origin = (0.0, -1.5),
        extent = (1.0, 3.0), periodic = (true, false)
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.2 - 0.01cospi(2x);
        lipschitz = 1.01
    )
    mixed = [
        TwoPhaseAMR._get(interface.fraction, ref)
            for ref in TwoPhaseAMR._cell_references(mesh)
            if 0 < TwoPhaseAMR._get(interface.fraction, ref) < 1
    ]
    @test 24 <= length(mixed) <= 28
    @test all(
        TwoPhaseAMR._get(interface.reconstruction, ref) == 0
            for ref in TwoPhaseAMR._cell_references(mesh)
            if TwoPhaseAMR._get(interface.fraction, ref) in (0.0, 1.0)
    )
end


@testset "interface reconstruction is independent of base-root partitioning" begin
    one_root = BlockTreeMesh(
        base_mesh = (32, 32), block_size = (32, 32),
        periodic = (true, true),
    )
    many_roots = BlockTreeMesh(
        base_mesh = (32, 32), block_size = (8, 8),
        periodic = (true, true),
    )
    levelset = (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27
    first = InterfaceField(one_root, levelset; lipschitz = 1)
    second = InterfaceField(many_roots, levelset; lipschitz = 1)

    function reconstruction_by_cell(interface)
        mesh = interface.mesh
        nx, ny = mesh.block_size
        return Dict(
            cellcenter(mesh, block, i, j) => (
                TwoPhaseAMR._get(interface.fraction, TwoPhaseAMR.CellRef(block, i, j)),
                TwoPhaseAMR._get(interface.normal_x, TwoPhaseAMR.CellRef(block, i, j)),
                TwoPhaseAMR._get(interface.normal_y, TwoPhaseAMR.CellRef(block, i, j)),
                TwoPhaseAMR._get(interface.curvature, TwoPhaseAMR.CellRef(block, i, j)),
            )
                for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        )
    end
    first_values = reconstruction_by_cell(first)
    second_values = reconstruction_by_cell(second)
    @test keys(first_values) == keys(second_values)
    @test all(
        all(
            (isnan(a) && isnan(b)) || isapprox(a, b; atol = 2.0e-12, rtol = 2.0e-12)
                for (a, b) in zip(first_values[key], second_values[key])
        )
            for key in keys(first_values)
    )
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
