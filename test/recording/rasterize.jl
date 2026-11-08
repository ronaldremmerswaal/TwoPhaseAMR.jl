using Test, TwoPhaseAMR

_pixel_centre(rect, size, i, j) = (
    rect[1] + (i - 0.5) * (rect[2] - rect[1]) / size[1],
    rect[3] + (j - 0.5) * (rect[4] - rect[3]) / size[2],
)

function _rasterize_test_mesh(periodic)
    return BlockTreeMesh(
        block_size = (4, 2), periodic = periodic, max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 1)
    )
end

# Staggered dual cell of a face: half cells on both sides along the normal,
# split at a periodic seam, and the face extent along the tangent.
function _dual_cell_contains(mesh, face, axis, point)
    tangent = 3 - axis
    abs(point[tangent] - face.center[tangent]) < face.area / 2 || return false
    if face isa TwoPhaseAMR.BoundaryFace
        half = mesh.blocks[face.cell.block].spacing[axis] / 2
        lower = face.side < 0 ? face.center[axis] : face.center[axis] - half
        return lower < point[axis] < lower + half
    end
    lower = face.center[axis] - mesh.blocks[face.minus.block].spacing[axis] / 2
    upper = face.center[axis] + mesh.blocks[face.plus.block].spacing[axis] / 2
    coordinate = point[axis]
    if mesh.periodic[axis]
        extent = mesh.extent[axis]
        coordinate < lower && (coordinate += extent)
        coordinate > upper && (coordinate -= extent)
    end
    return lower < coordinate < upper
end

function _expected_face_value(mesh, field::FaceField{Axis}, point) where {Axis}
    found = Float32[]
    index = 0
    for face in mesh.faces
        face.axis == Axis || continue
        index += 1
        _dual_cell_contains(mesh, face, Axis, point) &&
            push!(found, field.interior[index])
    end
    index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        index += 1
        _dual_cell_contains(mesh, face, Axis, point) &&
            push!(found, field.boundary[index])
    end
    return found
end

@testset "Rasterized cell fields equal the frame values at pixel centres" begin
    for periodic in ((true, true), (false, false))
        mesh = _rasterize_test_mesh(periodic)
        field = CellField(mesh)
        initialize!(field, (x, y) -> sinpi(3x) + 2y^2)
        registry = VisualizationFieldRegistry()
        register_field!(registry, :field, field)
        frame = TwoPhaseAMR._registry_frame(registry)
        rect = (-0.1, 1.1, -0.05, 1.05)
        size = (101, 83)
        image = rasterize(frame, :field, rect, size)
        @test Base.size(image) == size
        @test eltype(image) == Float32
        for j in 1:size[2], i in 1:size[1]
            point = _pixel_centre(rect, size, i, j)
            inside = all(axis -> 0 <= point[axis] < 1, 1:2)
            if inside
                cell = locate(frame.topology, point)
                index = TwoPhaseAMR._cell_index(cell, 4, 2)
                @test image[i, j] == Float32(frame[:field][index])
            else
                @test isnan(image[i, j])
            end
        end
    end
    mesh = _rasterize_test_mesh((false, false))
    registry = VisualizationFieldRegistry()
    register_field!(registry, :field, CellField(mesh))
    frame = TwoPhaseAMR._registry_frame(registry)
    @test isnothing(locate(frame.topology, (1.5, 0.5)))
    @test locate(frame, (0.1, 0.9)) == locate(frame.topology, (0.1, 0.9))
    @test_throws ArgumentError rasterize(frame, :field, (0, 1, 0, 1), (8, 8); phase_rendering = :blend)
    @test_throws ArgumentError rasterize(frame, :field, (0, 1, 0, 1), (0, 8))
    @test_throws ArgumentError rasterize(frame, :field, (0, 0, 0, 1), (8, 8))
    @test_throws ArgumentError rasterize(frame, :missing, (0, 1, 0, 1), (8, 8))
end

@testset "Face fields fill their staggered dual cells" begin
    # Coarse–fine interfaces, boundary half cells, and periodic seams.
    for periodic in ((true, true), (false, false), (true, false))
        mesh = _rasterize_test_mesh(periodic)
        for (placement, sample) in ((Face{1}(), (x, y) -> 2x - y), (Face{2}(), (x, y) -> x + 3y))
            field = FaceField(mesh, placement, sample)
            registry = VisualizationFieldRegistry()
            register_field!(registry, :face, field)
            frame = TwoPhaseAMR._registry_frame(registry)
            @test frame[:face] == visualization_values(field)
            rect = (0.0, 1.0, 0.0, 1.0)
            # Pixel centres (2i - 1)/(2n) with n = 3·2⁵ or 5·2⁵ never fall on
            # dyadic tile edges, which are multiples of 2⁻⁵ here.
            size = (96, 160)
            image = rasterize(frame, :face, rect, size)
            @test !any(isnan, image)
            matches = 0
            for j in 1:size[2], i in 1:size[1]
                expected = _expected_face_value(mesh, field, _pixel_centre(rect, size, i, j))
                matches += length(expected) == 1 && image[i, j] == only(expected)
            end
            @test matches == prod(size)
        end
    end
end

@testset "Sharp rendering classifies pixels by the PLIC plane" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 0)
    interface = InterfaceField(mesh, (x, y) -> x - 0.55; lipschitz = 1)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :scalar, PhaseScalar(interface; liquid = 1, gas = 2))
    register_field!(registry, :face, PhaseFaceScalar{2}(interface; liquid = 1, gas = 2))
    frame = TwoPhaseAMR._registry_frame(registry)
    @test length(only(frame.phases).mixed) == 8
    rect = (0.0, 1.0, 0.0, 1.0)
    size = (199, 13)
    for name in (:scalar, :face)
        sharp = rasterize(frame, name, rect, size; phase_rendering = :plic)
        majority = rasterize(frame, name, rect, size; phase_rendering = :cell)
        for j in 1:size[2], i in 1:size[1]
            x, _ = _pixel_centre(rect, size, i, j)
            @test sharp[i, j] == (x < 0.55 ? 1 : 2)
            # The mixed column holds 40 % liquid, so its majority is gas.
            @test majority[i, j] == (x < 0.5 ? 1 : 2)
        end
    end
end

@testset "Cameras are isotropic and fill the panel" begin
    domain = (0.0, 2.0, 0.0, 1.0)
    for size in ((800, 400), (400, 400), (1000, 300))
        camera = fit_camera(domain, size)
        xmin, xmax, ymin, ymax = visible_rect(camera, size)
        @test (xmax - xmin) / size[1] ≈ (ymax - ymin) / size[2] ≈ camera.scale
        @test xmin <= 0 && xmax >= 2 && ymin <= 0 && ymax >= 1
        @test (xmax - xmin ≈ 2) || (ymax - ymin ≈ 1)
        @test camera.center == (1.0, 0.5)

        box = (0.3, 0.5, 0.2, 0.25)
        zoomed = zoom_to_box(camera, box, size)
        zx0, zx1, zy0, zy1 = visible_rect(zoomed, size)
        ε = 1.0e-12
        @test zx0 <= 0.3 + ε && zx1 >= 0.5 - ε && zy0 <= 0.2 + ε && zy1 >= 0.25 - ε
        @test (zx1 - zx0 ≈ 0.2) || (zy1 - zy0 ≈ 0.05)
        @test zoom_to_box(camera, (0.3, 0.3, 0.2, 0.25), size) === camera

        anchor = (0.25size[1], 0.7size[2])
        world(camera) = (
            visible_rect(camera, size)[1] + anchor[1] * camera.scale,
            visible_rect(camera, size)[3] + anchor[2] * camera.scale,
        )
        closer = zoom(camera, 2.5, anchor, size)
        @test closer.scale ≈ camera.scale / 2.5
        @test all(world(closer) .≈ world(camera))
    end
    mesh = BlockTreeMesh(block_size = (4, 4), extent = (2.0, 1.0), base_mesh = (8, 4))
    @test fit_camera(mesh, (100, 50)).scale ≈ 0.02
    @test_throws ArgumentError ViewCamera((0.0, 0.0), 0.0)
    @test_throws ArgumentError visible_rect(ViewCamera((0.0, 0.0), 1.0), (0, 10))
    @test_throws ArgumentError zoom(ViewCamera((0.0, 0.0), 1.0), -1, (0, 0), (10, 10))
end
