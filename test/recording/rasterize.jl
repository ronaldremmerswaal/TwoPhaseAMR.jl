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

# The mesh schedule and physical dual-cell geometry are independent of the
# snapshot's face-tile lookup. Indices here are global within each schedule.
function _expected_face_references(mesh, axis, point)
    found = NamedTuple{(:boundary, :index), Tuple{Bool, Int}}[]
    for (index, face) in pairs(mesh.faces)
        face.axis == axis && _dual_cell_contains(mesh, face, axis, point) &&
            push!(found, (; boundary = false, index))
    end
    for (index, face) in pairs(mesh.boundary_faces)
        face.axis == axis && _dual_cell_contains(mesh, face, axis, point) &&
            push!(found, (; boundary = true, index))
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

@testset "Face probes agree with physical staggered dual cells" begin
    # Sample off tile boundaries on a mixed-level mesh. The mesh's global
    # face schedule, rather than its raster tiles, supplies the expected IDs.
    rect = (0.0, 1.0, 0.0, 1.0)
    size = (96, 160)
    for periodic in ((false, false), (true, false), (false, true), (true, true))
        mesh = _rasterize_test_mesh(periodic)
        interfaces = zeros(Int, 2)
        boundaries = zeros(Int, 2, 2)
        for (axis, placement) in ((1, Face{1}()), (2, Face{2}()))
            field = FaceField(mesh, placement, (x, y) -> 2x - 3y)
            registry = VisualizationFieldRegistry()
            register_field!(registry, :face, field)
            frame = TwoPhaseAMR._registry_frame(registry)
            image = rasterize(frame, :face, rect, size)
            interior_ids = findall(face -> face.axis == axis, mesh.faces)
            boundary_ids = findall(face -> face.axis == axis, mesh.boundary_faces)
            interior_local = Dict(global_id => local_id for (local_id, global_id) in pairs(interior_ids))
            boundary_local = Dict(global_id => local_id for (local_id, global_id) in pairs(boundary_ids))

            for j in ((1:7:155)..., 160), i in 1:5:96
                point = _pixel_centre(rect, size, i, j)
                expected = _expected_face_references(mesh, axis, point)
                @test length(expected) == 1
                reference = locate_face(frame, point, axis)
                @test reference isa FaceRef
                if length(expected) == 1 && reference isa FaceRef
                    expected_ref = only(expected)
                    @test (boundary = reference.boundary, index = reference.index) == expected_ref
                    @test locate_face(frame.topology, point, axis) == reference
                    if (boundary = reference.boundary, index = reference.index) == expected_ref
                        if reference.boundary
                            face = mesh.boundary_faces[reference.index]
                            boundaries[axis, face.side < 0 ? 1 : 2] += 1
                            @test image[i, j] == Float32(field.boundary[boundary_local[reference.index]])
                        else
                            face = mesh.faces[reference.index]
                            interfaces[axis] +=
                                mesh.blocks[face.minus.block].key.level !=
                                mesh.blocks[face.plus.block].key.level
                            @test image[i, j] == Float32(field.interior[interior_local[reference.index]])
                        end
                    end
                    for direction in 1:2, shift in (-1, 1)
                        shifted = ntuple(d -> point[d] + (d == direction ? shift : 0), 2)
                        if periodic[direction]
                            @test locate_face(frame, shifted, axis) == reference
                        else
                            @test isnothing(locate_face(frame, shifted, axis))
                        end
                    end
                end
            end
            @test_throws ArgumentError locate_face(frame, (0.3, 0.4), 0)
            @test_throws ArgumentError locate_face(frame.topology, (0.3, 0.4), 3)
        end
        @test all(>(0), interfaces)
        for axis in 1:2
            @test periodic[axis] ? all(iszero, boundaries[axis, :]) :
                all(>(0), boundaries[axis, :])
        end
    end

    # Wrapping uses the domain origin and extent, not unit-box coordinates.
    shifted_mesh = BlockTreeMesh(
        block_size = (4, 2), origin = (1.25, -0.75), extent = (1.5, 0.75),
        periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0,
    )
    shifted_fields = VisualizationFieldRegistry()
    register_field!(shifted_fields, :cell, CellField(shifted_mesh))
    shifted_frame = TwoPhaseAMR._registry_frame(shifted_fields)
    shifted_rect = (1.25, 2.75, -0.75, 0.0)
    for axis in 1:2, j in (1, 10, 82, 160), i in (1, 10, 53, 96)
        point = _pixel_centre(shifted_rect, size, i, j)
        expected = _expected_face_references(shifted_mesh, axis, point)
        @test length(expected) == 1
        reference = locate_face(shifted_frame, point, axis)
        @test reference isa FaceRef
        if length(expected) == 1 && reference isa FaceRef
            @test (boundary = reference.boundary, index = reference.index) == only(expected)
            for direction in 1:2, shift in (-2, -1, 1, 2)
                translated = ntuple(
                    d -> point[d] + (d == direction ? shift * shifted_mesh.extent[d] : 0), 2,
                )
                @test locate_face(shifted_frame, translated, axis) == reference
            end
        end
    end

    # A probe on an old frame keeps its original global references after AMR.
    mesh = BlockTreeMesh(block_size = (4, 4), max_level = 1, periodic = (false, false))
    field = CellField(mesh)
    registry = VisualizationFieldRegistry()
    register_field!(registry, :cell, field)
    old = TwoPhaseAMR._registry_frame(registry)
    for axis in 1:2
        tie = axis == 1 ? (0.375, 0.3) : (0.3, 0.375)
        below = ntuple(d -> tie[d] - (d == axis ? 1 / 1024 : 0), 2)
        above = ntuple(d -> tie[d] + (d == axis ? 1 / 1024 : 0), 2)
        @test locate_face(old, tie, axis) == locate_face(old, above, axis)
        @test locate_face(old, tie, axis) != locate_face(old, below, axis)
    end
    @test isnothing(locate_face(old, (1.0, 0.3), 1))
    @test isnothing(locate_face(old, (0.3, 1.0), 2))
    point = (0.37, 0.61)
    old_refs = [locate_face(old, point, axis) for axis in 1:2]
    @test all(ref -> ref isa FaceRef, old_refs)
    old_expected = [only(_expected_face_references(mesh, axis, point)) for axis in 1:2]
    @test old_refs == [FaceRef(ref.boundary, ref.index) for ref in old_expected]
    @test adapt!(AMRWorkspace(mesh, field), [:refine]; step = 1, force = true).changed
    @test [locate_face(old, point, axis) for axis in 1:2] == old_refs
    fresh = TwoPhaseAMR._registry_frame(registry)
    fresh_expected = [only(_expected_face_references(mesh, axis, point)) for axis in 1:2]
    fresh_refs = [locate_face(fresh, point, axis) for axis in 1:2]
    @test all(ref -> ref isa FaceRef, fresh_refs)
    @test fresh_refs == [FaceRef(ref.boundary, ref.index) for ref in fresh_expected]
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
