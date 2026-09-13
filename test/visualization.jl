@testset "Visualization geometry" begin
    mesh = BlockTreeMesh(
        block_size=(4, 2), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0))
    geometry = visualization_geometry(mesh)

    @test length(geometry.cells) == ncells(mesh)
    @test length(geometry.points) == 4ncells(mesh)
    @test length(geometry.triangles) == 2ncells(mesh)
    @test length(geometry.block_segments) == 8nblocks(mesh)
    @test geometry.cells[1] == CellRef(1, 1, 1)
    @test cell_for_vertex(geometry, 1) == geometry.cells[1]
    @test cell_for_vertex(geometry, 4) == geometry.cells[1]
    @test cell_for_vertex(geometry, 5) == geometry.cells[2]
    @test isnothing(cell_for_vertex(geometry, 0))
    @test isnothing(cell_for_vertex(geometry, length(geometry.points) + 1))

    for (cell_index, cell) in pairs(geometry.cells)
        block = mesh.blocks[cell.block]
        first_vertex = 4cell_index - 3
        @test geometry.points[first_vertex] ==
            (block.origin[1] + (cell.i - 1) * block.spacing[1],
             block.origin[2] + (cell.j - 1) * block.spacing[2])
        @test geometry.triangles[2cell_index - 1] ==
            (first_vertex, first_vertex + 1, first_vertex + 2)
        @test geometry.triangles[2cell_index] ==
            (first_vertex, first_vertex + 2, first_vertex + 3)
    end

    field = CellField(mesh)
    initialize!(field, (x, y) -> 2x - 3y)
    values = visualization_values(field)
    @test length(values) == ncells(mesh)
    for (index, cell) in pairs(geometry.cells)
        x, y = cellcenter(mesh, cell.block, cell.i, cell.j)
        @test values[index] == 2x - 3y
    end

    @test update_visualization!(NoVisualization(), field) isa NoVisualization
    @test isnothing(selected_cell(NoVisualization()))
    @test isnothing(selected_face(NoVisualization()))
end

@testset "Visualization face fields" begin
    mesh = BlockTreeMesh(
        block_size=(4, 2), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 1))

    for (placement, sample) in (
            (Face{1}(), (x, y) -> 2x - y),
            (Face{2}(), (x, y) -> x + 3y))
        field = FaceField(mesh, placement, sample)
        geometry = visualization_geometry(mesh, placement)
        values = visualization_values(field)
        rendered = TwoPhaseAMR._visualization_face_vertex_values(
            values, geometry)

        @test length(geometry.points) == 4length(geometry.faces)
        @test length(geometry.triangles) == 2length(geometry.faces)
        @test maximum(geometry.value_indices) == length(values)
        @test length(rendered) == length(geometry.points)
        @test face_for_vertex(geometry, 1) == geometry.faces[1]
        @test face_for_vertex(geometry, 4) == geometry.faces[1]
        @test face_for_vertex(geometry, 5) == geometry.faces[2]
        @test isnothing(face_for_vertex(geometry, 0))
        @test isnothing(face_for_vertex(
            geometry, length(geometry.points) + 1))

        tile_area = 0.0
        axis = placement isa Face{1} ? 1 : 2
        tangent = 3 - axis
        for (index, reference) in pairs(geometry.faces)
            face = reference.boundary ?
                mesh.boundary_faces[reference.index] :
                mesh.faces[reference.index]
            first_vertex = 4index - 3
            points = geometry.points[first_vertex:(first_vertex + 3)]
            normal_lower, normal_upper =
                extrema(point[axis] for point in points)
            tangent_lower, tangent_upper =
                extrema(point[tangent] for point in points)
            contains_face = normal_lower <= face.center[axis] <= normal_upper
            wrapped_face = mesh.periodic[axis] &&
                face.center[axis] == mesh.origin[axis] &&
                normal_upper == mesh.origin[axis] + mesh.extent[axis]
            @test contains_face || wrapped_face
            @test (tangent_lower + tangent_upper) / 2 ≈
                face.center[tangent]
            @test tangent_upper - tangent_lower ≈ face.area
            value = values[geometry.value_indices[index]]
            @test value == sample(face.center...)
            @test all(==(value),
                      rendered[first_vertex:(first_vertex + 3)])
            @test geometry.triangles[2index - 1] ==
                (first_vertex, first_vertex + 1, first_vertex + 2)
            @test geometry.triangles[2index] ==
                (first_vertex, first_vertex + 2, first_vertex + 3)
            tile_area += (normal_upper - normal_lower) *
                         (tangent_upper - tangent_lower)
        end
        @test tile_area ≈ prod(mesh.extent)
    end

    velocity = FaceVelocity(
        mesh, (x, y, time, axis) ->
            axis == 1 ? 2x - y + time : x + 3y - time;
        time=0.25)
    for placement in (Face{1}(), Face{2}())
        geometry = visualization_geometry(mesh, placement)
        values = visualization_values(velocity, placement)
        for (index, reference) in pairs(geometry.faces)
            face = reference.boundary ?
                mesh.boundary_faces[reference.index] :
                mesh.faces[reference.index]
            expected = face.axis == 1 ?
                2face.center[1] - face.center[2] + 0.25 :
                face.center[1] + 3face.center[2] - 0.25
            @test values[geometry.value_indices[index]] == expected
        end
    end
end

@testset "Derived two-component face visualization fields" begin
    mesh = BlockTreeMesh(
        block_size=(4, 4), periodic=(false, false), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 1 && key.j == 0))
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? x + y : 2x - y)
    geometry = visualization_geometry(mesh)

    magnitude = visualization_values(velocity, CenteredMagnitude())
    vorticity = visualization_values(velocity, Vorticity())
    components = (FaceField{1}(velocity), FaceField{2}(velocity))
    paired_magnitude = visualization_values(components, CenteredMagnitude())
    paired_vorticity = visualization_values(components, Vorticity())
    @test length(magnitude) == ncells(mesh)
    @test length(vorticity) == ncells(mesh)
    @test paired_magnitude == magnitude
    @test paired_vorticity == vorticity
    for (index, cell) in pairs(geometry.cells)
        x, y = cellcenter(mesh, cell.block, cell.i, cell.j)
        @test magnitude[index] ≈ hypot(x + y, 2x - y)
        @test vorticity[index] ≈ 1 atol=2e-13
    end
end

@testset "Visualization field registry" begin
    mesh = BlockTreeMesh(block_size=(4, 2), max_level=1)
    pressure = CellField(mesh)
    xflux = FaceField{1}(mesh)
    yflux = FaceField{2}(mesh)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : -x)
    registry = VisualizationFieldRegistry()

    @test register_field!(
        registry, :pressure, pressure; label="Pressure", units="Pa") === registry
    @test register_field!(registry, :xflux, xflux) === registry
    @test register_field!(
        registry, :velocity_x, velocity, Face{1}();
        label="Velocity x", colormap=:balance) === registry
    @test register_field!(
        registry, :velocity_y, velocity, Face{2}();
        label="Velocity y", colormap=:balance) === registry
    @test register_field!(
        registry, :velocity_magnitude, velocity, CenteredMagnitude();
        label="Velocity magnitude") === registry
    @test register_field!(
        registry, :vorticity, velocity, Vorticity();
        label="Vorticity", colormap=:balance) === registry
    @test register_field!(
        registry, :flux_magnitude, (xflux, yflux), CenteredMagnitude()) === registry
    @test register_field!(
        registry, :flux_vorticity, (xflux, yflux), Vorticity()) === registry
    @test field_names(registry) == [
        :pressure, :xflux, :velocity_x, :velocity_y,
        :velocity_magnitude, :vorticity, :flux_magnitude, :flux_vorticity]
    @test_throws ArgumentError register_field!(registry, :pressure, pressure)

    other_mesh = BlockTreeMesh(block_size=(4, 2), max_level=1)
    @test_throws ArgumentError register_field!(
        registry, :other, CellField(other_mesh))
    @test_throws ArgumentError register_field!(
        VisualizationFieldRegistry(), :invalid_pair,
        (xflux, FaceField{2}(other_mesh)), CenteredMagnitude())
end
