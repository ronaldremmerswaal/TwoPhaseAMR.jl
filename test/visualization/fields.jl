@testset "Visualization geometry" begin
    mesh = BlockTreeMesh(
        block_size = (4, 2), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0)
    )
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
            (
            block.origin[1] + (cell.i - 1) * block.spacing[1],
            block.origin[2] + (cell.j - 1) * block.spacing[2],
        )
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

function _plic_phase_areas(geometry)
    liquid = 0.0
    gas = 0.0
    for triangle in geometry.triangles
        polygon = [geometry.points[index] for index in triangle]
        area = TwoPhaseAMR._polygon_area(polygon)
        if geometry.liquid[first(triangle)]
            liquid += area
        else
            gas += area
        end
    end
    return liquid, gas
end

@testset "Discontinuous fields use sharp two-phase PLIC geometry" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 0)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.24;
        lipschitz = 1,
    )
    scalar = PhaseScalar(
        interface;
        liquid = (x, y) -> 1 + x + 2y,
        gas = (x, y) -> 4 - 2x + y,
    )
    staggered = PhaseFaceScalar{2}(
        interface;
        liquid = (x, y) -> 2 + x - y,
        gas = (x, y) -> 5 - x + 2y,
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :scalar, scalar)
    register_field!(registry, :staggered_scalar, staggered)
    @test registry.interfaces == [interface]

    topology = TwoPhaseAMR._visualization_topology(mesh)
    cell_source, face_source = (entry.source for entry in registry.entries)
    cell_values = TwoPhaseAMR._source_values(cell_source)
    @test cell_values.liquid == visualization_values(scalar.liquid)
    @test cell_values.gas == visualization_values(scalar.gas)
    cell_geometry = TwoPhaseAMR._phase_visualization_geometry(
        cell_source, topology,
    )
    liquid_area, gas_area = _plic_phase_areas(cell_geometry)
    @test liquid_area ≈ integrate(interface)
    @test liquid_area + gas_area ≈ 1
    cell_colors = TwoPhaseAMR._phase_visualization_vertex_values(
        cell_values, cell_geometry,
    )
    @test all(isfinite, cell_colors)
    @test any(cell_geometry.liquid)
    @test any(!, cell_geometry.liquid)
    @test all(
        value -> value in cell_values.liquid || value in cell_values.gas,
        cell_colors
    )

    face_geometry = TwoPhaseAMR._phase_visualization_geometry(
        face_source, topology,
    )
    face_liquid_area, face_gas_area = _plic_phase_areas(face_geometry)
    @test face_liquid_area + face_gas_area ≈ 1
    @test all(
        isfinite, TwoPhaseAMR._phase_visualization_vertex_values(
            TwoPhaseAMR._source_values(face_source), face_geometry,
        )
    )

    frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    for entry in registry.entries
        @test isequal(frame[entry.name], TwoPhaseAMR._source_values(entry.source))
        @test TwoPhaseAMR._frame_phase_geometry(frame, entry.source) !==
            TwoPhaseAMR._phase_visualization_geometry(entry.source, topology)
    end

    geometry = GeometryField(
        mesh, (x, y) -> y - 0.25; lipschitz = 1,
    )
    cut_interface = InterfaceField(
        mesh, (x, y) -> x - 0.5;
        geometry, lipschitz = 1,
    )
    cut_scalar = PhaseScalar(cut_interface; liquid = 2, gas = 3)
    cut_staggered = PhaseFaceScalar{1}(
        cut_interface; liquid = 2, gas = 3,
    )
    cut_registry = VisualizationFieldRegistry()
    register_field!(cut_registry, :scalar, cut_scalar)
    register_field!(cut_registry, :staggered_scalar, cut_staggered)
    cut_topology = TwoPhaseAMR._visualization_topology(mesh)
    cut_cell_geometry = TwoPhaseAMR._phase_visualization_geometry(
        cut_registry.entries[1].source, cut_topology,
    )
    cut_liquid_area, cut_gas_area = _plic_phase_areas(cut_cell_geometry)
    @test cut_liquid_area + cut_gas_area ≈ integrate(geometry.open_fraction)
    cut_face_geometry = TwoPhaseAMR._phase_visualization_geometry(
        cut_registry.entries[2].source, cut_topology,
    )
    cut_face_liquid, cut_face_gas = _plic_phase_areas(cut_face_geometry)
    @test cut_face_liquid + cut_face_gas < 1

    mixed_mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2, periodic = (true, true),
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    mixed_interface = InterfaceField(
        mixed_mesh, (x, y) -> hypot(x - 0.4, y - 0.55) - 0.21;
        lipschitz = 1,
    )
    mixed_topology = TwoPhaseAMR._visualization_topology(mixed_mesh)
    for axis in (1, 2)
        mixed_scalar = PhaseFaceScalar{axis}(
            mixed_interface; liquid = 1, gas = 2,
        )
        mixed_registry = VisualizationFieldRegistry()
        register_field!(mixed_registry, :scalar, mixed_scalar)
        mixed_geometry = TwoPhaseAMR._phase_visualization_geometry(
            only(mixed_registry.entries).source, mixed_topology,
        )
        mixed_liquid, mixed_gas = _plic_phase_areas(mixed_geometry)
        @test mixed_liquid + mixed_gas ≈ 1
        @test all(mixed_geometry.owners) do owner
            owner.boundary ?
                1 <= owner.index <= length(mixed_topology.boundary_faces) :
                1 <= owner.index <= length(mixed_topology.interior_faces)
        end
    end
end

@testset "Two-velocity visualization combines intrinsic phase fields" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 0)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.24;
        lipschitz = 1,
    )
    velocity = (
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? 1 + x : 2 + y,
        ),
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? -1 - y : 3 - x,
        ),
    )
    registry = VisualizationFieldRegistry()
    @test register_default_fields!(
        registry, :velocity, velocity, interface,
    ) === registry
    @test field_names(registry) == [
        :velocity_x, :velocity_y,
        :velocity_magnitude, :velocity_vorticity,
    ]
    @test registry.interfaces == [interface]

    for (index, entry) in pairs(registry.entries)
        values = TwoPhaseAMR._source_values(entry.source)
        if index == 1
            @test values.liquid == visualization_values(velocity[1], Face{1}())
            @test values.gas == visualization_values(velocity[2], Face{1}())
        elseif index == 2
            @test values.liquid == visualization_values(velocity[1], Face{2}())
            @test values.gas == visualization_values(velocity[2], Face{2}())
        elseif index == 3
            @test isequal(values.liquid, visualization_values(
                velocity[1], CenteredMagnitude(); interface, phase = :liquid,
            ))
            @test isequal(values.gas, visualization_values(
                velocity[2], CenteredMagnitude(); interface, phase = :gas,
            ))
        else
            @test isequal(values.liquid, visualization_values(
                velocity[1], Vorticity(); interface, phase = :liquid,
            ))
            @test isequal(values.gas, visualization_values(
                velocity[2], Vorticity(); interface, phase = :gas,
            ))
        end
    end

    frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    captured = copy(frame[:velocity_magnitude])
    fill!(velocity[1].interior, 99)
    fill!(velocity[1].boundary, 99)
    @test isequal(frame[:velocity_magnitude], captured)
    @test !isequal(
        TwoPhaseAMR._source_values(registry.entries[3].source), captured,
    )

    other_interface = InterfaceField(
        BlockTreeMesh(block_size = (8, 8)), (x, y) -> x - 0.5;
        lipschitz = 1,
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :velocity, velocity, other_interface,
    )
end

@testset "Visualization face fields" begin
    mesh = BlockTreeMesh(
        block_size = (4, 2), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 1)
    )

    for (placement, sample) in (
            (Face{1}(), (x, y) -> 2x - y),
            (Face{2}(), (x, y) -> x + 3y),
        )
        field = FaceField(mesh, placement, sample)
        geometry = visualization_geometry(mesh, placement)
        values = visualization_values(field)
        rendered = TwoPhaseAMR._visualization_face_vertex_values(
            values, geometry
        )

        @test length(geometry.points) == 4length(geometry.faces)
        @test length(geometry.triangles) == 2length(geometry.faces)
        @test maximum(geometry.value_indices) == length(values)
        @test length(rendered) == length(geometry.points)
        @test face_for_vertex(geometry, 1) == geometry.faces[1]
        @test face_for_vertex(geometry, 4) == geometry.faces[1]
        @test face_for_vertex(geometry, 5) == geometry.faces[2]
        @test isnothing(face_for_vertex(geometry, 0))
        @test isnothing(
            face_for_vertex(
                geometry, length(geometry.points) + 1
            )
        )

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
            @test all(
                ==(value),
                rendered[first_vertex:(first_vertex + 3)]
            )
            @test geometry.triangles[2index - 1] ==
                (first_vertex, first_vertex + 1, first_vertex + 2)
            @test geometry.triangles[2index] ==
                (first_vertex, first_vertex + 2, first_vertex + 3)
            tile_area += (normal_upper - normal_lower) *
                (tangent_upper - tangent_lower)
        end
        @test tile_area ≈ prod(mesh.extent)
    end

    # Each directional face field must cover the domain exactly once. Checking
    # every rectangle induced by tile edges detects both gaps and overlaps,
    # including boundary half-tiles and coarse--fine transitions.
    function test_exact_tiling(mesh, placement)
        geometry = visualization_geometry(mesh, placement)
        xedges = sort!(unique(point[1] for point in geometry.points))
        yedges = sort!(unique(point[2] for point in geometry.points))
        bounds = map(eachindex(geometry.faces)) do tile
            first_vertex = 4tile - 3
            points = geometry.points[first_vertex:(first_vertex + 3)]
            xlo, xhi = extrema(point[1] for point in points)
            ylo, yhi = extrema(point[2] for point in points)
            return (xlo, xhi, ylo, yhi)
        end
        for xi in 1:(length(xedges) - 1), yi in 1:(length(yedges) - 1)
            x = (xedges[xi] + xedges[xi + 1]) / 2
            y = (yedges[yi] + yedges[yi + 1]) / 2
            coverage = count(bounds) do (xlo, xhi, ylo, yhi)
                xlo < x < xhi && ylo < y < yhi
            end
            @test coverage == 1
        end
    end

    for periodic in ((true, true), (false, false), (true, false))
        tiling_mesh = BlockTreeMesh(
            block_size = (4, 2), periodic = periodic, max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 1)
        )
        test_exact_tiling(tiling_mesh, Face{1}())
        test_exact_tiling(tiling_mesh, Face{2}())
    end

    velocity = FaceVelocity(
        mesh, (x, y, time, axis) ->
        axis == 1 ? 2x - y + time : x + 3y - time;
        time = 0.25
    )
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
        block_size = (4, 4), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 1 && key.j == 0)
    )
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? x + y : 2x - y
    )
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
        @test vorticity[index] ≈ 1 atol = 2.0e-13
    end
end

@testset "Visualization field registry" begin
    mesh = BlockTreeMesh(block_size = (4, 2), max_level = 1)
    pressure = CellField(mesh)
    xflux = FaceField{1}(mesh)
    yflux = FaceField{2}(mesh)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? y : -x
    )
    registry = VisualizationFieldRegistry()

    @test register_field!(
        registry, :pressure, pressure; label = "Pressure", units = "Pa"
    ) === registry
    @test register_field!(registry, :xflux, xflux) === registry
    @test register_field!(
        registry, :velocity_x, velocity, Face{1}();
        label = "Velocity x", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :velocity_y, velocity, Face{2}();
        label = "Velocity y", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :velocity_magnitude, velocity, CenteredMagnitude();
        label = "Velocity magnitude"
    ) === registry
    @test register_field!(
        registry, :vorticity, velocity, Vorticity();
        label = "Vorticity", colormap = :balance
    ) === registry
    @test register_field!(
        registry, :flux_magnitude, (xflux, yflux), CenteredMagnitude()
    ) === registry
    @test register_field!(
        registry, :flux_vorticity, (xflux, yflux), Vorticity()
    ) === registry
    @test field_names(registry) == [
        :pressure, :xflux, :velocity_x, :velocity_y,
        :velocity_magnitude, :vorticity, :flux_magnitude, :flux_vorticity,
    ]
    @test_throws ArgumentError register_field!(registry, :pressure, pressure)

    other_mesh = BlockTreeMesh(block_size = (4, 2), max_level = 1)
    @test_throws ArgumentError register_field!(
        registry, :other, CellField(other_mesh)
    )
    @test_throws ArgumentError register_field!(
        VisualizationFieldRegistry(), :invalid_pair,
        (xflux, FaceField{2}(other_mesh)), CenteredMagnitude()
    )

    defaults = VisualizationFieldRegistry()
    @test register_default_fields!(
        defaults, :velocity, velocity, :pressure, pressure
    ) === defaults
    @test field_names(defaults) == [
        :velocity_x, :velocity_y, :velocity_magnitude,
        :velocity_vorticity, :pressure,
    ]
    @test [entry.label for entry in defaults.entries] == [
        "Velocity x", "Velocity y", "Velocity magnitude",
        "Velocity vorticity", "Pressure",
    ]
    @test [entry.colormap for entry in defaults.entries] == [
        :balance, :balance, :viridis, :balance, :viridis,
    ]

    tuple_defaults = VisualizationFieldRegistry()
    @test register_default_fields!(
        tuple_defaults, :mass_flux, (xflux, yflux)
    ) === tuple_defaults
    @test field_names(tuple_defaults) == [
        :mass_flux_x, :mass_flux_y, :mass_flux_magnitude,
        :mass_flux_vorticity,
    ]

    directional_default = VisualizationFieldRegistry()
    @test register_default_fields!(
        directional_default, :xflux, xflux
    ) === directional_default
    @test field_names(directional_default) == [:xflux]
    @test directional_default.entries[1].colormap == :balance

    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :pressure
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), "pressure", pressure
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :unsupported, zeros(2)
    )
    @test_throws ArgumentError register_default_fields!(
        VisualizationFieldRegistry(), :wrong_pair, (yflux, xflux)
    )

    atomic = VisualizationFieldRegistry()
    @test_throws ArgumentError register_default_fields!(
        atomic, :pressure, pressure, :pressure, pressure
    )
    @test isempty(field_names(atomic))
    @test_throws ArgumentError register_default_fields!(
        atomic, :pressure, pressure, :other, CellField(other_mesh)
    )
    @test isempty(field_names(atomic))

    existing = VisualizationFieldRegistry()
    register_field!(existing, :velocity_x, pressure)
    @test_throws ArgumentError register_default_fields!(
        existing, :velocity, velocity
    )
    @test field_names(existing) == [:velocity_x]

    reduction = (velocity, pressure, time, step, solver) -> time + step
    @test register_diagnostic!(
        registry, :progress, reduction;
        label = "Progress", units = "s", color = :red
    ) === registry
    @test diagnostic_names(registry) == [:progress]
    @test registry.diagnostics[1].label == "Progress"
    @test registry.diagnostics[1].units == "s"
    @test registry.diagnostics[1].color == :red
    @test_throws ArgumentError register_diagnostic!(
        registry, :progress, reduction
    )
end
