using Test, TwoPhaseAMR, LinearAlgebra

function _geometry_cell_values(field)
    return reduce(vcat, [vec(collect(interior(field, b))) for b in eachindex(field.mesh.blocks)])
end

function _open_phase_volume(interface, geometry)
    return sum(eachindex(interface.mesh.blocks)) do block
        sum(interior(interface.fraction, block)) *
            interface.mesh.blocks[block].volume
    end
end

function _circle_open_symmetric_difference(
        interface, centre, radius, wall_normal, wall_point;
        circle_sides = 4096,
    )
    mesh = interface.mesh
    circle = [
        (
            centre[1] + radius * cospi(2index / circle_sides),
            centre[2] + radius * sinpi(2index / circle_sides),
        ) for index in 0:(circle_sides - 1)
    ]
    wall_intercept = sum(wall_normal .* wall_point)
    difference = 0.0
    for ref in TwoPhaseAMR._cell_references(mesh)
        block = mesh.blocks[ref.block]
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        dx, dy = block.spacing
        conditional = TwoPhaseAMR._conditional_fraction(interface, ref)
        near_exact_interface = abs(
            hypot(x - centre[1], y - centre[2]) - radius,
        ) <= hypot(dx, dy)
        (near_exact_interface || 0 < conditional < 1) || continue

        exact = TwoPhaseAMR._clip_polygon(
            circle, 1.0, 0.0, x - dx / 2; keep_less = false,
        )
        exact = TwoPhaseAMR._clip_polygon(
            exact, 1.0, 0.0, x + dx / 2,
        )
        exact = TwoPhaseAMR._clip_polygon(
            exact, 0.0, 1.0, y - dy / 2; keep_less = false,
        )
        exact = TwoPhaseAMR._clip_polygon(
            exact, 0.0, 1.0, y + dy / 2,
        )
        exact = TwoPhaseAMR._clip_polygon(
            exact, wall_normal..., wall_intercept; keep_less = false,
        )

        reconstructed = NTuple{2, Float64}[
            (x - dx / 2, y - dy / 2),
            (x + dx / 2, y - dy / 2),
            (x + dx / 2, y + dy / 2),
            (x - dx / 2, y + dy / 2),
        ]
        reconstructed = TwoPhaseAMR._clip_polygon(
            reconstructed, wall_normal..., wall_intercept;
            keep_less = false,
        )
        overlap = exact
        if conditional <= 0
            empty!(reconstructed)
            overlap = similar(exact, 0)
        elseif conditional < 1
            normal_x = TwoPhaseAMR._get(interface.normal_x, ref) / dx
            normal_y = TwoPhaseAMR._get(interface.normal_y, ref) / dy
            intercept = TwoPhaseAMR._get(interface.intercept, ref) +
                normal_x * x + normal_y * y
            reconstructed = TwoPhaseAMR._clip_polygon(
                reconstructed, normal_x, normal_y, intercept,
            )
            overlap = TwoPhaseAMR._clip_polygon(
                exact, normal_x, normal_y, intercept,
            )
        end
        difference += max(
            0.0,
            TwoPhaseAMR._polygon_area(exact) +
                TwoPhaseAMR._polygon_area(reconstructed) -
                2TwoPhaseAMR._polygon_area(overlap),
        )
    end
    return difference
end

function _plic_symmetric_difference(first, second, geometry)
    first.mesh === second.mesh === geometry.mesh || throw(
        ArgumentError("interfaces and geometry must share one mesh"),
    )
    difference = zero(eltype(first))
    for ref in TwoPhaseAMR._cell_references(first.mesh)
        open_polygon = TwoPhaseAMR._open_cell_polygon(geometry, ref)
        first_fraction = TwoPhaseAMR._conditional_fraction(first, ref)
        second_fraction = TwoPhaseAMR._conditional_fraction(second, ref)
        first_fraction == second_fraction &&
            (first_fraction == 0 || first_fraction == 1) && continue

        function phase_polygon(field, fraction)
            fraction <= 0 && return similar(open_polygon, 0)
            fraction >= 1 && return open_polygon
            return TwoPhaseAMR._clip_polygon(
                open_polygon,
                TwoPhaseAMR._get(field.normal_x, ref),
                TwoPhaseAMR._get(field.normal_y, ref),
                TwoPhaseAMR._get(field.intercept, ref),
            )
        end
        first_polygon = phase_polygon(first, first_fraction)
        second_polygon = phase_polygon(second, second_fraction)
        overlap = if first_fraction <= 0 || second_fraction <= 0
            similar(open_polygon, 0)
        elseif second_fraction >= 1
            first_polygon
        else
            TwoPhaseAMR._clip_polygon(
                first_polygon,
                TwoPhaseAMR._get(second.normal_x, ref),
                TwoPhaseAMR._get(second.normal_y, ref),
                TwoPhaseAMR._get(second.intercept, ref),
            )
        end
        local_difference =
            TwoPhaseAMR._polygon_area(first_polygon) +
            TwoPhaseAMR._polygon_area(second_polygon) -
            2TwoPhaseAMR._polygon_area(overlap)
        difference += max(zero(local_difference), local_difference) *
            first.mesh.blocks[ref.block].volume
    end
    return difference
end

function _compact_plateau(value, inner, outer)
    magnitude = abs(value)
    magnitude <= inner && return 1.0
    magnitude >= outer && return 0.0
    coordinate = (magnitude - inner) / (outer - inner)
    return 1 - coordinate^2 * (3 - 2coordinate)
end

function _sloped_wall_translation_velocity(
        mesh, geometry, wall_normal, tangent, wall_point; speed,
    )
    streamfunction(x, y) = begin
        offset = (x - wall_point[1], y - wall_point[2])
        along = sum(tangent .* offset)
        normal = sum(wall_normal .* offset)
        speed * normal * _compact_plateau(along, 0.2, 0.32) *
            _compact_plateau(normal, 0.16, 0.3)
    end
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        aperture = geometry.aperture[index]
        aperture > 0 || continue
        axis = face.axis
        transverse = 3 - axis
        low = face.center[transverse] - face.area / 2
        high = face.center[transverse] + face.area / 2
        point(coordinate) = axis == 1 ?
            (face.center[1], coordinate) : (coordinate, face.center[2])
        first, last = point(low), point(high)
        first_distance = sum(
            wall_normal .* (first .- wall_point),
        )
        last_distance = sum(
            wall_normal .* (last .- wall_point),
        )
        if first_distance < 0 || last_distance < 0
            crossing = low + (high - low) * first_distance /
                (first_distance - last_distance)
            first_distance < 0 ? (low = crossing) : (high = crossing)
            first, last = point(low), point(high)
        end
        integrated_flux = axis == 1 ?
            streamfunction(last...) - streamfunction(first...) :
            -(streamfunction(last...) - streamfunction(first...))
        velocity.interior[index] = integrated_flux /
            (face.area * aperture)
    end
    return velocity
end

# Return the liquid trace as a physical interval relative to the face center.
function _component_liquid_face_interval(interface, face, ref)
    mesh = interface.mesh
    block = mesh.blocks[ref.block]
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    axis, tangent = face.axis, 3 - face.axis
    normal_coordinate = TwoPhaseAMR._wrapped_delta(
        face.center[axis] - center[axis], mesh.extent[axis], mesh.periodic[axis],
    ) / block.spacing[axis]
    lo, hi = TwoPhaseAMR._face_local_tangent_interval(mesh, ref, face)
    solid = interface.geometry.solid
    solid_fraction = TwoPhaseAMR._get(solid.fraction, ref)
    if solid_fraction > 0
        solid_normal = axis == 1 ?
            TwoPhaseAMR._get(solid.normal_x, ref) :
            TwoPhaseAMR._get(solid.normal_y, ref)
        solid_tangent = tangent == 1 ?
            TwoPhaseAMR._get(solid.normal_x, ref) :
            TwoPhaseAMR._get(solid.normal_y, ref)
        lo, hi = TwoPhaseAMR._clip_line_interval(
            lo, hi, solid_tangent,
            solid_normal * normal_coordinate - TwoPhaseAMR._get(solid.intercept, ref);
            keep_less = false,
        )
    end
    c = TwoPhaseAMR._conditional_fraction(interface, ref)
    if c < 1
        plane_normal = axis == 1 ?
            TwoPhaseAMR._get(interface.normal_x, ref) :
            TwoPhaseAMR._get(interface.normal_y, ref)
        plane_tangent = tangent == 1 ?
            TwoPhaseAMR._get(interface.normal_x, ref) :
            TwoPhaseAMR._get(interface.normal_y, ref)
        lo, hi = TwoPhaseAMR._clip_line_interval(
            lo, hi, plane_tangent,
            plane_normal * normal_coordinate -
                TwoPhaseAMR._get(interface.intercept, ref);
            keep_less = true,
        )
    end
    base = TwoPhaseAMR._wrapped_delta(
        center[tangent] - face.center[tangent],
        mesh.extent[tangent], mesh.periodic[tangent],
    )
    return base + lo * block.spacing[tangent], base + hi * block.spacing[tangent]
end

function _largest_liquid_component(interface)
    mesh = interface.mesh
    refs = [
        ref for ref in TwoPhaseAMR._cell_references(mesh)
            if TwoPhaseAMR._get(interface.fraction, ref) > 0
    ]
    neighbors = Dict(ref => CellRef[] for ref in refs)
    for (index, face) in pairs(mesh.faces)
        first, second = face.minus, face.plus
        haskey(neighbors, first) && haskey(neighbors, second) &&
            interface.geometry.aperture[index] > 0 || continue
        lo1, hi1 = _component_liquid_face_interval(interface, face, first)
        lo2, hi2 = _component_liquid_face_interval(interface, face, second)
        min(hi1, hi2) - max(lo1, lo2) >
            64eps(eltype(interface)) * face.area || continue
        push!(neighbors[first], second)
        push!(neighbors[second], first)
    end
    visited = Set{CellRef}()
    largest = Set{CellRef}()
    total_volume = largest_volume = zero(eltype(interface))
    for start in refs
        start in visited && continue
        pending = CellRef[start]
        component = Set{CellRef}([start])
        push!(visited, start)
        volume = zero(eltype(interface))
        while !isempty(pending)
            ref = pop!(pending)
            volume += TwoPhaseAMR._get(interface.fraction, ref) *
                mesh.blocks[ref.block].volume
            for neighbor in neighbors[ref]
                neighbor in visited && continue
                push!(visited, neighbor)
                push!(component, neighbor)
                push!(pending, neighbor)
            end
        end
        total_volume += volume
        if volume > largest_volume
            largest = component
            largest_volume = volume
        end
    end
    return largest, max(zero(total_volume), total_volume - largest_volume)
end

function _embedded_contact_positions(
        interface, wall_normal, tangent, wall_point; component = nothing,
    )
    mesh = interface.mesh
    wall_intercept = sum(wall_normal .* wall_point)
    positions = Float64[]
    for ref in TwoPhaseAMR._cell_references(mesh)
        isnothing(component) || ref in component || continue
        0 < TwoPhaseAMR._get(
                interface.geometry.solid.fraction, ref,
            ) < 1 || continue
        0 < TwoPhaseAMR._conditional_fraction(interface, ref) < 1 ||
            continue
        block = mesh.blocks[ref.block]
        dx, dy = block.spacing
        xcenter, ycenter = cellcenter(
            mesh, ref.block, ref.i, ref.j,
        )
        normal_x = TwoPhaseAMR._get(interface.normal_x, ref) / dx
        normal_y = TwoPhaseAMR._get(interface.normal_y, ref) / dy
        interface_intercept = TwoPhaseAMR._get(interface.intercept, ref) +
            normal_x * xcenter + normal_y * ycenter
        determinant = normal_x * wall_normal[2] -
            normal_y * wall_normal[1]
        abs(determinant) > 256eps() || continue
        x = (
            interface_intercept * wall_normal[2] -
                normal_y * wall_intercept
        ) / determinant
        y = (
            normal_x * wall_intercept -
                interface_intercept * wall_normal[1]
        ) / determinant
        abs(x - xcenter) <= dx * (0.5 + 256eps()) &&
            abs(y - ycenter) <= dy * (0.5 + 256eps()) || continue
        push!(
            positions,
            sum(tangent .* ((x, y) .- wall_point)),
        )
    end
    sort!(positions)
    return positions
end

function _sloped_wy_reverse_result(
        cells, initial_sweep_order; reconstruction_method = :lhf,
    )
    mesh = BlockTreeMesh(
        block_size = (cells, cells), periodic = (false, false),
    )
    slope = 0.25
    scale = sqrt(1 + slope^2)
    wall_normal = (-slope / scale, 1 / scale)
    tangent = (1 / scale, slope / scale)
    wall_point = (0.5, 0.25)
    radius = 0.12
    speed = 0.2
    half_period = 0.15
    wall(x, y) = sum(wall_normal .* ((x, y) .- wall_point))
    geometry = GeometryField(mesh, wall; lipschitz = 1)
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - wall_point[1], y - wall_point[2]) - radius;
        geometry, contact_angle = pi / 2, lipschitz = 1,
        atol = 1.0e-9, maxdepth = 24, reconstruction_method,
    )
    velocity = _sloped_wall_translation_velocity(
        mesh, geometry, wall_normal, tangent, wall_point; speed,
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity, geometry)
    maximum_divergence = maximum(
        abs(TwoPhaseAMR._get(divergence, ref))
            for ref in TwoPhaseAMR._cell_references(mesh)
            if TwoPhaseAMR._get(geometry.open_fraction, ref) > 0
    )
    base_velocity = copy(velocity.interior)
    workspace = WYAdvectionWorkspace(interface)
    initial_volume = _open_phase_volume(interface, geometry)
    maximum_volume_error = 0.0
    bounded = true
    time = 0.0
    step = 0
    function advance_to!(target, direction)
        while time < target - 16eps()
            velocity.interior .= direction .* base_velocity
            dt = min(
                stable_timestep(velocity, workspace; cfl = 0.4),
                target - time,
            )
            sweep_order = iseven(step) ?
                initial_sweep_order : reverse(initial_sweep_order)
            advect!(interface, velocity, dt, workspace; sweep_order)
            time += dt
            step += 1
            maximum_volume_error = max(
                maximum_volume_error,
                abs(_open_phase_volume(interface, geometry) - initial_volume),
            )
            bounded &= all(eachindex(mesh.blocks)) do block
                fraction = interior(interface.fraction, block)
                open = interior(geometry.open_fraction, block)
                all(
                    (-256eps() .<= fraction) .&
                        (fraction .<= open .+ 256eps())
                )
            end
        end
        return
    end
    advance_to!(half_period, 1.0)
    translated_centre = (
        wall_point[1] + speed * half_period * tangent[1],
        wall_point[2] + speed * half_period * tangent[2],
    )
    midpoint_reference = InterfaceField(
        mesh,
        (x, y) -> hypot(x - translated_centre[1], y - translated_centre[2]) -
            radius;
        geometry, contact_angle = pi / 2, lipschitz = 1,
        atol = 1.0e-9, maxdepth = 24, reconstruction_method,
    )
    midpoint_error = _plic_symmetric_difference(
        interface, midpoint_reference, geometry,
    )
    # At 16×16 a small detached numerical satellite may have its own wall
    # intersection. Track the principal liquid component and bound the
    # satellite volume separately; finer grids still count every contact.
    component, satellite_volume = cells == 16 ?
        _largest_liquid_component(interface) : (nothing, 0.0)
    contact_positions = _embedded_contact_positions(
        interface, wall_normal, tangent, wall_point; component,
    )
    exact_contacts = [
        -radius + speed * half_period,
        radius + speed * half_period,
    ]
    contact_error = length(contact_positions) == 2 ?
        maximum(abs.(contact_positions .- exact_contacts)) : Inf
    advance_to!(2half_period, -1.0)
    reverse_reference = InterfaceField(
        mesh,
        (x, y) -> hypot(x - wall_point[1], y - wall_point[2]) - radius;
        geometry, contact_angle = pi / 2, lipschitz = 1,
        atol = 1.0e-9, maxdepth = 24, reconstruction_method,
    )
    reverse_error = _plic_symmetric_difference(
        interface, reverse_reference, geometry,
    )
    return (;
        midpoint_error, reverse_error, contact_error,
        contact_count = length(contact_positions), maximum_volume_error,
        maximum_divergence, bounded, satellite_volume, initial_volume,
    )
end

function _maximum_aperture_to_cut_volume(geometry)
    mesh = geometry.mesh
    maximum_ratio = zero(eltype(geometry))
    for ref in TwoPhaseAMR._cell_references(mesh)
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        0 < open < 1 || continue
        volume = open * mesh.blocks[ref.block].volume
        for (index, face) in pairs(mesh.faces)
            (face.minus == ref || face.plus == ref) || continue
            maximum_ratio = max(
                maximum_ratio,
                geometry.aperture[index] * face.area / volume,
            )
        end
    end
    return maximum_ratio
end

@testset "Geometry-presence refinement indicator" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    # The solid boundary is deliberately aligned with cell and, after the
    # first refinement, block faces. Detection must therefore use halo jumps.
    geometry = GeometryField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    indicator = GeometryRefinementIndicator(
        geometry; level = 2, buffer_cells = 1,
        t_init = 1.0, t_end = 2.0,
    )
    interface_indicator = InterfaceRefinementIndicator(
        geometry.solid; level = 2, buffer_cells = 1,
        t_init = 1.0, t_end = 2.0,
    )
    labels = Vector{Symbol}(undef, nblocks(mesh))
    reference = similar(labels)
    @test indicator.field === geometry
    @test indicator.level == 2
    @test indicator.buffer_cells == 1
    @test indicator.field.solid.fraction_floor == sqrt(eps(Float64))
    @test indicator.t_init == 1.0
    @test indicator.t_end == 2.0
    @test_throws DimensionMismatch refinement_labels!(
        Symbol[], indicator; time = 0.0,
    )
    @test refinement_labels!(labels, indicator; time = 0.0) == [:coarsen]
    @test refinement_labels!(labels, indicator; time = 1.0) == [:refine]
    @test labels == refinement_labels!(
        reference, interface_indicator; time = 1.0,
    )

    workspace = AMRWorkspace(mesh)
    @test adapt!(
        workspace, indicator; step = 1, time = 1.0, force = true,
    ).changed
    @test all(block -> block.key.level == 1, mesh.blocks)
    resize!(reference, nblocks(mesh))
    @test refinement_labels!(workspace, indicator; time = 1.0) ==
        refinement_labels!(reference, interface_indicator; time = 1.0)
    @test adapt!(
        workspace, indicator; step = 2, time = 1.0, force = true,
    ).changed
    @test all(block -> block.key.level == 2, mesh.blocks)

    resize!(labels, nblocks(mesh))
    refinement_labels!(labels, indicator; time = 1.0)
    for (block, label) in zip(mesh.blocks, labels)
        xlo = block.origin[1]
        xhi = xlo + block.extent[1]
        expected = xlo <= 0.5 <= xhi ? :keep : :coarsen
        @test label == expected
    end
    @test count(==(:keep), labels) == 8
    @test count(==(:coarsen), labels) == 8

    default = GeometryRefinementIndicator(
        geometry; level = mesh.max_level,
    )
    @test default.t_init == -Inf
    @test default.t_end == Inf
    @test_throws ArgumentError GeometryRefinementIndicator(
        geometry; level = mesh.max_level + 1,
    )
    @test_throws ArgumentError GeometryRefinementIndicator(
        geometry; level = 1, buffer_cells = mesh.halo + 1,
    )
    @test_throws ArgumentError GeometryRefinementIndicator(
        geometry; level = 1, t_init = 2, t_end = 1,
    )
end

@testset "Grid-aligned solid faces are closed" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    geometry = GeometryField(
        mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    interface = InterfaceField(
        mesh, (x, y) -> -1.0; geometry, lipschitz = 1,
    )

    aligned_faces = Int[]
    for (index, face) in pairs(mesh.faces)
        minus_open = TwoPhaseAMR._get(
            geometry.open_fraction, face.minus,
        )
        plus_open = TwoPhaseAMR._get(
            geometry.open_fraction, face.plus,
        )
        (min(minus_open, plus_open), max(minus_open, plus_open)) ==
            (0.0, 1.0) || continue
        push!(aligned_faces, index)
        @test geometry.aperture[index] == 0
    end
    @test length(aligned_faces) == mesh.block_size[2]

    operator = PoissonOperator(
        interface, geometry; density = (1000.0, 1.0),
    )
    @test all(iszero, operator.face_coefficient[aligned_faces])
    acceleration = FaceVelocity(mesh)
    surface_tension!(
        acceleration, interface, operator, EllipticWorkspace(operator);
        sigma = 1.0, formulation = CGFM(),
    )
    @test all(iszero, acceleration.interior)
end


@testset "embedded geometry is independent of base-root partitioning" begin
    one_root = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (16, 16),
        periodic = (false, false),
    )
    four_roots = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8),
        periodic = (false, false),
    )
    levelset = (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27
    first = GeometryField(one_root, levelset; lipschitz = 1)
    second = GeometryField(four_roots, levelset; lipschitz = 1)

    function open_cells(geometry)
        mesh = geometry.mesh
        nx, ny = mesh.block_size
        return Dict(
            cellcenter(mesh, block, i, j) =>
                interior(geometry.open_fraction, block)[i, j]
                for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        )
    end
    function open_faces(geometry)
        return Dict(
            (face.center[1], face.center[2], face.axis) =>
                geometry.aperture[index]
                for (index, face) in pairs(geometry.mesh.faces)
        )
    end
    @test open_cells(first) == open_cells(second)
    @test open_faces(first) == open_faces(second)
end

@testset "PLIC geometry converges and produces filled solid visualization" begin
    errors = Float64[]
    radius = 0.21
    for cells in (16, 32, 64)
        mesh = BlockTreeMesh(block_size = (cells, cells))
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - radius;
            lipschitz = 1,
        )
        segments = plic_segments(geometry)
        midpoints = [
            ((first[1] + last[1]) / 2, (first[2] + last[2]) / 2)
                for (first, last) in segments
        ]
        squared_error = sum(
            (hypot(x - 0.5, y - 0.5) - radius)^2
                for (x, y) in midpoints
        )
        push!(errors, sqrt(squared_error / length(midpoints)))
        solid = plic_solid_geometry(geometry)
        triangulated_area = sum(solid.triangles) do triangle
            a, b, c = solid.points[triangle[1]], solid.points[triangle[2]],
                solid.points[triangle[3]]
            abs(
                (b[1] - a[1]) * (c[2] - a[2]) -
                    (b[2] - a[2]) * (c[1] - a[1]),
            ) / 2
        end
        @test triangulated_area ≈ integrate(geometry) atol = 2.0e-13
    end
    orders = log2.(errors[1:(end - 1)] ./ errors[2:end])
    # The LHF-to-WY default gives 1.499 on this coarse 16--64 cell sequence.
    @test minimum(orders) > 1.49
end

@testset "Cut-cell projection and centred convection preserve symmetry" begin
    mesh = BlockTreeMesh(
        block_size = (16, 16), periodic = (true, true),
    )
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.18;
        lipschitz = 1,
    )
    velocity = FaceVelocity(mesh)
    for index in eachindex(velocity.interior)
        velocity.interior[index] = sin(0.413index)
    end
    pressure = CellField(mesh)
    operator = PoissonOperator(geometry)
    @test issymmetric(operator.matrix)
    solver = EllipticSolver(operator; preconditioner = nothing)
    projection = ProjectionWorkspace(solver)
    projection_dt = 0.01
    original_velocity = copy(velocity.interior)
    @test_throws ArgumentError project!(
        velocity, pressure, solver, projection_dt, projection, geometry;
        compatibility = :invalid,
    )
    @test velocity.interior == original_velocity
    stats = project!(
        velocity, pressure, solver, projection_dt, projection, geometry;
        tolerance = 1.0e-11, max_cycles = 1000,
    )
    @test stats.converged
    divergence = CellField(mesh)
    divergence!(divergence, velocity, geometry)
    @test maximum(abs, _geometry_cell_values(divergence)) < 1.0e-8

    workspace = StaggeredVelocityAdvectionWorkspace(velocity, geometry)
    tendency = FaceVelocity(mesh)
    convection!(
        tendency, velocity, velocity, 0.0, workspace;
        scheme = Centred(), amr_interface_scheme = Centred(),
    )
    ux, uy = FaceField{1}(velocity), FaceField{2}(velocity)
    cx, cy = FaceField{1}(tendency), FaceField{2}(tendency)
    energy_rate =
        staggered_inner_product(ux, cx, workspace.xworkspace) +
        staggered_inner_product(uy, cy, workspace.yworkspace)
    scale = sqrt(
        (
            staggered_inner_product(ux, ux, workspace.xworkspace) +
                staggered_inner_product(uy, uy, workspace.yworkspace)
        ) * (
            staggered_inner_product(cx, cx, workspace.xworkspace) +
                staggered_inner_product(cy, cy, workspace.yworkspace)
        ),
    )
    @test abs(energy_rate) <= 2.0e-11 * scale

    diffusion = StaggeredDiffusionOperator(geometry; viscosity = 0.1)
    for component in (diffusion.x, diffusion.y)
        @test issymmetric(component.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(component.stiffness)))) > -1.0e-11
    end
end

@testset "Tree multigrid is robust to aperture/cut-volume ratio" begin
    ratios = Float64[]
    iterations = Int[]
    for relative_sliver in (1.0e-1, 1.0e-3, 1.0e-5, 1.0e-6)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            periodic = (false, false),
            refine = (key, origin, extent) -> key.level < 2,
        )
        spacing = mesh.blocks[1].spacing[1]
        intercept = 0.5 + 2spacing - relative_sliver * spacing
        solid = InterfaceField(mesh; fraction_floor = eps(Float64))
        initialize!(solid, (1.0, 1.0), intercept)
        geometry = GeometryField(solid)
        push!(ratios, _maximum_aperture_to_cut_volume(geometry))

        operator = PoissonOperator(geometry)
        exact = CellField(mesh)
        initialize!(exact, (x, y) -> sinpi(2x) * cospi(2y))
        right_hand_side = CellField(mesh)
        apply!(
            right_hand_side, operator, exact,
            EllipticWorkspace(operator),
        )
        solver = EllipticSolver(
            operator; preconditioner = :tree_geometric,
            preconditioner_options = (; geometry),
        )
        statistics = solve!(
            CellField(mesh), solver, right_hand_side;
            rtol = 1.0e-10, atol = 1.0e-12, max_cycles = 100,
        )
        @test statistics.converged
        push!(iterations, statistics.cycles)
        @test length(solver.preconditioner.geometry_levels) ==
            length(solver.preconditioner.operators)
    end
    @test all(diff(ratios) .> 0)
    @test ratios[end] / ratios[1] > 1.0e4
    @test maximum(iterations) <= 2minimum(iterations)

    # The uniform, constant-coefficient sliver experiment above does not
    # exercise the difficult combination present in LGPI: a mixed-level tree,
    # cut cells, and a 1000:1 jump in the pressure coefficient.  Keep this
    # static manufactured problem independent of the Navier--Stokes time loop
    # so that a large Krylov count diagnoses the preconditioner itself.
    mesh = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0, 0.0), extent = (1.0, 1.0),
        periodic = (false, false), max_level = 3,
        adaptation = MeshAdaptationOptions(
            update_interval = 1, coarsen_delay = 0,
        ),
    )
    elevation(x) = 19 / 30 + 3 / 10 * tanh(8.8 * (x - 0.775))
    interface_levelset(x, y) = y - elevation(x)
    geometry_levelset(x, y) = hypot(x / 0.9, y / (7 / 30)) - 1
    geometry = GeometryField(
        mesh, geometry_levelset;
        lipschitz = 30 / 7, atol = 1.0e-8, maxdepth = 28,
    )
    interface = InterfaceField(
        mesh, interface_levelset; geometry,
        lipschitz = hypot(1, 2.64), atol = 1.0e-8, maxdepth = 28,
    )
    indicators = (
        InterfaceRefinementIndicator(interface; level = 3),
        GeometryRefinementIndicator(geometry; level = 3),
    )
    adaptation = AMRWorkspace(mesh)
    register_adaptive_field!(adaptation, interface)
    register_adaptive_field!(adaptation, geometry)
    for pass in 1:4
        statistics = adapt!(
            adaptation, indicators; step = pass, force = true,
        )
        statistics.changed || break
        initialize!(
            geometry.solid, geometry_levelset;
            lipschitz = 30 / 7, atol = 1.0e-8, maxdepth = 28,
        )
        refresh_geometry!(geometry)
        initialize!(
            interface, interface_levelset;
            lipschitz = hypot(1, 2.64), atol = 1.0e-8, maxdepth = 28,
        )
    end
    # The solve below needs a composite (multi-level) mesh around the cut cells.
    @test length(unique(block.key.level for block in mesh.blocks)) > 1

    function manufactured_solve(operator; geometry = nothing)
        exact = CellField(mesh)
        initialize!(
            exact,
            (x, y) -> sinpi(x) * cospi(y) +
                0.2cospi(2x) * sinpi(3y),
        )
        right_hand_side = CellField(mesh)
        apply!(
            right_hand_side, operator, exact,
            EllipticWorkspace(operator),
        )
        options = isnothing(geometry) ? (;) : (; geometry)
        solver = EllipticSolver(
            operator; preconditioner = :tree_geometric,
            preconditioner_options = options,
        )
        statistics = solve!(
            CellField(mesh), solver, right_hand_side;
            rtol = 1.0e-10, atol = 1.0e-12, max_cycles = 500,
        )
        @test statistics.converged
        return (; statistics, solver)
    end

    controls = (
        manufactured_solve(PoissonOperator(mesh)),
        manufactured_solve(
            PoissonOperator(interface; density = (1000.0, 1.0)),
        ),
        manufactured_solve(PoissonOperator(geometry); geometry),
    )
    coupled = manufactured_solve(
        PoissonOperator(
            interface, geometry; density = (1000.0, 1.0),
        ); geometry,
    )
    control_cycles = map(control -> control.statistics.cycles, controls)
    coupled_cycles = coupled.statistics.cycles
    @test maximum(control_cycles) <= 20
    @test coupled_cycles <= 2maximum(control_cycles)

    function test_active_nullspace_transfer(preconditioner)
        for level in eachindex(preconditioner.transfer_refreshes)
            prolongation = preconditioner.levels[level].prolongation
            fine_active = Float64.(
                preconditioner.geometry_levels[level].active_cells,
            )
            coarse_active = Float64.(
                preconditioner.geometry_levels[level + 1].active_cells,
            )
            @test prolongation * coarse_active ≈ fine_active atol = 1.0e-14
        end
    end
    preconditioner = coupled.solver.preconditioner
    test_active_nullspace_transfer(preconditioner)

    # Interface advection refreshes the coefficient-dependent transfer values
    # without rebuilding their sparsity.  Covered rows must remain excluded.
    shifted = InterfaceField(
        mesh, (x, y) -> y - 0.75; geometry,
        lipschitz = 1.0, atol = 1.0e-8, maxdepth = 28,
    )
    refreshed = PoissonOperator(
        shifted, geometry; density = (1000.0, 1.0),
        template = coupled.solver.operator,
    )
    TwoPhaseAMR._refresh_geometric_multigrid!(preconditioner, refreshed)
    test_active_nullspace_transfer(preconditioner)

    exact = CellField(mesh)
    initialize!(
        exact,
        (x, y) -> sinpi(x) * cospi(y) +
            0.2cospi(2x) * sinpi(3y),
    )
    right_hand_side = CellField(mesh)
    apply!(
        right_hand_side, refreshed, exact,
        EllipticWorkspace(refreshed),
    )
    refreshed_statistics = solve!(
        CellField(mesh), coupled.solver, right_hand_side;
        rtol = 1.0e-10, atol = 1.0e-12, max_cycles = 500,
    )
    @test refreshed_statistics.converged
    @test refreshed_statistics.cycles <= 2maximum(control_cycles)
end

@testset "Geometry revision tracks the solid's PLIC planes" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1, periodic = (false, false),
    )
    geometry_version(geometry) = TwoPhaseAMR._interface_geometry_revision(geometry)
    geometry = GeometryField(mesh, (x, y) -> hypot(x, y) - 0.4)
    revision = geometry_version(geometry)
    refresh_geometry!(geometry)
    @test geometry_version(geometry) != revision
    # Re-initializing the solid changes the stored planes without refreshing
    # the open metrics. Every cache that reads those planes keys on the
    # combined revision, so it must change with them.
    revision = geometry_version(geometry)
    initialize!(geometry.solid, (1.0, 0.0), 0.1)
    @test geometry_version(geometry) != revision
    # A forced reconstruction also bumps it, even when the planes are equal.
    revision = geometry_version(geometry)
    reconstruct!(geometry.solid; force = true)
    @test geometry_version(geometry) != revision
    # An up-to-date solid is not reconstructed, so its version stays put.
    revision = geometry_version(geometry)
    reconstruct!(geometry.solid)
    @test geometry_version(geometry) == revision
    initialize!(geometry.solid, (1.0, 0.0), 0.1)
    @test geometry_version(geometry) == revision

    # Two geometries over one solid both see the solid change: the version is
    # a function of the solid's state, not a per-geometry back-reference.
    shared = InterfaceField(mesh)
    initialize!(shared, (0.0, 1.0), 0.1)
    first_geometry = GeometryField(shared)
    second_geometry = GeometryField(shared)
    first_version = geometry_version(first_geometry)
    second_version = geometry_version(second_geometry)
    initialize!(shared, (1.0, 0.0), -0.1)
    @test geometry_version(first_geometry) != first_version
    @test geometry_version(second_geometry) != second_version
end

@testset "Geometry hierarchy is rebuilt consistently after AMR" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(
            update_interval = 1, coarsen_delay = 0,
        ),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.2 - 0.15x)
    indicator = GeometryRefinementIndicator(geometry; level = 1)
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        geometry, adaptivity = Adaptivity(criteria = (indicator,)), dt = 1.0e-4,
    )
    old_preconditioner = flow.system.pressure_solver.preconditioner
    result = simulate!(
        flow, 1.0e-4; tolerance = 1.0e-10, max_cycles = 200,
    )
    @test result.mesh_updates == 1
    rebuilt = flow.system.pressure_solver.preconditioner
    @test rebuilt !== old_preconditioner
    @test rebuilt.reused_levels == 0

    fresh_solver = EllipticSolver(
        PoissonOperator(geometry); preconditioner = :tree_geometric,
        preconditioner_options = (; geometry),
    )
    fresh = fresh_solver.preconditioner
    @test length(rebuilt.operators) == length(fresh.operators)
    for (rebuilt_operator, fresh_operator) in
        zip(rebuilt.operators, fresh.operators)
        @test rebuilt_operator.volumes ≈ fresh_operator.volumes
        @test rebuilt_operator.face_coefficient ≈
            fresh_operator.face_coefficient
        @test rebuilt_operator.matrix ≈ fresh_operator.matrix
    end
end

@testset "Open boundaries preserve uniform single-phase through-flow" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = FreeSlip(), top = FreeSlip(),
    )
    velocity = FaceVelocity(mesh, inflow)
    enforce_velocity_boundary!(velocity, boundary)
    flux = sum(pairs(mesh.boundary_faces)) do (index, face)
        face.side * face.area * velocity.boundary[index]
    end
    @test abs(flux) < 2.0e-14
    advection = StaggeredVelocityAdvectionWorkspace(velocity)
    tendency = FaceVelocity(mesh)
    convection!(
        tendency, velocity, velocity, 0.001, advection;
        scheme = Centred(), boundary,
    )
    @test maximum(abs, tendency.interior) < 2.0e-14

    solver = NavierStokesSystem(
        mesh; density = 1.0, dynamic_viscosity = 0.01,
        boundary, scheme = Centred(),
    )
    pressure = CellField(mesh)
    workspace = NavierStokesWorkspace(solver)
    stats = step!(velocity, pressure, solver, 0.001, workspace)
    @test stats.converged
    @test maximum(abs, _geometry_cell_values(pressure)) < 2.0e-12
end

@testset "Triple-cell normals converge with matching and mismatched angles" begin
    # Aggregate over walls and contact positions instead of taking a maximum
    # over the two or three triple cells of a single drop.
    methods = (:lhf, :lhf_ghf, :wy)
    angles = (60, 90, 120)
    errors = Dict(
        (method, angle) => Tuple{Float64, Float64}[]
            for method in methods, angle in angles
    )
    function sample_median(values)
        sort!(values)
        n = length(values)
        return (values[cld(n, 2)] + values[fld(n, 2) + 1]) / 2
    end
    theta, offset, radius = deg2rad(60.0), 1.7, 4.2
    for cells in (32, 64, 128)
        normal_errors = Dict(key => Float64[] for key in keys(errors))
        curvature_errors = Dict(key => Float64[] for key in keys(errors))
        fitted = Dict(key => 0 for key in keys(errors))
        mesh = BlockTreeMesh(
            base_mesh = (cells, cells ÷ 2), block_size = (16, 8),
            extent = (16.0, 8.0), periodic = (false, false),
        )
        for slope in (0.0, 0.2, -0.3, 0.5)
            scale = hypot(1, slope)
            wall_normal = (-slope, 1.0) ./ scale
            wall_tangent = (1.0, slope) ./ scale
            wall(x, y) = (y - offset - slope * (x - 8)) / scale
            geometry = GeometryField(mesh, wall)
            for contact_x in (9.7, 10.9, 12.1)
                contact = (contact_x, offset + slope * (contact_x - 8))
                contact_normal =
                    cos(theta) .* wall_normal .+ sin(theta) .* wall_tangent
                centre = contact .- radius .* contact_normal
                drop(x, y) = hypot(x - centre[1], y - centre[2]) - radius
                default = InterfaceField(
                    mesh, drop; geometry, embedded_contact_angle = theta,
                    contact_angle = nothing,
                )
                @test default.reconstruction_method == :lhf
                for method in methods, angle in angles
                    field = method == :lhf && angle == 60 ? default :
                        InterfaceField(
                            copy(default.fraction); geometry,
                            contact_angle = nothing,
                            embedded_contact_angle = deg2rad(angle),
                            reconstruction_method = method,
                        )
                    key = (method, angle)
                    for ref in TwoPhaseAMR._cell_references(mesh)
                        c = TwoPhaseAMR._conditional_fraction(field, ref)
                        0 < c < 1 || continue
                        solid = TwoPhaseAMR._get(geometry.solid.fraction, ref)
                        if !(0 < solid < 1)
                            if cells == 32 && slope == 0.2 &&
                                    contact_x == 10.9 && angle == 60 && method == :wy
                                h = mesh.halo
                                values = field.conditional_fraction.blocks[ref.block]
                                normal = TwoPhaseAMR._wy_normal(
                                    values, ref.i + h, ref.j + h,
                                )
                                @test isequal(
                                    TwoPhaseAMR._get(field.normal_x, ref), normal[1],
                                )
                                @test isequal(
                                    TwoPhaseAMR._get(field.normal_y, ref), normal[2],
                                )
                                @test TwoPhaseAMR._get(field.reconstruction, ref) == 3
                            end
                            continue
                        end
                        fitted[key] +=
                            TwoPhaseAMR._get(field.reconstruction, ref) in (1, 2)
                        mx = TwoPhaseAMR._get(field.normal_x, ref)
                        my = TwoPhaseAMR._get(field.normal_y, ref)
                        a = TwoPhaseAMR._get(field.intercept, ref)
                        spacing = mesh.blocks[ref.block].spacing
                        normal = (mx / spacing[1], my / spacing[2])
                        normal = normal ./ hypot(normal...)
                        centroid = TwoPhaseAMR._interface_centroid(mx, my, a)
                        x = cellcenter(mesh, ref.block, ref.i, ref.j) .+
                            centroid .* spacing
                        exact = (x .- centre) ./ hypot((x .- centre)...)
                        push!(normal_errors[key], hypot((normal .- exact)...))
                        curvature = TwoPhaseAMR._get(field.curvature, ref)
                        if isfinite(curvature)
                            push!(curvature_errors[key], abs(curvature - inv(radius)))
                        end
                    end
                end
            end
        end
        for key in keys(errors)
            count = length(normal_errors[key])
            @test count >= 20
            @test fitted[key] >= 0.85 * count
            @test length(curvature_errors[key]) >= 0.9 * count
            push!(
                errors[key],
                (sample_median(normal_errors[key]), sample_median(curvature_errors[key])),
            )
        end
    end
    for method in methods, angle in angles
        normals = first.(errors[(method, angle)])
        @test all(diff(normals) .< 0)
        @test normals[end] < 0.5 * normals[1]
    end
    for method in methods
        curvature = last.(errors[(method, 60)])
        @test curvature[end] < curvature[1]
    end
    println("triple-cell median (normal, curvature) errors: ", errors)
end

@testset "Sliver triple-cell orientation does not reverse" begin
    cells = 128
    slope, offset, radius, contact_x = 0.2, 1.7, 4.2, 9.7
    theta = deg2rad(60.0)
    scale = hypot(1, slope)
    wall_normal = (-slope, 1.0) ./ scale
    wall_tangent = (1.0, slope) ./ scale
    contact = (contact_x, offset + slope * (contact_x - 8))
    exact_contact_normal =
        cos(theta) .* wall_normal .+ sin(theta) .* wall_tangent
    centre = contact .- radius .* exact_contact_normal
    wall(x, y) = (y - offset - slope * (x - 8)) / scale
    drop(x, y) = hypot(x - centre[1], y - centre[2]) - radius
    mesh = BlockTreeMesh(
        base_mesh = (cells, cells ÷ 2), block_size = (16, 8),
        extent = (16.0, 8.0), periodic = (false, false),
    )
    geometry = GeometryField(mesh, wall)
    initial = InterfaceField(
        mesh, drop; geometry, contact_angle = nothing,
        embedded_contact_angle = theta,
    )
    target = CellRef(2, 5, 5)
    @test 0 < TwoPhaseAMR._conditional_fraction(initial, target) < 1
    @test 0 < TwoPhaseAMR._get(geometry.solid.fraction, target) < 1
    for method in (:lhf, :lhf_ghf, :wy)
        reference_plane = nothing
        for angle in (60, 90, 120)
            field = InterfaceField(
                copy(initial.fraction); geometry, contact_angle = nothing,
                embedded_contact_angle = deg2rad(angle),
                reconstruction_method = method,
            )
            mx = TwoPhaseAMR._get(field.normal_x, target)
            my = TwoPhaseAMR._get(field.normal_y, target)
            a = TwoPhaseAMR._get(field.intercept, target)
            plane = (mx, my, a)
            isnothing(reference_plane) && (reference_plane = plane)
            @test isequal(plane, reference_plane)
            spacing = mesh.blocks[target.block].spacing
            normal = (mx / spacing[1], my / spacing[2])
            normal = normal ./ hypot(normal...)
            centroid = TwoPhaseAMR._interface_centroid(mx, my, a)
            x = cellcenter(mesh, target.block, target.i, target.j) .+
                centroid .* spacing
            exact = (x .- centre) ./ hypot((x .- centre)...)
            @test sum(normal .* exact) > 0
            @test hypot((normal .- exact)...) < 0.4
        end
    end
end

@testset "Solid-aware WY differences and masked columns" begin
    for T in (Float32, Float64)
        # A linear fraction field: removing one of the three column pairs
        # must retain the unmasked transverse estimate, not two thirds of it.
        values = T[
            0.5 - 0.25 * (i - 3) - 0.125 * (j - 3)
                for i in 1:5, j in 1:5
        ]
        open = ones(T, 5, 5)
        @test TwoPhaseAMR._solid_aware_wy_normal(values, open, 3, 3) ==
            TwoPhaseAMR._wy_normal(values, 3, 3) == (T(1), T(0.375))
        open[2, 2] = 0
        @test TwoPhaseAMR._solid_aware_wy_normal(values .* open, open, 3, 3) ==
            (T(1), T(0.375))

        # The x difference spans only one cell, while the y difference is
        # central. Scaling the former by two makes x the principal direction.
        values = fill(T(0.5), 5, 5)
        open = ones(T, 5, 5)
        values[2, 3] = T(0.75)
        open[4, 3] = 0
        values[3, 2], values[3, 4] = T(0.6875), T(0.3125)
        @test TwoPhaseAMR._solid_aware_wy_normal(values .* open, open, 3, 3) ==
            (T(1), T(-0.09375))

        # Even with zero central gradients the WY columns carry a transverse
        # contribution. Only a completely unsupported stencil defaults to 0.
        values = fill(T(0.5), 5, 5)
        values[2, 2] = values[2, 4] = T(0.75)
        open = ones(T, 5, 5)
        @test TwoPhaseAMR._solid_aware_wy_normal(values, open, 3, 3) ==
            TwoPhaseAMR._wy_normal(values, 3, 3) == (T(0.5), T(1))
        fill!(open, 0)
        open[3, 3] = 1
        @test TwoPhaseAMR._solid_aware_wy_normal(values .* open, open, 3, 3) ==
            (T(0), T(1))
    end
end


@testset "Triple-cell fit failure excludes covered WY neighbours" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> y - 0.4)
    fraction = CellField(mesh)
    target = CellRef(1, 4, 4)
    for ref in TwoPhaseAMR._cell_references(mesh)
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        # The exact liquid region is x <= center_x of column 4.
        value = ref.i < 4 ? open : ref.i == 4 ? open / 2 : 0.0
        TwoPhaseAMR._set!(fraction, value, ref)
    end
    for method in (:lhf, :lhf_ghf, :wy)
        field = InterfaceField(
            copy(fraction); geometry, contact_angle = nothing,
            embedded_contact_angle = nothing, reconstruction_method = method,
        )
        h = mesh.halo
        physical = field.fraction.blocks[1]
        open = geometry.open_fraction.blocks[1]
        i, j = target.i + h, target.j + h
        fallback = TwoPhaseAMR._solid_aware_wy_normal(physical, open, i, j)
        @test fallback == (1.0, 0.0)
        values = TwoPhaseAMR._fill_height_values!(field)[1]
        neighbor = TwoPhaseAMR._nearby_lhf_normal(
            field, values, 1, target.i, target.j, target, 0.5, fallback...,
        )
        @test neighbor == (1.0, 0.0)
        @test isnothing(
            TwoPhaseAMR._nearby_lhf_normal(
                field, values, 1, target.i, target.j, target, 0.5, -1.0, 0.0,
            ),
        )
        changed_covered = copy(physical)
        for dj in -1:1, di in -1:1
            open[i + di, j + dj] > 0 ||
                (changed_covered[i + di, j + dj] = 100.0)
        end
        @test TwoPhaseAMR._solid_aware_wy_normal(changed_covered, open, i, j) ==
            fallback
        # A coarse halo containing covered and liquid donors represents the
        # same conditional liquid fraction as a completely open liquid cell.
        reduced_open = copy(open)
        reduced_liquid = copy(physical)
        reduced_open[i - 1, j + 1] *= 0.25
        reduced_liquid[i - 1, j + 1] *= 0.25
        @test TwoPhaseAMR._solid_aware_wy_normal(
            reduced_liquid, reduced_open, i, j,
        ) == fallback

        # No usable height or centroid samples: force the real fallback path.
        unavailable = fill(NaN, mesh.block_size .+ 8)
        TwoPhaseAMR._reconstruct_interface_cell!(field, unavailable, 1, 4, 4)
        @test TwoPhaseAMR._get(field.reconstruction, target) == 3
        @test (
            TwoPhaseAMR._get(field.normal_x, target),
            TwoPhaseAMR._get(field.normal_y, target),
        ) == fallback
        @test isnan(TwoPhaseAMR._get(field.curvature, target))
        @test TwoPhaseAMR._get(field.intercept, target) ==
            TwoPhaseAMR._open_plane_intercept(geometry, target, 0.5, fallback...)
    end
end

@testset "Triple-cell curvature falls back to the unconstrained fit" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> y - 7 / 16)
    fraction = CellField(mesh)
    target = CellRef(1, 4, 4)
    for ref in TwoPhaseAMR._cell_references(mesh)
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        value = ref.i < 4 ? open : ref.i == 4 ? open / 2 : 0.0
        TwoPhaseAMR._set!(fraction, value, ref)
    end
    for method in (:lhf, :lhf_ghf, :wy)
        field = InterfaceField(
            copy(fraction); geometry, contact_angle = nothing,
            embedded_contact_angle = pi / 3, reconstruction_method = method,
        )
        # Three separated centroids with almost equal tangent coordinates.
        # Their unconstrained matrix passes the rank test; the much larger
        # contact row makes the constrained matrix fail its relative test.
        values = fill(NaN, mesh.block_size .+ 8)
        for di in -1:1
            ref = CellRef(1, target.i + di, target.j)
            values[target.i + 4 + di, target.j + 4] = 0.5
            TwoPhaseAMR._set!(field.previous_normal_x, 1.0, ref)
            TwoPhaseAMR._set!(field.previous_normal_y, 1.0, ref)
            TwoPhaseAMR._set!(field.previous_intercept, 2.0e-4 * di, ref)
        end
        TwoPhaseAMR._set!(field.normal_x, 1.0, target)
        TwoPhaseAMR._set!(field.normal_y, 0.0, target)
        TwoPhaseAMR._set!(field.intercept, 0.0, target)
        args = (
            field, values, 1, target.i, target.j,
            field.previous_normal_x, field.previous_normal_y,
            field.previous_intercept,
        )
        plane = TwoPhaseAMR._ghf_geometry(
            args...; enforce_contact = false, sample_radius = 3,
        )
        constrained = TwoPhaseAMR._ghf_geometry(
            args...; enforce_contact = true, sample_radius = 3,
        )
        @test !isnothing(plane)
        @test isnothing(constrained)
        if !isnothing(plane)
            TwoPhaseAMR._reconstruct_interface_cell!(
                field, values, 1, target.i, target.j,
            )
            @test TwoPhaseAMR._get(field.reconstruction, target) == 2
            @test TwoPhaseAMR._get(field.curvature, target) ==
                TwoPhaseAMR._clip_curvature_to_mesh(
                plane[3], mesh.blocks[1].spacing,
            )
        end
    end
end

@testset "Embedded GHF contact curvature converges on sloped walls" begin
    intercept, radius = 0.12, 0.3
    errors = Float64[]
    for cells in (16, 32, 64)
        mesh = BlockTreeMesh(block_size = (cells, cells))
        curvature_errors = Float64[]
        triple_count = fitted_count = 0
        for slope in (-0.08, 0.1, 0.25, 0.4)
            geometry = GeometryField(
                mesh, (x, y) -> y - intercept - slope * x,
            )
            for centre_x in (0.4, 0.5, 0.6)
                centre = (centre_x, 0.35)
                distance = (
                    centre[2] - intercept - slope * centre[1]
                ) / sqrt(1 + slope^2)
                contact_angle = acos(-distance / radius)
                interface = InterfaceField(
                    mesh,
                    (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
                    geometry, contact_angle, lipschitz = 1,
                    reconstruction_method = :lhf_ghf,
                )
                for ref in TwoPhaseAMR._cell_references(mesh)
                    0 < TwoPhaseAMR._get(geometry.solid.fraction, ref) < 1 ||
                        continue
                    0 < TwoPhaseAMR._conditional_fraction(interface, ref) < 1 ||
                        continue
                    triple_count += 1
                    fitted_count += TwoPhaseAMR._get(interface.reconstruction, ref) == 2
                    curvature = TwoPhaseAMR._get(interface.curvature, ref)
                    if isfinite(curvature)
                        push!(curvature_errors, abs(curvature - inv(radius)))
                    end
                end
            end
        end
        @test triple_count >= 20
        @test fitted_count >= 0.9 * triple_count
        @test length(curvature_errors) >= 0.9 * triple_count
        sort!(curvature_errors)
        n = length(curvature_errors)
        push!(
            errors,
            (curvature_errors[cld(n, 2)] + curvature_errors[fld(n, 2) + 1]) / 2,
        )
    end
    observed_order = log(errors[1] / errors[end]) / log(4)
    @test all(diff(errors) .< 0)
    @test observed_order > 0.8
end

@testset "Embedded height reconstruction converges in symmetric difference" begin
    wall_angle = pi / 12
    wall_normal = (-sin(wall_angle), cos(wall_angle))
    wall_point = (0.5, 0.22)
    centre = wall_point
    radius = 0.24
    errors = Float64[]
    resolutions = (16, 32, 64)
    for cells in resolutions
        mesh = BlockTreeMesh(
            block_size = (cells, cells), periodic = (false, false),
        )
        geometry = GeometryField(
            mesh,
            (x, y) -> sum(
                wall_normal .* ((x, y) .- wall_point),
            );
            lipschitz = 1,
        )
        interface = InterfaceField(
            mesh,
            (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
            geometry, contact_angle = pi / 2, lipschitz = 1,
            atol = 1.0e-9, maxdepth = 24,
            reconstruction_method = :lhf_ghf,
        )
        triple_cells = [
            ref for ref in TwoPhaseAMR._cell_references(mesh)
                if 0 < TwoPhaseAMR._get(
                    geometry.solid.fraction, ref,
                ) < 1 &&
                0 < TwoPhaseAMR._conditional_fraction(interface, ref) < 1
        ]
        bulk_lhf_cells = [
            ref for ref in TwoPhaseAMR._cell_references(mesh)
                if TwoPhaseAMR._get(interface.reconstruction, ref) == 1
        ]
        @test !isempty(triple_cells)
        @test !isempty(bulk_lhf_cells)
        # The triple-cell plane uses an unconstrained generalized height fit;
        # the embedded contact-angle row is used for curvature only.
        @test all(
            TwoPhaseAMR._get(interface.reconstruction, ref) == 2
                for ref in triple_cells
        )
        push!(
            errors,
            _circle_open_symmetric_difference(
                interface, centre, radius, wall_normal, wall_point,
            ),
        )
    end
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    println(
        "embedded height reconstruction symmetric difference " *
            "n=$resolutions errors=$errors rates=$rates",
    )
    @test all(diff(errors) .< 0)
    @test all(rates .>= 1.8)
end

@testset "Geometry-aware level-set initialization converges in open volume" begin
    angle = pi / 12
    normal = (-sin(angle), cos(angle))
    wall_point = (0.5, 0.22)
    radius = 0.16
    wall(x, y) = normal[1] * (x - wall_point[1]) +
        normal[2] * (y - wall_point[2])
    circle(x, y) = hypot(x - wall_point[1], y - wall_point[2]) - radius
    errors = Float64[]
    for cells in (16, 32, 64)
        mesh = BlockTreeMesh(block_size = (cells, cells))
        geometry = GeometryField(mesh, wall)
        interface = InterfaceField(
            mesh, circle; geometry, contact_angle = pi / 2,
            lipschitz = 1,
        )
        @test all(eachindex(mesh.blocks)) do block
            fraction = interior(interface.fraction, block)
            open = interior(geometry.open_fraction, block)
            all(
                (-128eps() .<= fraction) .&
                    (fraction .<= open .+ 128eps())
            )
        end
        push!(
            errors,
            _circle_open_symmetric_difference(
                interface, wall_point, radius, normal, wall_point,
            ),
        )
    end
    @test all(log2.(errors[1:(end - 1)] ./ errors[2:end]) .>= 1.8)
end

@testset "Embedded contact angles do not change PLIC planes" begin
    wall_angle = pi / 12
    wall_normal = (-sin(wall_angle), cos(wall_angle))
    wall_point = (0.5, 0.22)
    radius = 0.24
    mesh = BlockTreeMesh(
        block_size = (64, 64), periodic = (false, false),
    )
    geometry = GeometryField(
        mesh,
        (x, y) -> sum(wall_normal .* ((x, y) .- wall_point));
        lipschitz = 1,
    )
    fraction = InterfaceField(
        mesh,
        (x, y) -> hypot(x - wall_point[1], y - wall_point[2]) - radius;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    ).fraction
    for method in (:lhf, :lhf_ghf, :wy)
        first = InterfaceField(
            copy(fraction); geometry, contact_angle = nothing,
            embedded_contact_angle = pi / 3, reconstruction_method = method,
        )
        for angle in (nothing, pi / 2, 2pi / 3)
            second = InterfaceField(
                copy(fraction); geometry, contact_angle = nothing,
                embedded_contact_angle = angle, reconstruction_method = method,
            )
            curvature_changed = false
            planes_unchanged = true
            other_curvatures_unchanged = true
            triple_cells = 0
            for ref in TwoPhaseAMR._cell_references(mesh)
                c = TwoPhaseAMR._conditional_fraction(first, ref)
                0 < c < 1 || continue
                plane1 = (
                    TwoPhaseAMR._get(first.normal_x, ref),
                    TwoPhaseAMR._get(first.normal_y, ref),
                    TwoPhaseAMR._get(first.intercept, ref),
                )
                plane2 = (
                    TwoPhaseAMR._get(second.normal_x, ref),
                    TwoPhaseAMR._get(second.normal_y, ref),
                    TwoPhaseAMR._get(second.intercept, ref),
                )
                planes_unchanged &= isequal(plane1, plane2)
                solid = TwoPhaseAMR._get(geometry.solid.fraction, ref)
                k1 = TwoPhaseAMR._get(first.curvature, ref)
                k2 = TwoPhaseAMR._get(second.curvature, ref)
                if !(0 < solid < 1)
                    other_curvatures_unchanged &= isequal(k1, k2)
                    continue
                end
                triple_cells += 1
                isfinite(k1) && isfinite(k2) && (curvature_changed |= k1 != k2)
            end
            @test triple_cells > 0
            @test planes_unchanged
            @test other_curvatures_unchanged
            @test curvature_changed
        end
    end
end

@testset "Cut-cell WY swept flux is a bounded phase partition" begin
    mesh = BlockTreeMesh(block_size = (16, 16))
    angle = pi / 12
    normal = (-sin(angle), cos(angle))
    point = (0.5, 0.22)
    geometry = GeometryField(
        mesh,
        (x, y) -> normal[1] * (x - point[1]) +
            normal[2] * (y - point[2]),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - point[1], y - point[2]) - 0.16;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    liquid_violation = 0.0
    gas_violation = 0.0
    checked_partitions = 0
    for (index, face) in pairs(mesh.faces)
        aperture = geometry.aperture[index]
        aperture > 0 || continue
        for (donor, direction) in ((face.minus, 1.0), (face.plus, -1.0))
            open = TwoPhaseAMR._get(geometry.open_fraction, donor)
            open > 0 || continue
            target = 0.49open
            block = mesh.blocks[donor.block]
            velocity = direction * target * block.volume /
                (face.area * aperture)
            liquid_fraction = TwoPhaseAMR._swept_fraction(
                interface, face, donor, velocity, 1.0, target,
            )
            swept_liquid = target * liquid_fraction
            liquid = TwoPhaseAMR._get(interface.fraction, donor)
            gas = open - liquid
            liquid_violation = max(
                liquid_violation, -swept_liquid, swept_liquid - liquid,
            )
            swept_gas = target - swept_liquid
            gas_violation = max(
                gas_violation, -swept_gas, swept_gas - gas,
            )
            checked_partitions += 1
        end
    end
    @test checked_partitions > 0
    @test liquid_violation <= 512eps()
    @test gas_violation <= 512eps()
end

@testset "Cut-cell WY reverses wall-tangent contact-line translation" begin
    initial_sweep_order = (1, 2)
    resolutions = (16, 32, 64)
    # Exercise transport with the GHF fallback in ordinary mixed cells.
    results = [
        _sloped_wy_reverse_result(
            cells, initial_sweep_order;
            reconstruction_method = :lhf_ghf,
        )
            for cells in resolutions
    ]
    midpoint_errors = getproperty.(results, :midpoint_error)
    reverse_errors = getproperty.(results, :reverse_error)
    contact_errors = getproperty.(results, :contact_error)
    midpoint_rates = log2.(
        midpoint_errors[1:(end - 1)] ./ midpoint_errors[2:end],
    )
    reverse_rates = log2.(
        reverse_errors[1:(end - 1)] ./ reverse_errors[2:end],
    )
    println(
        "cut-cell WY n=$resolutions " *
            "midpoint_errors=$midpoint_errors " *
            "midpoint_rates=$midpoint_rates " *
            "reverse_errors=$reverse_errors " *
            "reverse_rates=$reverse_rates " *
            "contact_errors=$contact_errors " *
            "coarse_satellite_volume=$(results[1].satellite_volume) " *
            "coarse_satellite_fraction=$(results[1].satellite_volume / results[1].initial_volume)",
    )
    @test all(getproperty.(results, :bounded))
    @test all(getproperty.(results, :contact_count) .== 2)
    @test results[1].satellite_volume <= 1.0e-3 * results[1].initial_volume
    @test all(diff(contact_errors) .< 0)
    @test maximum(getproperty.(results, :maximum_volume_error)) <= 4096eps()
    @test all(
        result.maximum_divergence <= 256eps() * 128 for result in results
    )
    @test all(diff(midpoint_errors) .< 0)
    @test all(diff(reverse_errors) .< 0)
    @test all(midpoint_rates .>= 1.8)
    # A contact-cell crossing can make one adjacent pair pre-asymptotic. The
    # order across the full fourfold refinement is the robust reverse measure.
    reverse_order = log(reverse_errors[1] / reverse_errors[end]) / log(4)
    @test reverse_order >= 1.8
end

@testset "Geometry and phase PLIC transfer conserve physical volumes on AMR" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.18 - 0.15x)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.42) - 0.22;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    @test all(eachindex(mesh.blocks)) do block
        fraction = interior(interface.fraction, block)
        open = interior(geometry.open_fraction, block)
        all((-128eps() .<= fraction) .& (fraction .<= open .+ 128eps()))
    end
    for (index, face) in pairs(mesh.faces)
        liquid_aperture = (
            TwoPhaseAMR._face_aperture(interface, face, face.minus) +
                TwoPhaseAMR._face_aperture(interface, face, face.plus)
        ) / 2
        @test 0 <= liquid_aperture <= geometry.aperture[index] + 128eps()
    end
    solid_volume = integrate(geometry)
    liquid_volume = _open_phase_volume(interface, geometry)
    workspace = AMRWorkspace(mesh)
    register_adaptive_field!(workspace, interface)
    register_adaptive_field!(workspace, geometry)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    @test integrate(geometry) ≈ solid_volume atol = 5.0e-14
    @test _open_phase_volume(interface, geometry) ≈ liquid_volume atol = 5.0e-14
    @test all(eachindex(mesh.blocks)) do block
        fraction = interior(interface.fraction, block)
        open = interior(geometry.open_fraction, block)
        all((-128eps() .<= fraction) .& (fraction .<= open .+ 128eps()))
    end
    labels = [
        block.key.i == 0 && block.key.j == 0 ? :refine : :keep
            for block in mesh.blocks
    ]
    @test adapt!(workspace, labels; step = 2, force = true).changed
    @test isbalanced(mesh)
    @test integrate(geometry) ≈ solid_volume atol = 5.0e-14
    @test _open_phase_volume(interface, geometry) ≈ liquid_volume atol = 5.0e-14
    @test all(eachindex(mesh.blocks)) do block
        fraction = interior(interface.fraction, block)
        open = interior(geometry.open_fraction, block)
        all((-128eps() .<= fraction) .& (fraction .<= open .+ 128eps()))
    end

    # The cut metric symmetrically scales the existing non-diagonal AMR
    # vertex-patch metric instead of replacing it by a diagonal approximation.
    operator = PoissonOperator(geometry)
    @test operator.gradient_operator.inverse_metric isa
        TwoPhaseAMR.SparseInverseFaceMetric
    @test issymmetric(operator.matrix)
    velocity = FaceVelocity(mesh)
    for index in eachindex(velocity.interior)
        velocity.interior[index] = sin(0.902index)
    end
    pressure = CellField(mesh)
    solver = EllipticSolver(operator; preconditioner = :tree_geometric)
    projection = ProjectionWorkspace(solver)
    projection_dt = 0.01
    stats = project!(
        velocity, pressure, solver, projection_dt, projection, geometry;
        tolerance = 1.0e-10, max_cycles = 1000,
    )
    @test stats.converged
    residual = CellField(mesh)
    divergence!(residual, velocity, geometry)
    active_divergence = [
        abs(TwoPhaseAMR._get(residual, ref))
            for ref in TwoPhaseAMR._cell_references(mesh)
            if TwoPhaseAMR._get(geometry.open_fraction, ref) > 0
    ]
    reported_divergence = projection_dt * stats.final_linf
    @test maximum(active_divergence) <=
        reported_divergence + 256eps(eltype(geometry))
end

@testset "Combined two-phase cut-cell diffusion and solver smoke test" begin
    mesh = BlockTreeMesh(
        block_size = (12, 12), periodic = (true, true),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.18 - 0.1x)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.42) - 0.22;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    diffusion = StaggeredDiffusionOperator(
        interface; viscosity = (2.0, 0.4),
    )
    @test issymmetric(diffusion.stiffness)
    TwoPhaseAMR._update_dynamic_viscosity!(
        diffusion, interface, (1.5, 0.3),
    )
    @test issymmetric(diffusion.stiffness)

    solver = NavierStokesSystem(
        mesh; dynamic_viscosity = (2.0, 0.4), interface, geometry,
        density = (1.0, 0.1), pressure_preconditioner = nothing,
        diffusion_preconditioner = nothing,
    )
    velocity = FaceVelocity(mesh)
    pressure = CellField(mesh)
    workspace = NavierStokesWorkspace(solver)
    stats = step!(
        velocity, pressure, solver, 1.0e-4, workspace;
        tolerance = 1.0e-9, max_cycles = 1000,
    )
    @test stats.converged
    @test all(isfinite, velocity.interior)
    @test maximum(abs, velocity.interior) < 2.0e-12
end

# Reconstruct the uncondensed trace Jacobian from the open edge geometry.
# Box traces are shared by every Green row meeting a vertex, not independent
# vertex squares. PLIC traces and box traces must therefore be eliminated
# together in the weighted strain metric.
function _check_ls_stag_free_trace_projection(mesh, geometry, no_slip, free_slip; tilted)
    construction = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    refs = TwoPhaseAMR._cached_cell_references!(construction.cell_references, mesh)
    measures = copy(no_slip.stress_measures[1:(2 * length(refs))])
    cache = TwoPhaseAMR._ls_stress_topology!(TwoPhaseAMR._LSStressCache(Float64), geometry)
    free_construction = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    free_measures = copy(free_slip.stress_measures[1:(2 * length(refs))])
    free_cache = TwoPhaseAMR._ls_stress_topology!(TwoPhaseAMR._LSStressCache(Float64), geometry)
    retained_rows = collect(1:(2 * length(refs)))
    retained_free_rows = copy(retained_rows)
    dropped_rows = Int[]
    nrows = length(no_slip.stress_measures)
    box = Dict{Int, Vector{Float64}}()
    plic = Vector{Vector{Float64}}()
    aligned_walls = 0
    for (c, ref) in pairs(refs)
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        open > 0 || continue
        entries = cache.offsets[c]:(cache.offsets[c + 1] - 1)
        wall = !isone(open) || any(entries) do entry
            iszero(cache.faces[abs(cache.cell_faces[entry])].length)
        end
        regular = isone(open) && length(entries) == 4
        vertices = Int[]
        lengths = NTuple{2, Float64}[]
        for entry in entries
            face = cache.faces[abs(cache.cell_faces[entry])]
            for j in 1:2
                v = face.vertices[j]
                x, y = cache.lines[v]
                regular &= face.reaches[j] && x.regular && y.regular
                face.reaches[j] && x.active && y.active || continue
                pos = findfirst(==(v), vertices)
                if isnothing(pos)
                    push!(vertices, v)
                    push!(lengths, (0.0, 0.0))
                    pos = length(vertices)
                end
                lengths[pos] = Base.setindex(lengths[pos], face.length, face.axis)
            end
        end
        volume = open * mesh.blocks[ref.block].volume
        rectangles = sum(a * b / 4 for (a, b) in lengths; init = 0.0)
        scale = regular || iszero(rectangles) ? 1.0 : min(1.0, volume / (2rectangles))
        remainder = regular ? 0.0 : volume - scale * rectangles
        h = mesh.blocks[ref.block].spacing
        n1 = TwoPhaseAMR._get(geometry.solid.normal_x, ref) / h[1]
        n2 = TwoPhaseAMR._get(geometry.solid.normal_y, ref) / h[2]
        norm_n = hypot(n1, n2)
        sloped = !isone(open) && min(abs(n1), abs(n2)) > 64eps() * norm_n
        no_complete = remainder > 0
        free_complete = no_complete && (!wall || sloped)
        wall && !sloped && no_complete && (aligned_walls += 1)
        before = length(measures)
        free_before = length(free_measures)
        previous_vertices = count(!iszero, cache.vertex_rows)
        free_previous_vertices = count(!iszero, free_cache.vertex_rows)
        TwoPhaseAMR._ls_stress_cell!(cache, geometry, ref, c, measures, construction, NoSlip())
        TwoPhaseAMR._ls_stress_cell!(free_cache, geometry, ref, c, free_measures, free_construction, FreeSlip())
        new_vertices = count(!iszero, cache.vertex_rows) - previous_vertices
        free_new_vertices = count(!iszero, free_cache.vertex_rows) - free_previous_vertices
        @test new_vertices == free_new_vertices
        # NoSlip retains the aligned-wall completion; FreeSlip drops it.
        @test length(measures) - before == new_vertices + Int(no_complete)
        @test length(free_measures) - free_before == free_new_vertices + Int(free_complete)
        cell_row = no_complete ? before + 1 : 0
        if no_complete
            # The completion is inserted before any new vertex rows.
            @test isapprox(measures[cell_row], remainder; rtol = 128eps(), atol = 128eps())
            if free_complete
                push!(retained_rows, cell_row)
                push!(retained_free_rows, free_before + 1)
            else
                push!(dropped_rows, cell_row)
            end
        end
        for (pos, v) in pairs(vertices)
            area = scale * prod(lengths[pos]) / 4
            area > 0 || continue
            row = cache.vertex_rows[v]
            for line in cache.lines[v]
                (iszero(line.box_derivative) || iszero(line.columns[1])) && continue
                column = get!(box, v) do
                    zeros(nrows)
                end
                column[row] = -line.box_derivative
                no_complete && (column[cell_row] += area * line.box_derivative / remainder)
            end
        end
        if no_complete
            normal_flux = zeros(2)
            shear_flux = 0.0
            tangent = sloped ? (-n2 / norm_n, n1 / norm_n) : (0.0, 0.0)
            for entry in entries
                oriented = cache.cell_faces[entry]
                face = cache.faces[abs(oriented)]
                normal_flux[face.axis] += sign(oriented) * face.length
                for j in 1:2
                    face.reaches[j] || continue
                    v = face.vertices[j]
                    line = cache.lines[v][3 - face.axis]
                    if !iszero(line.box_derivative) && !iszero(line.columns[1])
                        column = get!(box, v) do
                            zeros(nrows)
                        end
                        column[cell_row] += sign(oriented) * face.length / (2remainder)
                    end
                    # The PLIC parameter has a unit endpoint trace even at
                    # a box line, independently of the box trace parameter.
                    endpoint = iszero(line.box_derivative) ? sum(line.trace) : 1.0
                    shear_flux += sign(oriented) * face.length *
                        tangent[3 - face.axis] * endpoint / 2
                end
            end
            if sloped
                column = zeros(nrows)
                column[2c - 1] = tangent[1] * normal_flux[1] / volume
                column[2c] = tangent[2] * normal_flux[2] / volume
                column[cell_row] = shear_flux / remainder
                push!(plic, column)
            end
        end
    end
    for v in eachindex(cache.vertex_rows)
        row, free_row = cache.vertex_rows[v], free_cache.vertex_rows[v]
        @test iszero(row) == iszero(free_row)
        iszero(row) && continue
        push!(retained_rows, row)
        push!(retained_free_rows, free_row)
    end
    # Isolated closed-face and Dirichlet rows are appended after LS rows.
    append!(retained_rows, (length(measures) + 1):nrows)
    append!(retained_free_rows, (length(free_measures) + 1):length(free_slip.stress_measures))
    @test length(retained_rows) == length(retained_free_rows)
    @test sort([retained_rows; dropped_rows]) == collect(1:nrows)
    @test sort(retained_free_rows) == collect(eachindex(free_slip.stress_measures))
    @test no_slip.stress_weights[retained_rows] == free_slip.stress_weights[retained_free_rows]
    surviving_E = Matrix(free_slip.strain[retained_free_rows, :])
    surviving_W = no_slip.stress_weights[retained_rows]
    @test maximum(
        abs, Matrix(free_slip.stiffness) -
            surviving_E' * (surviving_W .* surviving_E)
    ) <= 1024eps()
    @test length(dropped_rows) == aligned_walls
    dropped = [Float64.(collect(1:nrows) .== row) for row in dropped_rows]
    @test !isempty(box)
    if tilted
        ncut = count(ref -> 0 < TwoPhaseAMR._get(geometry.open_fraction, ref) < 1, refs)
        @test ncut > 0
        @test length(plic) == ncut
    else
        @test aligned_walls > 0
        @test isempty(plic)
    end
    B = hcat(collect(values(box))..., plic..., dropped...)
    A = Matrix(no_slip.strain)
    for column in plic
        # Every cut-cell trace must couple to at least one retained velocity.
        @test any(!iszero, A' * (no_slip.stress_weights .* column))
    end
    Wsqrt = sqrt.(no_slip.stress_weights)
    weighted_B = Wsqrt .* B
    weighted_A = Wsqrt .* A
    # SVD handles dependent trace columns without imposing an arbitrary gauge.
    released = weighted_A' * weighted_B * pinv(weighted_B) * weighted_A
    # Absolute 1024eps allows only accumulated Float64 round-off, not an
    # O(1) error in a wall coefficient or the shared-trace cross terms.
    @test maximum(abs, Matrix(no_slip.stiffness - free_slip.stiffness) - released) <= 1024eps()
    if tilted
        @test maximum(abs, hcat(collect(values(box))...)' * (no_slip.stress_weights .* hcat(plic...))) > 0
    end
    return nothing
end

@testset "Embedded free slip omits only physical wall-shear stresses" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.18 - 0.2x; lipschitz = hypot(1, 0.2),
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.45) - 0.2;
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )

    # Legacy wall-distance closure. It deletes the cut-wall shear rows, so
    # FreeSlip has strictly fewer stress rows; name the closure so a change
    # of default cannot silently change the reference.
    legacy_no_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = NoSlip(),
        closure = WallDistanceClosure(),
    )
    legacy_free_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = FreeSlip(),
        closure = WallDistanceClosure(),
    )
    @test legacy_no_slip.closure isa WallDistanceClosure
    @test legacy_free_slip.closure isa WallDistanceClosure
    @test issymmetric(legacy_free_slip.stiffness)
    @test size(legacy_free_slip.strain, 1) < size(legacy_no_slip.strain, 1)
    legacy_wall_shear = legacy_no_slip.stiffness - legacy_free_slip.stiffness
    @test any(value -> !iszero(value), legacy_wall_shear)
    @test minimum(eigvals(Symmetric(Matrix(legacy_wall_shear)))) >= -1024eps()

    # Default LS-STAG closure. Both stiffnesses are individually PSD, and
    # releasing the wall tangential trace cannot add dissipation: the actual
    # K_NoSlip - K_FreeSlip is PSD on this nonperiodic tilted wall. The
    # released trace is checked against a joint weighted Schur elimination
    # of the independent geometric trace columns below. Do not loosen these bounds.
    no_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = NoSlip(),
    )
    free_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = FreeSlip(),
    )
    @test no_slip.closure isa LSStagClosure
    @test free_slip.closure isa LSStagClosure
    @test issymmetric(free_slip.stiffness)
    for operator in (no_slip, free_slip)
        @test minimum(eigvals(Symmetric(Matrix(operator.stiffness)))) >=
            -1024eps()
    end
    wall_shear = no_slip.stiffness - free_slip.stiffness
    @test any(value -> !iszero(value), wall_shear)
    @test minimum(eigvals(Symmetric(Matrix(wall_shear)))) >= -1024eps()
    _check_ls_stag_free_trace_projection(mesh, geometry, no_slip, free_slip; tilted = true)

    # Changing the mixture next to the box/PLIC junction changes the weighted
    # trace minimizer, not just the diagonal stress weights.
    shifted_interface = InterfaceField(
        mesh, (x, y) -> y - 0.24 - 0.4x;
        geometry, lipschitz = hypot(1, 0.4),
    )
    refreshed_viscosity = (0.03, 0.2)
    old_weights = copy(free_slip.stress_weights)
    for operator in (no_slip, free_slip)
        TwoPhaseAMR._update_dynamic_viscosity!(
            operator, shifted_interface, refreshed_viscosity,
        )
    end
    @test any(!iszero, free_slip.stress_weights - old_weights)
    fresh_free_slip = StaggeredDiffusionOperator(
        shifted_interface; viscosity = refreshed_viscosity, boundary = FreeSlip(),
    )
    # Both comparisons allow only round-off from assembly and dense products.
    @test maximum(abs, Matrix(free_slip.stiffness - fresh_free_slip.stiffness)) <= 1024eps()
    E = Matrix(free_slip.strain)
    factorized = E' * (free_slip.stress_weights .* E)
    @test maximum(abs, Matrix(free_slip.stiffness) - factorized) <= 1024eps()
    @test minimum(eigvals(Symmetric(Matrix(no_slip.stiffness - free_slip.stiffness)))) >= -1024eps()
    _check_ls_stag_free_trace_projection(mesh, geometry, no_slip, free_slip; tilted = true)

    # A grid-aligned wall has closed faces but no PLIC completion. Check the
    # released box traces separately, including their PSD stiffness difference.
    aligned_geometry = GeometryField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    aligned_interface = InterfaceField(
        mesh, (x, y) -> y - 0.43; geometry = aligned_geometry, lipschitz = 1,
    )
    aligned_no_slip = StaggeredDiffusionOperator(
        aligned_interface; viscosity = (0.2, 0.03), boundary = NoSlip(),
    )
    aligned_free_slip = StaggeredDiffusionOperator(
        aligned_interface; viscosity = (0.2, 0.03), boundary = FreeSlip(),
    )
    aligned_loss = Matrix(aligned_no_slip.stiffness - aligned_free_slip.stiffness)
    @test any(!iszero, aligned_loss)
    @test minimum(eigvals(Symmetric(aligned_loss))) >= -1024eps()
    _check_ls_stag_free_trace_projection(
        mesh, aligned_geometry, aligned_no_slip, aligned_free_slip; tilted = false,
    )

    for closure in (WallDistanceClosure(), LSStagClosure())
        constant_free_slip = StaggeredDiffusionOperator(
            geometry; viscosity = 0.2, boundary = FreeSlip(), closure,
        )
        for component in (constant_free_slip.x, constant_free_slip.y)
            @test issymmetric(component.stiffness)
            @test minimum(eigvals(Symmetric(Matrix(component.stiffness)))) >= -1024eps()
        end
    end
end

@testset "Cut-cell AMR forces retain balanced pressure cancellation" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        periodic = (false, false),
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.12 - 0.1x)
    gravity_vector = (0.6, -0.8)
    interface = InterfaceField(
        mesh,
        (x, y) -> gravity_vector[1] * (x - 0.5) +
            gravity_vector[2] * (y - 0.55);
        geometry, contact_angle = pi / 2, lipschitz = 1,
    )
    density = (1000.0, 1.0)
    dt = 0.01

    function project_force!(velocity, operator)
        velocity.interior .*= dt
        solver = EllipticSolver(
            operator; preconditioner = :tree_geometric,
        )
        statistics = project!(
            velocity, CellField(mesh), solver, dt,
            ProjectionWorkspace(solver), geometry;
            tolerance = 1.0e-11, max_cycles = 1000,
        )
        @test statistics.converged
        @test maximum(abs, velocity.interior) < 1.0e-10
    end

    for formulation in (CSF(), CGFM())
        operator = PoissonOperator(
            interface, geometry; density,
        )
        @test reinterpret(UInt64, operator.face_coefficient) == reinterpret(
            UInt64,
            geometry.aperture .^ 2 ./ mixture_density(interface, density, geometry),
        )
        velocity = FaceVelocity(mesh)
        surface_tension!(
            velocity, interface, operator,
            EllipticWorkspace(operator);
            sigma = 1.0, formulation, curvature = 2.0,
        )
        project_force!(velocity, operator)
    end

    operator = PoissonOperator(interface, geometry; density)
    velocity = FaceVelocity(mesh)
    gravity!(
        velocity, interface, operator, EllipticWorkspace(operator);
        density, gravity = gravity_vector,
    )
    project_force!(velocity, operator)
end
