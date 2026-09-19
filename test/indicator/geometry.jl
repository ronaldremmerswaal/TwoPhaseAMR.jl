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

function _embedded_contact_positions(
        interface, wall_normal, tangent, wall_point,
    )
    mesh = interface.mesh
    wall_intercept = sum(wall_normal .* wall_point)
    positions = Float64[]
    for ref in TwoPhaseAMR._cell_references(mesh)
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

function _sloped_wy_reverse_result(cells, initial_sweep_order)
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
        atol = 1.0e-9, maxdepth = 24,
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
    midpoint_error = _circle_open_symmetric_difference(
        interface, translated_centre, radius, wall_normal, wall_point,
    )
    contact_positions = _embedded_contact_positions(
        interface, wall_normal, tangent, wall_point,
    )
    exact_contacts = [
        -radius + speed * half_period,
        radius + speed * half_period,
    ]
    contact_error = length(contact_positions) == 2 ?
        maximum(abs.(contact_positions .- exact_contacts)) : Inf
    advance_to!(2half_period, -1.0)
    reverse_error = _circle_open_symmetric_difference(
        interface, wall_point, radius, wall_normal, wall_point,
    )
    return (;
        midpoint_error, reverse_error, contact_error,
        contact_count = length(contact_positions), maximum_volume_error,
        maximum_divergence, bounded,
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
        formulation = GFM(),
    )
    @test all(iszero, operator.face_coefficient[aligned_faces])
    acceleration = FaceVelocity(mesh)
    surface_tension!(
        acceleration, interface, operator, EllipticWorkspace(operator);
        sigma = 1.0, formulation = GFM(),
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
    @test minimum(orders) > 1.5
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
    @test nblocks(mesh) == 49

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
        mesh; density = 1.0, dynamic_viscosity = 0.01, geometry,
        refinement_indicators = (indicator,), dt = 1.0e-4,
    )
    old_preconditioner = flow.solver.pressure_solver.preconditioner
    result = simulate!(
        flow, 1.0e-4; tolerance = 1.0e-10, max_cycles = 200,
    )
    @test result.mesh_updates == 1
    rebuilt = flow.solver.pressure_solver.preconditioner
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

    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.01,
        boundary, scheme = Centred(),
    )
    pressure = CellField(mesh)
    workspace = NavierStokesWorkspace(solver)
    stats = step!(velocity, pressure, solver, 0.001, workspace)
    @test stats.converged
    @test maximum(abs, _geometry_cell_values(pressure)) < 2.0e-12
end

@testset "Embedded GHF contact curvature converges on a sloped wall" begin
    slope = 0.25
    intercept = 0.12
    centre = (0.5, 0.38)
    radius = 0.3
    distance = (
        centre[2] - intercept - slope * centre[1]
    ) / sqrt(1 + slope^2)
    contact_angle = acos(-distance / radius)
    errors = Float64[]
    for cells in (32, 64, 128)
        mesh = BlockTreeMesh(block_size = (cells, cells))
        geometry = GeometryField(
            mesh, (x, y) -> y - intercept - slope * x,
        )
        interface = InterfaceField(
            mesh,
            (x, y) -> hypot(x - centre[1], y - centre[2]) - radius;
            geometry, contact_angle, lipschitz = 1,
        )
        triple_cells = [
            ref for ref in TwoPhaseAMR._cell_references(mesh)
                if 0 < TwoPhaseAMR._get(geometry.solid.fraction, ref) < 1 &&
                0 < TwoPhaseAMR._conditional_fraction(interface, ref) < 1
        ]
        curvature = [
            TwoPhaseAMR._get(interface.curvature, ref) for ref in triple_cells
        ]
        @test length(curvature) >= 2
        @test all(isfinite, curvature)
        @test all(
            TwoPhaseAMR._get(interface.reconstruction, ref) == 2
                for ref in triple_cells
        )
        push!(errors, maximum(abs.(curvature .- inv(radius))))
    end
    @test log2(errors[end - 1] / errors[end]) > 1
end

@testset "Embedded height reconstruction converges in symmetric difference" begin
    wall_angle = pi / 12
    wall_normal = (-sin(wall_angle), cos(wall_angle))
    wall_point = (0.5, 0.22)
    centre = wall_point
    radius = 0.24
    errors = Float64[]
    for cells in (24, 48, 96)
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
        # Contact-cell PLIC normals use the unconstrained generalized height
        # fit; only their separately computed curvature uses the wall angle.
        # No optimization-based LVIRA reconstruction is involved.
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
            "n=(24,48,96) errors=$errors rates=$rates",
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

@testset "Embedded contact angle changes curvature, not the PLIC plane" begin
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
    first = InterfaceField(
        copy(fraction); geometry, embedded_contact_angle = pi / 3,
    )
    second = InterfaceField(
        copy(fraction); geometry, embedded_contact_angle = 2pi / 3,
    )
    curvature_changed = false
    for ref in TwoPhaseAMR._cell_references(mesh)
        c = TwoPhaseAMR._conditional_fraction(first, ref)
        0 < c < 1 || continue
        @test TwoPhaseAMR._get(first.normal_x, ref) ≈
            TwoPhaseAMR._get(second.normal_x, ref) atol = 32eps()
        @test TwoPhaseAMR._get(first.normal_y, ref) ≈
            TwoPhaseAMR._get(second.normal_y, ref) atol = 32eps()
        @test TwoPhaseAMR._get(first.intercept, ref) ≈
            TwoPhaseAMR._get(second.intercept, ref) atol = 32eps()
        solid = TwoPhaseAMR._get(geometry.solid.fraction, ref)
        0 < solid < 1 || continue
        k1 = TwoPhaseAMR._get(first.curvature, ref)
        k2 = TwoPhaseAMR._get(second.curvature, ref)
        isfinite(k1) && isfinite(k2) &&
            (curvature_changed |= k1 != k2)
    end
    @test curvature_changed
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
            @test -512eps() <= swept_liquid <= liquid + 512eps()
            @test -512eps() <= target - swept_liquid <= gas + 512eps()
        end
    end
end

@testset "Cut-cell WY reverses wall-tangent contact-line translation" begin
    for initial_sweep_order in ((1, 2), (2, 1))
        results = [
            _sloped_wy_reverse_result(cells, initial_sweep_order)
                for cells in (32, 64, 128)
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
            "cut-cell WY initial_order=$initial_sweep_order " *
                "midpoint_errors=$midpoint_errors " *
                "midpoint_rates=$midpoint_rates " *
                "reverse_errors=$reverse_errors " *
                "reverse_rates=$reverse_rates " *
                "contact_errors=$contact_errors",
        )
        @test all(getproperty.(results, :bounded))
        @test all(getproperty.(results, :contact_count) .== 2)
        @test all(diff(contact_errors) .< 0)
        @test maximum(getproperty.(results, :maximum_volume_error)) <=
            4096eps()
        @test all(
            result.maximum_divergence <= 256eps() * 128
                for result in results
        )
        @test all(diff(midpoint_errors) .< 0)
        @test all(diff(reverse_errors) .< 0)
        @test all(midpoint_rates .>= 1.8)
        # A contact-cell crossing can make one adjacent pair pre-asymptotic.
        # The order across the full fourfold refinement remains the meaningful
        # reverse-test estimate.
        reverse_order = log(reverse_errors[1] / reverse_errors[end]) /
            log(4)
        @test reverse_order >= 1.8
    end
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

    solver = IncompressibleNavierStokesSolver(
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
    no_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = NoSlip(),
    )
    free_slip = StaggeredDiffusionOperator(
        interface; viscosity = (0.2, 0.03), boundary = Slip(),
    )
    @test issymmetric(free_slip.stiffness)
    @test size(free_slip.strain, 1) < size(no_slip.strain, 1)
    wall_shear = no_slip.stiffness - free_slip.stiffness
    @test any(value -> !iszero(value), wall_shear)
    @test minimum(eigvals(Symmetric(Matrix(wall_shear)))) >= -1024eps()

    constant_free_slip = StaggeredDiffusionOperator(
        geometry; viscosity = 0.2, boundary = Slip(),
    )
    for component in (constant_free_slip.x, constant_free_slip.y)
        @test issymmetric(component.stiffness)
        @test minimum(eigvals(Symmetric(Matrix(component.stiffness)))) >=
            -1024eps()
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

    for formulation in (CSF(), GFM())
        operator = PoissonOperator(
            interface, geometry; density, formulation,
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

@testset "Visualization frames retain PLIC interface and solid overlays" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2,
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.6; geometry,
    )
    registry = VisualizationFieldRegistry()
    register_default_fields!(registry, :volume_fraction, interface)
    @test registry.interfaces == [interface]
    @test registry.geometries == [geometry]
    frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    @test !isempty(frame.primitives[interface])
    @test !isempty(frame.primitives[geometry].triangles)
end
