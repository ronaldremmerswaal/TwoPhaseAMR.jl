using Test, TwoPhaseAMR, LinearAlgebra

function _geometry_cell_values(field)
    return reduce(vcat, [vec(collect(interior(field, b))) for b in eachindex(field.mesh.blocks)])
end

function _open_phase_volume(interface, geometry)
    return sum(eachindex(interface.mesh.blocks)) do block
        sum(
            interior(interface.fraction, block) .*
                interior(geometry.open_fraction, block),
        ) * interface.mesh.blocks[block].volume
    end
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
        rtol = 1.0e-11, max_cycles = 1000,
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
        solid = InterfaceField(mesh)
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
    indicator = InterfaceRefinementIndicator(geometry.solid; level = 1)
    flow = FlowSimulation(
        mesh; viscosity = 0.01, geometry,
        refinement_indicators = (indicator,), dt = 1.0e-4,
    )
    old_preconditioner = flow.solver.pressure_solver.preconditioner
    result = simulate!(
        flow, 1.0e-4; rtol = 1.0e-10, max_cycles = 200,
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
        mesh; viscosity = 0.01, boundary, scheme = Centred(),
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
                    0 < TwoPhaseAMR._get(interface.fraction, ref) < 1
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
    solid_volume = integrate(geometry)
    liquid_volume = _open_phase_volume(interface, geometry)
    workspace = AMRWorkspace(mesh)
    register_adaptive_field!(workspace, interface)
    register_adaptive_field!(workspace, geometry)
    @test adapt!(workspace, [:refine]; step = 1, force = true).changed
    @test integrate(geometry) ≈ solid_volume atol = 5.0e-14
    @test _open_phase_volume(interface, geometry) ≈ liquid_volume atol = 5.0e-14
    labels = [
        block.key.i == 0 && block.key.j == 0 ? :refine : :keep
            for block in mesh.blocks
    ]
    @test adapt!(workspace, labels; step = 2, force = true).changed
    @test isbalanced(mesh)
    @test integrate(geometry) ≈ solid_volume atol = 5.0e-14
    @test _open_phase_volume(interface, geometry) ≈ liquid_volume atol = 5.0e-14

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
        rtol = 1.0e-10, max_cycles = 1000,
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
    roundoff_factor = 1 + eps(eltype(geometry))^(1 / 3)
    @test maximum(active_divergence) <=
        roundoff_factor * reported_divergence
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
        mesh; viscosity = (2.0, 0.4), interface, geometry,
        density = (1.0, 0.1), pressure_preconditioner = nothing,
        diffusion_preconditioner = nothing,
    )
    velocity = FaceVelocity(mesh)
    pressure = CellField(mesh)
    workspace = NavierStokesWorkspace(solver)
    stats = step!(
        velocity, pressure, solver, 1.0e-4, workspace;
        rtol = 1.0e-9, max_cycles = 1000,
    )
    @test stats.converged
    @test all(isfinite, velocity.interior)
    @test maximum(abs, velocity.interior) < 2.0e-12
end

@testset "Cut-cell AMR forces retain balanced pressure cancellation" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        periodic = (false, false),
        refine = (key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.12 - 0.10x)
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
            rtol = 1.0e-11, atol = 1.0e-12, max_cycles = 1000,
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
