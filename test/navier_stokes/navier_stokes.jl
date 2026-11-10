using LinearAlgebra: Diagonal, diag, dot, isdiag
using SparseArrays: nonzeros

isdefined(@__MODULE__, :test_cell_values) ||
    include(joinpath(@__DIR__, "..", "test_utils.jl"))

function _velocity_l2_error(velocity, exact, time, solver)
    squared = 0.0
    total_volume = 0.0
    for component in (solver.diffusion.x, solver.diffusion.y)
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            volume = component.volumes[local_index]
            face = velocity.mesh.faces[global_index]
            value = exact(face.center[1], face.center[2], time, face.axis)
            squared += volume * (velocity.interior[global_index] - value)^2
            total_volume += volume
        end
    end
    return sqrt(squared / total_volume)
end

@testset "pressure failure reports convergence diagnostics" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? sinpi(2x) : cospi(2y),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.5)
    pressure = CellField(mesh)
    solver = IncompressibleNavierStokesSolver(
        mesh; interface, density = (1.0, 1.0),
        dynamic_viscosity = (0.0, 0.0),
        boundary = FreeSlip(), coarse_size = 4,
    )
    workspace = NavierStokesWorkspace(solver)
    exception = try
        simulate!(
            velocity, pressure, solver, 0.01, workspace;
            dt = 0.01, tolerance = 1.0e-12, max_cycles = 0,
        )
        nothing
    catch caught
        caught
    end
    @test exception isa ErrorException
    message = sprint(showerror, exception)
    for field in (
            "pressure projection failed to converge", "tolerance=",
            "dimensionless_residual=", "dt=", "time=", "max_cycles=0",
            "cycles=", "converged=false", "initial_l2=", "initial_linf=",
            "final_l2=", "final_linf=", "target_linf=",
            "residual_scale=", "removed_rhs_mean=", "method=",
            "preconditioner=",
        )
        @test occursin(field, message)
    end
end

@testset "short step does not turn inherited divergence into pressure" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    initial = (x, y, time, axis) -> axis == 1 ? 1.0e-5sinpi(2x) : 0.0
    force = (x, y, time, axis) -> axis == 1 ? 0.1time * sinpi(2x) : 0.0
    initial_velocity = FaceVelocity(mesh, initial)
    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 1.0e-3, coarse_size = 8,
    )
    workspace = NavierStokesWorkspace(solver)
    full_velocity = copy(initial_velocity)
    short_velocity = copy(initial_velocity)
    full_pressure = CellField(mesh)
    short_pressure = CellField(mesh)
    full_dt = 0.01
    ratio = 0.1
    full_result = simulate!(
        full_velocity, full_pressure, solver, full_dt, workspace;
        dt = full_dt, force, tolerance = 1.0e-12, max_cycles = 100,
    )
    @test full_result.pressure.converged
    short_result = simulate!(
        short_velocity, short_pressure, solver, ratio * full_dt, workspace;
        dt = full_dt, force, tolerance = 1.0e-12, max_cycles = 100,
    )
    @test short_result.pressure.converged
    full_change = Array(interior(full_pressure, 1))
    short_change = Array(interior(short_pressure, 1))
    local_variation = maximum(abs, full_change)
    excursion = maximum(abs, short_change .- ratio .* full_change)
    @test local_variation > 0
    @test excursion <= 3local_variation
    initial_divergence = CellField(mesh)
    short_divergence = CellField(mesh)
    divergence!(initial_divergence, initial_velocity)
    divergence!(short_divergence, short_velocity)
    @test maximum(
        abs,
        interior(short_divergence, 1) .-
            (1 - ratio) .* interior(initial_divergence, 1),
    ) <= 2.0e-9

    # Exact visualization checkpoints use an independent copy of the step;
    # it must inherit the physical proposal, not its own shorter interval.
    source_workspace = NavierStokesWorkspace(solver)
    options = (; tolerance = 1.0e-12, max_cycles = 100)
    procedure = TwoPhaseAMR._independent_step_procedure(
        solver, force, options, source_workspace,
    )
    TwoPhaseAMR._prepare_independent_step!(
        procedure, solver, source_workspace, full_dt,
    )
    checkpoint_velocity = copy(initial_velocity)
    checkpoint_pressure = CellField(mesh)
    checkpoint_stats = procedure(
        checkpoint_velocity, checkpoint_pressure, ratio * full_dt; time = 0.0,
    )
    expected_velocity = copy(initial_velocity)
    expected_pressure = CellField(mesh)
    expected_stats = simulate!(
        expected_velocity, expected_pressure, solver, ratio * full_dt,
        source_workspace; dt = full_dt, force, options...,
    ).pressure
    @test checkpoint_stats.converged && expected_stats.converged
    @test checkpoint_velocity.interior ≈ expected_velocity.interior atol = 1.0e-11
    @test interior(checkpoint_pressure, 1) ≈ interior(expected_pressure, 1) atol = 1.0e-10
end

@testset "split-domain projection retains physical component flux" begin
    # Opposing prescribed inflows balance globally but not in either
    # disconnected component (issues #95/#97). Unlike an outflow boundary,
    # these values are unchanged when the boundary is reimposed after project!.
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), periodic = (false, false),
    )
    geometry = GeometryField(mesh, (x, y) -> abs(y - 0.46) - 0.17; lipschitz = 1)
    operator = PoissonOperator(geometry)
    volumes, refs = operator.volumes, operator.references
    ys = [cellcenter(mesh, ref.block, ref.i, ref.j)[2] for ref in refs]
    lower = findall(i -> volumes[i] > 0 && ys[i] < 0.5, eachindex(volumes))
    upper = findall(i -> volumes[i] > 0 && ys[i] > 0.5, eachindex(volumes))
    inactive = findall(iszero, volumes)
    components = (lower, upper)
    anchors = sort([minimum(cells) for cells in components])
    boundary_area(cells_are_lower) = sum(
        face.area * geometry.boundary_aperture[index]
            for (index, face) in pairs(mesh.boundary_faces)
            if face.axis == 1 && face.side == -1 &&
            (face.center[2] < 0.5) == cells_are_lower
    )
    lower_area, upper_area = boundary_area(true), boundary_area(false)
    @test lower_area > 0 && upper_area > 0 && !isempty(inactive)
    upper_speed = -0.2 * lower_area / upper_area
    boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, time, axis) -> y < 0.5 ? 0.2 : upper_speed),
        right = NoSlip(), bottom = NoSlip(), top = NoSlip(),
    )
    predictor = FaceVelocity(
        mesh, (x, y, time, axis) ->
        axis == 1 ? 0.1sinpi(2x) * cospi(2y) : 0.0,
    )
    prepared = copy(predictor)
    enforce_velocity_boundary!(prepared, boundary; geometry)
    enforce_geometry_no_slip!(prepared, geometry)
    initial_divergence = CellField(mesh)
    divergence!(initial_divergence, prepared, geometry)
    initial_values = test_cell_values(initial_divergence)
    dt = 0.1
    means = [
        -dot(volumes[cells], initial_values[cells]) / (dt * sum(volumes[cells]))
            for cells in components
    ]
    mean_by_anchor = Dict(minimum(cells) => mean for (cells, mean) in zip(components, means))
    expected_means = [mean_by_anchor[anchor] for anchor in anchors]
    @test abs(dot(volumes, initial_values)) < 1.0e-12
    @test minimum(abs, means) > 1.0

    for specification in (nothing, :tree_geometric)
        @testset "$(specification)" begin
            options = isnothing(specification) ? (;) : (; geometry)
            solver = EllipticSolver(
                operator; preconditioner = specification,
                preconditioner_options = options, coarse_size = 16,
            )
            workspace = ProjectionWorkspace(solver)
            @test_throws ArgumentError project!(
                copy(predictor), CellField(mesh), solver, dt, workspace, geometry;
                velocity_boundary = boundary, compatibility = :error,
            )
            pressures = Vector{Float64}[]
            for (lower_offset, upper_offset) in ((0.0, 0.0), (11.0, -7.0))
                velocity = copy(predictor)
                pressure = CellField(
                    mesh, (x, y) -> y < 0.5 ? lower_offset : upper_offset,
                )
                for index in inactive
                    TwoPhaseAMR._set!(pressure, 23.0, refs[index])
                end
                stats = if iszero(lower_offset)
                    @test_logs (:warn,) project!(
                        velocity, pressure, solver, dt, workspace, geometry;
                        velocity_boundary = boundary, compatibility = :project,
                        tolerance = 1.0e-10, max_cycles = 500,
                    )
                else
                    @test_logs project!(
                        velocity, pressure, solver, dt, workspace, geometry;
                        velocity_boundary = boundary, compatibility = :project,
                        tolerance = 1.0e-10, max_cycles = 500,
                    )
                end
                values = test_cell_values(pressure)
                push!(pressures, values)
                final_divergence = CellField(mesh)
                divergence!(final_divergence, velocity, geometry)
                final_values = test_cell_values(final_divergence)
                @test stats.converged
                @test !stats.compatible
                @test stats.removed_rhs_component_anchors == anchors
                @test stats.removed_rhs_component_means ≈ expected_means atol = 1.0e-11 rtol = 0
                @test abs(stats.removed_rhs_mean) < 1.0e-11
                @test stats.target_linf <= 2.0e-8
                @test stats.final_linf <= 2.0e-8
                @test maximum(abs, values) < 10.0
                @test all(iszero, values[inactive])
                for (cells, mean) in zip(components, means)
                    @test abs(dot(volumes[cells], values[cells]) / sum(volumes[cells])) < 1.0e-12
                    # D u_new = -dt*removed_mean plus the true residual.
                    # 3e-9 includes dt*(tolerance/dt^2) and rounding.
                    @test maximum(abs, final_values[cells] .+ dt * mean) < 3.0e-9
                    @test abs(dot(volumes[cells], final_values[cells]) / sum(volumes[cells])) > 0.1
                end
                @test abs(dot(volumes, final_values)) < 1.0e-12
            end
            # Incoming per-component constants are pure gauge and covered
            # values have no physical meaning.
            @test pressures[1] ≈ pressures[2] atol = 1.0e-10 rtol = 0
        end
    end

    # Matching inflow on both sides balances each component separately.
    balanced_boundary = VelocityBoundaryConditions(
        left = boundary.left, right = Inflow(boundary.left.velocity),
        bottom = NoSlip(), top = NoSlip(),
    )
    solver = EllipticSolver(operator; preconditioner = nothing, coarse_size = 16)
    workspace = ProjectionWorkspace(solver)
    balanced = @test_logs project!(
        copy(predictor), CellField(mesh), solver, dt, workspace, geometry;
        velocity_boundary = balanced_boundary, tolerance = 1.0e-10, max_cycles = 500,
    )
    @test balanced.converged && balanced.compatible
    @test maximum(abs, balanced.removed_rhs_component_means) < 1.0e-10

    # The inherited-divergence policy halves this RHS for a reference step
    # twice as large. The means are strong values, but the physical comparison
    # uses dt^2, not an unscaled RHS value or the algebraic residual target.
    scaled_mean = dt^2 * maximum(abs, expected_means)
    @test 0 < scaled_mean < 1
    threshold = 0.75 * scaled_mean
    shortened = @test_logs project!(
        copy(predictor), CellField(mesh), solver, dt, workspace, geometry;
        velocity_boundary = boundary, tolerance = threshold, max_cycles = 500,
        _inherited_divergence = initial_divergence, _reference_dt = 2dt,
    )
    @test shortened.converged && shortened.compatible
    @test maximum(abs, shortened.removed_rhs_component_means) > threshold
    @test dt^2 * maximum(abs, shortened.removed_rhs_component_means) < threshold
    @test shortened.removed_rhs_component_means ≈
        expected_means ./ 2 atol = 1.0e-11 rtol = 0
    full = @test_logs (:warn,) project!(
        copy(predictor), CellField(mesh), solver, dt, workspace, geometry;
        velocity_boundary = boundary, tolerance = threshold, max_cycles = 500,
    )
    @test full.converged && !full.compatible
    @test full.removed_rhs_component_means ≈ expected_means atol = 1.0e-11 rtol = 0

    # The physical-step API and simulation result expose the same decision.
    # Multiple incompatible steps and a subsequent call share one warning.
    make_flow(bc) = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        geometry, boundary = bc,
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 16)),
    )
    step_flow = make_flow(boundary)
    step_stats = @test_logs (:warn,) step!(
        step_flow.velocity, step_flow.pressure, step_flow.solver, 0.001,
        step_flow.workspace; tolerance = 1.0e-9, max_cycles = 500,
    )
    @test step_stats.converged && !step_stats.compatible

    flow = make_flow(boundary)
    result = @test_logs (:warn,) simulate!(
        flow, 0.003; dt = 0.001, tolerance = 1.0e-9, max_cycles = 500,
    )
    @test result.steps == 3
    @test result.pressure.converged && !result.pressure.compatible
    @test :compatible in TwoPhaseAMR._projection_summary_fields(result.pressure)
    @test :removed_rhs_component_means in TwoPhaseAMR._projection_summary_fields(result.pressure)
    resumed = @test_logs simulate!(
        flow, 0.004; dt = 0.001, tolerance = 1.0e-9, max_cycles = 500,
    )
    @test resumed.steps == 1
    @test resumed.pressure.converged && !resumed.pressure.compatible

    balanced_flow = make_flow(balanced_boundary)
    balanced_result = @test_logs simulate!(
        balanced_flow, 0.002; dt = 0.001, tolerance = 1.0e-9, max_cycles = 500,
    )
    @test balanced_result.steps == 2
    @test balanced_result.pressure.converged && balanced_result.pressure.compatible
    @test :compatible in TwoPhaseAMR._projection_summary_fields(balanced_result.pressure)
end

@testset "connected projection preserves physical pressure mean" begin
    for cut in (false, true)
        mesh = BlockTreeMesh(
            block_size = (8, 8), periodic = cut ? (false, false) : (true, true),
        )
        geometry = cut ? GeometryField(
                mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.14;
                lipschitz = 1,
            ) : nothing
        operator = cut ? PoissonOperator(geometry) : PoissonOperator(mesh)
        solver = EllipticSolver(operator; coarse_size = 8)
        workspace = ProjectionWorkspace(solver)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sinpi(2x) : cospi(2y),
        )
        pressure = CellField(mesh, (x, y) -> 4.25 + 0.1cospi(2x) * cospi(2y))
        before = test_cell_values(pressure)
        stats = if cut
            @test_logs project!(
                velocity, pressure, solver, 0.1, workspace, geometry;
                tolerance = 1.0e-11, max_cycles = 500,
            )
        else
            @test_logs project!(
                velocity, pressure, solver, 0.1, workspace;
                tolerance = 1.0e-11, max_cycles = 500,
            )
        end
        @test stats.converged && stats.compatible
        @test length(stats.removed_rhs_component_anchors) == 1
        volumes = operator.volumes
        after = test_cell_values(pressure)
        # The solved increment has zero volume mean, so the old nonzero
        # physical-pressure mean survives to roundoff in either geometry.
        @test abs(dot(volumes, after - before) / sum(volumes)) < 1.0e-10
        inactive = findall(iszero, volumes)
        @test after[inactive] == before[inactive]
    end
end

@testset "uncut projection reports incompatible component flux once" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    coefficient = [
        face.axis == 1 && face.center[1] == 0.5 ? 0.0 : 1.0
            for face in mesh.faces
    ]
    solver = EllipticSolver(
        PoissonOperator(mesh; face_coefficient = coefficient); coarse_size = 8,
    )
    workspace = ProjectionWorkspace(solver)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 && x == 0.5 ? 0.2 : 0.0,
    )
    for first_projection in (true, false)
        stats = if first_projection
            @test_logs (:warn,) project!(
                copy(velocity), CellField(mesh), solver, 0.1, workspace;
                tolerance = 1.0e-10, max_cycles = 500,
            )
        else
            @test_logs project!(
                copy(velocity), CellField(mesh), solver, 0.1, workspace;
                tolerance = 1.0e-10, max_cycles = 500,
            )
        end
        @test stats.converged && !stats.compatible
        @test :compatible in propertynames(stats)
        @test 0.1^2 * maximum(abs, stats.removed_rhs_component_means) > 1.0e-10
    end
    # Reusing the elliptic solver outside projection must not retain the
    # previous physical-projection decision.
    generic = solve!(CellField(mesh), solver, CellField(mesh))
    @test generic.converged && generic.compatible
end

@testset "increment predictor preserves open-boundary projection" begin
    for cut in (false, true)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
        geometry = cut ? GeometryField(
                mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.14;
                lipschitz = 1,
            ) : nothing
        inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
        boundary = VelocityBoundaryConditions(
            left = Inflow(inflow), right = Outflow(),
            bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
        )
        solver = IncompressibleNavierStokesSolver(
            mesh; density = 1.0, dynamic_viscosity = 0.01, geometry, boundary,
            coarse_size = 8,
        )
        workspace = NavierStokesWorkspace(solver)
        initial = FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? 1 + 0.1sinpi(2x) : 0.1cospi(2y),
        )
        reference_velocity = copy(initial)
        predicted_velocity = copy(initial)
        reference_pressure = CellField(mesh)
        old_pressure = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
        dt = 0.01
        if cut
            # The convected predictor is not solid-masked until after
            # diffusion and forcing; masking it early changes cut-cell flow.
            tentative = FaceVelocity(mesh, (x, y, time, axis) -> 1.0)
            old = copy(tentative)
            blocked = findall(iszero, geometry.aperture)
            @test !isempty(blocked)
            TwoPhaseAMR._enforce_predictor_velocity_boundary!(tentative, old, solver, dt)
            @test all(==(1.0), tentative.interior[blocked])
            TwoPhaseAMR._enforce_predictor_velocity_boundary!(
                tentative, old, solver, dt; enforce_geometry = true,
            )
            @test all(iszero, tentative.interior[blocked])
        end
        TwoPhaseAMR._enforce_current_velocity_boundary!(reference_velocity, solver, dt)
        if cut
            @test_throws ArgumentError project!(
                reference_velocity, reference_pressure, solver.pressure_solver,
                dt, workspace.projection, geometry; _reference_dt = dt / 2,
            )
        else
            @test_throws ArgumentError project!(
                reference_velocity, reference_pressure, solver.pressure_solver,
                dt, workspace.projection; _reference_dt = dt / 2,
            )
        end
        options = (;
            velocity_boundary = boundary, boundary_time = dt,
            tolerance = 1.0e-13, max_cycles = 100,
        )
        full_stats = if cut
            project!(
                reference_velocity, reference_pressure, solver.pressure_solver,
                dt, workspace.projection, geometry; options...,
            )
        else
            project!(
                reference_velocity, reference_pressure, solver.pressure_solver,
                dt, workspace.projection; options...,
            )
        end
        @test full_stats.converged
        TwoPhaseAMR._enforce_current_velocity_boundary!(predicted_velocity, solver, dt)
        TwoPhaseAMR._pressure_predictor!(
            predicted_velocity, old_pressure, solver, workspace, dt,
        )
        incremental_stats = if cut
            TwoPhaseAMR._project_prepared!(
                predicted_velocity, old_pressure, solver.pressure_solver,
                dt, workspace.projection, geometry; options...,
            )
        else
            TwoPhaseAMR._project_prepared!(
                predicted_velocity, old_pressure, solver.pressure_solver,
                dt, workspace.projection; options...,
            )
        end
        @test incremental_stats.converged
        @test predicted_velocity.interior ≈
            reference_velocity.interior atol = 1.0e-10 rtol = 0
        @test predicted_velocity.boundary ≈
            reference_velocity.boundary atol = 1.0e-10 rtol = 0
        references = solver.pressure_solver.operator.references
        active = findall(>(0), solver.pressure_solver.operator.volumes)
        difference = [
            TwoPhaseAMR._get(old_pressure, references[index]) -
                TwoPhaseAMR._get(reference_pressure, references[index])
                for index in active
        ]
        @test maximum(abs, difference) <= 2.0e-7
    end
end

@testset "open channel pressure flux and divergence" begin
    # This is the non-square, outlet-refined channel from the projection
    # reference, including a nontrivial transverse predictor.
    profile(y) = 4y * (1 - y)
    for open in (Outflow(), ZeroGradientVelocity()), mixed in (false, true),
            preconditioner in (:tree_geometric, nothing)
        mesh = BlockTreeMesh(
            base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> mixed && key.level == 0 && key.i == 3,
        )
        boundary = VelocityBoundaryConditions(
            left = Inflow((x, y, t, axis) -> axis == 1 ? profile(y) : 0.0),
            right = open, bottom = NoSlip(), top = NoSlip(),
        )
        system = IncompressibleNavierStokesSolver(
            mesh; density = 2.0, dynamic_viscosity = 0.01,
            boundary, pressure_preconditioner = preconditioner, coarse_size = 8,
        )
        workspace = NavierStokesWorkspace(system)
        velocity = FaceVelocity(
            mesh, (x, y, t, axis) -> axis == 1 ?
                profile(y) + 0.3sinpi(2x) * sinpi(y) :
                0.2sinpi(x) * sinpi(2y),
        )
        enforce_velocity_boundary!(velocity, boundary)
        predictor = copy(velocity)
        pressure = CellField(mesh)
        dt = 0.01
        stats = project!(
            velocity, pressure, system.pressure_solver, dt, workspace.projection;
            velocity_boundary = boundary, tolerance = 1.0e-12, max_cycles = 1000,
            compatibility = :error,
        )
        @test stats.converged
        @test iszero(stats.removed_rhs_mean)
        operator = system.pressure_solver.operator
        @test !operator.singular
        increment = workspace.projection.increment
        open_change = 0.0
        for (index, face) in pairs(mesh.boundary_faces)
            if face.axis == 1 && face.side > 0
                spacing = mesh.blocks[face.cell.block].spacing[face.axis]
                gradient = -face.side * (1 / 2.0) *
                    TwoPhaseAMR._get(increment, face.cell) / (spacing / 2)
                @test workspace.projection.weighted_gradient.boundary[index] ≈ gradient atol = 1.0e-12
                @test velocity.boundary[index] ≈
                    predictor.boundary[index] - dt * gradient atol = 1.0e-12
                open_change = max(open_change, abs(velocity.boundary[index] - predictor.boundary[index]))
            else
                @test velocity.boundary[index] == predictor.boundary[index]
                @test iszero(workspace.projection.weighted_gradient.boundary[index])
            end
        end
        @test open_change > 1.0e-8
        divergence = CellField(mesh)
        divergence!(divergence, velocity)
        outlet = Set(
            face.cell for face in mesh.boundary_faces if face.axis == 1 && face.side > 0
        )
        near = maximum(abs(TwoPhaseAMR._get(divergence, ref)) for ref in outlet)
        elsewhere = maximum(
            abs(TwoPhaseAMR._get(divergence, ref))
                for ref in operator.references if !(ref in outlet)
        )
        if mixed && preconditioner === :tree_geometric
            @test dt * near <= 10dt * elsewhere
        end
        @test dt * max(near, elsewhere) ≈ dt^2 * stats.final_linf atol = 2.0e-14

        # A second projection must use, not re-extrapolate, the corrected flux.
        projected = copy(velocity)
        # Diffusion does not carry boundary storage. Its predictor must
        # extrapolate only the interior change, not reset the projected flux.
        for change in (0.0, 0.125)
            tentative = copy(projected)
            fill!(tentative.boundary, 0.0)
            for (index, face) in pairs(mesh.faces)
                face.axis == 1 && (tentative.interior[index] += change)
            end
            TwoPhaseAMR._enforce_predictor_velocity_boundary!(
                tentative, projected, system, 0.0,
            )
            for (index, face) in pairs(mesh.boundary_faces)
                if face.axis == 1 && face.side > 0
                    @test tentative.boundary[index] ≈ projected.boundary[index] + change atol = 1.0e-14
                else
                    @test tentative.boundary[index] == projected.boundary[index]
                end
            end
        end
        again = project!(
            velocity, CellField(mesh), system.pressure_solver, dt, workspace.projection;
            velocity_boundary = boundary, tolerance = 1.0e-12, max_cycles = 1000,
        )
        @test again.converged
        @test velocity.interior ≈ projected.interior atol = 2.0e-10 rtol = 0
        @test velocity.boundary ≈ projected.boundary atol = 2.0e-10 rtol = 0
    end
end

@testset "open projected flux is retained in shortened-step history" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, t, axis) -> axis == 1 ? 1.0 : 0.0),
        right = Outflow(), bottom = NoSlip(), top = NoSlip(),
    )
    system = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.01, boundary, coarse_size = 8,
    )
    workspace = NavierStokesWorkspace(system)
    velocity = FaceVelocity(
        mesh, (x, y, t, axis) -> axis == 1 ?
            1 + 0.2sinpi(x) * cospi(y) : 0.1cospi(x) * sinpi(2y),
    )
    enforce_velocity_boundary!(velocity, boundary)
    pressure = CellField(mesh)
    @test project!(
        velocity, pressure, system.pressure_solver, 0.01, workspace.projection;
        velocity_boundary = boundary, tolerance = 1.0e-12, max_cycles = 300,
    ).converged
    old = copy(velocity)
    stats = step!(
        velocity, pressure, system, 0.001, workspace;
        _reference_dt = 0.01, tolerance = 1.0e-10, max_cycles = 300,
    )
    @test stats.converged
    @test workspace.old_velocity.boundary == old.boundary
    @test maximum(
        abs(TwoPhaseAMR._get(workspace.projection.inherited_divergence, ref))
            for ref in system.pressure_solver.operator.references
    ) <= 2.0e-9
end

@testset "open pressure boundary survives a topology rebuild" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    marker = CellField(mesh; init = 1.0)
    indicator = InterpolationErrorIndicator(
        marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4, minimum_level = 1,
    )
    boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, t, axis) -> axis == 1 ? 1.0 : 0.0),
        right = Outflow(), bottom = FreeSlip(), top = FreeSlip(),
    )
    system = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.01, boundary,
        refinement_indicators = indicator, coarse_size = 8,
    )
    workspace = NavierStokesWorkspace(system)
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 1.0 : 0.0)
    pressure = CellField(mesh)
    enforce_velocity_boundary!(velocity, boundary)
    result = simulate!(
        velocity, pressure, system, 0.001, workspace;
        dt = 0.001, tolerance = 1.0e-10, max_cycles = 300,
    )
    @test result.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test !system.pressure_solver.operator.singular
    @test !system.pressure_solver.preconditioner.coarse_augmented
    @test system.pressure_solver.operator.boundary_sides == (false, true, false, false)
    stats = project!(
        velocity, CellField(mesh), system.pressure_solver, 0.01, workspace.projection;
        velocity_boundary = boundary, tolerance = 1.0e-11, max_cycles = 300,
    )
    @test stats.converged
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test maximum(
        abs(TwoPhaseAMR._get(divergence, ref))
            for ref in system.pressure_solver.operator.references
    ) <= 2.0e-8
end

function _taylor_green_error(nx; mixed = false)
    mesh = mixed ? BlockTreeMesh(
            block_size = (nx, nx), extent = (2pi, 2pi), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ) : BlockTreeMesh(block_size = (nx, nx), extent = (2pi, 2pi))
    viscosity = 0.02
    exact = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    velocity = FaceVelocity(mesh, exact)
    pressure = CellField(mesh)
    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = viscosity,
        boundary = FreeSlip(), coarse_size = 16
    )
    workspace = NavierStokesWorkspace(solver)
    final_time = 0.02
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt = 0.02 / nx, tolerance = 2.0e-10, max_cycles = 80
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    maximum_divergence = maximum(
        maximum(abs, interior(divergence, block))
            for block in eachindex(mesh.blocks)
    )
    return (
        _velocity_l2_error(velocity, exact, final_time, solver),
        maximum_divergence, result,
    )
end

@testset "Taylor--Green Navier--Stokes integration" begin
    coarse = _taylor_green_error(12)
    fine = _taylor_green_error(24)
    @test fine[1] < coarse[1]
    @test log2(coarse[1] / fine[1]) > 1.8
    @test coarse[2] <= 2.0e-8
    @test fine[2] <= 2.0e-8
    @test coarse[3].pressure.converged
    @test fine[3].pressure.converged

    mixed = _taylor_green_error(8; mixed = true)
    @test mixed[1] < 0.08
    @test mixed[2] <= 2.0e-8
end

@testset "single-phase material scaling" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    initial = (x, y, time, axis) ->
    axis == 1 ? sinpi(2x) : 0.25sinpi(2y)
    light = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.02,
        boundary = FreeSlip(), coarse_size = 8,
    )
    heavy = IncompressibleNavierStokesSolver(
        mesh; density = 7.0, dynamic_viscosity = 0.14,
        boundary = FreeSlip(), coarse_size = 8,
    )
    @test light.kinematic_viscosity == heavy.kinematic_viscosity == 0.02
    @test heavy.pressure_solver.operator.face_coefficient ≈
        light.pressure_solver.operator.face_coefficient ./ 7

    light_velocity = FaceVelocity(mesh, initial)
    heavy_velocity = FaceVelocity(mesh, initial)
    light_pressure = CellField(mesh)
    heavy_pressure = CellField(mesh)
    light_workspace = NavierStokesWorkspace(light)
    heavy_workspace = NavierStokesWorkspace(heavy)
    light_stats = project!(
        light_velocity, light_pressure, light.pressure_solver, 0.01,
        light_workspace.projection; tolerance = 1.0e-12, max_cycles = 80,
    )
    heavy_stats = project!(
        heavy_velocity, heavy_pressure, heavy.pressure_solver, 0.01,
        heavy_workspace.projection; tolerance = 1.0e-12, max_cycles = 80,
    )
    @test light_stats.converged
    @test heavy_stats.converged
    @test heavy_velocity.interior ≈ light_velocity.interior rtol = 64eps() atol =
        64eps()
    for block in eachindex(mesh.blocks)
        @test interior(heavy_pressure, block) ≈
            7 .* interior(light_pressure, block) rtol = 512eps() atol =
            512eps()
    end
end

@testset "Navier--Stokes timestep is convective only" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.25 : -0.125
    )
    pressure = CellField(mesh)
    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 100.0,
        boundary = FreeSlip(), coarse_size = 8
    )
    @test solver.amr_interface_scheme isa MonotonizedCentral
    @test_throws ArgumentError IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 1.0,
        scheme = LaxWendroff(), amr_interface_scheme = Upwind()
    )
    @test_throws ArgumentError IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = Inf
    )
    workspace = NavierStokesWorkspace(solver)

    expected = stable_timestep(velocity, workspace.advection; cfl = 0.35)
    @test stable_timestep(velocity, solver, workspace; cfl = 0.35) == expected

    fill!(velocity.interior, 0.0)
    result = simulate!(
        velocity, pressure, solver, 0.2, workspace;
        tolerance = 1.0e-10, max_cycles = 20
    )
    @test result.steps == 128
    @test result.time == 0.2
    @test result.pressure.converged
end

function _cavity_centerline(velocity, axis, coordinates)
    points = Tuple{Float64, Float64}[]
    for (index, face) in pairs(velocity.mesh.faces)
        face.axis == axis || continue
        abs(face.center[axis] - 0.5) <= 32eps() || continue
        push!(points, (face.center[3 - axis], velocity.interior[index]))
    end
    if axis == 1
        push!(points, (0.0, 0.0))
        push!(points, (1.0, 1.0))
    else
        push!(points, (0.0, 0.0))
        push!(points, (1.0, 0.0))
    end
    sort!(points; by = first)
    abscissae = first.(points)
    return map(coordinates) do coordinate
        left_index = clamp(
            searchsortedlast(abscissae, coordinate), 1, length(points) - 1
        )
        left = points[left_index]
        right = points[left_index + 1]
        fraction = (coordinate - left[1]) / (right[1] - left[1])
        return left[2] + fraction * (right[2] - left[2])
    end
end

@testset "Ghia Re=100 lid-driven cavity" begin
    # Tables I and II of Ghia, Ghia, and Shin, JCP 48 (1982), pp. 398--399,
    # DOI 10.1016/0021-9991(82)90058-4. End points are omitted because they
    # are prescribed wall values rather than independent validation samples.
    ghia_y = [
        0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
        0.7344, 0.6172, 0.5, 0.4531, 0.2813,
        0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
    ]
    ghia_u = [
        0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
        0.00332, -0.13641, -0.20581, -0.2109, -0.15662,
        -0.1015, -0.06434, -0.04775, -0.04192, -0.03717,
    ]
    ghia_x = [
        0.9688, 0.9609, 0.9531, 0.9453, 0.9063,
        0.8594, 0.8047, 0.5, 0.2344, 0.2266,
        0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
    ]
    ghia_v = [
        -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
        -0.22445, -0.24533, 0.05454, 0.17527, 0.17507,
        0.16077, 0.12317, 0.1089, 0.10091, 0.09233,
    ]

    mesh = BlockTreeMesh(block_size = (32, 32), periodic = (false, false))
    lid = NoSlip(
        (x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0
    )
    solver = IncompressibleNavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.01,
        boundary = lid, coarse_size = 32
    )
    workspace = NavierStokesWorkspace(solver)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    result = simulate!(
        velocity, pressure, solver, 15.0, workspace;
        dt = 0.005, tolerance = 1.0e-8, max_cycles = 100
    )
    @test result.pressure.converged
    @test all(iszero, velocity.boundary)

    numerical_u = _cavity_centerline(velocity, 1, ghia_y)
    numerical_v = _cavity_centerline(velocity, 2, ghia_x)
    u_error = numerical_u - ghia_u
    v_error = numerical_v - ghia_v
    @test sqrt(sum(abs2, u_error) / length(u_error)) < 0.004
    @test sqrt(sum(abs2, v_error) / length(v_error)) < 0.006
    @test maximum(abs, u_error) < 0.006
    @test maximum(abs, v_error) < 0.012

    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test maximum(abs, interior(divergence, 1)) <= 2.0e-8

    velocity_at_15 = copy(velocity.interior)
    settled = simulate!(
        velocity, pressure, solver, 16.0, workspace;
        time = 15.0, dt = 0.005, tolerance = 1.0e-8, max_cycles = 100
    )
    @test settled.pressure.converged
    @test maximum(abs, velocity.interior - velocity_at_15) < 1.0e-4
end

function _check_refreshed_staggered_solve(
        solver, diffusion, strength, tolerance,
    )
    reference_operator = StaggeredEllipticOperator(
        diffusion; reaction = solver.operator.reaction, diffusion = strength,
    )
    reference = StaggeredEllipticSolver(
        reference_operator;
        krylov = solver.krylov_factory,
        krylov_options = solver.krylov_options,
        preconditioner = solver.preconditioner_specification,
        preconditioner_options = solver.preconditioner_options,
        multigrid = TwoPhaseAMR._multigrid_options(solver),
    )
    @test solver.operator.diffusion == strength
    @test solver.operator.x.matrix == reference_operator.x.matrix
    @test solver.operator.y.matrix == reference_operator.y.matrix
    for (actual, expected) in ((solver.x, reference.x), (solver.y, reference.y))
        @test actual.action === actual.operator.matrix
        if actual.preconditioner isa TwoPhaseAMR.TreeStaggeredMultigridComponent
            hierarchy, fresh = actual.preconditioner, expected.preconditioner
            @test length(hierarchy.levels) == length(fresh.levels)
            for (level, fresh_level) in zip(hierarchy.levels, fresh.levels)
                @test level.matrix == fresh_level.matrix
                @test level.matrix_transpose == fresh_level.matrix_transpose
                @test level.inverse_diagonal == fresh_level.inverse_diagonal
                @test level.relaxation_weight == fresh_level.relaxation_weight
            end
            coarse = last(hierarchy.levels).matrix
            FT = eltype(hierarchy.coarse_factorization)
            rhs = [FT(sin(index)) for index in axes(coarse, 1)]
            factor_tolerance = eltype(coarse) === Float32 ? 1.0e-5 : 1.0e-10
            solution = hierarchy.coarse_factorization \ rhs
            @test coarse * solution ≈ rhs rtol =
                factor_tolerance atol = factor_tolerance
            @test solution ≈ fresh.coarse_factorization \ rhs rtol =
                factor_tolerance atol = factor_tolerance
        end
    end

    rhs = FaceVelocity(
        diffusion.mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y),
    )
    actual, expected = FaceVelocity(diffusion.mesh), FaceVelocity(diffusion.mesh)
    actual_stats = solve!(
        actual, solver, rhs; tolerance, max_cycles = 300,
    )
    expected_stats = solve!(
        expected, reference, rhs; tolerance, max_cycles = 300,
    )
    @test actual_stats.converged
    @test expected_stats.converged
    # Both independently solved systems obey the same strong-residual tolerance;
    # allow for the accumulation of that tolerance and floating-point roundoff.
    @test actual.interior ≈ expected.interior rtol = 50tolerance atol = 50tolerance
    return nothing
end

@testset "fixed-topology staggered diffusion refresh" begin
    mixed_boundary = VelocityBoundaryConditions(
        left = Inflow((x, y, time, axis) -> axis == 1 ? 0.1 : 0.0),
        right = Outflow(), bottom = FreeSlip(), top = NoSlip(),
    )
    for (T, boundary, periodic, adaptive) in (
            (Float64, FreeSlip(), (true, true), true),
            (Float32, NoSlip(), (false, false), false),
            (Float64, mixed_boundary, (false, false), false),
        )
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4),
            origin = (T(0), T(0)), extent = (T(1), T(1)), periodic = periodic,
            max_level = adaptive ? 2 : 0,
            refine = (key, origin, extent) -> adaptive &&
                (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)),
        )
        diffusion = StaggeredDiffusionOperator(
            mesh; viscosity = T(0.1), boundary, T,
        )
        solver = StaggeredEllipticSolver(
            StaggeredEllipticOperator(
                diffusion; reaction = T(1.3), diffusion = T(0.05),
            ); coarse_size = 16,
        )
        components = (solver.x, solver.y)
        workspaces = (solver.x.krylov, solver.y.krylov)
        vectors = map(components) do component
            (
                component.solution, component.right_hand_side,
                component.residual, component.product,
            )
        end
        hierarchies = map(component -> component.preconditioner, components)
        factors = map(hierarchy -> hierarchy.coarse_factorization, hierarchies)
        storage = map(hierarchies) do hierarchy
            map(hierarchy.levels) do level
                (
                    level, level.matrix, nonzeros(level.matrix),
                    level.matrix_transpose, nonzeros(level.matrix_transpose),
                    level.prolongation, level.inverse_diagonal,
                    level.solution, level.right_hand_side, level.residual,
                    level.temporary,
                )
            end
        end
        tolerance = T === Float32 ? T(1.0e-5) : T(1.0e-10)
        # Include a large change, a single representable increment, a genuinely
        # truncated shift, zero, and a return to nonzero.
        for strength in (T(0.4), nextfloat(T(0.4)), T(0.0007), zero(T), T(0.08))
            TwoPhaseAMR._update_staggered_elliptic!(solver, strength)
            @test (solver.x, solver.y) === components
            for axis in 1:2
                component = components[axis]
                hierarchy = hierarchies[axis]
                @test component.krylov === workspaces[axis]
                @test component.preconditioner === hierarchy
                @test (
                    component.solution, component.right_hand_side,
                    component.residual, component.product,
                ) === vectors[axis]
                @test hierarchy.coarse_factorization === factors[axis]
                for (level, retained) in zip(hierarchy.levels, storage[axis])
                    @test (
                        level, level.matrix, nonzeros(level.matrix),
                        level.matrix_transpose, nonzeros(level.matrix_transpose),
                        level.prolongation, level.inverse_diagonal,
                        level.solution, level.right_hand_side, level.residual,
                        level.temporary,
                    ) === retained
                end
            end
            _check_refreshed_staggered_solve(
                solver, diffusion, strength, tolerance,
            )
        end
        if T === Float32
            unpreconditioned = StaggeredEllipticSolver(
                StaggeredEllipticOperator(
                    diffusion; reaction = T(1.3), diffusion = T(0.05),
                ); preconditioner = nothing,
            )
            for strength in (T(0.4), zero(T), T(0.0007))
                TwoPhaseAMR._update_staggered_elliptic!(unpreconditioned, strength)
                _check_refreshed_staggered_solve(
                    unpreconditioned, diffusion, strength, tolerance,
                )
            end
        end
    end
end

@testset "zero-shift expansion and non-tree staggered refresh" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    zero_solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(diffusion; diffusion = 0.0); coarse_size = 8,
    )
    components = (zero_solver.x, zero_solver.y)
    workspaces = map(component -> component.krylov, components)
    factors = map(component -> component.preconditioner.coarse_factorization, components)
    @test all(workspace -> workspace isa TwoPhaseAMR.Krylov.BicgstabWorkspace, workspaces)
    TwoPhaseAMR._update_staggered_elliptic!(zero_solver, 0.02)
    for axis in 1:2
        @test components[axis].krylov === workspaces[axis]
        # Dense coarse storage covers both patterns; sparse symbolic analysis
        # must instead be rebuilt when zero diffusion omits off-diagonals.
        factor = components[axis].preconditioner.coarse_factorization
        if factors[axis] isa TwoPhaseAMR.DenseStaggeredCoarseLU
            @test factor === factors[axis]
        else
            @test factor !== factors[axis]
        end
    end
    _check_refreshed_staggered_solve(zero_solver, diffusion, 0.02, 1.0e-10)
    expanded = map(component -> component.operator.matrix, components)
    factors = map(component -> component.preconditioner.coarse_factorization, components)
    TwoPhaseAMR._update_staggered_elliptic!(zero_solver, 0.0)
    TwoPhaseAMR._update_staggered_elliptic!(zero_solver, 0.08)
    for axis in 1:2
        @test components[axis].operator.matrix === expanded[axis]
        @test components[axis].preconditioner.coarse_factorization === factors[axis]
    end
    _check_refreshed_staggered_solve(zero_solver, diffusion, 0.08, 1.0e-10)

    factory_calls = Ref(0)
    factory = matrix -> begin
        factory_calls[] += 1
        Diagonal(diag(matrix))
    end
    changing_factory = matrix -> isdiag(matrix) ? nothing : Diagonal(diag(matrix))
    prototype = StaggeredEllipticOperator(diffusion; diffusion = 0.03)
    prebuilt = (
        x = Diagonal(ones(length(prototype.x.volumes))),
        y = Diagonal(ones(length(prototype.y.volumes))),
    )
    for specification in (nothing, :ilu, factory, changing_factory, prebuilt)
        solver = StaggeredEllipticSolver(
            StaggeredEllipticOperator(diffusion; diffusion = 0.03);
            krylov = TwoPhaseAMR.Krylov.BicgstabWorkspace,
            preconditioner = specification,
        )
        components = (solver.x, solver.y)
        workspaces = map(component -> component.krylov, components)
        matrices = map(component -> component.operator.matrix, components)
        for strength in (0.12, 0.0, 0.07)
            calls_before = factory_calls[]
            TwoPhaseAMR._update_staggered_elliptic!(solver, strength)
            for axis in 1:2
                @test components[axis].krylov === workspaces[axis]
                @test components[axis].operator.matrix === matrices[axis]
            end
            specification === factory && @test factory_calls[] == calls_before + 2
            specification === prebuilt && @test solver.x.preconditioner === prebuilt.x
            if specification === changing_factory
                @test isnothing(solver.x.preconditioner) == iszero(strength)
                @test isnothing(solver.y.preconditioner) == iszero(strength)
            end
            _check_refreshed_staggered_solve(solver, diffusion, strength, 1.0e-10)
        end
    end
    for specification in (nothing, :ilu, changing_factory, prebuilt)
        solver = StaggeredEllipticSolver(
            StaggeredEllipticOperator(diffusion; diffusion = 0.0);
            krylov = TwoPhaseAMR.Krylov.BicgstabWorkspace,
            preconditioner = specification,
        )
        workspaces = (solver.x.krylov, solver.y.krylov)
        TwoPhaseAMR._update_staggered_elliptic!(solver, 0.07)
        @test (solver.x.krylov, solver.y.krylov) === workspaces
        if specification === changing_factory
            @test !isnothing(solver.x.preconditioner)
            @test !isnothing(solver.y.preconditioner)
        end
        _check_refreshed_staggered_solve(solver, diffusion, 0.07, 1.0e-10)
    end
    @test_throws ArgumentError TwoPhaseAMR._update_staggered_elliptic!(
        zero_solver, -0.1,
    )
end

@testset "staggered solvers do not mutate a shared operator" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    diffusion = StaggeredDiffusionOperator(mesh; viscosity = 0.1, boundary = FreeSlip())
    operator = StaggeredEllipticOperator(diffusion; diffusion = 0.02)
    initial_x = copy(operator.x.matrix)
    initial_y = copy(operator.y.matrix)
    first_solver = StaggeredEllipticSolver(operator; coarse_size = 8)
    second_solver = StaggeredEllipticSolver(operator; coarse_size = 8)
    @test first_solver.operator.x.matrix !== operator.x.matrix
    @test first_solver.operator.x.matrix !== second_solver.operator.x.matrix
    TwoPhaseAMR._update_staggered_elliptic!(first_solver, 0.08)
    @test operator.diffusion == 0.02
    @test operator.x.matrix == initial_x
    @test operator.y.matrix == initial_y
    @test second_solver.operator.x.matrix == initial_x
    @test second_solver.operator.y.matrix == initial_y
    TwoPhaseAMR._update_staggered_elliptic!(second_solver, 0.02)
    _check_refreshed_staggered_solve(second_solver, diffusion, 0.02, 1.0e-10)
end

@testset "large staggered coarse systems reuse sparse symbolic analysis" begin
    mesh = BlockTreeMesh(block_size = (18, 18))
    diffusion = StaggeredDiffusionOperator(mesh; viscosity = 0.1, boundary = FreeSlip())
    solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(diffusion; diffusion = 0.02); coarse_size = 16,
    )
    components = (solver.x, solver.y)
    factors = map(component -> component.preconditioner.coarse_factorization, components)
    symbols = map(factor -> factor.symbolic, factors)
    @test all(factor -> !(factor isa TwoPhaseAMR.DenseStaggeredCoarseLU), factors)
    TwoPhaseAMR._update_staggered_elliptic!(solver, 0.04)
    for axis in 1:2
        preconditioner = components[axis].preconditioner
        @test preconditioner.coarse_factorization === factors[axis]
        @test factors[axis].symbolic === symbols[axis]
        coarse = last(preconditioner.levels).matrix
        rhs = [sin(0.3index) for index in axes(coarse, 1)]
        @test coarse * (factors[axis] \ rhs) ≈ rhs rtol = 1.0e-10 atol = 1.0e-10
    end
    zero_solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(diffusion; diffusion = 0.0); coarse_size = 16,
    )
    zero_components = (zero_solver.x, zero_solver.y)
    zero_factors = map(component -> component.preconditioner.coarse_factorization, zero_components)
    TwoPhaseAMR._update_staggered_elliptic!(zero_solver, 0.04)
    for axis in 1:2
        preconditioner = zero_components[axis].preconditioner
        @test preconditioner.coarse_factorization !== zero_factors[axis]
        coarse = last(preconditioner.levels).matrix
        rhs = [sin(0.3index) for index in axes(coarse, 1)]
        @test coarse * (preconditioner.coarse_factorization \ rhs) ≈
            rhs rtol = 1.0e-10 atol = 1.0e-10
    end
end

@testset "staggered refresh after AMR hierarchy reuse" begin
    mesh = BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4), max_level = 1)
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(diffusion; diffusion = 0.03); coarse_size = 8,
    )
    old = TreeStaggeredGeometricMultigridPreconditioner(
        solver.operator, solver.x.preconditioner, solver.y.preconditioner,
    )
    marker = CellField(mesh)
    adaptation = AMRWorkspace(mesh, marker)
    labels = [
        block.key == BlockKey(0, 0, 0) ? :refine : :keep
            for block in mesh.blocks
    ]
    @test adapt!(adaptation, labels; step = 1, force = true).changed
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(diffusion; diffusion = 0.03);
        coarse_size = 8, reuse_preconditioner = old,
    )
    @test solver.x.preconditioner.reused_levels > 0
    @test solver.y.preconditioner.reused_levels > 0
    workspaces = (solver.x.krylov, solver.y.krylov)
    for strength in (0.19, 0.0, 0.004)
        TwoPhaseAMR._update_staggered_elliptic!(solver, strength)
        @test (solver.x.krylov, solver.y.krylov) === workspaces
        _check_refreshed_staggered_solve(solver, diffusion, strength, 1.0e-10)
    end
end
