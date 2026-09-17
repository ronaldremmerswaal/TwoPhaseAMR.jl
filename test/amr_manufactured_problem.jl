struct LocalizedManufacturedVortex{T}
    center::NTuple{2, T}
    concentration::T
    amplitude::T
    decay_rate::T
    viscosity::T
end

function _manufactured_velocity_l2_error(velocity, exact, time, solver)
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

function _periodic_bump_derivatives(coordinate, center, concentration)
    angle = 2pi * (coordinate - center)
    value = exp(concentration * (cos(angle) - 1))
    first_log = -2pi * concentration * sin(angle)
    second_log = -(2pi)^2 * concentration * cos(angle)
    third_log = (2pi)^3 * concentration * sin(angle)
    first = value * first_log
    second = value * (first_log^2 + second_log)
    third = value * (
        first_log^3 + 3first_log * second_log + third_log
    )
    return value, first, second, third
end

function _localized_velocity(
        problem::LocalizedManufacturedVortex,
        x, y, time, axis
    )
    xprofile = _periodic_bump_derivatives(
        x, problem.center[1], problem.concentration
    )
    yprofile = _periodic_bump_derivatives(
        y, problem.center[2], problem.concentration
    )
    amplitude = problem.amplitude * exp(-problem.decay_rate * time)
    return axis == 1 ? amplitude * xprofile[1] * yprofile[2] :
        -amplitude * xprofile[2] * yprofile[1]
end

function _localized_force(
        problem::LocalizedManufacturedVortex,
        x, y, time, axis
    )
    X, X1, X2, X3 = _periodic_bump_derivatives(
        x, problem.center[1], problem.concentration
    )
    Y, Y1, Y2, Y3 = _periodic_bump_derivatives(
        y, problem.center[2], problem.concentration
    )
    amplitude = problem.amplitude * exp(-problem.decay_rate * time)
    if axis == 1
        velocity = amplitude * X * Y1
        convection = amplitude^2 * X * X1 * (Y1^2 - Y * Y2)
        laplacian = amplitude * (X2 * Y1 + X * Y3)
    else
        velocity = -amplitude * X1 * Y
        convection = amplitude^2 * Y * Y1 * (X1^2 - X * X2)
        laplacian = -amplitude * (X3 * Y + X1 * Y2)
    end
    return -problem.decay_rate * velocity + convection -
        problem.viscosity * laplacian
end

function _localized_marker(problem::LocalizedManufacturedVortex, x, y)
    X = first(
        _periodic_bump_derivatives(
            x, problem.center[1], problem.concentration
        )
    )
    Y = first(
        _periodic_bump_derivatives(
            y, problem.center[2], problem.concentration
        )
    )
    return X * Y
end

function _localized_vortex_case(
        max_level;
        E_refine = 0.025,
        block_size = 8,
        dt = 5.0e-4,
        final_time = 6.0e-3
    )
    problem = LocalizedManufacturedVortex(
        (0.37, 0.53), 8.0, 0.05, 1.0, 2.0e-3
    )
    mesh = BlockTreeMesh(
        block_size = (block_size, block_size), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 1, coarsen_delay = typemax(Int)
        )
    )
    marker = CellField(mesh)
    initialize!(marker, (x, y) -> _localized_marker(problem, x, y))
    indicator = InterpolationErrorIndicator(
        marker; E_refine, E_coarsen = 0.0, relative_floor = 0.02
    )
    adaptation_workspace = AMRWorkspace(mesh, marker)
    for step in 1:max_level
        statistics = adapt!(
            adaptation_workspace, indicator; step, force = true
        )
        statistics.changed || break
    end
    exact = (x, y, time, axis) ->
    _localized_velocity(problem, x, y, time, axis)
    force = (x, y, time, axis) ->
    _localized_force(problem, x, y, time, axis)
    velocity = FaceVelocity(mesh, exact)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = problem.viscosity,
        boundary = FreeSlip(),
        coarse_size = 16, refinement_indicators = (indicator,)
    )
    workspace = NavierStokesWorkspace(solver)
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt, force, rtol = 2.0e-10, max_cycles = 100
    )
    error = _manufactured_velocity_l2_error(
        velocity, exact, final_time, solver
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    maximum_divergence = maximum(
        maximum(abs, interior(divergence, block))
            for block in eachindex(mesh.blocks)
    )
    return (
        error, maximum_divergence, ncells(mesh), nblocks(mesh),
        maximum(block.key.level for block in mesh.blocks), result,
        isbalanced(mesh),
    )
end

function _uniform_localized_vortex_case(
        block_size;
        dt = 5.0e-4,
        final_time = 6.0e-3
    )
    problem = LocalizedManufacturedVortex(
        (0.37, 0.53), 8.0, 0.05, 1.0, 2.0e-3
    )
    mesh = BlockTreeMesh(block_size = (block_size, block_size))
    exact = (x, y, time, axis) ->
    _localized_velocity(problem, x, y, time, axis)
    force = (x, y, time, axis) ->
    _localized_force(problem, x, y, time, axis)
    velocity = FaceVelocity(mesh, exact)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = problem.viscosity,
        boundary = FreeSlip(),
        coarse_size = 16
    )
    workspace = NavierStokesWorkspace(solver)
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt, force, rtol = 2.0e-10, max_cycles = 100
    )
    return (
        _manufactured_velocity_l2_error(
            velocity, exact, final_time, solver
        ),
        ncells(mesh), result,
    )
end
