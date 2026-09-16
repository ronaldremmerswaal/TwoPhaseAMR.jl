function _timestep_bound(
        velocity, solver, workspace, model::TwoPhaseModel; cfl = 0.45
    )
    bound = stable_timestep(velocity, workspace.two_phase.advection; cfl)
    h = minimum(minimum(block.spacing) for block in solver.mesh.blocks)
    rho = model.density
    if model.sigma > 0
        bound = min(bound, sqrt(sum(rho) * h^3 / (4pi * model.sigma)))
    end
    g = hypot(model.gravity...)
    if g > 0 && rho[1] != rho[2]
        bound = min(bound, sqrt(sum(rho) / abs(rho[1] - rho[2]) * h / g))
    end
    return bound
end
function _refresh_pressure_substep!(solver, workspace, model::TwoPhaseModel)
    # The topology and compatible gradient metric are fixed between AMR
    # updates, but both the fine operator and its coefficient-dependent
    # preconditioner must follow the advected interface. A frozen hierarchy
    # rapidly loses effectiveness at large density ratio.
    pressure_solver = solver.pressure_solver
    pressure_solver.operator = isnothing(solver.geometry) ? PoissonOperator(
        model.interface; density = model.density,
        formulation = model.formulation,
        template = pressure_solver.operator,
    ) : PoissonOperator(
        model.interface, solver.geometry; density = model.density,
        formulation = model.formulation,
        template = pressure_solver.operator,
    )
    specification = solver.pressure_preconditioner
    variable_density = model.density[1] != model.density[2]
    refreshable = variable_density && (
        specification isa Symbol ||
            applicable(specification, pressure_solver.operator.matrix)
    )
    if refreshable
        if specification in (:geometric, :tree_geometric) &&
                pressure_solver.preconditioner isa
                TreeGeometricMultigridPreconditioner
            _refresh_geometric_multigrid!(
                pressure_solver.preconditioner, pressure_solver.operator,
            )
        else
            pressure_solver.preconditioner =
                _construct_pressure_preconditioner(
                specification, pressure_solver.operator;
                coarse_size = solver.coarse_size,
                pre_sweeps = solver.pre_sweeps,
                post_sweeps = solver.post_sweeps,
                max_levels = solver.max_levels,
                options = solver.pressure_preconditioner_options
            )
        end
        pressure_solver.stats.preconditioner =
            _preconditioner_name(pressure_solver.preconditioner)
    end
    return workspace.projection
end

function _transport_substep!(
        velocity, solver, workspace, dt, time, model::TwoPhaseModel
    )
    phase = workspace.two_phase
    bound = _timestep_bound(
        velocity, solver, workspace, model; cfl = 0.5
    )
    dt <= bound * (1 + 32eps(eltype(velocity))) || throw(ArgumentError("time step exceeds two-phase transport or force stability bound"))
    advect!(
        model.interface, workspace.old_velocity, dt, phase.advection;
        scalars = model.scalars, momentum = workspace.convected, density = model.density,
        momentum_workspace = phase.momentum,
        sweep_order = phase.forward ? (1, 2) : (2, 1), scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme
    )
    phase.forward = !phase.forward
    return workspace.convected
end

function _diffusion_substep!(
        velocity, solver, workspace, dt, time, model::TwoPhaseModel;
        rtol = 1.0e-10, atol = 0, max_cycles = 500
    )
    if all(iszero, model.viscosity)
        _copy_velocity!(velocity, workspace.convected)
    else
        diffusion = _update_dynamic_viscosity!(
            workspace.two_phase.diffusion,
            model.interface, model.viscosity
        )
        density = _mixture_density!(
            workspace.two_phase.density, model.interface, model.density,
        )
        implicit_midpoint_diffusion!(
            velocity, workspace.convected, diffusion, dt,
            workspace.two_phase.diffusion_solver;
            old_velocity = workspace.old_velocity,
            density,
            rtol, atol, max_cycles
        )
    end
    return velocity
end

function _phase_force_substep!(
        velocity, solver, workspace, dt, model::TwoPhaseModel
    )
    phase = workspace.two_phase
    op = solver.pressure_solver.operator
    if model.sigma > 0
        surface_tension!(
            phase.acceleration, model.interface, op, workspace.projection.elliptic;
            sigma = model.sigma, formulation = model.formulation
        )
        @_parallel for index in 1:length(velocity.interior)
            @inbounds velocity.interior[index] +=
                dt * phase.acceleration.interior[index]
        end
    end
    if any(!iszero, model.gravity)
        gravity!(
            phase.acceleration, model.interface, op, workspace.projection.elliptic;
            density = model.density, gravity = model.gravity
        )
        @_parallel for index in 1:length(velocity.interior)
            @inbounds velocity.interior[index] +=
                dt * phase.acceleration.interior[index]
        end
    end
    return velocity
end
