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
    solver.pressure_solver = EllipticSolver(
        PoissonOperator(model.interface; density = model.density, formulation = model.formulation);
        krylov = solver.pressure_krylov, krylov_options = solver.pressure_krylov_options,
        preconditioner = solver.pressure_preconditioner,
        preconditioner_options = solver.pressure_preconditioner_options,
        coarse_size = solver.coarse_size, pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps, max_levels = solver.max_levels
    )
    return workspace.projection = ProjectionWorkspace(solver.pressure_solver)
end

function _transport_substep!(
        velocity, solver, workspace, dt, model::TwoPhaseModel
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
        velocity, solver, workspace, dt, time, model::TwoPhaseModel
    )
    if all(iszero, model.viscosity)
        _copy_velocity!(velocity, workspace.convected)
    else
        diffusion = _update_dynamic_viscosity!(
            workspace.two_phase.diffusion,
            model.interface, model.viscosity
        )
        implicit_midpoint_diffusion!(
            velocity, workspace.convected, diffusion, dt;
            density = mixture_density(model.interface, model.density)
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
        velocity.interior .+= dt .* phase.acceleration.interior
    end
    if any(!iszero, model.gravity)
        gravity!(
            phase.acceleration, model.interface, op, workspace.projection.elliptic;
            density = model.density, gravity = model.gravity
        )
        velocity.interior .+= dt .* phase.acceleration.interior
    end
    return velocity
end
