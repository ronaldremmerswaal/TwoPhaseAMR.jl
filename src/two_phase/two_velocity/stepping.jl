# Intrinsic phase-velocity stage implementations and public stepping path.
function _copy_velocity!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
    )
    for phase in 1:2
        _copy_velocity!(destination[phase], source[phase])
    end
    return destination
end

function _enforce_phase_velocity_boundary!(
        velocity::TwoPhaseVelocity, solver, time,
    )
    for phase in 1:2
        enforce_velocity_boundary!(
            velocity[phase], solver.boundary;
            time, geometry = solver.geometry,
        )
    end
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    return velocity
end


@inline _enforce_current_velocity_boundary!(
    velocity::TwoPhaseVelocity, solver, time,
) = _enforce_phase_velocity_boundary!(velocity, solver, time)

function _timestep_bound(
        velocity::TwoPhaseVelocity, solver, workspace,
        model::TwoPhaseModel; cfl = 0.45,
    )
    state = _two_velocity_flow_workspace!(workspace.two_phase, model)
    bound = stable_timestep(velocity, state.advection; cfl)
    h = minimum(minimum(block.spacing) for block in solver.mesh.blocks)
    rho = model.density
    if model.sigma > 0
        bound = min(
            bound, sqrt(sum(rho) * h^3 / (4pi * model.sigma)),
        )
    end
    g = hypot(model.gravity...)
    if g > 0 && rho[1] != rho[2]
        bound = min(
            bound,
            sqrt(sum(rho) / abs(rho[1] - rho[2]) * h / (2g)),
        )
    end
    return bound
end

function _transport_substep!(
        velocity::TwoPhaseVelocity, solver, workspace, dt, time,
        model::TwoPhaseModel,
    )
    phase = workspace.two_phase
    state = _two_velocity_flow_workspace!(phase, model)
    bound = _timestep_bound(
        velocity, solver, workspace, model; cfl = 0.5,
    )
    dt <= bound * (1 + 32eps(eltype(velocity[1]))) || throw(
        ArgumentError(
            "time step exceeds two-phase transport or force stability bound",
        ),
    )
    advect!(
        model.interface, state.old_velocity, dt, state.advection;
        scalars = model.scalars,
        momentum = state.convected, density = model.density,
        momentum_workspace = state.momentum,
        sweep_order = phase.forward ? (1, 2) : (2, 1),
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme,
    )
    phase.forward = !phase.forward
    return state.convected
end

"""Form the thesis one-velocity state from intrinsic phase velocities.

`density` is refreshed at the same time so the midpoint solve uses exactly the
same staggered phase masses as this average. No phase-weighted quantity is
stored outside this local construction.
"""
function _mass_weighted_one_velocity!(
        destination::FaceVelocity{T}, density::AbstractVector{T},
        velocity::TwoPhaseVelocity, model::TwoPhaseModel{T},
    ) where {T}
    return _mass_weighted_one_velocity!(
        destination, density, velocity, model, model.interface.geometry,
    )
end

function _mass_weighted_one_velocity!(
        destination::FaceVelocity{T}, density::AbstractVector{T},
        velocity::TwoPhaseVelocity, model::TwoPhaseModel{T}, geometry,
    ) where {T}
    interface = model.interface
    mesh = interface.mesh
    destination.mesh === mesh || throw(
        ArgumentError("mixture velocity and interface must share a mesh"),
    )
    _validate_two_phase_velocity(velocity, mesh, T)
    length(density) == length(mesh.faces) || throw(
        DimensionMismatch(
            "one mixture-density value is required per interior face",
        )
    )
    rho_liquid, rho_gas = model.density
    @_parallel for index in 1:length(mesh.faces)
        face = mesh.faces[index]
        liquid_fraction = _staggered_liquid_fraction(
            interface, geometry, face,
        )
        liquid_mass = rho_liquid * liquid_fraction
        gas_mass = rho_gas * (one(T) - liquid_fraction)
        total_mass = liquid_mass + gas_mass
        @inbounds begin
            density[index] = total_mass
            destination.interior[index] = (
                liquid_mass * velocity[1].interior[index] +
                    gas_mass * velocity[2].interior[index]
            ) / total_mass
        end
    end
    fill!(destination.boundary, zero(T))
    return destination
end

"""Apply one common viscous velocity increment to both intrinsic phases."""
function _apply_two_velocity_diffusion_update!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
        diffused_mixture::FaceVelocity, mixture::FaceVelocity,
    )
    mesh = mixture.mesh
    diffused_mixture.mesh === mesh || throw(
        ArgumentError("diffusion mixture fields must share a mesh"),
    )
    _validate_two_phase_velocity(destination, mesh, eltype(mixture))
    _validate_two_phase_velocity(source, mesh, eltype(mixture))
    @_parallel for index in 1:length(mesh.faces)
        @inbounds begin
            increment = diffused_mixture.interior[index] -
                mixture.interior[index]
            destination[1].interior[index] =
                source[1].interior[index] + increment
            destination[2].interior[index] =
                source[2].interior[index] + increment
        end
    end
    fill!(destination[1].boundary, zero(eltype(mixture)))
    fill!(destination[2].boundary, zero(eltype(mixture)))
    return destination
end

function _diffusion_substep!(
        velocity::TwoPhaseVelocity, solver, workspace, dt, time,
        model::TwoPhaseModel;
        tolerance = sqrt(eps(eltype(velocity[1]))), max_cycles = 500,
    )
    state = _two_velocity_flow_workspace!(workspace.two_phase, model)
    if all(iszero, model.dynamic_viscosity)
        _copy_velocity!(velocity, state.convected)
        return velocity
    end

    diffusion = _update_dynamic_viscosity!(
        workspace.two_phase.diffusion,
        model.interface, model.dynamic_viscosity,
    )
    density = workspace.two_phase.density
    _mass_weighted_one_velocity!(
        state.mixture_velocity, density, state.convected, model,
    )
    implicit_midpoint_diffusion!(
        state.diffused_mixture, state.mixture_velocity, diffusion, dt,
        workspace.two_phase.diffusion_solver;
        density, tolerance, max_cycles,
    )
    _apply_two_velocity_diffusion_update!(
        velocity, state.convected,
        state.diffused_mixture, state.mixture_velocity,
    )
    return velocity
end

function _add_common_phase_acceleration!(
        velocity::TwoPhaseVelocity, acceleration::FaceVelocity, dt,
    )
    for phase in 1:2
        @_parallel for index in 1:length(velocity[phase].interior)
            @inbounds velocity[phase].interior[index] +=
                dt * acceleration.interior[index]
        end
    end
    return velocity
end

function _phase_force_substep!(
        velocity::TwoPhaseVelocity, solver, workspace, dt,
        model::TwoPhaseModel,
    )
    phase = workspace.two_phase
    operator = solver.pressure_solver.operator
    # These accelerations are added to the predictor.  They therefore enter
    # both right-hand sides of the subsequent mimetic projection rather than
    # being applied after incompressibility has been enforced.
    if model.sigma > 0
        surface_tension!(
            phase.acceleration, model.interface, operator,
            workspace.projection.elliptic;
            sigma = model.sigma, formulation = model.formulation,
        )
        _add_common_phase_acceleration!(
            velocity, phase.acceleration, dt,
        )
    end
    if any(!iszero, model.gravity)
        gravity!(
            phase.acceleration, model.interface, operator,
            workspace.projection.elliptic;
            density = model.density, gravity = model.gravity,
        )
        _add_common_phase_acceleration!(
            velocity, phase.acceleration, dt,
        )
    end
    return velocity
end

"""Advance one two-velocity Navier--Stokes step.

The geometric interface and both intrinsic phase velocities use the same
transport stage as the passive two-velocity API.  Body force, surface tension,
and gravity modify the predictor, so their contribution is present in the two
right-hand sides of the subsequent three-block mimetic projection. Viscosity
acts implicitly on the phase-mass weighted one-velocity state; its common
increment is then applied to both phases, conserving combined momentum and
preserving the intrinsic velocity jump.
"""
function step!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
    ) where {T}
    model = solver.two_phase
    isnothing(model) && throw(
        ArgumentError(
            "a two-velocity step requires a solver configured with an interface",
        )
    )
    phase = workspace.two_phase
    isnothing(phase) && throw(
        ArgumentError(
            "a two-velocity step requires a two-phase workspace",
        )
    )
    _validate_two_phase_velocity(velocity, solver.mesh, T)
    pressure.mesh === solver.mesh || throw(
        ArgumentError("Navier--Stokes arguments must share a mesh"),
    )
    time_step = _positive_finite_step(T, dt)
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)

    state = _two_velocity_flow_workspace!(phase, model)
    _copy_velocity!(state.old_velocity, velocity)
    _enforce_phase_velocity_boundary!(
        state.old_velocity, solver, current_time,
    )
    _transport_substep!(
        velocity, solver, workspace, time_step, current_time, model,
    )
    midpoint = current_time + time_step / 2
    for phase_index in 1:2
        for (index, face) in pairs(solver.mesh.faces)
            @inbounds state.convected[phase_index].interior[index] +=
                time_step * _body_force(force, face, midpoint)
        end
    end
    _enforce_phase_velocity_boundary!(
        state.convected, solver, midpoint,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance, max_cycles = cycle_limit,
    )
    _enforce_phase_velocity_boundary!(
        velocity, solver, current_time + time_step,
    )
    if model.sigma > zero(T) || any(!iszero, model.gravity)
        # The established balanced-force kernels use the one-velocity
        # pressure Riesz map. Refresh it only when a force actually needs it;
        # the mimetic projection assembles its own three-block pressure block.
        _refresh_pressure_substep!(solver, workspace, model)
        _phase_force_substep!(
            velocity, solver, workspace, time_step, model,
        )
    end
    return project!(
        velocity, pressure, state.projection_solver, time_step,
        state.projection_workspace;
        tolerance = dimensionless_tolerance,
        max_iterations = cycle_limit,
    )
end

function stable_timestep(
        velocity::TwoPhaseVelocity,
        solver::IncompressibleNavierStokesSolver,
        workspace::NavierStokesWorkspace; cfl = 0.45,
    )
    isnothing(solver.two_phase) && throw(
        ArgumentError(
            "a two-velocity CFL bound requires a two-phase solver",
        )
    )
    return _timestep_bound(
        velocity, solver, workspace, solver.two_phase; cfl,
    )
end
