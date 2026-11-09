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
    if solver.boundary isa Union{NoSlip, FreeSlip}
        enforce_no_permeation!(velocity)
    else
        for phase in 1:2
            enforce_velocity_boundary!(
                velocity[phase], solver.boundary;
                time, geometry = solver.geometry, allow_absent = true,
            )
        end
    end
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    _mark_inactive_phase_velocities!(velocity, solver.two_phase.interface)
    return velocity
end


@inline _enforce_current_velocity_boundary!(
    velocity::TwoPhaseVelocity, solver, time,
) = _enforce_phase_velocity_boundary!(velocity, solver, time)

function _timestep_bound(
        velocity::TwoPhaseVelocity, solver, workspace,
        model::TwoPhaseModel{T, M, F, I}; cfl = 0.45,
    ) where {T, M, F, I}
    # Read the shared phase/WY workspace directly. The lazily constructed
    # projection state uses this same object, so frozen apertures prepared
    # here are available to the immediately following transport sweep.
    advection = _timestep_wy_workspace(workspace, model)
    return _two_velocity_timestep_bound(
        velocity, solver, advection, model, cfl,
    )
end

function _two_velocity_timestep_bound(
        velocity::TwoPhaseVelocity, solver,
        advection::WYAdvectionWorkspace, model::TwoPhaseModel, cfl,
    )
    bound = stable_timestep(velocity, advection; cfl)
    h = _minimum_mesh_spacing(solver.mesh)
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
        model::TwoPhaseModel; geometry_prepared = false,
    )
    phase = workspace.two_phase
    state = _two_velocity_flow_workspace!(phase, model)
    if !geometry_prepared
        # Direct `step!` and fixed-dt `simulate!` calls have no preceding CFL
        # preparation. This bound reconstructs the initial PLIC and freezes
        # the phase apertures exactly once before the first directional sweep.
        bound = _timestep_bound(
            velocity, solver, workspace, model; cfl = 0.5,
        )
        dt <= bound * (1 + 32eps(eltype(velocity[1]))) || throw(
            ArgumentError(
                "time step $dt exceeds $(_two_phase_limit_name(bound, solver, model)) stability bound $bound",
            ),
        )
    end
    _advect_transport_substep!(
        state.advection.interface_transport, model.interface,
        state.old_velocity, dt, state.advection;
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
            destination.interior[index] =
                !isnothing(geometry) && iszero(geometry.face_dual_volumes[index]) ?
                zero(T) : (
                    _phase_product(liquid_mass, velocity[1].interior[index]) +
                    _phase_product(gas_mass, velocity[2].interior[index])
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
                isnan(source[1].interior[index]) ? eltype(mixture)(NaN) :
                source[1].interior[index] + increment
            destination[2].interior[index] =
                isnan(source[2].interior[index]) ? eltype(mixture)(NaN) :
                source[2].interior[index] + increment
        end
    end
    for phase in 1:2, index in eachindex(destination[phase].boundary)
        destination[phase].boundary[index] = isnan(source[phase].boundary[index]) ?
            eltype(mixture)(NaN) : zero(eltype(mixture))
    end
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
    _mark_inactive_phase_velocities!(velocity, model.interface)
    return velocity
end

function _add_common_phase_acceleration!(
        velocity::TwoPhaseVelocity, acceleration::FaceVelocity, dt,
    )
    for phase in 1:2
        @_parallel for index in 1:length(velocity[phase].interior)
            isnan(velocity[phase].interior[index]) && continue
            @inbounds velocity[phase].interior[index] +=
                dt * acceleration.interior[index]
        end
    end
    return velocity
end

"""Save the old aperture-weighted mixture cell flux before interface transport."""
function _capture_two_velocity_inherited_flux!(solver, workspace, state)
    work = state.projection_workspace
    cache = work isa CMDGFMProjectionWorkspace ?
        work.pressure_cache : work.cache.pressure
    data = _projection_face_data(
        state.projection_solver, work; storage = cache.face_data,
    )
    cache.face_data = data
    projection = workspace.projection
    flux = projection.elliptic.input
    _cmdgfm_cell_flux!(
        flux, data, solver.two_phase.interface, state.old_velocity,
    )
    # The force stage reuses elliptic.input. Keep the old-time integrated flux
    # in cell scratch until that stage and the pressure predictor are done.
    return _copy_to_field!(
        projection.inherited_divergence, flux,
        solver.pressure_solver.operator.references,
    )
end

function _phase_force_pressure_operator!(projection, work::CMDGFMProjectionWorkspace)
    cache = work.pressure_cache
    data = _projection_face_data(projection, work; storage = cache.face_data)
    cache.face_data = data
    return _cmdgfm_pressure_cache!(cache, projection, data).operator
end

function _phase_force_pressure_operator!(projection, work::MimeticProjectionWorkspace)
    cache = work.cache
    data = _projection_face_data(projection, work; storage = cache.pressure.face_data)
    cache.pressure.face_data = data
    return _mimetic_pressure_cache!(cache, projection, data).pressure.operator
end

function _two_velocity_pressure_operator!(solver, workspace, state, model)
    if state.projection_solver isa CMDGFMProjectionSolver &&
            isnothing(solver.geometry)
        # This operator is passed directly into the uncut CMDGFM projection.
        # Its density must follow the just-advected, not the frozen, interface.
        reconstruct!(model.interface)
        _refresh_pressure_substep!(solver, workspace, model)
        return solver.pressure_solver.operator
    end
    # Mimetic and cut-cell CMDGFM use their own pressure operator. In
    # particular, the two-velocity cut-cell coefficient is A/rho, not A²/rho.
    return _phase_force_pressure_operator!(
        state.projection_solver, state.projection_workspace,
    )
end

function _phase_force_substep!(
        velocity::TwoPhaseVelocity, workspace, dt,
        model::TwoPhaseModel, operator,
    )
    phase = workspace.two_phase
    # Forces enter the predictor before the pressure projection.
    if model.sigma > 0
        surface_tension!(
            phase.acceleration, model.interface, operator,
            workspace.projection.elliptic;
            sigma = model.sigma, formulation = model.formulation,
            unresolved_curvature_faces = workspace.unresolved_curvature_faces,
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
transport stage as the passive two-velocity API. Body force, surface tension,
and gravity modify the predictor before subtracting the old physical pressure
gradient with the selected projection's current weighted map. For a shortened
step, the inherited old-aperture mixture flux enters the same projection with
the untruncated reference-step scaling. The projection corrects velocity with
the pressure increment and adds it to the physical pressure. Viscosity acts
implicitly on the phase-mass weighted one-velocity state; its common increment
is applied to both phases.
"""
function step!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
        _transport_geometry_prepared = false,
        _reference_dt = dt,
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
    reference_step = _positive_finite_step(T, _reference_dt, "reference dt")
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    _invalidate_simulation_step!(workspace)
    workspace.unresolved_curvature_faces[] = 0

    state = _two_velocity_flow_workspace!(phase, model)
    _copy_velocity!(state.old_velocity, velocity)
    _enforce_phase_velocity_boundary!(
        state.old_velocity, solver, current_time,
    )
    inherited_flux = time_step == reference_step ? nothing :
        _capture_two_velocity_inherited_flux!(solver, workspace, state)
    inherited_velocity = if !isnothing(inherited_flux) &&
            state.projection_solver isa CMDGFMProjectionSolver
        # These shared-velocity scratch fields are unused by the two-velocity
        # stages. Snapshot the same old-geometry extension as momentum
        # transport, without changing the frozen carrier in state.old_velocity.
        extended = (workspace.old_velocity, workspace.convected)
        _prepare_momentum_advection!(state.momentum, state.old_velocity)
        _copy_momentum_velocity!(extended, state.momentum)
    else
        nothing
    end
    # The CFL check must see the same boundary-enforced velocity as advection.
    _transport_substep!(
        state.old_velocity, solver, workspace, time_step, current_time, model;
        geometry_prepared = _transport_geometry_prepared,
    )
    midpoint = current_time + time_step / 2
    convected = state.convected
    _body_force_substep!(convected, time_step, midpoint, force)
    _enforce_phase_velocity_boundary!(
        convected, solver, midpoint,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance, max_cycles = cycle_limit,
    )
    _enforce_phase_velocity_boundary!(
        velocity, solver, current_time + time_step,
    )
    pressure_operator = _two_velocity_pressure_operator!(
        solver, workspace, state, model,
    )
    if model.sigma > zero(T) || any(!iszero, model.gravity)
        _phase_force_substep!(
            velocity, workspace, time_step, model, pressure_operator,
        )
    end
    _projection_weighted_gradient!(
        phase.acceleration, pressure, pressure_operator,
        workspace.projection.elliptic, model.interface.geometry,
    )
    _add_common_phase_acceleration!(velocity, phase.acceleration, -time_step)
    inherited_cell_flux = if isnothing(inherited_flux)
        nothing
    else
        _copy_from_field!(
            workspace.projection.elliptic.input, inherited_flux,
            pressure_operator.references,
        )
    end
    if !isnothing(inherited_velocity)
        return _project_two_velocity!(
            velocity, pressure, solver, workspace, state, time_step;
            pressure_prepared = true,
            tolerance = dimensionless_tolerance,
            max_iterations = cycle_limit,
            _inherited_cell_flux = inherited_cell_flux,
            _inherited_velocity = inherited_velocity,
            _reference_dt = reference_step,
        )
    end
    return _project_two_velocity!(
        velocity, pressure, solver, workspace, state, time_step;
        pressure_prepared = true,
        tolerance = dimensionless_tolerance,
        _inherited_cell_flux = inherited_cell_flux,
        _reference_dt = reference_step,
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
