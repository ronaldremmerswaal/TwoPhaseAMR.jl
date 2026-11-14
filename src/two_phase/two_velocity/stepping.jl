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
        velocity::TwoPhaseVelocity, solver, time, inactive_failures,
    )
    if solver.boundary isa Union{NoSlip, FreeSlip}
        enforce_no_permeation!(velocity)
    else
        for phase in 1:2
            enforce_velocity_boundary!(
                velocity[phase], solver.boundary;
                time, geometry = solver.geometry, allow_absent = true,
                _require_compatible_flux = true,
            )
        end
    end
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    _mark_inactive_phase_velocities!(
        velocity, solver.two_phase.interface, inactive_failures,
    )
    return velocity
end


@inline _enforce_current_velocity_boundary!(
    velocity::TwoPhaseVelocity, solver, time, inactive_failures,
) = _enforce_phase_velocity_boundary!(velocity, solver, time, inactive_failures)


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
        _check_transport_timestep(
            velocity, eltype(velocity[1]), solver, workspace, dt, model,
        )
    end
    _advect_transport_substep!(
        model.interface, state.old_velocity, dt, state.advection;
        scalars = model.scalars,
        momentum = state.convected, density = model.density,
        momentum_workspace = state.momentum,
        sweep_order = phase.forward ? (1, 2) : (2, 1),
        scheme = solver.options.advection.scheme,
        amr_interface_scheme = solver.options.advection.amr_interface_scheme,
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

"""Apply one common viscous increment, enforcing prescribed phase velocities."""
function _apply_two_velocity_diffusion_update!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
        diffused_mixture::FaceVelocity, mixture::FaceVelocity;
        dirichlet_faces = (),
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
    return _prescribe_two_velocity_diffusion_values!(destination, source, dirichlet_faces)
end

function _prescribe_two_velocity_diffusion_values!(destination, source, dirichlet_faces)
    T = eltype(destination[1])
    for phase in 1:2, index in dirichlet_faces
        @inbounds if !isnan(source[phase].interior[index])
            destination[phase].interior[index] = zero(T)
        end
    end
    for phase in 1:2, index in eachindex(destination[phase].boundary)
        destination[phase].boundary[index] = isnan(source[phase].boundary[index]) ?
            T(NaN) : zero(T)
    end
    return destination
end

"""Record the direct metric viscous solve of `MimeticProjection` in the
statistics of the diffusion solver: one cycle with the relative residual."""
function _record_mimetic_viscous_solve!(stats::LinearSolverStats{T}, residual, tolerance) where {T}
    stats.cycles = 1
    stats.converged = residual <= _dimensionless_tolerance(T, tolerance)
    stats.final_l2 = stats.final_linf = T(residual)
    stats.method = :cholesky
    stats.preconditioner = :none
    stats.converged || throw(
        ErrorException("mimetic viscous solve: relative residual $residual above the tolerance $tolerance"),
    )
    return stats
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
    if state.projection_solver isa MimeticProjectionSolver
        # Implicit midpoint of the phase values in the metric of the
        # projection, unconditionally non-increasing in that metric.
        result = _mimetic_viscous_step!(
            velocity, state.convected, state.projection_solver,
            state.projection_workspace, diffusion.stiffness, dt,
        )
        _record_mimetic_viscous_solve!(
            workspace.two_phase.diffusion_solver.stats, result.residual, tolerance,
        )
        _prescribe_two_velocity_diffusion_values!(
            velocity, state.convected, diffusion.dirichlet_faces,
        )
    else
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
            state.diffused_mixture, state.mixture_velocity;
            dirichlet_faces = diffusion.dirichlet_faces,
        )
    end
    # The common increment leaves the jump unchanged and the drag the mixture,
    # so the two updates commute and the full drag step needs no splitting.
    _interphase_drag!(velocity, model.interphase_drag, model, diffusion, dt)
    _mark_inactive_phase_velocities!(
        velocity, model.interface, workspace.two_phase.inactive_failures,
    )
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
    data = _two_velocity_projection_face_data!(state.projection_solver, work)
    projection = workspace.projection
    flux = projection.elliptic.input
    _cmdgfm_cell_flux!(
        flux, data, solver.two_phase.interface, state.old_velocity,
    )
    # The force stage and CMDGFM pressure predictor reuse elliptic.input.
    # Keep the old-time integrated flux in cell scratch until they are done.
    return _copy_to_field!(
        projection.inherited_divergence, flux,
        solver.pressure_solver.operator.references,
    )
end

function _two_velocity_projection_face_data!(projection::CMDGFMProjectionSolver, work)
    cache = work.pressure_cache
    data = _projection_face_data(projection, work; storage = cache.face_data)
    cache.face_data = data
    return data
end

function _two_velocity_projection_face_data!(projection::MimeticProjectionSolver, work)
    data = _projection_face_data(projection, work; storage = work.face_data)
    work.face_data = data
    return data
end

function _phase_force_pressure_operator!(projection, work::CMDGFMProjectionWorkspace)
    cache = work.pressure_cache
    data = _projection_face_data(projection, work; storage = cache.face_data)
    cache.face_data = data
    return _cmdgfm_pressure_cache!(cache, projection, data).operator
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
    # Cut-cell CMDGFM uses its own pressure operator. In particular, the
    # two-velocity cut-cell coefficient is A/rho, not A²/rho.
    return _phase_force_pressure_operator!(
        state.projection_solver, state.projection_workspace,
    )
end

function _phase_force_substep!(
        velocity::TwoPhaseVelocity, workspace, dt,
        model::TwoPhaseModel, operator::EllipticOperator,
    )
    return _apply_phase_forces!(velocity, workspace, dt, model, operator)
end

"""Apply phase forces and subtract the old CMDGFM pressure gradient."""
function _two_velocity_forces_and_pressure!(
        velocity, pressure, solver, workspace, state, model::TwoPhaseModel{T},
        time_step,
    ) where {T}
    phase = workspace.two_phase
    phase_forces = model.sigma > zero(T) || any(!iszero, model.gravity)
    pressure_operator = _two_velocity_pressure_operator!(
        solver, workspace, state, model,
    )
    if phase_forces
        _phase_force_substep!(
            velocity, workspace, time_step, model, pressure_operator,
        )
    end
    _projection_weighted_gradient!(
        phase.acceleration, pressure, pressure_operator,
        workspace.projection.elliptic, model.interface.geometry,
    )
    _add_common_phase_acceleration!(velocity, phase.acceleration, -time_step)
    return pressure_operator
end

"""Apply phase forces for the mimetic projection.

Surface tension is the per-phase capillary force through the mimetic gradient
map `𝓜⁻¹Dᵀ` (see `_mimetic_capillary_force!`); gravity is the uniform
acceleration of both phases (see `_mimetic_uniform_acceleration!`). The
mimetic projection is absolute, so no old pressure gradient is subtracted: an
`𝓜⁻¹Dᵀ` gradient in the predictor is removed exactly, and the projection
returns the full pressure.
"""
function _two_velocity_forces_and_pressure!(
        velocity, pressure, solver, workspace,
        state::TwoVelocityFlowWorkspace{<:Any, <:Any, <:Any, <:Any, <:MimeticProjectionSolver},
        model::TwoPhaseModel{T}, time_step,
    ) where {T}
    isnothing(solver.geometry) || throw(
        ArgumentError("MimeticProjection does not support embedded solid geometry"),
    )
    reconstruct!(model.interface)
    if model.sigma > zero(T)
        _mimetic_capillary_force!(
            velocity, state.projection_solver, state.projection_workspace, time_step;
            sigma = model.sigma, formulation = model.formulation,
            unresolved_curvature_faces = workspace.unresolved_curvature_faces,
        )
    end
    if any(!iszero, model.gravity)
        _mimetic_uniform_acceleration!(
            velocity, model.interface.mesh, model.gravity, time_step,
        )
    end
    # Only the cell references of this operator are used.
    return solver.pressure_solver.operator
end


"""Advance one two-velocity Navier--Stokes step.

The geometric interface and both intrinsic phase velocities use the same
transport stage as the passive two-velocity API. Body force, surface tension,
and gravity modify the momentum predictor. CMDGFM subtracts the old physical
pressure gradient from the predictor, then corrects velocity with a pressure
increment that is added to the physical pressure.

For a shortened CMDGFM step, the old-aperture mixture flux is scaled by
the reference step, which must be at least `dt`.
Viscosity acts implicitly on the phase-mass weighted one-velocity state; its
common increment is applied to both phases. With `MimeticProjection` the
phase values instead take the implicit-midpoint step of the same stress in
the projection's metric (see `_mimetic_viscous_step!`), which is that common
increment where the metric is the staggered phase mass. The interphase drag
of the model then relaxes the phase-velocity jump of the mixed faces.

The two-velocity projection keeps a Neumann pressure boundary and cannot
correct open normal velocities. Boundary preparation extrapolates each phase;
`Outflow()` balances its net flux, while an absolute phase flux exceeding
`256eps(T)` with only `ZeroGradientVelocity()` open sides throws
`ArgumentError`. This per-phase check sums outward intrinsic velocity times
face area and geometry aperture, omitting faces whose phase velocity is absent.
"""
function step!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::NavierStokesSystem{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
        _reference_dt = dt,
    ) where {T}
    geometry_prepared = workspace.transport_geometry_prepared
    workspace.transport_geometry_prepared = false
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
    time_step, reference_step, current_time, dimensionless_tolerance, cycle_limit =
        _prepare_step_controls!(
        velocity, solver, workspace, dt, _reference_dt, time,
        tolerance, max_cycles,
    )

    state = _two_velocity_flow_workspace!(phase, model)
    _copy_velocity!(state.old_velocity, velocity)
    _enforce_phase_velocity_boundary!(
        state.old_velocity, solver, current_time, phase.inactive_failures,
    )
    inherited_flux = time_step == reference_step ? nothing :
        _capture_two_velocity_inherited_flux!(solver, workspace, state)
    # The CFL check must see the same boundary-enforced velocity as advection.
    _transport_substep!(
        state.old_velocity, solver, workspace, time_step, current_time, model;
        geometry_prepared,
    )
    midpoint = current_time + time_step / 2
    convected = state.convected
    _body_force_substep!(convected, time_step, midpoint, force)
    _enforce_phase_velocity_boundary!(
        convected, solver, midpoint, phase.inactive_failures,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance, max_cycles = cycle_limit,
    )
    _enforce_phase_velocity_boundary!(
        velocity, solver, current_time + time_step, phase.inactive_failures,
    )
    pressure_operator = _two_velocity_forces_and_pressure!(
        velocity, pressure, solver, workspace, state, model, time_step,
    )
    inherited_cell_flux = if isnothing(inherited_flux)
        nothing
    else
        _copy_from_field!(
            workspace.projection.elliptic.input, inherited_flux,
            pressure_operator.references,
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
        solver::NavierStokesSystem,
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
