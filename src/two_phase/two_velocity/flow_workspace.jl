# Workspace state and pressure-projection construction owned by the
# intrinsic two-velocity model.
mutable struct TwoVelocityFlowWorkspace{A, M, V, F, S, W}
    advection::A
    momentum::M
    old_velocity::V
    convected::V
    mixture_velocity::F
    diffused_mixture::F
    projection_solver::S
    projection_workspace::W
end

function _two_velocity_projection_workspace(
        field, density, projection::MimeticProjection,
    )
    solver = _mimetic_projection_solver(
        field, projection.variant; density,
        preconditioner = projection.preconditioner,
        preconditioner_options = projection.preconditioner_options,
        active_mass_fraction = projection.active_mass_fraction,
        rank_tolerance = projection.rank_tolerance,
        patch_radius = projection.patch_radius,
        pressure_weight = projection.pressure_weight,
        commutator_weight = projection.commutator_weight,
        correction_cap = projection.correction_cap,
        fit_sweeps = projection.fit_sweeps,
    )
    return solver, MimeticProjectionWorkspace(solver)
end

function _two_velocity_projection_workspace(
        field, density, projection::CMDGFMProjection,
    )
    solver = CMDGFMProjectionSolver(
        field; density, preconditioner = projection.preconditioner,
        preconditioner_options = projection.preconditioner_options,
    )
    return solver, CMDGFMProjectionWorkspace(solver)
end


function _two_velocity_flow_workspace(
        model::TwoPhaseModel{T}, advection::WYAdvectionWorkspace,
    ) where {T}
    field = model.interface
    advection.interface === field || throw(
        ArgumentError(
            "two-velocity advection workspace has the wrong interface",
        )
    )
    old_velocity = (
        FaceVelocity(field.mesh, T), FaceVelocity(field.mesh, T),
    )
    convected = (
        FaceVelocity(field.mesh, T), FaceVelocity(field.mesh, T),
    )
    mixture_velocity = FaceVelocity(field.mesh, T)
    diffused_mixture = FaceVelocity(field.mesh, T)
    momentum = MomentumAdvectionWorkspace(field, old_velocity)
    solver, projection_workspace = _two_velocity_projection_workspace(
        field, model.density, model.pressure_projection,
    )
    return TwoVelocityFlowWorkspace(
        advection, momentum, old_velocity, convected,
        mixture_velocity, diffused_mixture, solver,
        projection_workspace,
    )
end

function _two_velocity_flow_workspace!(
        workspace::TwoPhaseFlowWorkspace{T, M, D, S, C, A, N, V}, model,
    ) where {T, M, D, S, C, A, N, V}
    state = workspace.two_velocity[]
    if isnothing(state)
        state = _two_velocity_flow_workspace(model, workspace.advection)
        workspace.two_velocity[] = state
    end
    return state::V
end

function _reconfigure_two_velocity_flow_workspace(state, model, advection)
    # Preserve Krylov and geometric templates across AMR. Topology-dependent
    # caches are validated against block keys at the next assembly.
    state isa TwoVelocityFlowWorkspace || return nothing
    state.projection_solver isa Union{CMDGFMProjectionSolver, MimeticProjectionSolver} ||
        return nothing
    mesh = model.interface.mesh
    for velocity in (
            state.old_velocity..., state.convected...,
            state.mixture_velocity, state.diffused_mixture,
        )
        _resize_zero_velocity!(velocity, mesh)
    end
    work = state.projection_workspace
    for velocity in (
            work.pressure_gradient, work.base_gradient,
            work.liquid_gradient, work.gas_gradient,
        )
        _resize_zero_velocity!(velocity, mesh)
    end
    resize!(work.jump, length(mesh.faces))
    fill!(work.jump, zero(eltype(mesh.origin)))
    _resize_face_field!(work.liquid_fraction_x, mesh)
    _resize_face_field!(work.liquid_fraction_y, mesh)
    # Old matrix indices no longer refer to this live mesh; never expose it
    # as a current assembled system before the next projection.
    work.matrix = spzeros(eltype(mesh.origin), 0, 0)
    if work isa CMDGFMProjectionWorkspace
        work.jump_block = spzeros(eltype(mesh.origin), 0, 0)
        empty!(work.stencils)
    else
        work.block_preconditioner = spzeros(eltype(mesh.origin), 0, 0)
        work.constraint_matrix = spzeros(eltype(mesh.origin), 0, 0)
        work.active_faces = Int[]
        empty!(work.constraint_locations)
        empty!(work.constraint_scales)
    end
    # The visible indices may alias the cache's gauge/active buffers. Detach
    # instead of clearing the cached state if a rebuild retains the topology.
    work.pressure_indices = Int[]
    work.mixed_faces = Int[]
    return TwoVelocityFlowWorkspace(
        advection,
        _reconfigure_momentum_workspace!(state.momentum, model.interface),
        state.old_velocity, state.convected, state.mixture_velocity,
        state.diffused_mixture, state.projection_solver, work
    )
end

function _project_two_velocity!(
        velocity, pressure, solver, workspace, state, dt;
        pressure_prepared = false, kwargs...
    )
    projection = state.projection_solver
    model = solver.two_phase
    compatible = projection isa CMDGFMProjectionSolver &&
        isnothing(solver.geometry)
    if compatible
        pressure_prepared || _refresh_pressure_substep!(solver, workspace, model)
        shared = nothing
        if projection.preconditioner === :block
            settings = _cmdgfm_block_settings(projection.preconditioner_options)
            candidate = solver.pressure_solver.preconditioner
            if candidate isa TreeGeometricMultigridPreconditioner &&
                    isempty(solver.pressure_preconditioner_options) &&
                    settings.refresh_pressure_coefficients &&
                    candidate.pre_sweeps == settings.pressure_pre_sweeps &&
                    candidate.post_sweeps == settings.pressure_post_sweeps &&
                    candidate.pruning == settings.pressure_pruning &&
                    solver.coarse_size == settings.pressure_coarse_size &&
                    solver.max_levels == settings.pressure_max_levels
                shared = candidate
            end
        end
        return project!(
            velocity, pressure, projection, dt, state.projection_workspace;
            _pressure_operator = solver.pressure_solver.operator,
            _pressure_multigrid = shared, kwargs...
        )
    end
    return project!(velocity, pressure, projection, dt, state.projection_workspace; kwargs...)
end

"""Restore the two-phase incompressibility constraint after an AMR transfer."""
function _project_regridded_two_velocity!(
        velocity::TwoPhaseVelocity, solver, workspace, dt, time;
        tolerance, max_iterations,
    )
    T = eltype(velocity[1])
    model = solver.two_phase
    state = _two_velocity_flow_workspace!(workspace.two_phase, model)
    _enforce_phase_velocity_boundary!(velocity, solver, time)
    # The coupled projector compares a volume defect and an absolute normal
    # velocity defect to the same tolerance. Divide large velocities by their
    # scale and multiply dt by that scale: dt*u, and hence the WY volume
    # defect, is unchanged. This keeps the jump tolerance relative to the
    # representable velocity scale without loosening volume conservation.
    velocity_scale = one(T)
    for component in velocity, values in (component.interior, component.boundary)
        for value in values
            isfinite(value) && (velocity_scale = max(velocity_scale, abs(value)))
        end
    end
    normalized_velocity = velocity_scale == one(T) ? velocity : state.old_velocity
    if velocity_scale != one(T)
        inverse_scale = inv(velocity_scale)
        for phase in 1:2
            source, destination = velocity[phase], normalized_velocity[phase]
            for (input, output) in (
                    (source.interior, destination.interior),
                    (source.boundary, destination.boundary),
                )
                @inbounds for index in eachindex(input, output)
                    output[index] = input[index] * inverse_scale
                end
            end
        end
    end
    # The transferred pressure is the physical pressure at the end of the
    # preceding step, not the pressure impulse for this geometric correction.
    # The ordinary projection divergence buffer is free here and is already
    # sized for the new mesh; the two-velocity projector does not use it.
    correction_pressure = workspace.projection.divergence
    # WY freezes the phase apertures for both directional sweeps. A
    # divergence-free mixture makes the indicator-weighted dilation cancel
    # cell by cell. Do not let a loose flow-solver tolerance reintroduce a
    # measurable volume jump on the first post-regrid step.
    correction_tolerance = min(T(tolerance) / velocity_scale, T(128) * eps(T))
    stats = _project_two_velocity!(
        normalized_velocity, correction_pressure, solver, workspace, state,
        T(dt) * velocity_scale;
        pressure_prepared = true, tolerance = correction_tolerance,
        max_iterations,
    )
    stats.converged || throw(
        ErrorException(
            _solver_failure_message(
                "post-regrid two-velocity projection", stats,
                correction_tolerance;
                details = "dt=$(dt), velocity_scale=$(velocity_scale), " *
                    "time=$(time), max_cycles=$(max_iterations)",
            ),
        ),
    )
    if velocity_scale != one(T)
        for phase in 1:2
            source, destination = normalized_velocity[phase], velocity[phase]
            for (input, output) in (
                    (source.interior, destination.interior),
                    (source.boundary, destination.boundary),
                )
                @inbounds for index in eachindex(input, output)
                    output[index] = input[index] * velocity_scale
                end
            end
        end
    end
    return stats
end
