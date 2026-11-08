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
    solver = MimeticProjectionSolver(
        field; density, variant = projection.variant,
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
    advection.interface === field || throw(ArgumentError(
        "two-velocity advection workspace has the wrong interface",
    ))
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
    # Preserve the Krylov, ILU/GMG history and geometric templates across AMR.
    # Topology-dependent cache data is validated against block keys at the
    # next assembly. The transport buffers have their existing AMR rebuild.
    state isa TwoVelocityFlowWorkspace || return nothing
    state.projection_solver isa CMDGFMProjectionSolver || return nothing
    mesh = model.interface.mesh
    for velocity in (state.old_velocity..., state.convected...,
            state.mixture_velocity, state.diffused_mixture)
        _resize_zero_velocity!(velocity, mesh)
    end
    work = state.projection_workspace
    for velocity in (work.pressure_gradient, work.base_gradient,
            work.liquid_gradient, work.gas_gradient)
        _resize_zero_velocity!(velocity, mesh)
    end
    resize!(work.jump, length(mesh.faces))
    fill!(work.jump, zero(eltype(mesh.origin)))
    _resize_face_field!(work.liquid_fraction_x, mesh)
    _resize_face_field!(work.liquid_fraction_y, mesh)
    # Old matrix indices no longer refer to this live mesh; never expose it
    # as a current assembled system before the next projection.
    work.matrix = spzeros(eltype(mesh.origin), 0, 0)
    work.jump_block = spzeros(eltype(mesh.origin), 0, 0)
    empty!(work.pressure_indices)
    empty!(work.mixed_faces)
    empty!(work.stencils)
    return TwoVelocityFlowWorkspace(advection,
        _reconfigure_momentum_workspace!(state.momentum, model.interface),
        state.old_velocity, state.convected, state.mixture_velocity,
        state.diffused_mixture, state.projection_solver, work)
end

function _project_two_velocity!(velocity, pressure, solver, workspace, state, dt;
        pressure_prepared = false, kwargs...)
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
        return project!(velocity, pressure, projection, dt, state.projection_workspace;
            _pressure_operator = solver.pressure_solver.operator,
            _pressure_multigrid = shared, kwargs...)
    end
    return project!(velocity, pressure, projection, dt, state.projection_workspace; kwargs...)
end
