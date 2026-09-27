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

function _two_velocity_flow_workspace!(workspace, model)
    state = workspace.two_velocity[]
    if isnothing(state)
        state = _two_velocity_flow_workspace(model, workspace.advection)
        workspace.two_velocity[] = state
    end
    return state
end
