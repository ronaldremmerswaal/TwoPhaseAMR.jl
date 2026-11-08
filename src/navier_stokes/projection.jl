"""Reusable divergence, gradient, and elliptic scratch for `project!`."""
struct ProjectionWorkspace{T, M <: BlockTreeMesh}
    divergence::CellField{T, M}
    weighted_gradient::FaceVelocity{T, M}
    elliptic::EllipticWorkspace{T}
end

function ProjectionWorkspace(solver::EllipticSolver{T}) where {T}
    mesh = solver.operator.mesh
    divergence = CellField(mesh, T)
    weighted_gradient = FaceVelocity(
        mesh, (x, y, time, axis) -> zero(T); T
    )
    return ProjectionWorkspace(
        divergence, weighted_gradient,
        EllipticWorkspace(solver.operator)
    )
end

function _validate_projection_compatibility(compatibility)
    compatibility in (:project, :error) || throw(
        ArgumentError("compatibility must be :project or :error"),
    )
    return nothing
end

"""
    project!(velocity, pressure, solver, dt, workspace;
             tolerance=sqrt(eps(T)), max_cycles=100)

Apply the pressure projection using exactly the weighted gradient in the
solver's Poisson operator. Physical face velocities are first set to zero; the
corresponding pressure correction has homogeneous Neumann data. The pressure is overwritten by the mean-zero
solution of `-diffusion*D*G_kappa*p = -D*u/dt`, then
`u <- u - dt*diffusion*G_kappa*p`. Following Basilisk, `tolerance`
bounds the maximum fractional cell-volume defect left by the projection:
`dt^2 * norm(residual, Inf) <= tolerance` (up to the roundoff floor).
"""
function project!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project
    ) where {T}
    _validate_projection_compatibility(compatibility)
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh ||
        throw(ArgumentError("projection arguments must share a mesh"))
    iszero(solver.operator.reaction) ||
        throw(ArgumentError("pressure projection requires a zero-reaction operator"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)

    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary; time = boundary_time,
        )
    divergence!(workspace.divergence, velocity)
    nx, ny = velocity.mesh.block_size
    halo = velocity.mesh.halo
    _parallel_block_columns(velocity.mesh) do block, first_j, last_j
        values = @inbounds workspace.divergence.blocks[block]
        @inbounds for j in first_j:last_j, i in 1:nx
            values[i + halo, j + halo] = -values[i + halo, j + halo] / dt
        end
    end
    stats = solve!(
        pressure, solver, workspace.divergence;
        rtol = zero(T), atol = dimensionless_tolerance / T(dt)^2,
        max_cycles, compatibility,
    )
    stats.residual_scale = inv(T(dt)^2)
    weighted_gradient!(
        workspace.weighted_gradient, pressure,
        solver.operator, workspace.elliptic
    )
    scale = T(dt) * solver.operator.diffusion
    for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -=
            scale * workspace.weighted_gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -=
            scale * workspace.weighted_gradient.boundary[index]
    end
    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary; time = boundary_time,
        )
    return stats
end
