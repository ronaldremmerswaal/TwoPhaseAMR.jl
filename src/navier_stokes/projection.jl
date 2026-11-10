"""Reusable pressure-increment, divergence, gradient, and elliptic scratch for `project!`."""
struct ProjectionWorkspace{T, M <: BlockTreeMesh}
    increment::CellField{T, M}
    divergence::CellField{T, M}
    inherited_divergence::CellField{T, M}
    weighted_gradient::FaceVelocity{T, M}
    elliptic::EllipticWorkspace{T}
end

function ProjectionWorkspace(solver::EllipticSolver{T}) where {T}
    mesh = solver.operator.mesh
    increment = CellField(mesh, T)
    divergence = CellField(mesh, T)
    inherited_divergence = CellField(mesh, T)
    weighted_gradient = FaceVelocity(
        mesh, (x, y, time, axis) -> zero(T); T
    )
    return ProjectionWorkspace(
        increment, divergence, inherited_divergence, weighted_gradient,
        EllipticWorkspace(solver.operator)
    )
end

function _validate_projection_compatibility(compatibility)
    compatibility in (:project, :error) || throw(
        ArgumentError("compatibility must be :project or :error"),
    )
    return nothing
end
function _validate_projection_reference_dt(dt, reference_dt)
    isfinite(reference_dt) && reference_dt >= dt || throw(
        ArgumentError("reference dt must be finite and at least dt"),
    )
    return nothing
end


"""Accumulate a mean-zero correction into the physical pressure."""
function _accumulate_pressure_increment!(pressure::CellField, increment::CellField)
    nx, ny = pressure.mesh.block_size
    halo = pressure.mesh.halo
    _parallel_block_columns(pressure.mesh) do block, first_j, last_j
        values = @inbounds pressure.blocks[block]
        correction = @inbounds increment.blocks[block]
        @inbounds for j in first_j:last_j, i in 1:nx
            values[i + halo, j + halo] += correction[i + halo, j + halo]
        end
    end
    fill_halo!(pressure)
    return pressure
end

"""Build the pressure-increment RHS, relaxing inherited divergence on the reference step."""
function _projection_rhs!(
        divergence::CellField{T}, dt, inherited_divergence, reference_dt,
    ) where {T}
    nx, ny = divergence.mesh.block_size
    halo = divergence.mesh.halo
    if isnothing(inherited_divergence)
        _parallel_block_columns(divergence.mesh) do block, first_j, last_j
            values = @inbounds divergence.blocks[block]
            @inbounds for j in first_j:last_j, i in 1:nx
                values[i + halo, j + halo] = -values[i + halo, j + halo] / dt
            end
        end
    else
        weight = (reference_dt - dt) / (dt * reference_dt)
        _parallel_block_columns(divergence.mesh) do block, first_j, last_j
            values = @inbounds divergence.blocks[block]
            inherited = @inbounds inherited_divergence.blocks[block]
            @inbounds for j in first_j:last_j, i in 1:nx
                index_i, index_j = i + halo, j + halo
                values[index_i, index_j] =
                    -values[index_i, index_j] / dt +
                    weight * inherited[index_i, index_j]
            end
        end
    end
    return divergence
end

"""
    project!(velocity, pressure, solver, dt, workspace;
             tolerance=sqrt(eps(T)), max_cycles=100)

Project a pressure-predicted velocity using the weighted gradient in the
solver's Poisson operator. The input pressure is the old physical pressure
and the velocity predictor must already include its pressure acceleration.
The mean-zero increment `φ` solves
`-diffusion*D*G_kappa*φ = -D*u*/dt`; the update is
`u <- u* - dt*diffusion*G_kappa*φ`, `p <- p + φ`.
Reusing a nonzero pressure field with an unpredicted velocity accumulates
projection impulses; use zero scratch pressure for an absolute projection.
Following Basilisk, `tolerance` bounds the fractional volume defect associated
with the linear-solve residual: `dt^2 * norm(residual, Inf) <= tolerance`
(up to the roundoff floor). A shortened simulation step can additionally
retain part of its inherited divergence.
"""
function project!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project, _inherited_divergence = nothing,
        _reference_dt = dt,
    ) where {T}
    _validate_projection_compatibility(compatibility)
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh ||
        throw(ArgumentError("projection arguments must share a mesh"))
    iszero(solver.operator.reaction) ||
        throw(ArgumentError("pressure projection requires a zero-reaction operator"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    _validate_projection_reference_dt(dt, _reference_dt)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)

    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary; time = boundary_time,
        )
    return _project_prepared!(
        velocity, pressure, solver, dt, workspace;
        velocity_boundary, boundary_time, tolerance = dimensionless_tolerance,
        max_cycles, compatibility, _inherited_divergence, _reference_dt,
    )
end

"""Project a velocity whose physical boundary was prepared before its pressure predictor."""
function _project_prepared!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt, workspace::ProjectionWorkspace{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project, _inherited_divergence = nothing,
        _reference_dt = dt,
    ) where {T}
    dimensionless_tolerance = T(tolerance)
    divergence!(workspace.divergence, velocity)
    _projection_rhs!(
        workspace.divergence, T(dt), _inherited_divergence, T(_reference_dt),
    )
    _zero_interior!(workspace.increment)
    stats = solve!(
        workspace.increment, solver, workspace.divergence;
        rtol = zero(T), atol = dimensionless_tolerance / T(dt)^2,
        max_cycles, compatibility,
    )
    stats.residual_scale = inv(T(dt)^2)
    weighted_gradient!(
        workspace.weighted_gradient, workspace.increment,
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
    _accumulate_pressure_increment!(pressure, workspace.increment)
    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary; time = boundary_time,
        )
    return stats
end
