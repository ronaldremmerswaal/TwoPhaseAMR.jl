# The `FaceVelocity` path is also the shared-velocity path for two-phase
# physics; two-phase stages extend this orchestration by dispatch.
function _enforce_current_velocity_boundary!(
        velocity::FaceVelocity, solver, time,
    )
    enforce_velocity_boundary!(
        velocity, solver.boundary; time, geometry = solver.geometry,
    )
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    return velocity
end


function _transport_substep!(
        velocity, solver, workspace, dt, time, ::Nothing
    )
    bound = _timestep_bound(workspace.old_velocity, solver, workspace, nothing; cfl = 0.5)
    dt <= bound * (1 + 32eps(eltype(velocity))) ||
        throw(ArgumentError("time step $dt exceeds advective stability bound $bound"))
    _copy_velocity!(workspace.convected, workspace.old_velocity)
    advect!(
        workspace.convected, workspace.old_velocity, dt,
        workspace.advection;
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme,
        boundary = solver.boundary, time,
    )
    return workspace.convected
end

function _diffusion_substep!(
        velocity, solver, workspace, dt, time, ::Nothing;
        tolerance = sqrt(eps(eltype(velocity))),
        max_cycles = 100
    )
    return implicit_midpoint_diffusion!(
        velocity, workspace.convected, workspace.old_velocity,
        solver.diffusion, dt, workspace.elliptic,
        workspace.diffusion_work; time, tolerance, max_cycles
    )
end

@inline _refresh_pressure_substep!(solver, workspace, ::Nothing) =
    workspace.projection

@inline _phase_force_substep!(velocity, solver, workspace, dt, ::Nothing) =
    velocity

@inline _timestep_bound(velocity, solver, workspace, ::Nothing; cfl) =
    stable_timestep(velocity, workspace.advection; cfl)

"""Apply the old pressure with the same map used by the increment projection."""
function _pressure_predictor!(
        velocity::FaceVelocity{T}, pressure, solver, workspace, dt,
    ) where {T}
    projection = workspace.projection
    operator = solver.pressure_solver.operator
    _projection_weighted_gradient!(
        projection.weighted_gradient, pressure, operator, projection.elliptic,
        solver.geometry,
    )
    scale = T(dt) * operator.diffusion
    @_parallel for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -=
            scale * projection.weighted_gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -=
            scale * projection.weighted_gradient.boundary[index]
    end
    return velocity
end


"""
    step!(velocity, pressure, solver, dt, workspace;
          time=0, force=nothing, tolerance=sqrt(eps(T)),
          max_cycles=nothing)

Advance one split momentum step followed by one incremental pressure projection.
Both single- and two-phase configurations execute the same staged procedure.
`force`, when present, is sampled as `force(x, y, time, axis)` at the time
midpoint and added to the transported state before diffusion, preserving the
single-phase solver's established splitting. Phase forces and the old
pressure gradient are added after diffusion, with the same coefficient and
cut-cell map used by the pressure-increment projection.
The internal `_reference_dt` defaults to `dt` and must not be smaller than
`dt`; a larger reference retains the corresponding fraction of the inherited
divergence defect instead of amplifying it in the pressure correction.
The transporting flux is
frozen at the old-time velocity while each coordinate sweep updates both
transported velocity components. No-permeation is automatic in non-periodic
directions.
"""
function step!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
        _transport_geometry_prepared = false,
        _reference_dt = dt,
    ) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    time_step = _positive_finite_step(T, dt)
    reference_step = _positive_finite_step(T, _reference_dt, "reference dt")
    reference_step >= time_step ||
        throw(ArgumentError("reference dt must be greater than or equal to dt"))
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    _invalidate_simulation_step!(workspace)
    workspace.unresolved_curvature_faces[] = 0
    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_velocity_boundary!(
        workspace.old_velocity, solver.boundary;
        time = current_time, geometry = solver.geometry,
    )
    isnothing(solver.geometry) || enforce_geometry_no_slip!(
        workspace.old_velocity, solver.geometry,
    )
    inherited_divergence = if time_step == reference_step
        nothing
    elseif isnothing(solver.geometry)
        divergence!(workspace.projection.inherited_divergence, workspace.old_velocity)
    else
        divergence!(
            workspace.projection.inherited_divergence,
            workspace.old_velocity, solver.geometry,
        )
    end
    model = solver.two_phase
    _transport_substep!(
        velocity, solver, workspace, time_step, current_time, model
    )
    midpoint = current_time + time_step / 2
    _body_force_substep!(workspace.convected, time_step, midpoint, force)
    enforce_velocity_boundary!(
        workspace.convected, solver.boundary;
        time = midpoint, geometry = solver.geometry,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
    enforce_velocity_boundary!(
        velocity, solver.boundary;
        time = current_time + time_step, geometry = solver.geometry,
    )
    _refresh_pressure_substep!(solver, workspace, model)
    _phase_force_substep!(velocity, solver, workspace, time_step, model)
    _enforce_current_velocity_boundary!(
        velocity, solver, current_time + time_step,
    )
    _pressure_predictor!(velocity, pressure, solver, workspace, time_step)

    if isnothing(solver.geometry)
        return _project_prepared!(
            velocity, pressure, solver.pressure_solver, time_step,
            workspace.projection;
            velocity_boundary = solver.boundary,
            boundary_time = current_time + time_step,
            _inherited_divergence = inherited_divergence,
            _reference_dt = reference_step,
            tolerance = dimensionless_tolerance,
            max_cycles = cycle_limit,
        )
    end
    return _project_prepared!(
        velocity, pressure, solver.pressure_solver, time_step,
        workspace.projection, solver.geometry;
        velocity_boundary = solver.boundary,
        boundary_time = current_time + time_step,
        _inherited_divergence = inherited_divergence,
        _reference_dt = reference_step,
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
end

"""Return the convective CFL bound for a Navier--Stokes step."""
function stable_timestep(
        velocity::FaceVelocity,
        solver::IncompressibleNavierStokesSolver,
        workspace::NavierStokesWorkspace;
        cfl = 0.45
    )
    return _timestep_bound(
        velocity, solver, workspace, solver.two_phase; cfl
    )
end
