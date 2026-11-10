# The `FaceVelocity` path is also the shared-velocity path for two-phase
# physics; two-phase stages extend this orchestration by dispatch.
function _enforce_current_velocity_boundary!(
        velocity::FaceVelocity, solver, time; preserve_open = false,
    )
    enforce_velocity_boundary!(
        velocity, solver.boundary; time, geometry = solver.geometry, preserve_open,
    )
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    return velocity
end
"""Extrapolate the predictor change without discarding the projected open flux."""
function _enforce_predictor_velocity_boundary!(
        velocity::FaceVelocity{T}, old_velocity::FaceVelocity{T}, solver, time;
        enforce_geometry = false,
    ) where {T}
    enforce_velocity_boundary!(
        velocity, solver.boundary; time, geometry = solver.geometry, preserve_open = true,
    )
    if enforce_geometry && !isnothing(solver.geometry)
        enforce_geometry_no_slip!(velocity, solver.geometry)
    end
    boundary = solver.boundary
    boundary isa VelocityBoundaryConditions || return velocity
    mesh = velocity.mesh
    geometry = solver.geometry
    sources = _boundary_extrapolation_schedule!(boundary, mesh)
    for (index, face) in pairs(mesh.boundary_faces)
        condition = _side_boundary(boundary, face.axis, face.side)
        condition isa Union{Outflow, ZeroGradientVelocity} || continue
        if !isnothing(geometry) && iszero(geometry.boundary_aperture[index])
            velocity.boundary[index] = zero(T)
            continue
        end
        numerator = zero(T)
        denominator = zero(T)
        for source in sources[index]
            aperture = isnothing(geometry) ? one(T) : geometry.aperture[source]
            weight = T(mesh.faces[source].area) * aperture
            @inbounds numerator += weight *
                (velocity.interior[source] - old_velocity.interior[source])
            denominator += weight
        end
        change = denominator > zero(T) ? numerator / denominator : zero(T)
        @inbounds velocity.boundary[index] = old_velocity.boundary[index] + change
    end
    return velocity
end


@inline _reference_step_message(::FaceVelocity) =
    "reference dt must be greater than or equal to dt"
@inline _reference_step_message(::TwoPhaseVelocity) =
    "reference dt must be at least dt"

function _prepare_step_controls!(
        velocity::Union{FaceVelocity, TwoPhaseVelocity},
        solver::NavierStokesSystem{T}, workspace, dt, reference_dt, time,
        tolerance, max_cycles,
    ) where {T}
    time_step = _positive_finite_step(T, dt)
    reference_step = _positive_finite_step(T, reference_dt, "reference dt")
    reference_step >= time_step ||
        throw(ArgumentError(_reference_step_message(velocity)))
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    _invalidate_simulation_step!(workspace)
    workspace.unresolved_curvature_faces[] = 0
    return time_step, reference_step, current_time, dimensionless_tolerance, cycle_limit
end

@inline _transport_limit_name(bound, solver, ::Nothing) = "advective"

function _check_transport_timestep(
        velocity, ::Type{T}, solver, workspace, dt, model,
    ) where {T}
    bound = _timestep_bound(velocity, solver, workspace, model; cfl = 0.5)
    dt <= bound * (1 + 1.0e-6 + 32eps(T)) ||
        throw(ArgumentError("time step $dt exceeds $(_transport_limit_name(bound, solver, model)) stability bound $bound"))
    return nothing
end

function _transport_substep!(
        velocity, solver, workspace, dt, time, ::Nothing
    )
    _check_transport_timestep(
        workspace.old_velocity, eltype(velocity), solver, workspace, dt, nothing,
    )
    _copy_velocity!(workspace.convected, workspace.old_velocity)
    advect!(
        workspace.convected, workspace.old_velocity, dt,
        workspace.advection;
        scheme = solver.options.advection.scheme,
        amr_interface_scheme = solver.options.advection.amr_interface_scheme,
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
    operator = _one_velocity_pressure_solver!(solver, workspace).operator
    _projection_weighted_gradient!(
        projection.weighted_gradient, pressure, operator, projection.elliptic,
        solver.geometry,
    )
    _apply_pressure_predictor!(
        velocity, projection.weighted_gradient, T(dt) * operator.diffusion,
    )
    return velocity
end

function _apply_pressure_predictor!(velocity::FaceVelocity, gradient::FaceVelocity, scale)
    @_parallel for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -= scale * gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -= scale * gradient.boundary[index]
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
        solver::NavierStokesSystem{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
        _transport_geometry_prepared = false,
        _reference_dt = dt,
    ) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    time_step, reference_step, current_time, dimensionless_tolerance, cycle_limit =
        _prepare_step_controls!(
        velocity, solver, workspace, dt, _reference_dt, time,
        tolerance, max_cycles,
    )
    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_velocity_boundary!(
        workspace.old_velocity, solver.boundary;
        time = current_time, geometry = solver.geometry, preserve_open = true,
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
    _enforce_predictor_velocity_boundary!(
        workspace.convected, workspace.old_velocity, solver, midpoint,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
    if isnothing(model) || !any(_pressure_boundary_sides(solver.boundary))
        _refresh_pressure_substep!(solver, workspace, model)
    end
    pressure_solver = _one_velocity_pressure_solver!(solver, workspace; refresh = true)
    _phase_force_substep!(velocity, solver, workspace, time_step, model)
    _enforce_predictor_velocity_boundary!(
        velocity, workspace.old_velocity, solver, current_time + time_step;
        enforce_geometry = true,
    )
    _pressure_predictor!(velocity, pressure, solver, workspace, time_step)

    return _project_prepared!(
        velocity, pressure, pressure_solver, time_step,
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
        solver::NavierStokesSystem,
        workspace::NavierStokesWorkspace;
        cfl = 0.45
    )
    return _timestep_bound(
        velocity, solver, workspace, solver.two_phase; cfl
    )
end
