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


"""
    step!(velocity, pressure, solver, dt, workspace;
          time=0, force=nothing, tolerance=sqrt(eps(T)),
          max_cycles=nothing)

Advance one split momentum step followed by exactly one pressure projection.
Both single- and two-phase configurations execute the same staged procedure.
`force`, when present, is sampled as `force(x, y, time, axis)` at the time
midpoint and added to the transported state before diffusion, preserving the
single-phase solver's established splitting. Thesis phase forces are added
after diffusion so hydrostatic equilibria are unchanged by the viscous solve.
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
        tolerance = sqrt(eps(T)), max_cycles = nothing
    ) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    time_step = _positive_finite_step(T, dt)
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_velocity_boundary!(
        workspace.old_velocity, solver.boundary;
        time = current_time, geometry = solver.geometry,
    )
    isnothing(solver.geometry) || enforce_geometry_no_slip!(
        workspace.old_velocity, solver.geometry,
    )
    model = solver.two_phase
    _transport_substep!(
        velocity, solver, workspace, time_step, current_time, model
    )
    midpoint = current_time + time_step / 2
    for (index, face) in pairs(solver.mesh.faces)
        @inbounds workspace.convected.interior[index] +=
            time_step * _body_force(force, face, midpoint)
    end
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

    if isnothing(solver.geometry)
        return project!(
            velocity, pressure, solver.pressure_solver, time_step,
            workspace.projection;
            velocity_boundary = solver.boundary,
            boundary_time = current_time + time_step,
            tolerance = dimensionless_tolerance,
            max_cycles = cycle_limit,
        )
    end
    return project!(
        velocity, pressure, solver.pressure_solver, time_step,
        workspace.projection, solver.geometry;
        velocity_boundary = solver.boundary,
        boundary_time = current_time + time_step,
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
