"""
    IncompressibleNavierStokesSolver(mesh; viscosity,
        boundary=NoSlip(), scheme=LW(), amr_interface_scheme=scheme,
        diffusion_method=:auto, pressure_method=:auto)

Fixed-tree, single-phase incompressible Navier--Stokes discretization. One time
step is split into:

1. a conservative Lax--Wendroff convection update (second order in space and
   time on regular staggered regions),
2. a Crank--Nicolson/implicit-midpoint staggered viscous solve, and
3. one compatible pressure projection.

The tree may contain static local refinement, but its topology must not change
while the solver or its workspaces are in use. The stored pressure is the
latest projection pressure. `pressure_method=:auto` uses
multigrid-preconditioned CG for the symmetric pressure equation; pass
`:multigrid` to use standalone V-cycles. `diffusion_method=:auto` selects
V-cycle-preconditioned BiCGStab for sufficiently stiff nonsymmetric viscous
systems and standalone multigrid for reaction-dominated steps.
"""
struct IncompressibleNavierStokesSolver{
        T,M<:BlockTreeMesh,D<:StaggeredDiffusionOperator,
        P<:GeometricMultigridSolver,S,A}
    mesh::M
    viscosity::T
    diffusion::D
    pressure_solver::P
    scheme::S
    amr_interface_scheme::A
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    diffusion_method::Symbol
end

const NavierStokesSolver = IncompressibleNavierStokesSolver

function IncompressibleNavierStokesSolver(
        mesh::M; viscosity, boundary=NoSlip(), scheme=LW(),
        amr_interface_scheme=scheme, coarse_size=64,
        pre_sweeps=2, post_sweeps=2, max_levels=32,
        diffusion_method=:auto, pressure_method=:auto,
        T=eltype(mesh.origin)) where {M<:BlockTreeMesh}
    viscosity_value = T(viscosity)
    viscosity_value > zero(T) ||
        throw(ArgumentError("viscosity must be strictly positive"))
    diffusion_method in (:multigrid, :bicgstab, :auto) ||
        throw(ArgumentError(
            "diffusion_method must be :multigrid, :bicgstab, or :auto"))
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity=viscosity_value, boundary, T)
    pressure_solver = GeometricMultigridSolver(
        PoissonOperator(mesh; T);
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        method=pressure_method)
    return IncompressibleNavierStokesSolver{
        T,M,typeof(diffusion),typeof(pressure_solver),
        typeof(scheme),typeof(amr_interface_scheme)}(
        mesh, viscosity_value, diffusion, pressure_solver,
        scheme, amr_interface_scheme, coarse_size, pre_sweeps,
        post_sweeps, max_levels, diffusion_method)
end

struct NavierStokesWorkspace{T,M<:BlockTreeMesh,A,P,E}
    old_velocity::FaceVelocity{T,M}
    convected::FaceVelocity{T,M}
    convection::FaceVelocity{T,M}
    diffusion_work::FaceVelocity{T,M}
    advection::A
    projection::P
    elliptic::E
end

function _zero_velocity(mesh::M, ::Type{T}) where {M<:BlockTreeMesh,T}
    return FaceVelocity(mesh, (x, y, time, axis) -> zero(T); T)
end

function NavierStokesWorkspace(
        solver::IncompressibleNavierStokesSolver{T,M}) where {T,M}
    old_velocity = _zero_velocity(solver.mesh, T)
    convected = _zero_velocity(solver.mesh, T)
    convection = _zero_velocity(solver.mesh, T)
    diffusion_work = _zero_velocity(solver.mesh, T)
    advection = StaggeredVelocityAdvectionWorkspace(old_velocity)
    projection = ProjectionWorkspace(solver.pressure_solver)
    elliptic = StaggeredEllipticSolver(
        StaggeredEllipticOperator(solver.diffusion; diffusion=zero(T));
        coarse_size=solver.coarse_size, pre_sweeps=solver.pre_sweeps,
        post_sweeps=solver.post_sweeps, max_levels=solver.max_levels,
        method=solver.diffusion_method)
    return NavierStokesWorkspace(
        old_velocity, convected, convection, diffusion_work,
        advection, projection, elliptic)
end

function _copy_velocity!(destination::FaceVelocity, source::FaceVelocity)
    destination.mesh === source.mesh ||
        throw(ArgumentError("velocities must share a mesh"))
    copyto!(destination.interior, source.interior)
    copyto!(destination.boundary, source.boundary)
    return destination
end

@inline _body_force(::Nothing, face, time) = zero(typeof(face.area))
@inline _body_force(force, face, time) =
    force(face.center[1], face.center[2], time, face.axis)

"""
    step!(velocity, pressure, solver, dt, workspace;
          time=0, force=nothing, projection_options...)

Advance one split second-order momentum step followed by exactly one pressure
projection. `force`, when present, is sampled as
`force(x, y, time, axis)` at the time midpoint and added to the convected
intermediate velocity. No-permeation is automatic in non-periodic directions.
"""
function step!(velocity::FaceVelocity{T}, pressure::CellField{T},
               solver::IncompressibleNavierStokesSolver{T}, dt,
               workspace::NavierStokesWorkspace{T};
               time=zero(T), force=nothing, kwargs...) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))

    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_no_permeation!(workspace.old_velocity)
    convection!(workspace.convection, workspace.old_velocity,
                workspace.old_velocity, dt, workspace.advection;
                scheme=solver.scheme,
                amr_interface_scheme=solver.amr_interface_scheme)
    midpoint = T(time) + T(dt) / 2
    for (index, face) in pairs(solver.mesh.faces)
        @inbounds workspace.convected.interior[index] =
            workspace.old_velocity.interior[index] -
            T(dt) * workspace.convection.interior[index] +
            T(dt) * _body_force(force, face, midpoint)
    end
    fill!(workspace.convected.boundary, zero(T))

    implicit_midpoint_diffusion!(
        velocity, workspace.convected, workspace.old_velocity,
        solver.diffusion, T(dt), workspace.elliptic,
        workspace.diffusion_work; time=T(time))
    return project!(velocity, pressure, solver.pressure_solver, T(dt),
                    workspace.projection; kwargs...)
end

"""Return the convective CFL bound for a Navier--Stokes step."""
function stable_timestep(velocity::FaceVelocity,
                         ::IncompressibleNavierStokesSolver,
                         workspace::NavierStokesWorkspace;
                         cfl=0.45)
    return stable_timestep(velocity, workspace.advection; cfl)
end

function _independent_solver(
        solver::IncompressibleNavierStokesSolver{T}) where {T}
    return IncompressibleNavierStokesSolver(
        solver.mesh; viscosity=solver.viscosity,
        boundary=solver.diffusion.boundary, scheme=solver.scheme,
        amr_interface_scheme=solver.amr_interface_scheme,
        coarse_size=solver.coarse_size, pre_sweeps=solver.pre_sweeps,
        post_sweeps=solver.post_sweeps, max_levels=solver.max_levels,
        diffusion_method=solver.diffusion_method,
        pressure_method=solver.pressure_solver.method, T)
end

function _independent_step_procedure(
        solver::IncompressibleNavierStokesSolver, force, options)
    independent_solver = nothing
    independent_workspace = nothing
    return function (velocity, pressure, dt; time)
        if isnothing(independent_solver)
            independent_solver = _independent_solver(solver)
            independent_workspace = NavierStokesWorkspace(independent_solver)
        end
        stats = step!(
            velocity, pressure, independent_solver, dt,
            independent_workspace; time, force, options...)
        stats.converged ||
            throw(ErrorException(
                "independent visualization step pressure projection failed to converge"))
        return stats
    end
end

"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              visualization=NoVisualization(), ...)

Advance to `final_time` on the fixed tree. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Returns
the final time, number of steps, and latest projection statistics. A supplied
visualization receives the initial state, every completed step, and the final
state. During initialization it is also given a lazy independent-step
procedure. A visualization may use that procedure to advance copied state
without replacing or subdividing a step on the actual simulation path.
"""
function simulate!(velocity::FaceVelocity{T}, pressure::CellField{T},
                   solver::IncompressibleNavierStokesSolver{T}, final_time,
                   workspace::NavierStokesWorkspace{T};
                   time=zero(T), dt=nothing, cfl=0.45,
                   force=nothing,
                   visualization::AbstractVisualization=NoVisualization(),
                   kwargs...) where {T}
    final_time >= time ||
        throw(ArgumentError("final_time must not precede the initial time"))
    current_time = T(time)
    step_count = 0
    stats = solver.pressure_solver.stats
    selected_step = isnothing(dt) ? T(Inf) : T(dt)
    time_tolerance = T(16) * eps(T) *
                     max(one(T), abs(T(final_time)), abs(current_time))
    independent_step = _independent_step_procedure(solver, force, kwargs)
    initialize_visualization!(
        visualization; time=current_time, step=step_count,
        final_time=T(final_time), velocity, pressure, solver,
        step_procedure=independent_step)
    try
        while T(final_time) - current_time > time_tolerance
            if isnothing(dt)
                bound = stable_timestep(
                    velocity, solver, workspace; cfl)
                selected_step = min(selected_step, bound)
            end
            proposed = isfinite(selected_step) ? selected_step :
                       T(final_time) - current_time
            step_size = min(proposed, T(final_time) - current_time)
            step_size > zero(T) ||
                throw(ArgumentError(
                    "time-step selection produced a non-positive step"))

            stats = step!(velocity, pressure, solver, step_size, workspace;
                          time=current_time, force, kwargs...)
            stats.converged ||
                throw(ErrorException("pressure projection failed to converge"))
            current_time = min(T(final_time), current_time + step_size)
            step_count += 1
            update_visualization!(
                visualization; event=:step, time=current_time,
                step=step_count, final_time=T(final_time),
                velocity, pressure, solver)
        end
        current_time = T(final_time)
        update_visualization!(
            visualization; event=:completed, time=current_time,
            step=step_count, final_time=current_time,
            velocity, pressure, solver)
    catch exception
        try
            update_visualization!(
                visualization; event=:failed, time=current_time,
                step=step_count, final_time=T(final_time),
                velocity, pressure, solver, exception)
        catch
            # Preserve the simulation or visualization error that stopped the run.
        end
        rethrow()
    end
    return (time=current_time, steps=step_count, pressure=stats)
end
