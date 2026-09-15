"""
    IncompressibleNavierStokesSolver(mesh; viscosity,
        boundary=NoSlip(), scheme=LW(),
        amr_interface_scheme=default_amr_interface_scheme(scheme),
        pressure_krylov=Krylov.CgWorkspace,
        diffusion_krylov=Krylov.BicgstabWorkspace,
        pressure_krylov_options=(;), diffusion_krylov_options=(;),
        pressure_preconditioner=:geometric,
        diffusion_preconditioner=:geometric,
        refinement_indicators=(), adaptive_fields=())

Single-phase incompressible Navier--Stokes discretization. One time
step is split into:

1. a conservative staggered convection update (Lax--Wendroff by default and
   optionally using a three-point limited AMR-interface flux), applied as
   sequential coordinate sweeps whose order alternates between time steps,
2. a Crank--Nicolson/implicit-midpoint staggered viscous solve, and
3. one compatible pressure projection.

The tree may contain static or dynamically adapted local refinement. Optional
`refinement_indicators` are evaluated after completed steps according to the
mesh's `MeshAdaptationOptions`; `velocity`, `pressure`, indicator fields, and
the additional `adaptive_fields` are transferred in place before
topology-dependent operators are rebuilt. The stored pressure is the latest
projection pressure. Krylov.jl CG and BiCGSTAB remain the pressure and
diffusion defaults respectively; their preconditioners are independent and
default to this package's geometric multigrid V-cycle.
"""
mutable struct IncompressibleNavierStokesSolver{
        T, M <: BlockTreeMesh, D <: StaggeredDiffusionOperator,
        S, A, PK, DK, PKO, DKO, PP, DP, PO, QO, RI, AF,
    }
    mesh::M
    viscosity::T
    diffusion::D
    # The compatible face metric is diagonal on a uniform mesh and sparse on
    # a mixed-level mesh. Dynamic adaptation may therefore change the
    # concrete EllipticSolver type while preserving its scalar type.
    pressure_solver::EllipticSolver{T}
    scheme::S
    amr_interface_scheme::A
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    pressure_krylov::PK
    diffusion_krylov::DK
    pressure_krylov_options::PKO
    diffusion_krylov_options::DKO
    pressure_preconditioner::PP
    diffusion_preconditioner::DP
    pressure_preconditioner_options::PO
    diffusion_preconditioner_options::QO
    refinement_indicators::RI
    adaptive_fields::AF
    two_phase::Union{Nothing, TwoPhaseModel{T, M}}
end

"""Concise alias for [`IncompressibleNavierStokesSolver`](@ref)."""
const NavierStokesSolver = IncompressibleNavierStokesSolver

function IncompressibleNavierStokesSolver(
        mesh::M; viscosity, boundary = NoSlip(), scheme = LW(),
        amr_interface_scheme = default_amr_interface_scheme(scheme), coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2, max_levels = 32,
        pressure_krylov = Krylov.CgWorkspace,
        diffusion_krylov = Krylov.BicgstabWorkspace,
        pressure_krylov_options = (;),
        diffusion_krylov_options = (;),
        pressure_preconditioner = :geometric,
        diffusion_preconditioner = :geometric,
        pressure_preconditioner_options = (;),
        diffusion_preconditioner_options = (;),
        refinement_indicators = (), adaptive_fields = (),
        interface = nothing, density = (1.0, 1.0), surface_tension = 0.0,
        surface_tension_formulation = CSF(), gravity = (0.0, 0.0), phase_scalars = (),
        T = eltype(mesh.origin)
    ) where {M <: BlockTreeMesh}
    model = if isnothing(interface)
        nothing
    else
        interface isa InterfaceField{T, M} && interface.mesh === mesh ||
            throw(ArgumentError("interface must share solver mesh and scalar type"))
        TwoPhaseModel(
            interface; density, viscosity, surface_tension, gravity,
            formulation = surface_tension_formulation, scalars = phase_scalars
        )
    end
    viscosity_value = isnothing(model) ? T(viscosity) :
        max(eps(T), maximum(model.viscosity ./ model.density))
    isfinite(viscosity_value) && viscosity_value > zero(T) ||
        throw(ArgumentError("viscosity must be strictly positive"))
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = viscosity_value, boundary, T
    )
    pressure_solver = EllipticSolver(
        isnothing(model) ? PoissonOperator(mesh; T) :
            PoissonOperator(interface; density = model.density, formulation = model.formulation); krylov = pressure_krylov,
        krylov_options = pressure_krylov_options,
        preconditioner = pressure_preconditioner,
        preconditioner_options = pressure_preconditioner_options,
        coarse_size, pre_sweeps, post_sweeps, max_levels
    )
    indicators = _as_configuration_tuple(refinement_indicators)
    all(indicator -> indicator isa AbstractRefinementIndicator, indicators) ||
        throw(
        ArgumentError(
            "refinement_indicators must contain refinement indicators"
        )
    )
    all(indicator -> _indicator_field(indicator).mesh === mesh, indicators) ||
        throw(ArgumentError("refinement indicator fields must use solver mesh"))
    fields = _as_configuration_tuple(adaptive_fields)
    all(field -> field isa AbstractMeshField, fields) ||
        throw(
        ArgumentError(
            "adaptive_fields must contain cell or face fields"
        )
    )
    all(field -> field.mesh === mesh, fields) ||
        throw(ArgumentError("adaptive fields must use solver mesh"))
    return IncompressibleNavierStokesSolver{
        T, M, typeof(diffusion),
        typeof(scheme), typeof(amr_interface_scheme),
        typeof(pressure_krylov), typeof(diffusion_krylov),
        typeof(pressure_krylov_options), typeof(diffusion_krylov_options),
        typeof(pressure_preconditioner), typeof(diffusion_preconditioner),
        typeof(pressure_preconditioner_options),
        typeof(diffusion_preconditioner_options),
        typeof(indicators), typeof(fields),
    }(
        mesh, viscosity_value, diffusion, pressure_solver,
        scheme, amr_interface_scheme, coarse_size, pre_sweeps,
        post_sweeps, max_levels, pressure_krylov, diffusion_krylov,
        pressure_krylov_options, diffusion_krylov_options,
        pressure_preconditioner, diffusion_preconditioner,
        pressure_preconditioner_options, diffusion_preconditioner_options,
        indicators, fields, model
    )
end

@inline _as_configuration_tuple(::Nothing) = ()
@inline _as_configuration_tuple(value::Tuple) = value
@inline _as_configuration_tuple(value::AbstractVector) = Tuple(value)
@inline _as_configuration_tuple(value) = (value,)

mutable struct NavierStokesWorkspace{T, M <: BlockTreeMesh}
    old_velocity::FaceVelocity{T, M}
    convected::FaceVelocity{T, M}
    diffusion_work::FaceVelocity{T, M}
    # Uniform and mixed-level meshes use different schedule/metric concrete
    # types. These partial types let a single workspace survive a regrid.
    advection::StaggeredVelocityAdvectionWorkspace{T, M}
    projection::ProjectionWorkspace{T, M}
    elliptic::StaggeredEllipticSolver{T}
    two_phase::Union{Nothing, TwoPhaseFlowWorkspace{T, M}}
end

function _replace_workspace!(
        destination::NavierStokesWorkspace,
        source::NavierStokesWorkspace
    )
    destination.old_velocity = source.old_velocity
    destination.convected = source.convected
    destination.diffusion_work = source.diffusion_work
    destination.advection = source.advection
    destination.projection = source.projection
    destination.elliptic = source.elliptic
    destination.two_phase = source.two_phase
    return destination
end

function _navier_stokes_configuration(
        solver::IncompressibleNavierStokesSolver;
        refinement_indicators = solver.refinement_indicators,
        adaptive_fields = solver.adaptive_fields
    )
    return (;
        viscosity = isnothing(solver.two_phase) ? solver.viscosity : solver.two_phase.viscosity,
        interface = isnothing(solver.two_phase) ? nothing : solver.two_phase.interface,
        density = isnothing(solver.two_phase) ? (1.0, 1.0) : solver.two_phase.density,
        surface_tension = isnothing(solver.two_phase) ? 0.0 : solver.two_phase.sigma,
        surface_tension_formulation = isnothing(solver.two_phase) ? CSF() : solver.two_phase.formulation,
        gravity = isnothing(solver.two_phase) ? (0.0, 0.0) : solver.two_phase.gravity,
        phase_scalars = isnothing(solver.two_phase) ? () : solver.two_phase.scalars,
        boundary = solver.diffusion.boundary,
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme,
        coarse_size = solver.coarse_size,
        pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps,
        max_levels = solver.max_levels,
        pressure_krylov = solver.pressure_krylov,
        diffusion_krylov = solver.diffusion_krylov,
        pressure_krylov_options = solver.pressure_krylov_options,
        diffusion_krylov_options = solver.diffusion_krylov_options,
        pressure_preconditioner = solver.pressure_preconditioner,
        diffusion_preconditioner = solver.diffusion_preconditioner,
        pressure_preconditioner_options =
            solver.pressure_preconditioner_options,
        diffusion_preconditioner_options =
            solver.diffusion_preconditioner_options,
        refinement_indicators,
        adaptive_fields,
        T = typeof(solver.viscosity),
    )
end

function _rebuild_topology_dependent_state!(
        solver::IncompressibleNavierStokesSolver,
        workspace::NavierStokesWorkspace; dt = nothing
    )
    configuration = _navier_stokes_configuration(solver)
    refreshed = IncompressibleNavierStokesSolver(
        solver.mesh; configuration...
    )
    solver.diffusion = refreshed.diffusion
    solver.pressure_solver = refreshed.pressure_solver
    return _replace_workspace!(
        workspace, NavierStokesWorkspace(solver; dt)
    )
end

function _navier_stokes_amr_workspace(
        velocity, pressure, solver::IncompressibleNavierStokesSolver
    )
    workspace = AMRWorkspace(solver.mesh, velocity, pressure)
    if !isnothing(solver.two_phase)
        register_adaptive_field!(workspace, solver.two_phase.interface)
        for scalar in solver.two_phase.scalars
            register_adaptive_field!(workspace, scalar)
        end
    end
    for field in solver.adaptive_fields
        register_adaptive_field!(workspace, field)
    end
    for indicator in solver.refinement_indicators
        _register_indicator_field!(workspace, indicator)
    end
    return workspace
end

function _zero_velocity(mesh::M, ::Type{T}) where {M <: BlockTreeMesh, T}
    return FaceVelocity(mesh, T)
end

function _positive_finite_step(::Type{T}, value, label = "dt") where {T}
    converted = try
        T(value)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("$label must convert to $T"))
    end
    isfinite(converted) && converted > zero(T) || throw(
        ArgumentError(
            "$label must be finite and strictly positive"
        )
    )
    return converted
end

function _finite_time(::Type{T}, value, label = "time") where {T}
    converted = try
        T(value)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("$label must convert to $T"))
    end
    isfinite(converted) || throw(ArgumentError("$label must be finite"))
    return converted
end

function _navier_stokes_elliptic_solver(
        solver::IncompressibleNavierStokesSolver{T}, diffusion
    ) where {T}
    return StaggeredEllipticSolver(
        StaggeredEllipticOperator(solver.diffusion; diffusion = T(diffusion));
        krylov = solver.diffusion_krylov,
        krylov_options = solver.diffusion_krylov_options,
        preconditioner = solver.diffusion_preconditioner,
        preconditioner_options = solver.diffusion_preconditioner_options,
        coarse_size = solver.coarse_size, pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps, max_levels = solver.max_levels
    )
end

"""
    NavierStokesWorkspace(solver; dt=nothing)

Allocate all reusable arrays, staggered schedules, Krylov workspaces, and
preconditioners required by `step!` and `simulate!`. When the fixed time step
is known, pass `dt` to build the shifted diffusion hierarchy at its final
coefficient and avoid rebuilding it on the first step. Construct the workspace
once for advanced low-level use; `FlowSimulation` owns one automatically.
"""
function NavierStokesWorkspace(
        solver::IncompressibleNavierStokesSolver{T, M};
        dt = nothing
    ) where {T, M}
    old_velocity = _zero_velocity(solver.mesh, T)
    convected = _zero_velocity(solver.mesh, T)
    diffusion_work = _zero_velocity(solver.mesh, T)
    advection = StaggeredVelocityAdvectionWorkspace(old_velocity)
    projection = ProjectionWorkspace(solver.pressure_solver)
    if isnothing(dt)
        diffusion = zero(T)
    else
        time_step = _positive_finite_step(T, dt, "workspace dt")
        diffusion = time_step * solver.viscosity / 2
    end
    elliptic = _navier_stokes_elliptic_solver(solver, diffusion)
    return NavierStokesWorkspace(
        old_velocity, convected, diffusion_work,
        advection, projection, elliptic,
        isnothing(solver.two_phase) ? nothing : TwoPhaseFlowWorkspace(
            solver.two_phase; boundary = solver.diffusion.boundary
        )
    )
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

function _transport_substep!(
        velocity, solver, workspace, dt, ::Nothing
    )
    _copy_velocity!(workspace.convected, workspace.old_velocity)
    advect!(
        workspace.convected, workspace.old_velocity, dt,
        workspace.advection;
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme
    )
    return workspace.convected
end

function _diffusion_substep!(
        velocity, solver, workspace, dt, time, ::Nothing
    )
    return implicit_midpoint_diffusion!(
        velocity, workspace.convected, workspace.old_velocity,
        solver.diffusion, dt, workspace.elliptic,
        workspace.diffusion_work; time
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
          time=0, force=nothing, projection_options...)

Advance one split second-order momentum step followed by exactly one pressure
projection. Both single- and two-phase configurations execute the same
sequence: transport, implicit diffusion, explicit body/interface forces, and
projection. `force`, when present, is sampled as
`force(x, y, time, axis)` at the time midpoint. The transporting flux is
frozen at the old-time velocity while each coordinate sweep updates both
transported velocity components. No-permeation is automatic in non-periodic
directions.
"""
function step!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing, kwargs...
    ) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    time_step = _positive_finite_step(T, dt)
    current_time = _finite_time(T, time)
    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_no_permeation!(workspace.old_velocity)
    model = solver.two_phase
    _transport_substep!(
        velocity, solver, workspace, time_step, model
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model
    )
    _refresh_pressure_substep!(solver, workspace, model)
    _phase_force_substep!(velocity, solver, workspace, time_step, model)

    midpoint = current_time + time_step / 2
    for (index, face) in pairs(solver.mesh.faces)
        @inbounds velocity.interior[index] +=
            time_step * _body_force(force, face, midpoint)
    end
    enforce_no_permeation!(velocity)
    return project!(
        velocity, pressure, solver.pressure_solver, time_step,
        workspace.projection; kwargs...
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

function _independent_solver(
        solver::IncompressibleNavierStokesSolver{T}
    ) where {T}
    configuration = _navier_stokes_configuration(
        solver; refinement_indicators = (), adaptive_fields = ()
    )
    return IncompressibleNavierStokesSolver(
        solver.mesh; configuration...
    )
end

function _independent_step_procedure(
        solver::IncompressibleNavierStokesSolver, force, options,
        source_workspace::NavierStokesWorkspace
    )
    independent_solver = nothing
    independent_workspace = nothing
    return function (velocity, pressure, dt; time)
        if isnothing(independent_solver)
            independent_solver = _independent_solver(solver)
            independent_workspace = NavierStokesWorkspace(independent_solver)
        end
        # The copied checkpoint state precedes the main step that just
        # completed. Reuse that step's sweep order, which is now the opposite
        # of the main workspace's next-order flag.
        independent_workspace.advection.sweep_forward[] =
            !source_workspace.advection.sweep_forward[]
        stats = step!(
            velocity, pressure, independent_solver, dt,
            independent_workspace; time, force, options...
        )
        stats.converged ||
            throw(
            ErrorException(
                "independent visualization step pressure projection failed to converge"
            )
        )
        return stats
    end
end

_visualization_step_procedure(
    ::NoVisualization, solver, force, options, workspace
) = nothing

_visualization_step_procedure(
    ::AbstractVisualization, solver, force, options, workspace
) =
    _independent_step_procedure(solver, force, options, workspace)

"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              visualization=NoVisualization(), ...)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Returns
the final time, number of steps, latest projection statistics, and the number
of topology changes. When the solver has refinement indicators, they are
evaluated after a successful physical step; field transfer does not add a
time-integration or pressure-correction step. A supplied
visualization receives the initial state, every completed step, and the final
state. During initialization it is also given a lazy independent-step
procedure. A visualization may use that procedure to advance copied state
without replacing or subdividing a step on the actual simulation path.
"""
function simulate!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, final_time,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), dt = nothing, cfl = 0.45,
        force = nothing,
        visualization::AbstractVisualization = NoVisualization(),
        kwargs...
    ) where {T}
    target_time = _finite_time(T, final_time, "final_time")
    current_time = _finite_time(T, time)
    target_time >= current_time ||
        throw(ArgumentError("final_time must not precede the initial time"))
    step_count = 0
    mesh_updates = 0
    stats = solver.pressure_solver.stats
    selected_step = isnothing(dt) ? T(Inf) : _positive_finite_step(T, dt)
    time_tolerance = T(16) * eps(T) *
        max(one(T), abs(target_time), abs(current_time))
    independent_step = _visualization_step_procedure(
        visualization, solver, force, kwargs, workspace
    )
    amr_workspace = isempty(solver.refinement_indicators) ? nothing :
        _navier_stokes_amr_workspace(
            velocity, pressure, solver
        )
    isnothing(amr_workspace) || _supports_dynamic_topology(visualization) ||
        throw(
        ArgumentError(
            "exact visualization checkpoints are not supported while dynamic mesh adaptation is enabled"
        )
    )
    initialize_visualization!(
        visualization; time = current_time, step = step_count,
        final_time = target_time, velocity, pressure, solver,
        step_procedure = independent_step
    )
    try
        while target_time - current_time > time_tolerance
            if isnothing(dt)
                bound = stable_timestep(
                    velocity, solver, workspace; cfl
                )
                selected_step = min(selected_step, bound)
            end
            proposed = isfinite(selected_step) ? selected_step :
                target_time - current_time
            step_size = min(proposed, target_time - current_time)
            step_size > zero(T) ||
                throw(
                ArgumentError(
                    "time-step selection produced a non-positive step"
                )
            )

            stats = step!(
                velocity, pressure, solver, step_size, workspace;
                time = current_time, force, kwargs...
            )
            stats.converged ||
                throw(ErrorException("pressure projection failed to converge"))
            current_time = min(target_time, current_time + step_size)
            step_count += 1
            solver.mesh.adaptation.completed_steps += 1
            if !isnothing(amr_workspace)
                amr_stats = adapt!(
                    amr_workspace, solver.refinement_indicators;
                    step = solver.mesh.adaptation.completed_steps
                )
                if amr_stats.changed
                    mesh_updates += 1
                    # Seed the replacement viscous hierarchy with the current
                    # shift. Fixed-step runs therefore do not build it twice
                    # after every topology change.
                    _rebuild_topology_dependent_state!(
                        solver, workspace; dt = step_size
                    )
                end
            end
            update_visualization!(
                visualization; event = :step, time = current_time,
                step = step_count, final_time = target_time,
                velocity, pressure, solver
            )
        end
        current_time = target_time
        update_visualization!(
            visualization; event = :completed, time = current_time,
            step = step_count, final_time = current_time,
            velocity, pressure, solver
        )
    catch exception
        try
            update_visualization!(
                visualization; event = :failed, time = current_time,
                step = step_count, final_time = target_time,
                velocity, pressure, solver, exception
            )
        catch
            # Preserve the simulation or visualization error that stopped the run.
        end
        rethrow()
    end
    return (
        time = current_time, steps = step_count, pressure = stats,
        mesh_updates,
    )
end
