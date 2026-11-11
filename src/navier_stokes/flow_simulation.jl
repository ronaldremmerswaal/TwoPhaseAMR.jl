struct FlowAMRCache{V, P, S, A, W}
    velocity::V
    pressure::P
    system::S
    adaptivity::A
    workspace::W
end
"""
    FlowSimulation(mesh, physics::SinglePhase; options...)
    FlowSimulation(mesh, physics::TwoPhase;
                   velocity_model=OneVelocityModel(), options...)
    FlowSimulation(velocity, pressure, system; workspace=nothing, options...)

High-level owner for an incompressible-flow calculation. It keeps the evolving
`velocity` and `pressure` together with their Navier--Stokes equation system
at `flow.system`, reusable `workspace`, current `time`, cumulative
step/mesh-update counters, optional default time step, and diagnostic histories.

Physics, velocity formulation, numerical methods, and adaptivity are explicit:

```julia
physics = SinglePhase(density=1.0, dynamic_viscosity=1e-3)
flow = FlowSimulation(
    mesh, physics; dt=1e-3,
    velocity=(x, y, time, axis) -> axis == 1 ? sinpi(2y) : 0.0,
    diagnostics=(KineticEnergy(),))
result = simulate!(flow, 1.0)
```

For two-phase physics, `velocity_model=OneVelocityModel()` selects one shared
velocity and `TwoVelocityModel(...)` selects separate liquid/gas velocities.
The formulation is never inferred from the shape of an initializer. A shared
initializer is accepted in either mode; two-velocity mode also accepts a
liquid/gas pair.

`dt` becomes the default for `simulate!` and lets the initial workspace build
the final shifted diffusion hierarchy once. `force`, when supplied, is the
default body force for `step!` and `simulate!`. Velocity may be zero, a constant
two-component tuple, a callable `f(x,y,time,axis)`, or compatible existing
field data. Pressure may be zero/a number, a callable `p(x,y)`, or a
`CellField`. The fully assembled constructor wraps existing fields, system,
and optional workspace without copying.

`flow.interface` and
`flow.diffusion_operator` expose the active phase interface and viscous
operator; the latter is `nothing` for inviscid two-phase flow.
"""
mutable struct FlowSimulation{T, M <: BlockTreeMesh, V, S, W, F, D}
    velocity::V
    pressure::CellField{T, M}
    system::S
    workspace::W
    time::T
    steps::Int
    mesh_updates::Int
    unresolved_curvature_faces::Int
    default_dt::Union{Nothing, T}
    force::F
    diagnostics::D
    amr_cache::Union{Nothing, FlowAMRCache}
end

function Base.getproperty(flow::FlowSimulation, name::Symbol)
    if name === :interface
        model = getfield(getfield(flow, :system), :two_phase)
        return isnothing(model) ? nothing : model.interface
    elseif name === :diffusion_operator
        solver = getfield(flow, :system)
        phase_workspace = getfield(getfield(flow, :workspace), :two_phase)
        return isnothing(phase_workspace) ? solver.diffusion :
            phase_workspace.diffusion
    end
    return getfield(flow, name)
end

function Base.propertynames(flow::FlowSimulation, ::Bool = false)
    names = fieldnames(typeof(flow))
    return (names..., :interface, :diffusion_operator)
end

function _flow_scalar_type(mesh, velocity, pressure, scalar_type)
    if !isnothing(scalar_type)
        scalar_type isa Type || throw(ArgumentError("T must be a type"))
        return scalar_type
    elseif velocity isa FaceVelocity
        return eltype(velocity)
    elseif velocity isa TwoPhaseVelocity
        return eltype(velocity[1])
    elseif pressure isa CellField
        return eltype(pressure)
    end
    return eltype(mesh.origin)
end

function _check_initial_field(
        field::AbstractMeshField, mesh,
        ::Type{T}, name
    ) where {T}
    field.mesh === mesh || throw(
        ArgumentError(
            "$name must use the FlowSimulation mesh"
        )
    )
    eltype(field) === T || throw(
        ArgumentError(
            "$name has element type $(eltype(field)); expected $T"
        )
    )
    return field
end

function _nonnegative_count(value, name)
    value isa Integer && !(value isa Bool) || throw(
        ArgumentError(
            "$name must be an integer"
        )
    )
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return try
        Int(value)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("$name does not fit in Int"))
    end
end

function _constant_velocity(mesh, values, ::Type{T}) where {T}
    length(values) == 2 || throw(
        ArgumentError(
            "a constant velocity must contain exactly two components"
        )
    )
    components = try
        (T(values[1]), T(values[2]))
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("constant velocity components must convert to $T"))
    end
    velocity = FaceVelocity(mesh, T)
    for (index, face) in pairs(mesh.faces)
        @inbounds velocity.interior[index] = components[face.axis]
    end
    for (index, face) in pairs(mesh.boundary_faces)
        @inbounds velocity.boundary[index] = components[face.axis]
    end
    return velocity
end

function _initial_velocity(mesh, initializer, ::Type{T}, time) where {T}
    if isnothing(initializer)
        return FaceVelocity(mesh, T)
    elseif initializer isa FaceVelocity
        return _check_initial_field(initializer, mesh, T, "velocity")
    elseif initializer isa TwoPhaseVelocity
        _validate_two_phase_velocity(initializer, mesh, T)
        return initializer
    elseif initializer isa Tuple && all(value -> value isa Number, initializer)
        return _constant_velocity(mesh, initializer, T)
    elseif initializer isa Tuple && length(initializer) == 2 && all(
            value -> applicable(
                value, mesh.origin[1], mesh.origin[2], time, 1,
            ),
            initializer,
        )
        return (
            FaceVelocity(mesh, initializer[1]; time, T),
            FaceVelocity(mesh, initializer[2]; time, T),
        )
    elseif applicable(initializer, mesh.origin[1], mesh.origin[2], time, 1)
        return FaceVelocity(mesh, initializer; time, T)
    end
    throw(
        ArgumentError(
            "velocity must be nothing, a two-component tuple, " *
                "a callable f(x, y, time, axis), a FaceVelocity, or a " *
                "liquid/gas pair of FaceVelocity fields or callables"
        )
    )
end

function _initial_pressure(mesh, initializer, ::Type{T}) where {T}
    if isnothing(initializer)
        return CellField(mesh, T)
    elseif initializer isa CellField
        return _check_initial_field(initializer, mesh, T, "pressure")
    elseif initializer isa Number
        return CellField(mesh, T; init = initializer)
    elseif applicable(initializer, mesh.origin[1], mesh.origin[2])
        return CellField(mesh, initializer; T)
    end
    throw(
        ArgumentError(
            "pressure must be nothing, a number, a callable p(x, y), " *
                "or a CellField"
        )
    )
end

function FlowSimulation(
        velocity::Union{FaceVelocity, TwoPhaseVelocity},
        pressure::CellField{T, M},
        solver::NavierStokesSystem{T, M};
        workspace = nothing, time = zero(T), dt = nothing, steps = 0,
        mesh_updates = 0, diagnostics = (), force = nothing,
    ) where {T, M <: BlockTreeMesh}
    velocity_mesh = velocity isa FaceVelocity ? velocity.mesh :
        velocity[1].mesh
    velocity_mesh === pressure.mesh === solver.mesh || throw(
        ArgumentError(
            "velocity, pressure, and solver must share one mesh"
        )
    )
    default_dt = isnothing(dt) ? nothing : _positive_finite_step(T, dt)
    built_workspace = isnothing(workspace) ?
        NavierStokesWorkspace(solver; dt = default_dt) : workspace
    built_workspace isa NavierStokesWorkspace{T, M} || throw(
        ArgumentError(
            "workspace must be a compatible NavierStokesWorkspace"
        )
    )
    built_workspace.old_velocity.mesh === solver.mesh || throw(
        ArgumentError(
            "workspace and solver must share one mesh"
        )
    )
    initial_time = _finite_time(T, time)
    initial_steps = _nonnegative_count(steps, "steps")
    initial_mesh_updates = _nonnegative_count(mesh_updates, "mesh_updates")
    velocity isa TwoPhaseVelocity &&
        _validate_two_phase_velocity(velocity, solver.mesh, T)
    if velocity isa TwoPhaseVelocity
        isnothing(solver.two_phase) && throw(
            ArgumentError(
                "a two-velocity FlowSimulation requires an interface",
            )
        )
        _mark_inactive_phase_velocities!(velocity, solver.two_phase.interface)
    end
    collection = DiagnosticCollection(T)
    flow = FlowSimulation{
        T, M, typeof(velocity), typeof(solver), typeof(built_workspace),
        typeof(force), typeof(collection),
    }(
        velocity, pressure, solver, built_workspace, initial_time,
        initial_steps, initial_mesh_updates, 0, default_dt, force, collection, nothing,
    )
    _configure_diagnostics!(
        collection, diagnostics, SimulationState(
            velocity, pressure, solver, initial_time, initial_steps,
            built_workspace,
        ),
    )
    return flow
end

function _flow_simulation(
        mesh::M, physics; numerics, adaptivity,
        velocity = nothing, pressure = nothing,
        time = 0, dt = nothing, T = nothing,
        diagnostics = (), force = nothing, system_options...
    ) where {M <: BlockTreeMesh}
    scalar_type = _flow_scalar_type(mesh, velocity, pressure, T)
    _check_real_type(
        scalar_type, "FlowSimulation requires T=Float32 or T=Float64"; exact = true,
    )
    initial_time = _finite_time(scalar_type, time)
    initial_velocity = _initial_velocity(
        mesh, velocity, scalar_type, initial_time
    )
    initial_pressure = _initial_pressure(mesh, pressure, scalar_type)
    solver = NavierStokesSystem(
        mesh, physics; options = numerics, adaptivity,
        T = scalar_type, system_options...
    )
    return FlowSimulation(
        initial_velocity, initial_pressure, solver;
        time = initial_time, dt, diagnostics, force,
    )
end


function _one_velocity_initializer(initializer)
    if initializer isa TwoPhaseVelocity ||
            initializer isa Tuple && length(initializer) == 2 && all(
            value -> applicable(value, 0.0, 0.0, 0.0, 1), initializer,
        )
        throw(
            ArgumentError(
                "OneVelocityModel requires one shared velocity initializer",
            ),
        )
    end
    return initializer
end

function _two_velocity_initializer(initializer)
    if initializer isa TwoPhaseVelocity ||
            initializer isa Tuple && length(initializer) == 2 && all(
            value -> applicable(value, 0.0, 0.0, 0.0, 1), initializer,
        )
        return initializer
    elseif isnothing(initializer)
        zero_velocity = (x, y, time, axis) -> 0.0
        return (zero_velocity, zero_velocity)
    elseif initializer isa FaceVelocity
        # Retain the supplied field as the liquid velocity so refinement
        # criteria attached to it continue to observe the evolving state.
        return (initializer, copy(initializer))
    elseif initializer isa Tuple && length(initializer) == 2 &&
            all(value -> value isa Number, initializer)
        shared = (x, y, time, axis) -> initializer[axis]
        return (shared, shared)
    elseif applicable(initializer, 0.0, 0.0, 0.0, 1)
        return (initializer, initializer)
    end
    throw(
        ArgumentError(
            "TwoVelocityModel velocity must be a shared velocity initializer " *
                "or a (liquid, gas) pair",
        ),
    )
end

"""
    FlowSimulation(mesh, physics::SinglePhase; numerics=NavierStokesOptions(),
                   adaptivity=Adaptivity(), options...)

Construct a single-phase incompressible-flow simulation with physics and
numerical configuration passed as separate typed objects.
"""
function FlowSimulation(
        mesh::BlockTreeMesh, physics::SinglePhase;
        velocity = nothing, pressure = nothing,
        time = 0, dt = nothing, T = nothing,
        geometry = nothing, boundary = NoSlip(),
        numerics = NavierStokesOptions(), adaptivity = Adaptivity(),
        diagnostics = (), force = nothing,
    )
    numerics isa NavierStokesOptions || throw(
        ArgumentError("numerics must be NavierStokesOptions"),
    )
    adaptivity isa Adaptivity || throw(
        ArgumentError("adaptivity must be Adaptivity"),
    )
    return _flow_simulation(
        mesh, physics; numerics, adaptivity,
        velocity, pressure, time, dt, T, geometry, boundary, diagnostics, force,
    )
end

"""
    FlowSimulation(mesh, physics::TwoPhase;
                   velocity_model=OneVelocityModel(),
                   numerics=NavierStokesOptions(),
                   adaptivity=Adaptivity(), options...)

Construct a one- or two-velocity two-phase simulation. The velocity model is
explicit and no longer inferred from the shape of `velocity`. In
`TwoVelocityModel`, a shared initializer is copied to both phases.
"""
function FlowSimulation(
        mesh::BlockTreeMesh, physics::TwoPhase;
        velocity_model::AbstractVelocityModel = OneVelocityModel(),
        velocity = nothing, pressure = nothing,
        time = 0, dt = nothing, T = nothing,
        geometry = physics.interface.geometry, boundary = NoSlip(),
        numerics = NavierStokesOptions(), adaptivity = Adaptivity(),
        diagnostics = (), force = nothing,
    )
    physics.interface.mesh === mesh || throw(
        ArgumentError("two-phase physics and simulation must share one mesh"),
    )
    numerics isa NavierStokesOptions || throw(
        ArgumentError("numerics must be NavierStokesOptions"),
    )
    adaptivity isa Adaptivity || throw(
        ArgumentError("adaptivity must be Adaptivity"),
    )
    initial_velocity = velocity_model isa TwoVelocityModel ?
        _two_velocity_initializer(velocity) :
        _one_velocity_initializer(velocity)
    pressure_projection = velocity_model isa TwoVelocityModel ?
        velocity_model.pressure_projection : CMDGFMProjection()
    return _flow_simulation(
        mesh, physics; numerics, adaptivity,
        two_velocity_projection = pressure_projection,
        velocity = initial_velocity, pressure, time, dt, T,
        geometry, boundary, diagnostics, force,
    )
end

function _simulation_flow(
        velocity, pressure::CellField{T, M}, solver, workspace,
        time, steps, diagnostics, force,
    ) where {T, M <: BlockTreeMesh}
    return FlowSimulation{
        T, M, typeof(velocity), typeof(solver), typeof(workspace),
        typeof(force), typeof(diagnostics),
    }(
        velocity, pressure, solver, workspace, time, steps, 0, 0, nothing,
        force, diagnostics, nothing,
    )
end

function _flow_amr_workspace!(flow::FlowSimulation)
    solver = flow.system
    isempty(solver.adaptivity.criteria) && return nothing
    cache = flow.amr_cache
    if isnothing(cache) || cache.velocity !== flow.velocity ||
            cache.pressure !== flow.pressure || cache.system !== solver ||
            cache.adaptivity !== solver.adaptivity
        amr = _navier_stokes_amr_workspace(flow.velocity, flow.pressure, solver)
        flow.amr_cache = FlowAMRCache(
            flow.velocity, flow.pressure, solver, solver.adaptivity, amr,
        )
        return amr
    end
    return cache.workspace
end

function _same_simulation_step(flow::FlowSimulation)
    selection = flow.workspace.simulation_step
    return selection.velocity === flow.velocity &&
        selection.pressure === flow.pressure && selection.solver === flow.system &&
        selection.time == flow.time &&
        selection.topology_generation == flow.system.mesh.topology_cache.generation
end

function _save_simulation_step!(flow::FlowSimulation, proposal, last_step, defect_step)
    selection = flow.workspace.simulation_step
    selection.topology_generation = flow.system.mesh.topology_cache.generation
    selection.time = flow.time
    selection.step = proposal
    selection.last_dt = last_step
    selection.defect_dt = defect_step
    selection.velocity = flow.velocity
    selection.pressure = flow.pressure
    selection.solver = flow.system
    return nothing
end

"""
    step!(flow::FlowSimulation, dt; force=flow.force, observer=NoObserver(),
          tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance one physical step, update the flow time and counters, adapt the mesh
when configured, and record diagnostics before emitting a `:step` event to
`observer`. Returns the pressure-projection statistics. A body force passed
to this call overrides the force stored in the flow for this step.

The field-level `step!(velocity, pressure, system, dt, workspace; ...)`
performs only the physical solver stages; it does not perform this bookkeeping
or adaptive mesh update.
"""
function step!(
        flow::FlowSimulation{T}, dt;
        force = flow.force, observer::AbstractObserver = NoObserver(),
        tolerance = sqrt(eps(T)), max_cycles = nothing,
        _proposal = dt, _reference_dt = nothing, _current_time = nothing,
        _next_time = nothing, _final_time = nothing, _independent_step = nothing,
        _step_started = time_ns(), _stats_ref = nothing,
        _defer_selection_save = false,
    ) where {T}
    velocity, pressure, solver, workspace =
        flow.velocity, flow.pressure, flow.system, flow.workspace
    time_step = _positive_finite_step(T, dt)
    current_time = isnothing(_current_time) ? _finite_time(T, flow.time) : _current_time
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    selection = workspace.simulation_step
    defect_step = _same_simulation_step(flow) && selection.last_dt > zero(T) ?
        selection.defect_dt : zero(T)
    reference_step = isnothing(_reference_dt) ?
        max(_proposal, defect_step, time_step) : _reference_dt
    amr_workspace = _flow_amr_workspace!(flow)
    isnothing(amr_workspace) || _supports_dynamic_topology(observer) ||
        throw(
        ArgumentError(
            "exact keyframes are not supported while dynamic mesh adaptation is enabled"
        )
    )
    _prepare_independent_step!(_independent_step, solver, workspace, reference_step)
    stats = step!(
        velocity, pressure, solver, dt, workspace;
        time = current_time, force, tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit, _reference_dt = reference_step,
    )
    isnothing(_stats_ref) || (_stats_ref[] = stats)
    stats.converged || throw(
        ErrorException(
            _solver_failure_message(
                "pressure projection", stats, dimensionless_tolerance;
                details = "dt=$(dt), time=$(current_time), max_cycles=$(cycle_limit)",
            )
        )
    )

    next_time = isnothing(_next_time) ? current_time + time_step : _next_time
    flow.time = next_time
    flow.steps += 1
    flow.unresolved_curvature_faces += workspace.unresolved_curvature_faces[]
    solver.mesh.adaptation.completed_steps += 1
    topology_changed = false
    defect_step = time_step
    if !isnothing(amr_workspace)
        amr_stats = adapt!(
            amr_workspace, solver.adaptivity.criteria;
            step = solver.mesh.adaptation.completed_steps, time = next_time,
        )
        if amr_stats.changed
            topology_changed = true
            flow.mesh_updates += 1
            _rebuild_topology_dependent_state!(solver, workspace; dt)
            # The new transfer defect belongs to the unshortened proposal.
            defect_step = T(_proposal)
            if velocity isa TwoPhaseVelocity
                _project_regridded_two_velocity!(
                    velocity, solver, workspace, _proposal, next_time;
                    tolerance = dimensionless_tolerance,
                    max_iterations = cycle_limit,
                )
            end
        end
    end
    if _defer_selection_save
        # The simulation loop persists identity and proposal once at run end.
        selection.last_dt = time_step
        selection.defect_dt = defect_step
    else
        _save_simulation_step!(flow, T(Inf), time_step, defect_step)
    end
    state = SimulationState(velocity, pressure, solver, next_time, flow.steps, workspace)
    _record_diagnostics!(flow.diagnostics, state; event = :step)
    final_time = isnothing(_final_time) ? next_time : _final_time
    _observe_safely!(
        observer, :step, state,
        StepInfo(;
            dt, projection = stats, topology_changed, final_time,
            wall_time = (time_ns() - _step_started) / 1.0e9,
        ),
    )
    return stats
end

"""
    simulate!(flow::FlowSimulation, final_time; dt=flow.default_dt, options...)

Advance `flow` from its stored time to the absolute `final_time`, reusing its
workspace. The low-level simulation result is returned with `total_steps`,
`total_mesh_updates`, and `total_unresolved_curvature_faces` added; the last is
the cumulative number of unresolved surface-tension faces over all calls.
Calling this method again continues the same run. For automatic stepping, the
unmodified step proposal is retained across consecutive calls. To avoid a
tiny final step, a proposal leaving less than `1e-6` of itself is stretched
to the target. A target between physical steps can change the
trajectory. If automatic step selection initially has an infinite
stability bound, the step is capped at one 128th of the requested interval.
The initial time is owned by `flow` and therefore cannot be passed as a
keyword.
"""
function simulate!(
        flow::FlowSimulation{T}, final_time;
        dt = flow.default_dt, force = flow.force, options...
    ) where {T}
    haskey(options, :time) && throw(
        ArgumentError(
            "FlowSimulation owns the initial time; set flow.time before resuming"
        )
    )
    haskey(options, :diagnostics) && throw(
        ArgumentError("FlowSimulation owns its diagnostic collection"),
    )
    haskey(options, :step_offset) && throw(
        ArgumentError("FlowSimulation owns its cumulative step count"),
    )
    result = _simulate!(
        flow, _finite_time(T, final_time, "final_time"); dt, force, options...
    )
    return (;
        result..., total_steps = flow.steps,
        total_mesh_updates = flow.mesh_updates,
        total_unresolved_curvature_faces = flow.unresolved_curvature_faces,
    )
end

"""Return the current convective CFL bound for a `FlowSimulation`."""
stable_timestep(flow::FlowSimulation; cfl = 0.45) =
    stable_timestep(flow.velocity, flow.system, flow.workspace; cfl)

"""
    project!(flow::FlowSimulation; dt=1, tolerance=sqrt(eps(T)),
             max_cycles=nothing)

Project the current one- or two-velocity state onto the configured discrete
incompressibility constraint. This is primarily useful for preparing an
analytical initial condition before the first time step. The projection uses
zero scratch pressure; its pressure is a geometric impulse, so the physical
`flow.pressure` is left unchanged. For a one-velocity flow, the returned
`LinearSolverStats.compatible` reports whether the component RHS means meet
the projection tolerance; an incompatible projection warns once per workspace.
"""
function project!(
        flow::FlowSimulation{T}; dt = one(T),
        tolerance = sqrt(eps(T)), max_cycles = nothing,
    ) where {T}
    time_step = _positive_finite_step(T, dt)
    selected_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(flow.system, max_cycles)
    _invalidate_simulation_step!(flow.workspace)
    correction_pressure = flow.workspace.projection.inherited_divergence
    _zero_interior!(correction_pressure)
    if flow.velocity isa TwoPhaseVelocity
        model = flow.system.two_phase
        isnothing(model) && throw(
            ArgumentError("two-velocity projection requires two-phase physics"),
        )
        phase = flow.workspace.two_phase
        state = _two_velocity_flow_workspace!(phase, model)
        return _project_two_velocity!(
            flow.velocity, correction_pressure, flow.system, flow.workspace, state,
            time_step;
            tolerance = selected_tolerance,
            max_iterations = cycle_limit,
        )
    end
    pressure_solver = _one_velocity_pressure_solver!(flow.system, flow.workspace; refresh = true)
    if isnothing(flow.system.geometry)
        return project!(
            flow.velocity, correction_pressure, pressure_solver,
            time_step, flow.workspace.projection;
            velocity_boundary = flow.system.boundary,
            boundary_time = flow.time,
            tolerance = selected_tolerance, max_cycles = cycle_limit,
        )
    end
    return project!(
        flow.velocity, correction_pressure, pressure_solver,
        time_step, flow.workspace.projection, flow.system.geometry;
        velocity_boundary = flow.system.boundary,
        boundary_time = flow.time,
        tolerance = selected_tolerance, max_cycles = cycle_limit,
    )
end

"""
    register_default_fields!(registry, flow::FlowSimulation;
                             interface_name=:volume_fraction,
                             velocity_name=:velocity,
                             pressure_name=:pressure)

Register the standard fields of a flow for visualization. The method selects
the appropriate one- or two-velocity representation automatically and adds
the phase interface when present.
"""
function register_default_fields!(
        registry::VisualizationFieldRegistry, flow::FlowSimulation;
        interface_name = :volume_fraction,
        velocity_name = :velocity, pressure_name = :pressure,
    )
    interface = flow.interface
    if isnothing(interface)
        register_default_fields!(
            registry, velocity_name, flow.velocity,
        )
    else
        register_default_fields!(
            registry, interface_name, interface,
        )
        if flow.velocity isa FaceVelocity
            register_default_fields!(registry, velocity_name, flow.velocity)
        else
            register_default_fields!(
                registry, velocity_name, flow.velocity, interface,
            )
        end
    end
    register_default_fields!(registry, pressure_name, flow.pressure)
    return registry
end

function Base.show(io::IO, flow::FlowSimulation)
    return print(
        io, "FlowSimulation(time=", flow.time,
        ", steps=", flow.steps,
        ", dt=", flow.default_dt,
        ", blocks=", nblocks(flow.system.mesh),
        ", cells=", ncells(flow.system.mesh), ")"
    )
end
