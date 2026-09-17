"""
    FlowSimulation(mesh; density, dynamic_viscosity,
                   velocity=nothing, pressure=nothing,
                   time=0, dt=nothing, T=nothing, solver_options...)
    FlowSimulation(velocity, pressure, solver;
                   workspace=nothing, time=0, dt=nothing,
                   steps=0, mesh_updates=0)

High-level owner for a single- or two-phase incompressible-flow calculation. It keeps
the evolving `velocity` and `pressure` together with their `solver`, reusable
`workspace`, current `time`, cumulative `steps`, and cumulative
`mesh_updates`, and optional default time step.

For standard use, pass a mesh, density, dynamic viscosity, and optionally
initial conditions:

```julia
flow = FlowSimulation(
    mesh; density=1.0, dynamic_viscosity=1e-3,
    dt=1e-3,
    velocity=(x, y, time, axis) -> axis == 1 ? sinpi(2y) : 0.0)
result = simulate!(flow, 1.0)
```

`dt` is optional; when supplied, it becomes the default for `simulate!` and
lets the initial workspace build the final shifted diffusion hierarchy once.
`velocity` may be `nothing` (zero), a two-number tuple, a callable
`f(x, y, time, axis)`, or an existing `FaceVelocity`. `pressure` may be
`nothing`/a number, a callable `p(x, y)`, or an existing `CellField`. Remaining
keywords configure `NavierStokesSolver`. The second constructor preserves the
fully configurable low-level path and optionally accepts an existing
`NavierStokesWorkspace`.

For single-phase flow, `density` and `dynamic_viscosity` are scalars. Pass an
`InterfaceField` through `interface`, together with `(liquid, gas)` tuples for
the two material properties, to select the one-velocity two-phase model.
`flow.interface` and
`flow.diffusion_operator` expose the active interface and viscous operator;
the latter is `nothing` for inviscid two-phase flow.
"""
mutable struct FlowSimulation{T, M <: BlockTreeMesh, S, W}
    velocity::FaceVelocity{T, M}
    pressure::CellField{T, M}
    solver::S
    workspace::W
    time::T
    steps::Int
    mesh_updates::Int
    default_dt::Union{Nothing, T}
end

function Base.getproperty(flow::FlowSimulation, name::Symbol)
    if name === :interface
        model = getfield(getfield(flow, :solver), :two_phase)
        return isnothing(model) ? nothing : model.interface
    elseif name === :diffusion_operator
        solver = getfield(flow, :solver)
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

"""Explicit alias for [`FlowSimulation`](@ref)."""
const NavierStokesSimulation = FlowSimulation

function _flow_scalar_type(mesh, velocity, pressure, scalar_type)
    if !isnothing(scalar_type)
        scalar_type isa Type || throw(ArgumentError("T must be a type"))
        return scalar_type
    elseif velocity isa FaceVelocity
        return eltype(velocity)
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
    elseif initializer isa Tuple
        return _constant_velocity(mesh, initializer, T)
    elseif applicable(initializer, mesh.origin[1], mesh.origin[2], time, 1)
        return FaceVelocity(mesh, initializer; time, T)
    end
    throw(
        ArgumentError(
            "velocity must be nothing, a two-component tuple, " *
                "a callable f(x, y, time, axis), or a FaceVelocity"
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
        velocity::FaceVelocity{T, M}, pressure::CellField{T, M},
        solver::IncompressibleNavierStokesSolver{T, M};
        workspace = nothing, time = zero(T), dt = nothing, steps = 0,
        mesh_updates = 0
    ) where {T, M <: BlockTreeMesh}
    velocity.mesh === pressure.mesh === solver.mesh || throw(
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
    return FlowSimulation{T, M, typeof(solver), typeof(built_workspace)}(
        velocity, pressure, solver, built_workspace, initial_time,
        initial_steps, initial_mesh_updates, default_dt
    )
end

function FlowSimulation(
        mesh::M; density, dynamic_viscosity,
        velocity = nothing, pressure = nothing,
        time = 0, dt = nothing, T = nothing,
        solver_options...
    ) where {M <: BlockTreeMesh}
    scalar_type = _flow_scalar_type(mesh, velocity, pressure, T)
    scalar_type in (Float32, Float64) || throw(
        ArgumentError(
            "FlowSimulation requires T=Float32 or T=Float64"
        )
    )
    initial_time = _finite_time(scalar_type, time)
    initial_velocity = _initial_velocity(
        mesh, velocity, scalar_type, initial_time
    )
    initial_pressure = _initial_pressure(mesh, pressure, scalar_type)
    solver = NavierStokesSolver(
        mesh; density, dynamic_viscosity,
        T = scalar_type, solver_options...
    )
    return FlowSimulation(
        initial_velocity, initial_pressure, solver; time = initial_time, dt
    )
end

"""
    simulate!(flow::FlowSimulation, final_time; dt=flow.default_dt, options...)

Advance `flow` from its stored time to the absolute `final_time`, reusing its
workspace. The low-level simulation result is returned with `total_steps` and
`total_mesh_updates` added. Calling this method again continues the same run.
The initial time is owned by `flow` and therefore cannot be passed as a
keyword.
"""
function simulate!(
        flow::FlowSimulation{T}, final_time;
        dt = flow.default_dt, options...
    ) where {T}
    haskey(options, :time) && throw(
        ArgumentError(
            "FlowSimulation owns the initial time; set flow.time before resuming"
        )
    )
    result = simulate!(
        flow.velocity, flow.pressure, flow.solver,
        _finite_time(T, final_time, "final_time"),
        flow.workspace; time = flow.time, dt, options...
    )
    flow.time = result.time
    flow.steps += result.steps
    flow.mesh_updates += result.mesh_updates
    return (;
        result..., total_steps = flow.steps,
        total_mesh_updates = flow.mesh_updates,
    )
end

"""Return the current convective CFL bound for a `FlowSimulation`."""
stable_timestep(flow::FlowSimulation; cfl = 0.45) =
    stable_timestep(flow.velocity, flow.solver, flow.workspace; cfl)

function Base.show(io::IO, flow::FlowSimulation)
    return print(
        io, "FlowSimulation(time=", flow.time,
        ", steps=", flow.steps,
        ", dt=", flow.default_dt,
        ", blocks=", nblocks(flow.solver.mesh),
        ", cells=", ncells(flow.solver.mesh), ")"
    )
end
