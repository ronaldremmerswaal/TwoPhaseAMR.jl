"""Return a read-only view of the current state owned by `flow`."""
simulation_state(flow::FlowSimulation) = SimulationState(
    flow.velocity, flow.pressure, flow.solver, flow.time, flow.steps,
)

"""
    add_diagnostic!(flow, diagnostic; schedule=EveryStep())

Attach a diagnostic to a simulation and immediately record its current state.
Diagnostics are evaluated during `simulate!` even when no visualization is
present.
"""
function add_diagnostic!(
        flow::FlowSimulation, diagnostic::AbstractDiagnostic;
        schedule = EveryStep(),
    )
    _add_diagnostic!(
        flow.diagnostics, diagnostic, simulation_state(flow); schedule,
    )
    return flow
end

"""
    record_diagnostics!(flow; event=:step)

Evaluate diagnostics that are due at the current state. This is useful for
manual time-stepping loops; [`simulate!`](@ref) calls it automatically.
"""
function record_diagnostics!(flow::FlowSimulation; event = :step)
    event in (:step, :completed) || throw(
        ArgumentError("event must be :step or :completed"),
    )
    _record_diagnostics!(flow.diagnostics, simulation_state(flow); event)
    return flow
end

"""Convenience overload for registering a named custom state reduction."""
function add_diagnostic!(
        flow::FlowSimulation, name::Symbol, measure;
        schedule = EveryStep(),
    )
    return add_diagnostic!(flow, Diagnostic(name, measure); schedule)
end

diagnostic_names(flow::FlowSimulation) = diagnostic_names(flow.diagnostics)

"""
    diagnostic_history(flow, name)

Return a detached typed history containing `times`, `steps`, and `values` for
the named simulation diagnostic.
"""
diagnostic_history(flow::FlowSimulation, name::Symbol) =
    diagnostic_history(flow.diagnostics, name)

function _summary_value(io::IO, value)
    value isa Union{AbstractString, Symbol} && return print(io, value)
    return show(IOContext(io, :compact => true, :limit => true), value)
end

function _summary_property(io::IO, label, value; indent = "  ")
    print(io, indent, label, ": ")
    _summary_value(io, value)
    return println(io)
end

_projection_iterations(stats) = hasproperty(stats, :iterations) ?
    stats.iterations : hasproperty(stats, :cycles) ? stats.cycles : nothing
_projection_initial_residual(stats) =
    hasproperty(stats, :initial_residual) ? stats.initial_residual :
    hasproperty(stats, :initial_linf) ? stats.initial_linf : nothing
_projection_final_residual(stats) =
    hasproperty(stats, :final_residual) ? stats.final_residual :
    hasproperty(stats, :final_linf) ? stats.final_linf : nothing

function _print_projection_summary(io::IO, stats)
    println(io, "  pressure projection:")
    hasproperty(stats, :converged) &&
        _summary_property(io, "converged", stats.converged; indent = "    ")
    iterations = _projection_iterations(stats)
    isnothing(iterations) ||
        _summary_property(io, "iterations", iterations; indent = "    ")
    initial = _projection_initial_residual(stats)
    final = _projection_final_residual(stats)
    if !isnothing(initial) || !isnothing(final)
        print(io, "    residual: ")
        isnothing(initial) ? print(io, "unknown") : _summary_value(io, initial)
        print(io, " -> ")
        isnothing(final) ? print(io, "unknown") : _summary_value(io, final)
        println(io)
    end
    for name in (
            :method, :preconditioner, :mixture_divergence,
            :mixture_volume_defect, :normal_velocity_defect,
            :normal_volume_defect, :tangential_defect,
            :momentum_identity_error,
        )
        hasproperty(stats, name) || continue
        _summary_property(
            io, replace(string(name), '_' => ' '), getproperty(stats, name);
            indent = "    ",
        )
    end
    return nothing
end

function _flow_model_name(flow::FlowSimulation)
    isnothing(flow.solver.two_phase) && return "single phase"
    return flow.velocity isa TwoPhaseVelocity ?
        "two phase / two velocity" : "two phase / one velocity"
end

function _print_physics_summary(io::IO, flow::FlowSimulation)
    solver = flow.solver
    _summary_property(io, "model", _flow_model_name(flow))
    _summary_property(io, "velocity field", nameof(typeof(flow.velocity)))
    _summary_property(io, "density", solver.density)
    _summary_property(io, "dynamic viscosity", solver.dynamic_viscosity)
    _summary_property(io, "boundary", nameof(typeof(solver.boundary)))
    _summary_property(io, "advection", nameof(typeof(solver.scheme)))
    _summary_property(io, "embedded geometry", !isnothing(solver.geometry))
    model = solver.two_phase
    isnothing(model) && return nothing
    _summary_property(io, "surface tension", model.sigma)
    _summary_property(io, "surface-tension method", nameof(typeof(model.formulation)))
    _summary_property(io, "gravity", model.gravity)
    if flow.velocity isa TwoPhaseVelocity
        _summary_property(
            io, "interface transport", nameof(typeof(model.interface_transport)),
        )
        _summary_property(
            io, "pressure formulation", nameof(typeof(model.pressure_projection)),
        )
    end
    return nothing
end

function _print_diagnostic_summary(io::IO, flow::FlowSimulation)
    names = diagnostic_names(flow)
    println(io, "  diagnostics (", length(names), "):")
    isempty(names) && return println(io, "    none")
    for name in names
        history = diagnostic_history(flow, name)
        print(io, "    ", name, " [", length(history), "]: ")
        first_value = first(history.values)
        last_value = last(history.values)
        _summary_value(io, first_value)
        if length(history) > 1
            print(io, " -> ")
            _summary_value(io, last_value)
            if applicable(-, last_value, first_value)
                print(io, "  (change ")
                _summary_value(io, last_value - first_value)
                print(io, ")")
            end
        end
        println(io)
    end
    return nothing
end

"""
    print_simulation_summary([io], flow, result)

Print a detailed summary of a completed [`FlowSimulation`](@ref) run. The
report describes the physical/numerical model, final mesh, step and AMR
counts, latest pressure-projection convergence, and the initial/final value of
every simulation-owned diagnostic. `result` is the named tuple returned by
[`simulate!`](@ref).
"""
function print_simulation_summary(
        io::IO, flow::FlowSimulation, result,
    )
    mesh = flow.solver.mesh
    levels = sort!(unique(block.key.level for block in mesh.blocks))
    println(io, "TwoPhaseAMR simulation summary")
    _print_physics_summary(io, flow)
    _summary_property(io, "final time", result.time)
    _summary_property(io, "steps this run", result.steps)
    hasproperty(result, :total_steps) &&
        _summary_property(io, "total steps", result.total_steps)
    _summary_property(io, "mesh updates this run", result.mesh_updates)
    hasproperty(result, :total_mesh_updates) &&
        _summary_property(io, "total mesh updates", result.total_mesh_updates)
    println(io, "  final mesh:")
    _summary_property(io, "leaf blocks", nblocks(mesh); indent = "    ")
    _summary_property(io, "active cells", ncells(mesh); indent = "    ")
    _summary_property(io, "levels", levels; indent = "    ")
    _summary_property(io, "block size", mesh.block_size; indent = "    ")
    _summary_property(
        io, "refinement criteria", length(flow.solver.refinement_indicators);
        indent = "    ",
    )
    _print_projection_summary(io, result.pressure)
    _print_diagnostic_summary(io, flow)
    return nothing
end

print_simulation_summary(flow::FlowSimulation, result) =
    print_simulation_summary(stdout, flow, result)

"""
    register_diagnostic!(registry, flow, diagnostic_name;
                         name=diagnostic_name, component=nothing, options...)

Expose a simulation-owned diagnostic in a live visualization. Evaluation and
history remain owned by `flow`; if the viewer asks for a point outside the
diagnostic's normal schedule, the manager records that point once and the
viewer reuses it. For vector-valued diagnostics, select a scalar `component`.
"""
function register_diagnostic!(
        registry::VisualizationFieldRegistry, flow::FlowSimulation,
        source_name::Symbol; name = source_name, component = nothing,
        kwargs...
    )
    _diagnostic_series(flow.diagnostics, source_name)
    reduction = function (velocity, pressure, time, step, solver)
        state = SimulationState(velocity, pressure, solver, time, step)
        value = _diagnostic_value_at!(flow.diagnostics, source_name, state)
        selected = isnothing(component) ? value : value[component]
        selected isa Real || throw(
            ArgumentError(
                "diagnostic :$source_name is $(typeof(value)); select a " *
                    "scalar component for visualization",
            ),
        )
        return selected
    end
    return register_diagnostic!(
        registry, Symbol(name), reduction; kwargs...,
    )
end
