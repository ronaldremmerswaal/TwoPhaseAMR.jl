"""Return a read-only view of the current state owned by `flow`."""
simulation_state(flow::FlowSimulation) = SimulationState(
    flow.velocity, flow.pressure, flow.solver, flow.time, flow.steps,
    flow.workspace,
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

"""
    add_diagnostic!(flow, name, measure; schedule=EveryStep(), metadata...)

Register a named custom state reduction; `metadata` keywords form its
[`DiagnosticInfo`](@ref) (see [`Diagnostic`](@ref)).
"""
function add_diagnostic!(
        flow::FlowSimulation, name::Symbol, measure;
        schedule = EveryStep(), metadata...,
    )
    return add_diagnostic!(flow, Diagnostic(name, measure; metadata...); schedule)
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

const _TwoVelocityProjectionStats =
    Union{CMDGFMProjectionStats, MimeticProjectionStats}
const _ProjectionSummaryStats =
    Union{LinearSolverStats, _TwoVelocityProjectionStats}

"""
    converged(stats)

Return the convergence decision recorded by a linear or two-velocity
projection solve. This internal accessor does not recompute that decision.
"""
converged(stats::_ProjectionSummaryStats) = stats.converged

"""
    iterations(stats)

Return the recorded iteration count, using `cycles` for `LinearSolverStats`
and `iterations` for the two-velocity projection statistics.
"""
iterations(stats::LinearSolverStats) = stats.cycles
iterations(stats::_TwoVelocityProjectionStats) = stats.iterations

"""
    initial_residual(stats)

Return the initial residual in the solver's native norm: the strong infinity
norm for `LinearSolverStats`, and the algebraic 2-norm for the two-velocity
projection statistics. These norms are not interchangeable across solvers.
"""
initial_residual(stats::LinearSolverStats) = stats.initial_linf
initial_residual(stats::_TwoVelocityProjectionStats) = stats.initial_residual

"""
    final_residual(stats)

Return the final residual in the same solver-specific norm as
[`initial_residual`](@ref).
"""
final_residual(stats::LinearSolverStats) = stats.final_linf
final_residual(stats::_TwoVelocityProjectionStats) = stats.final_residual

_projection_summary_fields(::LinearSolverStats) = (:method, :preconditioner)
_projection_summary_fields(::CMDGFMProjectionStats) = (
    :preconditioner, :mixture_divergence, :mixture_volume_defect,
    :normal_velocity_defect, :momentum_identity_error,
)
_projection_summary_fields(::MimeticProjectionStats) = (
    :preconditioner, :mixture_divergence, :mixture_volume_defect,
    :normal_velocity_defect, :tangential_defect, :momentum_identity_error,
)

function _print_projection_summary(io::IO, stats::_ProjectionSummaryStats)
    println(io, "  pressure projection:")
    _summary_property(io, "converged", converged(stats); indent = "    ")
    _summary_property(io, "iterations", iterations(stats); indent = "    ")
    print(io, "    residual: ")
    _summary_value(io, initial_residual(stats))
    print(io, " -> ")
    _summary_value(io, final_residual(stats))
    println(io)
    for name in _projection_summary_fields(stats)
        _summary_property(
            io, replace(string(name), '_' => ' '), getfield(stats, name);
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
[`simulate!`](@ref); the low-level result without cumulative counts is also
accepted. Residuals use each solver's native norm.
"""
function print_simulation_summary(
        io::IO, flow::FlowSimulation, result::NamedTuple,
    )
    mesh = flow.solver.mesh
    levels = sort!(unique(block.key.level for block in mesh.blocks))
    println(io, "TwoPhaseAMR simulation summary")
    _print_physics_summary(io, flow)
    _summary_property(io, "final time", result.time)
    _summary_property(io, "steps this run", result.steps)
    haskey(result, :total_steps) &&
        _summary_property(io, "total steps", result.total_steps)
    _summary_property(io, "mesh updates this run", result.mesh_updates)
    haskey(result, :total_mesh_updates) &&
        _summary_property(io, "total mesh updates", result.total_mesh_updates)
    unresolved = get(result, :unresolved_curvature_faces, 0)
    unresolved > 0 && _summary_property(
        io, "unresolved curvature faces this run", unresolved,
    )
    total_unresolved = get(result, :total_unresolved_curvature_faces, 0)
    total_unresolved > 0 && _summary_property(
        io, "total unresolved curvature faces", total_unresolved,
    )
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

print_simulation_summary(flow::FlowSimulation, result::NamedTuple) =
    print_simulation_summary(stdout, flow, result)
