"""Return a read-only view of the current state owned by `flow`."""
simulation_state(flow::FlowSimulation) = SimulationState(
    flow.velocity, flow.pressure, flow.system, flow.time, flow.steps,
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

const _ProjectionSummaryStats =
    Union{LinearSolverStats, CMDGFMProjectionStats}

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
iterations(stats::CMDGFMProjectionStats) = stats.iterations

"""
    initial_residual(stats)

Return the initial residual in the solver's native norm: the strong infinity
norm for `LinearSolverStats`, and the algebraic 2-norm for the two-velocity
projection statistics. These norms are not interchangeable across solvers.
"""
initial_residual(stats::LinearSolverStats) = stats.initial_linf
initial_residual(stats::CMDGFMProjectionStats) = stats.initial_residual

"""
    final_residual(stats)

Return the final residual in the same solver-specific norm as
[`initial_residual`](@ref).
"""
final_residual(stats::LinearSolverStats) = stats.final_linf
final_residual(stats::CMDGFMProjectionStats) = stats.final_residual

_projection_summary_fields(::LinearSolverStats) = (
    :compatible, :method, :preconditioner,
    :removed_rhs_component_anchors, :removed_rhs_component_means,
)
_projection_summary_fields(::CMDGFMProjectionStats) = (
    :preconditioner, :mixture_divergence, :mixture_volume_defect,
    :normal_velocity_defect, :momentum_identity_error,
)
function _print_projection_summary(
        io::IO, run::ProjectionRunStats, stats::_ProjectionSummaryStats,
    )
    println(io, "  pressure projection:")
    _summary_property(io, "solves this run", run.solves; indent = "    ")
    _summary_property(io, "iterations this run", run.iterations; indent = "    ")
    run.without_iterations > 0 && _summary_property(
        io, "solves that did no work", run.without_iterations; indent = "    ",
    )
    _summary_property(io, "converged", run.converged; indent = "    ")
    if _has_projection_reduction(run)
        _summary_property(
            io, "last working solve iterations", run.last_iterations;
            indent = "    ",
        )
        print(io, "    last working solve residual: ")
        _summary_value(io, run.last_initial_residual)
        print(io, " -> ")
        _summary_value(io, run.last_final_residual)
        println(io)
    else
        _summary_property(
            io, "last working solve residual", "none"; indent = "    ",
        )
    end
    for name in _projection_summary_fields(stats)
        _summary_property(
            io, replace(string(name), '_' => ' '), getfield(stats, name);
            indent = "    ",
        )
    end
    return nothing
end

function _flow_model_name(flow::FlowSimulation)
    isnothing(flow.system.two_phase) && return "single phase"
    return flow.velocity isa TwoPhaseVelocity ?
        "two phase / two velocity" : "two phase / one velocity"
end

function _print_physics_summary(io::IO, flow::FlowSimulation)
    solver = flow.system
    _summary_property(io, "model", _flow_model_name(flow))
    _summary_property(io, "velocity field", nameof(typeof(flow.velocity)))
    model = solver.two_phase
    density = isnothing(model) ? _single_phase_density(solver) : model.density
    viscosity = isnothing(model) ? eltype(flow.pressure)(solver.physics.dynamic_viscosity) : model.dynamic_viscosity
    _summary_property(io, "density", density)
    _summary_property(io, "dynamic viscosity", viscosity)
    _summary_property(io, "boundary", nameof(typeof(solver.boundary)))
    _summary_property(io, "advection", nameof(typeof(solver.options.advection.scheme)))
    _summary_property(io, "embedded geometry", !isnothing(solver.geometry))
    isnothing(model) && return nothing
    _summary_property(io, "surface tension", model.sigma)
    _summary_property(io, "surface-tension method", nameof(typeof(model.formulation)))
    _summary_property(io, "gravity", model.gravity)
    if flow.velocity isa TwoPhaseVelocity
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
counts, the run's accumulated pressure-projection work (iterations, the number
of solves that needed none, and the residual of the last solve that did work)
together with the latest solve's convergence flags, and the initial/final
value of every simulation-owned diagnostic. `result` is the named tuple
returned by [`simulate!`](@ref); the low-level result without cumulative
counts is also accepted. Residuals use each solver's native norm.
"""
function print_simulation_summary(
        io::IO, flow::FlowSimulation, result::NamedTuple,
    )
    mesh = flow.system.mesh
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
        io, "refinement criteria", length(flow.system.adaptivity.criteria);
        indent = "    ",
    )
    _print_projection_summary(io, result.projection, result.pressure)
    _print_diagnostic_summary(io, flow)
    return nothing
end

print_simulation_summary(flow::FlowSimulation, result::NamedTuple) =
    print_simulation_summary(stdout, flow, result)
