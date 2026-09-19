"""
    SimulationState(velocity, pressure, solver, time, step[, workspace])

Read-only view of one completed simulation state. Diagnostics receive this
object instead of a long positional argument list, so adding state metadata
does not require changing every diagnostic callback. `workspace` is available
for reductions, such as a CFL bound, that must reuse assembled simulation
state. It is `nothing` only for callers constructing a state manually.
"""
struct SimulationState{T, V, P, S, I, G, W}
    velocity::V
    pressure::P
    solver::S
    time::T
    step::Int
    interface::I
    geometry::G
    workspace::W
end

function SimulationState(
        velocity, pressure, solver, time, step, workspace = nothing,
    )
    model = hasproperty(solver, :two_phase) ? solver.two_phase : nothing
    interface = isnothing(model) || !hasproperty(model, :interface) ?
        nothing : model.interface
    solver_geometry = hasproperty(solver, :geometry) ? solver.geometry : nothing
    geometry = if !isnothing(solver_geometry)
        solver_geometry
    elseif !isnothing(interface)
        interface.geometry
    else
        nothing
    end
    return SimulationState(
        velocity, pressure, solver, time, Int(step), interface, geometry,
        workspace,
    )
end
