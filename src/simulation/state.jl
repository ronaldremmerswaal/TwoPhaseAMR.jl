"""
    SimulationState(velocity, pressure, solver, time, step)

Read-only view of one completed simulation state. Diagnostics receive this
object instead of a long positional argument list, so adding state metadata
does not require changing every diagnostic callback.
"""
struct SimulationState{T, V, P, S, I, G}
    velocity::V
    pressure::P
    solver::S
    time::T
    step::Int
    interface::I
    geometry::G
end

function SimulationState(velocity, pressure, solver, time, step)
    model = solver.two_phase
    interface = isnothing(model) ? nothing : model.interface
    geometry = if !isnothing(solver.geometry)
        solver.geometry
    elseif !isnothing(interface)
        interface.geometry
    else
        nothing
    end
    return SimulationState(
        velocity, pressure, solver, time, Int(step), interface, geometry,
    )
end
