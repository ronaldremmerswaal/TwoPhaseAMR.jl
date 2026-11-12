# Which sub-step of the first two-velocity step creates the tangential jump of
# jump_reproducer.jl? Re-executes the body of
# `step!(::TwoPhaseVelocity, …)` (src/two_phase/two_velocity/stepping.jl:277-363) with
# the same internal calls and measures max|u_g − u_l| (x-faces carrying both phases)
# after every sub-step. FlowSimulation does not project the initial velocity
# (src/navier_stokes/flow_simulation.jl:230-238 only masks absent phases).
#   julia --project=. jump_trace.jl  → results/jump_trace.txt
using TwoPhaseAMR
using Printf
const P = TwoPhaseAMR

const OUT = IOBuffer()
report(args...) = (println(args...); println(OUT, args...))

"Largest |u_g − u_l| over x-faces with both phases, its location and the distance to the interface in h."
function jump(velocity, mesh, a, h)
    ul, ug = velocity
    best, where = 0.0, (NaN, NaN, NaN, NaN)
    for (i, face) in pairs(mesh.faces)
        face.axis == 1 || continue
        l, g = ul.interior[i], ug.interior[i]
        (isfinite(l) && isfinite(g)) || continue
        d = abs(g - l)
        if d > best
            x, y = face.center
            best, where = d, (x, y, (y - a * cospi(2x)) / h, l)
        end
    end
    return best, where
end

function trace(n; δ = 0.05, a = 0.01, cfl = 0.4, dt_factor = 1.0)
    h = 1 / n
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (0.0, -0.5), extent = (1.0, 1.0),
        periodic = (true, false),
    )
    interface = InterfaceField(mesh, (x, y) -> y - a * cospi(2x); lipschitz = 1)
    physics = TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (1.0e-3, 1.0e-3))
    u0 = (x, y, t, axis) -> axis == 1 ? 0.5 * tanh(y / δ) : 0.0
    flow = FlowSimulation(mesh, physics; velocity_model = TwoVelocityModel(), velocity = u0, boundary = FreeSlip())
    dt = dt_factor * stable_timestep(flow; cfl)
    velocity, pressure, solver, workspace = flow.velocity, flow.pressure, flow.system, flow.workspace
    show(label, v) = (
        j = jump(v, mesh, a, h); report(
            @sprintf(
                "  %-46s max|u_g−u_l| = %.2e  at (x, y) = (%.3f, %+.4f), (y−η)/h = %+.2f, u_l = %+.4f",
                label, j[1], j[2]...
            )
        )
    )
    report(@sprintf("δ/h = %.1f, dt = %.3e", δ * n, dt))
    show("(a) after initialisation", velocity)
    # body of step! (stepping.jl:285-363)
    geometry_prepared = workspace.transport_geometry_prepared
    workspace.transport_geometry_prepared = false
    model = solver.two_phase
    phase = workspace.two_phase
    time_step, reference_step, current_time, tol, cycles = P._prepare_step_controls!(
        velocity, solver, workspace, dt, dt, 0.0, sqrt(eps()), nothing,
    )
    state = P._two_velocity_flow_workspace!(phase, model)
    P._copy_velocity!(state.old_velocity, velocity)
    P._enforce_phase_velocity_boundary!(state.old_velocity, solver, current_time, phase.inactive_failures)
    show("(b) old velocity after boundary enforcement", state.old_velocity)
    P._transport_substep!(state.old_velocity, solver, workspace, time_step, current_time, model; geometry_prepared)
    show("(c) after transport (state.convected)", state.convected)
    midpoint = current_time + time_step / 2
    P._enforce_phase_velocity_boundary!(state.convected, solver, midpoint, phase.inactive_failures)
    show("(d) after boundary enforcement", state.convected)
    P._diffusion_substep!(velocity, solver, workspace, time_step, current_time, model; tolerance = tol, max_cycles = cycles)
    show("(e) after viscous step", velocity)
    P._enforce_phase_velocity_boundary!(velocity, solver, current_time + time_step, phase.inactive_failures)
    operator = P._two_velocity_pressure_operator!(solver, workspace, state, model)
    P._projection_weighted_gradient!(phase.acceleration, pressure, operator, workspace.projection.elliptic, model.interface.geometry)
    P._add_common_phase_acceleration!(velocity, phase.acceleration, -time_step)
    show("(f) after old pressure gradient", velocity)
    P._project_two_velocity!(
        velocity, pressure, solver, workspace, state, time_step;
        pressure_prepared = true, tolerance = tol, _reference_dt = reference_step, max_iterations = cycles
    )
    show("(g) after CMDGFM projection (end of step 1)", velocity)
    return nothing
end

for n in (32, 64, 128)
    trace(n)
end
report("\nDependence on the time step (δ/h = 3.2):")
for f in (0.25, 0.0625)
    trace(64; dt_factor = f)
end
write(joinpath(@__DIR__, "results", "jump_trace.txt"), String(take!(OUT)))
