load_started = time_ns()
using TwoPhaseAMR
load_seconds = (time_ns() - load_started) / 1.0e9

construction = @timed begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8),
        periodic = (true, true),
    )
    physics = SinglePhase(density = 1.0, dynamic_viscosity = 0.01)
    FlowSimulation(
        mesh, physics;
        velocity = (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        boundary = FreeSlip(),
    )
end

flow = construction.value
dt = min(stable_timestep(flow; cfl = 0.2), 5.0e-4)
first_step = @timed simulate!(
    flow, flow.time + dt; dt, tolerance = 1.0e-7, max_cycles = 300,
)
steady_step = @timed simulate!(
    flow, flow.time + dt; dt, tolerance = 1.0e-7, max_cycles = 300,
)

println("TwoPhaseAMR latency probe")
println("Julia ", VERSION, ", Julia threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)
println(
    (;
        package_load_seconds = load_seconds,
        construction_seconds = construction.time,
        construction_bytes = construction.bytes,
        first_step_seconds = first_step.time,
        first_step_bytes = first_step.bytes,
        steady_step_seconds = steady_step.time,
        steady_step_bytes = steady_step.bytes,
    )
)
