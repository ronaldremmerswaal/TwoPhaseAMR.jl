using BenchmarkTools
using TwoPhaseAMR

function workflow_mesh(block_size = 16)
    return BlockTreeMesh(
        block_size = (block_size, block_size), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    )
end

function workflow_flow(mesh; dt = nothing)
    velocity = (x, y, time, axis) -> axis == 1 ?
        sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    return FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.01, velocity,
        boundary = FreeSlip(), coarse_size = 32, dt
    )
end

function first_flow_step(mesh, workspace_dt)
    flow = workflow_flow(mesh; dt = workspace_dt)
    return simulate!(
        flow, 1.0e-4; dt = 1.0e-4, tolerance = 1.0e-8, max_cycles = 80
    )
end

function report(label, trial)
    estimate = median(trial)
    return println(
        rpad(label, 31),
        " time = ", BenchmarkTools.prettytime(estimate.time),
        ", memory = ", BenchmarkTools.prettymemory(estimate.memory),
        ", allocs = ", estimate.allocs
    )
end

mesh = workflow_mesh()
flow = workflow_flow(mesh; dt = 1.0e-4)
dt = 1.0e-4
step!(
    flow.velocity, flow.pressure, flow.solver, dt, flow.workspace;
    tolerance = 1.0e-8, max_cycles = 80
)

setup = @benchmark workflow_flow($mesh; dt = $dt) samples = 30 evals = 1 seconds = 10
unseeded_first_step = @benchmark first_flow_step(
    $mesh, nothing
) samples = 20 evals = 1 seconds = 10
seeded_first_step = @benchmark first_flow_step(
    $mesh, $dt
) samples = 20 evals = 1 seconds = 10
low_level_step = @benchmark step!(
    $flow.velocity, $flow.pressure, $flow.solver, $dt, $flow.workspace;
    time = $flow.time, tolerance = 1.0e-8, max_cycles = 80
)
high_level_step = @benchmark simulate!(
    $flow, $flow.time + $dt; dt = $dt, tolerance = 1.0e-8, max_cycles = 80
)

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)
println("Single-phase workflow (", ncells(mesh), " cells)")
report("FlowSimulation setup", setup)
report("setup + first step, unseeded", unseeded_first_step)
report("setup + first step, seeded", seeded_first_step)
report("low-level step!", low_level_step)
report("one-step simulate!", high_level_step)
