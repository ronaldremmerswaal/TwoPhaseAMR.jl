using BenchmarkTools
using TwoPhaseAMR

function benchmark_case(mesh)
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.2 : -0.1)
    workspace = AdvectionWorkspace(scalar)
    dt = stable_timestep(velocity, workspace; cfl=0.5)

    setup = @benchmark BlockTreeMesh(
        block_size=$(mesh.block_size), max_level=$(mesh.max_level),
        refine=(key, origin, extent) -> key.level < $(mesh.max_level),
    ) samples=20 evals=1
    step = @benchmark advect!($scalar, $velocity, $dt, $workspace)
    halo = @benchmark fill_halo!($scalar)
    return (; setup, step, halo)
end

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)

function print_estimate(label, trial)
    estimate = median(trial)
    println("  ", rpad(label * ":", 7), estimate.time, " ns, ",
            estimate.memory, " bytes, ", estimate.allocs, " allocations")
end

for (name, mesh) in (
    "single block" => BlockTreeMesh(block_size=(32, 32)),
    "16 uniform blocks" => BlockTreeMesh(
        block_size=(16, 16), max_level=2,
        refine=(key, origin, extent) -> true,
    ),
)
    result = benchmark_case(mesh)
    println("\n", name, " (", ncells(mesh), " cells)")
    print_estimate("setup", result.setup)
    print_estimate("step", result.step)
    print_estimate("halo", result.halo)
end
