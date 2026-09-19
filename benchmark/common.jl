using BenchmarkTools
using LinearAlgebra: BLAS
using TwoPhaseAMR

const BENCHMARK_SAMPLES = parse(
    Int, get(ENV, "TWOPHASE_BENCHMARK_SAMPLES", "10"),
)
const BENCHMARK_SECONDS = parse(
    Float64, get(ENV, "TWOPHASE_BENCHMARK_SECONDS", "2"),
)

benchmark_levelset(x, y) = y - 0.5 - 0.04cospi(2x)

function benchmark_mesh(mixed::Bool; periodic = (true, false))
    return BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8),
        periodic = periodic, max_level = mixed ? 1 : 0,
        refine = (key, origin, extent) ->
        mixed && key.level == 0 && key.i == 0 && key.j == 0,
        adaptation = MeshAdaptationOptions(
            update_interval = 4, coarsen_delay = 2,
        ),
    )
end

function benchmark_velocity(x, y, time, axis)
    return axis == 1 ?
        0.1sinpi(2x) * cospi(y) :
        -0.2cospi(2x) * sinpi(y)
end

function single_phase_flow(mixed::Bool; diagnostics = ())
    mesh = benchmark_mesh(mixed)
    physics = SinglePhase(density = 1.0, dynamic_viscosity = 0.01)
    numerics = NavierStokesOptions(
        multigrid = MultigridOptions(coarse_size = 16),
    )
    return FlowSimulation(
        mesh, physics; velocity = benchmark_velocity,
        boundary = FreeSlip(), numerics, diagnostics,
    )
end

function two_phase_flow(
        mixed::Bool, velocity_model::AbstractVelocityModel;
        diagnostics = (),
    )
    mesh = benchmark_mesh(mixed)
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
    physics = TwoPhase(
        interface; density = (1000.0, 1.0),
        dynamic_viscosity = (0.01, 1.0e-4),
        surface_tension = 0.01,
    )
    numerics = NavierStokesOptions(
        multigrid = MultigridOptions(coarse_size = 16),
    )
    return FlowSimulation(
        mesh, physics; velocity_model, velocity = benchmark_velocity,
        boundary = FreeSlip(), numerics, diagnostics,
    )
end

benchmark_dt(flow) = min(stable_timestep(flow; cfl = 0.2), 5.0e-4)

function benchmark_step!(flow, dt)
    return simulate!(
        flow, flow.time + dt; dt, tolerance = 1.0e-7, max_cycles = 300,
    )
end

function print_benchmark_header()
    BLAS.set_num_threads(1)
    println("TwoPhaseAMR benchmark suite")
    println("Julia ", VERSION, ", Julia threads=", Threads.nthreads())
    println("CPU: ", Sys.cpu_info()[1].model)
    println(
        "samples=", BENCHMARK_SAMPLES,
        ", seconds/benchmark=", BENCHMARK_SECONDS,
        ", BLAS threads=1",
    )
    return nothing
end
