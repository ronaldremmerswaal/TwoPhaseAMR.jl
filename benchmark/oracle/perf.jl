# Run in a fresh process; includes package/example loading before first-step timings.
include("cases.jl")

function first_step(run)
    GC.gc()
    measurement = @timed run()
    return Dict(
        "seconds" => measurement.time, "bytes" => measurement.bytes,
        "steps" => measurement.value.flow.steps
    )
end

function taylor_64()
    mesh = BlockTreeMesh(base_mesh = (64, 64), block_size = (8, 8), extent = (2pi, 2pi), max_level = 0)
    velocity(x, y, time, axis) = exp(-0.04time) * (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    return FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.02);
        velocity, boundary = FreeSlip()
    )
end

function cavity_64()
    mesh = BlockTreeMesh(
        base_mesh = (64, 64), block_size = (8, 8),
        periodic = (false, false), max_level = 0
    )
    lid = NoSlip((x, y, time, axis) -> axis == 1 && y > 0.5 ? 1.0 : 0.0)
    return FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
        boundary = lid, dt = 0.001
    )
end

function warm_steps(flow; dt = 0.001, n = 10, warmup = 1)
    for _ in 1:warmup
        simulate!(flow, flow.time + dt; dt)
    end
    GC.gc()
    start_steps = flow.steps
    measurement = @timed for _ in 1:n
        simulate!(flow, flow.time + dt; dt)
    end
    steps = flow.steps - start_steps
    steps == n || error("expected $n steps, got $steps")
    return Dict(
        "seconds_per_step" => measurement.time / steps,
        "bytes_per_step" => measurement.bytes / steps, "steps" => steps
    )
end

function main()
    results = Dict{String, Any}(
        "cpu" => strip(split(first(filter(line -> startswith(line, "model name"), readlines("/proc/cpuinfo"))), ":"; limit = 2)[2]),
        "julia_version" => string(VERSION), "threads" => Threads.nthreads()
    )
    results["first_step_taylor_green"] = first_step(() -> TaylorGreen.run_example(max_level = 1, final_time = 0.001, dt = 0.001))
    results["first_step_stokes3"] = first_step(() -> Stokes.run_example(max_level = 1, final_time = 0.001, dt = 0.001))
    results["warm_taylor_green_64"] = warm_steps(taylor_64())
    results["warm_lid_cavity_64"] = warm_steps(cavity_64(); warmup = 5)
    for (name, model) in (
            ("warm_capillary_one", OneVelocityModel()),
            ("warm_capillary_cmdgfm", TwoVelocityModel(pressure_projection = CMDGFMProjection())),
            ("warm_capillary_mimetic", TwoVelocityModel(pressure_projection = MimeticProjection())),
        )
        flow = Capillary.run_example(max_level = 1, final_time = 0.001, dt = 0.001, velocity_model = model).flow
        results[name] = warm_steps(flow)
    end
    open(joinpath(@__DIR__, "perf_baseline.toml"), "w") do io
        TOML.print(io, results; sorted = true)
    end
    return TOML.print(stdout, results; sorted = true)
end
main()
