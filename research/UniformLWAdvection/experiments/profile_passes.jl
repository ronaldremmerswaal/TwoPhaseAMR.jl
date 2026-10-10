"""
Standalone diagnostic profiler for the installed UniformLWAdvection step.

Run with the main project's resolved dependencies, for example:

    julia --project=. --threads=4 research/UniformLWAdvection/experiments/profile_passes.jl \
        --backend=threads --flux=stored --nx=512 --ny=512 --steps=20 \
        --repeats=5 --output=/tmp/toy-passes.toml

This compares complete Strang trajectories with the same trajectory timed at
each of its three recompute sweeps, or at each of its six stored flux/gather
passes. State reset, compilation warmup, mass/error checks, and TOML output are
outside timed regions. Each pass synchronizes before its clock is stopped.
Separate allocation trials run the same trajectories after timing.

Per-pass clocks and wrapper calls perturb cache state and launch latency; the
sum of pass times is not an alternative complete-step timing. The full-step
path also includes validation and time bookkeeping. This experiment is for
attribution only, not the Stage 2 acceptance timing or a fair backend ranking.
`@allocated` is diagnostic and may not include every worker-task allocation.
Only a fixed constant carrier is profiled; there is no velocity refresh cost.
Affinity fields describe the process and calling-thread masks inherited from
the launcher; they do not prove the placement of every worker thread.
"""

using LinearAlgebra: BLAS
using Statistics: median
using TOML
using SHA

BLAS.set_num_threads(1)
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))
using UniformLWAdvection
const Toy = UniformLWAdvection

const ALLOWED = Set((
    "backend", "flux", "nx", "ny", "steps", "repeats", "warmups",
    "tile-x", "tile-y", "threshold", "batches-per-worker", "dt",
    "output",
))

function parse_options(args)
    options = Dict{String, String}()
    for argument in args
        startswith(argument, "--") && occursin('=', argument) ||
            error("expected --name=value, got $argument")
        name, value = split(argument[3:end], '='; limit = 2)
        name in ALLOWED || error("unknown option --$name")
        haskey(options, name) && error("duplicate option --$name")
        options[name] = value
    end
    return options
end

option(options, name, fallback) = get(options, name, fallback)
integer_option(options, name, fallback) = parse(Int, option(options, name, string(fallback)))

function positive_integer(options, name, fallback)
    value = integer_option(options, name, fallback)
    value > 0 || error("--$name must be positive")
    return value
end

function reset!(state, initial)
    copyto!(state.q, initial)
    state.t = 0.0
    return nothing
end

function allowed_cpus(path)
    isfile(path) || return "unavailable"
    for line in eachline(path)
        startswith(line, "Cpus_allowed_list:") || continue
        return strip(split(line, ':'; limit = 2)[2])
    end
    return "unavailable"
end

# These two operations use the same run_tiles! calls and pass IDs as the core
# stored-flux sweep. They are intentionally outside the package source.
@inline function stored_flux!(q, velocity, ratio, direction, workspace, pass)
    Toy.run_tiles!(
        tile -> Toy.flux_tile!(workspace.flux, q, velocity, ratio,
                               direction, workspace.carrier, tile),
        workspace, 2pass - 1,
    )
    return nothing
end

@inline function stored_gather!(out, q, ratio, direction, workspace, pass)
    Toy.run_tiles!(
        tile -> Toy.gather_tile!(out, q, workspace.flux, ratio,
                                 direction, tile),
        workspace, 2pass,
    )
    return nothing
end

function observe!(operation, slots::Vector{Float64}, index, mode)
    if mode === :clock
        start = time_ns()
        operation()
        slots[index] += (time_ns() - start) * 1.0e-9
    else
        operation()
    end
    return nothing
end

function observe_allocation!(operation, slots::Vector{Int}, index)
    slots[index] += @allocated operation()
    return nothing
end

function recompute_passes!(state, workspace, dt, steps, slots, mode)
    nx, ny = size(state.q)
    xratio, yratio = dt * nx * 0.5, dt * ny
    for _ in 1:steps
        if mode === :allocation
            observe_allocation!(slots, 1) do
                Toy.sweep!(workspace.first, state.q, state.u, xratio,
                           Val(1), workspace, 1)
            end
            observe_allocation!(slots, 2) do
                Toy.sweep!(workspace.second, workspace.first, state.v, yratio,
                           Val(2), workspace, 2)
            end
            observe_allocation!(slots, 3) do
                Toy.sweep!(state.q, workspace.second, state.u, xratio,
                           Val(1), workspace, 3)
            end
        else
            observe!(slots, 1, mode) do
                Toy.sweep!(workspace.first, state.q, state.u, xratio,
                           Val(1), workspace, 1)
            end
            observe!(slots, 2, mode) do
                Toy.sweep!(workspace.second, workspace.first, state.v, yratio,
                           Val(2), workspace, 2)
            end
            observe!(slots, 3, mode) do
                Toy.sweep!(state.q, workspace.second, state.u, xratio,
                           Val(1), workspace, 3)
            end
        end
        state.t += dt
    end
    return state
end

function stored_passes!(state, workspace, dt, steps, slots, mode)
    nx, ny = size(state.q)
    xratio, yratio = dt * nx * 0.5, dt * ny
    for _ in 1:steps
        if mode === :allocation
            observe_allocation!(slots, 1) do
                stored_flux!(state.q, state.u, xratio, Val(1), workspace, 1)
            end
            observe_allocation!(slots, 2) do
                stored_gather!(workspace.first, state.q, xratio,
                               Val(1), workspace, 1)
            end
            observe_allocation!(slots, 3) do
                stored_flux!(workspace.first, state.v, yratio,
                             Val(2), workspace, 2)
            end
            observe_allocation!(slots, 4) do
                stored_gather!(workspace.second, workspace.first, yratio,
                               Val(2), workspace, 2)
            end
            observe_allocation!(slots, 5) do
                stored_flux!(workspace.second, state.u, xratio,
                             Val(1), workspace, 3)
            end
            observe_allocation!(slots, 6) do
                stored_gather!(state.q, workspace.second, xratio,
                               Val(1), workspace, 3)
            end
        else
            observe!(slots, 1, mode) do
                stored_flux!(state.q, state.u, xratio, Val(1), workspace, 1)
            end
            observe!(slots, 2, mode) do
                stored_gather!(workspace.first, state.q, xratio,
                               Val(1), workspace, 1)
            end
            observe!(slots, 3, mode) do
                stored_flux!(workspace.first, state.v, yratio,
                             Val(2), workspace, 2)
            end
            observe!(slots, 4, mode) do
                stored_gather!(workspace.second, workspace.first, yratio,
                               Val(2), workspace, 2)
            end
            observe!(slots, 5, mode) do
                stored_flux!(workspace.second, state.u, xratio,
                             Val(1), workspace, 3)
            end
            observe!(slots, 6, mode) do
                stored_gather!(state.q, workspace.second, xratio,
                               Val(1), workspace, 3)
            end
        end
        state.t += dt
    end
    return state
end

function verify!(state, reference, initial_mass, dt, steps)
    target_time = steps * dt
    abs(state.t - target_time) <=
        128eps(Float64) * steps * max(1.0, target_time) ||
        error("profiled trajectory ended at wrong time")
    all(isfinite, state.q) || error("profiled trajectory contains nonfinite q")
    difference = maximum(abs, state.q .- reference)
    scale = max(1.0, maximum(abs, reference))
    difference <= 512eps(Float64) * scale ||
        error("pass trajectory changed the complete-step result: $difference")
    return Dict(
        "max_state_delta" => difference,
        "mass_error" => abs(Toy.mass(state) - initial_mass),
        "translation_rms_error" => Toy.translation_error(state),
        "checksum" => sum(state.q),
    )
end

function profile(options)
    backend = Symbol(option(options, "backend", "serial"))
    flux = Symbol(option(options, "flux", "recompute"))
    backend in (:serial, :threads, :spawn, :polyester) || error("invalid backend")
    flux in (:recompute, :stored) || error("invalid flux policy")
    nx = positive_integer(options, "nx", 512)
    ny = positive_integer(options, "ny", nx)
    steps = positive_integer(options, "steps", 20)
    repeats = positive_integer(options, "repeats", 5)
    warmups = positive_integer(options, "warmups", 2)
    tile_x = positive_integer(options, "tile-x", 128)
    tile_y = positive_integer(options, "tile-y", 32)
    threshold = integer_option(options, "threshold", 16384)
    threshold >= 0 || error("--threshold must be nonnegative")
    batches = positive_integer(options, "batches-per-worker", 4)

    state = Toy.State(nx, ny)
    Toy.initialize!(state)
    initial = copy(state.q)
    initial_mass = Toy.mass(state)
    workspace = Toy.Workspace(state; flux, backend,
        tile = (tile_x, tile_y), threshold, batches_per_worker = batches)
    default_dt = 0.35 / (abs(state.u_speed) * nx + abs(state.v_speed) * ny)
    dt = parse(Float64, option(options, "dt", string(default_dt)))
    Toy.validate_step(state, workspace, dt)
    pass_names = flux === :recompute ?
        ["x_half_1", "y_full", "x_half_2"] :
        ["x_half_1_flux", "x_half_1_gather", "y_full_flux",
         "y_full_gather", "x_half_2_flux", "x_half_2_gather"]
    pass_count = length(pass_names)
    run_passes! = flux === :recompute ? recompute_passes! : stored_passes!

    # Warm every path, including pass wrappers, before measuring. Instrument
    # worker IDs in a separate untimed step only.
    Toy.track_workers!(workspace, true)
    Toy.step!(state, workspace, dt)
    worker_ids = Toy.active_workers(workspace)
    Toy.track_workers!(workspace, false)
    for _ in 1:warmups
        reset!(state, initial)
        Toy.simulate!(state, workspace, dt, steps)
        reset!(state, initial)
        run_passes!(state, workspace, dt, steps, zeros(Float64, pass_count), :clock)
    end
    # The allocation mode uses different slot and wrapper specializations.
    # Compile them before the diagnostic byte counts are collected.
    reset!(state, initial)
    @allocated Toy.simulate!(state, workspace, dt, steps)
    reset!(state, initial)
    run_passes!(state, workspace, dt, steps, zeros(Int, pass_count), :allocation)

    full_samples = Float64[]
    pass_samples = [Float64[] for _ in 1:pass_count]
    reference = similar(initial)
    full_diagnostics = Dict{String, Float64}()
    pass_diagnostics = Dict{String, Float64}()
    for repetition in 1:repeats
        reset!(state, initial)
        start = time_ns()
        Toy.simulate!(state, workspace, dt, steps)
        push!(full_samples, (time_ns() - start) * 1.0e-9)
        if repetition == 1
            copyto!(reference, state.q)
            full_diagnostics = verify!(state, reference, initial_mass, dt, steps)
        else
            verify!(state, reference, initial_mass, dt, steps)
        end

        reset!(state, initial)
        slots = zeros(Float64, pass_count)
        run_passes!(state, workspace, dt, steps, slots, :clock)
        pass_diagnostics = verify!(state, reference, initial_mass, dt, steps)
        for i in eachindex(slots)
            push!(pass_samples[i], slots[i])
        end
    end

    reset!(state, initial)
    full_bytes = @allocated Toy.simulate!(state, workspace, dt, steps)
    verify!(state, reference, initial_mass, dt, steps)
    reset!(state, initial)
    pass_bytes = zeros(Int, pass_count)
    run_passes!(state, workspace, dt, steps, pass_bytes, :allocation)
    verify!(state, reference, initial_mass, dt, steps)

    passes = [Dict(
        "name" => pass_names[i],
        "elapsed_seconds" => pass_samples[i],
        "median_seconds" => median(pass_samples[i]),
        "allocated_bytes" => pass_bytes[i],
    ) for i in eachindex(pass_names)]
    source_path = normpath(joinpath(@__DIR__, "..", "src", "UniformLWAdvection.jl"))
    extension_path = normpath(joinpath(@__DIR__, "..", "ext",
                                   "UniformLWAdvectionPolyesterExt.jl"))
    return Dict(
        "schema" => 1,
        "kind" => "diagnostic_pass_profile_not_acceptance_timing",
        "julia_version" => string(VERSION),
        "toy_source_sha256" => bytes2hex(sha256(read(source_path))),
        "polyester_extension_sha256" => backend === :polyester ?
            bytes2hex(sha256(read(extension_path))) : "not_loaded",
        "backend" => string(backend), "flux" => string(flux),
        "nx" => nx, "ny" => ny, "steps" => steps,
        "repeats" => repeats, "warmups" => warmups,
        "dt" => dt, "tile_x" => tile_x, "tile_y" => tile_y,
        "threshold" => threshold, "batches_per_worker" => batches,
        "default_threads" => Threads.nthreads(:default),
        "blas_threads" => BLAS.get_num_threads(),
        "process_allowed_cpus" => allowed_cpus("/proc/self/status"),
        "calling_thread_allowed_cpus" => allowed_cpus("/proc/thread-self/status"),
        "observed_worker_ids_untimed" => worker_ids,
        "working_array_bytes" => Toy.memory_bytes(state, workspace),
        "full_elapsed_seconds" => full_samples,
        "full_median_seconds" => median(full_samples),
        "full_allocated_bytes" => full_bytes,
        "passes" => passes,
        "pass_median_sum_seconds" => sum(pass["median_seconds"] for pass in passes),
        "full_diagnostics" => full_diagnostics,
        "pass_diagnostics" => pass_diagnostics,
    )
end

function main(options)
    result = profile(options)
    destination = get(options, "output", "-")
    if destination == "-"
        TOML.print(stdout, result)
    else
        open(destination, "w") do io
            TOML.print(io, result)
        end
    end
    return nothing
end

const COMMAND_OPTIONS = parse_options(ARGS)
if get(COMMAND_OPTIONS, "backend", "serial") == "polyester"
    # Load the optional extension before entering the profiled call world.
    @eval using Polyester
end
main(COMMAND_OPTIONS)
