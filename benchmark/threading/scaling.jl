# Thread-scaling probe for one or more fixtures in a fresh process.
#
#     julia --project=. -t N benchmark/threading/scaling.jl CASE[,CASE...] [STEPS] [PROFILE_STEPS]
#
# Cases (`_fine`: one level finer):
#   taylor_green[_fine]  128² (256²) Taylor–Green, hierarchical: one 16² root
#                        refined to level 3 (4), no adaptation;
#   cylinder[_fine]      example 04 at max_level 3 (4), wake/geometry AMR and the
#                        example's diagnostics; `cylinder[_fine]_noforce` drops
#                        the example's serial `pressure_force` diagnostics;
#   stokes3_one[_fine]   example 10 with OneVelocityModel() at max_level 5 (6);
#   stokes3_two[_fine]   example 10 with TwoVelocityModel() at max_level 5 (6);
#   capillary[_fine]     CMDGFM capillary wave k = 4 (example 06 setup, base mesh
#                        32×96 cells in 8² blocks) with interface AMR to
#                        max_level 2 (3); the interface barely moves, so no regrid.
# The probe takes 5 warm-up steps, then STEPS fixed-dt steps in one `simulate!`
# call and reports the per-step solver wall time (`StepInfo.wall_time`: the
# step, mesh adaptation and diagnostics) and a SHA-256 of the final state
# (bitwise comparison across thread counts). The Stokes-3 cases warm up until
# one step has regridded and extend both windows until they hold a regrid step;
# regular (`steady_mean_ms`) and regrid steps are reported separately. With
# PROFILE_STEPS > 0 it profiles further steps one at a time, files each step's
# samples under its class (regular or regrid) and splits the main thread's
# samples into serial samples and samples inside a Polyester batch.
using LinearAlgebra
BLAS.set_num_threads(1)
using Printf
using Profile
using SHA
using Statistics
using TwoPhaseAMR

const EXAMPLES = joinpath(pkgdir(TwoPhaseAMR), "examples")
module Cylinder
    include(joinpath(Main.EXAMPLES, "04_cylinder_flow", "cylinder_flow.jl"))
end
module Stokes3
    include(joinpath(Main.EXAMPLES, "10_stokes3_wave", "stokes3_wave.jl"))
end

function taylor_green_flow(level)
    mesh = BlockTreeMesh(
        block_size = (16, 16), extent = (2pi, 2pi),
        max_level = level, refine = (key, origin, extent) -> key.level < level,
    )
    velocity(x, y, time, axis) = exp(-0.04time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.02);
        velocity, boundary = FreeSlip(),
    )
    return flow, stable_timestep(flow; cfl = 0.45)
end

function capillary_flow(; k = 4, max_level = 2)
    wavelength = 1.0
    wavenumber = 2pi / wavelength
    density = (1.0, 1.0e-3)
    surface_tension = 1.0
    dynamic_viscosity = density .* sqrt(surface_tension * wavelength / 3000.0)
    levelset = (x, y) -> y - 0.25 - 0.01cos(wavenumber * x)
    mesh = BlockTreeMesh(
        base_mesh = (8k, 24k), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (wavelength, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed || break
        initialize!(interface, levelset; lipschitz = 1)
    end
    physics = TwoPhase(interface; density, dynamic_viscosity, surface_tension)
    flow = FlowSimulation(
        mesh, physics;
        velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection()),
        boundary = FreeSlip(), adaptivity = Adaptivity(criteria = (refinement,)),
    )
    return flow, 0.5 * stable_timestep(flow)
end

function cylinder_flow(max_level; pressure_force = true)
    flow = Cylinder.run_example(; max_level, final_time = 0.0).flow
    pressure_force || drop_diagnostics!(flow, (:pressure_drag, :pressure_lift))
    return flow, stable_timestep(flow; cfl = 0.45)
end

"""Remove the example's `pressure_force` diagnostics (serial example code)."""
function drop_diagnostics!(flow, names)
    collection = flow.diagnostics
    filter!(series -> !(diagnostic_name(series.diagnostic) in names), collection.series)
    empty!(collection.indices)
    for (index, series) in enumerate(collection.series)
        collection.indices[diagnostic_name(series.diagnostic)] = index
    end
    return flow
end

function stokes3_flow(velocity_model, max_level)
    flow = Stokes3.run_example(; velocity_model, max_level, final_time = 0.0).flow
    return flow, 0.5 * stable_timestep(flow)
end

const FIXTURES = Dict(
    "taylor_green" => () -> taylor_green_flow(3),
    "taylor_green_fine" => () -> taylor_green_flow(4),
    "cylinder" => () -> cylinder_flow(3),
    "cylinder_fine" => () -> cylinder_flow(4),
    "cylinder_noforce" => () -> cylinder_flow(3; pressure_force = false),
    "cylinder_fine_noforce" => () -> cylinder_flow(4; pressure_force = false),
    "stokes3_one" => () -> stokes3_flow(OneVelocityModel(), 5),
    "stokes3_one_fine" => () -> stokes3_flow(OneVelocityModel(), 6),
    "stokes3_two" => () -> stokes3_flow(TwoVelocityModel(), 5),
    "stokes3_two_fine" => () -> stokes3_flow(TwoVelocityModel(), 6),
    "capillary" => () -> capillary_flow(max_level = 2),
    "capillary_fine" => () -> capillary_flow(max_level = 3),
)
const EXPECTED_CELLS = Dict("taylor_green" => 16_384, "taylor_green_fine" => 65_536, "capillary" => 7_680)

"""Collect the solver wall time and topology flag of every completed step."""
struct StepTimes <: AbstractObserver
    seconds::Vector{Float64}
    topology_changed::Vector{Bool}
end
StepTimes() = StepTimes(Float64[], Bool[])
function TwoPhaseAMR.observe!(observer::StepTimes, event::Symbol, state, info)
    event === :step || return nothing
    push!(observer.seconds, info.wall_time)
    push!(observer.topology_changed, info.topology_changed)
    return nothing
end
# A custom observer would receive an independent-step procedure (copied state
# advanced alongside the run); the timer does not use it.
TwoPhaseAMR._observer_step_procedure(::StepTimes, solver, force, options, workspace) = nothing

# One `simulate!` call per step: a multi-step call can end on a shortened step
# (accumulated time round-off), whose new dt rebuilds the dt-dependent
# preconditioners and would put a ~100 ms outlier into every window.
function run_steps!(flow, dt, steps; observer = NoObserver())
    for _ in 1:steps
        simulate!(flow, flow.time + dt; dt, observer)
    end
    return flow
end

function state_checksum(flow)
    context = SHA.SHA256_CTX()
    velocity = flow.velocity
    phases = velocity isa TwoPhaseAMR.TwoPhaseVelocity ? collect(velocity) : (velocity,)
    for phase in phases
        SHA.update!(context, reinterpret(UInt8, phase.interior))
    end
    fields = isnothing(flow.interface) ? (flow.pressure,) : (flow.pressure, flow.interface.fraction)
    for field in fields, block in eachindex(field.blocks)
        SHA.update!(context, reinterpret(UInt8, vec(collect(Float64, interior(field, block)))))
    end
    return bytes2hex(SHA.digest!(context))[1:16]
end

_is_gc_frame(frame) = frame.from_c &&
    any(prefix -> startswith(string(frame.func), prefix), ("jl_gc", "ijl_gc", "gc_"))

function _is_parallel_frame(frame)
    file = string(frame.file)
    return occursin("Polyester", file) || occursin("ThreadingUtilities", file) ||
        occursin("PolyesterWeave", file)
end

_frame_name(frame) = string(frame.func, " (", basename(string(frame.file)), ")")
const PACKAGE_SOURCE = joinpath(pkgdir(TwoPhaseAMR), "src")
const EXAMPLE_SOURCE = EXAMPLES
_in_package(frame) = !frame.from_c && (
    startswith(string(frame.file), PACKAGE_SOURCE) ||
        startswith(string(frame.file), EXAMPLE_SOURCE)
)

"""Main-thread profile samples of one step class (regular or regrid steps)."""
mutable struct ProfileBucket
    steps::Int
    serial::Int
    parallel::Int
    serial_self::Dict{String, Int}
    serial_inclusive::Dict{String, Int}
end
ProfileBucket() = ProfileBucket(0, 0, 0, Dict{String, Int}(), Dict{String, Int}())

"""Add the main-thread samples of `data` to `bucket`, split into serial
samples and samples inside a Polyester batch."""
function serial_profile!(bucket, data, main_thread)
    lidict = Profile.getdict(data)
    block_start = 1
    index = 1
    while index <= length(data)
        if Profile.is_block_end(data, index)
            thread = data[index - Profile.META_OFFSET_THREADID]
            if thread == main_thread
                frames = Base.StackTraces.StackFrame[]
                for ip in data[block_start:(index - 6)]
                    append!(frames, lidict[ip])
                end
                if any(_is_parallel_frame, frames)
                    bucket.parallel += 1
                else
                    bucket.serial += 1
                    package = filter(_in_package, frames)
                    # GC and samples without a package frame (task
                    # scheduler, runtime) are named separately.
                    name = if any(_is_gc_frame, frames)
                        "[GC]"
                    elseif isempty(package)
                        string("[outside] ", isempty(frames) ? "?" : frames[1].func)
                    else
                        _frame_name(first(package))
                    end
                    bucket.serial_self[name] = get(bucket.serial_self, name, 0) + 1
                    for name in unique(string(frame.func) for frame in package)
                        bucket.serial_inclusive[name] = get(bucket.serial_inclusive, name, 0) + 1
                    end
                end
            end
            block_start = index + 1
        end
        index += 1
    end
    return bucket
end

function print_profile(label, bucket)
    total = bucket.serial + bucket.parallel
    iszero(total) && return nothing
    @printf(
        "%s: main-thread samples=%d steps=%d serial=%.1f%% batch=%.1f%%\n",
        label, total, bucket.steps, 100bucket.serial / total, 100bucket.parallel / total,
    )
    println("$label serial self by function (innermost package frame), % of main-thread samples:")
    for (name, count) in first(sort!(collect(bucket.serial_self); by = last, rev = true), 60)
        @printf("  %5.1f%%  %s\n", 100count / total, name)
    end
    println("$label serial inclusive (package functions), % of main-thread samples:")
    for (name, count) in first(sort!(collect(bucket.serial_inclusive); by = last, rev = true), 45)
        @printf("  %5.1f%%  %s\n", 100count / total, name)
    end
    return nothing
end

ncells(flow) = sum(length(interior(flow.pressure, block)) for block in eachindex(flow.pressure.blocks))

function measure(case, steps, profile_steps)
    flow, dt = FIXTURES[case]()
    cells = ncells(flow)
    haskey(EXPECTED_CELLS, case) && cells != EXPECTED_CELLS[case] &&
        error("$case fixture has $cells cells, expected $(EXPECTED_CELLS[case])")
    # Warm up for 5 steps and, on the regridding Stokes-3 cases, until one step
    # has changed the topology, so the timed window holds no first-call
    # compilation.
    regridding = startswith(case, "stokes3")
    warmup = StepTimes()
    run_steps!(flow, dt, 5; observer = warmup)
    while regridding && !any(warmup.topology_changed) && length(warmup.seconds) < 40
        run_steps!(flow, dt, 1; observer = warmup)
    end
    regridding && !any(warmup.topology_changed) && error("$case did not regrid in 40 warm-up steps")
    GC.gc()
    blocks_before = length(flow.pressure.blocks)
    cells_before = ncells(flow)
    # The regridding cases extend the window until it holds a regrid step, so
    # both step classes are measured.
    observer = StepTimes()
    stats = @timed begin
        run_steps!(flow, dt, steps; observer)
        while regridding && !any(observer.topology_changed) && length(observer.seconds) < steps + 40
            run_steps!(flow, dt, 1; observer)
        end
    end
    times = observer.seconds
    regrid = observer.topology_changed
    steady = times[.!regrid]
    @printf(
        "case=%s threads=%d cells=%d->%d blocks=%d->%d dt=%.4e steps=%d regrid_steps=%d median_ms=%.3f mean_ms=%.3f min_ms=%.3f steady_mean_ms=%.3f regrid_mean_ms=%.3f bytes_per_step=%d checksum=%s\n",
        case, Threads.nthreads(:default), cells_before, ncells(flow), blocks_before,
        length(flow.pressure.blocks), dt, length(times), count(regrid),
        1.0e3median(times), 1.0e3mean(times), 1.0e3minimum(times),
        isempty(steady) ? NaN : 1.0e3mean(steady),
        any(regrid) ? 1.0e3mean(times[regrid]) : NaN,
        round(Int, stats.bytes / length(times)), state_checksum(flow),
    )
    println("step_ms=", join(round.(1.0e3 .* times; digits = 2), ","))
    flush(stdout)
    profile_steps > 0 || return nothing
    # Profile one step at a time and file its samples under its step class.
    Profile.init(n = 10^7, delay = 0.0005)
    main_thread = Threads.threadid()
    regular, regridded = ProfileBucket(), ProfileBucket()
    profile_times = StepTimes()
    while length(profile_times.seconds) < profile_steps ||
            (
            regridding && !any(profile_times.topology_changed) &&
                length(profile_times.seconds) < profile_steps + 40
        )
        Profile.clear()
        Profile.@profile run_steps!(flow, dt, 1; observer = profile_times)
        bucket = last(profile_times.topology_changed) ? regridded : regular
        bucket.steps += 1
        serial_profile!(bucket, Profile.fetch(include_meta = true), main_thread)
    end
    print_profile("regular steps", regular)
    print_profile("regrid steps", regridded)
    flush(stdout)
    return nothing
end

function main()
    cases = split(ARGS[1], ',')
    steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    profile_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    for case in cases
        measure(String(case), steps, profile_steps)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
