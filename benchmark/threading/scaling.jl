# Thread-scaling probe for one fixture in a fresh process.
#
#     julia --project=. -t N benchmark/threading/scaling.jl CASE [STEPS] [PROFILE_STEPS]
#
# CASE is `taylor_green` (16 384 leaves: one 16² root refined to level 3),
# `capillary` (CMDGFM capillary wave k = 4: base mesh 32×96 cells in 8² blocks,
# max_level 2, 7 680 cells) or `cylinder` (example 04 at its default max_level 3). The
# probe warms the flow, times
# STEPS fixed-dt steps one `simulate!` call each, prints a checksum of the final
# state (bitwise comparison across thread counts) and, if PROFILE_STEPS > 0,
# profiles further steps and splits the main thread's samples into serial
# samples and samples inside a Polyester batch.
using LinearAlgebra
BLAS.set_num_threads(1)
using Printf
using Profile
using SHA
using Statistics
using TwoPhaseAMR

module Cylinder
    include(joinpath(@__DIR__, "../../examples/04_cylinder_flow/cylinder_flow.jl"))
end

function taylor_green_flow()
    mesh = BlockTreeMesh(
        block_size = (16, 16), extent = (2pi, 2pi),
        max_level = 3, refine = (key, origin, extent) -> key.level < 3,
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

function cylinder_flow()
    flow = Cylinder.run_example(final_time = 0.0).flow
    return flow, stable_timestep(flow; cfl = 0.45)
end

const FIXTURES = Dict(
    "taylor_green" => taylor_green_flow,
    "capillary" => capillary_flow,
    "cylinder" => cylinder_flow,
)

step!(flow, dt) = simulate!(flow, flow.time + dt; dt)

function state_checksum(flow)
    context = SHA.SHA256_CTX()
    velocity = flow.velocity
    phases = velocity isa TwoPhaseAMR.TwoPhaseVelocity ? collect(velocity) : (velocity,)
    for phase in phases
        SHA.update!(context, reinterpret(UInt8, collect(Float64, phase.interior)))
    end
    for block in eachindex(flow.pressure.blocks)
        SHA.update!(context, reinterpret(UInt8, vec(collect(Float64, interior(flow.pressure, block)))))
    end
    return bytes2hex(SHA.digest!(context))[1:16]
end

function timed_steps(flow, dt, steps)
    times = Float64[]
    bytes = Int[]
    for _ in 1:steps
        stats = @timed step!(flow, dt)
        push!(times, stats.time)
        push!(bytes, stats.bytes)
    end
    return times, bytes
end

function _is_parallel_frame(frame)
    file = string(frame.file)
    return occursin("Polyester", file) || occursin("ThreadingUtilities", file) ||
        occursin("PolyesterWeave", file)
end

_frame_name(frame) = string(frame.func, " ", basename(string(frame.file)), ":", frame.line)
const PACKAGE_SOURCE = joinpath(pkgdir(TwoPhaseAMR), "src")
_in_package(frame) = !frame.from_c && startswith(string(frame.file), PACKAGE_SOURCE)

"""Split main-thread samples into serial and Polyester-batch samples."""
function serial_profile(data, main_thread)
    lidict = Profile.getdict(data)
    serial = 0
    parallel = 0
    serial_self = Dict{String, Int}()
    serial_inclusive = Dict{String, Int}()
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
                    parallel += 1
                else
                    serial += 1
                    package = filter(_in_package, frames)
                    if !isempty(package)
                        name = _frame_name(first(package))
                        serial_self[name] = get(serial_self, name, 0) + 1
                        for name in unique(string(frame.func) for frame in package)
                            serial_inclusive[name] = get(serial_inclusive, name, 0) + 1
                        end
                    end
                end
            end
            block_start = index + 1
        end
        index += 1
    end
    return serial, parallel, serial_self, serial_inclusive
end

function main()
    case = ARGS[1]
    steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    profile_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 0
    flow, dt = FIXTURES[case]()
    cells = sum(length(interior(flow.pressure, block)) for block in eachindex(flow.pressure.blocks))
    expected_cells = get(Dict("taylor_green" => 16_384, "capillary" => 7_680), case, cells)
    cells == expected_cells || error("$case fixture has $cells cells, expected $expected_cells")
    for _ in 1:5
        step!(flow, dt)
    end
    GC.gc()
    blocks_before = length(flow.pressure.blocks)
    times, bytes = timed_steps(flow, dt, steps)
    @printf(
        "case=%s threads=%d cells=%d blocks=%d->%d dt=%.4e steps=%d median_ms=%.3f min_ms=%.3f mean_ms=%.3f median_bytes=%d checksum=%s\n",
        case, Threads.nthreads(:default), cells, blocks_before, length(flow.pressure.blocks), dt, steps,
        1.0e3median(times), 1.0e3minimum(times), 1.0e3mean(times), round(Int, median(bytes)), state_checksum(flow),
    )
    profile_steps > 0 || return nothing
    Profile.init(n = 10^7, delay = 0.0005)
    main_thread = Threads.threadid()
    Profile.@profile for _ in 1:profile_steps
        step!(flow, dt)
    end
    serial, parallel, self, inclusive = serial_profile(Profile.fetch(include_meta = true), main_thread)
    total = serial + parallel
    @printf("main-thread samples=%d serial=%.1f%% batch=%.1f%%\n", total, 100serial / total, 100parallel / total)
    println("serial self (innermost package frame), % of main-thread samples:")
    for (name, count) in first(sort!(collect(self); by = last, rev = true), 30)
        @printf("  %5.1f%%  %s\n", 100count / total, name)
    end
    println("serial inclusive (package functions), % of main-thread samples:")
    for (name, count) in first(sort!(collect(inclusive); by = last, rev = true), 45)
        @printf("  %5.1f%%  %s\n", 100count / total, name)
    end
    return nothing
end

main()
