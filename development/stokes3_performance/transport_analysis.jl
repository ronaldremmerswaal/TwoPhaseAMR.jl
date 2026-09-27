# Focused, non-invasive time and allocation sampling of Stokes3 transport.
# Run one mode per process from the repository root, for example:
#   julia --project=. development/stokes3_performance/transport_analysis.jl one 0.05
#   julia --project=. development/stokes3_performance/transport_analysis.jl cmdgfm 0.05
# The run is valid for the AMR question only if MESH_UPDATES > 0.
include(joinpath(@__DIR__, "benchmark.jl"))
using Statistics

# An experimental function barrier for the no-embedded-solid Stokes3 case.
# This is not used by the production transport path. It tests whether the
# `geometry::Any` field is responsible for the per-face allocations in the
# current aperture-freezing loop.
function _freeze_no_geometry_kernel!(interface, work, ::Nothing)
    for index in eachindex(interface.mesh.faces)
        liquid, gas = TwoPhaseAMR._phase_face_apertures(
            interface, nothing, index,
        )
        @inbounds begin
            work.liquid_aperture[index] = liquid
            work.gas_aperture[index] = gas
        end
    end
    return work
end

_freeze_no_geometry!(interface, work) =
    _freeze_no_geometry_kernel!(interface, work, interface.geometry)

function geometry_microbenchmark(mode::Symbol; repetitions = 6)
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    interface = flow.solver.two_phase.interface
    reconstruction = [@timed(TwoPhaseAMR.reconstruct!(interface))
        for _ in 1:repetitions]
    @printf("GEOMETRY mode=%s operation=reconstruct median_wall=%.6f median_bytes=%.0f repetitions=%d\n",
        mode, median(item.time for item in reconstruction),
        median(item.bytes for item in reconstruction), repetitions)
    if mode == :cmdgfm
        state = TwoPhaseAMR._two_velocity_flow_workspace!(
            flow.workspace.two_phase, flow.solver.two_phase,
        )
        aperture = [@timed(TwoPhaseAMR._freeze_phase_apertures!(
                interface, state.advection,
            )) for _ in 1:repetitions]
        @printf("GEOMETRY mode=%s operation=freeze_apertures median_wall=%.6f median_bytes=%.0f repetitions=%d\n",
            mode, median(item.time for item in aperture),
            median(item.bytes for item in aperture), repetitions)
        liquid = copy(state.advection.liquid_aperture)
        gas = copy(state.advection.gas_aperture)
        _freeze_no_geometry!(interface, state.advection) # compile before timing
        same = isequal(liquid, state.advection.liquid_aperture) &&
            isequal(gas, state.advection.gas_aperture)
        candidate = [@timed(_freeze_no_geometry!(
                interface, state.advection,
            )) for _ in 1:repetitions]
        @printf("GEOMETRY mode=%s operation=freeze_no_geometry_candidate median_wall=%.6f median_bytes=%.0f repetitions=%d identical=%s\n",
            mode, median(item.time for item in candidate),
            median(item.bytes for item in candidate), repetitions, same)
    end
    fields = (
        interface.normal_x, interface.normal_y, interface.intercept,
        interface.curvature, interface.reconstruction,
    )
    snapshot = map(field -> copy.(field.blocks), fields)
    TwoPhaseAMR.reconstruct!(interface)
    identical = all(eachindex(fields)) do index
        all(isequal(before, after) for (before, after) in
            zip(snapshot[index], fields[index].blocks))
    end
    println("RECONSTRUCTION_IDEMPOTENT mode=", mode, " value=", identical)
    return nothing
end

function allocation_substage(names)
    contains_name(names, "_advect_phase_faces!") &&
        return :phase_face_momentum
    contains_name(names, "_advect_momentum_component!") &&
        return :one_velocity_momentum
    contains_name(names, "reconstruct!") &&
        return :interface_reconstruction
    contains_name(names, "_freeze_phase_apertures!") &&
        return :aperture_freeze
    contains_name(names, "stable_timestep") && return :transport_cfl
    contains_name(names, "_compute_directional_wy_fluxes!") &&
        return :wy_flux_construction
    contains_name(names, "_advect_wy!") && return :other_wy
    return :transport_wrapper
end

function allocation_summary(sample_rate)
    samples = Profile.Allocs.fetch().allocs
    sampled_bytes = Dict{Symbol, Int}()
    sampled_count = Dict{Symbol, Int}()
    transport_bytes = Dict{Symbol, Int}()
    transport_count = Dict{Symbol, Int}()
    package_sources = Dict{String, Int}()
    for allocation in samples
        names = [string(frame.func) for frame in allocation.stacktrace]
        group = stage(names)
        sampled_bytes[group] = get(sampled_bytes, group, 0) + allocation.size
        sampled_count[group] = get(sampled_count, group, 0) + 1
        group == :transport || continue
        substage = allocation_substage(names)
        transport_bytes[substage] =
            get(transport_bytes, substage, 0) + allocation.size
        transport_count[substage] =
            get(transport_count, substage, 0) + 1
        for frame in allocation.stacktrace
            filename = string(frame.file)
            occursin("TwoPhaseAMR.jl/src/", filename) || continue
            location = string(frame.func, "@", basename(filename), ":", frame.line)
            package_sources[location] =
                get(package_sources, location, 0) + allocation.size
            break
        end
    end
    println("ALLOC_PROFILE sample_rate=", sample_rate,
        " sampled_events=", length(samples),
        " sampled_bytes=", sum(values(sampled_bytes)))
    for (group, bytes) in sort!(collect(sampled_bytes); by = last, rev = true)
        @printf("ALLOC_STAGE %-28s %7d %12.3f MB_est\n",
            group, sampled_count[group], bytes / sample_rate / 1e6)
    end
    for (group, bytes) in sort!(collect(transport_bytes); by = last, rev = true)
        @printf("ALLOC_TRANSPORT %-24s %7d %12.3f MB_est\n",
            group, transport_count[group], bytes / sample_rate / 1e6)
    end
    for (location, bytes) in first(
            sort!(collect(package_sources); by = last, rev = true), 20,
        )
        @printf("ALLOC_SOURCE %-65s %12.3f MB_est\n",
            location, bytes / sample_rate / 1e6)
    end
end

function analyze_transport(mode::Symbol; final_time = 0.05,
        warmup_time = final_time, sample_rate = 0.005)
    mode in (:one, :cmdgfm) ||
        throw(ArgumentError("mode must be 'one' or 'cmdgfm'"))
    final_time > 0 || throw(ArgumentError("final_time must be positive"))
    sample_rate > 0 && sample_rate <= 1 ||
        throw(ArgumentError("sample_rate must be in (0, 1]"))
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    @printf("CONFIG mode=%s final_time=%.4f warmup_time=%.4f sample_rate=%.4g julia_threads=%d blas_threads=%d\n",
        mode, final_time, warmup_time, sample_rate,
        Threads.nthreads(), BLAS.get_num_threads())
    warm = @timed run_example(
        velocity_model = model, final_time = warmup_time,
    )
    @printf("WARMUP wall=%.3f compile=%.3f steps=%d updates=%d\n",
        warm.time, warm.compile_time,
        warm.value.result.steps, warm.value.result.mesh_updates)
    flush(stdout)

    timed_flow = run_example(velocity_model = model, final_time = 0.0).flow
    Profile.init(; n = 4_000_000, delay = 0.004)
    Profile.clear()
    timed = nothing
    Profile.@profile timed = @timed simulate!(timed_flow, final_time)
    @printf("TIMED mode=%s wall=%.3f compile=%.3f gc=%.3f bytes=%d steps=%d updates=%d final_blocks=%d final_faces=%d\n",
        mode, timed.time, timed.compile_time, timed.gctime,
        timed.bytes, timed.value.steps, timed.value.mesh_updates,
        length(timed_flow.solver.mesh.blocks),
        length(timed_flow.solver.mesh.faces))
    profile_summary(timed.time)
    flush(stdout)

    allocation_flow = run_example(
        velocity_model = model, final_time = 0.0,
    ).flow
    Profile.Allocs.clear()
    allocated = nothing
    Profile.Allocs.@profile sample_rate=sample_rate begin
        allocated = @timed simulate!(allocation_flow, final_time)
    end
    @printf("ALLOC_RUN mode=%s wall=%.3f compile=%.3f gc=%.3f bytes=%d steps=%d updates=%d final_blocks=%d final_faces=%d\n",
        mode, allocated.time, allocated.compile_time, allocated.gctime,
        allocated.bytes, allocated.value.steps, allocated.value.mesh_updates,
        length(allocation_flow.solver.mesh.blocks),
        length(allocation_flow.solver.mesh.faces))
    allocation_summary(sample_rate)
    if iszero(timed.value.mesh_updates) ||
            iszero(allocated.value.mesh_updates)
        @warn "No AMR update occurred: increase final_time for an AMR-valid comparison"
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(get(ARGS, 1, "one"))
    if get(ARGS, 2, "") == "micro"
        run_example(
            velocity_model = mode == :one ? OneVelocityModel() :
                TwoVelocityModel(pressure_projection = CMDGFMProjection()),
            final_time = 0.005,
        )
        geometry_microbenchmark(mode)
    else
        target = parse(Float64, get(ARGS, 2, "0.05"))
        analyze_transport(mode; final_time = target)
    end
end
