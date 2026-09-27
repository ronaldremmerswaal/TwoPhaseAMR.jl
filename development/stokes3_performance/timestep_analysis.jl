# Focused Stokes3 CFL measurements. Run from the repository root:
#   julia --project=. development/stokes3_performance/timestep_analysis.jl one
#   julia --project=. development/stokes3_performance/timestep_analysis.jl cmdgfm
# Reports complete solver-level query and its constituent operations on the
# initial mesh and after four AMR updates. Do not sum the component timings:
# they are separate repeated evaluations of the same frozen state.
using TwoPhaseAMR
using LinearAlgebra: BLAS
using Printf
using Profile
using Statistics: median

include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

function _model_for_timestep(mode)
    mode == :one && return OneVelocityModel()
    mode == :cmdgfm && return TwoVelocityModel(
        pressure_projection = CMDGFMProjection(),
    )
    throw(ArgumentError("mode must be one or cmdgfm"))
end

function _measure_timestep(label, f; repetitions = 8)
    f()
    GC.gc()
    samples = [@timed f() for _ in 1:repetitions]
    result = last(samples).value
    display_result = result isa Real ? repr(result) : string(nameof(typeof(result)))
    bytes = [sample.bytes for sample in samples]
    @printf("CALL %-25s median_ms=%9.3f median_bytes=%10.0f max_bytes=%d value=%s\n",
        label, 1e3 * median([sample.time for sample in samples]),
        median(bytes), maximum(bytes),
        display_result)
    return maximum(bytes)
end

function _measure_aperture_rebuild(field; repetitions = 8)
    # Diagnostic-only forced misses: an ordinary step sees one new state
    # after transport and the following CFL sees a cache hit.
    TwoPhaseAMR._current_phase_face_apertures!(field)
    GC.gc()
    samples = map(1:repetitions) do _
        field.phase_apertures_valid[] = false
        @timed TwoPhaseAMR._current_phase_face_apertures!(field)
    end
    @printf("CALL %-25s median_ms=%9.3f median_bytes=%10.0f max_bytes=%d value=phase_apertures\n",
        "aperture_forced_rebuild",
        1e3 * median([sample.time for sample in samples]),
        median([sample.bytes for sample in samples]),
        maximum(sample.bytes for sample in samples))
    return nothing
end

function _timestep_state(mode, label, flow)
    model = flow.solver.two_phase
    work = flow.workspace.two_phase.advection
    @printf("STATE mode=%s label=%s steps=%d updates=%d blocks=%d faces=%d\n",
        mode, label, flow.steps, flow.mesh_updates,
        length(flow.solver.mesh.blocks), length(flow.solver.mesh.faces))
    bytes = _measure_timestep(
        "solver_complete", () -> stable_timestep(flow),
    )
    iszero(bytes) || error("warmed solver-level CFL query allocated $bytes bytes")
    _measure_timestep("reconstruct", () -> reconstruct!(model.interface))
    if mode == :cmdgfm
        _measure_aperture_rebuild(model.interface)
        _measure_timestep("aperture_copy_to_WY", () ->
            TwoPhaseAMR._freeze_phase_apertures!(model.interface, work))
        _measure_timestep("frozen_rate_scan", () ->
            TwoPhaseAMR._frozen_interface_stable_timestep(
                flow.velocity, work; cfl = 0.45,
            ))
    else
        _measure_timestep("one_rate_scan", () ->
            stable_timestep(flow.velocity, work; cfl = 0.45))
    end
    flush(stdout)
    return nothing
end

function _timestep_allocations(mode; post_amr = true)
    model = _model_for_timestep(mode)
    # Warm the integrated call path, including the post-AMR path.
    post_amr && run_example(velocity_model = model, final_time = 0.005)
    initial = run_example(velocity_model = model, final_time = 0.0).flow
    _timestep_state(mode, "initial", initial)
    target = initial
    if post_amr
        evolved = run_example(velocity_model = model, final_time = 0.05).flow
        evolved.mesh_updates > 0 || error("expected AMR topology changes")
        _timestep_state(mode, "post_amr", evolved)
        target = evolved
    end

    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        stable_timestep(target)
    end
    sites = Dict{String, Tuple{Int, Int}}()
    for (sample_index, sample) in pairs(Profile.Allocs.fetch().allocs)
        if sample_index <= 8
            typename = sample.type isa DataType ? nameof(sample.type) :
                typeof(sample.type)
            @printf("ALLOC_SAMPLE size=%d type=%s\n",
                sample.size, typename)
        end
        frames = sample.stacktrace
        location_index = findfirst(frame -> occursin(
            "TwoPhaseAMR.jl/src/", string(frame.file),
        ), frames)
        frame = isnothing(location_index) ? last(frames) :
            frames[location_index]
        location = string(frame.func, "@",
            basename(string(frame.file)), ":", frame.line)
        calls, bytes = get(sites, location, (0, 0))
        sites[location] = (calls + 1, bytes + sample.size)
    end
    for (site, (calls, bytes)) in
            sort!(collect(sites); by = x -> x.second[2], rev = true)
        @printf("ALLOC_SITE calls=%d bytes=%d location=%s\n",
            calls, bytes, site)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    @printf("ENV julia=%s threads=%d blas_threads=%d\n",
        VERSION, Threads.nthreads(), BLAS.get_num_threads())
    _timestep_allocations(
        Symbol(get(ARGS, 1, "cmdgfm"));
        post_amr = get(ARGS, 2, "post_amr") != "initial",
    )
end
