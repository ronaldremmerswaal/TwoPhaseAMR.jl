# Production sampling with a finer breakdown of the old residual stage.
# No production methods are replaced. Source-line classification identifies
# the original inline predictor loop; named helpers identify the optimized
# loop. Profiles include real transport, AMR, pressure and default diagnostics.
#
# julia --project=. step_overhead_analysis.jl cmdgfm 0.5
# julia --project=. step_overhead_analysis.jl cmdgfm 0.08 allocs
include(joinpath(@__DIR__, "benchmark.jl"))

function predictor_line_ranges()
    ranges = Dict{String, UnitRange{Int}}()
    for relative in ("two_phase/two_velocity/stepping.jl", "single_phase/stepping.jl")
        filename = joinpath(dirname(pathof(TwoPhaseAMR)), relative)
        lines = readlines(filename)
        force_line = findfirst(line -> occursin("time_step * _body_force", line), lines)
        isnothing(force_line) && continue
        first_line = force_line
        while first_line > 1 && !occursin("for ", lines[first_line])
            first_line -= 1
        end
        if occursin("two_velocity", filename)
            first_line -= 1 # original outer phase loop
        end
        ranges[replace(filename, '\\' => '/')] = first_line:(force_line + 2)
    end
    return ranges
end

function overhead_category(frames, ranges)
    names = string.(getproperty.(frames, :func))
    parent = stage(names)
    parent in (:step_overhead, :simulation_overhead) || return parent
    contains_name(names, "_mark_inactive_phase_velocities!") && return :phase_masks
    (contains_name(names, "_enforce_phase_velocity_boundary!") ||
        contains_name(names, "enforce_velocity_boundary!") ||
        contains_name(names, "enforce_no_permeation!")) && return :boundary_enforcement
    (contains_name(names, "_copy_velocity!") || contains_name(names, "copyto!")) && return :velocity_copy
    (contains_name(names, "_add_body_force!") ||
        contains_name(names, "_body_force_substep!")) && return :external_force_predictor
    for frame in frames
        filename = replace(string(frame.file), '\\' => '/')
        for (source, interval) in ranges
            endswith(filename, source) || endswith(source, filename) || continue
            frame.line in interval && return :external_force_predictor
        end
    end
    contains_name(names, "_two_velocity_flow_workspace!") && return :intrinsic_workspace_lookup
    contains_name(names, "_project_two_velocity!") && return :projection_wrapper
    contains_name(names, "_navier_stokes_amr_workspace") && return :initial_amr_registration
    return parent == :step_overhead ? :other_step_controls : :other_simulation_controls
end

function overhead_profile_summary(wall, ranges)
    data, dictionary = Profile.retrieve(; include_meta = false)
    counts = Dict{Symbol, Int}()
    inclusive = Dict{String, Int}()
    frames = Base.StackTraces.StackFrame[]
    for ip in data
        if iszero(ip)
            isempty(frames) && continue
            category = overhead_category(frames, ranges)
            counts[category] = get(counts, category, 0) + 1
            if category in (:other_step_controls, :other_simulation_controls)
                for frame in unique(frames)
                    occursin("TwoPhaseAMR", string(frame.file)) || continue
                    key = string(frame.func, " ", frame.file, ":", frame.line)
                    inclusive[key] = get(inclusive, key, 0) + 1
                end
            end
            empty!(frames)
        else
            append!(frames, get(dictionary, ip, Base.StackTraces.StackFrame[]))
        end
    end
    total = sum(values(counts))
    active = total - get(counts, :unclassified, 0)
    println("OVERHEAD_SAMPLES total=", total, " active=", active)
    for (category, count) in sort!(collect(counts); by = last, rev = true)
        category == :unclassified && continue
        @printf("OVERHEAD_STAGE %-32s count=%6d seconds=%9.5f\n", category, count, wall * count / active)
    end
    for (key, count) in first(sort!(collect(inclusive); by = last, rev = true),
            min(20, length(inclusive)))
        println("OVERHEAD_INCLUSIVE count=", count, " ", key)
    end
end

function overhead_allocation_summary(rate, ranges)
    counts, sizes = Dict{Symbol, Int}(), Dict{Symbol, Int}()
    for allocation in Profile.Allocs.fetch().allocs
        category = overhead_category(allocation.stacktrace, ranges)
        counts[category] = get(counts, category, 0) + 1
        sizes[category] = get(sizes, category, 0) + allocation.size
    end
    for (category, size) in sort!(collect(sizes); by = last, rev = true)
        @printf("OVERHEAD_ALLOC %-32s count=%6d sampled_bytes=%12d estimated_GB=%9.5f\n",
            category, counts[category], size, size / rate / 1e9)
    end
    println("OVERHEAD_ALLOC sampling_rate=", rate,
        " note=event_sampling_not_exact_stage_bytes")
end

function analyze_step_overhead(mode; final_time = 0.5, allocations = false)
    mode in (:one, :cmdgfm) || throw(ArgumentError("expected one or cmdgfm"))
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    println("SOURCE ", pathof(TwoPhaseAMR), " runtime=", VERSION,
        " julia_threads=", Threads.nthreads(), " blas_threads=", BLAS.get_num_threads())
    run_example(velocity_model = model, final_time = min(0.05, final_time))
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    ranges = predictor_line_ranges()
    flush(stdout)
    if allocations
        rate = 0.0005
        Profile.Allocs.clear()
        measured = Profile.Allocs.@profile sample_rate=rate @timed simulate!(flow, final_time)
        overhead_allocation_summary(rate, ranges)
    else
        Profile.init(; n = 6_000_000, delay = 0.002)
        Profile.clear()
        measured = Profile.@profile @timed simulate!(flow, final_time)
        overhead_profile_summary(measured.time, ranges)
    end
    @printf("OVERHEAD_ADVANCE mode=%s time=%.3f wall=%.6f bytes=%d gc=%.6f compile=%.6f steps=%d updates=%d faces=%d pressure_converged=%s\n",
        mode, final_time, measured.time, measured.bytes, measured.gctime,
        measured.compile_time, measured.value.steps, measured.value.mesh_updates,
        length(flow.solver.mesh.faces), measured.value.pressure.converged)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    analyze_step_overhead(Symbol(get(ARGS, 1, "cmdgfm"));
        final_time = parse(Float64, get(ARGS, 2, "0.5")),
        allocations = get(ARGS, 3, "") == "allocs")
end
