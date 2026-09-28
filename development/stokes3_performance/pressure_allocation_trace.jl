# Attribute allocations on one real force-free Stokes3 predictor. This is a
# development diagnostic, not a whole-run or matched-mesh timing benchmark.
include(joinpath(@__DIR__, "pressure_initial_guess.jl"))
using Profile

function pressure_allocation_trace(; sample_rate = 0.05)
    model = TwoVelocityModel(pressure_projection = CMDGFMProjection())
    # Warm the same assembly/projection specializations, including AMR.
    run_example(velocity_model = model, final_time = 0.03)
    flow = run_example(velocity_model = model, final_time = 0.03).flow
    dt = 0.002
    state = pressure_predictor!(flow, dt)
    decompose_cached_cmdgfm(flow; dt)
    print_samples("ALLOCATION_STAGES", decompose_cached_cmdgfm(flow; dt))
    # Warm profiling's wrapper without counting compilation allocations.
    Profile.Allocs.@profile sample_rate = sample_rate nothing
    Profile.Allocs.clear()
    projection = @timed Profile.Allocs.@profile sample_rate = sample_rate begin
        M._project_two_velocity!(flow.velocity, flow.pressure, flow.solver,
            flow.workspace, state, dt; pressure_prepared = true,
            tolerance = sqrt(eps()), max_iterations = 1000)
    end
    @printf("ALLOCATION_PROJECTION seconds=%.8g bytes=%d compile=%.6f sample_rate=%.3g\n",
        projection.time, projection.bytes, projection.compile_time, sample_rate)
    groups = Dict{String, Tuple{Int, Int}}()
    for allocation in Profile.Allocs.fetch().allocs
        index = findfirst(allocation.stacktrace) do frame
            occursin("TwoPhaseAMR.jl/src/", String(frame.file))
        end
        key = if isnothing(index)
            "external/unattributed"
        else
            frame = allocation.stacktrace[index]
            string(frame.func, " at ", basename(String(frame.file)), ":", frame.line)
        end
        bytes, count = get(groups, key, (0, 0))
        groups[key] = (bytes + allocation.size, count + 1)
    end
    for (key, (bytes, count)) in sort!(collect(groups); by = entry -> last(entry)[1], rev = true)
        @printf("ALLOCATION_SOURCE sampled_bytes=%d sampled_count=%d source=%s\n", bytes, count, key)
    end
    flush(stdout)
    return groups
end

if abspath(PROGRAM_FILE) == @__FILE__
    pressure_allocation_trace()
end
