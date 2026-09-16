using Profile
using Printf

include(joinpath(@__DIR__, "stokes3_performance.jl"))

function _hotspot_options()
    pressure_pruning = Symbol(
        get(ENV, "STOKES3_PROFILE_PRESSURE_TREE_PRUNING", "all"),
    )
    diffusion_pruning = Symbol(
        get(ENV, "STOKES3_PROFILE_DIFFUSION_TREE_PRUNING", "all"),
    )
    return (;
        base_cells = parse(
            Int, get(ENV, "STOKES3_PROFILE_BASE_CELLS", "8"),
        ),
        max_level = parse(
            Int, get(ENV, "STOKES3_PROFILE_MAX_LEVEL", "5"),
        ),
        final_time = parse(
            Float64, get(ENV, "STOKES3_PROFILE_FINAL_TIME", "0.1"),
        ),
        pressure_preconditioner = :tree_geometric,
        diffusion_preconditioner = :tree_geometric,
        pressure_preconditioner_options = (; pruning = pressure_pruning),
        diffusion_preconditioner_options = (; pruning = diffusion_pruning),
        matrix_free_fine = lowercase(
            get(ENV, "STOKES3_PROFILE_MATRIX_FREE_FINE", "false"),
        ) in ("true", "yes", "1"),
    )
end

function _hotspot_case(options)
    case = stokes3_wave(;
        base_cells = options.base_cells,
        max_level = options.max_level,
        final_time = 0.0,
        pressure_preconditioner = options.pressure_preconditioner,
        diffusion_preconditioner = options.diffusion_preconditioner,
        pressure_preconditioner_options =
            options.pressure_preconditioner_options,
        diffusion_preconditioner_options =
            options.diffusion_preconditioner_options,
        matrix_free_fine = options.matrix_free_fine,
    )
    flow = case.flow
    amr_workspace = TP._navier_stokes_amr_workspace(
        flow.velocity, flow.pressure, flow.solver,
    )
    retained_bytes = Base.summarysize(
        (flow.solver, flow.workspace, amr_workspace),
    )
    return (; flow, amr_workspace, retained_bytes, selected_step = Inf)
end

function _hotspot_step!(state, final_time)
    return _profile_stokes3_step!(
        state.flow,
        state.amr_workspace,
        state.selected_step,
        final_time;
        cfl = 0.45,
        rtol = 1.0e-9,
        atol = 1.0e-11,
        max_cycles = 150,
        retained_bytes = state.retained_bytes,
    )
end

function _advance_to_step_before!(state, target_step, final_time)
    for expected_step in 1:(target_step - 1)
        record, selected_step, retained_bytes =
            _hotspot_step!(state, final_time)
        record.step == expected_step || error("unexpected Stokes-3 step")
        record.mesh_changed && error(
            "topology changed before requested profile step $(target_step)",
        )
        state = merge(state, (; selected_step, retained_bytes))
    end
    return state
end

function _warm_first_topology_change(options)
    state = _hotspot_case(options)
    while state.flow.time < options.final_time
        record, selected_step, retained_bytes =
            _hotspot_step!(state, options.final_time)
        println("warmup step=$(record.step), time=$(record.end_time)")
        if record.mesh_changed
            return record.step
        end
        state = merge(state, (; selected_step, retained_bytes))
    end
    error("no topology change occurred before t=$(options.final_time)")
end

function _rebuild_allocations(results)
    selected = filter(results.allocs) do allocation
        any(
            frame -> occursin(
                "_rebuild_topology_dependent_state!", String(frame.func),
            ),
            allocation.stacktrace,
        )
    end
    return Profile.Allocs.AllocResults(selected)
end

function run_stokes3_hotspot_profile()
    options = _hotspot_options()
    prefix = get(ENV, "STOKES3_HOTSPOT_PREFIX", "stokes3-hotspot")
    sample_rate = parse(
        Float64, get(ENV, "STOKES3_HOTSPOT_ALLOCATION_RATE", "0.002"),
    )
    0 < sample_rate <= 1 || throw(
        ArgumentError("allocation sample rate must be in (0, 1]"),
    )

    println("Warming the first default topology change and exact code paths")
    target_step = _warm_first_topology_change(options)
    println("Profiling topology-changing step $(target_step)")

    cpu_state = _advance_to_step_before!(
        _hotspot_case(options), target_step, options.final_time,
    )
    GC.gc()
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.0002)
    cpu_result = Profile.@profile _hotspot_step!(
        cpu_state, options.final_time,
    )
    cpu_record = first(cpu_result)
    cpu_record.mesh_changed || error("profiled step did not change topology")
    cpu_path = "$(prefix)-cpu.txt"
    open(cpu_path, "w") do io
        Profile.print(
            io; format = :flat, C = true, sortedby = :count,
            mincount = 2, threads = 1,
        )
    end

    allocation_state = _advance_to_step_before!(
        _hotspot_case(options), target_step, options.final_time,
    )
    GC.gc()
    Profile.Allocs.clear()
    allocation_result = Profile.Allocs.@profile sample_rate=sample_rate begin
        _hotspot_step!(allocation_state, options.final_time)
    end
    allocation_record = first(allocation_result)
    allocation_record.mesh_changed || error(
        "allocation-profiled step did not change topology",
    )
    rebuild_allocations = _rebuild_allocations(Profile.Allocs.fetch())
    allocation_path = "$(prefix)-allocations.txt"
    open(allocation_path, "w") do io
        Profile.Allocs.print(
            io, rebuild_allocations;
            format = :flat, C = false, sortedby = :count, mincount = 1024,
        )
    end

    rebuild_metrics = allocation_record.stages[:topology_rebuild]
    @printf(
        "Warmed rebuild: %.3f s, %.3f MiB, %d allocations, %.3f s GC, %.3f s compilation\n",
        rebuild_metrics.seconds,
        rebuild_metrics.bytes / 2.0^20,
        rebuild_metrics.allocations,
        rebuild_metrics.gc_seconds,
        rebuild_metrics.compile_seconds,
    )
    println(
        "Sampled $(length(rebuild_allocations.allocs)) rebuild allocations " *
        "at rate $(sample_rate)",
    )
    println("CPU stacks: $(abspath(cpu_path))")
    println("Allocation stacks: $(abspath(allocation_path))")
    return (; cpu_record, allocation_record, rebuild_allocations)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_stokes3_hotspot_profile()
end
