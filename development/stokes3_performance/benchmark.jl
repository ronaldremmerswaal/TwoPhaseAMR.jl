# Profile the default Stokes3 advance to t=0.5, one velocity model at a time.
# Run from the repository root with, for example,
#   julia --project=. development/stokes3_performance/benchmark.jl one
#   julia --project=. development/stokes3_performance/benchmark.jl cmdgfm
# The short run warms compilation. A new flow is then initialized at t=0,
# including its initial AMR and pressure projection, before the measured advance.
# The sampled stage times are estimates, not independent stopwatch measurements.
# Nested SUBSTAGE rows overlap and must not be summed.
using TwoPhaseAMR
using Profile
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

function stack_names(stack, dictionary)
    names = String[]
    for ip in stack
        for frame in get(dictionary, ip, Base.StackTraces.StackFrame[])
            push!(names, string(frame.func))
        end
    end
    return names
end

contains_name(names, needle) = any(name -> occursin(needle, name), names)

function stage(names)
    # Test nested operations before their callers. The first matching stage
    # owns each sample, so the stage rows are disjoint.
    contains_name(names, "_rebuild_two_phase_topology_dependent_state!") &&
        return :topology_rebuild
    contains_name(names, "adapt!") && return :mesh_adaptation
    contains_name(names, "_transport_substep!") && return :transport
    contains_name(names, "_diffusion_substep!") && return :diffusion
    contains_name(names, "_refresh_pressure_substep!") &&
        return :pressure_operator_refresh
    contains_name(names, "_phase_force_substep!") && return :phase_forces
    contains_name(names, "project!") && return :projection
    contains_name(names, "stable_timestep") && return :outer_time_step_bound
    contains_name(names, "_record_diagnostics!") && return :diagnostics
    contains_name(names, "step!") && return :step_overhead
    contains_name(names, "simulate!") && return :simulation_overhead
    return :unclassified
end

function profile_summary(wall)
    data, dictionary = Profile.retrieve(; include_meta = false)
    stacks = Vector{Vector{UInt}}()
    current = UInt[]
    for ip in data
        if iszero(ip)
            isempty(current) || push!(stacks, current)
            current = UInt[]
        else
            push!(current, ip)
        end
    end
    isempty(current) || push!(stacks, current)

    counts = Dict{Symbol, Int}()
    subcounts = Dict{Symbol, Int}()
    for stack in stacks
        names = stack_names(stack, dictionary)
        category = stage(names)
        counts[category] = get(counts, category, 0) + 1
        fragments = if category == :transport
            (
                :wy_advection => "_advect_wy!",
                :phase_face_momentum => "_advect_phase_faces!",
                :interface_reconstruction => "reconstruct!",
                :aperture_freeze => "_freeze_phase_apertures!",
                :inner_cfl => "stable_timestep",
            )
        elseif category == :projection
            (
                :cmdgfm_assembly => "_assemble_cmdgfm_system!",
                :cmdgfm_preconditioner => "_cmdgfm_preconditioner",
                :cmdgfm_krylov => "bicgstab!",
                :elliptic_solve => "solve!",
            )
        elseif category == :pressure_operator_refresh
            (:multigrid_refresh => "_refresh_geometric_multigrid!",)
        elseif category == :outer_time_step_bound
            (
                :cfl_reconstruction => "reconstruct!",
                :cfl_aperture_freeze => "_freeze_phase_apertures!",
                :frozen_cfl_scan => "_frozen_interface_stable_timestep",
            )
        elseif category == :diagnostics
            (
                :diagnostic_face_volumes => "_diagnostic_face_volumes",
                :diagnostic_face_fractions => "_staggered_liquid_fraction",
                :diagnostic_mixture_velocity => "_mass_weighted_one_velocity!",
                :diagnostic_divergence => "divergence!",
            )
        else
            ()
        end
        for (label, fragment) in fragments
            contains_name(names, fragment) || continue
            subcounts[label] = get(subcounts, label, 0) + 1
        end
    end

    total = length(stacks)
    active = total - get(counts, :unclassified, 0)
    println("PROFILE_SAMPLES total=", total, " active=", active,
        " unclassified=", get(counts, :unclassified, 0))
    active == 0 && return
    for (name, count) in sort!(collect(counts); by = last, rev = true)
        name == :unclassified && continue
        @printf("STAGE %-28s %7d %6.2f%% %8.3fs\n",
            name, count, 100 * count / active, wall * count / active)
    end
    for (name, count) in sort!(collect(subcounts); by = last, rev = true)
        @printf("SUBSTAGE %-25s %7d %6.2f%% %8.3fs\n",
            name, count, 100 * count / active, wall * count / active)
    end
end

function benchmark(mode::Symbol; final_time = 0.5, warmup_time = 0.05)
    mode in (:one, :cmdgfm) ||
        throw(ArgumentError("mode must be 'one' or 'cmdgfm'"))
    model = mode == :one ? OneVelocityModel() :
        TwoVelocityModel(pressure_projection = CMDGFMProjection())
    @printf("ENV mode=%s julia=%s julia_threads=%d blas_threads=%d max_level=5 final_time=%.3f\n",
        mode, VERSION, Threads.nthreads(), BLAS.get_num_threads(), final_time)
    warm = @timed run_example(
        velocity_model = model, final_time = warmup_time,
    )
    @printf("WARMUP mode=%s wall=%.3f compile=%.3f gc=%.3f steps=%d updates=%d\n",
        mode, warm.time, warm.compile_time, warm.gctime,
        warm.value.result.steps, warm.value.result.mesh_updates)
    flush(stdout)

    setup = @timed run_example(velocity_model = model, final_time = 0.0)
    flow = setup.value.flow
    @printf("SETUP mode=%s wall=%.3f compile=%.3f gc=%.3f blocks=%d faces=%d\n",
        mode, setup.time, setup.compile_time, setup.gctime,
        length(flow.solver.mesh.blocks), length(flow.solver.mesh.faces))
    flush(stdout)

    Profile.init(; n = 4_000_000, delay = 0.004)
    Profile.clear()
    advance = nothing
    Profile.@profile advance = @timed simulate!(flow, final_time)
    result = advance.value
    # This is only the final projection, not an iteration distribution.
    pressure_iterations = hasproperty(result.pressure, :iterations) ?
        result.pressure.iterations : result.pressure.cycles
    @printf("ADVANCE mode=%s wall=%.3f compile=%.3f gc=%.3f bytes=%d steps=%d updates=%d blocks=%d faces=%d final_pressure_iterations=%d\n",
        mode, advance.time, advance.compile_time, advance.gctime,
        advance.bytes, result.steps, result.mesh_updates,
        length(flow.solver.mesh.blocks), length(flow.solver.mesh.faces),
        pressure_iterations)
    if mode == :cmdgfm
        statistics = result.pressure
        @printf("CMDGFM_FINAL pressure_unknowns=%d jump_unknowns=%d pressure_hierarchy_reused=%s jump_factor_reused=%s\n",
            statistics.pressure_unknowns, statistics.jump_unknowns,
            statistics.pressure_hierarchy_reused,
            statistics.jump_factor_reused)
    end
    flush(stdout)
    profile_summary(advance.time)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(Symbol(get(ARGS, 1, "one")))
end
