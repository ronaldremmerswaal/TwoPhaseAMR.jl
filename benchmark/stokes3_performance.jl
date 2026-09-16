using LinearAlgebra: mul!
using Printf
using Statistics: mean, median, quantile
using TwoPhaseAMR

import TwoPhaseAMR as TP

include(joinpath(@__DIR__, "..", "examples", "stokes3_wave.jl"))

const STOKES3_STAGES = (
    :timestep_bound,
    :state_copy,
    :advection,
    :body_force,
    :viscosity_operator,
    :mixture_density,
    :diffusion_solver_refresh,
    :diffusion_solve,
    :pressure_operator,
    :pressure_preconditioner,
    :surface_tension,
    :gravity,
    :pressure_projection,
    :amr_indicators,
    :amr_transfer,
    :topology_rebuild,
)

struct Stokes3StageMetrics
    seconds::Float64
    bytes::Int
    allocations::Int
    gc_seconds::Float64
    compile_seconds::Float64
end

Stokes3StageMetrics() = Stokes3StageMetrics(0.0, 0, 0, 0.0, 0.0)

Base.:+(a::Stokes3StageMetrics, b::Stokes3StageMetrics) =
    Stokes3StageMetrics(
    a.seconds + b.seconds,
    a.bytes + b.bytes,
    a.allocations + b.allocations,
    a.gc_seconds + b.gc_seconds,
    a.compile_seconds + b.compile_seconds,
)

function _measure_stokes3_stage(function_to_measure)
    measurement = @timed function_to_measure()
    metrics = Stokes3StageMetrics(
        measurement.time,
        measurement.bytes,
        Base.gc_alloc_count(measurement.gcstats),
        measurement.gctime,
        measurement.compile_time,
    )
    return measurement.value, metrics
end

struct Stokes3StepRecord
    step::Int
    start_time::Float64
    end_time::Float64
    dt::Float64
    blocks_before::Int
    blocks_after::Int
    cells_after::Int
    faces_after::Int
    maximum_level::Int
    mesh_keys::String
    amr_attempted::Bool
    mesh_changed::Bool
    refined_blocks::Int
    coarsened_families::Int
    pressure_iterations::Int
    diffusion_iterations::Int
    retained_bytes::Int
    wall_seconds::Float64
    stages::Dict{Symbol, Stokes3StageMetrics}
end

"""Canonical block-key trace used to prove identical AMR evolution."""
function _stokes3_mesh_keys(mesh)
    keys = sort!(
        [(block.key.level, block.key.i, block.key.j) for block in mesh.blocks],
    )
    return join(("$(level):$(i):$(j)" for (level, i, j) in keys), ';')
end

@inline function _record_stage!(function_to_measure, stages, name)
    value, metrics = _measure_stokes3_stage(function_to_measure)
    stages[name] = metrics
    return value
end

function _refresh_stokes3_pressure_operator!(solver, model)
    pressure_solver = solver.pressure_solver
    pressure_solver.operator = PoissonOperator(
        model.interface;
        density = model.density,
        formulation = model.formulation,
        template = pressure_solver.operator,
    )
    return pressure_solver.operator
end

function _refresh_stokes3_pressure_preconditioner!(solver, model)
    pressure_solver = solver.pressure_solver
    specification = solver.pressure_preconditioner
    variable_density = model.density[1] != model.density[2]
    refreshable = variable_density && (
        specification isa Symbol ||
            applicable(specification, pressure_solver.operator.matrix)
    )
    if refreshable
        if specification === :geometric &&
                pressure_solver.preconditioner isa
                TP.GeometricMultigridPreconditioner
            TP._refresh_geometric_multigrid!(
                pressure_solver.preconditioner,
                pressure_solver.operator,
            )
        else
            pressure_solver.preconditioner =
                TP._construct_pressure_preconditioner(
                specification,
                pressure_solver.operator;
                coarse_size = solver.coarse_size,
                pre_sweeps = solver.pre_sweeps,
                post_sweeps = solver.post_sweeps,
                max_levels = solver.max_levels,
                options = solver.pressure_preconditioner_options,
            )
        end
        pressure_solver.stats.preconditioner =
            TP._preconditioner_name(pressure_solver.preconditioner)
    end
    return pressure_solver.preconditioner
end

function _solve_stokes3_diffusion!(flow, dt; rtol, atol, max_cycles)
    workspace = flow.workspace
    operator = workspace.two_phase.diffusion
    solver = workspace.two_phase.diffusion_solver
    old_velocity = workspace.old_velocity
    source = workspace.convected
    mul!(solver.product, operator.stiffness, old_velocity.interior)
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] =
            solver.inertia[index] * source.interior[index] -
            dt * solver.product[index] / 2
    end
    stats = TP._solve_variable_diffusion!(
        flow.velocity,
        solver,
        operator;
        rtol,
        atol,
        max_cycles,
    )
    stats.converged || error(
        "variable diffusion did not converge in $(stats.cycles) iterations"
    )
    return stats
end

function _profile_stokes3_step!(
        flow,
        amr_workspace,
        selected_step,
        final_time;
        cfl,
        rtol,
        atol,
        max_cycles,
        retained_bytes,
    )
    solver = flow.solver
    workspace = flow.workspace
    model = solver.two_phase
    isnothing(model) && error("the Stokes-3 benchmark requires two phases")
    stages = Dict(stage => Stokes3StageMetrics() for stage in STOKES3_STAGES)
    blocks_before = nblocks(solver.mesh)
    start_time = flow.time
    wall_start = time_ns()

    bound = _record_stage!(stages, :timestep_bound) do
        stable_timestep(flow; cfl)
    end
    selected_step = min(selected_step, bound)
    dt = min(selected_step, final_time - start_time)
    dt > 0 || error("the profiled time step must be positive")

    _record_stage!(stages, :state_copy) do
        TP._copy_velocity!(workspace.old_velocity, flow.velocity)
        enforce_no_permeation!(workspace.old_velocity)
    end
    _record_stage!(stages, :advection) do
        TP._transport_substep!(
            flow.velocity,
            solver,
            workspace,
            dt,
            model,
        )
    end
    _record_stage!(stages, :body_force) do
        midpoint = start_time + dt / 2
        for (index, face) in pairs(solver.mesh.faces)
            @inbounds workspace.convected.interior[index] +=
                dt * TP._body_force(nothing, face, midpoint)
        end
        fill!(workspace.convected.boundary, 0)
    end

    diffusion_stats = nothing
    if all(iszero, model.viscosity)
        _record_stage!(stages, :diffusion_solve) do
            TP._copy_velocity!(flow.velocity, workspace.convected)
        end
    else
        _record_stage!(stages, :viscosity_operator) do
            TP._update_dynamic_viscosity!(
                workspace.two_phase.diffusion,
                model.interface,
                model.viscosity,
            )
        end
        density = _record_stage!(stages, :mixture_density) do
            TP._mixture_density!(
                workspace.two_phase.density,
                model.interface,
                model.density,
            )
        end
        _record_stage!(stages, :diffusion_solver_refresh) do
            TP._update_variable_diffusion_solver!(
                workspace.two_phase.diffusion_solver,
                workspace.two_phase.diffusion,
                density,
                dt,
            )
        end
        diffusion_stats = _record_stage!(stages, :diffusion_solve) do
            _solve_stokes3_diffusion!(
                flow,
                dt;
                rtol,
                atol,
                max_cycles,
            )
        end
    end

    _record_stage!(stages, :pressure_operator) do
        _refresh_stokes3_pressure_operator!(solver, model)
    end
    _record_stage!(stages, :pressure_preconditioner) do
        _refresh_stokes3_pressure_preconditioner!(solver, model)
    end
    if model.sigma > 0
        _record_stage!(stages, :surface_tension) do
            surface_tension!(
                workspace.two_phase.acceleration,
                model.interface,
                solver.pressure_solver.operator,
                workspace.projection.elliptic;
                sigma = model.sigma,
                formulation = model.formulation,
            )
            for index in eachindex(flow.velocity.interior)
                @inbounds flow.velocity.interior[index] +=
                    dt * workspace.two_phase.acceleration.interior[index]
            end
        end
    end
    if any(!iszero, model.gravity)
        _record_stage!(stages, :gravity) do
            gravity!(
                workspace.two_phase.acceleration,
                model.interface,
                solver.pressure_solver.operator,
                workspace.projection.elliptic;
                density = model.density,
                gravity = model.gravity,
            )
            for index in eachindex(flow.velocity.interior)
                @inbounds flow.velocity.interior[index] +=
                    dt * workspace.two_phase.acceleration.interior[index]
            end
        end
    end
    pressure_stats = _record_stage!(stages, :pressure_projection) do
        enforce_no_permeation!(flow.velocity)
        project!(
            flow.velocity,
            flow.pressure,
            solver.pressure_solver,
            dt,
            workspace.projection;
            rtol,
            atol,
            max_cycles,
        )
    end
    pressure_stats.converged || error(
        "pressure projection did not converge in " *
            "$(pressure_stats.cycles) iterations"
    )

    flow.time = min(final_time, start_time + dt)
    flow.steps += 1
    solver.mesh.adaptation.completed_steps += 1
    labels = _record_stage!(stages, :amr_indicators) do
        refinement_labels!(
            amr_workspace,
            solver.refinement_indicators;
            time = flow.time,
        )
    end
    amr_stats = _record_stage!(stages, :amr_transfer) do
        adapt!(
            amr_workspace,
            labels;
            step = solver.mesh.adaptation.completed_steps,
        )
    end
    if amr_stats.changed
        flow.mesh_updates += 1
        _record_stage!(stages, :topology_rebuild) do
            TP._rebuild_topology_dependent_state!(
                solver,
                workspace;
                dt,
            )
        end
        retained_bytes = Base.summarysize((solver, workspace, amr_workspace))
    end

    wall_seconds = (time_ns() - wall_start) / 1.0e9
    diffusion_iterations = isnothing(diffusion_stats) ? 0 :
        diffusion_stats.cycles
    maximum_level = maximum(block.key.level for block in solver.mesh.blocks)
    record = Stokes3StepRecord(
        flow.steps,
        Float64(start_time),
        Float64(flow.time),
        Float64(dt),
        blocks_before,
        nblocks(solver.mesh),
        ncells(solver.mesh),
        length(solver.mesh.faces),
        maximum_level,
        _stokes3_mesh_keys(solver.mesh),
        amr_stats.attempted,
        amr_stats.changed,
        amr_stats.refined_blocks,
        amr_stats.coarsened_families,
        pressure_stats.cycles,
        diffusion_iterations,
        retained_bytes,
        wall_seconds,
        stages,
    )
    return record, selected_step, retained_bytes
end

function _profiled_stokes3_run!(
        flow,
        final_time;
        cfl = 0.45,
        rtol = 1.0e-9,
        atol = 1.0e-11,
        max_cycles = 150,
    )
    final_time > flow.time || error("final_time must exceed the flow time")
    amr_workspace = TP._navier_stokes_amr_workspace(
        flow.velocity,
        flow.pressure,
        flow.solver,
    )
    retained_bytes = Base.summarysize(
        (flow.solver, flow.workspace, amr_workspace)
    )
    selected_step = Inf
    records = Stokes3StepRecord[]
    tolerance = 16eps(eltype(flow.velocity)) * max(1, abs(final_time))
    while final_time - flow.time > tolerance
        record, selected_step, retained_bytes = _profile_stokes3_step!(
            flow,
            amr_workspace,
            selected_step,
            final_time;
            cfl,
            rtol,
            atol,
            max_cycles,
            retained_bytes,
        )
        push!(records, record)
        println("step=$(record.step), time=$(record.end_time)")
    end
    return records
end

function _stage_total(records, stage)
    return mapreduce(
        record -> record.stages[stage],
        +,
        records;
        init = Stokes3StageMetrics(),
    )
end

@inline _record_accounted_time(record) =
    sum(record.stages[stage].seconds for stage in STOKES3_STAGES)

function _print_stokes3_summary(records, setup, setup_retained_bytes)
    accounted = sum(_record_accounted_time, records)
    wall = sum(record.wall_seconds for record in records)
    compile_time = sum(
        record.stages[stage].compile_seconds
            for record in records for stage in STOKES3_STAGES
    )
    changed = filter(record -> record.mesh_changed, records)
    unchanged = filter(record -> !record.mesh_changed, records)
    step_times = [_record_accounted_time(record) for record in records]

    println("\nStokes-3 initialization")
    println("| metric | value |")
    println("|---|---:|")
    @printf("| warmed setup time | %.3f s |\n", setup.time)
    @printf("| setup allocated | %.3f GiB |\n", setup.bytes / 2.0^30)
    println("| setup allocations | $(Base.gc_alloc_count(setup.gcstats)) |")
    @printf("| setup GC time | %.3f s |\n", setup.gctime)
    @printf(
        "| retained solver/workspace state | %.3f MiB |\n",
        setup_retained_bytes / 2.0^20,
    )

    println("\nPer-step totals")
    println("| metric | value |")
    println("|---|---:|")
    println("| steps | $(length(records)) |")
    println("| topology changes | $(length(changed)) |")
    @printf("| measured stage time | %.3f s |\n", accounted)
    @printf("| instrumented wall time | %.3f s |\n", wall)
    @printf("| first-use compilation inside steps | %.3f s |\n", compile_time)
    @printf("| median step | %.3f ms |\n", 1.0e3median(step_times))
    @printf("| p95 step | %.3f ms |\n", 1.0e3quantile(step_times, 0.95))
    @printf("| maximum step | %.3f ms |\n", 1.0e3maximum(step_times))
    if !isempty(changed)
        @printf(
            "| mean step with topology change | %.3f ms |\n",
            1.0e3mean(_record_accounted_time(record) for record in changed),
        )
    end
    if !isempty(unchanged)
        @printf(
            "| mean step without topology change | %.3f ms |\n",
            1.0e3mean(_record_accounted_time(record) for record in unchanged),
        )
    end

    println("\nStage breakdown")
    println("| stage | total (s) | share | per step (ms) | allocated (GiB) | allocations | GC (s) | compile (s) |")
    println("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage in STOKES3_STAGES
        total = _stage_total(records, stage)
        share = iszero(accounted) ? 0.0 : 100total.seconds / accounted
        @printf(
            "| %s | %.3f | %.1f%% | %.3f | %.3f | %d | %.3f | %.3f |\n",
            string(stage),
            total.seconds,
            share,
            1.0e3total.seconds / length(records),
            total.bytes / 2.0^30,
            total.allocations,
            total.gc_seconds,
            total.compile_seconds,
        )
    end

    println("\nPer-step trace")
    println("| step | t | dt | blocks | cells | changed | refine/coarsen | step (ms) | allocated (MiB) | pressure/diffusion iterations |")
    println("|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|")
    for record in records
        bytes = sum(record.stages[stage].bytes for stage in STOKES3_STAGES)
        @printf(
            "| %d | %.6f | %.3e | %d | %d | %s | %d/%d | %.3f | %.3f | %d/%d |\n",
            record.step,
            record.end_time,
            record.dt,
            record.blocks_after,
            record.cells_after,
            record.mesh_changed ? "yes" : "no",
            record.refined_blocks,
            record.coarsened_families,
            1.0e3 * _record_accounted_time(record),
            bytes / 2.0^20,
            record.pressure_iterations,
            record.diffusion_iterations,
        )
    end
    return nothing
end

function _write_stokes3_csv(path, records)
    isempty(path) && return nothing
    open(path, "w") do io
        fixed = [
            "step", "start_time", "end_time", "dt", "blocks_before",
            "blocks_after", "cells_after", "faces_after", "maximum_level",
            "mesh_keys", "amr_attempted", "mesh_changed", "refined_blocks",
            "coarsened_families", "pressure_iterations",
            "diffusion_iterations", "retained_bytes", "wall_seconds",
        ]
        stage_columns = String[]
        for stage in STOKES3_STAGES
            append!(
                stage_columns,
                [
                    "$(stage)_seconds",
                    "$(stage)_bytes",
                    "$(stage)_allocations",
                    "$(stage)_gc_seconds",
                    "$(stage)_compile_seconds",
                ],
            )
        end
        println(io, join((fixed..., stage_columns...), ','))
        for record in records
            values = Any[
                record.step,
                record.start_time,
                record.end_time,
                record.dt,
                record.blocks_before,
                record.blocks_after,
                record.cells_after,
                record.faces_after,
                record.maximum_level,
                record.mesh_keys,
                record.amr_attempted,
                record.mesh_changed,
                record.refined_blocks,
                record.coarsened_families,
                record.pressure_iterations,
                record.diffusion_iterations,
                record.retained_bytes,
                record.wall_seconds,
            ]
            for stage in STOKES3_STAGES
                metrics = record.stages[stage]
                append!(
                    values,
                    (
                        metrics.seconds,
                        metrics.bytes,
                        metrics.allocations,
                        metrics.gc_seconds,
                        metrics.compile_seconds,
                    ),
                )
            end
            println(io, join(values, ','))
        end
    end
    return path
end

function _validate_stokes3_profiler!()
    options = (
        base_cells = 4,
        max_level = 0,
        final_time = 0.0,
        cells_per_radius = 2,
    )
    production = stokes3_wave(; options...)
    instrumented = stokes3_wave(; options...)
    final_time = 1.0e-5
    simulate!(
        production.flow,
        final_time;
        rtol = 1.0e-9,
        atol = 1.0e-11,
        max_cycles = 150,
    )
    _profiled_stokes3_run!(instrumented.flow, final_time)
    production.flow.velocity.interior ==
        instrumented.flow.velocity.interior ||
        error("instrumented velocity differs from production step!")
    production.flow.velocity.boundary ==
        instrumented.flow.velocity.boundary ||
        error("instrumented boundary velocity differs from production step!")
    all(
        first == second for (first, second) in zip(
                production.flow.pressure.blocks,
                instrumented.flow.pressure.blocks,
            )
    ) || error("instrumented pressure differs from production step!")
    all(
        first == second for (first, second) in zip(
                production.flow.interface.fraction.blocks,
                instrumented.flow.interface.fraction.blocks,
            )
    ) || error("instrumented interface differs from production step!")
    println("Profiler validation: production state reproduced bit for bit")
    return instrumented
end

function run_stokes3_performance_benchmark(;
        final_time = parse(
            Float64,
            get(ENV, "STOKES3_PROFILE_FINAL_TIME", "1.0"),
        ),
        base_cells = parse(
            Int,
            get(ENV, "STOKES3_PROFILE_BASE_CELLS", "8"),
        ),
        max_level = parse(
            Int,
            get(ENV, "STOKES3_PROFILE_MAX_LEVEL", "5"),
        ),
        csv_path = get(ENV, "STOKES3_PROFILE_CSV", ""),
    )
    final_time > 0 || throw(ArgumentError("final_time must be positive"))
    base_cells > 0 || throw(ArgumentError("base_cells must be positive"))
    max_level >= 0 || throw(ArgumentError("max_level must be nonnegative"))
    println("Julia $(VERSION), threads=$(Threads.nthreads())")
    println("CPU: $(Sys.cpu_info()[1].model)")
    println(
        "Stokes-3 configuration: base=$(base_cells)x$(base_cells), " *
            "max_level=$(max_level), final_time=$(final_time)",
    )

    # Compile the setup and all physical substeps on a small mesh before
    # measuring the requested configuration, and guard against drift between
    # this decomposed path and the production time step.
    _validate_stokes3_profiler!()
    GC.gc()

    setup = @timed stokes3_wave(
        base_cells = base_cells,
        max_level = max_level,
        final_time = 0.0,
    )
    case = setup.value
    setup_retained_bytes = Base.summarysize(
        (case.flow.solver, case.flow.workspace)
    )
    records = _profiled_stokes3_run!(case.flow, final_time)
    _print_stokes3_summary(records, setup, setup_retained_bytes)
    written = _write_stokes3_csv(csv_path, records)
    isnothing(written) || println("\nWrote per-step CSV to $(written)")
    return (; case, setup, records)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_stokes3_performance_benchmark()
end
