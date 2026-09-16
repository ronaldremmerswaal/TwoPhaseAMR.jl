using Printf
using Statistics: mean, median, quantile

include(joinpath(@__DIR__, "compare_stokes3_performance.jl"))

const STOKES3_EXACT_WORK_COLUMNS = (
    "step",
    "start_time",
    "end_time",
    "dt",
    "blocks_before",
    "blocks_after",
    "cells_after",
    "faces_after",
    "maximum_level",
    "mesh_keys",
    "amr_attempted",
    "mesh_changed",
    "refined_blocks",
    "coarsened_families",
    "pressure_iterations",
    "diffusion_iterations",
    "pressure_reused_levels",
    "diffusion_reused_levels",
)

function _requested_stokes3_threads()
    raw = get(ENV, "STOKES3_THREAD_COUNTS", "1,2,4,8")
    counts = unique(sort(parse.(Int, split(raw, ','))))
    all(>(0), counts) || error("thread counts must be positive")
    first(counts) == 1 || error("STOKES3_THREAD_COUNTS must include 1")
    return counts
end

function _check_identical_stokes3_work(reference, candidate)
    length(reference.rows) == length(candidate.rows) || error(
        "step counts differ: $(length(reference.rows)) and " *
            "$(length(candidate.rows))",
    )
    for (reference_row, candidate_row) in zip(reference.rows, candidate.rows)
        step = _trace_value(reference, reference_row, "step")
        for column in STOKES3_EXACT_WORK_COLUMNS
            reference_value = _trace_value(reference, reference_row, column)
            candidate_value = _trace_value(candidate, candidate_row, column)
            reference_value == candidate_value || error(
                "thread-count runs differ at step $step in $column: " *
                    "$reference_value != $candidate_value",
            )
        end
    end
    return nothing
end

function _trace_total(trace, suffix)
    return sum(
        _stage_total(trace, stage, suffix) for
            stage in STOKES3_COMPARISON_STAGES
    )
end

function _stokes3_scaling_metrics(trace)
    times = _accounted_step_times(trace)
    changing = _changed_step_times(trace, times, true)
    fixed = _changed_step_times(trace, times, false)
    return (
        steps = length(times),
        topology_changes = length(changing),
        seconds = sum(times),
        wall_seconds = sum(
            _trace_float(trace, row, "wall_seconds") for row in trace.rows
        ),
        median_step = median(times),
        p95_step = quantile(times, 0.95),
        mean_changing_step = mean(changing),
        mean_fixed_step = mean(fixed),
        bytes = _trace_total(trace, "bytes"),
        allocations = _trace_total(trace, "allocations"),
        gc_seconds = _trace_total(trace, "gc_seconds"),
        compile_seconds = _trace_total(trace, "compile_seconds"),
    )
end

function _write_stokes3_scaling_summary(path, traces, counts)
    metrics = Dict(count => _stokes3_scaling_metrics(traces[count]) for count in counts)
    baseline = metrics[1]
    open(path, "w") do io
        println(
            io,
            "threads,steps,topology_changes,seconds,speedup,efficiency," *
                "wall_seconds,median_step_seconds,p95_step_seconds," *
                "mean_changing_step_seconds,mean_fixed_step_seconds,bytes," *
                "allocations,gc_seconds,compile_seconds",
        )
        for count in counts
            item = metrics[count]
            speedup = baseline.seconds / item.seconds
            println(
                io,
                count, ',', item.steps, ',', item.topology_changes, ',',
                item.seconds, ',', speedup, ',', speedup / count, ',',
                item.wall_seconds, ',', item.median_step, ',', item.p95_step, ',',
                item.mean_changing_step, ',', item.mean_fixed_step, ',',
                item.bytes, ',', item.allocations, ',', item.gc_seconds, ',',
                item.compile_seconds,
            )
        end
    end
    return metrics
end

function _write_stokes3_stage_scaling(path, traces, counts)
    baseline = traces[1]
    open(path, "w") do io
        println(io, "stage,threads,seconds,speedup,bytes,allocations")
        for stage in STOKES3_COMPARISON_STAGES
            reference_seconds = _stage_total(baseline, stage, "seconds")
            for count in counts
                trace = traces[count]
                seconds = _stage_total(trace, stage, "seconds")
                println(
                    io, stage, ',', count, ',', seconds, ',',
                    reference_seconds / seconds, ',',
                    round(Int, _stage_total(trace, stage, "bytes")), ',',
                    round(Int, _stage_total(trace, stage, "allocations")),
                )
            end
        end
    end
    return path
end

function _print_stokes3_scaling(metrics, traces, counts)
    baseline = metrics[1]
    println("# Stokes-3 thread scaling\n")
    println(
        "Every thread count executed the same time steps, AMR decisions, " *
            "block-key sequence, and solver iteration history.\n",
    )
    println("| threads | accounted time | speedup | median step | fixed topology | changing topology | allocated |")
    println("|---:|---:|---:|---:|---:|---:|---:|")
    for count in counts
        item = metrics[count]
        @printf(
            "| %d | %.3f s | %.2fx | %.3f ms | %.3f ms | %.3f ms | %.3f GiB |\n",
            count, item.seconds, baseline.seconds / item.seconds,
            1.0e3item.median_step, 1.0e3item.mean_fixed_step,
            1.0e3item.mean_changing_step, item.bytes / 2.0^30,
        )
    end

    println("\n## Stage speedup relative to one thread\n")
    print("| stage | one-thread time |")
    for count in Iterators.drop(counts, 1)
        print(" $(count)T |")
    end
    println()
    print("|---|---:|")
    for _ in Iterators.drop(counts, 1)
        print("---:|")
    end
    println()
    for stage in STOKES3_COMPARISON_STAGES
        reference = _stage_total(traces[1], stage, "seconds")
        @printf("| %s | %.3f s |", stage, reference)
        for count in Iterators.drop(counts, 1)
            seconds = _stage_total(traces[count], stage, "seconds")
            @printf(" %.2fx |", reference / seconds)
        end
        println()
    end
    return nothing
end

function run_stokes3_thread_scaling(;
        output_directory = get(
            ENV,
            "STOKES3_THREAD_OUTPUT",
            joinpath(tempdir(), "stokes3-thread-scaling"),
        ),
        counts = _requested_stokes3_threads(),
        reuse = lowercase(
            get(ENV, "STOKES3_THREAD_REUSE_RESULTS", "false"),
        ) in ("1", "true", "yes"),
    )
    mkpath(output_directory)
    project = dirname(@__DIR__)
    worker = joinpath(@__DIR__, "stokes3_performance.jl")
    traces = Dict{Int, Stokes3Trace}()
    for count in counts
        output = joinpath(output_directory, "threads-$count.csv")
        if reuse && isfile(output)
            println("Reusing ", output)
        else
            command = `$(Base.julia_cmd()) --project=$project --threads=$count $worker`
            environment = copy(ENV)
            environment["OPENBLAS_NUM_THREADS"] = "1"
            environment["STOKES3_PROFILE_CSV"] = output
            environment["STOKES3_PROFILE_FINAL_TIME"] = get(
                ENV, "STOKES3_PROFILE_FINAL_TIME", "1.0",
            )
            println("Running Stokes-3 with ", count, " Julia thread(s)")
            run(setenv(command, environment))
        end
        traces[count] = _read_stokes3_trace(output)
    end

    reference = traces[1]
    for count in Iterators.drop(counts, 1)
        _check_identical_stokes3_work(reference, traces[count])
    end
    summary = joinpath(output_directory, "scaling.csv")
    stages = joinpath(output_directory, "stage-scaling.csv")
    metrics = _write_stokes3_scaling_summary(summary, traces, counts)
    _write_stokes3_stage_scaling(stages, traces, counts)
    _print_stokes3_scaling(metrics, traces, counts)
    println("\nWrote ", summary)
    println("Wrote ", stages)
    return (; traces, metrics, summary, stages)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_stokes3_thread_scaling()
end
