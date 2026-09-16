using Printf
using Statistics: mean, median, quantile

const STOKES3_COMPARISON_STAGES = (
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

struct Stokes3Trace
    path::String
    columns::Dict{String, Int}
    rows::Vector{Vector{String}}
end

function _read_stokes3_trace(path)
    return open(path, "r") do io
        eof(io) && throw(ArgumentError("empty Stokes-3 trace: $path"))
        header = String.(split(readline(io), ','; keepempty = true))
        columns = Dict(name => index for (index, name) in pairs(header))
        rows = Vector{Vector{String}}()
        for line in eachline(io)
            isempty(line) && continue
            row = String.(split(line, ','; keepempty = true))
            length(row) == length(header) || throw(
                ArgumentError(
                    "$path contains $(length(row)) values for " *
                        "$(length(header)) columns",
                ),
            )
            push!(rows, row)
        end
        isempty(rows) && throw(ArgumentError("Stokes-3 trace has no steps: $path"))
        return Stokes3Trace(path, columns, rows)
    end
end

function _trace_value(trace, row, column)
    index = get(trace.columns, column, 0)
    iszero(index) && throw(
        ArgumentError("$(trace.path) does not contain column $column"),
    )
    return row[index]
end

_trace_float(trace, row, column) =
    parse(Float64, _trace_value(trace, row, column))

function _check_identical_mesh_evolution(before, after)
    length(before.rows) == length(after.rows) || throw(
        ArgumentError(
            "step counts differ: $(length(before.rows)) before and " *
                "$(length(after.rows)) after",
        ),
    )
    exact_columns = (
        "step",
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
    )
    for (before_row, after_row) in zip(before.rows, after.rows)
        step = _trace_value(before, before_row, "step")
        for column in exact_columns
            before_value = _trace_value(before, before_row, column)
            after_value = _trace_value(after, after_row, column)
            before_value == after_value || throw(
                ErrorException(
                    "mesh evolution differs at step $step in $column: " *
                        "$before_value before, $after_value after",
                ),
            )
        end
    end
    return nothing
end

function _stage_total(trace, stage, suffix)
    column = "$(stage)_$(suffix)"
    return sum(_trace_float(trace, row, column) for row in trace.rows)
end

function _accounted_step_times(trace)
    return [
        sum(
            _trace_float(trace, row, "$(stage)_seconds") for
                stage in STOKES3_COMPARISON_STAGES
        ) for row in trace.rows
    ]
end

function _changed_step_times(trace, step_times, changed)
    return [
        step_times[index] for index in eachindex(trace.rows) if
            (_trace_value(trace, trace.rows[index], "mesh_changed") == "true") ==
            changed
    ]
end

_mean_or_nan(values) = isempty(values) ? NaN : mean(values)

function _print_comparison_row(label, before, after, unit)
    ratio = iszero(after) ? Inf : before / after
    reduction = iszero(before) ? 0.0 : 100 * (before - after) / before
    @printf(
        "| %s | %.3f %s | %.3f %s | %.2fx | %.1f%% |\n",
        label, before, unit, after, unit, ratio, reduction,
    )
    return nothing
end

function compare_stokes3_performance(before_path, after_path)
    before = _read_stokes3_trace(before_path)
    after = _read_stokes3_trace(after_path)
    _check_identical_mesh_evolution(before, after)

    before_times = _accounted_step_times(before)
    after_times = _accounted_step_times(after)
    before_changed = _changed_step_times(before, before_times, true)
    after_changed = _changed_step_times(after, after_times, true)
    before_unchanged = _changed_step_times(before, before_times, false)
    after_unchanged = _changed_step_times(after, after_times, false)

    time_differences = [
        abs(
            _trace_float(before, before_row, "end_time") -
                _trace_float(after, after_row, "end_time"),
        ) for (before_row, after_row) in zip(before.rows, after.rows)
    ]

    println("# Stokes-3 before/after comparison\n")
    println("Mesh evolution: **identical at every step**.")
    @printf(
        "Compared %d steps with %d topology changes; maximum physical-time difference is %.3e.\n\n",
        length(before.rows), length(before_changed), maximum(time_differences),
    )
    println("| metric | before | after | speedup | reduction |")
    println("|---|---:|---:|---:|---:|")
    _print_comparison_row(
        "accounted time", sum(before_times), sum(after_times), "s",
    )
    _print_comparison_row(
        "median step", 1.0e3median(before_times),
        1.0e3median(after_times), "ms",
    )
    _print_comparison_row(
        "p95 step", 1.0e3quantile(before_times, 0.95),
        1.0e3quantile(after_times, 0.95), "ms",
    )
    _print_comparison_row(
        "mean topology-changing step", 1.0e3 * _mean_or_nan(before_changed),
        1.0e3 * _mean_or_nan(after_changed), "ms",
    )
    _print_comparison_row(
        "mean fixed-topology step", 1.0e3 * _mean_or_nan(before_unchanged),
        1.0e3 * _mean_or_nan(after_unchanged), "ms",
    )
    before_bytes = sum(
        _stage_total(before, stage, "bytes") for
            stage in STOKES3_COMPARISON_STAGES
    ) / 2.0^30
    after_bytes = sum(
        _stage_total(after, stage, "bytes") for
            stage in STOKES3_COMPARISON_STAGES
    ) / 2.0^30
    _print_comparison_row("allocated", before_bytes, after_bytes, "GiB")

    println("\n## Stage totals\n")
    println("| stage | before | after | speedup | reduction |")
    println("|---|---:|---:|---:|---:|")
    for stage in STOKES3_COMPARISON_STAGES
        _print_comparison_row(
            string(stage),
            _stage_total(before, stage, "seconds"),
            _stage_total(after, stage, "seconds"),
            "s",
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error(
        "usage: julia --project=. benchmark/compare_stokes3_performance.jl " *
            "BEFORE.csv AFTER.csv",
    )
    compare_stokes3_performance(ARGS...)
end
