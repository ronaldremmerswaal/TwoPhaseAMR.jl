# Summarize bounded, instrumented Stage 2 component profiles.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))
using CairoMakie
using Printf

function collect_profiles(standard, affine)
    rows = Any[]
    for size in (2048, 4096), workers in (1, 4, 8)
        path = joinpath(standard, "n$size-p$workers.toml")
        sample = TOML.parsefile(path)
        (sample["nx"], sample["ny"], sample["default_threads"]) ==
            (size, size, workers) || error("standard profile metadata mismatch: $path")
        sample["full_diagnostics"]["max_state_delta"] == 0 &&
            sample["pass_diagnostics"]["max_state_delta"] == 0 ||
            error("standard profile numerical parity failed: $path")
        passes = sample["passes"]
        [item["name"] for item in passes] == ["x_half_1", "y_full", "x_half_2"] ||
            error("unexpected standard sweep names: $path")
        split_total = sum(item["median_seconds"] for item in passes)
        split_total > 0 || error("nonpositive split time: $path")
        push!(rows, Dict(
            "implementation" => "standard", "mode" => "observable_step",
            "size" => size, "workers" => workers, "steps" => sample["steps"],
            "complete_diagnostic_median_seconds" => sample["full_median_seconds"],
            "split_median_seconds" => split_total,
            "component_fractions" => Dict(item["name"] => item["median_seconds"] / split_total
                                          for item in passes),
            "complete_allocated_bytes" => sample["full_allocated_bytes"],
            "active_worker_ids" => sample["observed_worker_ids_untimed"],
            "working_array_bytes" => sample["working_array_bytes"],
            "numerical_delta" => sample["pass_diagnostics"]["max_state_delta"],
            "source" => path,
            "placement_scope" => "process affinity mask; no per-worker pin verification",
        ))
    end
    for workers in (1, 4, 8)
        path = joinpath(affine, "n2048-p$workers", "profile.toml")
        sample = TOML.parsefile(path)
        (sample["nx"], sample["ny"], sample["default_workers"]) ==
            (2048, 2048, workers) || error("affine profile metadata mismatch: $path")
        for (name, item) in sample["modes"]
            item["complete_checks"]["max_state_delta"] == 0 &&
                item["split_checks"]["max_state_delta"] == 0 ||
                error("affine profile numerical parity failed: $path")
            item["worker_probe"]["active_worker_ids"] == collect(1:workers) ||
                error("affine profile worker participation failed: $path")
            fractions = Dict(key => item["diagnostic_$(key)_fraction"]
                             for key in ("compute", "copyback", "validation"))
            push!(rows, Dict(
                "implementation" => "affine", "mode" => name, "size" => 2048,
                "workers" => workers, "steps" => sample["steps"],
                "complete_diagnostic_median_seconds" => item["complete_median_seconds"],
                "split_median_seconds" => item["split_clock_median"]["elapsed_seconds"],
                "component_fractions" => fractions,
                "complete_allocated_bytes" => median(item["complete_allocated_bytes"]),
                "active_worker_ids" => item["worker_probe"]["active_worker_ids"],
                "working_array_bytes" => item["numeric_array_bytes"],
                "numerical_delta" => item["split_checks"]["max_state_delta"],
                "source" => path,
                "placement_scope" => "verified per-worker Linux pinning",
                "diagnostic_halo_duplicate_update_fraction" =>
                    item["analytic_halo_duplicate_update_fraction"],
                "complete_gc_seconds" => item["complete_gc_seconds"],
            ))
        end
    end
    return rows
end

function plot_profiles(rows, destination)
    panels = [
        ("Standard 2048²", [r for r in rows if r["implementation"] == "standard" && r["size"] == 2048],
         [("x_half_1", :steelblue), ("y_full", :seagreen), ("x_half_2", :darkorange)]),
        ("Standard 4096²", [r for r in rows if r["implementation"] == "standard" && r["size"] == 4096],
         [("x_half_1", :steelblue), ("y_full", :seagreen), ("x_half_2", :darkorange)]),
        ("Affine 2048²", [r for r in rows if r["implementation"] == "affine"],
         [("compute", :steelblue), ("copyback", :darkorange), ("validation", :gray)]),
    ]
    figure = Figure(size=(1500, 520))
    for (panel_index, (title, items, components)) in enumerate(panels)
        labels = [r["implementation"] == "standard" ? "p$(r["workers"])" :
                  "p$(r["workers"]) $(first(r["mode"]))" for r in items]
        axis = Axis(figure[1, panel_index], title=title,
                    xlabel="Fraction of instrumented split trajectory",
                    yticks=(collect(1:length(labels)), labels))
        xlims!(axis, 0, 1.03)
        for (item_index, row) in enumerate(items)
            left = 0.0
            for (name, color) in components
                width = get(row["component_fractions"], name, 0.0)
                polygon = Point2f[(left, item_index - 0.35), (left + width, item_index - 0.35),
                                  (left + width, item_index + 0.35), (left, item_index + 0.35)]
                poly!(axis, polygon; color, label=name)
                left += width
            end
        end
        axislegend(axis; position=:rb)
    end
    Label(figure[0, :], "Diagnostic component clocks; not acceptance timing or worker busy time")
    for extension in ("svg", "png")
        save(joinpath(destination, "stage2-component-fractions.$extension"), figure)
    end
end

function main(args=ARGS)
    opts = stage2_options(args, ("standard", "affine", "output"))
    isnothing(opts) && return
    all(haskey(opts, name) for name in ("standard", "affine", "output")) ||
        error("--standard, --affine, and --output are required")
    destination = opts["output"]
    ispath(destination) && error("output already exists: $destination")
    mkpath(destination)
    rows = collect_profiles(opts["standard"], opts["affine"])
    failed_stderr = joinpath(opts["affine"], "n4096-p1.stderr.txt")
    occursin("tracked affine path changed state or clock", read(failed_stderr, String)) ||
        error("expected preserved affine 4096 p1 probe failure")
    report = Dict(
        "kind" => "stage2_diagnostic_components_not_acceptance",
        "script_sha256" => sha256file(@__FILE__),
        "standard_profiles_completed" => 6, "affine_2048_profiles_completed" => 3,
        "affine_4096_profiles_completed" => 0,
        "affine_4096_failure" => failed_stderr,
        "scope" => "split clocks and CPU sampling perturb scheduling/cache; no hardware counters or worker busy-time measure",
        "rows" => rows,
    )
    writejson(joinpath(destination, "component-report.json"), report)
    lines = ["# Stage 2 diagnostic component profiles", "",
        "The six standard profiles cover p1/4/8 at 2048² and 4096². Three affine profiles cover p1/4/8 at 2048². An affine 4096² p1 run failed the untimed tracked/untracked full-state-or-clock comparison; no later affine profiles ran. The failure remains unresolved.", "",
        "Each percentage below uses instrumented split-clock time, not the acceptance matrix. The standard profiler constrained the process to the selected CPU set but did not pin each worker. The affine profiler verified each worker's CPU mask. Neither profiler measured physical DRAM traffic or worker busy-time imbalance.", "",
        "| Implementation | Grid | Workers | Diagnostic full (s) | Split components | Allocated / full run |",
        "|:--|--:|--:|--:|:--|--:|"]
    for row in rows
        labels = join(["$name $(@sprintf("%.1f%%", 100 * value))"
                       for (name, value) in row["component_fractions"]], ", ")
        push!(lines, @sprintf("| %s %s | %d² | %d | %.3f | %s | %.1f KiB |",
            row["implementation"], row["mode"], row["size"], row["workers"],
            row["complete_diagnostic_median_seconds"], labels,
            row["complete_allocated_bytes"] / 1024))
    end
    append!(lines, ["", "The three standard Strang sweeps occupy similar shares at both mandatory sizes; no single pass explains the p4/p8 scaling miss. At 2048², affine observable copy-back consumes about 16–24% of the instrumented path at p4/p8, while amortized trajectory copy-back is below 1%. These fractions do not establish a DRAM-bandwidth cause. Julia CPU samples may include compilation/runtime frames; GC time was captured for affine complete trials only, and no causal hardware counter was available.", ""])
    write(joinpath(destination, "component-report.md"), join(lines, "\n"))
    plot_profiles(rows, destination)
    println("Wrote $(length(rows)) diagnostic profile rows to $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
