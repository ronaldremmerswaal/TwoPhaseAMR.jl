# Audit and plot a completed, frozen Stage 2 final scaling artifact.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
using CairoMakie
using Random
using Statistics

const COLORS = Dict("standard" => :steelblue, "fused_affine" => :darkorange,
                    "fused_amortized" => :mediumpurple, "halo" => :gray)
api_scope(variant) = variant == "fused_amortized" ? "amortized_trajectory" : "observable_step"
serial_eligible(variant, scope) = scope == "amortized_trajectory" || variant != "fused_amortized"
group_key(row) = (row["nx"], row["variant"], Tuple(row["tile"]), row["workers"])

function ratio_ci(numerators, denominators, seed; draws=20_000)
    rng = MersenneTwister(seed)
    estimates = Vector{Float64}(undef, draws)
    for index in eachindex(estimates)
        estimates[index] = median(rand(rng, numerators, length(numerators))) /
                           median(rand(rng, denominators, length(denominators)))
    end
    sort!(estimates)
    return [estimates[floor(Int, fraction * (draws - 1)) + 1]
            for fraction in (0.025, 0.975)]
end

function audit(raw, config, workload)
    expected = config["repeats"]
    groups = Dict{Tuple, Vector{Any}}()
    for row in raw
        push!(get!(groups, group_key(row), Any[]), row)
    end
    length(raw) == 56 * expected || error("expected $(56 * expected) final samples")
    all(length(rows) == expected for rows in values(groups)) ||
        error("every final configuration must have $expected independent samples")
    for name in ("project_sha256", "toy_tree_sha256", "experiment_sha256",
                 "runner_sha256", "configuration_sha256", "workload_sha256")
        length(Set(row[name] for row in raw)) == 1 ||
            error("source/configuration/workload changed: $name")
    end
    Set(row["project_sha256"] for row in raw) == Set([config["project_sha256"]]) ||
        error("project hash differs from configuration")
    for n in config["sizes"]
        rows = [row for row in raw if row["nx"] == n]
        Set(row["steps"] for row in rows) == Set([workload[string(n)]]) ||
            error("workload steps differ at $(n)²")
        for field in ("dt", "final_time", "translation_l2")
            length(Set(row[field] for row in rows)) == 1 ||
                error("physical trajectory or translation error differs at $(n)²")
        end
    end
    for row in raw
        row["variant"] == "halo" || haskey(row, "probe_state_delta") ||
            error("instrumented kernel lacks probe parity field")
        row["max_state_delta"] == 0 && get(row, "probe_state_delta", 0) == 0 &&
            row["mass_error"] <= 512eps(Float64) ||
            error("numerical parity, probe parity, or conservation failed")
        isfinite(row["translation_l2"]) || error("nonfinite analytic error")
        row["requested_workers"] == row["workers"] &&
            length(row["affinity"]) == row["workers"] ||
            error("requested worker or affinity metadata inconsistent")
        if row["variant"] == "halo"
            row["workers"] == 1 && row["active_workers"] == 0 &&
                occursin("not instrumented", row["active_worker_scope"]) ||
                error("halo must be an explicitly uninstrumented serial-only reference")
        else
            1 <= row["active_workers_min_pass"] <= row["active_workers"] <=
                row["tile_worker_capacity"] ||
                error("invalid actual compute worker participation")
        end
        if row["nx"] in (512, 1024, 2048, 4096) && row["workers"] in (4, 8)
            row["active_workers_min_pass"] == row["workers"] ||
                error("mandatory primary point failed per-pass participation")
        end
    end
    return groups
end

function build_report(groups, raw, config, workload)
    samples = Dict(key => [row["time_seconds"] for row in rows]
                   for (key, rows) in groups)
    eligible = Dict{Tuple, Vector{Tuple}}()
    for key in keys(groups)
        n, variant, _, p = key
        p == 1 || continue
        for scope in ("observable_step", "amortized_trajectory")
            serial_eligible(variant, scope) || continue
            push!(get!(eligible, (n, scope), Tuple[]), key)
        end
    end
    fastest = Dict(key => first(sort(keys; by=item->median(samples[item])))
                   for (key, keys) in eligible)
    result = Any[]
    for (key, rows) in sort!(collect(groups); by=first)
        n, variant, tile, p = key
        own = (n, variant, tile, 1)
        scope = api_scope(variant)
        reference = fastest[(n, scope)]
        times = samples[key]
        mid = median(times)
        own_mid = median(samples[own])
        ref_mid = median(samples[reference])
        seed = config["seed"] + n + p + sum(Int(c) for c in variant) + tile[1] + tile[2]
        self_ci = own == key ? [1.0, 1.0] : ratio_ci(samples[own], times, seed)
        abs_ci = reference == key ? [1.0, 1.0] : ratio_ci(samples[reference], times, seed + 10101)
        regression_ci = own == reference ? [1.0, 1.0] :
            ratio_ci(samples[own], samples[reference], seed + 20202)
        efficiency_ci = self_ci ./ p
        classification = p in (4, 8) ?
            (efficiency_ci[1] >= 0.5 ? "pass" :
             efficiency_ci[2] < 0.5 ? "fail" : "inconclusive") : "diagnostic"
        med(field) = median([row[field] for row in rows])
        push!(result, Dict(
            "size" => n, "variant" => variant, "tile" => collect(tile),
            "api_scope" => scope, "requested_workers" => p,
            "actual_workers_min" => minimum(row["active_workers"] for row in rows),
            "actual_workers_min_pass" => minimum(row["active_workers_min_pass"] for row in rows),
            "tile_worker_capacity" => rows[1]["tile_worker_capacity"],
            "worker_evidence_scope" => rows[1]["active_worker_scope"],
            "samples" => length(rows), "steps" => rows[1]["steps"],
            "time_median_seconds" => mid, "time_min_seconds" => minimum(times),
            "time_max_seconds" => maximum(times),
            "self_speedup" => own_mid / mid, "self_speedup_ci95" => self_ci,
            "self_efficiency" => own_mid / mid / p,
            "self_efficiency_ci95" => efficiency_ci,
            "fastest_serial_variant" => reference[2],
            "fastest_serial_tile" => collect(reference[3]),
            "fastest_serial_median_seconds" => ref_mid,
            "absolute_speedup" => ref_mid / mid, "absolute_speedup_ci95" => abs_ci,
            "absolute_efficiency" => ref_mid / mid / p,
            "absolute_efficiency_ci95" => abs_ci ./ p,
            "serial_regression_fraction" => own_mid / ref_mid - 1,
            "serial_regression_ci95" => regression_ci .- 1,
            "initial_serial_regression_budget" => 0.05,
            "self_median_above_target_at_this_count" =>
                p in (4, 8) && own_mid / mid / p >= 0.5,
            "self_target_ci_classification" => classification,
            "translation_l2" => rows[1]["translation_l2"],
            "mass_error_max" => maximum(row["mass_error"] for row in rows),
            "peak_rss_median_bytes" => med("peak_rss_bytes"),
            "numerical_array_bytes" => rows[1]["numerical_array_bytes"],
            "throughput_cell_steps_per_second" => n * n * rows[1]["steps"] / mid,
            "allocation_median_bytes" => med("allocation_bytes"),
            "allocation_median_bytes_per_step" => med("allocation_bytes") / rows[1]["steps"],
            "valid_sample_duration" => minimum(times) >= 0.2,
        ))
    end
    return Dict(
        "schema_version" => 1,
        "scope_rules" => Dict(
            "observable_step" => "fastest p1 among standard, halo, and fused_affine at identical size/trajectory; excludes amortized-only implementation",
            "amortized_trajectory" => "fastest p1 among all complete final-state trajectories, including standard and halo",
            "halo_evidence" => "p1 serial time and numerical parity only; kernel worker IDs not instrumented",
            "efficiency_denominator" => "requested Julia compute workers, even when tile count underoccupies them",
        ),
        "input_samples" => length(raw), "workload" => workload, "rows" => result,
    )
end

function write_markdown(report, path)
    lines = ["# Stage 2 final scaling audit", "",
        "Five independent fresh-process samples per configuration. Speedups use the same fixed steps per size. The absolute reference is the fastest measured one-worker complete trajectory within the indicated API scope. The observable-step scope excludes amortized-only implementations; the amortized scope includes standard and halo serial trajectories. Halo's p1 worker-ID count is uninstrumented, not zero computation.", "",
        "| Size | Scope/configuration | p | Active min/pass | Time median (s) | Self E | Self 95% CI | CI gate | Absolute E | Absolute 95% CI | Serial regression | Reference | Min sample (s) |",
        "|---:|---|---:|---:|---:|---:|---|---|---:|---|---:|---|---:|"]
    for row in report["rows"]
        row["size"] in (512, 1024, 2048, 4096) || continue
        tile = row["tile"]
        ref_tile = row["fastest_serial_tile"]
        label = "$(row["variant"]) $(tile[1])×$(tile[2])"
        reference = "$(row["fastest_serial_variant"]) $(ref_tile[1])×$(ref_tile[2])"
        ci = row["absolute_efficiency_ci95"]
        self_ci = row["self_efficiency_ci95"]
        workers = row["variant"] == "halo" ? "unobserved (serial)" :
            string(row["actual_workers_min_pass"])
        push!(lines, "| $(row["size"]) | $label | $(row["requested_workers"]) | $workers | " *
            "$(round(row["time_median_seconds"]; digits=4)) | " *
            "$(round(row["self_efficiency"]; digits=3)) | " *
            "[$(round(self_ci[1]; digits=3)), $(round(self_ci[2]; digits=3))] | " *
            "$(row["self_target_ci_classification"]) | " *
            "$(round(row["absolute_efficiency"]; digits=3)) | " *
            "[$(round(ci[1]; digits=3)), $(round(ci[2]; digits=3))] | " *
            "$(round(100 * row["serial_regression_fraction"]; digits=1))% | " *
            "$reference | $(round(row["time_min_seconds"]; digits=3)) |")
    end
    append!(lines, ["", "The primary target is **fixed-configuration self-efficiency** ≥0.50 at p4 or p8 on each mandatory 512², 2048², and 4096² complete-step fixture. Speedup against the fastest validated serial trajectory and the initial ≤5% one-worker regression budget are separate safeguards. Inspect confidence intervals and numerical gates before assigning a stage outcome. Small-grid underoccupancy and SMT are separate diagnostics. Worker IDs show participation; worker busy-time fractions were not measured.", ""])
    write(path, join(lines, "\n"))
end

function plots(report, destination)
    rows = report["rows"]
    for n in (512, 1024, 2048, 4096)
        items = [r for r in rows if r["size"] == n && r["requested_workers"] in (1, 2, 4, 8, 16)]
        policies = sort!(collect(Set((r["variant"], Tuple(r["tile"])) for r in items if r["variant"] != "halo")))
        fig = Figure(size=(1900, 470))
        axes = [Axis(fig[1, j], xlabel="Requested Julia workers", ylabel=label)
                for (j, label) in enumerate(("Wall time (s)", "Self efficiency T₁/(pTₚ)",
                                             "Speedup vs fastest serial", "Self speedup T₁/Tₚ"))]
        for (variant, tile) in policies
            curve = sort([r for r in items if r["variant"] == variant &&
                          Tuple(r["tile"]) == tile]; by=r->r["requested_workers"])
            length(curve) >= 2 || continue
            p = [r["requested_workers"] for r in curve]
            label = "$variant $(tile[1])×$(tile[2])"
            color = get(COLORS, variant, :black)
            for (axis, field) in zip(axes, ("time_median_seconds", "self_efficiency",
                                            "absolute_speedup", "self_speedup"))
                lines!(axis, p, [r[field] for r in curve]; label, color)
                scatter!(axis, p, [r[field] for r in curve]; color)
            end
        end
        hlines!(axes[2], [0.5]; color=:black, linestyle=:dash)
        axislegend(axes[1]; position=:rt)
        Label(fig[0, :], "$(n)² complete trajectory")
        for extension in ("svg", "png")
            save(joinpath(destination, "stage2-final-$n.$extension"), fig)
        end
    end
    standard = [r for r in rows if r["variant"] == "standard" &&
                r["requested_workers"] in (1, 4, 8, 16)]
    fig = Figure(size=(720, 480))
    axis = Axis(fig[1, 1], xscale=log10,
                xlabel="Numerical-array working set (MiB)",
                ylabel="Million cell-steps/s")
    for p in (1, 4, 8, 16)
        curve = sort([r for r in standard if r["requested_workers"] == p]; by=r->r["size"])
        lines!(axis, [r["numerical_array_bytes"] / 2^20 for r in curve],
               [r["throughput_cell_steps_per_second"] / 1e6 for r in curve]; label="$p workers")
    end
    axislegend(axis)
    Label(fig[0, :], "Standard LW throughput; frozen steps differ by size")
    for extension in ("svg", "png")
        save(joinpath(destination, "stage2-final-throughput-working-set.$extension"), fig)
    end
    selected = [r for r in rows if r["size"] in (512, 2048, 4096) &&
                r["requested_workers"] in (1, 4, 8)]
    labels = ["$(r["size"])² $(r["variant"]) $(r["tile"][1])×$(r["tile"][2]) p$(r["requested_workers"])" for r in selected]
    memory = Figure(size=(1600, max(600, 28 * length(selected))))
    left = Axis(memory[1, 1], xlabel="Numerical-array storage (MiB)",
                yticks=(1:length(labels), labels))
    right = Axis(memory[1, 2], xlabel="Timed allocation per step (bytes)",
                 yticks=(1:length(labels), fill("", length(labels))))
    barplot!(left, 1:length(selected), [r["numerical_array_bytes"] / 2^20 for r in selected];
             direction=:x, color=:steelblue)
    barplot!(right, 1:length(selected), [r["allocation_median_bytes_per_step"] for r in selected];
             direction=:x, color=:darkorange)
    for extension in ("svg", "png")
        save(joinpath(destination, "stage2-final-memory-allocation.$extension"), memory)
    end
end

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia report_final.jl RESULT_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: julia report_final.jl RESULT_DIRECTORY")
    path = args[1]
    raw_path = joinpath(path, "raw.jsonl")
    raw = readjsonl(raw_path)
    config = readjson(joinpath(path, "configuration.json"))
    workload = readjson(joinpath(path, "workload.json"))
    groups = audit(raw, config, workload)
    report = build_report(groups, raw, config, workload)
    report["report_script_sha256"] = sha256file(@__FILE__)
    report["raw_jsonl_sha256"] = sha256file(raw_path)
    report["original_summary_sha256"] = sha256file(joinpath(path, "summary.json"))
    report["config_sha256"] = config["configuration_sha256"]
    writejson(joinpath(path, "report-corrected.json"), report)
    write_markdown(report, joinpath(path, "report-corrected.md"))
    plots(report, path)
    println("Audited $(length(raw)) samples and wrote corrected scope report/figures to $path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
