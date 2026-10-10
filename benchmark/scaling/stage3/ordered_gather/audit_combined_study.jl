#!/usr/bin/env julia
"""Read-only pooled audit for adjacent, identical safe-study cohorts."""
module AuditCombinedOrderedGather

include(joinpath(@__DIR__, "audit_safe_study.jl"))
using .AuditSafeOrderedGather
using .AuditSafeOrderedGather.ScalingSupport
using JSON
using Random

const Audit = AuditSafeOrderedGather

function paired_median_interval(ratios; seed, draws=5000)
    rng = MersenneTwister(seed)
    samples = [median_value(rand(rng, ratios, length(ratios))) for _ in 1:draws]
    sort!(samples)
    return [percentile(samples, 0.025), percentile(samples, 0.975)]
end

function audit_cohort(output)
    config = readjson(joinpath(output, "configuration.json"))
    manifest = Audit.audit_sources(output, config)
    oracle_hash = Audit.audit_tooling(output, config, manifest)
    commands = Audit.audit_commands(output, config)
    rows, _, strongest_serial = Audit.audit_rows(output, config, oracle_hash)
    return (; output, config, oracle_hash, commands, rows, strongest_serial)
end

function main(args=ARGS)
    length(args) == 2 ||
        error("usage: audit_combined_study.jl COMPLETED_FIRST_DIR COMPLETED_TWO_REPEAT_DIR")
    cohorts = [audit_cohort(abspath(arg)) for arg in args]
    a, b = cohorts
    a.config["repeats"] in (1, 3) || error("first cohort must have one or three repeats")
    Audit.require_equal(b.config["repeats"], 2, "second cohort repeats")
    pairs = a.config["repeats"] + b.config["repeats"]
    for key in ("case", "steps", "minimum_seconds", "control_advection_sha256",
                "candidate_advection_sha256", "driver_sha256", "source_driver_sha256",
                "generator_sha256", "study_sha256", "validation_driver_sha256",
                "compact_test_sha256", "initial_sha256", "placement", "julia",
                "project", "depot", "source_tree_sha256")
        Audit.require_equal(a.config[key], b.config[key], "cross-cohort $key")
    end
    Audit.require_equal(a.oracle_hash, b.oracle_hash, "cross-cohort full-field oracle")
    rows = vcat(a.rows, b.rows)
    length(rows) == 6pairs || error("combined matrix timed-row count differs")
    medians = Dict{Tuple{String,Int},Float64}()
    values = Dict{Tuple{String,Int},Vector{Float64}}()
    ratios = Dict{Int,Vector{Float64}}()
    for p in Audit.WORKERS
        for variant in Audit.VARIANTS
            samples = Float64[]
            for cohort in cohorts, repeat in 0:cohort.config["repeats"]-1
                row = only(row for row in cohort.rows if
                           row["variant"] == variant && row["threads"] == p &&
                           row["repeat"] == repeat)
                push!(samples, row["ns_per_unknown_step"])
            end
            length(samples) == pairs || error("$variant p$p lacks $pairs process samples")
            values[variant,p] = samples
            medians[variant,p] = median_value(samples)
        end
        ratios[p] = Float64[]
        for cohort in cohorts, repeat in 0:cohort.config["repeats"]-1
            control = only(row["ns_per_unknown_step"] for row in cohort.rows if
                           row["variant"] == "control" && row["threads"] == p &&
                           row["repeat"] == repeat)
            candidate = only(row["ns_per_unknown_step"] for row in cohort.rows if
                             row["variant"] == "pair_select" && row["threads"] == p &&
                             row["repeat"] == repeat)
            push!(ratios[p], control / candidate)
        end
    end
    strongest = argmin([medians[variant,1] for variant in Audit.VARIANTS])
    strongest_variant = Audit.VARIANTS[strongest]
    serial = values[strongest_variant,1]
    serial_median = medians[strongest_variant,1]
    results = Any[]
    for p in Audit.WORKERS
        control = values["control",p]
        candidate = values["pair_select",p]
        same_worker_ci = bootstrap_ratio(control, candidate; seed=20261010+p)
        efficiency_ci = p == 1 ? nothing :
            bootstrap_ratio(serial, candidate, p; seed=20261110+p)
        push!(results, Dict(
            "workers" => p,
            "control_samples_ns_per_cell_advect" => control,
            "candidate_samples_ns_per_cell_advect" => candidate,
            "control_median_ns_per_cell_advect" => medians["control",p],
            "candidate_median_ns_per_cell_advect" => medians["pair_select",p],
            "control_over_candidate" => medians["control",p] / medians["pair_select",p],
            "control_over_candidate_95_interval" => same_worker_ci,
            "paired_ratios" => ratios[p],
            "paired_ratio_median" => median_value(ratios[p]),
            "paired_ratio_median_95_interval" =>
                paired_median_interval(ratios[p]; seed=20261210+p),
            "strongest_serial_efficiency" => serial_median /
                                                    (p*medians["pair_select",p]),
            "strongest_serial_efficiency_95_interval" => efficiency_ci))
    end
    JSON.print(stdout, Dict(
        "status" => "passed", "scope" => "uniform periodic fixed-dt complete advect!",
        "cohorts" => [Dict("output" => cohort.output,
                            "repeats" => cohort.config["repeats"],
                            "commands" => cohort.commands,
                            "timed_rows" => length(cohort.rows),
                            "oracle_sha256" => cohort.oracle_hash) for cohort in cohorts],
        "total_timed_rows" => length(rows),
        "independent_pairs_per_worker" => pairs,
        "meets_five_process_repetition_gate" => pairs >= 5,
        "strongest_serial_variant" => strongest_variant,
        "strongest_serial_ns_per_cell_advect" => serial_median,
        "results" => results), 2)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
