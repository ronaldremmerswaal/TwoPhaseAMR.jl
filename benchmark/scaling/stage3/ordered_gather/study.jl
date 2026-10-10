#!/usr/bin/env julia
"""Exploratory paired run of copied general-AMR ordered-gather implementations."""
module OrderedGatherStudy

include(joinpath(@__DIR__, "..", "narrow_ids", "study.jl"))
include(joinpath(@__DIR__, "make_probe.jl"))
using .BlockInteriorStudy
using .OrderedGatherProbe
using Random
using TOML

const Study = BlockInteriorStudy
const Probe = OrderedGatherProbe
const Support = Study.ScalingSupport
const VALIDATOR = joinpath(@__DIR__, "validation_driver.jl")
const TEST_FILE = joinpath(Study.ROOT, "test", "operators", "advection",
                           "cell_advection_compact.jl")
const CASES = ((1024, 1024, 1024), (2048, 2048, 128)) # N, block, fixed steps
const WORKERS = (1, 4, 8)
const REPEATS = 3
const CANDIDATE_LABEL = "pair_select"

function parse_args(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: study.jl --cohort COMPLETED_NARROW_IDS_MATRIX --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("--cohort and --output are required")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--cohort", "--output")) ||
        error("expected only --cohort and --output")
    cohort, output = abspath(options["--cohort"]), abspath(options["--output"])
    isdir(cohort) || error("narrow-ID cohort missing: $cohort")
    !ispath(output) || error("output must be a new directory: $output")
    return (; cohort, output)
end

function completed_cohort(cohort)
    Study.verify_control()
    configuration = Support.readjson(joinpath(cohort, "configuration.json"))
    configuration["phase"] == "matrix" || error("input is not a matrix cohort")
    available = Set((Int(pair[1]), Int(pair[2])) for pair in configuration["cases"])
    all((n,b) in available for (n,b,_) in CASES) ||
        error("one-block 1024/2048 cases are missing from the cohort")
    summary = Support.readjson(joinpath(cohort, "summary.json"))
    rows = Support.readjsonl(joinpath(cohort, "raw.jsonl"))
    expected = 2 * length(available) * length(WORKERS) * configuration["repeats"]
    summary["samples"] == expected && length(rows) == expected ||
        error("narrow-ID matrix has not completed")
    all(row["bitwise_equal"] && row["tolerance_violations"] == 0 for row in rows) ||
        error("narrow-ID matrix contains a field mismatch")
    source = joinpath(cohort, "current-source")
    clean = joinpath(source, Probe.RELATIVE)
    Support.sha256file(clean) == Probe.EXPECTED_CLEAN_SHA256 ||
        error("cohort source is not the frozen narrow-ID candidate")
    manifest = Support.readjson(joinpath(cohort, "manifest.json"))
    locked = get(manifest, joinpath(Study.ROOT, Probe.RELATIVE),
                 get(manifest, Probe.RELATIVE, nothing))
    locked == Probe.EXPECTED_CLEAN_SHA256 ||
        error("cohort manifest does not lock the narrow-ID candidate")
    placements = Support.readjson(joinpath(cohort, "placement.json"))
    all(length(placements[string(p)]) == p for p in WORKERS) ||
        error("cohort placement lacks a required worker count")
    return (; source, placements)
end

function write_batch(path, cases)
    open(path, "w") do io
        TOML.print(io, Dict("cases" => cases))
    end
    return path
end

function case_settings(n, b, steps, initial, field, report)
    return Dict{String,Any}("n" => n, "block" => b, "steps" => steps,
        "input" => initial, "output" => field, "report" => report)
end

function reference_fixtures(options, control, driver, pin1)
    fixtures = Dict{Int,Any}()
    oracle_cases = Dict{String,Any}[]
    for (n,b,steps) in CASES
        original = joinpath(options.cohort, "initial-n$n.bin")
        isfile(original) && filesize(original) == 8*n*n ||
            error("cohort initial field missing or incomplete: $original")
        initial = joinpath(options.output, "initial-n$n.bin")
        cp(original, initial)
        oracle = joinpath(options.output, "oracle-n$n-b$b-s$steps.bin")
        oracle_report = joinpath(options.output, "oracle-n$n-b$b-s$steps.toml")
        fixtures[n] = (; n, b, steps, initial, oracle, oracle_report,
                       initial_sha256=Support.sha256file(initial))
        push!(oracle_cases, case_settings(n,b,steps,initial,oracle,oracle_report))
    end
    batch = write_batch(joinpath(options.output, "oracle-batch.toml"), oracle_cases)
    Study.launch(options.output, driver, 1, pin1,
        ["source=$control", "mode=measure", "kernel=upwind",
         "cpus=$(join(pin1, ','))", "batch=$batch"])
    for (n,b,steps) in CASES
        fixture = fixtures[n]
        report = TOML.parsefile(fixture.oracle_report)
        report["source"] == control && report["n"] == n &&
            report["block"] == b && report["steps"] == steps &&
            report["field_sha256"] == Support.sha256file(fixture.oracle) ||
            error("fresh control oracle identity differs: n$n")
        fixtures[n] = merge(fixture,
            (; oracle_sha256=Support.sha256file(fixture.oracle)))
    end
    return fixtures, batch
end

function validate_copied_suite(output, label, source, workers, cpus, expected_hash)
    report = joinpath(output, "suite-$label-p$workers.toml")
    Study.launch(output, VALIDATOR, workers, cpus,
        [source, TEST_FILE, expected_hash, report])
    result = TOML.parsefile(report)
    result["status"] == "passed" &&
        result["loaded_advection_sha256"] == expected_hash &&
        result["loaded_advection_file"] == joinpath(source, Probe.RELATIVE) &&
        result["ordered_sum_method_file"] == realpath(joinpath(source, Probe.RELATIVE)) &&
        result["original_test_sha256"] == Support.sha256file(TEST_FILE) &&
        result["permutation_count"] == 24 &&
        result["order_checks"] == 222 &&
        result["canonical_fast_checks"] == 36 &&
        result["generic_fallback_checks"] == 186 &&
        (label == "control" ? isempty(result["generic_fallback_method_file"]) :
            result["generic_fallback_method_file"] ==
                realpath(joinpath(source, Probe.RELATIVE))) &&
        result["threads"] == workers || error("copied-module suite receipt differs: $label")
    return result
end

function schedule_pairs()
    pairs = [(repeat,p) for repeat in 0:REPEATS-1 for p in WORKERS]
    shuffle!(MersenneTwister(20261010), pairs)
    return [Dict("repeat" => repeat, "workers" => p,
                 "order" => collect(iseven(repeat+p) ?
                     ("control", CANDIDATE_LABEL) : (CANDIDATE_LABEL, "control")))
            for (repeat,p) in pairs]
end

function batch_files(output, schedule, fixtures)
    batches = Dict{Tuple{String,Int,Int},String}()
    for job in schedule, variant in job["order"]
        p, repeat = job["workers"], job["repeat"]
        stem = "$variant-p$p-r$repeat"
        cases = Dict{String,Any}[]
        for (n,b,steps) in CASES
            fixture = fixtures[n]
            push!(cases, case_settings(n,b,steps,fixture.initial,
                joinpath(output, "$stem-n$n.bin"),
                joinpath(output, "$stem-n$n.toml")))
        end
        batches[variant,p,repeat] =
            write_batch(joinpath(output, "$stem-batch.toml"), cases)
    end
    return batches
end

function locked_hashes(options, control, candidate, driver, fixtures,
                       oracle_batch, batches)
    paths = String[@__FILE__, joinpath(@__DIR__, "make_probe.jl"), VALIDATOR,
        joinpath(@__DIR__, "..", "narrow_ids", "study.jl"), TEST_FILE,
        joinpath(@__DIR__, "..", "..", "Support.jl"), driver,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(Study.ROOT, "benchmark", "scaling", "Project.toml"),
        joinpath(Study.ROOT, "benchmark", "scaling", "Manifest.toml"),
        joinpath(options.cohort, "manifest.json"),
        joinpath(options.cohort, "configuration.json"),
        joinpath(options.cohort, "placement.json"),
        joinpath(options.cohort, "raw.jsonl"),
        joinpath(options.cohort, "summary.json"),
        joinpath(control, Probe.RELATIVE), joinpath(candidate, Probe.RELATIVE),
        joinpath(candidate, "ordered-gather-receipt.json"),
        joinpath(candidate, "ordered-gather.patch"), oracle_batch]
    append!(paths, values(batches))
    for fixture in values(fixtures)
        append!(paths, (fixture.initial, fixture.oracle, fixture.oracle_report))
    end
    all(isfile, paths) || error("a required diagnostic source or fixture is missing")
    return Dict(path => Support.sha256file(path) for path in sort!(unique(paths)))
end

function validate_case(output, variant, source, n, b, steps, p, repeat,
                       fixture, cpus)
    stem = "$variant-p$p-r$repeat-n$n"
    field = joinpath(output, "$stem.bin")
    report = TOML.parsefile(joinpath(output, "$stem.toml"))
    for (key, expected) in ("mode" => "measure", "kernel" => "upwind",
                            "n" => n, "block" => b, "threads" => p,
                            "steps" => steps, "source" => source)
        report[key] == expected || error("$stem: $key differs")
    end
    report["dt"] == 0.35/n &&
        isfinite(report["time_seconds"]) && report["time_seconds"] >= 0.2 &&
        isapprox(report["ns_per_unknown_step"],
                 1e9*report["time_seconds"]/(n*n*steps); rtol=1e-12) ||
        error("$stem: short or invalid timed trajectory")
    [entry["cpu"] for entry in report["affinity"]] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in report["affinity"]) ||
        error("$stem: affinity differs")
    field_hash = Support.sha256file(field)
    report["field_sha256"] == field_hash &&
        field_hash == fixture.oracle_sha256 || error("$stem: final-field SHA differs")
    comparison = Study.full_field(field, fixture.oracle, n)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
        comparison["max_abs_vs_oracle"] == 0 ||
        error("$stem: full field differs from fresh control oracle")
    merge!(report, comparison)
    merge!(report, Dict("variant" => variant, "repeat" => repeat,
        "initial_sha256" => fixture.initial_sha256,
        "oracle_sha256" => fixture.oracle_sha256,
        "source_sha256" => Support.sha256file(joinpath(source, Probe.RELATIVE))))
    rm(field)
    return report
end

function summarize(rows)
    groups = Dict{String,Any}[]
    for (n,b,steps) in CASES, p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
                        row["n"] == n && row["threads"] == p &&
                        row["variant"] == "control" && row["repeat"] == repeat)
                   for repeat in 0:REPEATS-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
                          row["n"] == n && row["threads"] == p &&
                          row["variant"] == CANDIDATE_LABEL &&
                          row["repeat"] == repeat)
                     for repeat in 0:REPEATS-1]
        length(control) == length(candidate) == REPEATS ||
            error("incomplete paired diagnostic group")
        cmed, pmed = Support.median_value(control), Support.median_value(candidate)
        paired = [control[r]/candidate[r] for r in 1:REPEATS]
        push!(groups, Dict("n" => n, "block" => b, "steps" => steps,
            "workers" => p, "repeats" => REPEATS,
            "control_median_ns_per_unknown_step" => cmed,
            "candidate_median_ns_per_unknown_step" => pmed,
            "control_mad_ns_per_unknown_step" => Support.mad_value(control),
            "candidate_mad_ns_per_unknown_step" => Support.mad_value(candidate),
            "control_over_candidate_median_ratio" => cmed/pmed,
            "individual_paired_ratios" => paired))
    end
    return Dict("schema" => 1, "samples" => length(rows), "rows" => groups,
                "comparison" => "exploratory adjacent fresh-process pairs; same fixed trajectories")
end

function main(args=ARGS)
    options = parse_args(args)
    options === nothing && return
    final = completed_cohort(options.cohort)
    mkpath(options.output)
    control = joinpath(options.output, "control")
    candidate = joinpath(options.output, CANDIDATE_LABEL)
    mkpath(control)
    cp(joinpath(final.source, "src"), joinpath(control, "src"))
    receipt = Probe.make_probe(final.source, candidate)
    Support.source_hash(control) == Support.source_hash(final.source) ||
        error("control copy differs from narrow-ID cohort source")
    driver = joinpath(options.output, "driver.jl")
    cp(Study.DRIVER, driver)
    suite = Dict(
        "control" => validate_copied_suite(options.output, "control", control, 1,
                                            final.placements["1"],
                                            Probe.EXPECTED_CLEAN_SHA256),
        CANDIDATE_LABEL => validate_copied_suite(options.output, CANDIDATE_LABEL,
            candidate, 4, final.placements["4"],
            receipt["candidate_source_sha256"]))
    fixtures, oracle_batch =
        reference_fixtures(options, control, driver, final.placements["1"])
    schedule = schedule_pairs()
    batches = batch_files(options.output, schedule, fixtures)
    source_hashes = Dict("control" => Support.source_hash(control),
                         CANDIDATE_LABEL => Support.source_hash(candidate))
    frozen = locked_hashes(options, control, candidate, driver, fixtures,
                           oracle_batch, batches)
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "final_cohort" => options.cohort,
        "cases" => [[n,b,steps] for (n,b,steps) in CASES],
        "workers" => collect(WORKERS), "repeats" => REPEATS,
        "minimum_timed_seconds" => 0.2, "placement" => final.placements,
        "source_tree_sha256" => source_hashes, "locked_files_sha256" => frozen,
        "copied_suite" => suite, "probe_receipt" => receipt,
        "driver_sha256" => Support.sha256file(driver)))
    Support.writejson(joinpath(options.output, "job_order.json"), schedule)
    rows = Any[]
    for job in schedule
        repeat, p = job["repeat"], job["workers"]
        cpus = final.placements[string(p)]
        for variant in job["order"]
            locked_hashes(options, control, candidate, driver, fixtures,
                          oracle_batch, batches) == frozen &&
                all(Support.source_hash(root) == source_hashes[name]
                    for (name,root) in (("control",control),
                                        (CANDIDATE_LABEL,candidate))) ||
                error("diagnostic source, input, or environment changed")
            source = variant == "control" ? control : candidate
            batch = batches[variant,p,repeat]
            Study.launch(options.output, driver, p, cpus,
                ["source=$source", "mode=measure", "kernel=upwind",
                 "cpus=$(join(cpus, ','))", "batch=$batch"])
            locked_hashes(options, control, candidate, driver, fixtures,
                          oracle_batch, batches) == frozen ||
                error("diagnostic source or input changed during launch")
            for (n,b,steps) in CASES
                row = validate_case(options.output, variant, source, n, b, steps,
                                    p, repeat, fixtures[n], cpus)
                Support.appendjsonl(joinpath(options.output, "raw.jsonl"), row)
                push!(rows, row)
                println("$variant n$n p$p r$repeat: ",
                        round(row["ns_per_unknown_step"]; digits=3), " ns/cell-step")
            end
            flush(stdout)
        end
    end
    Support.writejson(joinpath(options.output, "summary.json"), summarize(rows))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
