#!/usr/bin/env julia
"""One-case, fresh-process matched CFL-inclusive ordered-gather diagnostic."""
module CFLOrderedGatherStudy

include(joinpath(@__DIR__, "..", "narrow_ids", "study.jl"))
include(joinpath(@__DIR__, "make_probe.jl"))
include(joinpath(@__DIR__, "make_checked_cfl_driver.jl"))
using .BlockInteriorStudy
using .OrderedGatherProbe
using .CheckedCFLOrderedGatherDriver
using TOML
using Random

const Study = BlockInteriorStudy
const Probe = OrderedGatherProbe
const Support = Study.ScalingSupport
const COHORT = joinpath(Study.ROOT, "benchmark", "scaling", "results",
                        "20261010-narrow-ids-final")
const CANDIDATE = "pair_select"
const VALIDATOR = joinpath(@__DIR__, "validation_driver.jl")
const TEST_FILE = joinpath(Study.ROOT, "test", "operators", "advection",
                           "cell_advection_compact.jl")

function options(args)
    length(args) == 8 || error("usage: cfl_study.jl --case N:BLOCK --steps N --repeats N --output NEW_DIR")
    o = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    Set(keys(o)) == Set(("--case", "--steps", "--repeats", "--output")) ||
        error("invalid options")
    pair = parse.(Int, split(o["--case"], ":"))
    length(pair) == 2 || error("case must be N:BLOCK")
    n, block = pair
    n in (1024, 2048) && block in (n, 8) && n % block == 0 ||
        error("unsupported case")
    steps, repeats = parse(Int, o["--steps"]), parse(Int, o["--repeats"])
    steps >= 1 && repeats >= 1 || error("positive steps/repeats required")
    output = abspath(o["--output"])
    !ispath(output) || error("output must be a new directory")
    return (; n, block, steps, repeats, output)
end

function append_event(output, event)
    Support.appendjsonl(joinpath(output, "events.jsonl"),
                        merge(Dict("unix_ns" => round(Int, time()*1e9)), event))
end

function checked_launch(output, driver, p, cpus, args, label)
    append_event(output, Dict("event" => "launch", "label" => label,
                              "workers" => p))
    Study.launch(output, driver, p, cpus, args)
    append_event(output, Dict("event" => "completed", "label" => label,
                              "workers" => p))
end

function run_case(output, driver, source, variant, p, repeat, cpus,
                  n, block, steps, initial, oracle; minimum=0.0)
    stem = "$variant-n$n-b$block-p$p-r$repeat"
    field = joinpath(output, "$stem.bin")
    report = joinpath(output, "$stem.toml")
    checked_launch(output, driver, p, cpus,
        ["source=$source", "mode=measure_cfl", "kernel=upwind_cfl", "n=$n",
         "block=$block", "steps=$steps", "cpus=$(join(cpus, ','))",
         "input=$initial", "output=$field", "report=$report"], stem)
    row = TOML.parsefile(report)
    for (key, expected) in (("source", source), ("n", n), ("block", block),
                            ("threads", p), ("steps", steps), ("blocks", (n÷block)^2),
                            ("faces", 2n*n))
        row[key] == expected || error("$stem: $key differs")
    end
    row["dt"] == 0.4row["cfl_bound"] && row["julia_version"] == "1.10.12" &&
        row["mode"] == "measure_cfl" &&
        row["workload"] == CheckedCFLOrderedGatherDriver.OrderedGatherCFLDriver.WORKLOAD &&
        row["cfl_scans_per_timed_step"] == 1 &&
        isfinite(row["time_seconds"]) && row["time_seconds"] >= minimum &&
        isfinite(row["ns_per_unknown_step"]) &&
        isapprox(row["ns_per_unknown_step"],
                 1e9row["time_seconds"]/(n*n*steps); rtol=1e-12) ||
        error("$stem: invalid/short timed trajectory")
    [entry["cpu"] for entry in row["affinity"]] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in row["affinity"]) ||
        error("$stem: affinity differs")
    row["field_sha256"] == Support.sha256file(field) ||
        error("$stem: field hash differs from report")
    comparison = Study.full_field(field, oracle, n)
    merge!(row, comparison)
    merge!(row, Dict("variant" => variant, "repeat" => repeat,
                     "initial_sha256" => Support.sha256file(initial),
                     "oracle_sha256" => Support.sha256file(oracle),
                     "source_sha256" => Support.sha256file(joinpath(source, Probe.RELATIVE))))
    Support.appendjsonl(joinpath(output, "raw.jsonl"), row)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
        comparison["max_abs_vs_oracle"] == 0 ||
        error("$stem: full field differs from fresh control oracle")
    row["time_seconds"] >= minimum || error("$stem: short interval")
    rm(field)
    return row
end

function summarize(rows, n, block, steps, repeats)
    groups = Any[]
    for p in (1,4,8)
        controls = [only(row["ns_per_unknown_step"] for row in rows if
            row["threads"] == p && row["variant"] == "control" &&
            row["repeat"] == r) for r in 0:repeats-1]
        candidates = [only(row["ns_per_unknown_step"] for row in rows if
            row["threads"] == p && row["variant"] == CANDIDATE &&
            row["repeat"] == r) for r in 0:repeats-1]
        cmed, xmed = Support.median_value(controls), Support.median_value(candidates)
        push!(groups, Dict("workers" => p, "control_median_ns_per_unknown_step" => cmed,
            "candidate_median_ns_per_unknown_step" => xmed,
            "control_over_candidate_ratio" => cmed/xmed,
            "candidate_parallel_efficiency" => nothing,
            "paired_ratios" => controls ./ candidates,
            "control_samples" => controls, "candidate_samples" => candidates))
    end
    one = groups[1]["candidate_median_ns_per_unknown_step"]
    for group in groups
        group["candidate_parallel_efficiency"] = one /
            (group["workers"] * group["candidate_median_ns_per_unknown_step"])
    end
    return Dict("schema" => 1, "case" => [n,block], "steps" => steps,
                "repeats" => repeats, "minimum_seconds" => 0.2,
                "samples" => length(rows), "groups" => groups,
                "scope" => "uniform periodic fixed-carrier CFL scan and upwind advect per timed step, dt=0.4*CFL bound; exploratory adjacent fresh-process pairs")
end

function main(args=ARGS)
    o = options(args)
    Study.verify_control()
    final_source = joinpath(COHORT, "current-source")
    Support.sha256file(joinpath(final_source, Probe.RELATIVE)) ==
        Probe.EXPECTED_CLEAN_SHA256 || error("frozen narrow-ID source differs")
    placement = Support.readjson(joinpath(COHORT, "placement.json"))
    mkpath(o.output)
    control = joinpath(o.output, "control")
    candidate = joinpath(o.output, CANDIDATE)
    mkpath(control)
    cp(joinpath(final_source, "src"), joinpath(control, "src"))
    receipt = Probe.make_probe(final_source, candidate)
    suites = Dict{String,Any}()
    for (name, source, p, expected) in
        (("control", control, 1, Probe.EXPECTED_CLEAN_SHA256),
         (CANDIDATE, candidate, 4, receipt["candidate_source_sha256"]))
        suite_report = joinpath(o.output, "suite-$name.toml")
        checked_launch(o.output, VALIDATOR, p, placement[string(p)],
                       [source, TEST_FILE, expected, suite_report], "suite-$name")
        suite = TOML.parsefile(suite_report)
        suite["status"] == "passed" && suite["original_test_sha256"] ==
            Support.sha256file(TEST_FILE) && suite["order_checks"] == 222 ||
            error("$name copied-source suite differs")
        suites[name] = suite
    end
    driver = joinpath(o.output, "checked_driver.jl")
    driver_hash = CheckedCFLOrderedGatherDriver.make_checked_cfl_driver(driver)
    initial_source = joinpath(COHORT, "initial-n$(o.n).bin")
    initial = joinpath(o.output, "initial-n$(o.n).bin")
    cp(initial_source, initial)
    filesize(initial) == 8o.n*o.n || error("initial field byte count differs")
    oracle = joinpath(o.output, "oracle.bin")
    oracle_report = joinpath(o.output, "oracle.toml")
    Support.writejson(joinpath(o.output, "configuration.json"), Dict(
        "schema" => 1, "case" => [o.n,o.block], "steps" => o.steps,
        "repeats" => o.repeats, "minimum_seconds" => 0.2,
        "control_advection_sha256" => Probe.EXPECTED_CLEAN_SHA256,
        "candidate_advection_sha256" => receipt["candidate_source_sha256"],
        "driver_sha256" => driver_hash, "source_driver_sha256" =>
            Support.sha256file(CheckedCFLOrderedGatherDriver.CheckedOrderedGatherDriver.SOURCE),
        "generator_sha256" => Support.sha256file(joinpath(@__DIR__, "make_checked_cfl_driver.jl")),
        "cfl_generator_sha256" => Support.sha256file(joinpath(@__DIR__, "make_cfl_driver.jl")),
        "checked_generator_sha256" => Support.sha256file(joinpath(@__DIR__, "make_checked_driver.jl")),
        "study_sha256" => Support.sha256file(@__FILE__),
        "validation_driver_sha256" => Support.sha256file(VALIDATOR),
        "compact_test_sha256" => Support.sha256file(TEST_FILE),
        "copied_source_suites" => suites,
        "initial_sha256" => Support.sha256file(initial),
        "placement" => placement, "julia" => Study.JULIA,
        "project" => Study.PROJECT, "depot" => Study.DEPOT,
        "source_tree_sha256" => Dict("control" => Support.source_hash(control),
            CANDIDATE => Support.source_hash(candidate))))
    cpus1 = placement["1"]
    checked_launch(o.output, driver, 1, cpus1,
        ["source=$control", "mode=measure_cfl", "kernel=upwind_cfl", "n=$(o.n)",
         "block=$(o.block)", "steps=$(o.steps)", "cpus=$(join(cpus1, ','))",
         "input=$initial", "output=$oracle", "report=$oracle_report"], "oracle")
    oracle_row = TOML.parsefile(oracle_report)
    oracle_row["field_sha256"] == Support.sha256file(oracle) &&
        all(isfinite, reinterpret(Float64, read(oracle))) ||
        error("oracle field invalid")
    append_event(o.output, Dict("event" => "oracle_validated",
        "field_sha256" => oracle_row["field_sha256"]))
    jobs = [(r,p) for r in 0:o.repeats-1 for p in (1,4,8)]
    shuffle!(MersenneTwister(20261010+o.n+o.block), jobs)
    Support.writejson(joinpath(o.output, "job_order.json"),
        [Dict("repeat" => r, "workers" => p,
              "order" => collect(iseven(r+p) ? ("control",CANDIDATE) :
                                      (CANDIDATE,"control"))) for (r,p) in jobs])
    rows = Any[]
    for (r,p) in jobs
        order = iseven(r+p) ? ("control",CANDIDATE) : (CANDIDATE,"control")
        for variant in order
            source = variant == "control" ? control : candidate
            row = run_case(o.output, driver, source, variant, p, r,
                           placement[string(p)], o.n, o.block, o.steps,
                           initial, oracle; minimum=0.2)
            push!(rows, row)
            println("$variant n$(o.n) b$(o.block) p$p r$r: ",
                    round(row["ns_per_unknown_step"]; digits=3), " ns/unknown-step")
            flush(stdout)
        end
    end
    Support.writejson(joinpath(o.output, "summary.json"),
                      summarize(rows, o.n, o.block, o.steps, o.repeats))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
