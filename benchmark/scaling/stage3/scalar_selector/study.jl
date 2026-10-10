#!/usr/bin/env julia
"""One-case copied-source scalar-selector study with checked mesh and oracle."""
module ScalarSelectorStudy

include(joinpath(@__DIR__, "..", "narrow_ids", "study.jl"))
include(joinpath(@__DIR__, "make_probe.jl"))
include(joinpath(@__DIR__, "..", "ordered_gather", "make_checked_driver.jl"))
include(joinpath(@__DIR__, "..", "ordered_gather", "make_checked_cfl_driver.jl"))
using .BlockInteriorStudy
using .ScalarSelectorProbe
using .CheckedOrderedGatherDriver
using .CheckedCFLOrderedGatherDriver
using Random
using TOML

const Study = BlockInteriorStudy
const Probe = ScalarSelectorProbe
const Support = Study.ScalingSupport
const COHORT = joinpath(Study.ROOT, "benchmark", "scaling", "results",
                        "20261010-narrow-ids-final")
const VALIDATOR = joinpath(@__DIR__, "validation_driver.jl")
const TEST_FILE = joinpath(Study.ROOT, "test", "operators", "advection",
                           "cell_advection_compact.jl")
const CANDIDATE = "scalar_selector"
const WORKERS = (1, 4, 8)

function options(args)
    length(args) in (10,12) || error("usage: study.jl --case 1024:8 --steps N --repeats N --workload fixed|cfl --output NEW_DIRECTORY [--suite-dir VALIDATED_DIRECTORY]")
    length(unique(args[1:2:end])) == length(args) ÷ 2 || error("duplicate option")
    o = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    required = Set(("--case", "--steps", "--repeats", "--workload", "--output"))
    issubset(required, Set(keys(o))) && all(key in union(required,Set(("--suite-dir",)))
                                   for key in keys(o)) ||
        error("invalid options")
    pair = parse.(Int, split(o["--case"], ':'))
    length(pair) == 2 || error("case must be N:BLOCK")
    n, block = pair
    n == 1024 && block in (8, 1024) || error("only 1024:8 or 1024:1024 are supported")
    steps, repeats = parse(Int, o["--steps"]), parse(Int, o["--repeats"])
    steps >= 1 && repeats >= 1 || error("positive steps/repeats required")
    workload = o["--workload"]
    workload in ("fixed", "cfl") || error("workload must be fixed or cfl")
    output = abspath(o["--output"])
    !ispath(output) || error("output must be a new directory")
    suite_dir = haskey(o,"--suite-dir") ? abspath(o["--suite-dir"]) : nothing
    isnothing(suite_dir) || isdir(suite_dir) || error("suite directory missing")
    return (; n, block, steps, repeats, workload, output, suite_dir)
end

function event(output, name; details=Dict{String,Any}())
    Support.appendjsonl(joinpath(output, "events.jsonl"),
        merge(Dict{String,Any}("event" => name,
                               "unix_ns" => round(Int, time()*1e9)), details))
end

function launch(output, script, p, cpus, args, label)
    event(output, "launch"; details=Dict("label" => label, "workers" => p))
    Study.launch(output, script, p, cpus, args)
    event(output, "completed"; details=Dict("label" => label, "workers" => p))
end

function workload_spec(workload)
    if workload == "fixed"
        return ("measure", "upwind",
                CheckedOrderedGatherDriver.make_checked_driver)
    end
    return ("measure_cfl", "upwind_cfl",
            CheckedCFLOrderedGatherDriver.make_checked_cfl_driver)
end

function checked_report(report, source, o, p, cpus, mode, kernel)
    for (key, value) in (("source",source), ("mode",mode), ("kernel",kernel),
                         ("n",o.n), ("block",o.block), ("steps",o.steps),
                         ("threads",p), ("blocks",(o.n÷o.block)^2),
                         ("faces",2*o.n*o.n), ("julia_version","1.10.12"))
        report[key] == value || error("report $key differs")
    end
    [entry["cpu"] for entry in report["affinity"]] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in report["affinity"]) ||
        error("worker affinity differs")
    if o.workload == "fixed"
        report["dt"] == 0.35/o.n || error("fixed step differs")
    else
        report["workload"] ==
            CheckedCFLOrderedGatherDriver.OrderedGatherCFLDriver.WORKLOAD &&
            report["cfl_scans_per_timed_step"] == 1 &&
            report["dt"] == 0.4*report["cfl_bound"] ||
            error("CFL-inclusive step definition differs")
    end
    elapsed, rate = report["time_seconds"], report["ns_per_unknown_step"]
    isfinite(elapsed) && elapsed > 0 && isfinite(rate) && rate > 0 &&
        isapprox(rate,1e9*elapsed/(o.n*o.n*o.steps); rtol=1e-12) ||
        error("timing normalization differs")
    return report
end

function locked_files(control, candidate, driver, initial, oracle)
    paths = (Study.JULIA, joinpath(Study.PROJECT,"Project.toml"),
             joinpath(Study.PROJECT,"Manifest.toml"), driver, initial, oracle,
             VALIDATOR, TEST_FILE, @__FILE__,
             joinpath(dirname(control),"suite-control.toml"),
             joinpath(dirname(control),"suite-$CANDIDATE.toml"),
             joinpath(dirname(control),"suite-provenance.toml"),
             joinpath(candidate,"scalar-selector.patch"),
             joinpath(candidate,"scalar-selector-receipt.json"),
             joinpath(@__DIR__, "make_probe.jl"),
             joinpath(@__DIR__, "..", "ordered_gather", "make_checked_driver.jl"),
             joinpath(@__DIR__, "..", "ordered_gather", "make_checked_cfl_driver.jl"),
             joinpath(@__DIR__, "..", "ordered_gather", "make_cfl_driver.jl"))
    return Dict(path => Support.sha256file(path) for path in paths),
        Dict("control" => Support.source_hash(control),
             CANDIDATE => Support.source_hash(candidate))
end

function validate_suite_report(suite, source, expected_sha, selector)
    advection = realpath(joinpath(source,Probe.RELATIVE))
    suite["status"] == "passed" && suite["order_checks"] == 2304 &&
        suite["packed_code_count"] == 256 &&
        suite["original_test_sha256"] == Support.sha256file(TEST_FILE) &&
        suite["loaded_advection_sha256"] == expected_sha &&
        suite["selector_expected"] == selector &&
        suite["ordered_sum_method_file"] == advection &&
        suite["loaded_advection_file"] == advection &&
        (selector ? suite["scalar_selector_method_file"] == advection :
                    isempty(suite["scalar_selector_method_file"])) ||
        error("copied-source suite receipt differs")
end

function suites(o, final_source, control, candidate, receipt, placement)
    if isnothing(o.suite_dir)
        for (name,source,p,expected,selector) in
            (("control",control,1,Probe.EXPECTED_CLEAN_SHA256,false),
             (CANDIDATE,candidate,4,receipt["candidate_source_sha256"],true))
            suite_path = joinpath(o.output,"suite-$name.toml")
            launch(o.output,VALIDATOR,p,placement[string(p)],
                   [source,TEST_FILE,expected,string(selector),suite_path],"suite-$name")
            validate_suite_report(TOML.parsefile(suite_path),source,expected,selector)
        end
        provenance = Dict("schema" => 1, "mode" => "fresh",
            "validator_sha256" => Support.sha256file(VALIDATOR),
            "test_sha256" => Support.sha256file(TEST_FILE),
            "control_report_sha256" =>
                Support.sha256file(joinpath(o.output,"suite-control.toml")),
            "candidate_report_sha256" =>
                Support.sha256file(joinpath(o.output,"suite-$CANDIDATE.toml")))
        open(joinpath(o.output,"suite-provenance.toml"),"w") do io
            TOML.print(io,provenance)
        end
        return provenance
    end
    original = TOML.parsefile(joinpath(o.suite_dir,"validation-runs.toml"))
    original["schema"] == 1 && original["status"] == "both passed" &&
        original["validator"] == VALIDATOR &&
        original["validator_sha256"] == Support.sha256file(VALIDATOR) &&
        original["test"] == TEST_FILE &&
        original["test_sha256"] == Support.sha256file(TEST_FILE) &&
        original["julia"] == Study.JULIA &&
        original["julia_sha256"] == Support.sha256file(Study.JULIA) &&
        original["project"] == Study.PROJECT &&
        original["depot"] == Study.DEPOT &&
        original["threads"] == 1 &&
        original["assertions_per_run"] == 14604 ||
        error("reused suite environment differs")
    length(original["runs"]) == 2 || error("reused suite run count differs")
    for (name,source,expected,selector,original_source) in
        (("control",control,Probe.EXPECTED_CLEAN_SHA256,false,final_source),
         (CANDIDATE,candidate,receipt["candidate_source_sha256"],true,o.suite_dir))
        run = only(run for run in original["runs"] if run["variant"] == name)
        run["source"] == original_source && run["expect_selector"] == selector &&
            run["source_advection_sha256"] == expected &&
            run["source_tree_sha256"] == Support.source_hash(source) &&
            run["source_tree_sha256"] == Support.source_hash(original_source) &&
            run["exit_code"] == 0 || error("$name reused suite source differs")
        original_report = joinpath(o.suite_dir,run["report"])
        Support.sha256file(original_report) == run["report_sha256"] ||
            error("$name reused suite report hash differs")
        validate_suite_report(TOML.parsefile(original_report),original_source,
                              expected,selector)
        cp(original_report,joinpath(o.output,"suite-$name.toml"))
    end
    cp(joinpath(o.suite_dir,"validation-runs.toml"),
       joinpath(o.output,"suite-provenance.toml"))
    event(o.output,"suites_reused"; details=Dict(
        "from" => o.suite_dir,
        "provenance_sha256" => Support.sha256file(joinpath(o.output,"suite-provenance.toml"))))
    return original
end

function assert_locked(expected_files, expected_sources, control, candidate,
                       driver, initial, oracle)
    files, sources = locked_files(control,candidate,driver,initial,oracle)
    files == expected_files && sources == expected_sources ||
        error("frozen source, driver, environment, or field changed during study")
end

function run_one(o, driver, source, variant, p, repeat, cpus, mode, kernel,
                 initial, oracle, oracle_hash, expected_files, expected_sources)
    assert_locked(expected_files, expected_sources,
                  joinpath(o.output,"control"), joinpath(o.output,CANDIDATE),
                  driver, initial, oracle)
    stem = "$variant-n$(o.n)-b$(o.block)-p$p-r$repeat"
    field = joinpath(o.output, "$stem.bin")
    report_path = joinpath(o.output, "$stem.toml")
    launch(o.output, driver, p, cpus,
        ["source=$source", "mode=$mode", "kernel=$kernel",
         "n=$(o.n)", "block=$(o.block)", "steps=$(o.steps)",
         "cpus=$(join(cpus, ','))", "input=$initial",
         "output=$field", "report=$report_path"], stem)
    row = checked_report(TOML.parsefile(report_path), source, o, p, cpus, mode, kernel)
    row["field_sha256"] == Support.sha256file(field) || error("$stem field report differs")
    comparison = Study.full_field(field, oracle, o.n)
    merge!(row, comparison)
    merge!(row, Dict("variant" => variant, "repeat" => repeat,
                     "initial_sha256" => Support.sha256file(initial),
                     "oracle_sha256" => oracle_hash,
                     "source_sha256" => Support.sha256file(joinpath(source, Probe.RELATIVE))))
    Support.appendjsonl(joinpath(o.output,"raw.jsonl"), row)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
        comparison["max_abs_vs_oracle"] == 0 ||
        error("$stem differs from fresh full-field oracle")
    row["time_seconds"] >= 0.2 || error("$stem shorter than 0.2 seconds")
    assert_locked(expected_files, expected_sources,
                  joinpath(o.output,"control"), joinpath(o.output,CANDIDATE),
                  driver, initial, oracle)
    rm(field)
    return row
end

function summarize(o, rows)
    groups = Any[]
    for p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
            row["variant"] == "control" && row["threads"] == p &&
            row["repeat"] == r) for r in 0:o.repeats-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
            row["variant"] == CANDIDATE && row["threads"] == p &&
            row["repeat"] == r) for r in 0:o.repeats-1]
        cmed, xmed = Support.median_value(control), Support.median_value(candidate)
        push!(groups, Dict("workers" => p, "control_samples" => control,
            "candidate_samples" => candidate,
            "control_median_ns_per_unknown_step" => cmed,
            "candidate_median_ns_per_unknown_step" => xmed,
            "control_over_candidate_ratio" => cmed/xmed,
            "paired_ratios" => control ./ candidate))
    end
    serial = min(groups[1]["control_median_ns_per_unknown_step"],
                 groups[1]["candidate_median_ns_per_unknown_step"])
    for group in groups
        group["strongest_serial_efficiency"] = serial /
            (group["workers"]*group["candidate_median_ns_per_unknown_step"])
    end
    return Dict("schema" => 1, "case" => [o.n,o.block],
                "workload" => o.workload, "steps" => o.steps,
                "repeats" => o.repeats, "samples" => length(rows),
                "minimum_seconds" => 0.2, "groups" => groups,
                "scope" => "uniform periodic exploratory fresh-process copied-source pairs")
end

function main(args=ARGS)
    o = options(args)
    Study.verify_control()
    final_source = joinpath(COHORT, "current-source")
    Support.sha256file(joinpath(final_source, Probe.RELATIVE)) ==
        Probe.EXPECTED_CLEAN_SHA256 || error("frozen narrow-ID source differs")
    placement = Support.readjson(joinpath(COHORT,"placement.json"))
    mkpath(o.output)
    control = joinpath(o.output,"control")
    candidate = joinpath(o.output,CANDIDATE)
    mkpath(control)
    cp(joinpath(final_source,"src"), joinpath(control,"src"))
    receipt = Probe.make_probe(final_source, candidate)
    Support.source_hash(control) == receipt["clean_tree_sha256"] &&
        Support.source_hash(candidate) == receipt["candidate_tree_sha256"] ||
        error("copied source trees differ from generator receipt")
    suite_receipt = suites(o,final_source,control,candidate,receipt,placement)
    mode,kernel,make_driver = workload_spec(o.workload)
    driver = joinpath(o.output,"checked_driver.jl")
    make_driver(driver)
    initial = joinpath(o.output,"initial.bin")
    cp(joinpath(COHORT,"initial-n$(o.n).bin"), initial)
    filesize(initial) == 8*o.n*o.n || error("initial field byte count differs")
    oracle = joinpath(o.output,"oracle.bin")
    oracle_report = joinpath(o.output,"oracle.toml")
    launch(o.output,driver,1,placement["1"],
        ["source=$control", "mode=$mode", "kernel=$kernel",
         "n=$(o.n)", "block=$(o.block)", "steps=$(o.steps)",
         "cpus=$(join(placement["1"],','))", "input=$initial",
         "output=$oracle", "report=$oracle_report"],"oracle")
    oracle_row = checked_report(TOML.parsefile(oracle_report), control, o,
                                1, placement["1"], mode, kernel)
    oracle_hash = Support.sha256file(oracle)
    oracle_row["field_sha256"] == oracle_hash &&
        all(isfinite,reinterpret(Float64,read(oracle))) || error("oracle field invalid")
    event(o.output,"oracle_validated"; details=Dict("field_sha256" => oracle_hash))
    files,sources = locked_files(control,candidate,driver,initial,oracle)
    Support.writejson(joinpath(o.output,"configuration.json"),Dict(
        "schema" => 1, "case" => [o.n,o.block], "steps" => o.steps,
        "repeats" => o.repeats, "workload" => o.workload,
        "minimum_seconds" => 0.2, "mode" => mode, "kernel" => kernel,
        "control_advection_sha256" => Probe.EXPECTED_CLEAN_SHA256,
        "candidate_advection_sha256" => receipt["candidate_source_sha256"],
        "source_tree_sha256" => sources, "locked_files_sha256" => files,
        "probe_receipt" => receipt, "suite_provenance" => suite_receipt,
        "placement" => placement,
        "julia" => Study.JULIA, "project" => Study.PROJECT,
        "depot" => Study.DEPOT, "oracle_sha256" => oracle_hash))
    jobs = [(repeat,p) for repeat in 0:o.repeats-1 for p in WORKERS]
    shuffle!(MersenneTwister(20261011+o.block+o.steps), jobs)
    Support.writejson(joinpath(o.output,"job_order.json"),
        [Dict("repeat" => repeat, "workers" => p,
              "order" => collect(iseven(repeat+p) ? ("control",CANDIDATE) :
                                                   (CANDIDATE,"control")))
         for (repeat,p) in jobs])
    rows = Any[]
    for (repeat,p) in jobs
        order = iseven(repeat+p) ? ("control",CANDIDATE) : (CANDIDATE,"control")
        for variant in order
            source = variant == "control" ? control : candidate
            row = run_one(o,driver,source,variant,p,repeat,placement[string(p)],
                          mode,kernel,initial,oracle,oracle_hash,files,sources)
            push!(rows,row)
            println("$variant n$(o.n) b$(o.block) $mode p$p r$repeat: ",
                    round(row["ns_per_unknown_step"];digits=3), " ns/unknown-step")
            flush(stdout)
        end
    end
    Support.writejson(joinpath(o.output,"summary.json"),summarize(o,rows))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
