#!/usr/bin/env julia
"""Reusable checked copied-source Stage 3 study for prepared probe sources."""
module CopiedSourceStudy

include(joinpath(@__DIR__, "..", "scalar_selector", "study.jl"))
using .ScalarSelectorStudy
using Random
using TOML

const Shared = ScalarSelectorStudy
const Study = Shared.Study
const Support = Shared.Support
const RELATIVE = joinpath("src","operators","advection","cell_advection.jl")
const COHORT = Shared.COHORT
const TEST_FILE = Shared.TEST_FILE
const WORKERS = (1,4,8)

function options(args)
    keys = args[1:2:end]
    length(args) == 18 && length(unique(keys)) == 9 ||
        error("usage: study.jl --case 1024:8 --steps N --repeats N --workload fixed|cfl --candidate-source DIR --candidate-label LABEL --suite-dir DIR --probe-receipt FILE --output NEW_DIR")
    o = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    Set(keys) == Set(("--case","--steps","--repeats","--workload",
                       "--candidate-source","--candidate-label","--suite-dir",
                       "--probe-receipt","--output")) || error("invalid options")
    pair = parse.(Int,split(o["--case"],':'))
    length(pair) == 2 || error("case must be N:BLOCK")
    n,block = pair
    n == 1024 && block in (8,1024) || error("only 1024:8 and 1024:1024 are supported")
    steps,repeats = parse(Int,o["--steps"]),parse(Int,o["--repeats"])
    steps >= 1 && repeats >= 1 || error("positive steps/repeats required")
    workload = o["--workload"]
    workload in ("fixed","cfl") || error("workload must be fixed or cfl")
    label = o["--candidate-label"]
    occursin(r"^[a-z][a-z0-9_]*$",label) && label != "control" ||
        error("invalid candidate label")
    candidate_source = abspath(o["--candidate-source"])
    suite_dir = abspath(o["--suite-dir"])
    receipt_path = abspath(o["--probe-receipt"])
    output = abspath(o["--output"])
    isdir(candidate_source) && isdir(suite_dir) && isfile(receipt_path) ||
        error("prepared candidate, suite directory, or probe receipt missing")
    !ispath(output) || error("output must be a new directory")
    return (;n,block,steps,repeats,workload,label,candidate_source,
            suite_dir,receipt_path,output)
end

function prepare_sources(o, frozen)
    control = joinpath(o.output,"control")
    candidate = joinpath(o.output,o.label)
    mkpath(control)
    mkpath(candidate)
    cp(joinpath(frozen,"src"),joinpath(control,"src"))
    cp(joinpath(o.candidate_source,"src"),joinpath(candidate,"src"))
    receipt = Support.readjson(o.receipt_path)
    receipt["source"] == frozen && receipt["output"] == o.candidate_source &&
        receipt["clean_source_sha256"] == Shared.Probe.EXPECTED_CLEAN_SHA256 &&
        receipt["clean_tree_sha256"] == Support.source_hash(frozen) &&
        receipt["candidate_source_sha256"] ==
            Support.sha256file(joinpath(candidate,RELATIVE)) &&
        receipt["candidate_tree_sha256"] == Support.source_hash(candidate) &&
        Support.source_hash(control) == receipt["clean_tree_sha256"] ||
        error("prepared probe source identity differs")
    patch = only(path for path in readdir(o.candidate_source;join=true)
                 if endswith(path,".patch"))
    Support.sha256file(patch) == receipt["patch_sha256"] ||
        error("prepared probe patch hash differs")
    cp(o.receipt_path,joinpath(o.output,"probe-receipt.json"))
    cp(patch,joinpath(o.output,"probe.patch"))
    return control,candidate,receipt
end

function reused_suites(o, frozen, control, candidate, receipt)
    manifest_path = joinpath(o.suite_dir,"validation-runs.toml")
    manifest = TOML.parsefile(manifest_path)
    validator = manifest["validator"]
    manifest["schema"] == 1 && manifest["status"] == "both passed" &&
        isfile(validator) && manifest["validator_sha256"] == Support.sha256file(validator) &&
        manifest["test"] == TEST_FILE &&
        manifest["test_sha256"] == Support.sha256file(TEST_FILE) &&
        manifest["julia"] == Study.JULIA &&
        manifest["julia_sha256"] == Support.sha256file(Study.JULIA) &&
        manifest["project"] == Study.PROJECT && manifest["depot"] == Study.DEPOT &&
        manifest["threads"] == 1 && manifest["assertions_per_run"] > 0 ||
        error("prepared validation environment differs")
    for (key,path) in (("project_toml_sha256",joinpath(Study.PROJECT,"Project.toml")),
                       ("project_manifest_sha256",joinpath(Study.PROJECT,"Manifest.toml")),
                       ("focused_test_sha256",get(manifest,"focused_test","")),
                       ("candidate_receipt_sha256",o.receipt_path),
                       ("candidate_patch_sha256",only(path for path in
                           readdir(o.candidate_source;join=true) if endswith(path,".patch"))))
        if haskey(manifest,key)
            isfile(path) && manifest[key] == Support.sha256file(path) ||
                error("prepared validation input differs: $key")
        end
    end
    if haskey(manifest,"generator_sha256")
        manifest["generator_sha256"] == receipt["generator_sha256"] ||
            error("prepared probe generator hash differs")
    end
    length(manifest["runs"]) == 2 || error("prepared suite run count differs")
    for (name,prepared,copied,expected,flag) in
        (("control",frozen,control,Shared.Probe.EXPECTED_CLEAN_SHA256,false),
         (o.label,o.candidate_source,candidate,receipt["candidate_source_sha256"],true))
        run = only(row for row in manifest["runs"] if row["variant"] == name)
        run["source"] == prepared && run["source_advection_sha256"] == expected &&
            run["source_tree_sha256"] == Support.source_hash(prepared) &&
            run["source_tree_sha256"] == Support.source_hash(copied) &&
            run["exit_code"] == 0 || error("$name prepared suite source differs")
        for (key,value) in run
            if startswith(key,"expect_")
                value isa Bool && value == flag ||
                    error("$name suite expectation differs: $key")
            end
        end
        report_path = joinpath(o.suite_dir,run["report"])
        report = TOML.parsefile(report_path)
        advection = realpath(joinpath(prepared,RELATIVE))
        Support.sha256file(report_path) == run["report_sha256"] &&
            report["status"] == "passed" &&
            report["original_test_sha256"] == Support.sha256file(TEST_FILE) &&
            report["loaded_advection_sha256"] == expected &&
            report["loaded_advection_file"] == advection &&
            report["copied_module_file"] ==
                realpath(joinpath(prepared,"src","TwoPhaseAMR.jl")) &&
            report["julia_version"] == "1.10.12" &&
            report["threads"] == 1 ||
            error("$name prepared suite report differs")
        method_keys = [key for key in keys(report) if endswith(key,"_method_file")]
        isempty(method_keys) && error("$name suite reports no loaded method path")
        for key in method_keys
            expected_method = (key == "ordered_sum_method_file" || flag) ?
                advection : ""
            report[key] == expected_method ||
                error("$name loaded method path differs: $key")
        end
        for (key,value) in report
            if endswith(key,"_expected") && value isa Bool
                value == flag || error("$name suite expectation differs: $key")
            elseif endswith(key,"_test_sha256") &&
                   key != "original_test_sha256" &&
                   key != "adapted_test_sha256"
                file_key = replace(key,"_sha256" => "_file")
                haskey(report,file_key) && isfile(report[file_key]) &&
                    value == Support.sha256file(report[file_key]) ||
                    error("$name focused test hash differs: $key")
            end
        end
        cp(report_path,joinpath(o.output,"suite-$name.toml"))
    end
    for threaded in get(manifest,"threaded_reports",Any[])
        threaded["exit_code"] == 0 && threaded["threads"] >= 1 ||
            error("threaded suite receipt differs")
        source_report = joinpath(o.suite_dir,threaded["report"])
        Support.sha256file(source_report) == threaded["report_sha256"] ||
            error("threaded suite report hash differs")
        report = TOML.parsefile(source_report)
        report["status"] == "passed" &&
            report["threads"] == threaded["threads"] &&
            report["loaded_advection_sha256"] == receipt["candidate_source_sha256"] ||
            error("threaded suite report differs")
        for (key,value) in report
            if endswith(key,"_expected") && value isa Bool
                value == true || error("threaded suite expectation differs: $key")
            end
        end
        cp(source_report,joinpath(o.output,"suite-threaded-p$(threaded["threads"]).toml"))
    end
    cp(manifest_path,joinpath(o.output,"suite-provenance.toml"))
    Shared.event(o.output,"suites_reused";details=Dict(
        "from" => o.suite_dir,
        "provenance_sha256" => Support.sha256file(joinpath(o.output,"suite-provenance.toml"))))
    return manifest
end

function locked(o,control,candidate,driver,initial,oracle)
    validator = TOML.parsefile(joinpath(o.output,"suite-provenance.toml"))["validator"]
    paths = String[Study.JULIA,joinpath(Study.PROJECT,"Project.toml"),
             joinpath(Study.PROJECT,"Manifest.toml"),
             @__FILE__,joinpath(@__DIR__,"..","scalar_selector","study.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_checked_driver.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_checked_cfl_driver.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_cfl_driver.jl"),
             TEST_FILE,validator,driver,initial,oracle,
             joinpath(o.output,"probe-receipt.json"),joinpath(o.output,"probe.patch"),
             joinpath(o.output,"suite-provenance.toml"),
             joinpath(o.output,"suite-control.toml"),
             joinpath(o.output,"suite-$(o.label).toml")]
    for variant in ("control",o.label)
        report = TOML.parsefile(joinpath(o.output,"suite-$variant.toml"))
        append!(paths,[value for (key,value) in report if
                       endswith(key,"_test_file") && value isa String])
    end
    provenance = TOML.parsefile(joinpath(o.output,"suite-provenance.toml"))
    for threaded in get(provenance,"threaded_reports",Any[])
        push!(paths,joinpath(o.output,"suite-threaded-p$(threaded["threads"]).toml"))
    end
    files = Dict(path => Support.sha256file(path) for path in paths)
    sources = Dict("control" => Support.source_hash(control),
                   o.label => Support.source_hash(candidate))
    return files,sources
end

function check_locked(o,control,candidate,driver,initial,oracle,files,sources)
    current_files,current_sources = locked(o,control,candidate,driver,initial,oracle)
    current_files == files && current_sources == sources ||
        error("copied source, driver, environment, or field changed during study")
end

function measure_one(o,control,candidate,driver,initial,oracle,oracle_hash,
                     mode,kernel,placement,files,sources,variant,p,repeat)
    check_locked(o,control,candidate,driver,initial,oracle,files,sources)
    source = variant == "control" ? control : candidate
    cpus = placement[string(p)]
    stem = "$variant-n$(o.n)-b$(o.block)-p$p-r$repeat"
    field = joinpath(o.output,"$stem.bin")
    report_path = joinpath(o.output,"$stem.toml")
    Shared.launch(o.output,driver,p,cpus,
        ["source=$source","mode=$mode","kernel=$kernel",
         "n=$(o.n)","block=$(o.block)","steps=$(o.steps)",
         "cpus=$(join(cpus,','))","input=$initial",
         "output=$field","report=$report_path"],stem)
    report = Shared.checked_report(TOML.parsefile(report_path),source,o,p,cpus,
                                   mode,kernel)
    report["field_sha256"] == Support.sha256file(field) || error("$stem report hash differs")
    comparison = Study.full_field(field,oracle,o.n)
    merge!(report,comparison)
    merge!(report,Dict("variant" => variant,"repeat" => repeat,
                       "initial_sha256" => Support.sha256file(initial),
                       "oracle_sha256" => oracle_hash,
                       "source_sha256" => Support.sha256file(joinpath(source,RELATIVE))))
    Support.appendjsonl(joinpath(o.output,"raw.jsonl"),report)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
        comparison["max_abs_vs_oracle"] == 0 ||
        error("$stem differs from fresh full-field oracle")
    report["time_seconds"] >= 0.2 || error("$stem timed interval below 0.2 seconds")
    check_locked(o,control,candidate,driver,initial,oracle,files,sources)
    rm(field)
    return report
end

function summary(o,rows)
    groups = Any[]
    for p in WORKERS
        control = [only(row["ns_per_unknown_step"] for row in rows if
            row["variant"] == "control" && row["threads"] == p &&
            row["repeat"] == repeat) for repeat in 0:o.repeats-1]
        candidate = [only(row["ns_per_unknown_step"] for row in rows if
            row["variant"] == o.label && row["threads"] == p &&
            row["repeat"] == repeat) for repeat in 0:o.repeats-1]
        cmed,xmed = Support.median_value(control),Support.median_value(candidate)
        push!(groups,Dict("workers" => p,"control_samples" => control,
            "candidate_samples" => candidate,"control_median_ns_per_unknown_step" => cmed,
            "candidate_median_ns_per_unknown_step" => xmed,
            "control_over_candidate_ratio" => cmed/xmed,
            "paired_ratios" => control ./ candidate))
    end
    serial = min(groups[1]["control_median_ns_per_unknown_step"],
                 groups[1]["candidate_median_ns_per_unknown_step"])
    for group in groups
        group["strongest_serial_efficiency"] =
            serial/(group["workers"]*group["candidate_median_ns_per_unknown_step"])
    end
    return Dict("schema" => 1,"candidate_label" => o.label,
        "case" => [o.n,o.block],"steps" => o.steps,"repeats" => o.repeats,
        "workload" => o.workload,"minimum_seconds" => 0.2,
        "samples" => length(rows),"groups" => groups,
        "scope" => "uniform periodic exploratory fresh-process copied-source pairs")
end

function main(args=ARGS)
    o = options(args)
    Study.verify_control()
    frozen = joinpath(COHORT,"current-source")
    Support.sha256file(joinpath(frozen,RELATIVE)) ==
        Shared.Probe.EXPECTED_CLEAN_SHA256 || error("frozen baseline differs")
    placement = Support.readjson(joinpath(COHORT,"placement.json"))
    mkpath(o.output)
    control,candidate,receipt = prepare_sources(o,frozen)
    validation = reused_suites(o,frozen,control,candidate,receipt)
    mode,kernel,make_driver = Shared.workload_spec(o.workload)
    driver = joinpath(o.output,"checked_driver.jl")
    make_driver(driver)
    initial = joinpath(o.output,"initial.bin")
    cp(joinpath(COHORT,"initial-n$(o.n).bin"),initial)
    filesize(initial) == 8*o.n*o.n || error("initial field byte count differs")
    oracle = joinpath(o.output,"oracle.bin")
    oracle_report = joinpath(o.output,"oracle.toml")
    Shared.launch(o.output,driver,1,placement["1"],
        ["source=$control","mode=$mode","kernel=$kernel",
         "n=$(o.n)","block=$(o.block)","steps=$(o.steps)",
         "cpus=$(join(placement["1"],','))","input=$initial",
         "output=$oracle","report=$oracle_report"],"oracle")
    oracle_row = Shared.checked_report(TOML.parsefile(oracle_report),control,o,1,
                                       placement["1"],mode,kernel)
    oracle_hash = Support.sha256file(oracle)
    oracle_row["field_sha256"] == oracle_hash &&
        all(isfinite,reinterpret(Float64,read(oracle))) || error("oracle field invalid")
    Shared.event(o.output,"oracle_validated";details=Dict("field_sha256" => oracle_hash))
    files,sources = locked(o,control,candidate,driver,initial,oracle)
    Support.writejson(joinpath(o.output,"configuration.json"),Dict(
        "schema" => 1,"candidate_label" => o.label,
        "candidate_source" => o.candidate_source,"suite_dir" => o.suite_dir,
        "case" => [o.n,o.block],"steps" => o.steps,"repeats" => o.repeats,
        "workload" => o.workload,"mode" => mode,"kernel" => kernel,
        "minimum_seconds" => 0.2,"placement" => placement,
        "control_advection_sha256" => Shared.Probe.EXPECTED_CLEAN_SHA256,
        "candidate_advection_sha256" => receipt["candidate_source_sha256"],
        "probe_receipt" => receipt,"suite_provenance" => validation,
        "locked_files_sha256" => files,"source_tree_sha256" => sources,
        "oracle_sha256" => oracle_hash,"julia" => Study.JULIA,
        "project" => Study.PROJECT,"depot" => Study.DEPOT))
    jobs = [(repeat,p) for repeat in 0:o.repeats-1 for p in WORKERS]
    shuffle!(MersenneTwister(20261012+o.block+o.steps),jobs)
    Support.writejson(joinpath(o.output,"job_order.json"),
        [Dict("repeat" => repeat,"workers" => p,
              "order" => collect(iseven(repeat+p) ? ("control",o.label) :
                                                   (o.label,"control")))
         for (repeat,p) in jobs])
    rows = Any[]
    for (repeat,p) in jobs
        for variant in (iseven(repeat+p) ? ("control",o.label) : (o.label,"control"))
            row = measure_one(o,control,candidate,driver,initial,oracle,oracle_hash,
                              mode,kernel,placement,files,sources,variant,p,repeat)
            push!(rows,row)
            println("$variant n$(o.n) b$(o.block) $mode p$p r$repeat: ",
                    round(row["ns_per_unknown_step"];digits=3)," ns/unknown-step")
            flush(stdout)
        end
    end
    Support.writejson(joinpath(o.output,"summary.json"),summary(o,rows))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
