#!/usr/bin/env julia
"""Five-process production-only CFL-inclusive Stage 3 measurement."""
module ProductionCFLStudy

include(joinpath(@__DIR__, "study.jl"))
using .CopiedSourceStudy
using Random
using TOML

const Shared = CopiedSourceStudy
const Study = Shared.Study
const Support = Shared.Support
const COHORT = Shared.COHORT
const RELATIVE = Shared.RELATIVE
const EXPECTED_SOURCE_SHA256 = Shared.Shared.Probe.EXPECTED_CLEAN_SHA256
const EXPECTED_TREE_SHA256 =
    "0bb8826fb52d143ae0fa5f84e043f4ed4d4feeadb4c7328d6fbda33cbf7095a2"
const WORKERS = (1,4,8)

function options(args)
    iseven(length(args)) && length(unique(args[1:2:end])) == length(args)÷2 ||
        error("usage: production_cfl_study.jl --phase setup|measure --case 1024:8 --steps N --repeats 5 --output NEW_DIR [--setup-gate PASSED_DIR]")
    o = Dict(args[i] => args[i+1] for i in 1:2:length(args))
    phase = get(o,"--phase","")
    required = Set(("--phase","--case","--steps","--repeats","--output"))
    Set(keys(o)) == (phase == "measure" ? union(required,Set(("--setup-gate",))) :
                     required) || error("invalid phase or options")
    pair = parse.(Int, split(o["--case"], ':'))
    length(pair) == 2 || error("case must be N:BLOCK")
    n,block = pair
    n == 1024 && block in (8,1024) || error("only 1024:8 and 1024:1024 are supported")
    steps,repeats = parse(Int,o["--steps"]),parse(Int,o["--repeats"])
    steps >= 1 && repeats == 5 || error("positive steps and exactly five repetitions required")
    output = abspath(o["--output"])
    !ispath(output) || error("output must be a new directory")
    setup_gate = phase == "measure" ? abspath(o["--setup-gate"]) : nothing
    phase == "setup" || isdir(setup_gate) || error("setup gate directory missing")
    return (;n,block,steps,repeats,output,phase,setup_gate,workload="cfl")
end

function locked(o,source,driver,initial,oracle)
    paths = String[Study.JULIA,joinpath(Study.PROJECT,"Project.toml"),
             joinpath(Study.PROJECT,"Manifest.toml"),@__FILE__,
             joinpath(@__DIR__,"study.jl"),
             joinpath(@__DIR__,"..","scalar_selector","study.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_checked_driver.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_cfl_driver.jl"),
             joinpath(@__DIR__,"..","ordered_gather","make_checked_cfl_driver.jl"),
             joinpath(COHORT,"placement.json"),
             driver,initial,oracle]
    if o.setup_gate !== nothing
        append!(paths,[joinpath(o.setup_gate,name) for name in
                       ("setup-summary.json","setup-rows.jsonl","commands.jsonl",
                        "checked_driver.jl")])
    end
    return Dict(path => Support.sha256file(path) for path in paths),
        Support.source_hash(source)
end

function check_locked(o,source,driver,initial,oracle,files,tree)
    current_files,current_tree = locked(o,source,driver,initial,oracle)
    current_files == files && current_tree == tree ||
        error("production source, driver, environment, or field changed during study")
    if o.setup_gate !== nothing
        Support.source_hash(joinpath(o.setup_gate,"control")) ==
            EXPECTED_TREE_SHA256 || error("setup-gate source changed during study")
    end
end

function run_setup(o,source,driver,placement)
    files = Dict(path => Support.sha256file(path) for path in
        (Study.JULIA,joinpath(Study.PROJECT,"Project.toml"),
         joinpath(Study.PROJECT,"Manifest.toml"),@__FILE__,
         joinpath(@__DIR__,"study.jl"),joinpath(COHORT,"placement.json"),driver))
    tree = Support.source_hash(source)
    for p in WORKERS
        current = Dict(path => Support.sha256file(path) for path in keys(files))
        current == files && Support.source_hash(source) == tree ||
            error("production setup inputs changed")
        report_path = joinpath(o.output,"setup-p$p.toml")
        Shared.Shared.launch(o.output,driver,p,placement[string(p)],
            ["source=$source","mode=setup","kernel=upwind_cfl","n=$(o.n)",
             "block=$(o.block)","report=$report_path"],"setup-p$p")
        report = TOML.parsefile(report_path)
        for (key,expected) in (("source",source),("mode","setup"),
                               ("kernel","upwind_cfl"),("n",o.n),
                               ("block",o.block),("threads",p),
                               ("blocks",(o.n÷o.block)^2),("faces",2*o.n*o.n))
            report[key] == expected || error("setup p$p report $key differs")
        end
        Support.appendjsonl(joinpath(o.output,"setup-rows.jsonl"),report)
    end
    current = Dict(path => Support.sha256file(path) for path in keys(files))
    current == files && Support.source_hash(source) == tree ||
        error("production setup inputs changed")
    Support.writejson(joinpath(o.output,"setup-summary.json"),Dict(
        "schema"=>1,"status"=>"passed","case"=>[o.n,o.block],
        "workers"=>collect(WORKERS),"source_advection_sha256"=>EXPECTED_SOURCE_SHA256,
        "source_tree_sha256"=>tree,"driver_sha256"=>Support.sha256file(driver),
        "julia_sha256"=>Support.sha256file(Study.JULIA),
        "project_toml_sha256"=>Support.sha256file(joinpath(Study.PROJECT,"Project.toml")),
        "project_manifest_sha256"=>Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml")),
        "placement_sha256"=>Support.sha256file(joinpath(COHORT,"placement.json")),
        "setup_rows_sha256"=>Support.sha256file(joinpath(o.output,"setup-rows.jsonl")),
        "commands_sha256"=>Support.sha256file(joinpath(o.output,"commands.jsonl")),
        "locked_files_sha256"=>files))
end

function verify_setup_gate(o,driver)
    gate = Support.readjson(joinpath(o.setup_gate,"setup-summary.json"))
    gate["status"] == "passed" && gate["case"] == [o.n,o.block] &&
        gate["workers"] == collect(WORKERS) &&
        gate["source_advection_sha256"] == EXPECTED_SOURCE_SHA256 &&
        gate["source_tree_sha256"] == EXPECTED_TREE_SHA256 &&
        gate["driver_sha256"] == Support.sha256file(driver) &&
        gate["julia_sha256"] == Support.sha256file(Study.JULIA) &&
        gate["project_toml_sha256"] ==
            Support.sha256file(joinpath(Study.PROJECT,"Project.toml")) &&
        gate["project_manifest_sha256"] ==
            Support.sha256file(joinpath(Study.PROJECT,"Manifest.toml")) &&
        gate["placement_sha256"] ==
            Support.sha256file(joinpath(COHORT,"placement.json")) &&
        gate["setup_rows_sha256"] ==
            Support.sha256file(joinpath(o.setup_gate,"setup-rows.jsonl")) &&
        gate["commands_sha256"] ==
            Support.sha256file(joinpath(o.setup_gate,"commands.jsonl")) &&
        Support.source_hash(joinpath(o.setup_gate,"control")) ==
            EXPECTED_TREE_SHA256 || error("production setup gate identity differs")
    all(isfile(path) && Support.sha256file(path) == hash for
        (path,hash) in gate["locked_files_sha256"]) ||
        error("production setup gate tooling changed")
    rows = Support.readjsonl(joinpath(o.setup_gate,"setup-rows.jsonl"))
    commands = Support.readjsonl(joinpath(o.setup_gate,"commands.jsonl"))
    length(rows) == length(commands) == length(WORKERS) &&
        all(command["exit_code"] == 0 for command in commands) &&
        [row["threads"] for row in rows] == collect(WORKERS) ||
        error("production setup gate coverage differs")
    return Support.sha256file(joinpath(o.setup_gate,"setup-summary.json"))
end

function validated_report(path,source,o,p,cpus,mode,kernel)
    row = Shared.Shared.checked_report(TOML.parsefile(path),source,o,p,cpus,
                                       mode,kernel)
    row["carrier_policy"] == "prescribed constant owned-face velocity" &&
        row["dt_policy"] ==
            "0.4 * stable_timestep(velocity, workspace; cfl=0.9) each step" &&
        isfinite(row["cfl_bound"]) && row["cfl_bound"] > 0 ||
        error("CFL-inclusive workload policy differs")
    return row
end

function main(args=ARGS)
    o = options(args)
    Study.verify_control()
    frozen = joinpath(COHORT,"current-source")
    Support.sha256file(joinpath(frozen,RELATIVE)) == EXPECTED_SOURCE_SHA256 &&
        Support.source_hash(frozen) == EXPECTED_TREE_SHA256 ||
        error("frozen production source differs")
    placement = Support.readjson(joinpath(COHORT,"placement.json"))
    mkpath(o.output)
    source = joinpath(o.output,"control")
    mkpath(source)
    cp(joinpath(frozen,"src"),joinpath(source,"src"))
    Support.source_hash(source) == EXPECTED_TREE_SHA256 ||
        error("copied production source differs")
    mode,kernel,make_driver = Shared.Shared.workload_spec(o.workload)
    driver = joinpath(o.output,"checked_driver.jl")
    make_driver(driver)
    if o.phase == "setup"
        run_setup(o,source,driver,placement)
        return nothing
    end
    gate_hash = verify_setup_gate(o,driver)
    initial = joinpath(o.output,"initial.bin")
    cp(joinpath(COHORT,"initial-n$(o.n).bin"),initial)
    filesize(initial) == 8*o.n*o.n || error("initial field byte count differs")
    oracle = joinpath(o.output,"oracle.bin")
    oracle_report = joinpath(o.output,"oracle.toml")
    cpus1 = placement["1"]
    Shared.Shared.launch(o.output,driver,1,cpus1,
        ["source=$source","mode=$mode","kernel=$kernel","n=$(o.n)",
         "block=$(o.block)","steps=$(o.steps)","cpus=$(join(cpus1,','))",
         "input=$initial","output=$oracle","report=$oracle_report"],"oracle")
    oracle_row = validated_report(oracle_report,source,o,1,cpus1,mode,kernel)
    oracle_hash = Support.sha256file(oracle)
    oracle_row["field_sha256"] == oracle_hash &&
        all(isfinite,reinterpret(Float64,read(oracle))) ||
        error("production oracle field invalid")
    Shared.Shared.event(o.output,"oracle_validated";
                        details=Dict("field_sha256" => oracle_hash))
    files,tree = locked(o,source,driver,initial,oracle)
    Support.writejson(joinpath(o.output,"configuration.json"),Dict(
        "schema"=>1,"case"=>[o.n,o.block],"steps"=>o.steps,
        "repeats"=>o.repeats,"workload"=>o.workload,"mode"=>mode,
        "kernel"=>kernel,"minimum_seconds"=>0.2,"placement"=>placement,
        "source_advection_sha256"=>EXPECTED_SOURCE_SHA256,
        "source_tree_sha256"=>tree,"locked_files_sha256"=>files,
        "setup_gate"=>o.setup_gate,"setup_gate_sha256"=>gate_hash,
        "oracle_sha256"=>oracle_hash,"julia"=>Study.JULIA,
        "project"=>Study.PROJECT,"depot"=>Study.DEPOT))
    jobs = [(repeat,p) for repeat in 0:o.repeats-1 for p in WORKERS]
    shuffle!(MersenneTwister(20261014+o.block+o.steps),jobs)
    Support.writejson(joinpath(o.output,"job_order.json"),
        [Dict("repeat"=>repeat,"workers"=>p) for (repeat,p) in jobs])
    rows = Any[]
    for (repeat,p) in jobs
        check_locked(o,source,driver,initial,oracle,files,tree)
        stem = "control-n$(o.n)-b$(o.block)-p$p-r$repeat"
        field = joinpath(o.output,"$stem.bin")
        report = joinpath(o.output,"$stem.toml")
        cpus = placement[string(p)]
        Shared.Shared.launch(o.output,driver,p,cpus,
            ["source=$source","mode=$mode","kernel=$kernel","n=$(o.n)",
             "block=$(o.block)","steps=$(o.steps)","cpus=$(join(cpus,','))",
             "input=$initial","output=$field","report=$report"],stem)
        row = validated_report(report,source,o,p,cpus,mode,kernel)
        row["cfl_bound"] == oracle_row["cfl_bound"] ||
            error("$stem CFL bound differs from oracle")
        row["field_sha256"] == Support.sha256file(field) ||
            error("$stem report field hash differs")
        comparison = Study.full_field(field,oracle,o.n)
        merge!(row,comparison)
        merge!(row,Dict("variant"=>"control","repeat"=>repeat,
                        "initial_sha256"=>Support.sha256file(initial),
                        "oracle_sha256"=>oracle_hash,
                        "source_sha256"=>Support.sha256file(joinpath(source,RELATIVE))))
        Support.appendjsonl(joinpath(o.output,"raw.jsonl"),row)
        comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
            comparison["max_abs_vs_oracle"] == 0 ||
            error("$stem differs from full-field oracle")
        row["time_seconds"] >= 0.2 || error("$stem timed interval below 0.2 seconds")
        check_locked(o,source,driver,initial,oracle,files,tree)
        rm(field)
        push!(rows,row)
        println("$stem: ",round(row["ns_per_unknown_step"];digits=3),
                " ns/unknown/complete step")
        flush(stdout)
    end
    groups = Any[]
    serial = nothing
    for p in WORKERS
        samples = [only(row["ns_per_unknown_step"] for row in rows if
                    row["threads"] == p && row["repeat"] == repeat)
                   for repeat in 0:o.repeats-1]
        median = Support.median_value(samples)
        p == 1 && (serial = median)
        push!(groups,Dict("workers"=>p,"samples"=>samples,
                          "median_ns_per_unknown_complete_step"=>median,
                          "parallel_efficiency"=>serial/(p*median)))
    end
    Support.writejson(joinpath(o.output,"summary.json"),Dict(
        "schema"=>1,"scope"=>"uniform periodic fixed-carrier per-step CFL plus advect production baseline",
        "case"=>[o.n,o.block],"steps"=>o.steps,"repeats"=>o.repeats,
        "workload"=>o.workload,"minimum_seconds"=>0.2,
        "samples"=>length(rows),"groups"=>groups))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
