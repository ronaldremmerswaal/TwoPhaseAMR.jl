#!/usr/bin/env julia
module BlockInteriorStudy

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport
using JSON
using Random
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const BASELINE = joinpath(ROOT, "benchmark", "scaling", "results",
                          "20261010-stage3-narrow-ids-control")
const PROJECT = joinpath(BASELINE, "benchmark-project")
const JULIA = "/tmp/julia-1.10.12/bin/julia"
const DEPOT = "/tmp/twophase-julia110-depot:/home/ronald/.julia"
const DRIVER = joinpath(@__DIR__, "..", "main_driver.jl")
const TOY_DRIVER = joinpath(ROOT, "benchmark", "scaling", "cross_language", "julia_driver.jl")
const DEFAULT_CASES = [(1024,1024), (2048,2048), (1024,8)]
const CONTROL_REVISION = "5f0081d9b2677ddc676c855f9cb5db4d218165dd"
const CONTROL_ADVECTION_SHA256 =
    "cecb8463cccada2e16d90f5245b506842d9cc90f6df9c380bb69fee673e4f115"
const CONTROL_EMBEDDED_SHA256 =
    "bc9b95f79e4ac353ca5c1236ad138f8c9d5996953053c11a361b9054c86ff33e"

function verify_control()
    receipt = readjson(joinpath(BASELINE, "freeze.json"))
    receipt["git_revision"] == CONTROL_REVISION || error("wrong committed control revision")
    for (relative, expected) in receipt["files"]
        path = joinpath(BASELINE, relative)
        isfile(path) && sha256file(path) == expected ||
            error("committed control file changed: $relative")
    end
    source_hash(BASELINE) == receipt["source_tree_sha256"] ||
        error("committed control source tree changed")
    for (relative, expected) in
        (joinpath("src", "operators", "advection", "cell_advection.jl") =>
             CONTROL_ADVECTION_SHA256,
         joinpath("src", "indicator", "embedded_operators.jl") =>
             CONTROL_EMBEDDED_SHA256)
        sha256file(joinpath(BASELINE, relative)) == expected ||
            error("committed control source identity differs: $relative")
    end
    for (path, key) in
        ((DRIVER, "main_driver_sha256"),
         (joinpath(ROOT, "benchmark", "scaling", "Support.jl"), "support_sha256"),
         (joinpath(ROOT, "benchmark", "scaling", "Project.toml"),
          "tooling_project_sha256"),
         (joinpath(ROOT, "benchmark", "scaling", "Manifest.toml"),
          "tooling_manifest_sha256"),
         (JULIA, "numerical_julia_sha256"))
        sha256file(path) == receipt[key] || error("control tooling changed: $path")
    end
    return receipt
end

function parse_args(args)
    if any(arg -> arg in ("-h","--help"), args)
        println("usage: julia --project=benchmark/scaling ",
                "benchmark/scaling/stage3/narrow_ids/study.jl pilot|matrix ",
                "--output DIR [--pilot-steps N] [--steps-file FILE] ",
                "[--repeats N] [--min-seconds S] [--cases N:BLOCK ...]")
        return nothing
    end
    isempty(args) && error("usage: study.jl pilot|matrix --output DIR [options]")
    phase = popfirst!(args)
    phase in ("pilot", "matrix") || error("phase must be pilot or matrix")
    options = Dict{String,Any}("phase" => phase, "pilot_steps" => 16,
                               "repeats" => 5, "minimum" => 0.2,
                               "cases" => copy(DEFAULT_CASES))
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--cases"
            cases = Tuple{Int,Int}[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                pair = split(args[i], ':')
                length(pair) == 2 || error("case must be N:BLOCK")
                push!(cases, (parse(Int,pair[1]),parse(Int,pair[2])))
                i += 1
            end
            options["cases"] = cases
            continue
        end
        i < length(args) || error("missing value for $flag")
        value = args[i+1]
        if flag == "--output"
            options["output"] = abspath(value)
        elseif flag == "--steps-file"
            options["steps_file"] = abspath(value)
        elseif flag == "--pilot-steps"
            options["pilot_steps"] = parse(Int,value)
        elseif flag == "--repeats"
            options["repeats"] = parse(Int,value)
        elseif flag == "--min-seconds"
            options["minimum"] = parse(Float64,value)
        else
            error("unknown option: $flag")
        end
        i += 2
    end
    haskey(options,"output") || error("--output required")
    phase == "matrix" && !haskey(options,"steps_file") && error("matrix requires --steps-file")
    cases = options["cases"]
    isempty(cases) && error("at least one case required")
    length(unique(cases)) == length(cases) || error("duplicate cases")
    all(n >= 1 && b >= 1 && n % b == 0 for (n,b) in cases) || error("invalid N:BLOCK case")
    options["repeats"] >= 1 && options["pilot_steps"] >= 1 || error("positive counts required")
    isfinite(options["minimum"]) && options["minimum"] >= 0 ||
        error("--min-seconds must be finite and nonnegative")
    phase == "matrix" && options["minimum"] < 0.2 &&
        error("accepted matrix requires --min-seconds >= 0.2")
    return options
end

function source_files(root)
    paths = String[]
    for (directory, _, files) in walkdir(joinpath(root,"src"))
        append!(paths, joinpath(directory,file) for file in files if endswith(file,".jl"))
    end
    return sort!(paths)
end

function manifest()
    paths = vcat(source_files(ROOT), source_files(BASELINE),
                 [@__FILE__, joinpath(@__DIR__, "summarize.jl"),
                  joinpath(@__DIR__, "plot.jl"),
                  joinpath(@__DIR__, "make_worker_probe.jl"),
                  joinpath(@__DIR__, "worker_probe_study.jl"),
                  joinpath(@__DIR__, "setup_study.jl"),
                  joinpath(@__DIR__, "..", "..", "Support.jl"),
                  DRIVER, TOY_DRIVER, JULIA,
                  joinpath(PROJECT,"Project.toml"), joinpath(PROJECT,"Manifest.toml"),
                  joinpath(ROOT,"benchmark","scaling","Project.toml"),
                  joinpath(ROOT,"benchmark","scaling","Manifest.toml"),
                  joinpath(Sys.BINDIR,Base.julia_exename()),
                  joinpath(BASELINE,"freeze.json")])
    all(isfile, paths) || error("missing cohort source, environment, or executable")
    result = Dict{String,String}()
    for path in paths
        key = isabspath(path) && startswith(path, ROOT * "/") ? relpath(path,ROOT) : path
        result[key] = sha256file(path)
    end
    result["tooling_julia_version"] = string(VERSION)
    return result
end

function launch(output, script, workers, cpus, arguments)
    command = [JULIA, "--startup-file=no", "--project=$PROJECT",
               "--threads=$workers", "--gcthreads=1", script, arguments...]
    environment = Dict("JULIA_DEPOT_PATH" => DEPOT, "OPENBLAS_NUM_THREADS" => "1",
                       "MKL_NUM_THREADS" => "1", "JULIA_NUM_GC_THREADS" => "1")
    started = round(Int,time()*1e9)
    result = run_capture(command; env=environment, cpus=cpus)
    receipt = Dict("started_unix_ns" => started, "ended_unix_ns" => round(Int,time()*1e9),
                   "command" => command, "cpus" => cpus, "exit_code" => result.exitcode,
                   "stdout" => result.stdout, "stderr" => result.stderr,
                   "process_seconds" => result.seconds)
    appendjsonl(joinpath(output,"commands.jsonl"), receipt)
    if result.exitcode != 0
        failure = joinpath(output,"failure-$started.json")
        writejson(failure,receipt)
        error("Julia launch failed; see $failure")
    end
    return receipt
end

function full_field(actual, oracle, n)
    expected_bytes = 8n*n
    filesize(actual) == expected_bytes && filesize(oracle) == expected_bytes ||
        error("full-field byte count mismatch")
    a = reinterpret(Float64, read(actual))
    b = reinterpret(Float64, read(oracle))
    all(isfinite, a) && all(isfinite, b) || error("nonfinite field")
    maximum_error = 0.0
    violations = 0
    @inbounds for index in eachindex(a,b)
        delta = abs(a[index]-b[index])
        maximum_error = max(maximum_error,delta)
        violations += delta > 1e-12 + 1e-12abs(b[index])
    end
    return Dict("max_abs_vs_oracle" => maximum_error,
                "tolerance_violations" => violations,
                "bitwise_equal" => sha256file(actual) == sha256file(oracle))
end

function write_batch(path, cases)
    open(path,"w") do io
        TOML.print(io, Dict("cases" => cases))
    end
end

function fixture(output, n, block, steps, cpu)
    initial = joinpath(output,"initial-n$n.bin")
    if !isfile(initial)
        launch(output,TOY_DRIVER,1,cpu,
               ["mode=init", "nx=$n", "ny=$n", "output=$initial"])
    end
    filesize(initial) == 8n*n || error("incomplete initial field")
    oracle = joinpath(output,"oracle-n$n-b$block-s$steps.bin")
    report = joinpath(output,"oracle-n$n-b$block-s$steps.toml")
    if !isfile(oracle) || !isfile(report)
        launch(output,DRIVER,1,cpu,
               ["source=$BASELINE", "mode=measure", "kernel=upwind",
                "n=$n", "block=$block", "steps=$steps", "cpus=$(cpu[1])",
                "input=$initial", "output=$oracle", "report=$report"])
    end
    filesize(oracle) == 8n*n || error("incomplete baseline oracle")
    record = TOML.parsefile(report)
    record["source"] == BASELINE && record["field_sha256"] == sha256file(oracle) &&
        record["steps"] == steps && record["n"] == n && record["block"] == block ||
        error("baseline oracle identity or hash mismatch")
    return initial, oracle
end

function validate(output, variant, n, block, workers, repeat, steps, cpus,
                  initial, oracle, min_seconds)
    stem = "$variant-n$n-b$block-p$workers-r$repeat"
    field, report = joinpath(output,"$stem.bin"), joinpath(output,"$stem.toml")
    row = TOML.parsefile(report)
    for (key, expected) in ("n" => n, "block" => block, "steps" => steps,
                            "threads" => workers, "kernel" => "upwind")
        row[key] == expected || error("$stem: $key differs")
    end
    source = variant == "baseline" ? BASELINE : ROOT
    row["source"] == source || error("$stem: unexpected source")
    row["dt"] == 0.35/n && isfinite(row["time_seconds"]) || error("$stem: invalid time")
    affinity = row["affinity"]
    [entry["cpu"] for entry in affinity] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in affinity) ||
        error("$stem: worker affinity mismatch")
    row["time_seconds"] >= min_seconds || error("$stem: short timed interval")
    row["field_sha256"] == sha256file(field) || error("$stem: field hash mismatch")
    comparison = full_field(field,oracle,n)
    comparison["tolerance_violations"] == 0 && comparison["bitwise_equal"] &&
        row["field_sha256"] == sha256file(oracle) ||
        error("$stem: full-field mismatch")
    merge!(row,comparison)
    merge!(row, Dict("variant" => variant, "repeat" => repeat,
                     "initial_sha256" => sha256file(initial),
                     "oracle_sha256" => sha256file(oracle),
                     "source_sha256" => sha256file(joinpath(source,"src","operators",
                                                             "advection","cell_advection.jl"))))
    rm(field)
    return row
end

"""Recheck every retained accepted row before skipping it on cohort resume."""
function validate_prior_row(row, cases, steps_by_n, repeats, minimum, pins, output)
    variant = row["variant"]
    n, block, workers, repeat = row["n"], row["block"], row["threads"], row["repeat"]
    variant in ("baseline","current") && (n,block) in cases &&
        workers in (1,4,8) && 0 <= repeat < repeats || error("unexpected prior job key")
    steps = steps_by_n[string(n)]
    source = variant == "baseline" ? BASELINE : ROOT
    for (key, expected) in ("steps" => steps, "kernel" => "upwind",
                            "mode" => "measure", "source" => source,
                            "julia_version" => "1.10.12",
                            "blocks" => (n÷block)^2)
        row[key] == expected || error("prior row $key differs")
    end
    row["dt"] == 0.35/n || error("prior row timestep differs")
    seconds = row["time_seconds"]
    ns = row["ns_per_unknown_step"]
    isfinite(seconds) && seconds > 0 && seconds >= minimum && isfinite(ns) && ns > 0 &&
        isapprox(ns,1e9seconds/(n*n*steps);rtol=1e-12,atol=0) ||
        error("prior row timed interval differs")
    isfinite(row["checksum"]) && isfinite(row["sum_squares"]) ||
        error("prior row field reductions differ")
    cpus = pins[string(workers)]
    affinity = row["affinity"]
    [entry["cpu"] for entry in affinity] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in affinity) ||
        error("prior row affinity differs")
    row["source_sha256"] == sha256file(joinpath(source,"src","operators",
        "advection","cell_advection.jl")) || error("prior row source hash differs")
    initial = joinpath(output,"initial-n$n.bin")
    oracle = joinpath(output,"oracle-n$n-b$block-s$steps.bin")
    initial_hash,oracle_hash = sha256file(initial),sha256file(oracle)
    row["initial_sha256"] == initial_hash && row["oracle_sha256"] == oracle_hash &&
        row["field_sha256"] == oracle_hash && row["bitwise_equal"] &&
        row["tolerance_violations"] == 0 && row["max_abs_vs_oracle"] == 0 ||
        error("prior row field/oracle hashes differ")
    return (variant,n,block,workers,repeat)
end

function main(args=copy(ARGS))
    options = parse_args(args)
    options === nothing && return
    verify_control()
    phase, cases = options["phase"], options["cases"]
    steps_by_n = haskey(options, "steps_file") ?
        Dict(string(k) => Int(v) for (k,v) in readjson(options["steps_file"])) :
        Dict(string(n) => options["pilot_steps"] for (n,_) in cases)
    all(get(steps_by_n,string(n),0) >= 1 for (n,_) in cases) || error("missing fixed steps")
    output = options["output"]
    if isdir(output) && !isfile(joinpath(output,"manifest.json"))
        error("existing output directory has no cohort manifest")
    end
    mkpath(output)
    current_manifest = manifest()
    locked = joinpath(output,"manifest.json")
    repeats = phase == "pilot" ? 1 : options["repeats"]
    minimum = phase == "pilot" ? 0.0 : options["minimum"]
    allowed,nodes = topology()
    pins = Dict(string(p) => placement_mask(p,"spread",allowed,nodes) for p in (1,4,8))
    if isfile(locked)
        readjson(locked) == current_manifest || error("source or environment changed during cohort")
    else
        writejson(locked,current_manifest)
        mkpath(joinpath(output,"current-source"))
        cp(joinpath(ROOT,"src"),joinpath(output,"current-source","src"))
        writejson(joinpath(output,"configuration.json"),Dict(
            "phase" => phase, "cases" => [[n,b] for (n,b) in cases],
            "steps_by_n" => steps_by_n, "repeats" => repeats,
            "minimum_timed_seconds" => minimum, "baseline" => BASELINE,
            "control_revision" => CONTROL_REVISION,
            "control_advection_sha256" => CONTROL_ADVECTION_SHA256,
            "candidate_advection_sha256" => sha256file(joinpath(ROOT,"src",
                "operators","advection","cell_advection.jl")),
            "project" => PROJECT, "julia" => JULIA, "depot" => DEPOT,
            "tooling_julia" => joinpath(Sys.BINDIR,Base.julia_exename()),
            "tooling_julia_version" => string(VERSION),
            "machine" => machine_metadata(allowed,nodes),
            "velocity" => [0.7,-0.3], "dt" => "0.35/n",
            "method" => "general forward Euler upwind"))
    end
    config = readjson(joinpath(output,"configuration.json"))
    config["phase"] == phase && config["cases"] == [[n,b] for (n,b) in cases] &&
        config["steps_by_n"] == steps_by_n && config["repeats"] == repeats &&
        config["minimum_timed_seconds"] == minimum &&
        config["control_revision"] == CONTROL_REVISION &&
        config["control_advection_sha256"] == CONTROL_ADVECTION_SHA256 &&
        config["candidate_advection_sha256"] == sha256file(joinpath(ROOT,"src",
            "operators","advection","cell_advection.jl")) ||
        error("cohort configuration differs")
    detected_machine = machine_metadata(allowed,nodes)
    for key in ("allowed_cpus", "cpu_topology", "governors", "boost", "julia", "os")
        config["machine"][key] == detected_machine[key] ||
            error("machine $key changed during cohort")
    end
    placement_path = joinpath(output,"placement.json")
    if isfile(placement_path)
        readjson(placement_path) == pins || error("CPU placement changed during cohort")
    else
        writejson(placement_path,pins)
    end
    for (n,b) in cases
        fixture(output,n,b,steps_by_n[string(n)],pins["1"])
    end
    for n in unique(first.(cases))
        if (n,n) in cases && (n,8) in cases
            one = joinpath(output,"oracle-n$n-b$n-s$(steps_by_n[string(n)]).bin")
            many = joinpath(output,"oracle-n$n-b8-s$(steps_by_n[string(n)]).bin")
            full_field(one,many,n)["bitwise_equal"] || error("baseline layout oracles differ")
        end
    end
    raw = joinpath(output,"raw.jsonl")
    prior = readjsonl(raw)
    completed = Set(validate_prior_row(row,cases,steps_by_n,repeats,minimum,pins,output)
                    for row in prior)
    length(completed) == length(prior) || error("duplicate accepted rows")
    jobs = [(repeat,p,variant) for repeat in 0:(repeats-1) for p in (1,4,8)
            for variant in ("baseline","current")]
    shuffle!(MersenneTwister(20261010),jobs)
    for (repeat,p,variant) in jobs
        pending = [(n,b) for (n,b) in cases if
                   !((variant,n,b,p,repeat) in completed)]
        isempty(pending) && continue
        manifest() == current_manifest || error("source changed before launch")
        shuffle!(MersenneTwister(20261010 + 100repeat + p + (variant == "current")),pending)
        batch = Dict{String,Any}[]
        for (n,b) in pending
            stem = "$variant-n$n-b$b-p$p-r$repeat"
            push!(batch,Dict("mode" => "measure", "kernel" => "upwind",
                             "n" => n, "block" => b, "steps" => steps_by_n[string(n)],
                             "input" => joinpath(output,"initial-n$n.bin"),
                             "output" => joinpath(output,"$stem.bin"),
                             "report" => joinpath(output,"$stem.toml")))
        end
        batch_path = joinpath(output,"batch-$variant-p$p-r$repeat.toml")
        write_batch(batch_path,batch)
        source = variant == "baseline" ? BASELINE : ROOT
        launch(output,DRIVER,p,pins[string(p)],
               ["source=$source", "batch=$batch_path", "cpus=$(join(pins[string(p)],','))"])
        manifest() == current_manifest || error("source changed during launch")
        for (n,b) in pending
            initial,oracle = fixture(output,n,b,steps_by_n[string(n)],pins["1"])
            row = validate(output,variant,n,b,p,repeat,steps_by_n[string(n)],
                           pins[string(p)],initial,oracle,minimum)
            appendjsonl(raw,row)
            push!(completed,(variant,n,b,p,repeat))
            println("$variant n$n b$b p$p r$repeat: ",
                    round(row["ns_per_unknown_step"];digits=3), " ns/cell-step, ",
                    round(row["time_seconds"];digits=3),"s")
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
