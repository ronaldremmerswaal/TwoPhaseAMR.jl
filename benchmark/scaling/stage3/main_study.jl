"""Serialized, pinned, fresh-process Stage 3 main-package study."""
module MainStudy

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using JSON
using Random
using Statistics
using TOML

export ROOT, JULIA, DEPOT, FREEZE, FROZEN_SOURCE, FROZEN_PROJECT,
       TOY_DRIVER, DRIVER, COUNTS, SIZES, digest, source_manifest,
       run_julia, full_field, toy_fixture, upwind_fixture, sample_paths,
       collect_sample, main_sample, write_batch, main

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", "..", ".."))
const JULIA = "/tmp/julia-1.10.12/bin/julia"
const DEPOT = "/tmp/twophase-julia110-depot:/home/ronald/.julia"
const FREEZE = joinpath(ROOT, "benchmark", "scaling", "results",
                        "20260930-stage3-production-baseline-freeze")
const FROZEN_SOURCE = joinpath(FREEZE, "source-snapshot")
const FROZEN_PROJECT = joinpath(FREEZE, "frozen-project")
const TOY_DRIVER = joinpath(ROOT, "benchmark", "scaling", "cross_language", "julia_driver.jl")
const DRIVER = joinpath(HERE, "main_driver.jl")
const COUNTS = (1, 2, 4, 8, 16, 32)
const SIZES = (512, 1024, 2048)

digest(path) = sha256file(path)

function source_manifest()
    paths = String[]
    for directory in (joinpath(ROOT, "src"), joinpath(FROZEN_SOURCE, "src"),
                      joinpath(ROOT, "research", "UniformLWAdvection", "src"))
        for (base, _, names) in walkdir(directory)
            append!(paths, joinpath(base, name) for name in names if endswith(name, ".jl"))
        end
    end
    sort!(paths)
    append!(paths, [DRIVER, TOY_DRIVER, @__FILE__,
                    joinpath(HERE, "..", "Support.jl"),
                    joinpath(FROZEN_PROJECT, "Project.toml"),
                    joinpath(FROZEN_PROJECT, "Manifest.toml"), JULIA])
    return Dict((startswith(path, ROOT * "/") ? relpath(path, ROOT) : path) => digest(path)
                for path in paths)
end

function run_julia(driver, project, threads, arguments, cpus, log)
    command = [JULIA, "--startup-file=no", "--project=$project",
               "--threads=$threads", "--gcthreads=1", string(driver),
               string.(arguments)...]
    env = Dict("JULIA_DEPOT_PATH" => DEPOT, "OPENBLAS_NUM_THREADS" => "1",
               "MKL_NUM_THREADS" => "1", "JULIA_NUM_GC_THREADS" => "1")
    result = run_capture(command; env, cpus)
    receipt = Dict("command" => command, "wall_seconds" => result.seconds,
                   "exit_code" => result.exitcode, "stdout" => result.stdout,
                   "stderr" => result.stderr)
    receipt_dir = get(ENV, "STAGE3_FAILURE_DIR", "/tmp")
    mkpath(receipt_dir)
    appendjsonl(joinpath(receipt_dir, "commands.jsonl"), receipt)
    if result.exitcode != 0
        writejson(joinpath(receipt_dir, "failure-$(time_ns()).json"), receipt)
        error("Julia failed $(result.exitcode): $(join(command, ' '))\n" *
              "stdout:\n$(result.stdout)\nstderr:\n$(result.stderr)")
    end
    push!(log, receipt)
    return nothing
end

function full_field(actual, oracle, n)
    expected_bytes = n * n * sizeof(Float64)
    filesize(actual) == expected_bytes && filesize(oracle) == expected_bytes ||
        error("full-field byte count mismatch")
    a = reinterpret(Float64, read(actual))
    b = reinterpret(Float64, read(oracle))
    max_abs = 0.0
    violations = 0
    for index in eachindex(a)
        isfinite(a[index]) || error("nonfinite main field")
        delta = abs(a[index] - b[index])
        max_abs = max(max_abs, delta)
        violations += delta > 1e-12 + 1e-12 * abs(b[index])
    end
    return Dict("field_sha256" => digest(actual),
                "max_abs_vs_oracle" => max_abs,
                "tolerance_violations" => violations)
end

function toy_fixture(out, n, steps, cpus, log)
    initial = joinpath(out, "initial-$n.bin")
    oracle = joinpath(out, "toy-oracle-$n-s$steps.bin")
    report = joinpath(out, "toy-oracle-$n-s$steps.toml")
    if !isfile(initial)
        run_julia(TOY_DRIVER, FROZEN_PROJECT, 1,
                  ["mode=init", "nx=$n", "ny=$n", "output=$initial"], cpus[1:1], log)
    end
    if !isfile(oracle)
        run_julia(TOY_DRIVER, FROZEN_PROJECT, 1,
                  ["mode=oracle", "nx=$n", "ny=$n", "steps=$steps",
                   "input=$initial", "output=$oracle", "report=$report"],
                  cpus[1:1], log)
    end
    return initial, oracle
end

function upwind_fixture(out, n, block, steps, cpus, initial, log)
    field = joinpath(out, "upwind-oracle-$n-b$block-s$steps.bin")
    report = joinpath(out, "upwind-oracle-$n-b$block-s$steps.toml")
    if !isfile(field)
        run_julia(DRIVER, FROZEN_PROJECT, 1,
                  ["source=$FROZEN_SOURCE", "mode=measure", "kernel=upwind",
                   "n=$n", "block=$block", "steps=$steps", "cpus=$(cpus[1])",
                   "input=$initial", "output=$field", "report=$report"],
                  cpus[1:1], log)
    end
    return field
end

function sample_paths(out, variant, n, block, p, repeat)
    stem = "upwind-$variant-$n-b$block-p$p-r$repeat"
    return joinpath(out, "$stem.bin"), joinpath(out, "$stem.toml")
end

function collect_sample(out, variant, n, block, p, steps, cpus,
                        initial, oracle, repeat, log; minimum_seconds=0.0)
    source = variant == "frozen" ? FROZEN_SOURCE : ROOT
    field, report = sample_paths(out, variant, n, block, p, repeat)
    record = TOML.parsefile(report)
    expected = Dict("n" => n, "block" => block, "steps" => steps,
                    "threads" => p, "kernel" => "upwind")
    for (key, value) in expected
        get(record, key, nothing) == value ||
            error("sample $report: $key differs from requested $value")
    end
    record["dt"] == 0.35 / n || error("sample $report: timestep differs")
    affinity = get(record, "affinity", Any[])
    [entry["cpu"] for entry in affinity] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in affinity) ||
        error("sample $report: worker pinning differs")
    isfinite(record["time_seconds"]) && record["time_seconds"] > 0 ||
        error("sample $report: invalid timed duration")
    record["time_seconds"] >= minimum_seconds ||
        error("timed sample shorter than $(minimum_seconds)s, retained $field")
    record["field_sha256"] == digest(field) ||
        error("sample $report: field digest differs")
    merge!(record, Dict(
        "repeat" => repeat, "variant" => variant,
        "source_sha256" => digest(joinpath(source, "src", "TwoPhaseAMR.jl")),
        "advection_source_sha256" => digest(joinpath(source, "src", "operators", "advection", "cell_advection.jl")),
        "driver_sha256" => digest(DRIVER), "initial_sha256" => digest(initial),
        "toy_lw_reference_sha256" => digest(oracle),
    ))
    upwind_oracle = upwind_fixture(out, n, block, steps, cpus[1:1], initial, log)
    evidence = full_field(field, upwind_oracle, n)
    merge!(record, evidence)
    record["upwind_oracle_sha256"] = digest(upwind_oracle)
    evidence["tolerance_violations"] == 0 ||
        error("upwind full-field mismatch, retained $field")
    rm(field)
    return record
end

collect_sample(out, variant, n, block, p, steps, cpus,
               initial, oracle, repeat, log, minimum_seconds) =
    collect_sample(out, variant, n, block, p, steps, cpus,
                   initial, oracle, repeat, log; minimum_seconds)

function main_sample(out, variant, n, block, p, steps, cpus,
                     initial, oracle, repeat, log)
    source = variant == "frozen" ? FROZEN_SOURCE : ROOT
    field, report = sample_paths(out, variant, n, block, p, repeat)
    run_julia(DRIVER, FROZEN_PROJECT, p,
              ["source=$source", "mode=measure", "kernel=upwind", "n=$n",
               "block=$block", "steps=$steps", "cpus=$(join(cpus, ','))",
               "input=$initial", "output=$field", "report=$report"], cpus, log)
    return collect_sample(out, variant, n, block, p, steps, cpus,
                          initial, oracle, repeat, log)
end

function write_batch(path, cases)
    open(path, "w") do io
        for case in cases
            println(io, "[[cases]]")
            for (key, value) in case
                encoded = value isa Integer ? string(value) : JSON.json(string(value))
                println(io, "$key = $encoded")
            end
            println(io)
        end
    end
    return path
end

function parse_args(args)
    if isempty(args) || args[1] in ("--help", "-h")
        println("usage: julia main_study.jl pilot|matrix --output DIR [--steps-file JSON] [--repeats N] [--sizes N ...] [--counts P ...] [--layouts one|many|both] [--variant frozen|current|both]")
        return nothing
    end
    phase = args[1]
    phase in ("pilot", "matrix") || error("phase must be pilot or matrix")
    opts = Dict{String, Any}("phase" => phase, "repeats" => 5,
                             "sizes" => collect(SIZES), "pilot-steps" => 2,
                             "kernel" => "upwind", "min-seconds" => 0.2,
                             "variant" => "current", "counts" => collect(COUNTS),
                             "layouts" => "both")
    index = 2
    while index <= length(args)
        name = args[index]
        startswith(name, "--") || error("unexpected argument: $name")
        name = name[3:end]
        if name in ("sizes", "counts")
            values = Int[]
            index += 1
            while index <= length(args) && !startswith(args[index], "--")
                push!(values, parse(Int, args[index]))
                index += 1
            end
            isempty(values) && error("--$name needs at least one value")
            opts[name] = values
        else
            index < length(args) || error("--$name needs a value")
            value = args[index + 1]
            opts[name] = name in ("repeats", "pilot-steps") ? parse(Int, value) :
                         name == "min-seconds" ? parse(Float64, value) : value
            index += 2
        end
    end
    haskey(opts, "output") || error("--output is required")
    opts["kernel"] == "upwind" || error("only upwind kernel is supported")
    opts["variant"] in ("frozen", "current", "both") || error("invalid variant")
    opts["layouts"] in ("one", "many", "both") || error("invalid layouts")
    return opts
end

function main(args=ARGS)
    opts = parse_args(args)
    isnothing(opts) && return
    out = abspath(opts["output"])
    mkpath(out)
    ENV["STAGE3_FAILURE_DIR"] = out
    manifest = source_manifest()
    manifest_path = joinpath(out, "source-manifest.json")
    if isfile(manifest_path)
        readjson(manifest_path) == manifest ||
            error("source manifest changed since this output directory was created")
    else
        writejson(manifest_path, manifest)
        snapshot = joinpath(out, "source-snapshot")
        mkpath(snapshot)
        cp(joinpath(ROOT, "src"), joinpath(snapshot, "main-src"))
        cp(joinpath(ROOT, "research", "UniformLWAdvection", "src"),
           joinpath(snapshot, "toy-src"))
        cp(DRIVER, joinpath(snapshot, "main_driver.jl"))
        cp(TOY_DRIVER, joinpath(snapshot, "toy_driver.jl"))
        cp(@__FILE__, joinpath(snapshot, "main_study.jl"))
    end
    allowed, nodes = topology()
    placements = Dict(p => placement_mask(p, "spread", allowed, nodes)
                      for p in opts["counts"])
    writejson(joinpath(out, "placement.json"),
              Dict(string(p) => mask for (p, mask) in placements))
    log, rows = Any[], Any[]
    prior = Any[]
    layouts = Dict(n => opts["layouts"] == "one" ? (n,) :
                      opts["layouts"] == "many" ? (8,) : (n, 8)
                   for n in opts["sizes"])
    if opts["phase"] == "pilot"
        jobs = [(n, b, p, v) for n in opts["sizes"] for b in layouts[n]
                for p in opts["counts"] for v in
                    (opts["variant"] == "both" ? ("frozen", "current") : (opts["variant"],))]
        for (n, b, p, v) in jobs
            initial, oracle = toy_fixture(out, n, opts["pilot-steps"], placements[1], log)
            row = main_sample(out, v, n, b, p, opts["pilot-steps"],
                              placements[p], initial, oracle, 0, log)
            push!(rows, row)
            println("pilot $(opts["kernel"]) $v $n b$b p$p: $(row["time_seconds"])s/$(opts["pilot-steps"]) steps")
        end
    else
        haskey(opts, "steps-file") || error("matrix requires --steps-file")
        steps_by_size = Dict(parse(Int, key) => Int(value)
                             for (key, value) in readjson(opts["steps-file"]))
        layout_oracles = Dict{String, Any}()
        for n in opts["sizes"]
            steps = steps_by_size[n]
            initial, _ = toy_fixture(out, n, steps, placements[1], log)
            paths = Dict(b => upwind_fixture(out, n, b, steps, placements[1], initial, log)
                         for b in layouts[n])
            if length(paths) == 2
                one, many = paths[n], paths[8]
                comparison = full_field(one, many, n)
                comparison["tolerance_violations"] == 0 && digest(one) == digest(many) ||
                    error("frozen one/many layout oracle differs at $n")
                layout_oracles[string(n)] = Dict(
                    "one_sha256" => digest(one), "many_sha256" => digest(many),
                    "max_abs" => comparison["max_abs_vs_oracle"])
            end
        end
        writejson(joinpath(out, "layout-oracle-evidence.json"), layout_oracles)
        raw_path = joinpath(out, "raw.jsonl")
        prior = readjsonl(raw_path)
        completed = Set((row["repeat"], row["n"], row["block"],
                         row["threads"], row["variant"]) for row in prior)
        length(completed) == length(prior) || error("duplicate completed matrix configurations")
        rng = MersenneTwister(20260930)
        pairs = [(r, p) for r in 0:(opts["repeats"] - 1) for p in opts["counts"]]
        shuffle!(rng, pairs)
        batches = Tuple[]
        for (r, p) in pairs
            variants = opts["variant"] == "both" ? ["frozen", "current"] :
                       [opts["variant"]]
            shuffle!(rng, variants)
            for v in variants
                configs = [(n, b) for n in opts["sizes"] for b in layouts[n]
                           if (r, n, b, p, v) ∉ completed]
                shuffle!(rng, configs)
                isempty(configs) || push!(batches, (r, p, v, configs))
            end
        end
        for (index, (r, p, v, configs)) in enumerate(batches)
            source_manifest() == manifest || error("source changed during matrix")
            cases, fixtures = Any[], Tuple[]
            for (n, b) in configs
                steps = steps_by_size[n]
                initial, oracle = toy_fixture(out, n, steps, placements[1], log)
                upwind_fixture(out, n, b, steps, placements[1], initial, log)
                field, report = sample_paths(out, v, n, b, p, r)
                push!(cases, Dict("mode" => "measure", "kernel" => "upwind",
                                  "n" => n, "block" => b, "steps" => steps,
                                  "input" => initial, "output" => field, "report" => report))
                push!(fixtures, (n, b, steps, initial, oracle))
            end
            batch_path = joinpath(out, "batch-r$r-p$p-$v.toml")
            write_batch(batch_path, cases)
            source = v == "frozen" ? FROZEN_SOURCE : ROOT
            run_julia(DRIVER, FROZEN_PROJECT, p,
                      ["source=$source", "batch=$batch_path", "kernel=upwind",
                       "cpus=$(join(placements[p], ','))"], placements[p], log)
            source_manifest() == manifest || error("source changed during matrix batch")
            validated = [collect_sample(out, v, n, b, p, steps, placements[p],
                                        initial, oracle, r, log;
                                        minimum_seconds=opts["min-seconds"])
                         for (n, b, steps, initial, oracle) in fixtures]
            for row in validated
                appendjsonl(raw_path, row)
            end
            append!(rows, validated)
            println("batch $index/$(length(batches)) $v p$p repeat$r: " *
                    "$(length(validated)) cases, min $(minimum(row["time_seconds"] for row in validated))s")
        end
    end
    complete_rows = opts["phase"] == "matrix" ? [prior; rows] : rows
    writejson(joinpath(out, "$(opts["phase"]).json"), complete_rows)
    writejson(joinpath(out, "$(opts["phase"])-commands.json"), log)
end

end # module MainStudy

if abspath(PROGRAM_FILE) == @__FILE__
    MainStudy.main()
end
