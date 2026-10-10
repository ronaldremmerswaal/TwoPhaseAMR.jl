"""Serialized fresh-process pilot for the frozen toy and full-step tile pipeline."""
module CachePipelinePilot

include(joinpath(@__DIR__, "..", "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport
using Dates
using Random
using Statistics
using TOML

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", "..", "..", ".."))
const DRIVER = joinpath(HERE, "timing_driver.jl")
const TOY = joinpath(ROOT, "research", "UniformLWAdvection", "src", "UniformLWAdvection.jl")
const PIPELINE = joinpath(HERE, "CachePipeline.jl")
const POLYESTER_EXT = joinpath(ROOT, "research", "UniformLWAdvection", "ext",
                              "UniformLWAdvectionPolyesterExt.jl")
const SUPPORT = joinpath(ROOT, "benchmark", "scaling", "Support.jl")
const DEFAULT_STEPS = Dict(512=>512, 1024=>128, 2048=>32, 4096=>8)

utcstamp() = string(Dates.now(Dates.UTC), "Z")

function cli(args)
    known = Set(("output", "julia", "project", "depot", "sizes", "workers",
                 "pipeline-tiles", "toy-tile", "steps", "repeats", "seed",
                 "placement", "min-seconds", "cohort", "failed-receipt"))
    options = Dict{String,Vector{String}}()
    index = 1
    while index <= length(args)
        startswith(args[index], "--") || error("expected --option: $(args[index])")
        key = args[index][3:end]
        key in known || error("unknown option --$key")
        index += 1
        values = String[]
        while index <= length(args) && !startswith(args[index], "--")
            push!(values, args[index])
            index += 1
        end
        options[key] = values
    end
    return options
end

single(options, key, default=nothing) = begin
    values = get(options, key, default === nothing ? String[] : [string(default)])
    length(values) == 1 || error("--$key needs one value")
    only(values)
end

multi(options, key, default) = get(options, key, default)

function tile(value)
    dimensions = split(lowercase(value), 'x'; limit=2)
    length(dimensions) == 2 || error("tile must be WIDTHxHEIGHT")
    result = [parse(Int, part) for part in dimensions]
    minimum(result) > 0 || error("tile dimensions must be positive")
    return result
end

function parse_options(args)
    options = cli(args)
    output = abspath(single(options, "output"))
    julia = abspath(single(options, "julia", "/tmp/julia-1.10.12/bin/julia"))
    project = abspath(single(options, "project", "/tmp/twophase-compat110"))
    depot = abspath(single(options, "depot", "/tmp/twophase-julia110-depot"))
    sizes = parse.(Int, multi(options, "sizes", ["512", "1024", "2048", "4096"]))
    workers = parse.(Int, multi(options, "workers", ["1", "2", "4", "8", "16", "32"]))
    pipeline_tiles = tile.(multi(options, "pipeline-tiles", ["128x32"]))
    toy_tile = tile(single(options, "toy-tile", "512x16"))
    repeats = parse(Int, single(options, "repeats", "1"))
    seed = parse(Int, single(options, "seed", "30092027"))
    placement = single(options, "placement", "spread")
    min_seconds = parse(Float64, single(options, "min-seconds", "0"))
    ispath(output) && error("output already exists: $output")
    !isempty(sizes) && !isempty(workers) && minimum(sizes) > 0 &&
        minimum(workers) > 0 && repeats > 0 && min_seconds >= 0 ||
        error("sizes, workers, repeats, and minimum duration must be valid")
    placement in ("spread", "compact") || error("placement must be spread or compact")
    steps = copy(DEFAULT_STEPS)
    for value in multi(options, "steps", String[])
        pair = split(value, ':'; limit=2)
        length(pair) == 2 || error("steps must be N:STEPS")
        n, count = parse.(Int, pair)
        n > 0 && count > 0 || error("size and steps must be positive")
        steps[n] = count
    end
    all(haskey(steps, n) for n in sizes) || error("a step count is required for every size")
    return (; output, julia, project, depot, sizes, workers, pipeline_tiles,
            toy_tile, repeats, seed, placement, min_seconds,
            steps=Dict(n=>steps[n] for n in sizes))
end

function source_paths(options)
    paths = [TOY, POLYESTER_EXT, PIPELINE, DRIVER, @__FILE__,
             joinpath(HERE, "resume.jl"), joinpath(HERE, "summarize.jl"),
             joinpath(HERE, "README.md"), SUPPORT,
             joinpath(ROOT, "benchmark", "scaling", "Project.toml"),
             joinpath(ROOT, "benchmark", "scaling", "Manifest.toml"),
             joinpath(options.project, "Project.toml"),
             joinpath(options.project, "Manifest.toml"), options.julia]
    missing = filter(path -> !isfile(path), paths)
    isempty(missing) || error("required pilot source or dependency missing: $(join(missing, ", "))")
    return paths
end

sources(options) = Dict(path=>sha256file(path) for path in source_paths(options))

function launch(options, mode, n, steps, cpus, initial, output, report;
                oracle=nothing, tile_size=nothing)
    command = String[options.julia, "--startup-file=no", "--project=$(options.project)",
                     "--threads=$(length(cpus))", "--gcthreads=1", DRIVER,
                     "mode=$mode", "nx=$n", "ny=$n", "output=$output"]
    if mode != "init"
        append!(command, ["steps=$steps", "input=$initial", "report=$report",
                          "cpus=$(join(cpus, ','))"])
        oracle === nothing || push!(command, "oracle=$oracle")
        tile_size === nothing || append!(command,
            ["tile-x=$(tile_size[1])", "tile-y=$(tile_size[2])"])
    end
    environment = Dict("OPENBLAS_NUM_THREADS"=>"1", "MKL_NUM_THREADS"=>"1",
        "JULIA_NUM_GC_THREADS"=>"1",
        "JULIA_DEPOT_PATH"=>options.depot * ":" * joinpath(homedir(), ".julia"))
    result = run_capture(command; env=environment, cpus)
    receipt = Dict{String,Any}(
        "mode"=>mode, "n"=>n, "workers"=>length(cpus), "cpus"=>cpus,
        "command"=>command, "cwd"=>pwd(), "environment"=>environment,
        "returncode"=>result.exitcode, "termsignal"=>result.termsignal,
        "stdout"=>result.stdout,
        "stderr"=>result.stderr, "process_wall_seconds"=>result.seconds,
        "timestamp_utc"=>utcstamp())
    appendjsonl(joinpath(options.output, "commands.jsonl"), receipt)
    if result.exitcode != 0
        writejson(joinpath(options.output, "failure-$(time_ns()).json"), receipt)
        error("failed $mode n=$n p=$(length(cpus)): $(result.stderr)\n$(result.stdout)")
    end
    return mode == "init" ? nothing : TOML.parsefile(report)
end

function summarize_raw(output)
    groups = Dict{Tuple{Int,Int,String,Int,Int},Vector{Float64}}()
    for row in readjsonl(joinpath(output, "raw.jsonl"))
        key = (row["nx"], row["workers"], row["mode"], row["tile_x"], row["tile_y"])
        push!(get!(groups, key, Float64[]), row["ns_per_unknown_step"])
    end
    rows = [Dict("n"=>key[1], "workers"=>key[2], "mode"=>key[3],
                 "tile"=>[key[4], key[5]], "samples"=>length(values),
                 "median_ns_per_unknown_step"=>median(values))
            for (key, values) in sort!(collect(groups); by=first)]
    writejson(joinpath(output, "summary.json"), rows)
    return rows
end

function main(args=ARGS)
    options = parse_options(args)
    allowed, nodes = topology()
    cpus = Dict(count=>placement_mask(count, options.placement, allowed, nodes)
                for count in options.workers)
    pin1 = placement_mask(1, options.placement, allowed, nodes)
    mkpath(joinpath(options.output, "fields"))
    before = sources(options)
    snapshot = joinpath(options.output, "source-snapshot")
    mkpath(snapshot)
    for source in source_paths(options)
        source == options.julia && continue
        relative = startswith(source, ROOT * "/") ? relpath(source, ROOT) :
                   joinpath("resolved-project", basename(source))
        target = joinpath(snapshot, relative)
        mkpath(dirname(target))
        cp(source, target; force=true)
    end
    config = Dict{String,Any}(
        "created_utc"=>utcstamp(), "sizes"=>options.sizes,
        "workers"=>options.workers, "steps"=>options.steps,
        "julia"=>options.julia, "project"=>options.project,
        "depot"=>options.depot, "pipeline_tiles"=>options.pipeline_tiles,
        "toy_tile"=>options.toy_tile, "repeats"=>options.repeats,
        "seed"=>options.seed, "placement"=>options.placement,
        "minimum_sample_seconds"=>options.min_seconds,
        "cpu_orders"=>cpus, "machine"=>machine_metadata(allowed, nodes),
        "sources"=>before,
        "timer"=>"full step including pipeline halo work and public-q copyback")
    writejson(joinpath(options.output, "configuration.json"), config)
    fields = joinpath(options.output, "fields")
    oracle_hashes = Dict{Int,String}()
    for n in options.sizes
        initial = joinpath(fields, "q0-$n.bin")
        oracle = joinpath(fields, "oracle-$n.bin")
        launch(options, "init", n, 0, pin1, nothing, initial, nothing)
        record = launch(options, "oracle", n, options.steps[n], pin1, initial,
                        oracle, joinpath(fields, "oracle-$n.toml"); tile_size=options.toy_tile)
        sources(options) == before || error("source changed during oracle preparation")
        oracle_hashes[n] = record["field_sha256"]
        sha256file(oracle) == record["field_sha256"] || error("oracle field hash mismatch")
        println("oracle $(n)² steps=$(options.steps[n]) ready")
    end
    jobs = Vector{Any}()
    layouts = vcat([("toy", options.toy_tile)],
                   [("pipeline", layout) for layout in options.pipeline_tiles])
    for repeat in 0:(options.repeats-1), n in options.sizes,
        count in options.workers, (mode, layout) in layouts
        push!(jobs, [repeat, n, count, mode, layout])
    end
    shuffle!(MersenneTwister(options.seed), jobs)
    writejson(joinpath(options.output, "job_order.json"), jobs)
    raw = joinpath(options.output, "raw.jsonl")
    for (index, job) in enumerate(jobs)
        repeat, n, count, mode, layout = job
        sources(options) == before || error("source changed during pilot")
        output, report = joinpath(fields, "sample.bin"), joinpath(fields, "sample.toml")
        record = launch(options, mode, n, options.steps[n], cpus[count],
            joinpath(fields, "q0-$n.bin"), output, report;
            oracle=joinpath(fields, "oracle-$n.bin"), tile_size=layout)
        sources(options) == before || error("source changed during sample")
        record["field_mismatches"] == 0 && record["field_sha256"] == oracle_hashes[n] ||
            error("full field differs from serial oracle")
        record["q_alias_preserved"] || error("public q alias was replaced")
        all(==(count), record["active_worker_counts"]) ||
            error("worker participation incomplete")
        [item["observed_cpu"] for item in record["affinity"]] == cpus[count] ||
            error("worker pinning mismatch")
        if record["time_seconds"] < options.min_seconds
            writejson(joinpath(options.output, "failure-short-$(index-1).json"),
                Dict("sample_index"=>index-1, "mode"=>mode, "n"=>n,
                     "workers"=>count, "seconds"=>record["time_seconds"],
                     "minimum_seconds"=>options.min_seconds, "record"=>record))
            error("timed sample shorter than required minimum")
        end
        merge!(record, Dict("repeat"=>repeat, "sample_index"=>index-1,
                            "placement"=>options.placement,
                            "requested_cpus"=>cpus[count], "source_manifest"=>before))
        appendjsonl(raw, record)
        rm(output)
        rm(report)
        println("$index/$(length(jobs)) $mode $(n)² p$count tile=$layout " *
                "$(round(record["ns_per_unknown_step"]; digits=3)) ns")
    end
    summarize_raw(options.output)
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CachePipelinePilot.main()
