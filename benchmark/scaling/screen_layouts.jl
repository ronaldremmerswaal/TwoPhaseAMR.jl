# Bounded, fresh-process screen of experimental LW storage layouts.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))

const SCREEN_VARIANTS = Set(("standard", "halo", "fused", "fused_affine", "fused_amortized"))

function parse_screen_args(args)
    opts = Dict{String, Any}("project" => STAGE2_ROOT,
                             "variants" => ["standard", "halo", "fused"],
                             "tiles" => ["1024x32"], "seed" => "29092026")
    index = 1
    while index <= length(args)
        key = args[index]
        if key in ("-h", "--help")
            println("usage: julia screen_layouts.jl --output PATH [--julia PATH] [--project PATH] [--depot PATH] [--variants NAME ...] [--tiles NxM ...] [--seed INT]")
            return nothing
        end
        startswith(key, "--") || error("unexpected argument: $key")
        name = key[3:end]
        name in ("julia", "project", "depot", "output", "variants", "tiles", "seed") ||
            error("unknown option $key")
        if name in ("variants", "tiles")
            values = String[]
            index += 1
            while index <= length(args) && !startswith(args[index], "--")
                push!(values, args[index])
                index += 1
            end
            isempty(values) && error("$key needs at least one value")
            opts[name] = values
        else
            index < length(args) || error("$key needs a value")
            opts[name] = args[index + 1]
            index += 2
        end
    end
    haskey(opts, "output") || error("--output is required")
    all(variant in SCREEN_VARIANTS for variant in opts["variants"]) ||
        error("unknown layout variant")
    return opts
end

function parse_tiles(values)
    result = Tuple{Int, Int}[]
    for value in values
        parts = split(lowercase(value), 'x')
        length(parts) == 2 || error("tiles must be positive NxM shapes")
        tile = (parse(Int, parts[1]), parse(Int, parts[2]))
        minimum(tile) >= 1 || error("tiles must be positive NxM shapes")
        push!(result, tile)
    end
    return result
end

function main(args=ARGS)
    opts = parse_screen_args(args)
    isnothing(opts) && return
    tiles = parse_tiles(opts["tiles"])
    julia = stage2_julia_path(get(opts, "julia", nothing))
    version = command_output(julia, "--version")
    startswith(lowercase(version), "julia version 1.10.") ||
        error("screen freezes Julia 1.10 syntax and baseline; got $version")
    destination = opts["output"]
    ispath(destination) && error("output already exists: $destination")
    allowed, nodes = topology()
    masks = Dict(p => placement_mask(p, "spread", allowed, nodes) for p in (1, 4, 8))
    project_hash = source_hash(opts["project"])
    experiment_hash = stage2_digest(STAGE2_SOURCES)
    mkpath(joinpath(destination, "samples"))
    stage2_copy_snapshot(destination, STAGE2_SOURCES)
    stage2_copy_project(destination, opts["project"])
    steps = Dict("512" => 1024, "4096" => 16)
    config = Dict(
        "schema_version" => 1, "kind" => "bounded_layout_screen",
        "created_utc" => stage2_utc(), "project" => opts["project"],
        "project_revision" => project_revision(opts["project"]),
        "project_sha256" => project_hash, "toy_tree_sha256" => toy_tree_hash(),
        "experiment_sha256" => experiment_hash,
        "julia" => julia, "julia_version" => version,
        "depot" => get(opts, "depot", nothing), "variants" => opts["variants"],
        "sizes" => [512, 4096], "steps" => steps, "workers" => [1, 4, 8],
        "repeats" => 2, "backend" => "polyester", "flux" => "recompute",
        "tiles" => [collect(tile) for tile in tiles], "threshold" => 16384,
        "placement" => "spread", "pinning" => "per-worker Linux affinity",
        "cpu_masks" => Dict(string(p) => mask for (p, mask) in masks),
        "seed" => parse(Int, opts["seed"]),
    )
    writejson(joinpath(destination, "configuration.json"), config)
    jobs = [(variant, tile, n, p, repeat) for repeat in 0:1
            for variant in config["variants"] for tile in tiles
            for n in config["sizes"] for p in config["workers"]]
    rng = MersenneTwister(config["seed"])
    per_repeat = length(jobs) ÷ 2
    shuffle!(rng, view(jobs, 1:per_repeat))
    shuffle!(rng, view(jobs, (per_repeat + 1):length(jobs)))
    raw_path = joinpath(destination, "raw.jsonl")
    open(raw_path, "w") do _ end
    for (index, (variant, tile, n, p, repeat)) in enumerate(jobs)
        stage2_digest(STAGE2_SOURCES) == experiment_hash &&
            source_hash(opts["project"]) == project_hash ||
            error("source changed during frozen layout screen")
        sample_path = joinpath(destination, "samples",
                               "r$repeat-$variant-$(tile[1])x$(tile[2])-$n-p$p.toml")
        mask = masks[p]
        command = [julia, "--startup-file=no", "--project=$(opts["project"])",
                   "--threads=$p", "--gcthreads=1", STAGE2_SCREEN_SCRIPT,
                   "variant=$variant", "nx=$n", "ny=$n",
                   "steps=$(steps[string(n)])", "backend=polyester",
                   "tile-x=$(tile[1])", "tile-y=$(tile[2])", "threshold=16384",
                   "cpus=$(join(mask, ','))", "output=$sample_path"]
        process = run_capture(command; env=stage2_env(get(opts, "depot", nothing)), cpus=mask)
        process.exitcode == 0 ||
            error("sample $index/$(length(jobs)) failed\n$(process.stdout)\n$(process.stderr)")
        sample = TOML.parsefile(sample_path)
        sample["variant"] == variant && sample["nx"] == n &&
            sample["steps"] == steps[string(n)] && sample["workers"] == p &&
            sample["tile"] == collect(tile) || error("screen sample differs from frozen job")
        sample["max_state_delta"] <= 1e-12 && sample["mass_error"] <= 1e-12 ||
            error("layout screen numerical check failed")
        length(sample["affinity"]) == p &&
            [entry["cpu"] for entry in sample["affinity"]] == mask ||
            error("screen sample worker pinning mismatch")
        merge!(sample, Dict("repeat" => repeat,
                            "process_elapsed_seconds" => process.seconds,
                            "project_sha256" => project_hash,
                            "experiment_sha256" => experiment_hash,
                            "timestamp_utc" => stage2_utc()))
        appendjsonl(raw_path, sample)
        println("$index/$(length(jobs)) $variant tile$(tile[1])x$(tile[2]) $(n)² p$p r$(repeat+1): $(sample["time_seconds"])s")
    end
    println("Completed $(length(jobs)) independent exploratory layout samples in $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
