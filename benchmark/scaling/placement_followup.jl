# Bounded p4/p8 ordinary, compact, and spread placement comparison.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))

const PLACEMENT_DRIVER = joinpath(@__DIR__, "driver.jl")

function main(args=ARGS)
    opts = stage2_options(args, ("julia", "project", "depot", "final", "output", "seed");
        defaults=Dict{String, Any}("project" => STAGE2_ROOT, "seed" => "29092026"))
    isnothing(opts) && return
    all(haskey(opts, name) for name in ("final", "output")) ||
        error("--final and --output are required")
    julia = stage2_julia_path(get(opts, "julia", nothing))
    project, destination = opts["project"], opts["output"]
    ispath(destination) && error("output already exists: $destination")
    final_config = readjson(joinpath(opts["final"], "configuration.json"))
    workload = readjson(joinpath(opts["final"], "workload.json"))
    steps = workload["1024"]
    steps == 2198 || error("expected frozen 1024² workload of 2198 steps")
    source_hash(project) == final_config["project_sha256"] &&
        toy_tree_hash() == final_config["toy_tree_sha256"] ||
        error("project or toy source differs from final matrix")
    tracked = [@__FILE__, PLACEMENT_DRIVER, joinpath(@__DIR__, "run.jl"),
               joinpath(@__DIR__, "Support.jl"), joinpath(@__DIR__, "Stage2Common.jl")]
    runner_hash = stage2_digest(tracked)
    allowed, nodes = topology()
    masks = Dict((mode, p) => placement_mask(p, mode, allowed, nodes)
                 for mode in ("ordinary", "compact", "spread") for p in (4, 8))
    seed = parse(Int, opts["seed"])
    config = Dict(
        "schema_version" => 1, "kind" => "stage2_placement_diagnostic",
        "created_utc" => stage2_utc(), "julia" => julia,
        "version" => command_output(julia, "--version"), "project" => project,
        "project_revision" => project_revision(project),
        "project_sha256" => final_config["project_sha256"],
        "toy_tree_sha256" => final_config["toy_tree_sha256"],
        "runner_sha256" => runner_hash, "final_artifact" => opts["final"],
        "size" => 1024, "steps" => steps, "backend" => "polyester",
        "flux" => "recompute", "tile" => [1024, 32], "threshold" => 16384,
        "workers" => [4, 8], "placements" => ["ordinary", "compact", "spread"],
        "repeats" => 5,
        "ordinary_policy" => "unrestricted inherited process affinity, no individual worker pinning",
        "compact_spread_policy" => "process mask plus one distinct pinned CPU per Julia worker",
        "cpu_masks" => Dict("$mode-p$p" => mask for ((mode, p), mask) in masks),
        "seed" => seed,
    )
    mkpath(joinpath(destination, "samples"))
    stage2_copy_snapshot(destination, tracked)
    stage2_copy_snapshot(destination,
        [joinpath(STAGE2_ROOT, "research", "UniformLWAdvection", part, name)
         for (part, name) in (("src", "UniformLWAdvection.jl"),
                              ("ext", "UniformLWAdvectionPolyesterExt.jl"))])
    stage2_copy_project(destination, project)
    writejson(joinpath(destination, "configuration.json"), config)
    raw_path = joinpath(destination, "raw.jsonl")
    open(raw_path, "w") do _ end
    rng = MersenneTwister(seed)
    for repeat in 0:4
        jobs = [(mode, p) for mode in config["placements"] for p in config["workers"]]
        shuffle!(rng, jobs)
        for (mode, p) in jobs
            stage2_digest(tracked) == runner_hash &&
                source_hash(project) == config["project_sha256"] &&
                toy_tree_hash() == config["toy_tree_sha256"] ||
                error("source changed during placement comparison")
            mask = masks[(mode, p)]
            sample_path = joinpath(destination, "samples", "r$repeat-$mode-p$p.toml")
            command = [julia, "--startup-file=no", "--project=$project",
                       "--threads=$p", "--gcthreads=1", PLACEMENT_DRIVER,
                       "case=toy", "backend=polyester", "nx=1024", "ny=1024",
                       "tile=32", "steps=$steps", "seconds=0.5",
                       "output=$sample_path", "flux=recompute", "tile-x=1024",
                       "threshold=16384"]
            mode == "ordinary" || append!(command,
                ["pin-workers=true", "pin-cpus=$(join(mask, ','))"])
            process = run_capture(command;
                env=stage2_env(get(opts, "depot", nothing)),
                cpus=mode == "ordinary" ? nothing : mask)
            process.exitcode == 0 ||
                error("placement sample failed: $(join(command, ' '))\n$(process.stdout)\n$(process.stderr)")
            sample = TOML.parsefile(sample_path)
            sample["steps"] == steps && sample["active_workers"] <= p &&
                sample["diagnostics"]["mass_error"] <= 1e-10 ||
                error("placement sample failed workload, worker, or numerical check")
            merge!(sample, Dict(
                "repeat" => repeat, "placement" => mode, "requested_workers" => p,
                "process_elapsed_seconds" => process.seconds,
                "runner_sha256" => runner_hash, "timestamp_utc" => stage2_utc(),
            ))
            appendjsonl(raw_path, sample)
            println("placement $(repeat+1)/5 $mode p$p: $(sample["time_seconds"])s; actual $(sample["active_workers"])")
        end
    end
    raw = readjsonl(raw_path)
    grouped = Dict{Tuple, Vector{Any}}()
    for row in raw
        push!(get!(grouped, (row["placement"], row["requested_workers"]), Any[]), row)
    end
    summary = Any[]
    for ((mode, p), rows) in sort!(collect(grouped); by=first)
        times = [row["time_seconds"] for row in rows]
        spread = [row["time_seconds"] for row in grouped[("spread", p)]]
        ratio = median(spread) / median(times)
        ci = mode == "spread" ? [1.0, 1.0] : stage2_ratio_ci(spread, times, seed + p)
        push!(summary, Dict(
            "placement" => mode, "requested_workers" => p,
            "actual_workers_min" => minimum(row["active_workers"] for row in rows),
            "samples" => length(rows), "time_median_seconds" => median(times),
            "time_min_seconds" => minimum(times), "speedup_vs_spread" => ratio,
            "speedup_vs_spread_ci95" => ci, "pin_workers" => mode != "ordinary",
        ))
    end
    writejson(joinpath(destination, "summary.json"), summary)
    println("Completed $(length(raw)) placement samples in $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
