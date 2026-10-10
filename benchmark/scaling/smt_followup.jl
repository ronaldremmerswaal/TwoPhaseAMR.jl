# Separate p24/p32 SMT diagnostic using the frozen Stage 2 workloads.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))

function main(args=ARGS)
    opts = stage2_options(args, ("final", "output", "julia", "project", "depot", "seed");
        defaults=Dict{String, Any}("project" => STAGE2_ROOT, "seed" => "29092026"))
    isnothing(opts) && return
    all(haskey(opts, name) for name in ("final", "output")) ||
        error("--final and --output are required")
    julia = stage2_julia_path(get(opts, "julia", nothing))
    destination = opts["output"]
    ispath(destination) && error("output already exists: $destination")
    final_config = readjson(joinpath(opts["final"], "configuration.json"))
    workload = readjson(joinpath(opts["final"], "workload.json"))
    final_rows = readjsonl(joinpath(opts["final"], "raw.jsonl"))
    length(final_rows) == 280 || error("final Stage 2 reference needs 280 samples")
    project = opts["project"]
    experiment_hash = stage2_digest(STAGE2_SOURCES)
    experiment_hash == final_config["experiment_sha256"] &&
        source_hash(project) == final_config["project_sha256"] &&
        toy_tree_hash() == final_config["toy_tree_sha256"] ||
        error("source differs from frozen final matrix")
    allowed, nodes = topology()
    masks = Dict(p => placement_mask(p, "spread", allowed, nodes) for p in (24, 32))
    script_hash = sha256file(@__FILE__)
    seed = parse(Int, opts["seed"])
    config = Dict(
        "schema_version" => 1, "kind" => "stage2_smt_diagnostic",
        "created_utc" => stage2_utc(), "julia" => julia,
        "version" => command_output(julia, "--version"), "project" => project,
        "project_sha256" => final_config["project_sha256"],
        "toy_tree_sha256" => final_config["toy_tree_sha256"],
        "experiment_sha256" => experiment_hash,
        "runner_sha256" => script_hash,
        "final_artifact" => opts["final"],
        "final_config_sha256" => final_config["configuration_sha256"],
        "placement" => "spread, physical cores first, then SMT siblings",
        "cpu_masks" => Dict(string(p) => mask for (p, mask) in masks),
        "repeats" => 5, "workers" => [24, 32], "seed" => seed,
        "coverage" => "standard1024, standard2048, standard4096 only; affine SMT excluded after intermittent p24 numerical parity failure; 512 excluded because standard has16 tiles",
        "excluded_affine_smt_artifact" =>
            "benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/failure.json",
    )
    mkpath(joinpath(destination, "samples"))
    stage2_copy_snapshot(destination, [@__FILE__; STAGE2_SOURCES])
    stage2_copy_project(destination, project)
    writejson(joinpath(destination, "configuration.json"), config)
    jobs = [(n, "standard", (1024, 32), p)
            for n in (1024, 2048, 4096) for p in (24, 32)]
    length(jobs) == 6 || error("standard-only SMT matrix needs six jobs per repeat")
    raw_path = joinpath(destination, "raw.jsonl")
    open(raw_path, "w") do _ end
    rng = MersenneTwister(seed)
    for repeat in 0:4
        batch = copy(jobs)
        shuffle!(rng, batch)
        for (n, variant, tile, p) in batch
            stage2_digest(STAGE2_SOURCES) == config["experiment_sha256"] &&
                source_hash(project) == config["project_sha256"] &&
                toy_tree_hash() == config["toy_tree_sha256"] ||
                error("source changed during SMT sampling")
            mask = masks[p]
            sample_path = joinpath(destination, "samples",
                "r$repeat-$n-$variant-$(tile[1])x$(tile[2])-p$p.toml")
            command = [julia, "--startup-file=no", "--project=$project",
                       "--threads=$p", "--gcthreads=1", STAGE2_SCREEN_SCRIPT,
                       "variant=$variant", "nx=$n", "ny=$n",
                       "steps=$(workload[string(n)])", "backend=polyester",
                       "tile-x=$(tile[1])", "tile-y=$(tile[2])", "threshold=16384",
                       "cpus=$(join(mask, ','))", "output=$sample_path"]
            process = run_capture(command;
                env=stage2_env(get(opts, "depot", nothing)), cpus=mask)
            process.exitcode == 0 ||
                error("SMT sample failed: $(join(command, ' '))\n$(process.stdout)\n$(process.stderr)")
            sample = TOML.parsefile(sample_path)
            sample["variant"] == variant && sample["workers"] == p &&
                sample["steps"] == workload[string(n)] && sample["tile"] == collect(tile) ||
                error("SMT sample differs from frozen job")
            sample["max_state_delta"] == 0 && sample["probe_state_delta"] == 0 &&
                sample["mass_error"] <= 512eps(Float64) ||
                error("SMT numerical parity or conservation failed")
            pass_counts = [length(ids) for ids in sample["compute_pass_worker_ids"]]
            actual = minimum(pass_counts)
            capacity = min(p, cld(n, tile[1]) * cld(n, tile[2]))
            1 <= actual <= length(sample["active_worker_ids"]) <= capacity ||
                error("invalid SMT kernel participation")
            [entry["cpu"] for entry in sample["affinity"]] == mask ||
                error("SMT pinning mismatch")
            merge!(sample, Dict(
                "repeat" => repeat, "requested_workers" => p,
                "actual_workers_min_pass" => actual,
                "tile_worker_capacity" => capacity, "underoccupied" => actual < p,
                "process_elapsed_seconds" => process.seconds,
                "project_sha256" => config["project_sha256"],
                "toy_tree_sha256" => config["toy_tree_sha256"],
                "experiment_sha256" => experiment_hash,
                "runner_sha256" => script_hash, "timestamp_utc" => stage2_utc(),
            ))
            appendjsonl(raw_path, sample)
            println("SMT $(repeat+1)/5 $(n)² $variant p$p: $(sample["time_seconds"])s; actual/pass $actual")
        end
    end
    final_grouped = Dict{Tuple, Vector{Float64}}()
    for row in final_rows
        key = (row["nx"], row["variant"], Tuple(row["tile"]), row["workers"])
        push!(get!(final_grouped, key, Float64[]), row["time_seconds"])
    end
    smt_rows = readjsonl(raw_path)
    smt_grouped = Dict{Tuple, Vector{Float64}}()
    for row in smt_rows
        key = (row["nx"], row["variant"], Tuple(row["tile"]), row["workers"])
        push!(get!(smt_grouped, key, Float64[]), row["time_seconds"])
    end
    summary = Any[]
    for (key, times) in sort!(collect(smt_grouped); by=first)
        n, variant, tile, p = key
        base = final_grouped[(n, variant, tile, 1)]
        sixteen = final_grouped[(n, variant, tile, 16)]
        mid = median(times)
        speedup = median(base) / mid
        ratio = median(sixteen) / mid
        push!(summary, Dict(
            "size" => n, "variant" => variant, "tile" => collect(tile),
            "requested_workers" => p,
            "actual_workers_min_pass" => minimum(row["actual_workers_min_pass"]
                for row in smt_rows if row["nx"] == n && row["variant"] == variant &&
                                       Tuple(row["tile"]) == tile && row["workers"] == p),
            "samples" => length(times), "time_median_seconds" => mid,
            "time_min_seconds" => minimum(times), "S_p" => speedup,
            "S_p_ci95" => stage2_ratio_ci(base, times, seed + n + p),
            "E_p_requested" => speedup / p,
            "physical_core_normalized_S_p_over_16" => speedup / 16,
            "T16_over_Tp" => ratio,
            "T16_over_Tp_ci95" => stage2_ratio_ci(sixteen, times, seed + n + p + 16),
            "comparison_window" =>
                "p1/p16 from earlier final matrix; p24/p32 from separate SMT acquisition",
        ))
    end
    writejson(joinpath(destination, "summary.json"), summary)
    println("Completed $(length(smt_rows)) SMT samples in $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
