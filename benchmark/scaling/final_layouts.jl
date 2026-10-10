# Five-repeat strong-scaling evidence for frozen standard and affine LW scopes.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))
using JSON

const INITIAL_STEPS = Dict(128 => 4096, 256 => 4096, 512 => 1024,
                           1024 => 256, 2048 => 64, 4096 => 16)
const FINAL_SIZES = [128, 256, 512, 1024, 2048, 4096]
const FINAL_THREADS = [1, 2, 4, 8, 16]

function final_jobs()
    jobs = Tuple[]
    for n in FINAL_SIZES
        append!(jobs, [(n, "standard", (1024, 32), p) for p in FINAL_THREADS])
        if n >= 2048
            for variant in ("fused_affine", "fused_amortized")
                append!(jobs, [(n, variant, (512, 8), p) for p in FINAL_THREADS])
            end
        end
        if n in (512, 2048, 4096)
            push!(jobs, (n, "halo", (1024, 32), 1))
            push!(jobs, (n, "fused_amortized", (1024, 32), 1))
        end
    end
    return jobs
end

function final_summary(records, seed)
    grouped = Dict{Tuple, Vector{Any}}()
    for row in records
        key = (row["nx"], row["variant"], Tuple(row["tile"]), row["workers"])
        push!(get!(grouped, key, Any[]), row)
    end
    result = Any[]
    for (key, rows) in sort!(collect(grouped); by=first)
        n, variant, tile, p = key
        times = [r["time_seconds"] for r in rows]
        mid = median(times)
        base = get(grouped, (n, variant, tile, 1), Any[])
        base_times = [r["time_seconds"] for r in base]
        speedup = isempty(base) ? nothing : median(base_times) / mid
        ci = isempty(base) ? nothing : p == 1 ? [1.0, 1.0] :
             stage2_ratio_ci(base_times, times, seed + n + p + sum(Int(c) for c in variant))
        capacity = min(p, cld(n, tile[1]) * cld(n, tile[2]))
        med(field) = median([r[field] for r in rows])
        push!(result, Dict(
            "nx" => n, "ny" => n, "variant" => variant, "tile" => collect(tile),
            "workers" => p, "api_scope" =>
                variant == "fused_amortized" ? "amortized_trajectory" : "observable_step",
            "samples" => length(rows), "steps" => rows[1]["steps"],
            "seconds_median" => mid, "seconds_min" => minimum(times),
            "seconds_max" => maximum(times),
            "seconds_mad" => median([abs(t - mid) for t in times]),
            "self_speedup" => speedup, "self_speedup_ci95" => ci,
            "self_efficiency" => isnothing(speedup) ? nothing : speedup / p,
            "self_efficiency_ci95" => isnothing(ci) ? nothing : ci ./ p,
            "active_workers_min" => minimum(r["active_workers"] for r in rows),
            "active_workers_min_pass" => minimum(r["active_workers_min_pass"] for r in rows),
            "copyback_workers_min" => minimum(r["copyback_active_workers"] for r in rows),
            "active_worker_scope" => rows[1]["active_worker_scope"],
            "tile_worker_capacity" => capacity,
            "mass_error_max" => maximum(r["mass_error"] for r in rows),
            "state_delta_max" => maximum(r["max_state_delta"] for r in rows),
            "translation_l2_max" => maximum(r["translation_l2"] for r in rows),
            "allocation_median_bytes" => med("allocation_bytes"),
            "peak_rss_median_bytes" => med("peak_rss_bytes"),
            "numerical_array_bytes" => rows[1]["numerical_array_bytes"],
        ))
    end
    for row in result
        n, scope = row["nx"], row["api_scope"]
        references = [item for item in result if item["nx"] == n &&
                      item["api_scope"] == scope && item["workers"] == 1]
        isempty(references) && continue
        fastest = first(sort(references; by=item->item["seconds_median"]))
        base_key = (n, fastest["variant"], Tuple(fastest["tile"]), 1)
        current_key = (n, row["variant"], Tuple(row["tile"]), row["workers"])
        base_times = [sample["time_seconds"] for sample in grouped[base_key]]
        current_times = [sample["time_seconds"] for sample in grouped[current_key]]
        ratio = fastest["seconds_median"] / row["seconds_median"]
        ci = base_key == current_key ? [1.0, 1.0] :
             stage2_ratio_ci(base_times, current_times, seed + n + row["workers"] + 314159)
        merge!(row, Dict(
            "fastest_serial_variant" => fastest["variant"],
            "fastest_serial_tile" => fastest["tile"],
            "fastest_serial_seconds" => fastest["seconds_median"],
            "speedup_vs_fastest_serial" => ratio,
            "speedup_vs_fastest_serial_ci95" => ci,
            "efficiency_vs_fastest_serial" => ratio / row["workers"],
            "efficiency_vs_fastest_serial_ci95" => ci ./ row["workers"],
        ))
    end
    return result
end

function main(args=ARGS)
    opts = stage2_options(args, ("julia", "project", "depot", "output", "seconds", "repeats", "seed");
        defaults=Dict{String, Any}("project" => STAGE2_ROOT, "seconds" => "0.5",
                                   "repeats" => "5", "seed" => "29092026"))
    isnothing(opts) && return
    haskey(opts, "output") || error("--output is required")
    seconds = parse(Float64, opts["seconds"])
    repeats = parse(Int, opts["repeats"])
    seed = parse(Int, opts["seed"])
    seconds > 0 && repeats >= 1 || error("seconds and repeats must be positive")
    destination = opts["output"]
    ispath(destination) && error("output already exists: $destination")
    julia = stage2_julia_path(get(opts, "julia", nothing))
    version = command_output(julia, "--version")
    startswith(lowercase(version), "julia version 1.10.") ||
        error("frozen final script expects Julia 1.10.x; got $version")
    project = opts["project"]
    allowed, nodes = topology()
    masks = Dict(p => placement_mask(p, "spread", allowed, nodes) for p in FINAL_THREADS)
    source_before = stage2_digest(STAGE2_SOURCES)
    runner_before = stage2_digest([@__FILE__; STAGE2_SOURCES])
    project_before = source_hash(project)
    toy_before = toy_tree_hash()
    mkpath(joinpath(destination, "samples"))
    stage2_copy_snapshot(destination, [@__FILE__; STAGE2_SOURCES])
    stage2_copy_project(destination, project)
    config = Dict(
        "schema_version" => 1, "kind" => "final_lw_scaling",
        "created_utc" => stage2_utc(), "julia" => julia,
        "julia_version" => version, "project" => project,
        "project_revision" => project_revision(project),
        "project_sha256" => project_before, "toy_tree_sha256" => toy_before,
        "experiment_sha256" => source_before, "runner_sha256" => runner_before,
        "depot" => get(opts, "depot", nothing), "placement" => "spread",
        "pin_workers" => true,
        "cpu_masks" => Dict(string(p) => mask for (p, mask) in masks),
        "blas_threads" => 1, "gc_threads" => 1,
        "sizes" => FINAL_SIZES, "threads" => FINAL_THREADS,
        "repeats" => repeats, "seconds_target" => seconds,
        "general_step_policy" => ["standard", [1024, 32]],
        "large_grid_candidates" => [["fused_affine", [512, 8]],
                                    ["fused_amortized", [512, 8]]],
        "crossover_cells" => 2048^2,
        "serial_references" => [["halo", [1024, 32]],
                                ["fused_amortized", [1024, 32]]],
        "caveat" => "amortized trajectory is not an individually observable step API",
        "machine" => machine_metadata(allowed, nodes), "seed" => seed,
    )
    config["configuration_sha256"] = bytes2hex(SHA.sha256(JSON.json(config)))
    writejson(joinpath(destination, "configuration.json"), config)
    raw_path, pilot_path = joinpath(destination, "raw.jsonl"),
                           joinpath(destination, "pilots.jsonl")
    open(raw_path, "w") do _ end
    open(pilot_path, "w") do _ end
    workload_hash = Ref{Union{Nothing, String}}(nothing)
    function check_source()
        stage2_digest(STAGE2_SOURCES) == source_before &&
            stage2_digest([@__FILE__; STAGE2_SOURCES]) == runner_before &&
            source_hash(project) == project_before && toy_tree_hash() == toy_before ||
            error("project, toy, experiment, or runner source changed during measurements")
    end
    function invoke(n, variant, tile, p, steps, kind, repeat)
        check_source()
        mask = masks[p]
        sample_path = joinpath(destination, "samples",
            "$kind-r$repeat-$variant-$(tile[1])x$(tile[2])-$n-p$p.toml")
        command = [julia, "--startup-file=no", "--project=$project",
                   "--threads=$p", "--gcthreads=1", STAGE2_SCREEN_SCRIPT,
                   "variant=$variant", "nx=$n", "ny=$n", "steps=$steps",
                   "backend=polyester", "tile-x=$(tile[1])", "tile-y=$(tile[2])",
                   "threshold=16384", "cpus=$(join(mask, ','))", "output=$sample_path"]
        process = run_capture(command; env=stage2_env(get(opts, "depot", nothing)), cpus=mask)
        process.exitcode == 0 ||
            error("sample failed: $(join(command, ' '))\n$(process.stdout)\n$(process.stderr)")
        check_source()
        sample = TOML.parsefile(sample_path)
        for name in ("time_seconds", "max_state_delta", "mass_error", "translation_l2",
                     "checksum", "dt", "final_time", "observed_final_time",
                     "active_worker_ids", "compute_pass_worker_ids", "copyback_worker_ids",
                     "active_worker_scope", "allocation_bytes", "peak_rss_bytes",
                     "numerical_array_bytes")
            haskey(sample, name) || error("sample missing required field $name")
        end
        for name in ("time_seconds", "max_state_delta", "mass_error", "translation_l2",
                     "checksum", "dt", "final_time", "observed_final_time")
            isfinite(sample[name]) || error("nonfinite sample field $name")
        end
        sample["variant"] == variant && sample["nx"] == n && sample["ny"] == n &&
            sample["steps"] == steps && sample["workers"] == p &&
            sample["tile"] == collect(tile) || error("sample differs from frozen job")
        sample["max_state_delta"] <= 1e-12 && sample["mass_error"] <= 1e-12 ||
            error("sample numerical parity/conservation failed")
        length(sample["affinity"]) == p &&
            [entry["cpu"] for entry in sample["affinity"]] == mask ||
            error("sample worker pinning mismatch")
        active = length(sample["active_worker_ids"])
        passes = [length(ids) for ids in sample["compute_pass_worker_ids"]]
        min_pass = isempty(passes) ? 0 : minimum(passes)
        copyback = length(sample["copyback_worker_ids"])
        capacity = min(p, cld(n, tile[1]) * cld(n, tile[2]))
        merge!(sample, Dict(
            "active_workers" => active, "active_workers_min_pass" => min_pass,
            "copyback_active_workers" => copyback, "requested_workers" => p,
            "tile_worker_capacity" => capacity,
            "underoccupied_by_tile_count" => capacity < p,
        ))
        if variant == "halo" && p == 1 &&
                sample["active_worker_scope"] == "halo prototype not instrumented"
            nothing
        else
            1 <= min_pass <= active <= capacity ||
                error("invalid kernel participation $active; requested $p, capacity $capacity")
        end
        if n in (512, 1024, 2048, 4096) && p in (4, 8)
            min_pass == p || error("primary-gate per-pass participation differs")
        end
        sample["time_seconds"] > 0 || error("invalid sample time")
        merge!(sample, Dict(
            "kind" => kind, "repeat" => repeat,
            "process_elapsed_seconds" => process.seconds,
            "project_sha256" => project_before, "toy_tree_sha256" => toy_before,
            "experiment_sha256" => source_before, "runner_sha256" => runner_before,
            "configuration_sha256" => config["configuration_sha256"],
            "workload_sha256" => workload_hash[], "timestamp_utc" => stage2_utc(),
        ))
        appendjsonl(kind == "pilot" ? pilot_path : raw_path, sample)
        return sample
    end
    workloads = Dict{String, Int}()
    for n in FINAL_SIZES
        pilots = [("standard", (1024, 32), p) for p in (1, 8, 16)]
        n >= 2048 && append!(pilots,
            [("fused_amortized", (512, 8), p) for p in (8, 16)])
        estimates = Tuple[]
        for (variant, tile, p) in pilots
            sample = invoke(n, variant, tile, p, INITIAL_STEPS[n], "pilot", 0)
            push!(estimates, (sample["time_seconds"] / INITIAL_STEPS[n], variant, tile, p))
            println("pilot $(n)² $variant p$p: $(sample["time_seconds"])s/$(INITIAL_STEPS[n]) steps")
        end
        fastest = minimum(estimates)
        workloads[string(n)] = ceil(Int, seconds / fastest[1])
        reached = false
        for margin in 1:3
            check = invoke(n, fastest[2], fastest[3], fastest[4],
                           workloads[string(n)], "pilot", margin)
            if check["time_seconds"] >= 0.35
                reached = true
                break
            end
            workloads[string(n)] = ceil(Int, workloads[string(n)] *
                                            (0.55 / check["time_seconds"]))
        end
        reached || error("unable to freeze meaningful $(n)² trajectory")
        println("frozen $(n)²: $(workloads[string(n)]) steps")
    end
    workload_path = joinpath(destination, "workload.json")
    writejson(workload_path, workloads)
    workload_hash[] = sha256file(workload_path)
    all_jobs = final_jobs()
    rng = MersenneTwister(seed)
    records = Any[]
    for repeat in 0:(repeats - 1)
        jobs = copy(all_jobs)
        shuffle!(rng, jobs)
        for (n, variant, tile, p) in jobs
            sample = invoke(n, variant, tile, p, workloads[string(n)], "final", repeat)
            push!(records, sample)
            println("repeat $(repeat+1)/$repeats $(n)² $variant $tile p$p: $(sample["time_seconds"])s; active $(sample["active_workers"])")
            writejson(joinpath(destination, "summary.json"), final_summary(records, seed))
        end
    end
    println("Completed $(length(records)) final samples in $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
