"""Full-field validated dormant Fortran padding and per-worker PMU controls."""
module CacheCliffFortranControls

include(joinpath(@__DIR__, "cache_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using Dates
using JSON
using Random
using SHA
using Statistics

const BaseStudy = CacheCliffStudy
const Matched = BaseStudy.Matched
const HERE = @__DIR__
const ROOT = Matched.ROOT
const DIAG = joinpath(Matched.FORTRAN, "diagnostics")
const RESULT_ROOT = joinpath(ROOT, "benchmark", "scaling", "results",
                             "20260930-cache-cliff")

function options(args)
    parsed = Matched.cli(args)
    mode = Matched.single(parsed, "mode")
    mode in ("pmu", "padding") || error("mode must be pmu or padding")
    output = abspath(Matched.single(parsed, "output"))
    result_root = abspath(Matched.single(parsed, "result-root", RESULT_ROOT))
    binary = abspath(Matched.single(parsed, "binary", joinpath(DIAG, "build", "lw_omp_padded")))
    julia = abspath(Matched.single(parsed, "julia", "/tmp/julia-1.10.12/bin/julia"))
    project = abspath(Matched.single(parsed, "project", "/tmp/twophase-compat110"))
    depot = abspath(Matched.single(parsed, "depot", "/tmp/twophase-julia110-depot"))
    repeats = parse(Int, Matched.single(parsed, "repeats", mode == "padding" ? "5" : "3"))
    seed = parse(Int, Matched.single(parsed, "seed", "30092028"))
    !ispath(output) || error("output already exists")
    repeats > 0 || error("repeats must be positive")
    all(isfile, (binary, julia)) && isdir(project) ||
        error("diagnostic binary, Julia, or project missing")
    return (; mode, output, result_root, binary, julia, project, depot, repeats, seed)
end

function source_paths(options)
    paths = String[joinpath(HERE, "fortran_controls.jl"),
        joinpath(HERE, "cache_study.jl"), joinpath(HERE, "report.jl"),
        joinpath(HERE, "..", "cross_language", "study.jl"), Matched.DRIVER,
        joinpath(HERE, "..", "Support.jl"),
        joinpath(HERE, "..", "Project.toml"), joinpath(HERE, "..", "Manifest.toml"),
        joinpath(Matched.TOY, "src", "UniformLWAdvection.jl"),
        joinpath(Matched.TOY, "ext", "UniformLWAdvectionPolyesterExt.jl"),
        joinpath(DIAG, "lw_omp_padded.f90"), joinpath(DIAG, "perf_counters.c"),
        joinpath(DIAG, "Makefile"), joinpath(DIAG, "README.md"), options.binary,
        joinpath(options.project, "Project.toml"), joinpath(options.project, "Manifest.toml")]
    missing = filter(path->!isfile(path), paths)
    isempty(missing) || error("required diagnostic source missing: $(join(missing, ", "))")
    return paths
end

manifest(options) = Dict(path=>sha256file(path) for path in source_paths(options))

function snapshot(output, hashes)
    for name in keys(hashes)
        relative = startswith(name, ROOT * "/") ? relpath(name, ROOT) :
                   joinpath("resolved-project", basename(name))
        target = joinpath(output, "source-snapshot", relative)
        mkpath(dirname(target))
        cp(name, target; force=true)
    end
end

function inputs(options, output, jobs, mask1)
    fixtures = Dict{String,Any}()
    for (n,_,_,steps,_,_) in jobs
        key = "$n:$steps"
        haskey(fixtures, key) && continue
        initial, oracle = BaseStudy.initial_and_oracle(options, output, n, steps, mask1)
        fixtures[key] = Dict("q0_sha256"=>sha256file(initial),
                             "oracle_sha256"=>sha256file(oracle))
    end
    writejson(joinpath(output, "fixtures.json"), fixtures)
    return fixtures
end

function plan(mode, root)
    ladder = Dict(parse(Int,k)=>Int(v) for (k,v) in
        readjson(joinpath(root, "ladder", "workload.json")))
    placement = Dict(parse(Int,k)=>Int(v) for (k,v) in
        readjson(joinpath(root, "placement", "workload.json")))
    if mode == "padding"
        return [(n,16,"spread",ladder[n],pad,0)
                for n in (1024,2048) for pad in (0,8,32)]
    end
    return vcat([(n,16,"spread",ladder[n],0,counter)
                 for n in (1024,1280,1408,2048) for counter in (1,2)],
                [(n,8,place,placement[n],0,counter)
                 for n in (1024,1280) for place in ("compact","spread")
                 for counter in (1,2)])
end

function invoke(options, n, p, place, steps, pad, counters, cpus, initial, dump)
    command = String[options.binary, "--nx", string(n), "--ny", string(n),
        "--steps", string(steps), "--threads", string(p),
        "--tile-x", "512", "--tile-y", "16", "--pad", string(pad),
        "--counters", string(counters), "--cpus", join(cpus, ','),
        "--input", initial, "--output", dump]
    stdout, wall = Matched.launch(command, cpus, Matched.execution_env(cpus, options.depot))
    isempty(stdout) && error("diagnostic did not emit JSON")
    record = JSON.parse(last(split(stdout, '\n')))
    observed = Tuple(record[key] for key in
        ("nx", "ny", "steps", "workers", "tile_x", "tile_y", "padding", "leading_dimension"))
    observed == (n,n,steps,p,512,16,pad,n+pad) || error("diagnostic workload mismatch")
    record["affinity_cpu_ids"] == cpus && record["observed_cpu_ids"] == cpus &&
        record["pin_verified"] && all(==(p), record["active_worker_counts"]) ||
        error("diagnostic pinning or worker participation mismatch")
    isapprox(record["dt"], .35/n; rtol=2e-15) || error("diagnostic timestep mismatch")
    isapprox(record["final_time"], steps*record["dt"]; rtol=1e-12, atol=1e-12) ||
        error("diagnostic final time mismatch")
    isapprox(record["seconds_per_unknown_step"],
             record["time_seconds"]/(n*n*steps); rtol=2e-12) ||
        error("diagnostic normalized time mismatch")
    merge!(record, Dict("process_wall_seconds"=>wall,
        "requested_cpu_order"=>cpus, "placement"=>place,
        "counter_mode"=>counters,
        "ns_per_unknown_step"=>1e9*record["time_seconds"]/(n*n*steps)))
    if counters != 0
        events = record["counter_events"]
        length(events) == 5 && record["counter_scope"] !== nothing ||
            error("diagnostic counter event set incomplete")
        expected = [("cycles",0,0), ("instructions",0,1), ("generic_cache_misses",0,3)]
        if counters == 1
            append!(expected, [("zen3_demand_mem_io_local",4,0x0843),
                               ("zen3_demand_internal_cache",4,0x0243)])
        else
            append!(expected, [("zen3_hwprefetch_mem_io_local",4,0x085a),
                               ("zen3_hwprefetch_internal_cache",4,0x025a)])
        end
        [(event["name"],event["linux_perf_type"],event["config"]) for event in events] ==
            expected || error("unexpected PMU event definitions")
        for event in events
            workers = event["per_worker"]
            length(workers) == p &&
                sum(worker["value"] for worker in workers) == event["aggregate_raw"] ||
                error("counter worker/raw sum mismatch")
            fractions = [worker["time_running_ns"] / worker["time_enabled_ns"]
                         for worker in workers]
            minimum(fractions) >= .99 || error("counter multiplexing >1%")
            scaled = sum(worker["value"] * worker["time_enabled_ns"] /
                         worker["time_running_ns"] for worker in workers)
            isapprox(scaled, event["aggregate_scaled"]; rtol=1e-10, atol=2) ||
                error("scaled counter mismatch")
            event["min_running_fraction"] = minimum(fractions)
            event["events_per_unknown_step"] = scaled/(n*n*steps)
        end
    else
        record["counter_events"] === nothing || error("counter-off record includes events")
    end
    return record
end

function median_ci(values, seed; replicas=10000)
    rng = MersenneTwister(seed)
    samples = sort!([median(rand(rng, values, length(values))) for _ in 1:replicas])
    return [samples[floor(Int,.025*(replicas-1))+1],
            samples[floor(Int,.975*(replicas-1))+1]]
end

function summarize(output, config)
    raw = readjsonl(joinpath(output, "raw.jsonl"))
    groups = Dict{Tuple{Int,Int,String,Int,Int,Int},Vector{Any}}()
    for row in raw
        key = (row["nx"],row["workers"],row["placement"],row["steps"],
               row["padding"],row["counter_mode"])
        push!(get!(groups, key, Any[]), row)
    end
    rows = Dict{String,Any}[]
    for (key, samples) in sort!(collect(groups); by=first)
        n,p,place,steps,pad,mode = key
        length(samples) == config["repeats"] &&
            Set(row["repeat"] for row in samples) == Set(0:config["repeats"]-1) ||
            error("incomplete diagnostic group $key")
        times = Float64[row["ns_per_unknown_step"] for row in samples]
        item = Dict{String,Any}(
            "nx"=>n, "workers"=>p, "placement"=>place, "steps"=>steps,
            "padding"=>pad, "counter_mode"=>mode, "samples"=>length(samples),
            "median_ns_per_unknown_step"=>median(times),
            "time_ci95"=>median_ci(times,n+p+pad+mode),
            "field_max_abs_error"=>maximum(row["field_max_abs_error"] for row in samples),
            "events"=>Any[])
        if mode != 0
            names = [event["name"] for event in samples[1]["counter_events"]]
            for name in names
                event_rows = [only([event for event in row["counter_events"]
                                    if event["name"] == name]) for row in samples]
                rates = Float64[event["events_per_unknown_step"] for event in event_rows]
                push!(item["events"], Dict(
                    "name"=>name,
                    "median_raw_count"=>median(event["aggregate_raw"] for event in event_rows),
                    "median_scaled_count"=>median(event["aggregate_scaled"] for event in event_rows),
                    "median_events_per_unknown_step"=>median(rates),
                    "events_per_unknown_step_ci95"=>median_ci(rates,
                        n+p+mode+sum(Int(char) for char in name)),
                    "minimum_worker_running_fraction"=>
                        minimum(event["min_running_fraction"] for event in event_rows)))
            end
        end
        push!(rows, item)
    end
    summary = Dict{String,Any}(
        "schema"=>1, "mode"=>config["mode"], "samples"=>length(raw),
        "rows"=>rows, "configuration_sha256"=>config["configuration_sha256"],
        "analysis_script_sha256"=>sha256file(@__FILE__))
    writejson(joinpath(output, "summary.json"), summary)
    return summary
end

function main(args=ARGS)
    options = CacheCliffFortranControls.options(args)
    jobs = plan(options.mode, options.result_root)
    allowed, nodes = topology()
    masks = Dict(place=>Dict(p=>placement_mask(p,place,allowed,nodes)
                             for p in (1,8,16))
                 for place in ("spread","compact"))
    mkpath(joinpath(options.output, "fields"))
    before = manifest(options)
    snapshot(options.output, before)
    writejson(joinpath(options.output, "source_sha256.json"), before)
    config = Dict{String,Any}(
        "schema"=>1, "mode"=>options.mode, "created_utc"=>Matched.utcstamp(),
        "jobs"=>jobs, "repeats"=>options.repeats, "seed"=>options.seed,
        "cpu_orders"=>masks, "binary"=>options.binary, "julia"=>options.julia,
        "project"=>options.project, "depot"=>options.depot,
        "source_sha256"=>before, "machine"=>machine_metadata(allowed,nodes),
        "cgroup"=>Matched.read_cgroup(),
        "counter_scope"=>"per-worker user-only whole trajectory plus counter-region boundaries; counter-on cohort separate from timing",
        "normalizer"=>"1e9 * timed complete-trajectory seconds / (nx*ny*steps)")
    config["configuration_sha256"] = bytes2hex(SHA.sha256(codeunits(JSON.json(config))))
    writejson(joinpath(options.output, "configuration.json"), config)
    fixtures = inputs(options, options.output, jobs, masks["spread"][1])
    manifest(options) == before || error("diagnostic source changed during references")
    rng = MersenneTwister(options.seed)
    schedule = Tuple{Int,Int,Int,String,Int,Int,Int}[]
    for repeat in 0:options.repeats-1
        shuffled = copy(jobs)
        shuffle!(rng, shuffled)
        for (n,p,place,steps,pad,counters) in shuffled
            push!(schedule, (repeat,n,p,place,steps,pad,counters))
        end
    end
    writejson(joinpath(options.output, "job_order.json"), schedule)
    raw = joinpath(options.output, "raw.jsonl")
    touch(raw)
    for (index, (repeat,n,p,place,steps,pad,counters)) in enumerate(schedule)
        manifest(options) == before || error("diagnostic source changed during samples")
        cpus = masks[place][p]
        initial = joinpath(options.output, "fields", "q0-$n.bin")
        oracle = joinpath(options.output, "fields", "oracle-$n-s$steps.bin")
        dump = joinpath(options.output, "fields", "sample-$(lpad(index-1,3,'0')).bin")
        record = invoke(options,n,p,place,steps,pad,counters,cpus,initial,dump)
        evidence = Matched.compare_fields(dump,oracle,n*n)
        key = "$n:$steps"
        merge!(record, evidence, Dict(
            "repeat"=>repeat, "index"=>index-1, "fixture"=>fixtures[key],
            "configuration_sha256"=>config["configuration_sha256"],
            "timestamp_utc"=>Matched.utcstamp()))
        evidence["field_mismatches"] == 0 && evidence["field_max_abs_error"] == 0.0 &&
            evidence["field_sha256"] == fixtures[key]["oracle_sha256"] ||
            error("non-exact full-field parity; dump retained $dump")
        abs(evidence["field_sum"]/(n*n)-1.0) <= 2e-12 ||
            error("mass conservation failure; dump retained $dump")
        appendjsonl(raw, record)
        rm(dump)
        println("$index/$(length(schedule)) $(options.mode) N$n p$p $place " *
                "pad$pad counters$counters $(round(record["ns_per_unknown_step"]; digits=3)) ns/unknown/step")
    end
    manifest(options) == before || error("diagnostic source changed after samples")
    summary = summarize(options.output, config)
    println("complete $(options.mode): $(summary["samples"]) samples")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CacheCliffFortranControls.main()
