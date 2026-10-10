"""Controlled follow-up for the 1024² to 2048² LW cache/scaling cliff."""
module CacheCliffStudy

include(joinpath(@__DIR__, "..", "cross_language", "study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using Dates
using JSON
using Random
using SHA

const Matched = CrossLanguageStudy
const HERE = @__DIR__
const ROOT = Matched.ROOT
const LADDER = (768, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304)
const PLACEMENT = (1024, 1280, 1536, 2048)
const SHORT = (1024, 2048)

function options(args)
    parsed = Matched.cli(args)
    phase = Matched.single(parsed, "phase")
    phase in ("ladder", "placement", "steps") || error("unknown phase $phase")
    output = abspath(Matched.single(parsed, "output"))
    julia = abspath(Matched.single(parsed, "julia", "/tmp/julia-1.10.12/bin/julia"))
    project = abspath(Matched.single(parsed, "project", "/tmp/twophase-compat110"))
    depot = abspath(Matched.single(parsed, "depot", "/tmp/twophase-julia110-depot"))
    fortran = abspath(Matched.single(parsed, "fortran", joinpath(Matched.FORTRAN, "build", "lw_omp")))
    workload_from = haskey(parsed, "workload-from") ?
        abspath(Matched.single(parsed, "workload-from")) : nothing
    placement_sizes = parse.(Int, get(parsed, "placement-sizes", string.(PLACEMENT)))
    target_seconds = parse(Float64, Matched.single(parsed, "target-seconds", "0.5"))
    repeats = parse(Int, Matched.single(parsed, "repeats", phase == "steps" ? "5" : "3"))
    seed = parse(Int, Matched.single(parsed, "seed", "30092027"))
    !ispath(output) || error("output already exists")
    repeats >= 1 && target_seconds > 0 || error("invalid repeats or target seconds")
    all(isfile, (julia, fortran)) && isdir(project) ||
        error("Julia, Fortran, or child project missing")
    return (; phase, output, julia, project, depot, fortran, workload_from,
            placement_sizes, target_seconds, repeats, seed)
end

function source_paths(options)
    paths = String[joinpath(HERE, "cache_study.jl"), joinpath(HERE, "report.jl"),
        joinpath(HERE, "..", "Support.jl"),
        joinpath(HERE, "..", "Project.toml"), joinpath(HERE, "..", "Manifest.toml"),
        joinpath(HERE, "..", "cross_language", "study.jl"),
        Matched.DRIVER, joinpath(Matched.TOY, "src", "UniformLWAdvection.jl"),
        joinpath(Matched.TOY, "ext", "UniformLWAdvectionPolyesterExt.jl"),
        joinpath(Matched.FORTRAN, "lw_omp.f90"), joinpath(Matched.FORTRAN, "Makefile"),
        joinpath(Matched.FORTRAN, "test.jl"), options.fortran,
        joinpath(options.project, "Project.toml"),
        joinpath(options.project, "Manifest.toml")]
    missing = filter(path->!isfile(path), paths)
    isempty(missing) || error("required study source missing: $(join(missing, ", "))")
    return paths
end

source_manifest(options) = Dict(path=>sha256file(path) for path in source_paths(options))

function snapshot(options, output, manifest)
    for name in keys(manifest)
        relative = startswith(name, ROOT * "/") ? relpath(name, ROOT) :
                   joinpath("resolved-project", basename(name))
        target = joinpath(output, "source-snapshot", relative)
        mkpath(dirname(target))
        cp(name, target; force=true)
    end
end

function initial_and_oracle(options, output, nx, steps, mask1)
    fields = joinpath(output, "fields")
    initial, oracle = joinpath(fields, "q0-$nx.bin"),
                      joinpath(fields, "oracle-$nx-s$steps.bin")
    isfile(initial) || Matched.create_initial(options, nx, mask1, initial)
    if !isfile(oracle)
        record = Matched.invoke("julia", options, nx, steps, 1, mask1, initial,
                                oracle, joinpath(fields, "oracle.toml"), "oracle")
        Matched.compare_fields(oracle, oracle, nx*nx)
        println("oracle $(nx)² steps=$steps $(round(record["time_seconds"]; digits=3))s")
    end
    return initial, oracle
end

function workload(options, output, mask16)
    if options.phase == "steps"
        return Dict("$n:$steps"=>steps for n in SHORT for steps in (64,256))
    end
    sizes = options.phase == "ladder" ? LADDER : options.placement_sizes
    previous = options.workload_from === nothing ? Dict() : readjson(options.workload_from)
    pilots = Any[]
    frozen = Dict{Int,Int}()
    for nx in sizes
        if haskey(previous, string(nx))
            frozen[nx] = Int(previous[string(nx)])
            continue
        end
        initial = joinpath(output, "fields", "q0-$nx.bin")
        Matched.create_initial(options, nx, mask16[1:1], initial)
        rates = Float64[]
        pilot_steps = nx >= 1792 ? 64 : nx >= 1408 ? 128 : 256
        for language in ("julia", "fortran")
            dump = joinpath(output, "fields", "pilot-$language-$nx.bin")
            record = Matched.invoke(language, options, nx, pilot_steps, 16, mask16,
                initial, dump, joinpath(output, "fields", "pilot.toml"), "pilot")
            push!(rates, record["time_seconds"] / pilot_steps)
            push!(pilots, record)
            rm(dump)
        end
        frozen[nx] = max(1, ceil(Int, options.target_seconds / minimum(rates)))
        println("frozen $(nx)²: $(frozen[nx]) steps")
    end
    for row in pilots
        appendjsonl(joinpath(output, "pilots.jsonl"), row)
    end
    return frozen
end

function jobs_for(options, work)
    if options.phase == "ladder"
        return [(n,16,"spread",work[n]) for n in LADDER]
    elseif options.phase == "placement"
        return [(n,8,place,work[n]) for n in options.placement_sizes
                for place in ("compact", "spread")]
    end
    return [(n,16,"spread",steps) for n in SHORT for steps in (64,256)]
end

function balanced_schedule(jobs, repeats, seed)
    rng = MersenneTwister(seed)
    schedule = Tuple{Int,Int,Int,String,Int,String}[]
    for repeat in 0:repeats-1
        pairs = copy(jobs)
        shuffle!(rng, pairs)
        firsts = vcat(fill("julia", length(pairs) ÷ 2),
                      fill("fortran", length(pairs)-length(pairs) ÷ 2))
        shuffle!(rng, firsts)
        for (job, language) in zip(pairs, firsts)
            n,p,place,steps = job
            other = language == "julia" ? "fortran" : "julia"
            push!(schedule, (repeat,n,p,place,steps,language),
                            (repeat,n,p,place,steps,other))
        end
    end
    return schedule
end

function main(args=ARGS)
    options = CacheCliffStudy.options(args)
    allowed, nodes = topology()
    cpu_orders = Dict(mode=>Dict(p=>placement_mask(p, mode, allowed, nodes)
                                for p in (1,8,16))
                      for mode in ("spread", "compact"))
    mkpath(joinpath(options.output, "fields"))
    before = source_manifest(options)
    snapshot(options, options.output, before)
    writejson(joinpath(options.output, "source_sha256.json"), before)
    flags = collect(eachline(joinpath(Matched.FORTRAN, "Makefile")))
    config = Dict{String,Any}(
        "schema"=>1, "phase"=>options.phase, "created_utc"=>Matched.utcstamp(),
        "tile"=>collect(Matched.TILE), "repeats"=>options.repeats, "seed"=>options.seed,
        "placement_sizes"=>options.phase == "placement" ? options.placement_sizes : nothing,
        "workload_source"=>options.workload_from,
        "julia"=>options.julia, "project"=>options.project, "depot"=>options.depot,
        "fortran"=>options.fortran, "fortran_flags"=>flags[2],
        "cpu_orders"=>cpu_orders, "source_sha256"=>before,
        "machine"=>machine_metadata(allowed, nodes), "cgroup"=>Matched.read_cgroup(),
        "normalizer"=>"1e9 * timed full-trajectory seconds / (nx*ny*steps)",
        "short_sample_policy"=>"steps phase is diagnostic; sub-0.2-second samples retained and flagged")
    config["configuration_sha256"] = bytes2hex(SHA.sha256(codeunits(JSON.json(config))))
    writejson(joinpath(options.output, "configuration.json"), config)
    work = workload(options, options.output, cpu_orders["spread"][16])
    writejson(joinpath(options.output, "workload.json"), work)
    jobs = jobs_for(options, work)
    for (nx,_,_,steps) in jobs
        initial_and_oracle(options, options.output, nx, steps, cpu_orders["spread"][1])
    end
    fixtures = Dict("$nx:$steps"=>Dict(
        "q0_sha256"=>sha256file(joinpath(options.output, "fields", "q0-$nx.bin")),
        "oracle_sha256"=>sha256file(joinpath(options.output, "fields", "oracle-$nx-s$steps.bin")))
        for (nx,_,_,steps) in jobs)
    writejson(joinpath(options.output, "fixtures.json"), fixtures)
    source_manifest(options) == before || error("source changed during calibration/oracles")
    schedule = balanced_schedule(jobs, options.repeats, options.seed)
    writejson(joinpath(options.output, "job_order.json"), schedule)
    raw = joinpath(options.output, "raw.jsonl")
    touch(raw)
    for (index, (repeat,n,p,place,steps,language)) in enumerate(schedule)
        source_manifest(options) == before || error("source changed during diagnostic study")
        cpus = cpu_orders[place][p]
        initial = joinpath(options.output, "fields", "q0-$n.bin")
        oracle = joinpath(options.output, "fields", "oracle-$n-s$steps.bin")
        dump = joinpath(options.output, "fields", "sample-$(lpad(index-1,3,'0'))-$language.bin")
        record = Matched.invoke(language, options, n, steps, p, cpus, initial, dump,
                                joinpath(options.output, "fields", "sample.toml"), "sample")
        evidence = Matched.compare_fields(dump, oracle, n*n)
        merge!(record, evidence, Dict(
            "repeat"=>repeat, "sample_index"=>index-1, "placement"=>place,
            "ns_per_unknown_step"=>1e9*record["time_seconds"]/(n*n*steps),
            "tile_count"=>cld(n,512)*cld(n,16),
            "partial_x_tile_width"=>n%512, "partial_y_tile_height"=>n%16,
            "fixture"=>fixtures["$n:$steps"],
            "configuration_sha256"=>config["configuration_sha256"],
            "short_diagnostic"=>record["time_seconds"] < .2,
            "timestamp_utc"=>Matched.utcstamp()))
        evidence["field_mismatches"] == 0 || error("full-field mismatch, dump retained: $dump")
        all(==(p), record["active_worker_counts"]) ||
            error("worker participation mismatch, dump retained: $dump")
        abs(evidence["field_sum"]/(n*n)-1.0) <= 2e-12 ||
            error("mass conservation failure, dump retained: $dump")
        options.phase == "steps" || !record["short_diagnostic"] ||
            error("screen sample shorter than 0.2s, dump retained: $dump")
        appendjsonl(raw, record)
        rm(dump)
        println("$index/$(length(schedule)) $(options.phase) $language $(n)² p$p $place s$steps: " *
                "$(round(record["ns_per_unknown_step"]; digits=3)) ns/unknown/step")
    end
    source_manifest(options) == before || error("source changed after samples")
    include(joinpath(HERE, "report.jl"))
    CacheCliffReport.summarize(options.output)
    println("complete $(options.phase): $(length(schedule)) samples")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CacheCliffStudy.main()
