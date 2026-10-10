#!/usr/bin/env julia
module ScalingRunner

include(joinpath(@__DIR__,"Support.jl"))
using .ScalingSupport
using Dates
using JSON
using Random
using SHA
using Statistics
using TOML

const ROOT = normpath(joinpath(@__DIR__,"..",".."))
const DRIVER = joinpath(@__DIR__,"driver.jl")

function runner_hash()
    files = [DRIVER, @__FILE__, joinpath(@__DIR__,"plot.jl"),
             joinpath(@__DIR__,"compare.jl"), joinpath(@__DIR__,"screen.jl"),
             joinpath(@__DIR__,"Support.jl"),
             joinpath(@__DIR__,"Project.toml"),joinpath(@__DIR__,"Manifest.toml")]
    all(isfile,files) || error("missing Julia scaling runner dependency")
    context = SHA.SHA2_256_CTX()
    for path in files
        SHA.update!(context,codeunits(basename(path)))
        SHA.update!(context,read(path))
    end
    return bytes2hex(SHA.digest!(context))
end

function parse_cli(args)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: run.jl [--project DIR] [--julia EXEC] [--depot DIR] ",
                "[--cases CASE ...] [--sizes N ...] [--shapes NxM ...] ",
                "[--threads P ...] [--repeats N] [--seconds S] ",
                "[--placement ordinary|compact|spread] [--pin-workers] ",
                "[--backend NAME] [--tile N] [--tile-x N] [--threshold N] ",
                "[--block N] [--launch-items N] [--flux recompute|stored] ",
                "[--steps N] [--workload FILE] [--allow-large-upwind] ",
                "[--allow-large-wy] [--diagnostics summary|sampled] ",
                "[--diagnostic-interval N] [--seed N] [--output DIR] ",
                "[--profile] [--smoke]")
        return nothing
    end
    options = Dict{String,Any}(
        "project" => ROOT, "julia" => joinpath(Sys.BINDIR,Base.julia_exename()),
        "depot" => nothing, "profile" => false,
        "cases" => ["stream","stencil","gather","launch","toy"],
        "sizes" => [512,1024,2048,4096], "shapes" => String[],
        "threads" => [1,2,4,8,16], "repeats" => 5, "seconds" => 0.25,
        "placement" => "ordinary", "pin_workers" => false,
        "backend" => "auto", "tile" => 32, "tile_x" => 128,
        "threshold" => 16384, "block" => 32, "launch_items" => nothing,
        "flux" => "recompute", "steps" => 0, "workload" => nothing,
        "allow_large_upwind" => false, "allow_large_wy" => false,
        "diagnostics" => "summary", "diagnostic_interval" => 10,
        "seed" => 29092026, "output" => nothing, "smoke" => false)
    boolean = Dict("--profile" => "profile", "--pin-workers" => "pin_workers",
                   "--allow-large-upwind" => "allow_large_upwind",
                   "--allow-large-wy" => "allow_large_wy", "--smoke" => "smoke")
    many = Dict("--cases" => "cases", "--sizes" => "sizes",
                "--shapes" => "shapes", "--threads" => "threads")
    integer = Set(("tile","tile_x","threshold","block","launch_items","steps",
                   "diagnostic_interval","seed","repeats"))
    i = 1
    while i <= length(args)
        flag = args[i]
        if haskey(boolean,flag)
            options[boolean[flag]] = true
            i += 1
        elseif haskey(many,flag)
            values = String[]
            i += 1
            while i <= length(args) && !startswith(args[i],"--")
                push!(values,args[i]); i += 1
            end
            isempty(values) && flag in ("--cases","--threads") &&
                error("$flag needs one or more values")
            key = many[flag]
            options[key] = key in ("sizes","threads") ? parse.(Int,values) : values
        else
            startswith(flag,"--") || error("unknown option: $flag")
            key = replace(flag[3:end],'-' => '_')
            haskey(options,key) || error("unknown option: $flag")
            i < length(args) || error("missing value for $flag")
            value = args[i+1]
            options[key] = key in integer ? parse(Int,value) :
                           key == "seconds" ? parse(Float64,value) : value
            i += 2
        end
    end
    if options["smoke"]
        options["cases"] = ["stream","stencil","gather","launch","upwind","upwind_complete"]
        options["sizes"] = [128]
        options["threads"] = [1,2]
        options["repeats"] = 1
        options["seconds"] = 0.01
    end
    return options
end

function validate_options(o)
    shapes = [(n,n) for n in o["sizes"]]
    for shape in o["shapes"]
        parts = split(lowercase(shape),'x')
        length(parts) == 2 || error("shapes must use NxM")
        push!(shapes,(parse(Int,parts[1]),parse(Int,parts[2])))
    end
    !isempty(shapes) && all(nx > 0 && ny > 0 for (nx,ny) in shapes) ||
        error("provide positive sizes or shapes")
    o["repeats"] >= 1 && all(>(0),o["threads"]) && all(>(0),o["sizes"]) ||
        error("repeats, threads, and sizes must be positive")
    all(o[key] >= 1 for key in ("tile","tile_x","block","diagnostic_interval")) &&
        o["threshold"] >= 0 && o["steps"] >= 0 || error("invalid dimensions or steps")
    o["placement"] in ("ordinary","compact","spread") || error("unknown placement")
    o["flux"] in ("recompute","stored") || error("unknown flux")
    o["diagnostics"] in ("summary","sampled") || error("unknown diagnostics")
    o["seconds"] > 0 && isfinite(o["seconds"]) || error("seconds must be finite and positive")
    package_cases = Set(("upwind","upwind_complete","wy_complete","wy_kernel"))
    if any(case in package_cases for case in o["cases"])
        o["backend"] in ("auto","package") || error("package cases require package backend")
    end
    o["pin_workers"] && o["placement"] == "ordinary" &&
        error("--pin-workers requires compact or spread placement")
    largest = maximum(max(nx,ny) for (nx,ny) in shapes)
    any(case in ("upwind","upwind_complete") for case in o["cases"]) &&
        largest > 1024 && !o["allow_large_upwind"] &&
        error("upwind above 1024² requires --allow-large-upwind")
    any(case in ("wy_complete","wy_kernel") for case in o["cases"]) &&
        largest > 256 && !o["allow_large_wy"] &&
        error("WY above 256² requires --allow-large-wy")
    memory = memory_metadata()
    available = parse(Int,first(split(get(memory,"MemAvailable","0 kB")))) * 1024
    largest_area = maximum(nx*ny for (nx,ny) in shapes)
    if o["allow_large_wy"] && any(case in ("wy_complete","wy_kernel") for case in o["cases"])
        700largest_area <= available/2 || error("estimated WY working set exceeds half available memory")
    end
    if o["allow_large_upwind"] && any(case in ("upwind","upwind_complete") for case in o["cases"])
        500largest_area <= available/2 || error("estimated upwind working set exceeds half available memory")
    end
    resolved = Sys.which(o["julia"])
    resolved !== nothing || error("Julia executable unavailable: $(o["julia"])")
    o["julia"] = resolved
    version = command_output(o["julia"],"--version")
    m = match(r"(?i)julia version (\d+)\.(\d+)",version)
    m !== nothing && parse(Int,m.captures[1]) == 1 || error("unsupported Julia version: $version")
    o["julia_minor"] = parse(Int,m.captures[2])
    o["julia_version_text"] = version
    o["project"] = abspath(o["project"])
    o["workload"] !== nothing && (o["workload"] = abspath(o["workload"]))
    o["output"] !== nothing && (o["output"] = abspath(o["output"]))
    o["depot"] !== nothing && (o["depot"] = abspath(o["depot"]))
    return shapes
end

function invoke(o, case, nx, ny, count, backend, steps, seconds, mask,
                output, more; profile_output=nothing)
    source_before = source_hash(o["project"])
    runner_before = runner_hash()
    toy_before = case == "toy" ? toy_tree_hash() : nothing
    env = Dict("OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
               "JULIA_NUM_GC_THREADS" => "1")
    if o["depot"] !== nothing
        env["JULIA_DEPOT_PATH"] = o["depot"] * ":" * joinpath(homedir(),".julia")
    end
    flag = o["julia_minor"] >= 12 ? "--threads=$count,0" : "--threads=$count"
    command = [o["julia"],"--startup-file=no","--project=$(o["project"])",
               flag,DRIVER,"case=$case","backend=$backend","nx=$nx","ny=$ny",
               "tile=$(o["tile"])","steps=$steps","seconds=$seconds","output=$output"]
    append!(command,["$key=$value" for (key,value) in more])
    if o["pin_workers"]
        append!(command,["pin-workers=true","pin-cpus=$(join(mask,','))"])
    end
    profile_output !== nothing && append!(command,["profile=true","profile-output=$profile_output"])
    started = time_ns()
    result = run_capture(command;env=env,cpus=o["placement"] == "ordinary" ? nothing : mask)
    if result.exitcode != 0
        receipt = Dict("command" => command, "requested_affinity" => mask,
                       "exit_code" => result.exitcode, "termsignal" => result.termsignal,
                       "stdout" => result.stdout, "stderr" => result.stderr,
                       "process_elapsed_seconds" => (time_ns()-started)/1e9)
        directory = get(o,"receipt_dir",dirname(output))
        writejson(joinpath(directory,"failure-$(round(Int,time()*1e9)).json"),receipt)
    end
    result.exitcode == 0 || error("$(join(command,' '))\n$(result.stdout)\n$(result.stderr)")
    record = TOML.parsefile(output)
    record["process_elapsed_seconds"] = (time_ns()-started)/1e9
    record["runtime"]["threads_default"] == count || error("Julia worker count differs")
    record["stdout"] = strip(result.stdout)
    record["requested_workers"] = count
    record["requested_affinity"] = mask
    record["project"] = o["project"]
    record["project_revision"] = project_revision(o["project"])
    record["source_tree_sha256"] = source_before
    source_hash(o["project"]) == source_before && runner_hash() == runner_before ||
        error("project or runner source changed during benchmark process")
    case == "toy" && toy_tree_hash() != toy_before && error("toy source changed")
    record["runner_sha256"] = runner_before
    record["toy_tree_sha256"] = toy_before
    record["placement"] = o["placement"]
    record["pin_workers"] = o["pin_workers"]
    record["profile_output"] = profile_output
    return record
end

function summary(records, seed)
    grouped = Dict{Tuple,Vector{Any}}()
    for record in records
        key = (record["case"],record["backend"],record["nx"],record["ny"],
               record["requested_workers"],record["placement"])
        push!(get!(grouped,key,Any[]),record)
    end
    result = Dict{String,Any}[]
    rng = MersenneTwister(seed)
    for key in sort!(collect(keys(grouped)))
        samples = grouped[key]
        one_key = (key[1],key[2],key[3],key[4],1,key[6])
        one = get(grouped,one_key,nothing)
        times = [row["time_seconds"] for row in samples]
        center = median_value(times)
        baseline = one === nothing ? nothing : median_value([row["time_seconds"] for row in one])
        speedup = baseline === nothing ? nothing : baseline/center
        interval = if key[5] == 1
            [1.0,1.0]
        elseif one !== nothing && length(one) >= 2 && length(samples) >= 2
            one_times = [row["time_seconds"] for row in one]
            replicas = [begin
                            sampled_parallel = rand(rng,times,length(times))
                            sampled_serial = rand(rng,one_times,length(one_times))
                            median_value(sampled_serial)/median_value(sampled_parallel)
                        end for _ in 1:2000]
            sort!(replicas)
            [replicas[51],replicas[1950]]
        else
            [nothing,nothing]
        end
        gate = "screen_only"
        if key[5] in (4,8) && length(times) >= 5
            gate = interval[1] === nothing ? "inconclusive" :
                   interval[1] >= 0.5key[5] ? "timing_target_supported" :
                   interval[2] < 0.5key[5] ? "timing_target_missed" : "inconclusive"
        end
        active_gate = all(row["active_workers"] == key[5] for row in samples)
        gate == "timing_target_supported" && !active_gate && (gate = "invalid_inactive_workers")
        push!(result,Dict(
            "case" => key[1], "backend" => key[2], "nx" => key[3], "ny" => key[4],
            "workers" => key[5], "placement" => key[6], "samples" => length(times),
            "time_median_seconds" => center, "time_mad_seconds" => mad_value(times),
            "speedup" => speedup, "efficiency" => speedup === nothing ? nothing : speedup/key[5],
            "speedup_ci95_bootstrap" => interval, "efficiency_gate_50pct" => gate,
            "all_requested_workers_active_in_probe" => active_gate,
            "active_workers_min" => minimum(row["active_workers"] for row in samples),
            "active_workers_max" => maximum(row["active_workers"] for row in samples),
            "worker_probe_imbalance_median" => median_value([row["worker_probe_imbalance_ratio"] for row in samples]),
            "allocation_median_bytes" => median_value([row["allocation_bytes"] for row in samples]),
            "gc_median_seconds" => median_value([row["gc_seconds"] for row in samples]),
            "peak_rss_median_bytes" => median_value([row["peak_rss_bytes_linux"] for row in samples]),
            "mass_error_max" => maximum(get(row["diagnostics"],"mass_error",0) for row in samples),
            "l2_error_max" => maximum(get(row["diagnostics"],"l2_error",0) for row in samples),
            "working_set_bytes" => samples[1]["fixture"]["working_set_bytes"],
            "steps" => samples[1]["steps"],
            "throughput_cells_per_second" => samples[1]["fixture"]["cells"]*samples[1]["steps"]/center))
    end
    return result
end

function main(args=copy(ARGS))
    o = parse_cli(args)
    o === nothing && return
    shapes = validate_options(o)
    o["depot"] !== nothing && mkpath(o["depot"])
    allowed,nodes = topology()
    for count in o["threads"]
        placement_mask(count,o["placement"],allowed,nodes)
    end
    date = Dates.format(now(UTC),dateformat"yyyymmddTHHMMSS") * "Z"
    destination = o["output"] === nothing ? joinpath(ROOT,"benchmark","scaling","results",date) : o["output"]
    mkpath(destination)
    o["receipt_dir"] = destination
    raw = joinpath(destination,"raw.jsonl")
    !ispath(raw) || error("raw.jsonl already exists: $destination")
    touch(raw)
    config = Dict{String,Any}(
        "schema_version" => 1, "created_utc" => date, "cases" => o["cases"],
        "sizes" => o["sizes"], "shapes" => o["shapes"], "threads" => o["threads"],
        "repeats" => o["repeats"], "seconds_target" => o["seconds"],
        "placement" => o["placement"], "pin_workers" => o["pin_workers"],
        "backend" => o["backend"], "tile" => o["tile"], "tile_x" => o["tile_x"],
        "threshold" => o["threshold"], "block" => o["block"],
        "launch_items" => o["launch_items"], "diagnostics" => o["diagnostics"],
        "diagnostic_interval" => o["diagnostic_interval"], "flux" => o["flux"],
        "seed" => o["seed"], "project" => o["project"],
        "steps_override" => o["steps"], "workload_source" => o["workload"],
        "profile" => o["profile"], "depot" => o["depot"],
        "project_revision" => project_revision(o["project"]),
        "workspace_revision" => command_output("git","-C",ROOT,"rev-parse","HEAD"),
        "source_tree_sha256" => source_hash(o["project"]),
        "runner_sha256" => runner_hash(),
        "tooling_project_sha256" => sha256file(joinpath(@__DIR__,"Project.toml")),
        "tooling_manifest_sha256" => sha256file(joinpath(@__DIR__,"Manifest.toml")),
        "tooling_julia" => joinpath(Sys.BINDIR,Base.julia_exename()),
        "tooling_julia_version" => string(VERSION),
        "toy_tree_sha256" => "toy" in o["cases"] ? toy_tree_hash() : nothing,
        "workspace_dirty" => command_output("git","-C",ROOT,"status","--porcelain"),
        "julia" => o["julia"], "julia_version_text" => o["julia_version_text"],
        "threads_flag_policy" => o["julia_minor"] >= 12 ? "p,0" : "p",
        "machine" => machine_metadata(allowed,nodes))
    config["configuration_sha256"] = bytes2hex(SHA.sha256(codeunits(JSON.json(config))))
    writejson(joinpath(destination,"configuration.json"),config)
    work = Any[]
    frozen = o["workload"] === nothing ? Dict{String,Any}() : readjson(o["workload"])
    mktempdir(prefix="twophase-scaling-") do temporary
        report = joinpath(temporary,"sample.toml")
        for case in o["cases"], (nx,ny) in shapes
            backend = o["backend"] == "auto" ?
                      (case in ("upwind","upwind_complete","wy_complete","wy_kernel") ? "package" : "threads") :
                      o["backend"]
            count = maximum(o["threads"])
            mask = placement_mask(count,o["placement"],allowed,nodes)
            extra = Dict{String,Any}()
            case == "toy" && merge!(extra,Dict("flux" => o["flux"],
                "tile-x" => o["tile_x"],"threshold" => o["threshold"]))
            case in ("upwind","upwind_complete","wy_complete","wy_kernel") &&
                (extra["block"] = o["block"])
            if case in ("wy_complete","wy_kernel")
                extra["diagnostics"] = o["diagnostics"]
                extra["diagnostic-interval"] = o["diagnostic_interval"]
            end
            case == "launch" && o["launch_items"] !== nothing &&
                (extra["launch-items"] = o["launch_items"])
            key = "$case:$nx:$ny"
            steps = o["steps"] > 0 ? o["steps"] :
                    haskey(frozen,key) ? Int(frozen[key]) :
                    Int(invoke(o,case,nx,ny,count,backend,0,o["seconds"],mask,report,extra)["steps"])
            frozen[key] = steps
            for repeat in 0:(o["repeats"]-1), p in o["threads"]
                push!(work,(case,nx,ny,p,backend,steps,copy(extra),repeat))
            end
            println("frozen $case/$backend $nx×$ny: $steps steps")
            flush(stdout)
        end
        writejson(joinpath(destination,"workload.json"),frozen)
        rng = MersenneTwister(o["seed"])
        records = Any[]
        for repeat in 0:(o["repeats"]-1)
            jobs = [job for job in work if job[end] == repeat]
            shuffle!(rng,jobs)
            for (case,nx,ny,count,backend,steps,extra,_) in jobs
                mask = placement_mask(count,o["placement"],allowed,nodes)
                profile_output = o["profile"] ? joinpath(destination,
                    "$case-$(nx)x$(ny)-p$count-repeat$repeat.profile.txt") : nothing
                row = invoke(o,case,nx,ny,count,backend,steps,o["seconds"],
                             mask,report,extra;profile_output=profile_output)
                row["source_tree_sha256"] == config["source_tree_sha256"] &&
                    row["runner_sha256"] == config["runner_sha256"] ||
                    error("source differs from frozen configuration")
                case == "toy" && row["toy_tree_sha256"] != config["toy_tree_sha256"] &&
                    error("toy source differs from frozen configuration")
                row["configuration_sha256"] = config["configuration_sha256"]
                row["repeat"] = repeat
                row["timestamp_utc"] = string(now(UTC))
                appendjsonl(raw,row)
                push!(records,row)
                println("repeat $(repeat+1)/$(o["repeats"]) $(row["stdout"])")
                flush(stdout)
                writejson(joinpath(destination,"summary.json"),summary(records,o["seed"]))
            end
        end
        println("Saved $(length(records)) independent process samples to $destination")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
