"""Matched, serialized fresh-process Julia/Fortran LW strong-scaling study."""
module CrossLanguageStudy

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using Dates
using JSON
using Random
using SHA
using Statistics
using TOML

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", "..", ".."))
const TOY = joinpath(ROOT, "research", "UniformLWAdvection")
const FORTRAN = joinpath(TOY, "fortran")
const DRIVER = joinpath(HERE, "julia_driver.jl")
const TILE = (512, 16)
const SIZES = (512, 1024, 2048, 4096)
const COUNTS = (1, 2, 4, 8, 16, 32)
const PILOT_STEPS = Dict(512=>256, 1024=>64, 2048=>16, 4096=>4)

utcstamp() = string(Dates.now(Dates.UTC), "Z")

function cli(args)
    known = Set(("julia", "project", "depot", "fortran", "output", "repeats",
                 "target-seconds", "seed", "workload", "phase", "workload-from",
                 "placement-sizes", "mode", "padding", "counters",
                 "result-root", "binary"))
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

function parse_options(args)
    options = cli(args)
    output = abspath(single(options, "output"))
    julia = abspath(single(options, "julia", "/tmp/julia-1.10.12/bin/julia"))
    project = abspath(single(options, "project", "/tmp/twophase-compat110"))
    depot = abspath(single(options, "depot", "/tmp/twophase-julia110-depot"))
    fortran = abspath(single(options, "fortran", joinpath(FORTRAN, "build", "lw_omp")))
    repeats = parse(Int, single(options, "repeats", "5"))
    target_seconds = parse(Float64, single(options, "target-seconds", "0.5"))
    seed = parse(Int, single(options, "seed", "30092026"))
    workload = haskey(options, "workload") ? abspath(single(options, "workload")) : nothing
    !ispath(output) || error("output already exists: $output")
    repeats >= 1 && target_seconds > 0 || error("repeats and target seconds must be positive")
    all(ispath, (julia, fortran, project)) ||
        error("Julia, Fortran executable, or resolved Julia project missing")
    return (; julia, project, depot, fortran, output, repeats, target_seconds, seed, workload)
end

function frozen_files(binary, project)
    paths = String[joinpath(HERE, "study.jl"), DRIVER,
                   joinpath(HERE, "summarize.jl"), joinpath(HERE, "README.md"),
                   joinpath(HERE, "..", "Support.jl"),
                   joinpath(HERE, "..", "Project.toml"),
                   joinpath(HERE, "..", "Manifest.toml"),
                   joinpath(TOY, "Project.toml")]
    for directory in (joinpath(TOY, "src"), joinpath(TOY, "ext"))
        for (root, _, names) in walkdir(directory)
            append!(paths, joinpath(root, name) for name in names if endswith(name, ".jl"))
        end
    end
    for (root, _, names) in walkdir(FORTRAN)
        append!(paths, joinpath(root, name) for name in names
                if endswith(name, ".f90") || endswith(name, ".F90"))
    end
    append!(paths, [joinpath(FORTRAN, "Makefile"), joinpath(FORTRAN, "README.md"),
                    joinpath(project, "Project.toml"), joinpath(project, "Manifest.toml"),
                    binary])
    missing = filter(path -> !isfile(path), paths)
    isempty(missing) || error("required comparison source missing: $(join(missing, ", "))")
    baseline = joinpath(project, "baseline.json")
    isfile(baseline) && push!(paths, baseline)
    return Dict(path=>sha256file(path) for path in sort(unique(paths)))
end

function compare_fields(actual, expected, count; atol=1e-12, rtol=1e-12)
    filesize(actual) == 8count && filesize(expected) == 8count ||
        error("full-field byte count mismatch")
    max_abs = max_scaled = actual_sum = actual_sumsq = 0.0
    mismatches = 0
    open(actual) do a
        open(expected) do b
            remaining = count
            while remaining > 0
                width = min(remaining, 65536)
                va = read!(a, Vector{Float64}(undef, width))
                vb = read!(b, Vector{Float64}(undef, width))
                for k in eachindex(va)
                    isfinite(va[k]) && isfinite(vb[k]) || error("nonfinite field value")
                    difference = abs(va[k] - vb[k])
                    scale = atol + rtol * abs(vb[k])
                    max_abs = max(max_abs, difference)
                    max_scaled = max(max_scaled, difference / scale)
                    mismatches += difference > scale
                    actual_sum += va[k]
                    actual_sumsq += va[k] * va[k]
                end
                remaining -= width
            end
        end
    end
    return Dict{String,Any}(
        "field_sha256"=>sha256file(actual), "field_max_abs_error"=>max_abs,
        "field_max_tolerance_fraction"=>max_scaled,
        "field_mismatches"=>mismatches, "field_sum"=>actual_sum,
        "field_sum_squares"=>actual_sumsq, "field_atol"=>atol, "field_rtol"=>rtol)
end

function execution_env(cpus, depot)
    return Dict("OPENBLAS_NUM_THREADS"=>"1", "MKL_NUM_THREADS"=>"1",
        "JULIA_NUM_GC_THREADS"=>"1", "OMP_NUM_THREADS"=>string(length(cpus)),
        "OMP_DYNAMIC"=>"FALSE", "OMP_MAX_ACTIVE_LEVELS"=>"1",
        "OMP_WAIT_POLICY"=>"ACTIVE", "GOMP_SPINCOUNT"=>"300000",
        "OMP_PROC_BIND"=>"close",
        "OMP_PLACES"=>join(["{$cpu}" for cpu in cpus], ","),
        "JULIA_DEPOT_PATH"=>depot * ":" * joinpath(homedir(), ".julia"))
end

function launch(command, cpus, environment)
    result = run_capture(command; env=environment, cpus)
    result.exitcode == 0 || error("command failed ($(result.exitcode)): $command\n" *
                                "stdout: $(result.stdout)\nstderr: $(result.stderr)")
    return strip(result.stdout), result.seconds
end

function invoke(language, options, nx, steps, workers, cpus, input_path,
                output_path, report_path, kind)
    environment = execution_env(cpus, options.depot)
    if language == "julia"
        mode = kind == "oracle" ? "oracle" : "measure"
        command = String[options.julia, "--startup-file=no", "--project=$(options.project)",
            "--threads=$workers", "--gcthreads=1", DRIVER, "mode=$mode", "nx=$nx",
            "ny=$nx", "steps=$steps", "tile-x=$(TILE[1])", "tile-y=$(TILE[2])",
            "input=$input_path", "output=$output_path", "report=$report_path"]
        kind == "oracle" || push!(command, "cpus=$(join(cpus, ','))")
        _, process_wall = launch(command, cpus, environment)
        record = TOML.parsefile(report_path)
    elseif language == "fortran"
        command = String[options.fortran, "--nx", string(nx), "--ny", string(nx),
            "--steps", string(steps), "--threads", string(workers),
            "--tile-x", string(TILE[1]), "--tile-y", string(TILE[2]),
            "--cpus", join(cpus, ','), "--input", input_path, "--output", output_path]
        stdout, process_wall = launch(command, cpus, environment)
        isempty(stdout) && error("Fortran did not emit a JSON record")
        record = JSON.parse(last(split(stdout, '\n')))
    else
        error("unknown language $language")
    end
    merge!(record, Dict("process_wall_seconds"=>process_wall,
                        "requested_cpu_order"=>cpus, "language"=>language,
                        "kind"=>kind))
    (record["nx"], record["ny"], record["steps"], record["workers"]) ==
        (nx, nx, steps, workers) || error("reported workload or team differs from command")
    (record["tile_x"], record["tile_y"]) == TILE ||
        error("reported tile differs from frozen layout")
    isapprox(record["dt"], 0.35/nx; rtol=2e-15) ||
        error("reported timestep differs from contract")
    isfinite(record["time_seconds"]) && record["time_seconds"] > 0 ||
        error("invalid timed trajectory")
    if language == "julia" && kind != "oracle"
        affinity = record["affinity"]
        [item["cpu"] for item in affinity] == cpus &&
            [item["observed_cpu"] for item in affinity] == cpus ||
            error("Julia per-worker binding mismatch")
    elseif language == "fortran"
        record["affinity_cpu_ids"] == cpus && record["observed_cpu_ids"] == cpus &&
            record["pin_verified"] === true || error("Fortran per-worker binding mismatch")
    end
    isapprox(record["final_time"], steps*record["dt"]; rtol=1e-12, atol=1e-12) ||
        error("trajectory final time mismatch")
    isapprox(record["seconds_per_unknown_step"],
             record["time_seconds"]/(nx*nx*steps); rtol=2e-12) ||
        error("reported per-unknown step metric mismatch")
    return record
end

function read_cgroup()
    paths = ("/sys/fs/cgroup/cpu.max", "/proc/self/cgroup")
    return Dict(path=>strip(read(path, String)) for path in paths if isfile(path))
end

function create_initial(options, nx, cpus, output)
    command = String[options.julia, "--startup-file=no", "--project=$(options.project)",
        "--threads=1", DRIVER, "mode=init", "nx=$nx", "ny=$nx", "output=$output"]
    launch(command, cpus, execution_env(cpus, options.depot))
    filesize(output) == 8nx*nx || error("initial field byte count mismatch")
end

function make_jobs(rng, repeats; sizes=SIZES, counts=COUNTS)
    jobs = Tuple{Int,Int,Int,String}[]
    for repeat in 0:repeats-1
        pairs = [(n,p) for n in sizes for p in counts]
        shuffle!(rng, pairs)
        order = vcat(fill("julia", length(pairs) ÷ 2),
                     fill("fortran", length(pairs) - length(pairs) ÷ 2))
        shuffle!(rng, order)
        for ((n,p), first) in zip(pairs, order)
            second = first == "julia" ? "fortran" : "julia"
            push!(jobs, (repeat,n,p,first), (repeat,n,p,second))
        end
    end
    return jobs
end

function snapshot_sources(source, output)
    for name in keys(source)
        original = abspath(name)
        relative = startswith(original, ROOT * "/") ? relpath(original, ROOT) :
                   joinpath("resolved-project", basename(original))
        target = joinpath(output, "source-snapshot", relative)
        mkpath(dirname(target))
        cp(original, target; force=true)
    end
end

function main(args=ARGS)
    options = parse_options(args)
    allowed, nodes = topology()
    cpus = Dict(p=>placement_mask(p, "spread", allowed, nodes) for p in COUNTS)
    source = frozen_files(options.fortran, options.project)
    rng = MersenneTwister(options.seed)
    fields = joinpath(options.output, "fields")
    mkpath(fields)
    snapshot_sources(source, options.output)
    writejson(joinpath(options.output, "source_sha256.json"), source)
    flags = [strip(split(line, '='; limit=2)[2]) for line in
             eachline(joinpath(FORTRAN, "Makefile")) if startswith(line, "FFLAGS =")]
    config = Dict{String,Any}(
        "schema"=>1, "created_utc"=>utcstamp(), "sizes"=>collect(SIZES),
        "threads"=>collect(COUNTS), "tile"=>collect(TILE),
        "repeats"=>options.repeats, "target_seconds"=>options.target_seconds,
        "seed"=>options.seed, "cpu_order"=>cpus,
        "julia"=>options.julia,
        "julia_version"=>command_output(options.julia, "--version"),
        "julia_project"=>options.project, "julia_depot"=>options.depot,
        "fortran_binary"=>options.fortran,
        "fortran_version"=>first(split(command_output("gfortran", "--version"), '\n')),
        "fortran_flags"=>isempty(flags) ? "unknown" : first(flags),
        "OMP_DYNAMIC"=>"FALSE", "OMP_MAX_ACTIVE_LEVELS"=>"1",
        "OMP_WAIT_POLICY"=>"ACTIVE", "GOMP_SPINCOUNT"=>"300000",
        "OMP_PROC_BIND"=>"close", "OMP_PLACES"=>"explicit singleton selected CPUs",
        "OPENBLAS_NUM_THREADS"=>"1", "JULIA_NUM_GC_THREADS"=>"1",
        "metric"=>"1e9 * timed complete trajectory seconds / (nx*ny*steps)",
        "source_sha256"=>source, "machine"=>machine_metadata(allowed, nodes),
        "cgroup"=>read_cgroup(),
        "git_revision"=>command_output("git", "-C", ROOT, "rev-parse", "HEAD"))
    config["configuration_sha256"] = bytes2hex(SHA.sha256(codeunits(JSON.json(config))))
    writejson(joinpath(options.output, "configuration.json"), config)
    if options.workload !== nothing
        steps = Dict(parse(Int,key)=>Int(value) for (key,value) in readjson(options.workload))
    else
        pilots = Any[]
        steps = Dict{Int,Int}()
        for nx in SIZES
            initial = joinpath(fields, "q0-$nx.bin")
            create_initial(options, nx, cpus[1], initial)
            rates = Float64[]
            for language in ("julia", "fortran"), p in (4,8,16,32)
                output = joinpath(fields, "pilot-$language-$nx-p$p.bin")
                report = joinpath(fields, "pilot.toml")
                record = invoke(language, options, nx, PILOT_STEPS[nx], p, cpus[p],
                                initial, output, report, "pilot")
                push!(pilots, record)
                push!(rates, record["time_seconds"] / PILOT_STEPS[nx])
                rm(output)
            end
            steps[nx] = max(1, ceil(Int, options.target_seconds / minimum(rates)))
            println("calibrated $(nx)²: $(steps[nx]) complete steps")
        end
        for record in pilots
            appendjsonl(joinpath(options.output, "pilots.jsonl"), record)
        end
    end
    Set(keys(steps)) == Set(SIZES) && all(>(0), values(steps)) ||
        error("workload must have positive steps for all primary meshes")
    writejson(joinpath(options.output, "workload.json"), steps)
    for nx in SIZES
        initial = joinpath(fields, "q0-$nx.bin")
        isfile(initial) || create_initial(options, nx, cpus[1], initial)
        oracle = joinpath(fields, "oracle-$nx.bin")
        invoke("julia", options, nx, steps[nx], 1, cpus[1], initial,
               oracle, joinpath(fields, "oracle-$nx.toml"), "oracle")
        compare_fields(oracle, oracle, nx*nx)
        println("validated reference $(nx)², steps=$(steps[nx])")
    end
    fixtures = Dict(string(nx)=>Dict(
        "initial_field_sha256"=>sha256file(joinpath(fields, "q0-$nx.bin")),
        "reference_field_sha256"=>sha256file(joinpath(fields, "oracle-$nx.bin")),
        "steps"=>steps[nx], "dt"=>0.35/nx) for nx in SIZES)
    writejson(joinpath(options.output, "fixtures.json"), fixtures)
    frozen_files(options.fortran, options.project) == source ||
        error("source changed during workload calibration")
    jobs = make_jobs(rng, options.repeats)
    writejson(joinpath(options.output, "job_order.json"), jobs)
    raw = joinpath(options.output, "raw.jsonl")
    touch(raw)
    for (index, (repeat, nx, p, language)) in enumerate(jobs)
        frozen_files(options.fortran, options.project) == source ||
            error("source changed after configuration freeze")
        field = joinpath(fields, "sample-r$repeat-$language-$nx-p$p.bin")
        record = invoke(language, options, nx, steps[nx], p, cpus[p],
            joinpath(fields, "q0-$nx.bin"), field, joinpath(fields, "sample.toml"), "sample")
        evidence = compare_fields(field, joinpath(fields, "oracle-$nx.bin"), nx*nx)
        merge!(record, evidence, Dict(
            "repeat"=>repeat, "sample_index"=>index-1,
            "configuration_sha256"=>config["configuration_sha256"],
            "initial_field_sha256"=>fixtures[string(nx)]["initial_field_sha256"],
            "reference_field_sha256"=>fixtures[string(nx)]["reference_field_sha256"],
            "source_manifest_sha256"=>sha256file(joinpath(options.output, "source_sha256.json")),
            "timestamp_utc"=>utcstamp()))
        evidence["field_mismatches"] == 0 || error("full-field mismatch; dump retained: $field")
        isapprox(record["mass"], evidence["field_sum"]/(nx*nx); rtol=2e-12, atol=2e-12) ||
            error("reported mass differs from full field: $field")
        abs(evidence["field_sum"]/(nx*nx) - 1.0) <= 2e-12 ||
            error("mass conservation failed: $field")
        record["time_seconds"] >= 0.2 || error("sample shorter than 0.2 s: $field")
        all(==(p), record["active_worker_counts"]) ||
            error("worker participation failed: $field")
        appendjsonl(raw, record)
        rm(field)
        normalized = 1e9*record["time_seconds"]/(nx*nx*steps[nx])
        println("$index/$(length(jobs)) $language $(nx)² p$p r$(repeat+1): " *
                "$(round(normalized; digits=3)) ns/cell/step")
    end
    frozen_files(options.fortran, options.project) == source ||
        error("source changed after measurements")
    include(joinpath(HERE, "summarize.jl"))
    CrossLanguageSummary.summarize_study(options.output)
    println("completed $(length(jobs)) fresh-process samples: $(options.output)")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CrossLanguageStudy.main()
