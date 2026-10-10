module ScalingSupport

using Dates
using JSON
using Random
using SHA
using Statistics

export readjson, writejson, appendjsonl, readjsonl, sha256file, source_hash,
       toy_tree_hash, project_revision, command_output, topology, placement_mask,
       machine_metadata, memory_metadata, run_capture, median_value, mad_value,
       percentile, bootstrap_ratio, writecsv

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

readjson(path) = JSON.parsefile(path)

function writejson(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, value, 2)
        println(io)
    end
    return path
end

function appendjsonl(path, value)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, JSON.json(value))
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || error("fsync failed: $path")
    end
    return path
end

function readjsonl(path)
    isfile(path) || return Any[]
    return [JSON.parse(line) for line in eachline(path) if !isempty(strip(line))]
end

sha256file(path) = bytes2hex(SHA.sha256(read(path)))

function source_hash(project)
    files = String[]
    for subdir in ("src", "ext")
        directory = joinpath(project, subdir)
        if isdir(directory)
            selected = String[]
            for (root, _, names) in walkdir(directory)
                append!(selected, joinpath(root, name) for name in names if endswith(name, ".jl"))
            end
            append!(files, sort!(selected))
        end
    end
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(project, name)
        isfile(path) && push!(files, path)
    end
    context = SHA.SHA2_256_CTX()
    for path in files
        SHA.update!(context, codeunits(relpath(path, project)))
        SHA.update!(context, read(path))
    end
    return bytes2hex(SHA.digest!(context))
end

toy_tree_hash() = source_hash(joinpath(ROOT, "research", "UniformLWAdvection"))

function command_output(args...)
    try
        return strip(read(pipeline(Cmd(String[string(arg) for arg in args]); stderr=devnull), String))
    catch
        return "unavailable"
    end
end

function project_revision(project)
    snapshot = joinpath(project, "baseline.json")
    if isfile(snapshot)
        return get(readjson(snapshot), "git_commit", "unavailable")
    end
    return command_output("git", "-C", project, "rev-parse", "HEAD")
end

function topology()
    mask = zeros(UInt8, 128)
    ccall(:sched_getaffinity, Cint, (Cint, Csize_t, Ptr{UInt8}),
          0, length(mask), mask) == 0 || error("sched_getaffinity failed")
    allowed = Int[]
    for cpu in 0:(8length(mask)-1)
        !iszero(mask[cpu ÷ 8 + 1] & (UInt8(1) << (cpu % 8))) && push!(allowed, cpu)
    end
    nodes = Dict{Int,Dict{String,String}}()
    for cpu in allowed
        directory = "/sys/devices/system/cpu/cpu$cpu"
        function read_attribute(relative, fallback)
            path = joinpath(directory, relative)
            return isfile(path) ? strip(read(path, String)) : fallback
        end
        nodes[cpu] = Dict("package" => read_attribute("topology/physical_package_id", "0"),
                          "core" => read_attribute("topology/core_id", string(cpu)),
                          "l3" => read_attribute("cache/index3/id", "unknown"))
    end
    return allowed, nodes
end

function placement_mask(count, mode, allowed, nodes)
    count > 0 || error("worker count must be positive")
    mode == "ordinary" && return collect(allowed)
    mode in ("spread", "compact") || error("unknown placement mode: $mode")
    cores = Dict{Tuple{String,String},Vector{Int}}()
    for cpu in allowed
        node = nodes[cpu]
        push!(get!(cores, (node["package"], node["core"]), Int[]), cpu)
    end
    first_cpu = Dict(key => minimum(cpus) for (key, cpus) in cores)
    groups = Dict{String,Vector{Int}}()
    for cpu in values(first_cpu)
        push!(get!(groups, nodes[cpu]["l3"], Int[]), cpu)
    end
    foreach(sort!, values(groups))
    order = Int[]
    if mode == "compact"
        for group in sort!(collect(keys(groups)))
            append!(order, groups[group])
        end
    else
        for index in 1:maximum(length.(values(groups)))
            for group in sort!(collect(keys(groups)))
                index <= length(groups[group]) && push!(order, groups[group][index])
            end
        end
    end
    selected = Set(values(first_cpu))
    append!(order, (cpu for cpu in allowed if !(cpu in selected)))
    count <= length(order) || error("requested $count workers but affinity permits $(length(order)) CPUs")
    return order[1:count]
end

function memory_metadata()
    result = Dict{String,String}()
    path = "/proc/meminfo"
    isfile(path) || return result
    for line in eachline(path)
        if startswith(line, "MemTotal:") || startswith(line, "MemAvailable:")
            key, value = split(line, ':'; limit=2)
            result[key] = strip(value)
        end
    end
    return result
end

function machine_metadata(allowed, nodes)
    governors = Dict{String,String}()
    for cpu in allowed
        path = "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor"
        isfile(path) && (governors[string(cpu)] = strip(read(path, String)))
    end
    boost = Dict{String,String}()
    for path in ("/sys/devices/system/cpu/cpufreq/boost", "/sys/devices/system/cpu/amd_pstate/status")
        isfile(path) && (boost[path] = strip(read(path, String)))
    end
    return Dict("os" => string(Sys.KERNEL), "julia" => string(VERSION),
                "allowed_cpus" => allowed,
                "cpu_topology" => Dict(string(k) => v for (k,v) in nodes),
                "governors" => governors, "boost" => boost,
                "memory" => memory_metadata())
end

"""Run one process; `cpus` binds the entire process with Linux taskset."""
function run_capture(command::Vector{String}; env=Dict{String,String}(), cpus=nothing)
    actual = cpus === nothing ? command :
             vcat(["taskset", "-c", join(cpus, ',')], command)
    out, err = IOBuffer(), IOBuffer()
    started = time_ns()
    try
        process = run(pipeline(setenv(ignorestatus(Cmd(actual)), merge(Dict(ENV), Dict(env))),
                               stdout=out, stderr=err))
        signal = process.termsignal
        code = signal > 0 ? -signal : process.exitcode
        return (exitcode=code, termsignal=signal, stdout=String(take!(out)),
                stderr=String(take!(err)), seconds=(time_ns()-started)/1e9)
    catch exception
        return (exitcode=-1, termsignal=0, stdout=String(take!(out)),
                stderr=String(take!(err)) * "\nlaunch error: " * sprint(showerror,exception),
                seconds=(time_ns()-started)/1e9)
    end
end

function median_value(values)
    sorted = sort!(collect(values))
    n = length(sorted)
    n > 0 || error("median of empty collection")
    return isodd(n) ? sorted[(n+1)÷2] :
           (sorted[n÷2] + sorted[n÷2+1]) / 2
end
mad_value(values) = median_value([abs(value - median_value(values)) for value in values])

function percentile(sorted_values, fraction)
    0 <= fraction <= 1 || error("fraction outside [0,1]")
    location = fraction * (length(sorted_values)-1) + 1
    lo, hi = floor(Int, location), ceil(Int, location)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (location-lo)
end

function bootstrap_ratio(numerators, denominators, factor=1; seed=1, draws=5000)
    rng = MersenneTwister(seed)
    samples = Vector{Float64}(undef, draws)
    for i in eachindex(samples)
        a = median_value(rand(rng, numerators, length(numerators)))
        b = median_value(rand(rng, denominators, length(denominators)))
        samples[i] = a / (factor*b)
    end
    sort!(samples)
    return [percentile(samples, 0.025), percentile(samples, 0.975)]
end

function writecsv(path, rows; fields=nothing)
    isempty(rows) && error("cannot write empty CSV")
    fields === nothing && (fields = sort!(collect(keys(first(rows)))))
    escape(value) = begin
        s = value === nothing ? "" : value isa AbstractString ? value : JSON.json(value)
        occursin(r"[,\"\r\n]", s) ? "\"" * replace(s, '"' => "\"\"") * "\"" : s
    end
    open(path, "w") do io
        println(io, join(fields, ','))
        for row in rows
            println(io, join((escape(get(row, field, nothing)) for field in fields), ','))
        end
    end
    return path
end

end # module
