# Shared Julia helpers for the historical Stage 2 scaling entrypoints.
using Dates
using Random
using SHA
using Statistics
using TOML

const STAGE2_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const STAGE2_EXPERIMENTS = joinpath(STAGE2_ROOT, "research", "UniformLWAdvection", "experiments")
const STAGE2_SCREEN_SCRIPT = joinpath(STAGE2_EXPERIMENTS, "screen_layouts.jl")
const STAGE2_SOURCES = [
    joinpath(@__DIR__, "screen_layouts.jl"), joinpath(@__DIR__, "run.jl"),
    joinpath(@__DIR__, "Support.jl"), joinpath(@__DIR__, "Stage2Common.jl"),
    joinpath(@__DIR__, "Project.toml"),
    STAGE2_SCREEN_SCRIPT,
    [joinpath(STAGE2_EXPERIMENTS, name) for name in
     ("halo.jl", "fused.jl", "fused_affine.jl", "fused_affine_polyester.jl")]...,
    joinpath(STAGE2_ROOT, "research", "UniformLWAdvection", "src", "UniformLWAdvection.jl"),
    joinpath(STAGE2_ROOT, "research", "UniformLWAdvection", "ext", "UniformLWAdvectionPolyesterExt.jl"),
    joinpath(STAGE2_ROOT, "research", "UniformLWAdvection", "Project.toml"),
]

function stage2_options(args, names; defaults=Dict{String, Any}(), flags=Set{String}())
    opts = Dict{String, Any}(defaults)
    index = 1
    while index <= length(args)
        key = args[index]
        if key in ("-h", "--help")
            println("Options: ", join(["--" * name for name in names], ", "))
            return nothing
        end
        startswith(key, "--") || error("unexpected positional argument: $key")
        name = key[3:end]
        name in names || error("unknown option $key")
        if name in flags
            opts[name] = true
            index += 1
        else
            index < length(args) || error("$key requires a value")
            opts[name] = args[index + 1]
            index += 2
        end
    end
    return opts
end

stage2_utc() = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "+00:00"
stage2_stamp() = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS") * "Z"

function stage2_digest(paths; root=STAGE2_ROOT)
    context = SHA.SHA2_256_CTX()
    for path in paths
        update!(context, codeunits(relpath(path, root)))
        update!(context, read(path))
    end
    return bytes2hex(digest!(context))
end

function stage2_copy_snapshot(destination, paths; root=STAGE2_ROOT)
    for source in paths
        target = joinpath(destination, "source-snapshot", relpath(source, root))
        mkpath(dirname(target))
        cp(source, target; force=true)
    end
end

function stage2_copy_project(destination, project)
    for name in ("Project.toml", "Manifest.toml", "baseline.json")
        source = joinpath(project, name)
        isfile(source) && cp(source, joinpath(destination, name); force=true)
    end
    receipt = "/tmp/twophase-julia110-release.json"
    isfile(receipt) && cp(receipt, joinpath(destination, basename(receipt)); force=true)
end

function stage2_env(depot)
    env = Dict("OPENBLAS_NUM_THREADS" => "1", "MKL_NUM_THREADS" => "1",
               "JULIA_NUM_GC_THREADS" => "1")
    if !isnothing(depot)
        env["JULIA_DEPOT_PATH"] = "$depot:$(joinpath(homedir(), ".julia"))"
    end
    return env
end

function stage2_julia_path(value)
    candidate = isnothing(value) ? Sys.which("julia") :
        (isfile(value) ? abspath(value) : Sys.which(value))
    isnothing(candidate) && error("Julia executable unavailable; pass --julia")
    return string(candidate)
end

function stage2_ratio_ci(numerators, denominators, seed; draws=20_000)
    rng = MersenneTwister(seed)
    estimates = Vector{Float64}(undef, draws)
    for index in eachindex(estimates)
        estimates[index] = median(rand(rng, numerators, length(numerators))) /
                           median(rand(rng, denominators, length(denominators)))
    end
    sort!(estimates)
    return [estimates[floor(Int, fraction * (draws - 1)) + 1]
            for fraction in (0.025, 0.975)]
end
