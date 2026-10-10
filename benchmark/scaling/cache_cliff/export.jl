"""Export cache-cliff evidence and figures to documentation assets."""
module CacheCliffAssets

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: sha256file, writejson

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_RESULT = joinpath(ROOT, "benchmark", "scaling", "results",
                                "20260930-cache-cliff")
const DEFAULT_ASSETS = joinpath(ROOT, "docs", "src", "assets", "cache-cliff")
const CORE = ("configuration.json", "source_sha256.json", "workload.json",
              "fixtures.json", "job_order.json", "raw.jsonl", "summary.json", "summary.csv")

function export_assets(result, assets)
    manifest = Dict{String,Any}("source_result"=>result,
                                "sha256"=>Dict{String,String}())
    for phase in ("ladder", "placement", "placement-low", "steps", "pmu", "padding")
        folder = joinpath(result, phase)
        isdir(folder) || continue
        names = [name for name in CORE if isfile(joinpath(folder, name))]
        append!(names, [name for name in readdir(folder)
                        if endswith(name, ".svg") || endswith(name, ".png")])
        isfile(joinpath(folder, "pilots.jsonl")) && push!(names, "pilots.jsonl")
        for name in names
            source, target = joinpath(folder, name), joinpath(assets, phase, name)
            mkpath(dirname(target))
            cp(source, target; force=true)
            manifest["sha256"]["$phase/$name"] = sha256file(target)
        end
    end
    preflight = joinpath(result, "fortran-diagnostics", "preflight.json")
    if isfile(preflight)
        target = joinpath(assets, "fortran-diagnostics", "preflight.json")
        mkpath(dirname(target))
        cp(preflight, target; force=true)
        manifest["sha256"]["fortran-diagnostics/preflight.json"] = sha256file(target)
    end
    writejson(joinpath(assets, "asset-manifest.json"), manifest)
    return length(manifest["sha256"])
end

function main(args=ARGS)
    result, assets = DEFAULT_RESULT, DEFAULT_ASSETS
    iseven(length(args)) || error("usage: julia export.jl [--result DIR] [--assets DIR]")
    for i in 1:2:length(args)
        if args[i] == "--result"
            result = abspath(args[i+1])
        elseif args[i] == "--assets"
            assets = abspath(args[i+1])
        else
            error("unknown option $(args[i])")
        end
    end
    println("exported $(export_assets(result, assets)) files")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CacheCliffAssets.main()
