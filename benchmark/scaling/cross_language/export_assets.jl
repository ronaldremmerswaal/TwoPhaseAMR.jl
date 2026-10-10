"""Copy auditable Julia/Fortran comparison evidence into documentation assets."""
module CrossLanguageAssets

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: sha256file, writejson

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_RESULT = joinpath(ROOT, "benchmark", "scaling", "results",
                                "20260930-julia-fortran-matched")
const DEFAULT_ASSETS = joinpath(ROOT, "docs", "src", "assets", "julia-fortran-matched")

function export_assets(result, assets)
    mkpath(assets)
    names = String["raw.jsonl", "configuration.json", "workload.json", "fixtures.json",
                   "source_sha256.json", "job_order.json", "pilots.jsonl",
                   "summary.json", "comparison.csv"]
    append!(names, ["$kind-$n.$extension" for n in (512,1024,2048,4096)
                    for kind in ("comparison", "speedup")
                    for extension in ("svg", "png")])
    append!(names, [joinpath("fortran-validation", name) for name in
                    ("convergence.json", "debug-oracle-receipt.json", "debug-oracle.log")])
    relative = startswith(result, ROOT * "/") ? relpath(result, ROOT) : result
    manifest = Dict{String,Any}("source_result"=>relative,
                                "sha256"=>Dict{String,String}())
    for name in names
        source, target = joinpath(result, name), joinpath(assets, name)
        isfile(source) || error("missing expected asset: $source")
        mkpath(dirname(target))
        cp(source, target; force=true)
        manifest["sha256"][name] = sha256file(target)
    end
    writejson(joinpath(assets, "asset-manifest.json"), manifest)
    return length(names)
end

function main(args=ARGS)
    result, assets = DEFAULT_RESULT, DEFAULT_ASSETS
    iseven(length(args)) || error("usage: julia export_assets.jl [--result DIR] [--assets DIR]")
    for i in 1:2:length(args)
        if args[i] == "--result"
            result = abspath(args[i+1])
        elseif args[i] == "--assets"
            assets = abspath(args[i+1])
        else
            error("unknown option $(args[i])")
        end
    end
    count = export_assets(result, assets)
    println("exported $count files to $assets")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CrossLanguageAssets.main()
