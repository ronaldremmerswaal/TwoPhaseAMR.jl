#!/usr/bin/env julia
module BlockInteriorSetupStudy

include(joinpath(@__DIR__, "study.jl"))
using .BlockInteriorStudy
using TOML
const Support = BlockInteriorStudy.ScalingSupport

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: setup_study.jl COHORT_DIRECTORY OUTPUT_DIRECTORY")
        return
    end
    length(args) == 2 || error("usage: setup_study.jl COHORT_DIRECTORY OUTPUT_DIRECTORY")
    cohort,output = abspath.(args)
    config = Support.readjson(joinpath(cohort,"configuration.json"))
    pins = Support.readjson(joinpath(cohort,"placement.json"))["1"]
    source = Dict("baseline" => BlockInteriorStudy.BASELINE,
                  "current" => joinpath(cohort,"current-source"))
    mkpath(output)
    rows = Any[]
    for (n,block) in config["cases"], variant in ("baseline","current")
        root = source[variant]
        report_path = joinpath(output,"setup-$variant-n$n-b$block.toml")
        BlockInteriorStudy.launch(output,BlockInteriorStudy.DRIVER,1,pins,
            ["source=$root", "mode=setup", "kernel=upwind",
             "n=$n", "block=$block", "report=$report_path"])
        row = TOML.parsefile(report_path)
        row["n"] == n && row["block"] == block && row["source"] == root &&
            row["mode"] == "setup" || error("setup identity differs")
        row["variant"] = variant
        row["source_sha256"] = Support.sha256file(joinpath(root,"src","operators",
                                                          "advection","cell_advection.jl"))
        push!(rows,row)
        Support.writejson(joinpath(output,"setup-evidence.json"),rows)
        println("$variant n$n b$block: total ",
                round(row["total_summary_bytes"]/2^20;digits=1)," MiB")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
