"""Record four general-upwind setup and resident-size diagnostics."""
module Stage3SetupStudy

include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: writejson
using TOML

const Study = MainStudy

function main(args=ARGS)
    length(args) == 1 || error("usage: julia setup_study.jl OUTPUT")
    output = abspath(args[1])
    mkpath(output)
    ENV["STAGE3_FAILURE_DIR"] = output
    log = Any[]
    rows = Dict{String,Any}[]
    for (variant, source) in (("frozen", Study.FROZEN_SOURCE),
                              ("current", Study.ROOT)), block in (1024,8)
        report_path = joinpath(output, "$variant-1024-b$block.toml")
        Study.run_julia(Study.DRIVER, Study.FROZEN_PROJECT, 1,
            String["source=$source", "mode=setup", "kernel=upwind",
                   "n=1024", "block=$block", "report=$report_path"], [0], log)
        row = TOML.parsefile(report_path)
        row["n"] == 1024 && row["block"] == block && row["mode"] == "setup" ||
            error("setup report identity differs")
        row["variant"] = variant
        row["advection_source_sha256"] = Study.digest(joinpath(source,
            "src", "operators", "advection", "cell_advection.jl"))
        push!(rows, row)
        writejson(joinpath(output, "setup-evidence.json"), rows)
        total_mib = row["total_summary_bytes"] / 2^20
        println("$variant b$block: total $(round(total_mib; digits=1)) MiB, " *
                "mesh $(round(row["mesh_setup_seconds"]; digits=2))s, " *
                "workspace $(round(row["workspace_setup_seconds"]; digits=2))s")
    end
    length(rows) == 4 || error("setup study incomplete")
    println("validated four setup/resident-size observations")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3SetupStudy.main()
