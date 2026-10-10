#!/usr/bin/env julia
module NarrowIDStorageStudy

include(joinpath(@__DIR__, "study.jl"))
using .BlockInteriorStudy
using TOML
const Support = BlockInteriorStudy.ScalingSupport

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: id_storage_study.jl COHORT_DIRECTORY NEW_OUTPUT_DIRECTORY")
        return
    end
    length(args) == 2 || error("usage: id_storage_study.jl COHORT_DIRECTORY NEW_OUTPUT_DIRECTORY")
    cohort, output = abspath.(args)
    !ispath(output) || error("output exists: $output")
    config = Support.readjson(joinpath(cohort, "configuration.json"))
    config["phase"] == "matrix" || error("expected accepted matrix")
    BlockInteriorStudy.verify_control()
    sources = Dict("baseline" => BlockInteriorStudy.BASELINE,
                   "current" => joinpath(cohort, "current-source"))
    candidate_sha = Support.sha256file(joinpath(sources["current"], "src", "operators",
                                                   "advection", "cell_advection.jl"))
    candidate_sha == config["candidate_advection_sha256"] || error("candidate source differs")
    pins = Support.readjson(joinpath(cohort, "placement.json"))["1"]
    mkpath(output)
    rows = Any[]
    driver = joinpath(@__DIR__, "id_storage_driver.jl")
    for (n, block) in config["cases"], variant in ("baseline", "current")
        source = sources[variant]
        stem = "$variant-n$n-b$block"
        report_path = joinpath(output, "$stem.toml")
        BlockInteriorStudy.launch(output, driver, 1, pins,
            ["source=$source", "mode=setup", "kernel=upwind", "n=$n",
             "block=$block", "report=$report_path"])
        row = TOML.parsefile(report_path)
        row["mode"] == "setup" && row["source"] == source &&
            row["n"] == n && row["block"] == block || error("setup identity differs: $stem")
        expected_length = (n ÷ block)^2 * (block - 1) * block
        row["x_face_id_length"] == row["y_face_id_length"] == expected_length ||
            error("face ID lengths differ: $stem")
        expected_type = variant == "baseline" ? string(Int) : string(Int32)
        row["x_face_id_eltype"] == row["y_face_id_eltype"] == expected_type ||
            error("face ID element type differs: $stem")
        width = variant == "baseline" ? sizeof(Int) : sizeof(Int32)
        row["x_face_id_payload_bytes"] == row["y_face_id_payload_bytes"] ==
            width * expected_length || error("face ID payload differs: $stem")
        row["variant"] = variant
        row["source_sha256"] = Support.sha256file(joinpath(source, "src", "operators",
                                                          "advection", "cell_advection.jl"))
        row["driver_sha256"] = Support.sha256file(driver)
        row["control_revision"] = config["control_revision"]
        push!(rows, row)
        Support.writejson(joinpath(output, "id-storage-evidence.json"), rows)
        println("$stem: ", expected_type, ", ID payload ",
                row["x_face_id_payload_bytes"] + row["y_face_id_payload_bytes"],
                " B, deduplicated graph ", row["total_summary_bytes"], " B")
        flush(stdout)
    end
    length(rows) == 2length(config["cases"]) || error("incomplete storage study")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
