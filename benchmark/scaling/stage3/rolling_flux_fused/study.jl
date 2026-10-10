#!/usr/bin/env julia
"""Check the row-fused parent chain, then run the frozen copied-source study."""
module RollingFluxFusedStudy

include(joinpath(@__DIR__, "..", "copied_source", "study.jl"))
using .CopiedSourceStudy

const Generic = CopiedSourceStudy
const Support = Generic.Support
const PARENT_SHA256 =
    "9383d5b7800300353b2f054efac51b14be48f16aeaabf3eec886f4dc1eeb85b6"

function check_parent(options)
    options.label == "rolling_flux_fused" || error("expected row-fused candidate label")
    receipt = Support.readjson(options.receipt_path)
    parent = receipt["parent"]
    parent_receipt_file = joinpath(parent, "rolling-flux-receipt.json")
    isdir(parent) && isfile(parent_receipt_file) || error("parent probe missing")
    parent_receipt = Support.readjson(parent_receipt_file)
    parent_source = joinpath(parent, Generic.RELATIVE)
    receipt["parent_source_sha256"] == PARENT_SHA256 &&
        Support.sha256file(parent_source) == PARENT_SHA256 &&
        receipt["parent_tree_sha256"] == Support.source_hash(parent) &&
        receipt["parent_receipt_sha256"] ==
            Support.sha256file(parent_receipt_file) &&
        parent_receipt["output"] == parent &&
        parent_receipt["candidate_source_sha256"] == PARENT_SHA256 &&
        parent_receipt["candidate_tree_sha256"] == receipt["parent_tree_sha256"] &&
        parent_receipt["source"] == receipt["source"] &&
        parent_receipt["clean_source_sha256"] == receipt["clean_source_sha256"] &&
        parent_receipt["clean_tree_sha256"] == receipt["clean_tree_sha256"] ||
        error("row-fused parent identity differs")
    return Dict(
        "schema" => 1,
        "candidate_receipt_sha256" => Support.sha256file(options.receipt_path),
        "parent_source_sha256" => PARENT_SHA256,
        "parent_tree_sha256" => receipt["parent_tree_sha256"],
        "parent_receipt_sha256" => receipt["parent_receipt_sha256"],
        "wrapper_sha256" => Support.sha256file(@__FILE__),
        "generic_study_sha256" =>
            Support.sha256file(joinpath(@__DIR__, "..", "copied_source", "study.jl")))
end

function main()
    options = Generic.options(ARGS)
    before = check_parent(options)
    Generic.main()
    after = check_parent(options)
    before == after || error("row-fused parent changed during study")
    Support.writejson(joinpath(options.output, "parent-verification.json"), after)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
