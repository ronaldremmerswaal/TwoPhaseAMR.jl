#!/usr/bin/env julia
"""Make a copied row-fused rolling-flux probe from the frozen two-pass probe."""
module RollingFluxFusedProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_PARENT_SHA256 =
    "9383d5b7800300353b2f054efac51b14be48f16aeaabf3eec886f4dc1eeb85b6"

function replace_once(source, original, replacement)
    anchor = findfirst(original, source)
    anchor === nothing && error("source anchor missing: $original")
    findnext(original, source, last(anchor) + 1) === nothing ||
        error("source anchor repeated: $original")
    return replace(source, original => replacement; count=1)
end

function candidate_source(parent)
    result = replace_once(parent,
        "function _cell_advection_rolling_increments!(",
        "function _cell_advection_rolling_rows!(")
    result = replace_once(result,
        "        y_area = @inbounds workspace.y_areas[b]\n" *
        "        cell_offset = (b - 1) * cells_per_block\n",
        "        y_area = @inbounds workspace.y_areas[b]\n" *
        "        volume = mesh.blocks[b].volume\n" *
        "        cell_offset = (b - 1) * cells_per_block\n")
    result = replace_once(result,
        "                # Keep the T-valued integrated sum until every flux has read\n" *
        "                # the unchanged scalar. Division can promote beyond T.\n" *
        "                increment[i + h, j + h] = integrated\n",
        "                # Every flux reading this row has been computed before the\n" *
        "                # first scalar write. Preserve promoted change arithmetic.\n" *
        "                change = integrated / volume\n" *
        "                increment[i + h, j + h] = change\n" *
        "                values[i + h, j + h] += change\n")
    apply_function = findfirst(
        "function _cell_advection_apply_increments!(scalar::CellField, workspace::AdvectionWorkspace)\n",
        result)
    apply_function === nothing && error("separate update pass anchor missing")
    next_function = findnext(
        "function _cell_advection_regular_rates!(velocity::FaceVelocity, workspace::AdvectionWorkspace)\n",
        result, last(apply_function) + 1)
    next_function === nothing && error("regular rates anchor missing")
    before = result[firstindex(result):prevind(result, first(apply_function))]
    after = result[first(next_function):lastindex(result)]
    result = before * after
    result = replace_once(result,
        "    _cell_advection_rolling_increments!(scalar, velocity, dt, workspace)\n" *
        "    _cell_advection_apply_increments!(scalar, workspace)\n",
        "    _cell_advection_rolling_rows!(scalar, velocity, dt, workspace)\n")
    return result
end

function make_probe(parent, output)
    parent, output = abspath(parent), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    source = joinpath(parent, RELATIVE)
    receipt_file = joinpath(parent, "rolling-flux-receipt.json")
    receipt = readjson(receipt_file)
    receipt["output"] == parent &&
        receipt["candidate_source_sha256"] == EXPECTED_PARENT_SHA256 &&
        sha256file(source) == EXPECTED_PARENT_SHA256 &&
        receipt["candidate_tree_sha256"] == source_hash(parent) ||
        error("parent rolling-flux source identity differs")
    control = receipt["source"]
    control_source = joinpath(control, RELATIVE)
    sha256file(control_source) == receipt["clean_source_sha256"] &&
        source_hash(control) == receipt["clean_tree_sha256"] ||
        error("original control source identity differs")
    fused = candidate_source(read(source, String))
    mkpath(output)
    cp(joinpath(parent, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, fused)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".row-fused", source, target])
    diff.exitcode == 1 || error("expected copied-source patch")
    patch = joinpath(output, "rolling-flux-fused.patch")
    write(patch, diff.stdout)
    result = Dict{String,Any}(
        "schema" => 1,
        "source" => control,
        "clean_source_sha256" => sha256file(control_source),
        "clean_tree_sha256" => source_hash(control),
        "parent" => parent,
        "output" => output,
        "parent_receipt_sha256" => sha256file(receipt_file),
        "parent_source_sha256" => sha256file(source),
        "parent_tree_sha256" => source_hash(parent),
        "candidate_source_sha256" => sha256file(target),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "ordinary general-AMR upwind row-fused rolling advection; cut-cell/CFL unchanged",
    )
    writejson(joinpath(output, "rolling-flux-fused-receipt.json"), result)
    return result
end

function main(args=ARGS)
    length(args) == 2 || error("usage: make_probe.jl FROZEN_ROLLING_SOURCE NEW_OUTPUT")
    result = make_probe(args[1], args[2])
    println("created copied row-fused probe: ", result["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
