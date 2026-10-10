#!/usr/bin/env julia
"""Copy a frozen source and add setup-only assertions around its unchanged sort."""
module WorkspaceSetupProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const THREADING = joinpath("src", "threading.jl")
const ADVECTION = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_THREADING_SHA256 =
    "811d1d79b1843d1faa21abdcfa5cc8cb55b969b10bec99a6a6c124f8f458abc4"
const EXPECTED_ADVECTION_SHA256 = Dict(
    "control" => "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541",
    "rolling_flux_fused" => "d3d8edeb29cef090ca0f4168ba9103f44346425c1d55be40f80905f4e2e21027",
)

function replace_once(source, original, replacement)
    anchor = findfirst(original, source)
    anchor === nothing && error("source anchor missing: $original")
    findnext(original, source, last(anchor) + 1) === nothing ||
        error("source anchor repeated: $original")
    return replace(source, original => replacement; count=1)
end

const SCHEDULE_CHECKS = raw"""
# Setup-only assertions. Activated by a copied-source validation driver for
# the frozen periodic 1024² mesh with 8×8 leaves; no sort algorithm is changed.
const _stage3_setup_schedule_checks = Ref(false)

function _stage3_check_targets_before_sort(targets, outputs)
    outputs == 1_048_576 || error("unexpected number of owned cells: $outputs")
    length(targets) == 1_835_008 ||
        error("unexpected edge incidence count: $(length(targets))")
    for (index, target) in pairs(targets)
        1 <= target <= outputs ||
            error("target $index lies outside owned cells: $target")
    end
    println("setup_check=targets_valid outputs=$outputs entries=$(length(targets))")
    flush(stdout)
    return nothing
end

function _stage3_check_permutation_after_sort(permutation, entries)
    length(permutation) == entries || error("sort permutation length differs")
    seen = falses(entries)
    for (position, index) in pairs(permutation)
        1 <= index <= entries || error("sort permutation index $position differs: $index")
        !seen[index] || error("sort permutation repeats index $index")
        seen[index] = true
    end
    all(seen) || error("sort permutation omits a contribution")
    println("setup_check=permutation_valid entries=$entries")
    flush(stdout)
    return nothing
end

function _stage3_check_groups_after_sort(targets, offsets, permutation, outputs)
    length(offsets) == outputs + 1 || error("group offset length differs")
    offsets[1] == 1 || error("first group offset differs")
    offsets[end] == length(targets) + 1 || error("last group offset differs")
    for output in 1:outputs
        first = offsets[output]
        stop = offsets[output + 1]
        1 <= first <= stop <= length(targets) + 1 ||
            error("invalid group offsets for output $output")
        prior = 0
        for position in first:(stop - 1)
            original_index = permutation[position]
            targets[original_index] == output ||
                error("wrong target in output group $output at $position")
            original_index > prior ||
                error("original incidence order changed in output group $output")
            prior = original_index
        end
    end
    println("setup_check=groups_valid outputs=$outputs")
    flush(stdout)
    return nothing
end

"""

function checked_threading(original)
    result = replace_once(original,
        "\"\"\"Group contribution indices by a one-based output index.\"\"\"\n",
        SCHEDULE_CHECKS * "\"\"\"Group contribution indices by a one-based output index.\"\"\"\n")
    result = replace_once(result,
        "    permutation = sortperm(targets)\n",
        "    _stage3_setup_schedule_checks[] &&\n" *
        "        _stage3_check_targets_before_sort(targets, outputs)\n" *
        "    permutation = sortperm(targets)\n" *
        "    _stage3_setup_schedule_checks[] &&\n" *
        "        _stage3_check_permutation_after_sort(permutation, length(targets))\n")
    result = replace_once(result,
        "    return offsets, permutation\nend\n",
        "    _stage3_setup_schedule_checks[] &&\n" *
        "        _stage3_check_groups_after_sort(targets, offsets, permutation, outputs)\n" *
        "    return offsets, permutation\nend\n")
    return result
end

function checked_advection(original)
    return replace_once(original,
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n",
        "    length(targets) == length(face_indices) == length(face_signs) ==\n" *
        "        length(face_kinds) || error(\"edge incidence vector lengths differ\")\n" *
        "    offsets, permutation = _grouped_output_schedule(targets, ncells(mesh))\n")
end

function make_probe(variant, source, output)
    haskey(EXPECTED_ADVECTION_SHA256, variant) || error("unknown variant: $variant")
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    sha256file(joinpath(source, THREADING)) == EXPECTED_THREADING_SHA256 ||
        error("frozen threading source differs")
    sha256file(joinpath(source, ADVECTION)) == EXPECTED_ADVECTION_SHA256[variant] ||
        error("frozen advection source differs")
    threaded = checked_threading(read(joinpath(source, THREADING), String))
    advected = checked_advection(read(joinpath(source, ADVECTION), String))
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    write(joinpath(output, THREADING), threaded)
    write(joinpath(output, ADVECTION), advected)
    patches = String[]
    for relative in (THREADING, ADVECTION)
        diff = run_capture(["diff", "-u", "--label", relative,
                            "--label", relative * ".setup-check",
                            joinpath(source, relative), joinpath(output, relative)])
        diff.exitcode == 1 || error("expected setup-only patch for $relative")
        push!(patches, diff.stdout)
    end
    patch = joinpath(output, "setup-check.patch")
    write(patch, join(patches, "\n"))
    receipt = Dict{String,Any}(
        "schema" => 1,
        "variant" => variant,
        "source" => source,
        "output" => output,
        "original_threading_sha256" => EXPECTED_THREADING_SHA256,
        "original_advection_sha256" => EXPECTED_ADVECTION_SHA256[variant],
        "original_tree_sha256" => source_hash(source),
        "instrumented_threading_sha256" => sha256file(joinpath(output, THREADING)),
        "instrumented_advection_sha256" => sha256file(joinpath(output, ADVECTION)),
        "instrumented_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "setup-only assertions; sort and advection algorithms unchanged",
    )
    writejson(joinpath(output, "setup-check-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    length(args) == 3 || error("usage: make_probe.jl VARIANT FROZEN_SOURCE NEW_OUTPUT")
    receipt = make_probe(args[1], args[2], args[3])
    println("created setup-only copied source: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
