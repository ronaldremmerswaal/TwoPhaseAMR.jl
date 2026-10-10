#!/usr/bin/env julia
"""Make a copied-source all-order scalar-selection cell-advection probe."""
module ScalarSelectorProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_CLEAN_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const START_ANCHOR = "@inline function _cell_advection_ordered_sum("
const END_ANCHOR = "\n@inline function _cell_advection_regular_sum("

# The two low bits select a/b/c/d in every packed position. `ifelse` only
# selects already computed scalar fluxes; all four sums keep their original
# left-to-right order and promoted zero(T) accumulator.
const CANDIDATE_HELPER = raw"""
@inline function _cell_advection_select_flux(a, b, c, d, direction::UInt8)
    low = ifelse(iszero(direction & 0x01), a, b)
    high = ifelse(iszero(direction & 0x01), c, d)
    return ifelse(iszero(direction & 0x02), low, high)
end

@inline function _cell_advection_ordered_sum(a, b, c, d, order::UInt8, ::Type{T}) where {T}
    integrated = zero(T)
    integrated += _cell_advection_select_flux(a, b, c, d, order & 0x03)
    integrated += _cell_advection_select_flux(a, b, c, d, (order >> 2) & 0x03)
    integrated += _cell_advection_select_flux(a, b, c, d, (order >> 4) & 0x03)
    integrated += _cell_advection_select_flux(a, b, c, d, order >> 6)
    return integrated
end
"""

function candidate_source(original)
    first_range = findfirst(START_ANCHOR, original)
    first_range === nothing && error("ordered-sum helper anchor missing")
    last_range = findnext(END_ANCHOR, original, last(first_range) + 1)
    last_range === nothing && error("regular-sum helper anchor missing")
    length(split(original, START_ANCHOR)) == 2 ||
        error("ordered-sum helper anchor is not unique")
    before = original[firstindex(original):prevind(original, first(first_range))]
    after = original[first(last_range):lastindex(original)]
    return before * CANDIDATE_HELPER * after
end

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) && sha256file(clean) == EXPECTED_CLEAN_SHA256 ||
        error("source is not the frozen narrow-ID implementation")
    candidate = candidate_source(read(clean, String))
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".all-order-selector", clean, target])
    diff.exitcode == 1 || error("expected exactly one copied-source diff")
    patch = joinpath(output, "scalar-selector.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_source_sha256" => sha256file(clean),
        "candidate_source_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "copied ordered four-face sum on every AMR leaf",
        "selection" => "balanced scalar ifelse for all 256 packed UInt8 codes",
    )
    writejson(joinpath(output, "scalar-selector-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_probe.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("usage: --source SOURCE --output NEW_DIRECTORY")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied scalar-selector probe: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
