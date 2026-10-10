#!/usr/bin/env julia
"""Copy the frozen narrow-ID source and replace only the ordered-sum helper."""
module OrderedGatherProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const EXPECTED_CLEAN_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const START_ANCHOR = "@inline function _cell_advection_ordered_sum("
const END_ANCHOR = "\n@inline function _cell_advection_regular_sum("

# The fallback retains the original packed-order body for every other code.
# The fast path selects already signed terms and adds them in exactly the same
# order, beginning with zero(T), for both advection and promoted CFL sums.
const CANDIDATE_HELPER = raw"""
@inline function _cell_advection_ordered_sum_generic(a, b, c, d, order::UInt8, ::Type{T}) where {T}
    fluxes = (a, b, c, d)
    integrated = zero(T)
    integrated += fluxes[Int(order & 0x03) + 1]
    integrated += fluxes[Int((order >> 2) & 0x03) + 1]
    integrated += fluxes[Int((order >> 4) & 0x03) + 1]
    integrated += fluxes[Int(order >> 6) + 1]
    return integrated
end

@inline function _cell_advection_ordered_sum(a, b, c, d, order::UInt8, ::Type{T}) where {T}
    if order == 0xe4 || order == 0xe1 || order == 0xb4 || order == 0xb1
        x_reverse = !iszero(order & 0x01)
        y_reverse = !iszero(order & 0x10)
        first_x = ifelse(x_reverse, b, a)
        second_x = ifelse(x_reverse, a, b)
        first_y = ifelse(y_reverse, d, c)
        second_y = ifelse(y_reverse, c, d)
        integrated = zero(T)
        integrated += first_x
        integrated += second_x
        integrated += first_y
        integrated += second_y
        return integrated
    end
    return _cell_advection_ordered_sum_generic(a, b, c, d, order, T)
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
    original = read(clean, String)
    candidate = candidate_source(original)
    candidate != original || error("ordered-sum transformation made no change")
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".paired-order", clean, target])
    diff.exitcode == 1 || error("expected exactly one copied-source diff")
    patch = joinpath(output, "ordered-gather.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_source_sha256" => sha256file(clean),
        "candidate_source_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "candidate_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "copied ordered four-face sum only; every AMR leaf",
        "fallback" => "original packed-order helper for all noncanonical codes")
    writejson(joinpath(output, "ordered-gather-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_probe.jl --source FROZEN_NARROW_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("usage: --source SOURCE --output NEW_DIRECTORY")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied ordered-gather probe: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
