#!/usr/bin/env julia
"""Make an untimed instrumented copy of the frozen row-fused candidate."""
module RollingFluxFusedCountProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file, source_hash, run_capture

const RELATIVE = joinpath("src", "operators", "advection", "cell_advection.jl")
const ORIGINAL = "@inline _cell_advection_regular_flux(dt, speed, state, area, face_id, ::Nothing) =\n    dt * speed * state * area"
const INSTRUMENTED = """const _rolling_flux_face_calls = Ref{Vector{Threads.Atomic{Int}}}()

@inline function _cell_advection_regular_flux(dt, speed, state, area, face_id, ::Nothing)
    Threads.atomic_add!(_rolling_flux_face_calls[][face_id], 1)
    return dt * speed * state * area
end"""

function make_probe(candidate, output)
    candidate, output = abspath(candidate), abspath(output)
    !ispath(output) || error("instrumented output already exists: $output")
    source = joinpath(candidate, RELATIVE)
    receipt_file = joinpath(candidate, "rolling-flux-fused-receipt.json")
    receipt = readjson(receipt_file)
    receipt["output"] == candidate &&
        receipt["candidate_source_sha256"] == sha256file(source) &&
        receipt["candidate_tree_sha256"] == source_hash(candidate) ||
        error("frozen row-fused candidate identity differs")
    original = read(source, String)
    length(split(original, ORIGINAL)) == 2 ||
        error("regular-flux helper anchor must occur exactly once")
    mkpath(output)
    cp(joinpath(candidate, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, replace(original, ORIGINAL => INSTRUMENTED; count=1))
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".call-count", source, target])
    diff.exitcode == 1 || error("expected exactly one instrumentation diff")
    patch = joinpath(output, "regular-flux-call-count.patch")
    write(patch, diff.stdout)
    count_receipt = Dict{String,Any}(
        "schema" => 1,
        "candidate" => candidate,
        "output" => output,
        "candidate_receipt_sha256" => sha256file(receipt_file),
        "candidate_source_sha256" => sha256file(source),
        "candidate_tree_sha256" => source_hash(candidate),
        "instrumented_source_sha256" => sha256file(target),
        "instrumented_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "untimed row-fused plain-advection regular-face helper call count",
    )
    writejson(joinpath(output, "regular-flux-call-count-receipt.json"), count_receipt)
    return count_receipt
end

function main(args=ARGS)
    length(args) == 2 || error("usage: make_count_probe.jl FUSED_CANDIDATE NEW_OUTPUT")
    result = make_probe(args[1], args[2])
    println("instrumented row-fused copied source: ", result["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
