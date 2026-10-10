#!/usr/bin/env julia
"""Copy frozen mesh source and stop after a full RawSide backing-array scan."""
module MeshCrashRawCheckpoint

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: sha256file, source_hash, run_capture, writejson

const RELATIVE = joinpath("src", "mesh", "topology", "mesh.jl")
const CLEAN_SHA256 =
    "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"
const ANCHOR = """
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
"""
const INJECTED = """
    _raw_checkpoint!(vertical, horizontal, nx, ny, length(blocks))
    error("RawSide checkpoint stopped before face matching")
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
"""
const HELPERS = raw"""

mutable struct _RawCheckpointStats
    counts::Vector{Int}
    invalid::Int
    first_invalid::Vector{Any}
    invalid_by_line::Dict{String,Int}
    fnv64::UInt64
    sum64::UInt64
end

_RawCheckpointStats() = _RawCheckpointStats(zeros(Int,4), 0, Any[],
    Dict{String,Int}(), UInt64(0xcbf29ce484222325), zero(UInt64))

function _scan_raw_table!(stats, table, axis, nx, ny, block_count)
    for coordinate in sort!(collect(keys(table)))
        lower, upper = table[coordinate]
        # _insert_side! stores positive faces in `lower`, negative in `upper`.
        for (polarity, sides) in ((1,lower),(-1,upper))
            stage = (axis-1)*2 + (polarity == -1 ? 1 : 2)
            backing = reinterpret(UInt64, sides)
            length(backing) == 5length(sides) ||
                error("RawSide backing-word count mismatch")
            for (index, raw) in pairs(sides)
                stats.counts[stage] += 1
                cell = raw.cell
                tangent = axis == 1 ? cell.j : cell.i
                normal = axis == 1 ? cell.i : cell.j
                period = axis == 1 ? nx : ny
                expected_key = mod(normal - (polarity == -1 ? 1 : 0), period)
                valid = block_count == 1 && cell.block == 1 &&
                    1 <= cell.i <= nx && 1 <= cell.j <= ny &&
                    raw.lo == tangent-1 && raw.hi == tangent &&
                    coordinate == expected_key
                word_index = 5(index-1)
                for offset in 1:5
                    word = backing[word_index+offset]
                    stats.fnv64 = (stats.fnv64 ⊻ word) * UInt64(0x100000001b3)
                    stats.sum64 += word
                end
                if !valid
                    stats.invalid += 1
                    line = "$(axis):$(coordinate)"
                    stats.invalid_by_line[line] =
                        get(stats.invalid_by_line, line, 0) + 1
                    if length(stats.first_invalid) < 10
                        push!(stats.first_invalid, Dict(
                            "axis" => axis, "coordinate" => coordinate,
                            "polarity" => polarity, "vector_index" => index,
                            "lo" => raw.lo, "hi" => raw.hi,
                            "cell" => [cell.block, cell.i, cell.j],
                            "backing_words_hex" => [
                                string(backing[word_index+offset]; base=16, pad=16)
                                for offset in 1:5]))
                    end
                end
            end
        end
    end
    return nothing
end

function _raw_checkpoint!(vertical, horizontal, nx, ny, block_count)
    isbitstype(RawSide) && sizeof(RawSide) == 5sizeof(UInt64) &&
        fieldoffset(RawSide,1) == 0 && fieldoffset(RawSide,2) == 8 &&
        fieldoffset(RawSide,3) == 16 &&
        isbitstype(CellRef) && sizeof(CellRef) == 3sizeof(UInt64) &&
        fieldoffset(CellRef,1) == 0 && fieldoffset(CellRef,2) == 8 &&
        fieldoffset(CellRef,3) == 16 || error("unexpected RawSide/CellRef layout")
    stats = _RawCheckpointStats()
    _scan_raw_table!(stats, vertical, 1, nx, ny, block_count)
    _scan_raw_table!(stats, horizontal, 2, nx, ny, block_count)
    expected = nx*ny
    report = Dict{String,Any}(
        "schema" => 1, "fixture" => "periodic one-block 2048x2048",
        "nx" => nx, "ny" => ny, "block_count" => block_count,
        "counts" => stats.counts, "expected_count_each" => expected,
        "population_ok" => all(count -> count == expected, stats.counts),
        "invalid_count" => stats.invalid,
        "first_invalid" => stats.first_invalid,
        "invalid_by_line" => stats.invalid_by_line,
        "backing_fnv64_hex" => string(stats.fnv64; base=16, pad=16),
        "backing_sum64_hex" => string(stats.sum64; base=16, pad=16),
        "rawside_size" => sizeof(RawSide),
        "cellref_size" => sizeof(CellRef),
        "rawside_field_offsets" => [fieldoffset(RawSide,i) for i in 1:3],
        "cellref_field_offsets" => [fieldoffset(CellRef,i) for i in 1:3])
    path = ENV["TWOPHASE_RAW_CHECKPOINT_OUTPUT"]
    open(path, "w") do io
        Main.TOML.print(io, report)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("RawSide checkpoint fsync failed")
    end
    return nothing
end
"""

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("checkpoint output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) && sha256file(clean) == CLEAN_SHA256 ||
        error("mesh source differs from frozen control")
    original = read(clean, String)
    length(split(original, ANCHOR)) == 2 ||
        error("RawSide checkpoint anchor missing or repeated")
    candidate = replace(original, ANCHOR => INJECTED; count=1) * HELPERS
    mkpath(output)
    cp(joinpath(source,"src"),joinpath(output,"src"))
    target = joinpath(output,RELATIVE)
    write(target,candidate)
    diff = run_capture(["diff","-u","--label",RELATIVE,
        "--label",RELATIVE*".raw-checkpoint",clean,target])
    diff.exitcode == 1 || error("expected copied-source patch")
    patch = joinpath(output,"raw-checkpoint.patch")
    write(patch,diff.stdout)
    receipt = Dict{String,Any}(
        "schema"=>1,"source"=>source,"output"=>output,
        "clean_mesh_sha256"=>sha256file(clean),
        "checkpoint_mesh_sha256"=>sha256file(target),
        "clean_tree_sha256"=>source_hash(source),
        "checkpoint_tree_sha256"=>source_hash(output),
        "generator_sha256"=>sha256file(@__FILE__),
        "patch_sha256"=>sha256file(patch),
        "scope"=>"full RawSide backing-array scan after filling tables; stop before face matching")
    writejson(joinpath(output,"raw-checkpoint-receipt.json"),receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: make_raw_checkpoint.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args)==4 || error("--source and --output required")
    options=Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    Set(keys(options))==Set(("--source","--output")) ||
        error("only --source and --output are accepted")
    receipt=make_probe(options["--source"],options["--output"])
    println("created RawSide checkpoint: ",receipt["output"])
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

end # module
