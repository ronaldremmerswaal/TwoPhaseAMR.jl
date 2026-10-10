#!/usr/bin/env julia
"""Instrument a copied frozen mesh source at RawSide-to-InteriorFace production."""
module MeshCrashFaceProducerProbe

include(joinpath(@__DIR__, "..", "..", "Support.jl"))
using .ScalingSupport: sha256file, source_hash, run_capture, writejson

const RELATIVE = joinpath("src", "mesh", "topology", "mesh.jl")
const CLEAN_SHA256 =
    "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"

function unique_replace(source, before, after)
    length(split(source, before)) == 2 || error("mesh probe anchor missing or repeated")
    return replace(source, before => after; count=1)
end

const BUILD_OLD = """
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(faces, boundaries, vertical, 1, origin, extent, (sx, sy))
    _match_sides!(faces, boundaries, horizontal, 2, origin, extent, (sx, sy))
    return faces, boundaries
end
"""
const BUILD_NEW = """
    probe = _FaceProducerProbe(nx, ny, length(blocks))
    _probe_raw_table!(probe, vertical, 1)
    _probe_raw_table!(probe, horizontal, 2)
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(faces, boundaries, vertical, 1, origin, extent, (sx, sy), probe)
    _match_sides!(faces, boundaries, horizontal, 2, origin, extent, (sx, sy), probe)
    _probe_final_faces!(probe, faces)
    _probe_report!(probe, length(faces), length(boundaries))
    error("face producer probe stopped before dual-volume construction")
end
"""
const MATCH_SIGNATURE_OLD =
    "function _match_sides!(faces, boundaries, table, axis, origin, extent, lattice_size)"
const MATCH_SIGNATURE_NEW =
    "function _match_sides!(faces, boundaries, table, axis, origin, extent, lattice_size, probe)"
const MATCH_PUSH_OLD =
    "                push!(faces, InteriorFace(lside.cell, uside.cell, axis, center, (hi - lo) * tangent_scale))"
const MATCH_PUSH_NEW = join((
    "                _probe_raw_pair!(probe, lside, uside, axis, coordinate, length(faces) + 1)",
    "                face = InteriorFace(lside.cell, uside.cell, axis, center, (hi - lo) * tangent_scale)",
    "                _probe_fresh_face!(probe, face, lside, uside, axis, coordinate, length(faces) + 1)",
    "                push!(faces, face)",
    "                _probe_pushed_face!(probe, faces[end], face, axis, coordinate, length(faces))",
), "\n")

# This code is appended to the copied module. Its tests are specific to the
# frozen one-block, periodic, 2048² fixture; production topology is untouched.
const INSTRUMENTATION = raw"""

mutable struct _FaceProducerProbe
    nx::Int
    ny::Int
    block_count::Int
    counts::Dict{String,Int}
    first::Dict{String,Vector{Any}}
    lines::Dict{String,Dict{String,Int}}
end

function _FaceProducerProbe(nx, ny, block_count)
    stages = ("raw_pre_match", "raw_at_match", "fresh", "after_push", "final")
    return _FaceProducerProbe(nx, ny, block_count,
        Dict(stage => 0 for stage in stages),
        Dict(stage => Any[] for stage in stages),
        Dict(stage => Dict{String,Int}() for stage in stages))
end

@inline function _probe_cell_valid(probe, cell)
    return cell.block == 1 && probe.block_count == 1 &&
           1 <= cell.i <= probe.nx && 1 <= cell.j <= probe.ny
end

@inline function _probe_raw_valid(probe, raw, axis)
    tangential = axis == 1 ? raw.cell.j : raw.cell.i
    return _probe_cell_valid(probe, raw.cell) &&
           raw.lo == tangential - 1 && raw.hi == tangential
end

@inline function _probe_same_cell(a, b)
    return a.block == b.block && a.i == b.i && a.j == b.j
end

@inline function _probe_same_face(a, b)
    return _probe_same_cell(a.minus, b.minus) &&
           _probe_same_cell(a.plus, b.plus) && a.axis == b.axis &&
           isequal(a.center[1], b.center[1]) &&
           isequal(a.center[2], b.center[2]) && isequal(a.area, b.area)
end

@inline function _probe_face_valid(probe, face, axis)
    return _probe_cell_valid(probe, face.minus) &&
           _probe_cell_valid(probe, face.plus) && face.axis == axis &&
           isfinite(face.center[1]) && isfinite(face.center[2]) &&
           isfinite(face.area) && face.area > 0
end

function _probe_raw_words(value)
    isbitstype(typeof(value)) || error("raw-word value is not isbits")
    sizeof(value) % sizeof(UInt64) == 0 || error("raw-word size is not word aligned")
    return [string(word; base=16, pad=16) for word in reinterpret(UInt64, [value])]
end

function _probe_layout(::Type{T}) where T
    return Dict("sizeof" => sizeof(T),
                "field_offsets" => [fieldoffset(T, i) for i in 1:fieldcount(T)])
end

function _probe_raw_record(raw)
    return Dict("lo" => raw.lo, "hi" => raw.hi,
                "cell" => [raw.cell.block, raw.cell.i, raw.cell.j],
                "raw_words_hex" => _probe_raw_words(raw))
end

function _probe_face_record(face)
    return Dict("minus" => [face.minus.block, face.minus.i, face.minus.j],
                "plus" => [face.plus.block, face.plus.i, face.plus.j],
                "axis" => face.axis,
                "center" => [face.center[1], face.center[2]],
                "area" => face.area,
                "raw_words_hex" => _probe_raw_words(face))
end

function _probe_note!(probe, stage, axis, line, index, record)
    probe.counts[stage] += 1
    label = "$(axis):$(line)"
    lines = probe.lines[stage]
    lines[label] = get(lines, label, 0) + 1
    if length(probe.first[stage]) < 10
        record["index"] = index
        record["axis_expected"] = axis
        record["line"] = line
        push!(probe.first[stage], record)
    end
    return nothing
end

function _probe_raw_table!(probe, table, axis)
    for (coordinate, (lower, upper)) in table
        for (side_name, sides) in (("lower", lower), ("upper", upper))
            for (index, raw) in pairs(sides)
                if !_probe_raw_valid(probe, raw, axis)
                    record = length(probe.first["raw_pre_match"]) < 10 ?
                        merge(_probe_raw_record(raw),
                              Dict("side" => side_name, "coordinate" => coordinate)) :
                        Dict{String,Any}()
                    _probe_note!(probe, "raw_pre_match", axis,
                                 coordinate, index, record)
                end
            end
        end
    end
    return nothing
end

function _probe_raw_pair!(probe, lower, upper, axis, coordinate, index)
    if !_probe_raw_valid(probe, lower, axis) ||
       !_probe_raw_valid(probe, upper, axis)
        record = length(probe.first["raw_at_match"]) < 10 ?
            Dict("lower" => _probe_raw_record(lower),
                 "upper" => _probe_raw_record(upper)) : Dict{String,Any}()
        _probe_note!(probe, "raw_at_match", axis, coordinate, index, record)
    end
    return nothing
end

function _probe_fresh_face!(probe, face, lower, upper, axis, coordinate, index)
    if !_probe_face_valid(probe, face, axis) ||
       !_probe_same_cell(face.minus, lower.cell) ||
       !_probe_same_cell(face.plus, upper.cell)
        record = length(probe.first["fresh"]) < 10 ?
            Dict("face" => _probe_face_record(face),
                 "lower" => _probe_raw_record(lower),
                 "upper" => _probe_raw_record(upper)) : Dict{String,Any}()
        _probe_note!(probe, "fresh", axis, coordinate, index, record)
    end
    return nothing
end

function _probe_pushed_face!(probe, stored, fresh, axis, coordinate, index)
    if !_probe_face_valid(probe, stored, axis) ||
       !_probe_same_face(stored, fresh)
        record = length(probe.first["after_push"]) < 10 ?
            Dict("stored" => _probe_face_record(stored),
                 "fresh" => _probe_face_record(fresh)) : Dict{String,Any}()
        _probe_note!(probe, "after_push", axis, coordinate, index, record)
    end
    return nothing
end

function _probe_final_faces!(probe, faces)
    expected_per_axis = probe.nx * probe.ny
    for (index, face) in pairs(faces)
        axis = index <= expected_per_axis ? 1 : 2
        if !_probe_face_valid(probe, face, axis)
            center_coordinate = face.center[axis]
            line = isfinite(center_coordinate) && 0 <= center_coordinate <= 1 ?
                round(Int, center_coordinate * (axis == 1 ? probe.nx : probe.ny)) :
                "out_of_range"
            record = length(probe.first["final"]) < 10 ?
                _probe_face_record(face) : Dict{String,Any}()
            _probe_note!(probe, "final", axis, line, index, record)
        end
    end
    return nothing
end

function _probe_report!(probe, face_count, boundary_count)
    path = get(ENV, "TWOPHASE_FACE_PROBE_OUTPUT", "")
    isempty(path) && error("TWOPHASE_FACE_PROBE_OUTPUT is required")
    summary = Dict{String,Any}(
        "schema" => 1, "fixture" => "periodic one-block 2048x2048",
        "face_count" => face_count, "boundary_count" => boundary_count,
        "expected_face_count" => 2 * probe.nx * probe.ny,
        "counts" => probe.counts, "first_invalid" => probe.first,
        "invalid_by_line" => probe.lines,
        "layouts" => Dict("RawSide" => _probe_layout(RawSide),
                          "CellRef" => _probe_layout(CellRef),
                          "InteriorFace_Float64" => _probe_layout(InteriorFace{Float64})))
    open(path, "w") do io
        Main.TOML.print(io, summary)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("face probe fsync failed")
    end
    return nothing
end
"""

function make_probe(source, output)
    source, output = abspath(source), abspath(output)
    !ispath(output) || error("probe output already exists: $output")
    clean = joinpath(source, RELATIVE)
    isfile(clean) && sha256file(clean) == CLEAN_SHA256 ||
        error("mesh source is not the frozen narrow-ID control")
    original = read(clean, String)
    candidate = unique_replace(original, BUILD_OLD, BUILD_NEW)
    candidate = unique_replace(candidate, MATCH_SIGNATURE_OLD, MATCH_SIGNATURE_NEW)
    candidate = unique_replace(candidate, MATCH_PUSH_OLD, MATCH_PUSH_NEW)
    candidate *= INSTRUMENTATION
    mkpath(output)
    cp(joinpath(source, "src"), joinpath(output, "src"))
    target = joinpath(output, RELATIVE)
    write(target, candidate)
    diff = run_capture(["diff", "-u", "--label", RELATIVE,
                        "--label", RELATIVE * ".producer-probe", clean, target])
    diff.exitcode == 1 || error("expected one copied-source diff")
    patch = joinpath(output, "face-producer.patch")
    write(patch, diff.stdout)
    receipt = Dict{String,Any}(
        "schema" => 1, "source" => source, "output" => output,
        "clean_mesh_sha256" => sha256file(clean),
        "probe_mesh_sha256" => sha256file(target),
        "clean_tree_sha256" => source_hash(source),
        "probe_tree_sha256" => source_hash(output),
        "generator_sha256" => sha256file(@__FILE__),
        "patch_sha256" => sha256file(patch),
        "scope" => "copied mesh RawSide, fresh face, after-push, final face checks; stop before dual volumes")
    writejson(joinpath(output, "face-producer-receipt.json"), receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_face_producer_probe.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 4 || error("--source and --output are required")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    Set(keys(options)) == Set(("--source", "--output")) ||
        error("expected only --source and --output")
    receipt = make_probe(options["--source"], options["--output"])
    println("created copied face producer probe: ", receipt["output"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
