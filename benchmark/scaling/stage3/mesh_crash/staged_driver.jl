#!/usr/bin/env julia
"""Validate a frozen 2048 face schedule before the original dual-volume loop."""

using SHA
using TOML

length(ARGS) in (2,3) ||
    error("usage: staged_driver.jl FROZEN_SOURCE EVENTS_TOML [staged|phase_scan]")
source, events_path = ARGS[1:2]
mode = length(ARGS) == 3 ? ARGS[3] : "staged"
mode in ("staged", "phase_scan") || error("unknown diagnostic mode: $mode")
const SOURCE = abspath(source)
const MODULE_FILE = joinpath(SOURCE, "src", "TwoPhaseAMR.jl")
const MESH_FILE = joinpath(SOURCE, "src", "mesh", "topology", "mesh.jl")
const ADVECTION_FILE = joinpath(SOURCE, "src", "operators", "advection", "cell_advection.jl")
const MESH_SHA256 = "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"
const ADVECTION_SHA256 = "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(MESH_FILE) == MESH_SHA256 &&
    hashfile(ADVECTION_FILE) == ADVECTION_SHA256 ||
    error("staged source differs from the frozen control")
Base.include(Main, MODULE_FILE)
using .TwoPhaseAMR
method = only(collect(methods(TwoPhaseAMR._face_dual_volumes)))
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("dual-volume method loaded outside frozen source")

function rss_kib()
    match_result = match(r"(?m)^VmRSS:\s+(\d+) kB$", read("/proc/self/status", String))
    return match_result === nothing ? -1 : parse(Int, match_result.captures[1])
end

function event!(phase; faces=-1, blocks=-1, halo=-1, face_index=0,
                stats=Dict{String,Any}())
    entry = Dict{String,Any}(
        "phase" => phase, "faces" => faces, "blocks" => blocks,
        "halo_entries" => halo, "face_index" => face_index,
        "time_ns" => time_ns(), "rss_kib" => rss_kib(),
        "julia_version" => string(VERSION),
        "julia_executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "source" => SOURCE, "mesh_sha256" => hashfile(MESH_FILE),
        "advection_sha256" => hashfile(ADVECTION_FILE),
        "dual_volume_method_file" => realpath(String(method.file)),
        "threads" => Threads.nthreads(:default))
    merge!(entry, stats)
    open(events_path, "a") do io
        TOML.print(io, Dict("events" => [entry]))
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || error("event fsync failed")
    end
    return nothing
end

function validate_faces(blocks, faces, n)
    length(blocks) == 1 || error("one-block fixture made $(length(blocks)) blocks")
    length(faces) == 2n*n || error("unexpected face count $(length(faces))")
    min_block, max_block = typemax(Int), typemin(Int)
    min_axis, max_axis = typemax(Int), typemin(Int)
    min_i, max_i = typemax(Int), typemin(Int)
    min_j, max_j = typemax(Int), typemin(Int)
    min_area, max_area = Inf, -Inf
    for (index, face) in pairs(faces)
        axis = face.axis
        minus, plus = face.minus, face.plus
        valid = checkbounds(Bool, blocks, minus.block) &&
                checkbounds(Bool, blocks, plus.block) &&
                1 <= axis <= 2 &&
                1 <= minus.i <= n && 1 <= minus.j <= n &&
                1 <= plus.i <= n && 1 <= plus.j <= n
        if !valid
            event!("invalid_face"; faces=length(faces), blocks=length(blocks),
                   face_index=index)
            error("invalid face $index: minus=$minus plus=$plus axis=$axis")
        end
        # Checked array/tuple loads prove every reference consumed by the
        # original dual-volume expression is valid at scan time.
        minus_spacing = blocks[minus.block].spacing[axis]
        plus_spacing = blocks[plus.block].spacing[axis]
        isfinite(face.area) && face.area > 0 &&
            isfinite(minus_spacing) && minus_spacing > 0 &&
            isfinite(plus_spacing) && plus_spacing > 0 ||
            error("invalid geometry at face $index")
        min_block = min(min_block, minus.block, plus.block)
        max_block = max(max_block, minus.block, plus.block)
        min_axis = min(min_axis, axis)
        max_axis = max(max_axis, axis)
        min_i = min(min_i, minus.i, plus.i)
        max_i = max(max_i, minus.i, plus.i)
        min_j = min(min_j, minus.j, plus.j)
        max_j = max(max_j, minus.j, plus.j)
        min_area = min(min_area, face.area)
        max_area = max(max_area, face.area)
    end
    return Dict{String,Any}(
        "min_block" => min_block, "max_block" => max_block,
        "min_axis" => min_axis, "max_axis" => max_axis,
        "min_i" => min_i, "max_i" => max_i,
        "min_j" => min_j, "max_j" => max_j,
        "min_area" => min_area, "max_area" => max_area)
end

@inline _face_bits(value::Int) = reinterpret(UInt64, value)
@inline _face_bits(value::Float64) = reinterpret(UInt64, value)

function face_record(index, face)
    minus, plus = face.minus, face.plus
    fields = (_face_bits(minus.block), _face_bits(minus.i), _face_bits(minus.j),
              _face_bits(plus.block), _face_bits(plus.i), _face_bits(plus.j),
              _face_bits(face.axis), _face_bits(face.center[1]),
              _face_bits(face.center[2]), _face_bits(face.area))
    return Dict{String,Any}(
        "index" => index,
        "minus" => [minus.block, minus.i, minus.j],
        "plus" => [plus.block, plus.i, plus.j],
        "axis" => face.axis,
        "center" => [face.center[1], face.center[2]],
        "area" => face.area,
        "field_bits_hex" => [string(bits; base=16, pad=16) for bits in fields])
end

function reread_faces(faces, records)
    return [face_record(record["index"], faces[record["index"]])
            for record in records]
end

function scan_faces(blocks, faces, n)
    first_invalid = Dict{String,Any}[]
    invalid_count = 0
    checksum = UInt64(0xcbf29ce484222325)
    field_sum = zero(UInt64)
    for (index, face) in pairs(faces)
        minus, plus, axis = face.minus, face.plus, face.axis
        fields = (_face_bits(minus.block), _face_bits(minus.i), _face_bits(minus.j),
                  _face_bits(plus.block), _face_bits(plus.i), _face_bits(plus.j),
                  _face_bits(axis), _face_bits(face.center[1]),
                  _face_bits(face.center[2]), _face_bits(face.area))
        for bits in fields
            checksum = (checksum ⊻ bits) * UInt64(0x100000001b3)
            field_sum += bits
        end
        valid = checkbounds(Bool, blocks, minus.block) &&
                checkbounds(Bool, blocks, plus.block) &&
                1 <= axis <= 2 &&
                1 <= minus.i <= n && 1 <= minus.j <= n &&
                1 <= plus.i <= n && 1 <= plus.j <= n &&
                isfinite(face.center[1]) && isfinite(face.center[2]) &&
                isfinite(face.area) && face.area > 0
        if valid
            a = blocks[minus.block].spacing[axis]
            b = blocks[plus.block].spacing[axis]
            valid = isfinite(a) && a > 0 && isfinite(b) && b > 0
        end
        if !valid
            invalid_count += 1
            if length(first_invalid) < 10
                record = face_record(index, face)
                reread = face_record(index, faces[index])
                for (key, value) in reread
                    key == "index" && continue
                    record["immediate_reread_$key"] = value
                end
                push!(first_invalid, record)
            end
        end
    end
    return Dict{String,Any}(
        "invalid_count" => invalid_count,
        "first_invalid" => first_invalid,
        "field_fnv64_hex" => string(checksum; base=16, pad=16),
        "field_sum64_hex" => string(field_sum; base=16, pad=16))
end

function staged_mesh(mode)
    n = 2048
    block_size = base_mesh = (n,n)
    origin, extent, periodic = (0.0,0.0), (1.0,1.0), (true,true)
    max_level, halo = 0, 1
    refine = (key, origin, extent) -> false
    event!("before_root")
    root = TwoPhaseAMR._build_root_grid(origin, extent, base_mesh,
                                        block_size, periodic, max_level, refine)
    event!("after_root")
    nodes, blocks = TwoPhaseAMR._leaf_blocks(root, block_size)
    event!("after_leaf_blocks"; blocks=length(blocks))
    faces, boundaries = TwoPhaseAMR._build_face_schedules(
        blocks, base_mesh, block_size, origin, extent, periodic, max_level)
    before_scan = if mode == "phase_scan"
        stats = scan_faces(blocks, faces, n)
        event!("scanned_after_faces"; faces=length(faces),
               blocks=length(blocks), stats)
        stats
    else
        event!("after_face_schedule"; faces=length(faces), blocks=length(blocks))
        nothing
    end
    halo_entries, halo_donors, halo_weights = TwoPhaseAMR._build_halo_schedule(
        root, nodes, blocks, block_size, halo, origin, extent, periodic)
    isempty(boundaries) || error("periodic one-block mesh has boundary faces")
    length(halo_donors) == length(halo_weights) ||
        error("halo donor and weight arrays differ")
    if mode == "phase_scan"
        after_scan = scan_faces(blocks, faces, n)
        after_scan["prior_invalid_at_after_halo"] =
            reread_faces(faces, before_scan["first_invalid"])
        event!("scanned_after_halo"; faces=length(faces), blocks=length(blocks),
               halo=length(halo_entries), stats=after_scan)
        if after_scan["invalid_count"] > 0
            event!("reread_after_halo_invalid"; faces=length(faces),
                   blocks=length(blocks), halo=length(halo_entries),
                   stats=Dict{String,Any}(
                       "rereads" => reread_faces(faces, after_scan["first_invalid"])))
        end
        if before_scan["invalid_count"] > 0 || after_scan["invalid_count"] > 0
            event!("skipped_dual_invalid_face"; faces=length(faces),
                   blocks=length(blocks), halo=length(halo_entries))
            return nothing
        end
    else
        event!("after_halo_schedule"; faces=length(faces), blocks=length(blocks),
               halo=length(halo_entries))
        stats = validate_faces(blocks, faces, n)
        event!("validated_faces"; faces=length(faces), blocks=length(blocks),
               halo=length(halo_entries), stats)
    end
    dual = TwoPhaseAMR._face_dual_volumes(blocks, faces)
    length(dual) == length(faces) || error("dual-volume length mismatch")
    expected = 1.0 / (n*n)
    checksum, minimum_value, maximum_value = 0.0, Inf, -Inf
    for (index, value) in pairs(dual)
        if !(isfinite(value) && value == expected)
            event!("invalid_dual_volume"; faces=length(faces),
                   blocks=length(blocks), face_index=index)
            error("invalid dual volume at face $index: $value != $expected")
        end
        checksum += value
        minimum_value = min(minimum_value, value)
        maximum_value = max(maximum_value, value)
    end
    checksum == expected * length(dual) || error("dual-volume checksum differs")
    event!("original_dual_complete"; faces=length(faces), blocks=length(blocks),
           halo=length(halo_entries), stats=Dict{String,Any}(
               "dual_checksum" => checksum,
               "dual_min" => minimum_value, "dual_max" => maximum_value,
               "dual_expected" => expected))
    return nothing
end

event!("module_loaded")
staged_mesh(mode)
event!("complete")
