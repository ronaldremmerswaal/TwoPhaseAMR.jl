#!/usr/bin/env julia
"""Copy frozen mesh with same-process RawSide, x-before/x-after, and y scans."""
module MeshCrashFullFaceCheckpoint

include(joinpath(@__DIR__, "make_vertical_checkpoint.jl"))
using .MeshCrashVerticalCheckpoint

const Vertical = MeshCrashVerticalCheckpoint
const Support = Vertical.Support
const RELATIVE = Vertical.RELATIVE
const CLEAN_SHA256 = Vertical.CLEAN_SHA256
const BUILD_NEW = """
    raw_report = _raw_checkpoint!(vertical, horizontal, nx, ny, length(blocks))
    raw_report["invalid_count"] == 0 && raw_report["population_ok"] ||
        error("RawSide checkpoint failed before matching")
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(faces, boundaries, vertical, 1, origin, extent, (sx, sy))
    x_before = _face_segment_checkpoint!(faces, nx, ny, length(blocks),
        1, 1, nx*ny, nx*ny, "TWOPHASE_X_BEFORE")
    x_before["invalid_count"] == 0 && x_before["population_ok"] ||
        error("x prefix invalid before horizontal matching")
    _match_sides!(faces, boundaries, horizontal, 2, origin, extent, (sx, sy))
    _face_segment_checkpoint!(faces, nx, ny, length(blocks),
        1, 1, nx*ny, 2nx*ny, "TWOPHASE_X_AFTER")
    _face_segment_checkpoint!(faces, nx, ny, length(blocks),
        2, nx*ny+1, 2nx*ny, 2nx*ny, "TWOPHASE_Y_AFTER")
    error("full-face checkpoint stopped before halos and dual volumes")
    return faces, boundaries
end
"""

const FACE_HELPER = raw"""

function _face_segment_checkpoint!(faces, nx, ny, block_count,
                                   axis, first_index, last_index,
                                   expected_total, path_key)
    F = InteriorFace{Float64}
    isbitstype(F) && sizeof(F) == 10sizeof(UInt64) &&
        fieldoffset(F,1) == 0 && fieldoffset(F,2) == 24 &&
        fieldoffset(F,3) == 48 && fieldoffset(F,4) == 56 &&
        fieldoffset(F,5) == 72 ||
        error("unexpected InteriorFace{Float64} layout")
    backing = reinterpret(UInt64, faces)
    length(backing) == 10length(faces) ||
        error("face backing-word count mismatch")
    expected_segment = nx*ny
    population_ok = length(faces) == expected_total &&
        last_index-first_index+1 == expected_segment &&
        checkbounds(Bool, faces, first_index) &&
        checkbounds(Bool, faces, last_index)
    population_ok || error("face segment population or bounds mismatch")
    invalid = 0
    first_invalid = Any[]
    invalid_by_line = Dict{String,Int}()
    fnv64 = UInt64(0xcbf29ce484222325)
    sum64 = zero(UInt64)
    for index in first_index:last_index
        face = faces[index]
        minus, plus = face.minus, face.plus
        valid = block_count == 1 &&
            minus.block == 1 && plus.block == 1 &&
            1 <= minus.i <= nx && 1 <= minus.j <= ny &&
            1 <= plus.i <= nx && 1 <= plus.j <= ny &&
            face.axis == axis
        if axis == 1
            valid &= plus.i == mod(minus.i,nx)+1 &&
                plus.j == minus.j &&
                face.center[1] == mod(minus.i,nx)/nx &&
                face.center[2] == (minus.j-0.5)/ny &&
                face.area == 1.0/ny
        else
            valid &= plus.i == minus.i &&
                plus.j == mod(minus.j,ny)+1 &&
                face.center[1] == (minus.i-0.5)/nx &&
                face.center[2] == mod(minus.j,ny)/ny &&
                face.area == 1.0/nx
        end
        word_index = 10(index-1)
        for offset in 1:10
            word = backing[word_index+offset]
            fnv64 = (fnv64 ⊻ word) * UInt64(0x100000001b3)
            sum64 += word
        end
        if !valid
            invalid += 1
            line_coordinate = face.center[axis]
            line = isfinite(line_coordinate) && 0 <= line_coordinate <= 1 ?
                string(round(Int,line_coordinate*(axis==1 ? nx : ny))) :
                "out_of_range"
            invalid_by_line[line] = get(invalid_by_line,line,0)+1
            if length(first_invalid) < 10
                push!(first_invalid,Dict(
                    "index"=>index,
                    "minus"=>[minus.block,minus.i,minus.j],
                    "plus"=>[plus.block,plus.i,plus.j],
                    "axis"=>face.axis,
                    "center"=>[face.center[1],face.center[2]],
                    "area"=>face.area,
                    "backing_words_hex"=>[
                        string(backing[word_index+offset];base=16,pad=16)
                        for offset in 1:10]))
            end
        end
    end
    report = Dict{String,Any}(
        "schema"=>1,"fixture"=>"periodic one-block 2048x2048",
        "axis_expected"=>axis,
        "first_index"=>first_index,"last_index"=>last_index,
        "face_array_length"=>length(faces),
        "expected_total"=>expected_total,
        "population_ok"=>population_ok,
        "invalid_count"=>invalid,"first_invalid"=>first_invalid,
        "invalid_by_line"=>invalid_by_line,
        "backing_fnv64_hex"=>string(fnv64;base=16,pad=16),
        "backing_sum64_hex"=>string(sum64;base=16,pad=16),
        "face_size"=>sizeof(F),
        "face_field_offsets"=>[fieldoffset(F,i) for i in 1:5])
    path = ENV[path_key]
    open(path,"w") do io
        Main.TOML.print(io,report)
        flush(io)
        ccall(:fsync,Cint,(Cint,),Base.fd(io)) == 0 ||
            error("face segment checkpoint fsync failed")
    end
    return report
end
"""

function make_probe(source,output)
    source,output=abspath(source),abspath(output)
    !ispath(output) || error("full-face checkpoint output exists")
    clean=joinpath(source,RELATIVE)
    isfile(clean) && Support.sha256file(clean)==CLEAN_SHA256 ||
        error("mesh source differs from frozen control")
    raw_helpers=Vertical.replace_once(Vertical.Raw.HELPERS,
        Vertical.RAW_RETURN_OLD,Vertical.RAW_RETURN_NEW)
    candidate=Vertical.replace_once(read(clean,String),Vertical.BUILD_OLD,
        BUILD_NEW)*raw_helpers*FACE_HELPER
    mkpath(output)
    cp(joinpath(source,"src"),joinpath(output,"src"))
    target=joinpath(output,RELATIVE)
    write(target,candidate)
    diff=Support.run_capture(["diff","-u","--label",RELATIVE,
        "--label",RELATIVE*".full-face-checkpoint",clean,target])
    diff.exitcode==1 || error("expected copied-source patch")
    patch=joinpath(output,"full-face-checkpoint.patch")
    write(patch,diff.stdout)
    receipt=Dict{String,Any}(
        "schema"=>1,"source"=>source,"output"=>output,
        "clean_mesh_sha256"=>Support.sha256file(clean),
        "checkpoint_mesh_sha256"=>Support.sha256file(target),
        "clean_tree_sha256"=>Support.source_hash(source),
        "checkpoint_tree_sha256"=>Support.source_hash(output),
        "generator_sha256"=>Support.sha256file(@__FILE__),
        "vertical_generator_sha256"=>Support.sha256file(joinpath(@__DIR__,"make_vertical_checkpoint.jl")),
        "raw_generator_sha256"=>Support.sha256file(joinpath(@__DIR__,"make_raw_checkpoint.jl")),
        "patch_sha256"=>Support.sha256file(patch),
        "scope"=>"same-process RawSide, x-before, x-after, y-after backing scans; unchanged matchers")
    Support.writejson(joinpath(output,"full-face-checkpoint-receipt.json"),receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg->arg in ("-h","--help"),args)
        println("usage: make_full_face_checkpoint.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args)==4 || error("--source and --output required")
    options=Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    Set(keys(options))==Set(("--source","--output")) ||
        error("only --source and --output accepted")
    receipt=make_probe(options["--source"],options["--output"])
    println("created full-face checkpoint: ",receipt["output"])
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

end # module
