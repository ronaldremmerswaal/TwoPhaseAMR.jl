#!/usr/bin/env julia
"""Copy frozen mesh source with RawSide and vertical-face checkpoints."""
module MeshCrashVerticalCheckpoint

include(joinpath(@__DIR__, "make_raw_checkpoint.jl"))
using .MeshCrashRawCheckpoint

const Raw = MeshCrashRawCheckpoint
const Support = Raw.ScalingSupport
const RELATIVE = Raw.RELATIVE
const CLEAN_SHA256 = Raw.CLEAN_SHA256
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
    raw_report = _raw_checkpoint!(vertical, horizontal, nx, ny, length(blocks))
    raw_report["invalid_count"] == 0 && raw_report["population_ok"] ||
        error("RawSide checkpoint failed before matching")
    T = eltype(blocks[1].origin)
    faces = InteriorFace{T}[]
    boundaries = BoundaryFace{T}[]
    _match_sides!(faces, boundaries, vertical, 1, origin, extent, (sx, sy))
    _vertical_checkpoint!(faces, nx, ny, length(blocks))
    error("vertical-face checkpoint stopped before horizontal matching")
    _match_sides!(faces, boundaries, horizontal, 2, origin, extent, (sx, sy))
    return faces, boundaries
end
"""
const RAW_RETURN_OLD = """
            error("RawSide checkpoint fsync failed")
    end
    return nothing
end
"""
const RAW_RETURN_NEW = """
            error("RawSide checkpoint fsync failed")
    end
    return report
end
"""
const VERTICAL_HELPER = raw"""

function _vertical_checkpoint!(faces, nx, ny, block_count)
    F = InteriorFace{Float64}
    isbitstype(F) && sizeof(F) == 10sizeof(UInt64) &&
        fieldoffset(F,1) == 0 && fieldoffset(F,2) == 24 &&
        fieldoffset(F,3) == 48 && fieldoffset(F,4) == 56 &&
        fieldoffset(F,5) == 72 ||
        error("unexpected InteriorFace{Float64} layout")
    backing = reinterpret(UInt64, faces)
    length(backing) == 10length(faces) ||
        error("vertical-face backing-word count mismatch")
    expected_count = nx*ny
    invalid = 0
    first_invalid = Any[]
    invalid_by_line = Dict{String,Int}()
    fnv64 = UInt64(0xcbf29ce484222325)
    sum64 = zero(UInt64)
    for (index, face) in pairs(faces)
        minus, plus = face.minus, face.plus
        valid = block_count == 1 &&
            minus.block == 1 && plus.block == 1 &&
            1 <= minus.i <= nx && 1 <= minus.j <= ny &&
            1 <= plus.i <= nx && 1 <= plus.j <= ny &&
            face.axis == 1 && plus.i == mod(minus.i,nx)+1 &&
            plus.j == minus.j &&
            face.center[1] == mod(minus.i,nx)/nx &&
            face.center[2] == (minus.j-0.5)/ny &&
            face.area == 1.0/ny
        word_index = 10(index-1)
        for offset in 1:10
            word = backing[word_index+offset]
            fnv64 = (fnv64 ⊻ word) * UInt64(0x100000001b3)
            sum64 += word
        end
        if !valid
            invalid += 1
            line = isfinite(face.center[1]) && 0 <= face.center[1] <= 1 ?
                string(round(Int,face.center[1]*nx)) : "out_of_range"
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
        "face_count"=>length(faces),"expected_count"=>expected_count,
        "population_ok"=>length(faces)==expected_count,
        "invalid_count"=>invalid,"first_invalid"=>first_invalid,
        "invalid_by_line"=>invalid_by_line,
        "backing_fnv64_hex"=>string(fnv64;base=16,pad=16),
        "backing_sum64_hex"=>string(sum64;base=16,pad=16),
        "face_size"=>sizeof(F),
        "face_field_offsets"=>[fieldoffset(F,i) for i in 1:5])
    path = ENV["TWOPHASE_VERTICAL_CHECKPOINT_OUTPUT"]
    open(path,"w") do io
        Main.TOML.print(io,report)
        flush(io)
        ccall(:fsync,Cint,(Cint,),Base.fd(io)) == 0 ||
            error("vertical-face checkpoint fsync failed")
    end
    return nothing
end
"""

function replace_once(source,before,after)
    length(split(source,before)) == 2 || error("vertical checkpoint anchor missing/repeated")
    return replace(source,before=>after;count=1)
end

function make_probe(source,output)
    source,output=abspath(source),abspath(output)
    !ispath(output) || error("vertical checkpoint output exists")
    clean=joinpath(source,RELATIVE)
    isfile(clean) && Support.sha256file(clean)==CLEAN_SHA256 ||
        error("mesh source differs from frozen control")
    helpers=replace_once(Raw.HELPERS,RAW_RETURN_OLD,RAW_RETURN_NEW)
    candidate=replace_once(read(clean,String),BUILD_OLD,BUILD_NEW)*helpers*VERTICAL_HELPER
    mkpath(output)
    cp(joinpath(source,"src"),joinpath(output,"src"))
    target=joinpath(output,RELATIVE)
    write(target,candidate)
    diff=Support.run_capture(["diff","-u","--label",RELATIVE,
        "--label",RELATIVE*".vertical-checkpoint",clean,target])
    diff.exitcode==1 || error("expected copied-source patch")
    patch=joinpath(output,"vertical-checkpoint.patch")
    write(patch,diff.stdout)
    receipt=Dict{String,Any}(
        "schema"=>1,"source"=>source,"output"=>output,
        "clean_mesh_sha256"=>Support.sha256file(clean),
        "checkpoint_mesh_sha256"=>Support.sha256file(target),
        "clean_tree_sha256"=>Support.source_hash(source),
        "checkpoint_tree_sha256"=>Support.source_hash(output),
        "generator_sha256"=>Support.sha256file(@__FILE__),
        "raw_helper_generator_sha256"=>Support.sha256file(joinpath(@__DIR__,"make_raw_checkpoint.jl")),
        "patch_sha256"=>Support.sha256file(patch),
        "scope"=>"pre-match RawSide scan, unchanged vertical matcher, vertical face backing scan, stop")
    Support.writejson(joinpath(output,"vertical-checkpoint-receipt.json"),receipt)
    return receipt
end

function main(args=ARGS)
    if any(arg->arg in ("-h","--help"),args)
        println("usage: make_vertical_checkpoint.jl --source FROZEN_SOURCE --output NEW_DIRECTORY")
        return nothing
    end
    length(args)==4 || error("--source and --output required")
    options=Dict(args[i]=>args[i+1] for i in 1:2:length(args))
    Set(keys(options))==Set(("--source","--output")) ||
        error("only --source and --output accepted")
    receipt=make_probe(options["--source"],options["--output"])
    println("created vertical checkpoint: ",receipt["output"])
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

end # module
