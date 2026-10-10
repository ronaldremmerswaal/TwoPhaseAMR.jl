#!/usr/bin/env julia
"""Tiny periodic x-before/x-after/y scan with an injected backing-word error."""

include(joinpath(@__DIR__,"check_raw_checkpoint.jl"))

faces = M.InteriorFace{Float64}[]
boundaries = M.BoundaryFace{Float64}[]
M._match_sides!(faces,boundaries,small_table(1),1,
                (0.0,0.0),(1.0,1.0),(2,2))
path = tempname()
ENV["TWOPHASE_X_BEFORE"] = path
before = M._face_segment_checkpoint!(faces,2,2,1,1,1,4,4,
    "TWOPHASE_X_BEFORE")
before["invalid_count"]==0 || error("valid x faces rejected before y match")
M._match_sides!(faces,boundaries,small_table(2),2,
                (0.0,0.0),(1.0,1.0),(2,2))
ENV["TWOPHASE_X_AFTER"] = tempname()
after = M._face_segment_checkpoint!(faces,2,2,1,1,1,4,8,
    "TWOPHASE_X_AFTER")
after["invalid_count"]==0 &&
    after["backing_fnv64_hex"]==before["backing_fnv64_hex"] &&
    after["backing_sum64_hex"]==before["backing_sum64_hex"] ||
    error("valid x prefix changed after y match")
ENV["TWOPHASE_Y_AFTER"] = tempname()
y = M._face_segment_checkpoint!(faces,2,2,1,2,5,8,8,
    "TWOPHASE_Y_AFTER")
y["invalid_count"]==0 && isempty(boundaries) ||
    error("valid y suffix rejected")

bad_faces=copy(faces)
old=bad_faces[8]
bad_faces[8]=M.InteriorFace(M.CellRef(262145,old.minus.i,old.minus.j),
    old.plus,old.axis,old.center,old.area)
ENV["TWOPHASE_Y_AFTER"] = tempname()
bad=M._face_segment_checkpoint!(bad_faces,2,2,1,2,5,8,8,
    "TWOPHASE_Y_AFTER")
bad["invalid_count"]==1 &&
    bad["first_invalid"][1]["backing_words_hex"][1]==
        string(UInt64(262145);base=16,pad=16) ||
    error("injected y-face backing-word error was missed")
println("synthetic same-process x-prefix/y-suffix checks passed")
