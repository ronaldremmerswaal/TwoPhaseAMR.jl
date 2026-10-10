#!/usr/bin/env julia
"""Tiny wrapped vertical-face and injected bad-block probe check."""

include(joinpath(@__DIR__,"check_raw_checkpoint.jl"))

faces = M.InteriorFace{Float64}[]
boundaries = M.BoundaryFace{Float64}[]
M._match_sides!(faces,boundaries,small_table(1),1,
                (0.0,0.0),(1.0,1.0),(2,2))
length(faces)==4 && isempty(boundaries) ||
    error("tiny periodic vertical matcher produced wrong face population")
good_path = tempname()
ENV["TWOPHASE_VERTICAL_CHECKPOINT_OUTPUT"] = good_path
M._vertical_checkpoint!(faces,2,2,1)
good = TOML.parsefile(good_path)
good["invalid_count"]==0 && good["population_ok"] ||
    error("valid tiny vertical faces were rejected")

bad_faces = copy(faces)
old = bad_faces[1]
bad_faces[1] = M.InteriorFace(M.CellRef(262145,old.minus.i,old.minus.j),
    old.plus,old.axis,old.center,old.area)
bad_path = tempname()
ENV["TWOPHASE_VERTICAL_CHECKPOINT_OUTPUT"] = bad_path
M._vertical_checkpoint!(bad_faces,2,2,1)
bad = TOML.parsefile(bad_path)
bad["invalid_count"]==1 || error("injected bad face was not uniquely rejected")
bad["first_invalid"][1]["backing_words_hex"][1] ==
    string(UInt64(262145);base=16,pad=16) ||
    error("bad face backing word differs")
println("synthetic vertical face geometry and backing-word checks passed")
