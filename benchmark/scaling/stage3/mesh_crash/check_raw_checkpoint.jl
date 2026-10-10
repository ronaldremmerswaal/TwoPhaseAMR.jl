#!/usr/bin/env julia
"""Tiny synthetic polarity/wrap and bad-block check for a copied checkpoint."""

using SHA
using TOML

length(ARGS)==2 || error("usage: check_raw_checkpoint.jl COPIED_SOURCE EXPECTED_MESH_SHA256")
source, expected_mesh = abspath(ARGS[1]), ARGS[2]
mesh_file = joinpath(source,"src","mesh","topology","mesh.jl")
bytes2hex(SHA.sha256(read(mesh_file))) == expected_mesh ||
    error("copied mesh source hash differs")
Base.include(Main,joinpath(source,"src","TwoPhaseAMR.jl"))
using .TwoPhaseAMR
M = TwoPhaseAMR
for function_name in (:_scan_raw_table!, :_raw_checkpoint!)
    f = getfield(M,function_name)
    method = only(collect(methods(f)))
    realpath(String(method.file)) == realpath(mesh_file) ||
        error("loaded checkpoint method outside copied mesh source")
end

function small_table(axis)
    table = Dict{Int,Tuple{Vector{M.RawSide},Vector{M.RawSide}}}()
    for j in 1:2, i in 1:2
        cell = M.CellRef(1,i,j)
        normal = axis==1 ? i : j
        tangent = axis==1 ? j : i
        M._insert_side!(table,mod(normal-1,2),-1,tangent-1,tangent,cell)
        M._insert_side!(table,mod(normal,2),1,tangent-1,tangent,cell)
    end
    # Positive side of the final cell and negative side of the first cell
    # meet at the periodic key zero.
    lower,upper = table[0]
    any(raw->(axis==1 ? raw.cell.i : raw.cell.j)==2,lower) ||
        error("positive periodic-wrap side missing")
    any(raw->(axis==1 ? raw.cell.i : raw.cell.j)==1,upper) ||
        error("negative periodic-wrap side missing")
    return table
end

vertical,horizontal = small_table(1),small_table(2)
stats = M._RawCheckpointStats()
M._scan_raw_table!(stats,vertical,1,2,2,1)
M._scan_raw_table!(stats,horizontal,2,2,2,1)
stats.counts == [4,4,4,4] && stats.invalid==0 ||
    error("valid lower-positive/upper-negative synthetic sides failed")
M._insert_side!(vertical,0,1,0,1,M.CellRef(262145,2,1))
bad = M._RawCheckpointStats()
M._scan_raw_table!(bad,vertical,1,2,2,1)
bad.invalid == 1 && bad.counts == [4,5,0,0] ||
    error("injected bad-block side was not uniquely rejected")
bad.first_invalid[1]["cell"][1] == 262145 ||
    error("bad-block receipt lost original value")
bad.first_invalid[1]["backing_words_hex"][3] ==
    string(UInt64(262145);base=16,pad=16) ||
    error("bad-block backing array word differs")
println("synthetic RawSide polarity/wrap and backing-word checks passed")
