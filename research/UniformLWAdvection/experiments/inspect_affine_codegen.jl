# Untimed, read-only compiler diagnostic for the frozen affine tile kernel.
# Run only after performance measurements finish. This does not execute a
# numerical tile or modify any benchmarked source.
using InteractiveUtils, SHA
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"..")))
using Polyester
using UniformLWAdvection

include(joinpath(@__DIR__,"fused_affine.jl"))
const Affine = UniformLWFusedAffineExperiment

default_dir = normpath(joinpath(@__DIR__,"..","..","..","benchmark",
    "scaling","results","20260929-stage2-smt-baseline110-pinned-spread",
    "codegen"))
output_dir = get(ENV, "TWOPHASE_CODEGEN_DIR", default_dir)
mkpath(output_dir)
open(joinpath(output_dir,"metadata.txt"),"w") do io
    println(io,"Julia ",VERSION)
    for file in ("fused.jl","fused_affine.jl","fused_affine_polyester.jl",
                 "inspect_affine_codegen.jl")
        path=joinpath(@__DIR__,file)
        println(io,bytes2hex(sha256(read(path))),"  ",path)
    end
end

state = State(128, 96; u=0.61, v=-0.27)
initialize!(state)
workspace = Affine.FusedWorkspace(
    state; backend=:polyester, tile=(32,8), threshold=0,
)
scratch = workspace.scratch[1]
tile = workspace.tiles[first(workspace.chunks[1])]
q0, out = state.q, workspace.next

# Polyester's `@batch` routes captured array arguments through
# StrideArraysCore.object_and_preserve; the two PtrArray objects here have the
# same types as q0/out captured by the p>=2 batch closure. `state` and
# `workspace` are captured as ordinary mutable structs, so their nested
# u/v/scratch matrices remain Matrix values.
pointer_q0, preserve_q0 = Polyester.object_and_preserve(q0)
pointer_out, preserve_out = Polyester.object_and_preserve(out)

function dump_kernel(label, q, destination)
    types = Tuple{
        typeof(state), typeof(workspace), typeof(scratch), typeof(tile),
        typeof(q), typeof(destination), Float64, Float64,
    }
    path = joinpath(output_dir, label * ".llvm")
    open(path, "w") do io
        code_llvm(io, Affine.affine_tile!, types;
                  optimize=true, raw=true, debuginfo=:none)
    end
    typed_path = joinpath(output_dir, label * ".typed")
    open(typed_path, "w") do io
        code_warntype(io, Affine.affine_tile!, types; optimize=true)
    end
    println(label, " q=", typeof(q), " out=", typeof(destination),
            " llvm=", path, " typed=", typed_path)
end

function dump_copy(label, source)
    # In the batch copy-back closure `s.q` is nested inside captured State
    # and stays Matrix, while the directly captured source becomes PtrArray.
    types = Tuple{typeof(state.q), typeof(source), typeof(tile)}
    path = joinpath(output_dir, label * ".llvm")
    open(path, "w") do io
        code_llvm(io, Affine.BaseFused.copy_tile!, types;
                  optimize=true, raw=true, debuginfo=:none)
    end
    typed_path = joinpath(output_dir, label * ".typed")
    open(typed_path, "w") do io
        code_warntype(io, Affine.BaseFused.copy_tile!, types; optimize=true)
    end
    println(label, " source=", typeof(source),
            " llvm=", path, " typed=", typed_path)
end

GC.@preserve preserve_q0 preserve_out begin
    dump_kernel("serial_matrix", q0, out)
    dump_kernel("polyester_ptrarray", pointer_q0, pointer_out)
    dump_copy("copyback_matrix", out)
    dump_copy("copyback_ptrarray", pointer_out)
end

# Search the resulting LLVM for vector widths, loop-vectorization metadata,
# `jl_bounds_error`, and scalar division/modulo calls. These are static
# compiler artifacts; they do not establish a runtime slowdown by themselves.
