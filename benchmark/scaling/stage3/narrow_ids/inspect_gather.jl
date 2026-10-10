#!/usr/bin/env julia
module NarrowIdsGatherIR

using InteractiveUtils
using SHA
using TOML

function representative_gather_columns!(workspace, scalar, increment,
                                        block::Int, first_j::Int, last_j::Int)
    # Exact ordered-gather loop body from the production advect! callback.
    # Runtime cell and packed order values are loaded from the concrete workspace.
    mesh = scalar.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    increment_block = @inbounds increment.blocks[block]
    scalar_block = @inbounds scalar.blocks[block]
    volume = mesh.blocks[block].volume
    cell_offset = (block - 1) * cells_per_block
    @inbounds for j in first_j:last_j, i in 1:nx
        cell = cell_offset + (j - 1) * nx + i
        integrated = Main.TwoPhaseAMR._cell_advection_sum(
            workspace, block, i, j, nx, ny, cell, true, Float64)
        change = integrated / volume
        increment_block[i + h, j + h] = change
        scalar_block[i + h, j + h] += change
    end
    return nothing
end

function inspect_loaded(source, output, advection)
    mesh = Main.TwoPhaseAMR.BlockTreeMesh(base_mesh=(64,64),
        block_size=(64,64), periodic=(true,true))
    scalar = Main.TwoPhaseAMR.CellField(mesh, Float64)
    increment = Main.TwoPhaseAMR.CellField(mesh, Float64)
    workspace = Main.TwoPhaseAMR.AdvectionWorkspace(scalar)
    velocity = Main.TwoPhaseAMR.FaceVelocity(mesh, (x,y,t,axis) ->
        axis == 1 ? 0.7 : -0.3)
    signature = Tuple{typeof(workspace), typeof(scalar), typeof(increment),
                      Int, Int, Int}
    actual_signature = Tuple{typeof(scalar), typeof(velocity), Float64,
                             typeof(workspace)}
    sum_signature = Tuple{typeof(workspace), Int, Int, Int, Int, Int,
                          Int, Bool, Type{Float64}}
    sum_method = which(Main.TwoPhaseAMR._cell_advection_sum, sum_signature)
    advect_method = which(Main.TwoPhaseAMR.advect!, actual_signature)
    for method in (sum_method, advect_method)
        loaded_file = string(method.file)
        isfile(loaded_file) && realpath(loaded_file) == realpath(advection) ||
            error("generated-code method did not load copied advection source")
    end
    mkpath(output)
    open(joinpath(output, "representative-gather-typed.txt"), "w") do io
        show(io, code_typed(representative_gather_columns!, signature;
                            optimize=true))
    end
    open(joinpath(output, "representative-gather-llvm.txt"), "w") do io
        code_llvm(io, representative_gather_columns!, signature; optimize=true)
    end
    open(joinpath(output, "representative-gather-native.txt"), "w") do io
        code_native(io, representative_gather_columns!, signature;
                    syntax=:intel)
    end
    open(joinpath(output, "production-advect-typed.txt"), "w") do io
        show(io, code_typed(Main.TwoPhaseAMR.advect!, actual_signature;
                            optimize=true))
    end
    open(joinpath(output, "production-advect-llvm.txt"), "w") do io
        code_llvm(io, Main.TwoPhaseAMR.advect!, actual_signature;
                  optimize=true)
    end
    orders = Dict{String,Int}()
    for j in 2:63, i in 2:63
        cell = (j - 1)*64 + i
        key = string(@inbounds workspace.interior_order[cell]; base=16, pad=2)
        orders[key] = get(orders,key,0) + 1
    end
    receipt = Dict(
        "source" => source,
        "advection_sha256" => bytes2hex(sha256(read(advection))),
        "julia_version" => string(VERSION),
        "cpu_name" => Sys.CPU_NAME,
        "workspace_type" => string(typeof(workspace)),
        "scalar_type" => string(typeof(scalar)),
        "velocity_type" => string(typeof(velocity)),
        "loop_signature" => string(signature),
        "production_signature" => string(actual_signature),
        "sum_method_file" => string(sum_method.file),
        "sum_method_line" => sum_method.line,
        "advect_method_file" => string(advect_method.file),
        "advect_method_line" => advect_method.line,
        "order_histogram_hex" => orders,
        "scope" => "read-only generated-code inspection; representative copied gather body is not the exact threaded production closure, and production advect entry is separate",
    )
    open(joinpath(output, "receipt.toml"), "w") do io
        TOML.print(io, receipt)
    end
    println("saved generated code and runtime-order receipt to $output")
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: inspect_gather.jl SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY")
        return
    end
    length(args) == 2 || error("usage: inspect_gather.jl SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY")
    source, output = abspath.(args)
    !ispath(output) || error("output already exists")
    path = joinpath(source, "src", "TwoPhaseAMR.jl")
    advection = joinpath(source, "src", "operators", "advection", "cell_advection.jl")
    isfile(path) && isfile(advection) || error("source is incomplete")
    Base.include(Main, path)
    Base.invokelatest(inspect_loaded, source, output, advection)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
