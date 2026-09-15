using BenchmarkTools
using Printf: @printf
using TwoPhaseAMR

function mixed_mesh(block_size)
    return BlockTreeMesh(
        block_size = (block_size, block_size), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
end

function setup_trial(field, build_three_point_stencils)
    # The private false branch constructs the same conservative two-point AMR
    # graph and final schedule, but skips predecessor search. It exists to
    # isolate setup cost; production workspaces always include the stencils.
    return @benchmark TwoPhaseAMR._staggered_advection_workspace(
        $field, $build_three_point_stencils
    ) samples = 200 evals = 1 seconds = 10
end

function print_trial(label, trial)
    estimate = median(trial)
    @printf(
        "  %-12s %10.3f ms  %9.3f MiB  %8d allocations\n",
        label, estimate.time / 1.0e6, estimate.memory / 2.0^20,
        estimate.allocs
    )
    return estimate
end

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)
println("AMR staggered-advection workspace setup")
for block_size in (8, 16, 32)
    mesh = mixed_mesh(block_size)
    field = FaceField{1}(
        mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y)
    )
    two_point = setup_trial(field, false)
    three_point = setup_trial(field, true)
    println(
        "\nblock size ", block_size, " (", nblocks(mesh), " blocks, ",
        ncells(mesh), " cells, ", length(field.interior), " x-faces)"
    )
    two_estimate = print_trial("two-point", two_point)
    three_estimate = print_trial("three-point", three_point)
    @printf(
        "  %-12s %10.3fx time  %9.3fx allocated bytes\n",
        "overhead", three_estimate.time / two_estimate.time,
        three_estimate.memory / two_estimate.memory
    )
end
