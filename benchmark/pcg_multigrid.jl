using TwoPhaseAMR

function graded_mesh(max_level)
    return BlockTreeMesh(
        block_size=(8, 8), max_level=max_level,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level < max_level &&
             origin[1] < 0.5 && origin[2] < 0.5),
    )
end

function zero_cells!(field)
    for block in eachindex(field.mesh.blocks)
        fill!(interior(field, block), 0)
    end
    return field
end

function median_sample(values)
    ordered = sort(values)
    return ordered[cld(length(ordered), 2)]
end

function benchmark_solve(operator, rhs, method; samples=7)
    solver = GeometricMultigridSolver(
        operator; coarse_size=32, method)
    solution = CellField(operator.mesh)
    solve!(solution, solver, rhs; rtol=1e-10, max_cycles=100)

    times = Float64[]
    cycles = Int[]
    for _ in 1:samples
        zero_cells!(solution)
        elapsed = @elapsed stats = solve!(
            solution, solver, rhs; rtol=1e-10, max_cycles=100)
        stats.converged || error("$method failed to converge")
        push!(times, elapsed)
        push!(cycles, stats.cycles)
    end
    zero_cells!(solution)
    solve!(solution, solver, rhs; rtol=1e-10, max_cycles=100)
    zero_cells!(solution)
    bytes = @allocated solve!(
        solution, solver, rhs; rtol=1e-10, max_cycles=100)
    return (time=median_sample(times), cycles=last(cycles), bytes,
            residual=solver.stats.final_linf,
            hierarchy_levels=length(solver.levels))
end

println("| Max tree level | Cells | Hierarchy levels | Method | Iterations | Median time (ms) | Allocated bytes | LInf residual |")
println("|---:|---:|---:|---|---:|---:|---:|---:|")
for max_level in (2, 4, 6, 7)
    mesh = graded_mesh(max_level)
    rhs = CellField(mesh)
    initialize!(rhs, (x, y) ->
        8pi^2 * sinpi(2x) * cospi(2y) +
        2pi^2 * sinpi(2x + 2y))
    operator = PoissonOperator(mesh)
    samples = max_level == 7 ? 3 : max_level == 6 ? 5 : 7
    for method in (:multigrid, :pcg)
        result = benchmark_solve(operator, rhs, method; samples)
        println("| $max_level | $(ncells(mesh)) | " *
                "$(result.hierarchy_levels) | $method | " *
                "$(result.cycles) | $(round(1e3result.time; digits=3)) | " *
                "$(result.bytes) | $(result.residual) |")
    end
end
