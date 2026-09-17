using TwoPhaseAMR

const MAX_ITERATIONS = 1000
const RTOL = 1.0e-9

function graded_mesh(max_level)
    return BlockTreeMesh(
        block_size = (8, 8), max_level = max_level,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (
            key.level < max_level &&
                origin[1] < 0.5 && origin[2] < 0.5
        ),
    )
end

function zero_velocity!(velocity)
    fill!(velocity.interior, 0)
    fill!(velocity.boundary, 0)
    return velocity
end

function median_sample(values)
    ordered = sort(values)
    return ordered[cld(length(ordered), 2)]
end

function benchmark_solve(solution, solver, rhs, name; samples)
    zero_velocity!(solution)
    solve!(
        solution, solver, rhs;
        rtol = RTOL, max_cycles = MAX_ITERATIONS
    )

    times = Float64[]
    stats = nothing
    for _ in 1:samples
        zero_velocity!(solution)
        elapsed = @elapsed stats = solve!(
            solution, solver, rhs;
            rtol = RTOL, max_cycles = MAX_ITERATIONS
        )
        stats.converged || error(
            "$name failed to converge " *
                "(rtol=$(RTOL), max_cycles=$(MAX_ITERATIONS), " *
                "stats=$(repr(stats)))",
        )
        push!(times, elapsed)
    end
    zero_velocity!(solution)
    solve!(
        solution, solver, rhs;
        rtol = RTOL, max_cycles = MAX_ITERATIONS
    )
    zero_velocity!(solution)
    bytes = @allocated solve!(
        solution, solver, rhs;
        rtol = RTOL, max_cycles = MAX_ITERATIONS
    )
    return (
        time = median_sample(times),
        iterations = (stats.x.cycles, stats.y.cycles), bytes,
        residual = max(stats.x.final_linf, stats.y.final_linf),
    )
end

println("| Max tree level | Staggered unknowns | Relative stiffness | Preconditioner | BiCGSTAB x/y iterations | Median time (ms) | Allocated bytes | LInf residual |")
println("|---:|---:|---:|---|---:|---:|---:|---:|")
for max_level in (2, 4, 6)
    mesh = graded_mesh(max_level)
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 1.0, boundary = FreeSlip()
    )
    maximum_rate = max(
        diffusion.x.maximum_row_sum,
        diffusion.y.maximum_row_sum
    )
    rhs = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) + 0.1cospi(6y) :
            cospi(2x) * sinpi(2y) + 0.1sinpi(6x)
    )
    solution = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    unknowns = length(diffusion.x.volumes) + length(diffusion.y.volumes)
    samples = max_level == 6 ? 3 : 7
    for relative_stiffness in (0.45, 4.0, 40.0)
        operator = StaggeredEllipticOperator(
            diffusion; diffusion = relative_stiffness / maximum_rate
        )
        for preconditioner in (nothing, :tree_geometric)
            solver = StaggeredEllipticSolver(
                operator; coarse_size = 32, preconditioner
            )
            name = isnothing(preconditioner) ? :none : preconditioner
            result = benchmark_solve(
                solution, solver, rhs, name; samples
            )
            iterations = "$(result.iterations[1])/$(result.iterations[2])"
            println(
                "| $max_level | $unknowns | $relative_stiffness | " *
                    "$name | $iterations | " *
                    "$(round(1.0e3result.time; digits = 3)) | " *
                    "$(result.bytes) | $(result.residual) |"
            )
        end
    end
end
