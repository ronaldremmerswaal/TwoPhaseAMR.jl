using Statistics: median
import AlgebraicMultigrid
import IncompleteLU
import Krylov
import LimitedLDLFactorizations
import TwoPhaseAMR as TP

const SAMPLES = 5
const RTOL = 1.0e-9

function graded_mesh()
    return TP.BlockTreeMesh(
        block_size = (16, 16), max_level = 3,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < 3 && origin[1] < 0.5 && origin[2] < 0.5),
    )
end

function clear!(field::TP.CellField)
    for block in eachindex(field.mesh.blocks)
        fill!(TP.interior(field, block), 0)
    end
    return field
end

function clear!(field::TP.FaceVelocity)
    fill!(field.interior, 0)
    fill!(field.boundary, 0)
    return field
end

function warm_median!(solution, solver, right_hand_side; max_cycles = 1000)
    clear!(solution)
    TP.solve!(solution, solver, right_hand_side; rtol = RTOL, max_cycles)
    times = Float64[]
    stats = nothing
    for _ in 1:SAMPLES
        clear!(solution)
        push!(
            times, @elapsed stats = TP.solve!(
                solution, solver, right_hand_side; rtol = RTOL, max_cycles
            )
        )
        stats.converged || error("linear solve did not converge")
    end
    return median(times), stats
end

function pressure_comparison(mesh)
    operator = TP.PoissonOperator(mesh)
    rhs = TP.CellField(mesh)
    TP.initialize!(
        rhs, (x, y) ->
        8pi^2 * sinpi(2x) * cospi(2y) +
            2pi^2 * sinpi(2x + 2y) + 0.2sinpi(6x - 4y)
    )
    solution = TP.CellField(mesh)
    cases = (
        (
            name = :none, krylov = Krylov.CgWorkspace, preconditioner = nothing,
            options = (;),
        ),
        (
            name = :geometric, krylov = Krylov.CgWorkspace,
            preconditioner = :geometric, options = (;),
        ),
        (
            name = :amg, krylov = Krylov.CgWorkspace, preconditioner = :amg,
            options = (;),
        ),
        (
            name = :ichol, krylov = Krylov.CgWorkspace, preconditioner = :ichol,
            options = (; memory = 4, shift = 1.0e-8),
        ),
        # Generic ILU is nonsymmetric, so this row must not use CG.
        (
            name = :ilu, krylov = Krylov.BicgstabWorkspace, preconditioner = :ilu,
            options = (; drop_tolerance = 1.0e-3),
        ),
    )

    rows = []
    for case in cases
        # Compile package-specific setup before measuring it.
        TP.EllipticSolver(
            operator; krylov = case.krylov,
            preconditioner = case.preconditioner,
            preconditioner_options = case.options, coarse_size = 32
        )
        GC.gc()
        setup = @elapsed solver = TP.EllipticSolver(
            operator; krylov = case.krylov,
            preconditioner = case.preconditioner,
            preconditioner_options = case.options, coarse_size = 32
        )
        elapsed, stats = warm_median!(solution, solver, rhs)
        bytes = isnothing(solver.preconditioner) ? 0 :
            Base.summarysize(solver.preconditioner)
        push!(
            rows, (
                case.name, stats.method, setup, elapsed,
                stats.cycles, stats.final_linf, bytes,
            )
        )
    end
    return rows
end

function diffusion_comparison(mesh)
    diffusion = TP.StaggeredDiffusionOperator(
        mesh; viscosity = 1.0, boundary = TP.FreeSlip()
    )
    maximum_rate = max(
        diffusion.x.maximum_row_sum,
        diffusion.y.maximum_row_sum
    )
    operator = TP.StaggeredEllipticOperator(
        diffusion; diffusion = 40 / maximum_rate
    )
    rhs = TP.FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) + 0.1cospi(6y) :
            cospi(2x) * sinpi(2y) + 0.1sinpi(6x)
    )
    solution = TP.FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    cases = (
        (name = :none, preconditioner = nothing, options = (;)),
        (name = :geometric, preconditioner = :geometric, options = (;)),
        (name = :amg, preconditioner = :amg, options = (;)),
        (
            name = :ichol, preconditioner = :ichol,
            options = (; memory = 4, shift = 1.0e-8),
        ),
        (
            name = :ilu, preconditioner = :ilu,
            options = (; drop_tolerance = 1.0e-3),
        ),
    )

    rows = []
    for case in cases
        TP.StaggeredEllipticSolver(
            operator; krylov = Krylov.BicgstabWorkspace,
            preconditioner = case.preconditioner,
            preconditioner_options = case.options, coarse_size = 32
        )
        GC.gc()
        setup = @elapsed solver = TP.StaggeredEllipticSolver(
            operator; krylov = Krylov.BicgstabWorkspace,
            preconditioner = case.preconditioner,
            preconditioner_options = case.options, coarse_size = 32
        )
        elapsed, stats = warm_median!(solution, solver, rhs)
        bytes = isnothing(solver.x.preconditioner) ? 0 :
            Base.summarysize(solver.x.preconditioner) +
            Base.summarysize(solver.y.preconditioner)
        push!(
            rows, (
                case.name, :bicgstab, setup, elapsed,
                (stats.x.cycles, stats.y.cycles),
                max(stats.x.final_linf, stats.y.final_linf), bytes,
            )
        )
    end
    return rows
end

function diffusion_krylov_comparison(mesh)
    diffusion = TP.StaggeredDiffusionOperator(
        mesh; viscosity = 1.0, boundary = TP.FreeSlip()
    )
    maximum_rate = max(
        diffusion.x.maximum_row_sum,
        diffusion.y.maximum_row_sum
    )
    operator = TP.StaggeredEllipticOperator(
        diffusion; diffusion = 40 / maximum_rate
    )
    rhs = TP.FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) + 0.1cospi(6y) :
            cospi(2x) * sinpi(2y) + 0.1sinpi(6x)
    )
    solution = TP.FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    cases = (
        (name = :bicgstab, krylov = Krylov.BicgstabWorkspace, options = (;)),
        (
            name = :gmres,
            krylov = (matrix, rhs) -> Krylov.GmresWorkspace(
                matrix, rhs; memory = 20
            ),
            options = (; restart = true, reorthogonalization = true),
        ),
        (
            name = :fgmres,
            krylov = (matrix, rhs) -> Krylov.FgmresWorkspace(
                matrix, rhs; memory = 20
            ),
            options = (; restart = true, reorthogonalization = true),
        ),
    )

    rows = []
    for case in cases
        TP.StaggeredEllipticSolver(
            operator; krylov = case.krylov, krylov_options = case.options,
            preconditioner = :geometric, coarse_size = 32
        )
        GC.gc()
        setup = @elapsed solver = TP.StaggeredEllipticSolver(
            operator; krylov = case.krylov, krylov_options = case.options,
            preconditioner = :geometric, coarse_size = 32
        )
        elapsed, stats = warm_median!(solution, solver, rhs)
        workspace_bytes = Base.summarysize(solver.x.krylov) +
            Base.summarysize(solver.y.krylov)
        push!(
            rows, (
                case.name, setup, elapsed,
                (stats.x.cycles, stats.y.cycles),
                max(stats.x.final_linf, stats.y.final_linf),
                workspace_bytes,
            )
        )
    end
    return rows
end

mesh = graded_mesh()
println("Mesh: $(TP.ncells(mesh)) cells, $(length(mesh.faces)) owned faces")

println("\nPressure Poisson")
println("| Preconditioner | Krylov | Setup (ms) | Warm solve (ms) | Iterations | LInf residual | Stored bytes |")
println("|---|---|---:|---:|---:|---:|---:|")
for (name, method, setup, elapsed, iterations, residual, bytes) in
    pressure_comparison(mesh)
    println(
        "| $name | $method | $(round(1.0e3setup; digits = 3)) | " *
            "$(round(1.0e3elapsed; digits = 3)) | $iterations | " *
            "$(round(residual; sigdigits = 4)) | $bytes |"
    )
end

println("\nStaggered implicit diffusion (relative stiffness 40)")
println("| Preconditioner | Krylov | Setup (ms) | Warm solve (ms) | x/y iterations | LInf residual | Stored bytes |")
println("|---|---|---:|---:|---:|---:|---:|")
for (name, method, setup, elapsed, iterations, residual, bytes) in
    diffusion_comparison(mesh)
    println(
        "| $name | $method | $(round(1.0e3setup; digits = 3)) | " *
            "$(round(1.0e3elapsed; digits = 3)) | " *
            "$(iterations[1])/$(iterations[2]) | " *
            "$(round(residual; sigdigits = 4)) | $bytes |"
    )
end
println(
    "\nNamed AMG, ILU, and incomplete-Cholesky diffusion " *
        "preconditioners are built from the symmetric stable core while " *
        "BiCGSTAB still applies the exact corrected operator. ILU pressure " *
        "uses BiCGSTAB because generic ILU is not an SPD preconditioner for CG."
)

println("\nStaggered Krylov methods with geometric multigrid")
println("| Krylov | Setup (ms) | Warm solve (ms) | x/y iterations | LInf residual | Workspace bytes |")
println("|---|---:|---:|---:|---:|---:|")
for (method, setup, elapsed, iterations, residual, bytes) in
    diffusion_krylov_comparison(mesh)
    println(
        "| $method | $(round(1.0e3setup; digits = 3)) | " *
            "$(round(1.0e3elapsed; digits = 3)) | " *
            "$(iterations[1])/$(iterations[2]) | " *
            "$(round(residual; sigdigits = 4)) | $bytes |"
    )
end
