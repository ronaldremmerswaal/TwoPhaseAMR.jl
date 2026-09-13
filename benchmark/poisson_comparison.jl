using BenchmarkTools
using Random
using Statistics
import IncompressibleNavierStokes as INS
import TwoPhaseAMR as TP

const GRID_SIZE = 64
const RELATIVE_TOLERANCE = 1e-10
const SAMPLES = 100

function norms(values)
    return (l2=sqrt(sum(abs2, values) / length(values)),
            linf=maximum(abs, values))
end

function two_phase_case(n, rhs_values)
    mesh = TP.BlockTreeMesh(block_size=(n, n))
    rhs = TP.CellField(mesh)
    copyto!(TP.interior(rhs, 1), rhs_values)
    pressure = TP.CellField(mesh)
    operator = TP.PoissonOperator(mesh)
    solver = TP.EllipticSolver(operator)
    pressure_interior = TP.interior(pressure, 1)

    fill!(pressure_interior, 0)
    TP.solve!(pressure, solver, rhs; rtol=RELATIVE_TOLERANCE)
    trial = @benchmark begin
        fill!($pressure_interior, 0)
        TP.solve!($pressure, $solver, $rhs; rtol=$RELATIVE_TOLERANCE)
    end samples=SAMPLES evals=1
    solve_stats = deepcopy(solver.stats)

    velocity = TP.FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y),
    )
    projected = copy(velocity)
    projection_pressure = TP.CellField(mesh)
    projection_workspace = TP.ProjectionWorkspace(solver)
    divergence = TP.CellField(mesh)
    TP.divergence!(divergence, projected)
    initial_divergence = norms(vec(TP.interior(divergence, 1)))
    TP.project!(projected, projection_pressure, solver, 1.0,
                projection_workspace; rtol=RELATIVE_TOLERANCE)
    TP.divergence!(divergence, projected)
    projected_divergence = norms(vec(TP.interior(divergence, 1)))
    return (; trial, stats=solve_stats, initial_divergence,
            projected_divergence)
end

function incompressible_navier_stokes_case(n, rhs_values)
    coordinate = LinRange(0.0, 1.0, n + 1)
    setup = INS.Setup(
        x=(coordinate, coordinate),
        boundary_conditions=(;
            u=((INS.PeriodicBC(), INS.PeriodicBC()),
               (INS.PeriodicBC(), INS.PeriodicBC())),
        ),
    )
    rhs = INS.scalarfield(setup)
    # IncompressibleNavierStokes solves the integrated finite-volume system.
    # TwoPhaseAMR accepts the corresponding strong-form right-hand side.
    rhs[setup.Ip] .= rhs_values ./ n^2
    pressure = copy(rhs)
    solver = INS.psolver_cg(
        setup; reltol=RELATIVE_TOLERANCE, abstol=0.0, maxiter=n^2)

    copyto!(pressure, rhs)
    INS.poisson!(solver, pressure)
    trial = @benchmark begin
        copyto!($pressure, $rhs)
        INS.poisson!($solver, $pressure)
    end samples=SAMPLES evals=1

    INS.apply_bc_p!(pressure, 0.0, setup)
    residual = INS.laplacian(pressure, setup)
    @. residual = (residual - rhs) * n^2
    residual_norm = norms(view(residual, setup.Ip))

    # Count one representative CG solve without adding a counter to the timed
    # closure. The reference API intentionally does not return iteration data.
    iterations = Ref(0)
    reference_preconditioner = INS.create_laplace_diag(setup)
    function counting_preconditioner(destination, source)
        iterations[] += 1
        reference_preconditioner(destination, source)
    end
    counting_solver = INS.psolver_cg(
        setup; reltol=RELATIVE_TOLERANCE, abstol=0.0, maxiter=n^2,
        preconditioner=counting_preconditioner)
    counting_pressure = copy(rhs)
    INS.poisson!(counting_solver, counting_pressure)

    velocity = INS.velocityfield(
        setup,
        (axis, x, y) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y);
        doproject=false,
    )
    projected = copy(velocity)
    projection_pressure = INS.scalarfield(setup)
    initial_divergence = norms(view(INS.divergence(projected, setup), setup.Ip))
    INS.project!(projected, setup; psolver=solver, p=projection_pressure)
    projected_divergence = norms(view(INS.divergence(projected, setup), setup.Ip))
    return (; trial, iterations=iterations[], residual_norm,
            initial_divergence, projected_divergence)
end

function print_trial(label, trial)
    estimate = median(trial)
    println(label)
    println("  median_ns: ", estimate.time)
    println("  memory_bytes: ", estimate.memory)
    println("  allocations: ", estimate.allocs)
end

println("Poisson comparison")
println("  Julia: ", VERSION)
println("  CPU: ", Sys.cpu_info()[1].model)
println("  threads: ", Threads.nthreads())
println("  grid: ", GRID_SIZE, " x ", GRID_SIZE)
println("  precision: Float64")
println("  boundaries: periodic x/y")
println("  relative tolerance: ", RELATIVE_TOLERANCE)
println("  initial guess: zero")

rng = Xoshiro(0x2a17)
rhs_values = randn(rng, GRID_SIZE, GRID_SIZE)
rhs_values .-= mean(rhs_values)
two_phase = two_phase_case(GRID_SIZE, rhs_values)
reference = incompressible_navier_stokes_case(GRID_SIZE, rhs_values)
print_trial("TwoPhaseAMR geometric multigrid", two_phase.trial)
println("  cycles: ", two_phase.stats.cycles)
println("  final_residual_l2: ", two_phase.stats.final_l2)
println("  final_residual_linf: ", two_phase.stats.final_linf)
println("  projected_divergence: ", two_phase.projected_divergence)
print_trial("IncompressibleNavierStokes preconditioned CG", reference.trial)
println("  iterations: ", reference.iterations)
println("  final_residual: ", reference.residual_norm)
println("  projected_divergence: ", reference.projected_divergence)
println("median_time_ratio: ",
        median(two_phase.trial).time / median(reference.trial).time)
