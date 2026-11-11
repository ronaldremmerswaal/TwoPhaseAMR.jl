SUITE["linear solvers"] = BenchmarkGroup()
SUITE["two-velocity projection"] = BenchmarkGroup()

function poisson_fixture(mixed::Bool)
    mesh = benchmark_mesh(mixed; periodic = (true, true))
    operator = PoissonOperator(mesh)
    right_hand_side = CellField(
        mesh, (x, y) -> sinpi(2x) * cospi(2y),
    )
    solution = CellField(mesh)
    workspace = EllipticWorkspace(operator)
    solver = EllipticSolver(
        operator; preconditioner = :tree_geometric, coarse_size = 16,
    )
    return (; operator, right_hand_side, solution, workspace, solver)
end

function apply_poisson!(case)
    return apply!(
        case.solution, case.operator, case.right_hand_side, case.workspace,
    )
end

function solve_poisson!(case)
    return solve!(
        case.solution, case.solver, case.right_hand_side;
        rtol = 1.0e-8, max_cycles = 200,
    )
end

for (topology, mixed) in ("uniform" => false, "mixed" => true)
    SUITE["linear solvers"]["Poisson apply / $topology"] =
        @benchmarkable apply_poisson!(case) setup = (
        case = poisson_fixture($mixed)
    ) evals = 1
    SUITE["linear solvers"]["Poisson solve / $topology"] =
        @benchmarkable solve_poisson!(case) setup = (
        case = poisson_fixture($mixed)
    ) evals = 1
end

function two_velocity_projection_fixture()
    mesh = benchmark_mesh(false)
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
    density = (1000.0, 1.0)
    solver = CMDGFMProjectionSolver(interface; density)
    workspace = CMDGFMProjectionWorkspace(solver)
    velocity = (
        FaceVelocity(mesh, benchmark_velocity),
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? -0.05 : 0.0,
        ),
    )
    pressure = CellField(mesh)
    return (; velocity, pressure, solver, workspace, dt = 1.0e-3)
end

function project_two_velocity!(case)
    return project!(
        case.velocity, case.pressure, case.solver, case.dt, case.workspace;
        tolerance = 1.0e-7, max_iterations = 500,
    )
end

SUITE["two-velocity projection"]["CMDGFM"] =
    @benchmarkable project_two_velocity!(case) setup = (
    case = two_velocity_projection_fixture()
) evals = 1
