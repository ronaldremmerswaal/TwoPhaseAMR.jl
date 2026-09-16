using BenchmarkTools
using TwoPhaseAMR

function _estimate(trial)
    value = median(trial)
    return (; time = value.time, memory = value.memory, allocs = value.allocs)
end

function _print_comparison(label, single, two_phase)
    a, b = _estimate(single), _estimate(two_phase)
    return println(
        rpad(label, 25),
        lpad(round(a.time / 1.0e6; digits = 3), 10), " ms  ",
        lpad(round(b.time / 1.0e6; digits = 3), 10), " ms  ",
        lpad(round(b.time / a.time; digits = 2), 8), "x  ",
        lpad(a.memory, 12), " B  ", lpad(b.memory, 12), " B"
    )
end

function _print_extension(label, trial)
    value = _estimate(trial)
    return println(
        rpad(label, 25), lpad("-", 10), "     ",
        lpad(round(value.time / 1.0e6; digits = 3), 10), " ms  ",
        lpad("-", 8), "   ", lpad("-", 12), "    ",
        lpad(value.memory, 12), " B"
    )
end

function two_phase_benchmark_case(mesh; samples = 20)
    levelset = (x, y) -> hypot(x - 0.5, y - 0.58) - 0.2
    interface_template = InterfaceField(mesh, levelset; lipschitz = 1)
    cell_template = CellField(
        mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y)
    )
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            0.15sinpi(2x)^2 * sinpi(2y) :
            -0.15sinpi(2x) * sinpi(2y)^2
    )
    phase_work = WYAdvectionWorkspace(interface_template)
    dt = stable_timestep(velocity, phase_work; cfl = 0.2)
    cell_dt = stable_timestep(
        velocity, AdvectionWorkspace(cell_template); cfl = 0.2
    )

    cell_advection = @benchmark advect!(scalar, $velocity, $cell_dt, work) setup = (
        scalar = copy($cell_template);
        work = AdvectionWorkspace(scalar)
    ) evals = 1 samples = samples
    interface_advection = @benchmark advect!(
        interface, $velocity, $dt, work
    ) setup = (
        interface = copy($interface_template);
        work = WYAdvectionWorkspace(interface)
    ) evals = 1 samples = samples
    reconstruction = @benchmark reconstruct!(interface) setup = (
        interface = copy($interface_template)
    ) evals = 1 samples = samples

    single_momentum = @benchmark advect!(
        destination, $velocity, $dt, work; scheme = MC()
    ) setup = (
        destination = copy($velocity);
        work = StaggeredVelocityAdvectionWorkspace(destination)
    ) evals = 1 samples = samples
    phase_momentum = @benchmark advect!(
        interface, $velocity, $dt, work;
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum_work, scheme = MC()
    ) setup = (
        interface = copy($interface_template);
        destination = FaceVelocity($mesh);
        work = WYAdvectionWorkspace(interface);
        momentum_work = MomentumAdvectionWorkspace(interface)
    ) evals = 1 samples = samples

    constant_diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.01, boundary = FreeSlip()
    )
    variable_diffusion = StaggeredDiffusionOperator(
        interface_template; viscosity = (0.01, 1.0e-4),
        boundary = FreeSlip()
    )
    single_diffusion_setup = @benchmark StaggeredDiffusionOperator(
        $mesh; viscosity = 0.01, boundary = FreeSlip()
    ) evals = 1 samples = samples
    phase_diffusion_setup = @benchmark StaggeredDiffusionOperator(
        $interface_template; viscosity = (0.01, 1.0e-4),
        boundary = FreeSlip()
    ) evals = 1 samples = samples
    single_diffusion_apply = @benchmark diffusion!(
        destination, $velocity, $constant_diffusion
    ) setup = (destination = FaceVelocity($mesh)) evals = 1 samples = samples
    phase_diffusion_apply = @benchmark diffusion!(
        destination, $velocity, $variable_diffusion
    ) setup = (destination = FaceVelocity($mesh)) evals = 1 samples = samples
    phase_diffusion_refresh = @benchmark TwoPhaseAMR._update_dynamic_viscosity!(
        operator, $interface_template, (0.01, 1.0e-4)
    ) setup = (
        operator = StaggeredDiffusionOperator(
            $interface_template; viscosity = (0.01, 1.0e-4),
            boundary = FreeSlip()
        )
    ) evals = 1 samples = samples
    diffusion_dt = min(dt, 0.01)
    single_diffusion_solve = @benchmark implicit_midpoint_diffusion!(
        destination, $velocity, $velocity, $constant_diffusion,
        $diffusion_dt, solver, work;
        rtol = 1.0e-9, max_cycles = 200
    ) setup = (
        destination = copy($velocity);
        work = FaceVelocity($mesh);
        elliptic = StaggeredEllipticOperator(
            $constant_diffusion;
            diffusion = $diffusion_dt * $constant_diffusion.viscosity / 2
        );
        solver = StaggeredEllipticSolver(
            elliptic; preconditioner = :geometric
        )
    ) evals = 1 samples = samples
    phase_density = mixture_density(
        interface_template, (1000.0, 1.0)
    )
    phase_diffusion_solve = @benchmark implicit_midpoint_diffusion!(
        destination, $velocity, $variable_diffusion,
        $diffusion_dt, solver;
        old_velocity = $velocity, density = $phase_density,
        rtol = 1.0e-9, max_cycles = 200
    ) setup = (
        destination = copy($velocity);
        solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
            $variable_diffusion, $phase_density, $diffusion_dt;
            krylov = TwoPhaseAMR.Krylov.FgmresWorkspace,
            preconditioner = :geometric
        )
    ) evals = 1 samples = samples

    rhs = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    single_poisson = PoissonOperator(mesh)
    phase_poisson = PoissonOperator(
        interface_template; density = (1000.0, 1.0)
    )
    single_poisson_setup = @benchmark PoissonOperator($mesh) evals = 1 samples = samples
    phase_poisson_setup = @benchmark PoissonOperator(
        $interface_template; density = (1000.0, 1.0)
    ) evals = 1 samples = samples
    phase_poisson_refresh = @benchmark PoissonOperator(
        $interface_template; density = (1000.0, 1.0), template = operator
    ) setup = (
        operator = PoissonOperator(
            $interface_template; density = (1000.0, 1.0)
        )
    ) evals = 1 samples = samples
    single_poisson_apply = @benchmark apply!(
        destination, $single_poisson, $rhs, work
    ) setup = (
        destination = CellField($mesh);
        work = EllipticWorkspace($single_poisson)
    ) evals = 1 samples = samples
    phase_poisson_apply = @benchmark apply!(
        destination, $phase_poisson, $rhs, work
    ) setup = (
        destination = CellField($mesh);
        work = EllipticWorkspace($phase_poisson)
    ) evals = 1 samples = samples
    single_poisson_solve = @benchmark solve!(
        solution, solver, $rhs; rtol = 1.0e-9, max_cycles = 100
    ) setup = (
        solution = CellField($mesh);
        solver = EllipticSolver($single_poisson; preconditioner = :geometric)
    ) evals = 1 samples = samples
    phase_poisson_solve = @benchmark solve!(
        solution, solver, $rhs; rtol = 1.0e-9, max_cycles = 100
    ) setup = (
        solution = CellField($mesh);
        solver = EllipticSolver(
            $phase_poisson; krylov = TwoPhaseAMR.Krylov.FgmresWorkspace,
            preconditioner = :geometric
        )
    ) evals = 1 samples = samples

    return (;
        cell_advection, interface_advection, reconstruction,
        single_momentum, phase_momentum,
        single_diffusion_setup, phase_diffusion_setup,
        single_diffusion_apply, phase_diffusion_apply,
        phase_diffusion_refresh, single_diffusion_solve,
        phase_diffusion_solve,
        single_poisson_setup, phase_poisson_setup,
        phase_poisson_refresh,
        single_poisson_apply, phase_poisson_apply,
        single_poisson_solve, phase_poisson_solve,
    )
end

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)
benchmark_samples = parse(Int, get(ENV, "TWOPHASE_BENCHMARK_SAMPLES", "20"))
println(
    rpad("kernel", 25), lpad("single", 10), "     ",
    lpad("two-phase", 10), "    ratio  ",
    lpad("single bytes", 12), "  ", lpad("two-phase bytes", 12)
)

for (name, mesh) in (
        "uniform" => BlockTreeMesh(block_size = (24, 24)),
        "mixed" => BlockTreeMesh(
            block_size = (12, 12), max_level = 2,
            refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0)
        ),
    )
    println("\n", name, ": ", ncells(mesh), " cells, ", length(mesh.faces), " faces")
    result = two_phase_benchmark_case(mesh; samples = benchmark_samples)
    _print_comparison(
        "cell/interface advection",
        result.cell_advection, result.interface_advection
    )
    _print_extension("interface reconstruction", result.reconstruction)
    _print_comparison(
        "momentum advection", result.single_momentum, result.phase_momentum
    )
    _print_comparison(
        "diffusion setup",
        result.single_diffusion_setup, result.phase_diffusion_setup
    )
    _print_comparison(
        "diffusion apply",
        result.single_diffusion_apply, result.phase_diffusion_apply
    )
    _print_extension(
        "diffusion coeff refresh", result.phase_diffusion_refresh
    )
    _print_comparison(
        "implicit diffusion solve", result.single_diffusion_solve,
        result.phase_diffusion_solve
    )
    _print_comparison(
        "Poisson setup", result.single_poisson_setup,
        result.phase_poisson_setup
    )
    _print_extension("Poisson coeff refresh", result.phase_poisson_refresh)
    _print_comparison(
        "Poisson apply", result.single_poisson_apply,
        result.phase_poisson_apply
    )
    _print_comparison(
        "Poisson solve", result.single_poisson_solve,
        result.phase_poisson_solve
    )
end
