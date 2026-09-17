using BenchmarkTools
using LinearAlgebra: ldiv!
using TwoPhaseAMR

estimate(trial) = let value = median(trial)
    (; time_ms = value.time / 1.0e6, value.memory, value.allocs)
end

function cut_cell_benchmarks(; cells = 32, samples = 20)
    mesh = BlockTreeMesh(
        block_size = (cells, cells), periodic = (true, true),
    )
    levelset = (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2
    geometry = GeometryField(
        mesh, levelset; lipschitz = 1, atol = 1.0e-7,
    )
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x)^2 * sinpi(2y) :
            -sinpi(2x) * sinpi(2y)^2,
    )
    single_advection = StaggeredVelocityAdvectionWorkspace(velocity)
    cut_advection = StaggeredVelocityAdvectionWorkspace(velocity, geometry)
    destination = FaceVelocity(mesh)
    single_diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.01,
    )
    cut_diffusion = StaggeredDiffusionOperator(
        geometry; viscosity = 0.01,
    )
    single_poisson = PoissonOperator(mesh)
    cut_poisson = PoissonOperator(geometry)
    rhs = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    vector = [sin(0.3index) for index in eachindex(cut_poisson.volumes)]

    return (
        geometry_initialization = @benchmark GeometryField(
            $mesh, $levelset; lipschitz = 1, atol = 1.0e-7,
        ) evals = 1 samples = samples,
        geometry_refresh = @benchmark refresh_geometry!($geometry) evals = 1 samples = samples,
        single_advection = @benchmark convection!(
            $destination, $velocity, $velocity, 0.0, $single_advection;
            scheme = Centred(),
        ) evals = 1 samples = samples,
        cut_advection = @benchmark convection!(
            $destination, $velocity, $velocity, 0.0, $cut_advection;
            scheme = Centred(),
        ) evals = 1 samples = samples,
        single_diffusion_setup = @benchmark StaggeredDiffusionOperator(
            $mesh; viscosity = 0.01,
        ) evals = 1 samples = samples,
        cut_diffusion_setup = @benchmark StaggeredDiffusionOperator(
            $geometry; viscosity = 0.01,
        ) evals = 1 samples = samples,
        single_diffusion_apply = @benchmark diffusion!(
            $destination, $velocity, $single_diffusion,
        ) evals = 1 samples = samples,
        cut_diffusion_apply = @benchmark diffusion!(
            $destination, $velocity, $cut_diffusion,
        ) evals = 1 samples = samples,
        single_poisson_setup = @benchmark PoissonOperator($mesh) evals = 1 samples = samples,
        cut_poisson_setup = @benchmark PoissonOperator($geometry) evals = 1 samples = samples,
        single_poisson_apply = @benchmark apply!(
            output, $single_poisson, $rhs, work,
        ) setup = (
            output = CellField($mesh);
            work = EllipticWorkspace($single_poisson)
        ) evals = 1 samples = samples,
        cut_poisson_apply = @benchmark apply!(
            output, $cut_poisson, $rhs, work,
        ) setup = (
            output = CellField($mesh);
            work = EllipticWorkspace($cut_poisson)
        ) evals = 1 samples = samples,
        cut_pressure_vcycle = let solver = EllipticSolver(
                cut_poisson; preconditioner = :tree_geometric,
            )
            @benchmark ldiv!(output, preconditioner, $vector) setup = (
                preconditioner = $(solver.preconditioner);
                output = similar($vector)
            ) evals = 1 samples = samples
        end,
    )
end

samples = parse(Int, get(ENV, "CUTCELL_BENCHMARK_SAMPLES", "20"))
results = cut_cell_benchmarks(; samples)
for name in propertynames(results)
    println(rpad(string(name), 28), estimate(getproperty(results, name)))
end
