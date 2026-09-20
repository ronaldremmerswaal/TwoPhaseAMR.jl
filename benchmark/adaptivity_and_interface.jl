SUITE["mesh and interface"] = BenchmarkGroup()
SUITE["adaptivity"] = BenchmarkGroup()

SUITE["mesh and interface"]["uniform mesh setup"] =
    @benchmarkable benchmark_mesh(false) evals = 1
SUITE["mesh and interface"]["mixed mesh setup"] =
    @benchmarkable benchmark_mesh(true) evals = 1
SUITE["mesh and interface"]["level-set initialization"] =
    @benchmarkable InterfaceField(
    mesh, benchmark_levelset; lipschitz = 2,
) setup = (mesh = benchmark_mesh(true)) evals = 1
SUITE["mesh and interface"]["PLIC and height reconstruction"] =
    @benchmarkable reconstruct!(interface) setup = (
    mesh = benchmark_mesh(true);
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
) evals = 1
SUITE["mesh and interface"]["interface height"] =
    @benchmarkable interface_height(interface) setup = (
    mesh = benchmark_mesh(true);
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
) evals = 1

function adaptivity_fixture()
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (8, 8), max_level = 2,
        periodic = (true, false),
    )
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
    velocity = FaceVelocity(mesh, benchmark_velocity)
    criterion = InterfaceRefinementIndicator(interface; level = 2)
    workspace = AMRWorkspace(mesh, interface, velocity)
    return (; workspace, criterion)
end

SUITE["adaptivity"]["refine and transfer"] = @benchmarkable adapt!(
    case.workspace, case.criterion; step = 1, force = true,
) setup = (case = adaptivity_fixture()) evals = 1
