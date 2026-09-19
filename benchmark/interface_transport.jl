SUITE["interface transport"] = BenchmarkGroup()

function transport_fixture(mixed::Bool, method)
    mesh = benchmark_mesh(mixed)
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
    liquid = FaceVelocity(mesh, benchmark_velocity)
    if isnothing(method)
        workspace = WYAdvectionWorkspace(interface)
        dt = stable_timestep(liquid, workspace; cfl = 0.1)
        return (; interface, velocity = liquid, workspace, dt)
    end
    gas = FaceVelocity(
        mesh,
        (x, y, time, axis) ->
        axis == 1 ? -0.04sinpi(2x) * cospi(y) :
            0.08cospi(2x) * sinpi(y),
    )
    velocity = (liquid, gas)
    workspace = WYAdvectionWorkspace(
        interface, velocity; interface_transport = method,
    )
    dt = stable_timestep(velocity, workspace; cfl = 0.1)
    return (; interface, velocity, workspace, dt)
end

transport_step!(case) = advect!(
    case.interface, case.velocity, case.dt, case.workspace,
)

for (topology, mixed) in ("uniform" => false, "mixed" => true)
    SUITE["interface transport"]["one velocity / $topology"] =
        @benchmarkable transport_step!(case) setup = (
        case = transport_fixture($mixed, nothing)
    ) evals = 1
    SUITE["interface transport"]["frozen two velocity / $topology"] =
        @benchmarkable transport_step!(case) setup = (
        case = transport_fixture($mixed, FrozenInterfaceFluxes())
    ) evals = 1
    SUITE["interface transport"]["projected two velocity / $topology"] =
        @benchmarkable transport_step!(case) setup = (
        case = transport_fixture($mixed, ProjectedLiquidVelocity())
    ) evals = 1
end
