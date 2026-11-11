SUITE["interface transport"] = BenchmarkGroup()

function transport_fixture(mixed::Bool; two_velocity = false)
    mesh = benchmark_mesh(mixed)
    interface = InterfaceField(mesh, benchmark_levelset; lipschitz = 2)
    liquid = FaceVelocity(mesh, benchmark_velocity)
    if !two_velocity
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
    workspace = WYAdvectionWorkspace(interface, velocity)
    dt = stable_timestep(velocity, workspace; cfl = 0.1)
    return (; interface, velocity, workspace, dt)
end

transport_step!(case) = advect!(
    case.interface, case.velocity, case.dt, case.workspace,
)

for (topology, mixed) in ("uniform" => false, "mixed" => true)
    SUITE["interface transport"]["one velocity / $topology"] =
        @benchmarkable transport_step!(case) setup = (
        case = transport_fixture($mixed)
    ) evals = 1
    SUITE["interface transport"]["frozen two velocity / $topology"] =
        @benchmarkable transport_step!(case) setup = (
        case = transport_fixture($mixed; two_velocity = true)
    ) evals = 1
end
