SUITE["setup"] = BenchmarkGroup()
SUITE["stepping"] = BenchmarkGroup()

SUITE["setup"]["single phase"] = @benchmarkable single_phase_flow(false) evals = 1
SUITE["setup"]["one velocity"] = @benchmarkable two_phase_flow(
    false, OneVelocityModel(),
) evals = 1
SUITE["setup"]["two velocity"] = @benchmarkable two_phase_flow(
    false, TwoVelocityModel(),
) evals = 1

SUITE["stepping"]["single phase / uniform"] = @benchmarkable benchmark_step!(
    flow, dt,
) setup = (
    flow = single_phase_flow(false);
    dt = benchmark_dt(flow)
) evals = 1

SUITE["stepping"]["single phase / mixed"] = @benchmarkable benchmark_step!(
    flow, dt,
) setup = (
    flow = single_phase_flow(true);
    dt = benchmark_dt(flow)
) evals = 1

SUITE["stepping"]["one velocity / mixed"] = @benchmarkable benchmark_step!(
    flow, dt,
) setup = (
    flow = two_phase_flow(true, OneVelocityModel());
    dt = benchmark_dt(flow)
) evals = 1

SUITE["stepping"]["two velocity CMDGFM / mixed"] =
    @benchmarkable benchmark_step!(flow, dt) setup = (
    flow = two_phase_flow(true, TwoVelocityModel());
    dt = benchmark_dt(flow)
) evals = 1

SUITE["stepping"]["diagnostics / one velocity"] =
    @benchmarkable benchmark_step!(flow, dt) setup = (
    flow = two_phase_flow(
        false, OneVelocityModel();
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(),
        ),
    );
    dt = benchmark_dt(flow)
) evals = 1
