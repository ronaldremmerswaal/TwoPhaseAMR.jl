using Random
using TwoPhaseAMR

"""
    run_example(; reynolds=400, max_level=4, final_time=10,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run freely decaying periodic turbulence with pressure-driven AMR. Pass
`start_studio=studio` after loading TwoPhaseAMRStudio to follow the run live.
`recorder_options` overrides the recording schedule; other keywords are passed
to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 400.0,
        max_level = 4,
        final_time = 10.0,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    rng = MersenneTwister(20250308)
    modes = [
        (
            kx = kx, ky = ky,
            amplitude = exp(-0.5 * ((hypot(kx, ky) - 6) / 2)^2) *
                randn(rng) / hypot(kx, ky),
            phase = 2pi * rand(rng),
        )
            for kx in 0:9 for ky in -9:9
            if !(kx == 0 && ky <= 0) && 3 <= hypot(kx, ky) <= 9
    ]
    velocity = function (x, y, time, axis)
        return sum(modes; init = 0.0) do mode
            sine = sin(mode.kx * x + mode.ky * y + mode.phase)
            axis == 1 ? -mode.amplitude * mode.ky * sine :
                mode.amplitude * mode.kx * sine
        end
    end

    mesh = BlockTreeMesh(
        block_size = (8, 8), extent = (2pi, 2pi),
        max_level = max_level,
        refine = (key, origin, extent) -> key.level < max_level,
        adaptation = MeshAdaptationOptions(
            minimum_level = min(2, max_level), update_interval = 5,
            coarsen_delay = 25,
        ),
    )
    pressure = CellField(mesh)
    refinement = InterpolationErrorIndicator(
        pressure; E_refine = 0.6, E_coarsen = 0.2,
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = inv(reynolds),
    )
    flow_options = (;
        velocity, pressure, boundary = FreeSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(), CellCount(),
        ),
        numerics = NavierStokesOptions(
            advection = AdvectionOptions(scheme = LaxWendroff()),
        ),
        adaptivity = Adaptivity(criteria = (refinement,)),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    projection = project!(flow)
    projection.converged || error("initial pressure projection did not converge")
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    recorder = Recorder(
        fields; diagnostics = flow,
        merge(
            (;
                keyframes = EquispacedTimes(21),
                preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2),
            ),
            recorder_options,
        )...,
    )
    driver = isnothing(start_studio) ? nothing : TwoPhaseAMR.StudioDriver(flow, recorder)
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    result = try
        isnothing(driver) ? simulate!(flow, final_time; observer = recorder) :
            simulate!(driver, final_time)
    catch
        isnothing(app) || close(app)
        rethrow()
    end
    return (; flow, result, recorder, app)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; start_studio = studio)
    else
        run = run_example()
    end
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
    if !isnothing(run.app)
        try
            wait(run.app)
        finally
            close(run.app)
        end
    end
end
