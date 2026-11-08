using Random
using TwoPhaseAMR

"""
    run_example(; reynolds=400, max_level=4, final_time=10,
                visualize=false, kwargs...)

Run freely decaying periodic turbulence with pressure-driven AMR. Additional
keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 400.0,
        max_level = 4,
        final_time = 10.0,
        visualize = false,
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
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_diagnostic!(fields, flow, :kinetic_energy)
        register_diagnostic!(fields, flow, :cells)
        viewer = TwoPhaseAMR.visualize(
            fields; field = :velocity_vorticity, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 25, live_dt = final_time / 1000,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    result = simulate!(
        flow, final_time; visualization = observer,
    )
    isnothing(visualization) || wait_visualization!(visualization.session)
    return (; flow, result, visualization)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    GLMakie.activate!()
    run = run_example(visualize = true)
    print_simulation_summary(run.flow, run.result)
    scene = visualization_figure(run.visualization.viewer).scene
    scene.events.window_open[] && wait(scene)
end
