using TwoPhaseAMR

"""Third-order finite-depth Stokes-wave initial condition."""
function stokes3_initial_condition(;
        steepness = 0.55, depth = 0.5, gravity = 1.0,
    )
    k = 2pi
    chi = inv(tanh(k * depth))
    phi1 = inv(cosh(k * depth))
    phi2 = 3(chi^2 - 1)^2 / (8chi * cosh(2k * depth))
    phi3 = (chi^2 - 1) * (chi^2 + 3) * (9chi^2 - 13) /
        (64cosh(3k * depth))
    eta2 = chi * (3chi^2 - 1) / 4
    eta31 = -3(chi^4 - 3chi^2 + 3) / 8
    eta33 = 3(8chi^6 + (chi^2 - 1)^2) / 64
    frequency = sqrt(
        gravity * k *
            (1 + steepness^2 * (9(chi^2 - 1)^2 / 8 + chi^2)) / chi,
    )
    amplitude = steepness / k
    elevation(x) = amplitude * (
        cos(k * x) + steepness * eta2 * cos(2k * x) +
            steepness^2 * (
            eta31 * cos(k * x) + eta33 * cos(3k * x)
        )
    )
    levelset(x, y) = y - elevation(x)
    function liquid_velocity(x, y, time, axis)
        y > elevation(x) && return 0.0
        if axis == 1
            return amplitude * gravity / frequency * (
                k * phi1 * cosh(k * (y + depth)) * cos(k * x) +
                    2k * steepness * phi2 * cosh(2k * (y + depth)) *
                    cos(2k * x) +
                    3k * steepness^2 * phi3 * cosh(3k * (y + depth)) *
                    cos(3k * x)
            )
        end
        return amplitude * gravity / frequency * (
            k * phi1 * sinh(k * (y + depth)) * sin(k * x) +
                2k * steepness * phi2 * sinh(2k * (y + depth)) *
                sin(2k * x) +
                3k * steepness^2 * phi3 * sinh(3k * (y + depth)) *
                sin(3k * x)
        )
    end
    return (; levelset, liquid_velocity, elevation, frequency)
end

"""Construct the Stokes-wave flow and return it with the requested end time."""
function setup_stokes3(;
        steepness = 0.55,
        bond = 1.0e3,
        reynolds = 4.0e4,
        density_ratio = 850.0,
        max_level = 5,
        final_time = nothing,
        kwargs...,
    )
    depth = 0.5
    gravity = 1.0
    density = (1.0, inv(density_ratio))
    liquid_viscosity = sqrt(gravity) / reynolds
    dynamic_viscosity = (liquid_viscosity, liquid_viscosity / 51.15)
    surface_tension =
        (density[1] - density[2]) * gravity / (bond * (2pi)^2)
    initial = stokes3_initial_condition(; steepness, depth, gravity)

    mesh = BlockTreeMesh(
        block_size = (8, 8), origin = (-0.5, -depth),
        extent = (1.0, 1.0), periodic = (true, false),
        max_level = max_level,
        adaptation = MeshAdaptationOptions(coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, initial.levelset)
    liquid_velocity = FaceVelocity(mesh, initial.liquid_velocity)
    refinement = (
        InterfaceRefinementIndicator(interface; level = max_level, t_end = 0.01),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        VorticityRefinementIndicator(
            liquid_velocity; refine_tolerance = 0.2,
            coarsen_tolerance = 0.2 / 3, maximum_level = max_level,
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, liquid_velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, initial.levelset)
        set_velocities!(liquid_velocity, initial.liquid_velocity)
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = (0.0, -gravity),
    )
    velocity_model = get(kwargs, :velocity_model, OneVelocityModel())
    initial_velocity = velocity_model isa TwoVelocityModel ?
        (liquid_velocity, FaceVelocity(mesh)) : liquid_velocity
    flow_options = (;
        velocity = initial_velocity, boundary = FreeSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        ),
        adaptivity = Adaptivity(criteria = refinement),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    projection = project!(flow)
    projection.converged || error("initial pressure projection did not converge")
    target_time = isnothing(final_time) ? 2pi / initial.frequency : final_time
    return (; flow, target_time, initial_projection = projection)
end

"""
    run_example(; steepness=0.55, bond=1000, reynolds=4e4,
                density_ratio=850, max_level=5, final_time=nothing,
                visualize=false, kwargs...)

Run a third-order Stokes wave for one wave period by default. Two-velocity
mode gives the analytical velocity to the liquid and starts the gas at rest.
The one-velocity closure samples the corresponding piecewise shared field.
Additional keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(; visualize = false, kwargs...)
    setup = setup_stokes3(; kwargs...)
    flow = setup.flow
    target_time = setup.target_time
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_diagnostic!(fields, flow, :kinetic_energy)
        viewer = TwoPhaseAMR.visualize(
            fields; field = :volume_fraction, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 21, live_dt = target_time / 200,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    run_simulation() = simulate!(
        flow, target_time; visualization = observer,
    )
    result = isnothing(visualization) ? run_simulation() :
        fetch(Threads.@spawn run_simulation())
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
