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
    # The potential flow, analytic on the whole domain: every active liquid
    # value takes it, wherever its face centre lies relative to η.
    function liquid_velocity(x, y, time, axis)
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

"""One-velocity start of a liquid flow over gas at rest: each face takes the
liquid's momentum over its mass, `ρ_l α u_l / (ρ_l α + ρ_g (1 - α))`, with `α`
the face's liquid fraction of the mass-compatible interpolant."""
function mass_weighted_velocity!(velocity, interface, density, liquid_velocity)
    set_velocities!(velocity, liquid_velocity)
    # The interpolant is linear in α: densities (2, 1) give 1 + α.
    fraction = mixture_density(interface, (2.0, 1.0)) .- 1
    for (index, alpha) in pairs(fraction)
        liquid = density[1] * alpha
        velocity.interior[index] *= liquid / (liquid + density[2] * (1 - alpha))
    end
    fill!(velocity.boundary, 0.0)
    return velocity
end

"""
    run_example(; steepness=0.55, bond=1000, reynolds=4e4,
                density_ratio=850, max_level=5, final_time=nothing,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run a third-order Stokes wave for one wave period by default. Every active
liquid value takes the analytical potential flow and the gas starts at rest:
two-velocity mode sets the phase values, the one-velocity closure the
mass-weighted mixture velocity of each face.
Pass `start_studio=studio` after loading TwoPhaseAMRStudio to follow the run
live. `recorder_options` overrides the recording schedule; other keywords are
passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        steepness = 0.55,
        bond = 1.0e3,
        reynolds = 4.0e4,
        density_ratio = 850.0,
        max_level = 5,
        final_time = nothing,
        start_studio = nothing,
        recorder_options = (;),
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
    velocity_model = get(kwargs, :velocity_model, OneVelocityModel())
    two_velocity = velocity_model isa TwoVelocityModel
    velocity = FaceVelocity(mesh)
    initialize_velocity!() = two_velocity ?
        set_velocities!(velocity, initial.liquid_velocity) :
        mass_weighted_velocity!(velocity, interface, density, initial.liquid_velocity)
    initialize_velocity!()
    refinement = (
        InterfaceRefinementIndicator(interface; level = max_level, t_end = 0.01),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.2,
            coarsen_tolerance = 0.2 / 3, maximum_level = max_level,
            interface = two_velocity ? interface : nothing,
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, initial.levelset)
        initialize_velocity!()
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = (0.0, -gravity),
    )
    initial_velocity = two_velocity ? (velocity, FaceVelocity(mesh)) : velocity
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
        isnothing(driver) ? simulate!(flow, target_time; observer = recorder) :
            simulate!(driver, target_time)
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
