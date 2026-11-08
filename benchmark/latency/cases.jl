# Setups mirror the corresponding examples through the point immediately before
# simulate!, including their initial AMR passes. `Example` is the included example.

function setup_case(::Val{:taylor_green}, ::Val{two_velocity}; max_level = 1, dt = 0.001) where {two_velocity}
    two_velocity && error("Taylor–Green is single phase")
    viscosity = inv(50.0)
    mesh = BlockTreeMesh(
        block_size = (12, 12), extent = (2pi, 2pi), max_level = max_level,
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level < max_level && key.i == 0 && key.j == 0),
    )
    velocity = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    physics = SinglePhase(density = 1.0, dynamic_viscosity = viscosity)
    flow_options = (;
        velocity, boundary = FreeSlip(),
        diagnostics = (KineticEnergy(), LinearMomentum(), DivergenceNorm(), LeafBlockCount()),
        dt,
    )
    return FlowSimulation(mesh, physics; flow_options...)
end

function setup_case(::Val{:capillary_wave}, ::Val{two_velocity}; max_level = 1, dt = 0.001) where {two_velocity}
    wavelength = 1.0
    wavenumber = 2pi / wavelength
    density = (1.0, 1.0e-3)
    surface_tension = 1.0
    kinematic_viscosity = sqrt(surface_tension * wavelength / 3000.0)
    dynamic_viscosity = density .* kinematic_viscosity
    amplitude = 0.01
    levelset = (x, y) -> y - 0.25 - amplitude * cos(wavenumber * x)
    mesh = BlockTreeMesh(
        base_mesh = (8, 24), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (wavelength, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed || break
        initialize!(interface, levelset; lipschitz = 1)
    end
    physics = TwoPhase(interface; density, dynamic_viscosity, surface_tension)
    analytical_amplitude(time) = amplitude * Example.prosperetti_amplitude(
        time; liquid_density = density[1], gas_density = density[2],
        kinematic_viscosity, gravity = 0.0, surface_tension, wavenumber,
    )
    extra = two_velocity ? (; dt, velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection())) : (; dt)
    flow_options = (;
        boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(:amplitude, state -> interface_height(state.interface) - 0.25),
            Diagnostic(:analytical_amplitude, state -> analytical_amplitude(state.time)),
        ),
        adaptivity = Adaptivity(criteria = max_level > 0 ? (refinement,) : ()),
        extra...,
    )
    return FlowSimulation(mesh, physics; flow_options...)
end

function setup_case(::Val{:stokes3_wave}, ::Val{two_velocity}; max_level = 1, dt = 0.001) where {two_velocity}
    depth = 0.5
    gravity = 1.0
    density = (1.0, inv(850.0))
    liquid_viscosity = sqrt(gravity) / 4.0e4
    dynamic_viscosity = (liquid_viscosity, liquid_viscosity / 51.15)
    surface_tension = (density[1] - density[2]) * gravity / (1.0e3 * (2pi)^2)
    initial = Example.stokes3_initial_condition(; steepness = 0.55, depth, gravity)
    mesh = BlockTreeMesh(
        block_size = (8, 8), origin = (-0.5, -depth), extent = (1.0, 1.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, initial.levelset)
    liquid_velocity = FaceVelocity(mesh, initial.liquid_velocity)
    refinement = (
        InterfaceRefinementIndicator(interface; level = max_level, t_end = 0.01),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        VorticityRefinementIndicator(
            liquid_velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.2 / 3,
            maximum_level = max_level, interface = two_velocity ? interface : nothing,
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, liquid_velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed || break
        initialize!(interface, initial.levelset)
        set_velocities!(liquid_velocity, initial.liquid_velocity)
    end
    physics = TwoPhase(interface; density, dynamic_viscosity, surface_tension, gravity = (0.0, -gravity))
    extra = two_velocity ? (; dt, velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection())) : (; dt)
    initial_velocity = two_velocity ? (liquid_velocity, FaceVelocity(mesh)) : liquid_velocity
    flow_options = (;
        velocity = initial_velocity, boundary = FreeSlip(),
        diagnostics = (KineticEnergy(), LinearMomentum(), PhaseVolume(), DivergenceNorm(), LeafBlockCount()),
        adaptivity = Adaptivity(criteria = refinement),
        extra...,
    )
    return FlowSimulation(mesh, physics; flow_options...)
end

function initial_projection!(flow)
    projection = project!(flow)
    projection.converged || error("initial pressure projection did not converge")
    return nothing
end

function one_step!(flow, dt)
    previous = flow.steps
    simulate!(flow, flow.time + dt; dt)
    flow.steps == previous + 1 || error("expected one step, got $(flow.steps - previous)")
    return nothing
end
