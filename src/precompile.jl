using PrecompileTools: @setup_workload, @compile_workload

# Keep these fixtures small and independent of the examples and test suite.
function _precompile_taylor_green()
    mesh = BlockTreeMesh(
        block_size = (8, 8), extent = (2pi, 2pi), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0,
    )
    velocity = (x, y, time, axis) -> exp(-0.04time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    flow = FlowSimulation(
        mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.02);
        velocity, boundary = FreeSlip(), dt = 0.001,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(), LeafBlockCount(),
        ),
    )
    stable_timestep(flow)
    result = simulate!(flow, 0.001; dt = 0.001)
    print_simulation_summary(IOBuffer(), flow, result)
    return nothing
end

function _precompile_stokes_wave()
    mesh = BlockTreeMesh(
        block_size = (8, 8), origin = (-0.5, -0.5), extent = (1.0, 1.0),
        periodic = (true, false), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 2),
    )
    levelset = (x, y) -> y - 0.05cospi(2x)
    velocity = (x, y, time, axis) -> axis == 1 ?
        sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    interface = InterfaceField(mesh, levelset)
    liquid_velocity = FaceVelocity(mesh, velocity)
    refinement = (
        InterfaceRefinementIndicator(interface; level = 1, t_end = 0.01),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8),
        VorticityRefinementIndicator(
            liquid_velocity; refine_tolerance = 0.2,
            coarsen_tolerance = 0.2 / 3, maximum_level = 1,
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, liquid_velocity)
    if adapt!(adaptation, refinement; step = 1, force = true).changed
        initialize!(interface, levelset)
        set_velocities!(liquid_velocity, velocity)
    end
    physics = TwoPhase(
        interface; density = (1.0, 0.001),
        dynamic_viscosity = (0.000025, 0.0000005),
        surface_tension = 0.000025, gravity = (0.0, -1.0),
    )
    flow = FlowSimulation(
        mesh, physics; velocity = liquid_velocity, boundary = FreeSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(), DivergenceNorm(),
            LeafBlockCount(),
        ),
        adaptivity = Adaptivity(criteria = refinement), dt = 0.001,
    )
    project!(flow)
    stable_timestep(flow)
    result = simulate!(flow, 0.001; dt = 0.001)
    _rebuild_topology_dependent_state!(flow.system, flow.workspace; dt = 0.001)
    print_simulation_summary(IOBuffer(), flow, result)
    return nothing
end

function _precompile_capillary_wave(velocity_model)
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (8, 8), origin = (0.0, -0.5),
        extent = (1.0, 1.0), periodic = (true, false), max_level = 1,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    levelset = (x, y) -> y - 0.25 - 0.01cospi(2x)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = 1)
    adaptation = AMRWorkspace(mesh, interface)
    if adapt!(adaptation, refinement; step = 1, force = true).changed
        initialize!(interface, levelset; lipschitz = 1)
    end
    physics = TwoPhase(
        interface; density = (1.0, 0.001),
        dynamic_viscosity = (0.018, 0.000018), surface_tension = 1.0,
    )
    options = (;
        boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(:amplitude, state -> interface_height(state.interface) - 0.25),
            Diagnostic(:analytical_amplitude, state -> 0.01cos(state.time)),
        ),
        adaptivity = Adaptivity(criteria = (refinement,)), dt = 0.001,
    )
    flow = velocity_model isa OneVelocityModel ?
        FlowSimulation(mesh, physics; options...) :
        FlowSimulation(mesh, physics; options..., velocity_model)
    stable_timestep(flow)
    simulate!(flow, 0.001; dt = 0.001)
    return nothing
end

function _precompile_workload()
    _precompile_taylor_green()
    _precompile_stokes_wave()
    _precompile_capillary_wave(OneVelocityModel())
    _precompile_capillary_wave(TwoVelocityModel(pressure_projection = CMDGFMProjection()))
    return nothing
end

@setup_workload begin
    @compile_workload begin
        # Four refined blocks can reach the small-loop Polyester threshold on a
        # multithreaded Julia. Keep image generation serial and restore the
        # thread pool even if a workload fails.
        Polyester.disable_polyester_threads(_precompile_workload)
    end
end
