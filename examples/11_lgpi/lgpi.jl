using TwoPhaseAMR

"""Scaled large gas-pocket impact geometry used by the LGPI benchmark."""
function lgpi_initial_condition(; scale = 20.0)
    wavelength = 20.0 / scale
    domain = (wavelength, 3wavelength / 5)
    mean_height = 7.6 / scale
    amplitude = 3.6 / scale
    horizontal_shift = 15.5 / scale
    steepness = 0.44scale
    bathymetry_axes = (18.0 / scale, 2.8 / scale)
    elevation(x) = mean_height + amplitude *
        tanh(steepness * (x - horizontal_shift))
    interface_levelset(x, y) = y - elevation(x)
    bathymetry_levelset(x, y) = hypot(
        x / bathymetry_axes[1], y / bathymetry_axes[2],
    ) - 1
    return (;
        wavelength, domain, interface_levelset, bathymetry_levelset, bathymetry_axes,
        interface_lipschitz = hypot(1, amplitude * steepness),
        bathymetry_lipschitz = maximum(inv, bathymetry_axes),
    )
end

"""
    run_example(; scale=20, max_level=4, final_time=nothing, refinement_ymin=nothing,
                velocity_model=TwoVelocityModel(),
                numerics=NavierStokesOptions(advection=AdvectionOptions(scheme=MonotonizedCentral())),
                kwargs...)

Run the scaled large gas-pocket impact benchmark. The default end time is
`1.2` gravity-wave periods. By default refinement indicators begin above the
first coarse block boundary beyond the bathymetry top plus two coarse cells.
The default two-velocity model uses frozen interface fluxes and CMDGFM
projection. Momentum is advected with the monotonized-central limiter by
default: Lax–Wendroff leaves visible dispersive noise in the vorticity of this
violent impact. Additional keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        scale = 20.0,
        max_level = 4,
        final_time = nothing,
        refinement_ymin = nothing,
        velocity_model = TwoVelocityModel(),
        numerics = NavierStokesOptions(
            advection = AdvectionOptions(scheme = MonotonizedCentral()),
        ),
        kwargs...,
    )
    initial = lgpi_initial_condition(; scale)
    base_mesh = (40, 24)
    block_size = (4, 4)
    coarse_dy = initial.domain[2] / base_mesh[2]
    block_height = block_size[2] * coarse_dy
    ymin = isnothing(refinement_ymin) ?
        ceil((initial.bathymetry_axes[2] + 2coarse_dy) / block_height) *
        block_height + coarse_dy / 2 : refinement_ymin
    mesh = BlockTreeMesh(
        base_mesh = base_mesh, block_size = block_size,
        extent = initial.domain, periodic = (false, false),
        max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 10,
        ),
    )
    geometry = GeometryField(
        mesh, initial.bathymetry_levelset;
        lipschitz = initial.bathymetry_lipschitz,
        atol = 1.0e-8, maxdepth = 28,
    )
    interface = InterfaceField(
        mesh, initial.interface_levelset; geometry,
        lipschitz = initial.interface_lipschitz,
        atol = 1.0e-8, maxdepth = 28,
    )
    velocity = FaceVelocity(mesh)
    fixed_level = min(1, max_level)
    refinement = (
        InterfaceRefinementIndicator(interface; level = fixed_level, ymin),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.25,
            coarsen_tolerance = 0.25 / 3,
            maximum_level = max_level, ymin,
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, geometry, velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(
            geometry.solid, initial.bathymetry_levelset;
            lipschitz = initial.bathymetry_lipschitz,
            atol = 1.0e-8, maxdepth = 28,
        )
        refresh_geometry!(geometry)
        initialize!(
            interface, initial.interface_levelset;
            lipschitz = initial.interface_lipschitz,
            atol = 1.0e-8, maxdepth = 28,
        )
        fill!(velocity.interior, 0)
        fill!(velocity.boundary, 0)
    end

    density = (1.0e3, 1.0)
    physics = TwoPhase(
        interface; density,
        dynamic_viscosity = (1.0e-3, 1.75e-5),
        surface_tension = 7.3e-2, gravity = (0.0, -9.81),
    )
    flow_options = (;
        velocity, velocity_model, geometry, numerics, boundary = FreeSlip(),
        diagnostics = (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        ),
        adaptivity = Adaptivity(criteria = refinement),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    gravity_time = sqrt(
        sum(density) / (density[1] - density[2]) *
            2pi * initial.wavelength / 9.81,
    )
    target_time = isnothing(final_time) ? 1.2gravity_time : final_time
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_geometry!(fields, geometry)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(21), preview = nothing,
    )
    result = simulate!(flow, target_time; observer = recorder)
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = run_example()
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
end
