using TwoPhaseAMR

const LGPI_REFERENCE = (
    full_scale_wavelength = 20.0,
    mean_height = 7.6,
    amplitude = 3.6,
    steepness = 0.44,
    horizontal_shift = 15.5,
    bathymetry_half_axes = (18.0, 2.8),
    density = (1.0e3, 1.0),
    dynamic_viscosity = (1.0e-3, 1.75e-5),
    surface_tension = 7.3e-2,
    gravity = 9.81,
)

"""
    lgpi_initial_condition(; scale=5)

Return the scaled large gas-pocket impact (LGPI) geometry from the thesis.
The liquid initially occupies the region below

`z = (z0 + A*tanh(k*(scale*x - x0))) / scale`

in a `wavelength × 3wavelength/5` tank. The solid in the lower-left corner is
the first-quadrant part of the scaled ellipse with full-scale half-axes
18 m and 2.8 m. A level-set value at or below zero denotes liquid or solid,
respectively.
"""
function lgpi_initial_condition(; scale = 5.0)
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("scale must be finite and positive"))
    reference = LGPI_REFERENCE
    wavelength = reference.full_scale_wavelength / scale
    domain = (wavelength, 3wavelength / 5)
    mean_height = reference.mean_height / scale
    amplitude = reference.amplitude / scale
    horizontal_shift = reference.horizontal_shift / scale
    scaled_steepness = reference.steepness * scale
    bathymetry_half_axes = reference.bathymetry_half_axes ./ scale

    elevation(x) = mean_height + amplitude * tanh(
        scaled_steepness * (x - horizontal_shift),
    )
    interface_levelset(x, z) = z - elevation(x)
    bathymetry_levelset(x, z) = hypot(
        x / bathymetry_half_axes[1], z / bathymetry_half_axes[2],
    ) - 1

    # Bounds on the Euclidean gradient of the two implicit functions. They
    # enable certified pure-cell detection during level-set integration.
    interface_lipschitz = hypot(1, amplitude * scaled_steepness)
    bathymetry_lipschitz = maximum(inv, bathymetry_half_axes)
    return (;
        scale, wavelength, domain, mean_height, amplitude,
        horizontal_shift, scaled_steepness, bathymetry_half_axes,
        elevation, interface_levelset, bathymetry_levelset,
        interface_lipschitz, bathymetry_lipschitz,
    )
end

"""
    lgpi_setup(; scale=5, block_size=(10, 6), max_level=5,
               interface_level=min(3, max_level), ...)

Construct the two-dimensional LGPI case from the thesis at geometrical scale
`1:scale`. The default `scale=5` gives a 4 m wave in a 4 m by 2.4 m tank.
Water and air start at rest and use the dimensional material properties from
the thesis. Surface tension, gravity, and the elliptic bathymetry are active;
all domain and embedded walls are free-slip.

The interface-presence criterion maintains a resolved interface band from the
initial condition onward. Vorticity AMR may refine dynamically up to
`max_level`; the fixed geometry-presence criterion resolves the bathymetry to
the same level as the initial interface by default. Initial AMR resamples both
analytic level sets. Runtime adaptation uses conservative geometric PLIC
transfer.
"""
function lgpi_setup(;
        scale = 5.0, block_size = (10, 6), max_level = 5,
        interface_level = min(3, max_level),
        geometry_level = interface_level,
        vorticity_refine_tolerance = 0.25,
        vorticity_coarsen_tolerance = vorticity_refine_tolerance / 3,
        adaptation_interval = 5,
        density = LGPI_REFERENCE.density,
        dynamic_viscosity = LGPI_REFERENCE.dynamic_viscosity,
        surface_tension = LGPI_REFERENCE.surface_tension,
        gravity = LGPI_REFERENCE.gravity,
        pressure_preconditioner = :tree_geometric,
        pressure_preconditioner_options = (;),
        diffusion_preconditioner = :tree_geometric,
        diffusion_preconditioner_options = (;),
    )
    0 <= interface_level <= max_level || throw(
        ArgumentError("interface_level must lie within mesh limits"),
    )
    0 <= geometry_level <= max_level || throw(
        ArgumentError("geometry_level must lie within mesh limits"),
    )
    adaptation_interval isa Integer && adaptation_interval > 0 || throw(
        ArgumentError("adaptation_interval must be a positive integer"),
    )
    all(value -> isfinite(value) && value > 0, density) ||
        throw(ArgumentError("phase densities must be finite and positive"))
    all(value -> isfinite(value) && value > 0, dynamic_viscosity) || throw(
        ArgumentError("phase dynamic viscosities must be finite and positive"),
    )
    isfinite(surface_tension) && surface_tension > 0 || throw(
        ArgumentError("surface_tension must be finite and positive"),
    )
    isfinite(gravity) && gravity > 0 ||
        throw(ArgumentError("gravity must be finite and positive"))

    initial = lgpi_initial_condition(; scale)
    mesh = BlockTreeMesh(
        block_size = block_size, origin = (0.0, 0.0),
        extent = initial.domain, periodic = (false, false),
        max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval,
            coarsen_delay = 2adaptation_interval,
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
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = interface_level,
    )
    vorticity_indicator = VorticityRefinementIndicator(
        velocity;
        refine_tolerance = vorticity_refine_tolerance,
        coarsen_tolerance = vorticity_coarsen_tolerance,
        minimum_level = 0, maximum_level = max_level,
    )
    geometry_indicator = GeometryRefinementIndicator(
        geometry; level = geometry_level,
    )
    indicators = (
        interface_indicator, vorticity_indicator, geometry_indicator,
    )

    initial_adaptation = AMRWorkspace(mesh)
    register_adaptive_field!(initial_adaptation, interface)
    register_adaptive_field!(initial_adaptation, geometry)
    register_adaptive_field!(initial_adaptation, velocity)
    for pass in 1:(max_level + 1)
        stats = adapt!(
            initial_adaptation, indicators; step = pass, force = true,
        )
        stats.changed || break
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

    flow = FlowSimulation(
        mesh; interface, geometry, velocity, density, dynamic_viscosity,
        surface_tension, surface_tension_formulation = GFM(),
        gravity = (0.0, -gravity), boundary = FreeSlip(),
        refinement_indicators = indicators,
        pressure_preconditioner, pressure_preconditioner_options,
        diffusion_preconditioner, diffusion_preconditioner_options,
    )
    gravity_time = sqrt(
        (density[1] + density[2]) / (density[1] - density[2]) *
            2pi * initial.wavelength / gravity,
    )
    interface_cells_per_wavelength = block_size[1] * (1 << interface_level)
    maximum_cells_per_wavelength = block_size[1] * (1 << max_level)
    return (;
        flow, mesh, geometry, interface, velocity = flow.velocity,
        indicators, interface_indicator, vorticity_indicator,
        geometry_indicator, initial, density, dynamic_viscosity,
        surface_tension, gravity, gravity_time,
        interface_cells_per_wavelength, maximum_cells_per_wavelength,
    )
end

"""Build and advance the thesis LGPI case without opening a viewer."""
function lgpi(;
        final_gravity_times = 0.6, final_time = nothing,
        dt = nothing, cfl = 0.35,
        visualization::AbstractVisualization = NoVisualization(),
        tolerance = 1.0e-8, max_cycles = 500, kwargs...,
    )
    case = lgpi_setup(; kwargs...)
    target = isnothing(final_time) ?
        final_gravity_times * case.gravity_time : final_time
    isfinite(target) && target >= 0 ||
        throw(ArgumentError("final time must be finite and nonnegative"))
    result = target > 0 ? simulate!(
            case.flow, target; dt, cfl, visualization,
            tolerance, max_cycles,
        ) : nothing
    return (; case..., result, final_time = target)
end

"""
    visualize_lgpi(; final_gravity_times=0.6, ...)

Run the LGPI example with live vorticity, volume-fraction, velocity, pressure,
PLIC-interface, solid-geometry, and AMR-block views. Vorticity is selected
initially and uses a symmetric automatic color scale.
"""
function visualize_lgpi(;
        final_gravity_times = 1, final_time = nothing,
        checkpoints = 21, live_dt = nothing,
        dt = nothing, cfl = 0.35,
        display = true, asynchronous = true, viewer = nothing,
        tolerance = 1.0e-8, max_cycles = 500, kwargs...,
    )
    case = lgpi_setup(; kwargs...)
    target = isnothing(final_time) ?
        final_gravity_times * case.gravity_time : final_time
    isfinite(target) && target > 0 ||
        throw(ArgumentError("final time must be finite and positive"))
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    isnothing(viewer) && Threads.nthreads() == 1 && @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI",
    )

    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields, :volume_fraction, case.interface,
        :velocity, case.flow.velocity, :pressure, case.flow.pressure,
    )
    register_diagnostic!(
        fields, :leaf_blocks,
        (velocity, pressure, time, step, solver) -> nblocks(case.mesh);
        label = "Leaf blocks", color = :darkorange,
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :velocity_vorticity, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1150, 760)),
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt, diagnostic_every = 1, asynchronous,
    )
    simulation = Threads.@spawn simulate!(
        case.flow, target; dt, cfl, visualization = session,
        tolerance, max_cycles,
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time = target,
    )
end

function wait_for_lgpi!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_lgpi()
    println(
        "Started the LGPI case through t/Tg=" *
            "$(run.final_time / run.gravity_time) on " *
            "$(nblocks(run.mesh)) blocks.",
    )
    result = wait_for_lgpi!(run)
    println(
        "Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates on $(nblocks(run.mesh)) blocks. Close the window to exit.",
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
