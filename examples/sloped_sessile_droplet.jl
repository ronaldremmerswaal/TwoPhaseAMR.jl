using TwoPhaseAMR

function sloped_sessile_kinetic_energy(
        interface, velocity, density, geometry,
    )
    face_density = mixture_density(interface, density)
    volumes = TwoPhaseAMR._geometry_face_dual_volumes(geometry)
    energy = zero(eltype(velocity))
    for index in eachindex(velocity.interior)
        energy += face_density[index] * abs2(velocity.interior[index]) *
            volumes[index] / 2
    end
    return energy
end

"""
    sloped_sessile_droplet_setup(; wall_angle=pi/12, ...)

Build the arbitrary-geometry sessile-droplet case from the thesis. The solid
is a PLIC plane inclined by `wall_angle`; gravity is normal to that plane.
The initial circular cap has the prescribed contact angle but is not the
finite-Bond equilibrium, so its relaxation is visible. Fixed interface
refinement, curvature refinement, and fixed refinement of the solid boundary
are active by default.
"""
function sloped_sessile_droplet_setup(;
        base_cells = 8, max_level = 2,
        interface_level = min(1, max_level),
        geometry_level = min(1, max_level),
        wall_angle = pi / 12, wall_point = (0.5, 0.22),
        contact_angle = pi / 2, initial_radius = 0.16,
        density = (1.0, 1.0e-3), surface_tension = 1.0,
        bond = 1.21, laplace = 5.5e2,
        gas_liquid_viscosity_ratio = 0.0175,
        cells_per_radius = 8.0, adaptation_interval = 2,
    )
    0 <= interface_level <= max_level || throw(
        ArgumentError("interface_level must lie within mesh limits"),
    )
    0 <= geometry_level <= max_level || throw(
        ArgumentError("geometry_level must lie within mesh limits"),
    )
    0 < contact_angle < pi || throw(
        ArgumentError("contact_angle must lie in (0, pi)"),
    )
    tangent = (cos(wall_angle), sin(wall_angle))
    normal = (-sin(wall_angle), cos(wall_angle))
    signed_wall_distance(x, y) =
        normal[1] * (x - wall_point[1]) +
        normal[2] * (y - wall_point[2])
    mesh = BlockTreeMesh(
        block_size = (base_cells, base_cells), max_level = max_level,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval,
            coarsen_delay = adaptation_interval,
        ),
    )
    # Negative signed distance is the solid half-space.
    geometry = GeometryField(mesh, signed_wall_distance)
    centre = (
        wall_point[1] - initial_radius * cos(contact_angle) * normal[1],
        wall_point[2] - initial_radius * cos(contact_angle) * normal[2],
    )
    levelset = (x, y) -> hypot(x - centre[1], y - centre[2]) -
        initial_radius
    interface = InterfaceField(
        mesh, levelset; geometry, contact_angle, lipschitz = 1,
        atol = 1.0e-7, maxdepth = 24,
    )
    gravity_magnitude = bond * surface_tension /
        ((density[1] - density[2]) * initial_radius^2)
    gravity = (-gravity_magnitude * normal[1], -gravity_magnitude * normal[2])
    liquid_viscosity = sqrt(
        surface_tension * density[1] * initial_radius / laplace
    )
    viscosity = (
        liquid_viscosity,
        gas_liquid_viscosity_ratio * liquid_viscosity,
    )
    indicators = (
        InterfaceRefinementIndicator(interface; level = interface_level),
        CurvatureRefinementIndicator(
            interface; cells_per_radius,
            minimum_level = 0, maximum_level = max_level,
        ),
        InterfaceRefinementIndicator(
            geometry.solid; level = geometry_level,
        ),
    )
    if max_level > 0
        initial_adaptation = AMRWorkspace(mesh)
        register_adaptive_field!(initial_adaptation, interface)
        register_adaptive_field!(initial_adaptation, geometry)
        for step in 1:(max_level + 1)
            stats = adapt!(
                initial_adaptation, indicators; step, force = true,
            )
            stats.changed || break
            # During construction, evaluate the analytic circle on every new
            # mesh.  Runtime AMR retains the conservative PLIC transfer; using
            # it here would preserve the root mesh's coarse circular chord.
            initialize!(
                interface, levelset; lipschitz = 1,
                atol = 1.0e-7, maxdepth = 24,
            )
        end
    end
    initial_volume = _open_phase_volume_example(interface, geometry)
    flow = FlowSimulation(
        mesh; interface, geometry, density, viscosity,
        surface_tension, surface_tension_formulation = GFM(),
        gravity, boundary = NoSlip(),
        refinement_indicators = max_level > 0 ? indicators : (),
    )
    capillary_time = sqrt(
        sum(density) * initial_radius^3 / surface_tension
    )
    return (;
        mesh, geometry, interface, flow, indicators, tangent, normal,
        centre, contact_angle, initial_radius, density, viscosity,
        surface_tension, gravity, initial_volume, capillary_time,
    )
end

function _open_phase_volume_example(interface, geometry)
    return sum(eachindex(interface.mesh.blocks)) do block
        sum(interior(interface.fraction, block)) *
            interface.mesh.blocks[block].volume
    end
end

function sloped_sessile_droplet(;
        capillary_times = 4.0, final_time = nothing, dt = nothing,
        visualization::AbstractVisualization = NoVisualization(), kwargs...,
    )
    case = sloped_sessile_droplet_setup(; kwargs...)
    target = isnothing(final_time) ?
        capillary_times * case.capillary_time : final_time
    result = simulate!(
        case.flow, target; dt, visualization,
        rtol = 1.0e-9, atol = 1.0e-11, max_cycles = 1000,
    )
    return (; case..., result)
end

function visualize_sloped_sessile_droplet(;
        capillary_times = 4.0, final_time = nothing,
        checkpoints = 21, live_dt = nothing,
        display = true, viewer = nothing, kwargs...,
    )
    case = sloped_sessile_droplet_setup(; kwargs...)
    target = isnothing(final_time) ?
        capillary_times * case.capillary_time : final_time
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields, :volume_fraction, case.interface,
        :velocity, case.flow.velocity, :pressure, case.flow.pressure,
    )
    register_diagnostic!(
        fields, :kinetic_energy,
        (velocity, pressure, time, step, solver) ->
        sloped_sessile_kinetic_energy(
            case.interface, velocity, case.density, case.geometry,
        ); label = "Kinetic energy", color = :darkorange,
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1050, 760)),
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt, diagnostic_every = 1,
        asynchronous = false,
    )
    result = simulate!(
        case.flow, target; visualization = session,
        rtol = 1.0e-9, atol = 1.0e-11, max_cycles = 1000,
    )
    return (;
        case..., result, fields, viewer = selected_viewer, session,
        final_time = target,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    case = visualize_sloped_sessile_droplet()
    println((
        time = case.result.time, steps = case.result.steps,
        blocks = nblocks(case.mesh),
        volume_error = abs(
            _open_phase_volume_example(case.interface, case.geometry) -
                case.initial_volume,
        ),
    ))
    scene = visualization_figure(case.viewer).scene
    scene.events.window_open[] && wait(scene)
end
