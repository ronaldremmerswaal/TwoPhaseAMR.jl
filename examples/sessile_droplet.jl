using TwoPhaseAMR

"""Compute the Young--Laplace sessile-drop profile used by the thesis test."""
function sessile_equilibrium_profile(
        initial_radius, contact_angle, density, sigma, gravity;
        points = 800
    )
    target_area = initial_radius^2 * (
        contact_angle - sin(contact_angle) * cos(contact_angle)
    )
    coefficient = (density[1] - density[2]) * gravity / sigma
    initial_angle = pi - contact_angle

    function trajectory(initial_curvature, length)
        ds = length / points
        angle = initial_angle
        curvature = initial_curvature
        x = 0.0
        y = 0.0
        xs = Vector{Float64}(undef, points + 1)
        ys = similar(xs)
        xs[1] = x
        ys[1] = y
        derivative(state) = (
            state[2], -coefficient * sin(state[1]),
            cos(state[1]), sin(state[1]),
        )
        for index in 1:points
            state = (angle, curvature, x, y)
            k1 = derivative(state)
            middle1 = ntuple(i -> state[i] + ds * k1[i] / 2, 4)
            k2 = derivative(middle1)
            middle2 = ntuple(i -> state[i] + ds * k2[i] / 2, 4)
            k3 = derivative(middle2)
            endpoint = ntuple(i -> state[i] + ds * k3[i], 4)
            k4 = derivative(endpoint)
            angle, curvature, x, y = ntuple(
                i -> state[i] + ds *
                    (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6,
                4
            )
            xs[index + 1] = x
            ys[index + 1] = y
        end
        xs .-= last(xs)
        area = 0.0
        for index in 1:points
            area -= (ys[index] + ys[index + 1]) *
                (xs[index + 1] - xs[index])
        end
        return angle, area, reverse(xs), reverse(ys)
    end

    unknown = [inv(initial_radius), initial_radius * contact_angle]
    for _ in 1:15
        angle, area, _, _ = trajectory(unknown...)
        residual = [
            angle - pi,
            (area - target_area) / target_area,
        ]
        maximum(abs, residual) < 2.0e-12 && break
        jacobian = zeros(2, 2)
        for column in 1:2
            perturbed = copy(unknown)
            increment = 1.0e-6max(abs(unknown[column]), 1.0)
            perturbed[column] += increment
            angle_p, area_p, _, _ = trajectory(perturbed...)
            jacobian[:, column] .= (
                [
                    angle_p - pi,
                    (area_p - target_area) / target_area,
                ] .- residual
            ) ./ increment
        end
        unknown .-= jacobian \ residual
        all(>(0), unknown) ||
            error("sessile-profile iteration left the physical branch")
    end
    angle, area, x, y = trajectory(unknown...)
    abs(angle - pi) < 1.0e-10 ||
        error("sessile profile failed its symmetry condition")
    abs(area - target_area) < 1.0e-10target_area ||
        error("sessile profile failed its volume condition")
    return (; x, y, area = target_area)
end

function sessile_equilibrium_levelset(profile, centre_x)
    function height(radius)
        radius <= first(profile.x) && return first(profile.y)
        radius >= last(profile.x) && return 0.0
        upper = searchsortedfirst(profile.x, radius)
        lower = upper - 1
        weight = (radius - profile.x[lower]) /
            (profile.x[upper] - profile.x[lower])
        return (1 - weight) * profile.y[lower] +
            weight * profile.y[upper]
    end
    contact_radius = last(profile.x)
    return function (x, y)
        radius = abs(x - centre_x)
        return radius <= contact_radius ? y - height(radius) :
            hypot(radius - contact_radius, y)
    end
end

"""
    sessile_droplet_setup(; base_cells=16, max_level=2,
                          interface_level=1, ...)

Build the finite-Laplace sessile-drop equilibrium from the thesis. The default
case has `Bo=1.21`, `La=550`, density ratio 1000, and a right-angle contact
condition on the bottom wall. It starts from an equal-volume circular cap,
which differs from the gravity-deformed Young--Laplace equilibrium. Interface
and curvature criteria drive AMR during the ensuing relaxation.
"""
function sessile_droplet_setup(;
        base_cells = 8, max_level = 3,
        interface_level = min(1, max_level), contact_angle = pi / 2,
        initial_radius = 0.2, density = (1.0, 1.0e-3),
        surface_tension = 1.0, bond = 1.21, laplace = 5.5e2,
        gas_liquid_viscosity_ratio = 0.0175,
        cells_per_radius = 8.0, adaptation_interval = 5,
        profile_points = 800
    )
    base_cells >= 2 || throw(ArgumentError("base_cells must be at least two"))
    0 <= interface_level <= max_level ||
        throw(ArgumentError("interface_level must lie within mesh limits"))
    0 < contact_angle < pi ||
        throw(ArgumentError("contact_angle must lie in (0, pi)"))
    initial_radius > 0 ||
        throw(ArgumentError("initial_radius must be positive"))
    all(>(0), density) ||
        throw(ArgumentError("phase densities must be positive"))
    surface_tension > 0 ||
        throw(ArgumentError("surface_tension must be positive"))
    bond > 0 || throw(ArgumentError("bond must be positive"))
    laplace > 0 || throw(ArgumentError("laplace must be positive"))

    gravity_magnitude = bond * surface_tension /
        ((density[1] - density[2]) * initial_radius^2)
    liquid_viscosity = sqrt(
        surface_tension * density[1] * initial_radius / laplace
    )
    viscosity = (
        liquid_viscosity,
        gas_liquid_viscosity_ratio * liquid_viscosity,
    )
    profile = sessile_equilibrium_profile(
        initial_radius, contact_angle, density,
        surface_tension, gravity_magnitude; points = profile_points
    )
    equilibrium_levelset = sessile_equilibrium_levelset(profile, 0.5)
    circle_centre_y = -initial_radius * cos(contact_angle)
    initial_height = initial_radius * (1 - cos(contact_angle))
    equilibrium_height = first(profile.y)
    initial_levelset = (x, y) ->
    hypot(x - 0.5, y - circle_centre_y) - initial_radius
    mesh = BlockTreeMesh(
        block_size = (base_cells, base_cells), max_level = max_level,
        periodic = (false, false),
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval, coarsen_delay = 2
        )
    )
    interface = InterfaceField(
        mesh, initial_levelset; lipschitz = 1,
        contact_angle = (nothing, nothing, contact_angle, nothing)
    )
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = interface_level
    )
    curvature_indicator = CurvatureRefinementIndicator(
        interface; cells_per_radius, minimum_level = interface_level
    )
    indicators = (interface_indicator, curvature_indicator)
    if max_level > 0
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:(max_level + 1)
            stats = adapt!(
                adaptation, indicators; step, force = true
            )
            stats.changed || break
            # The circular initial geometry is resampled only during mesh
            # construction. Evolution uses geometric PLIC transfer.
            initialize!(interface, initial_levelset; lipschitz = 1)
        end
    end
    initial_volume = integrate(interface)
    flow = FlowSimulation(
        mesh; interface, density, viscosity,
        surface_tension,
        surface_tension_formulation = GFM(),
        gravity = (0.0, -gravity_magnitude), boundary = NoSlip(),
        refinement_indicators = max_level > 0 ? indicators : ()
    )
    capillary_time = sqrt(
        sum(density) * initial_radius^3 / surface_tension
    )
    return (;
        mesh, interface, flow, profile, initial_levelset,
        equilibrium_levelset, indicators,
        density, viscosity, surface_tension, gravity_magnitude,
        initial_volume, initial_height, equilibrium_height, capillary_time,
    )
end

function _sessile_target(case, capillary_times, final_time)
    capillary_times > 0 ||
        throw(ArgumentError("capillary_times must be positive"))
    target = isnothing(final_time) ?
        capillary_times * case.capillary_time : final_time
    target > case.flow.time ||
        throw(ArgumentError("final_time must exceed the current flow time"))
    return target
end

"""Build and advance the finite-Laplace sessile-droplet equilibrium."""
function sessile_droplet(;
        capillary_times = 8.0, final_time = nothing, dt = nothing,
        visualization::AbstractVisualization = NoVisualization(), kwargs...
    )
    case = sessile_droplet_setup(; kwargs...)
    target = _sessile_target(case, capillary_times, final_time)
    result = simulate!(
        case.flow, target; dt, visualization,
        rtol = 1.0e-9, atol = 1.0e-11, max_cycles = 300
    )
    return (; case..., result)
end

"""
    visualize_sessile_droplet(; capillary_times=8, checkpoints=21, ...)

Run the sessile-drop equilibrium with live interface, velocity, pressure, and
volume views. GLMakie must be loaded by the caller.
"""
function visualize_sessile_droplet(;
        capillary_times = 8.0, final_time = nothing, dt = nothing,
        checkpoints = 21, live_dt = nothing, display = true,
        asynchronous = true, viewer = nothing, kwargs...
    )
    case = sessile_droplet_setup(; kwargs...)
    target = _sessile_target(case, capillary_times, final_time)
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    isnothing(viewer) && Threads.nthreads() == 1 && @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI"
    )

    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields,
        :volume_fraction, case.interface.fraction,
        :velocity, case.flow.velocity,
        :pressure, case.flow.pressure
    )
    register_diagnostic!(
        fields, :volume_fraction_error,
        (velocity, pressure, time, step, solver) ->
        (integrate(case.interface) - case.initial_volume) /
            case.initial_volume;
        label = "Relative liquid-volume error", color = :darkorange
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1050, 760))
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt, asynchronous
    )
    simulation = Threads.@spawn simulate!(
        case.flow, target; dt, visualization = session,
        rtol = 1.0e-9, atol = 1.0e-11, max_cycles = 300
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time = target,
    )
end

function wait_for_sessile_droplet!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_sessile_droplet()
    println(
        "Started the sessile droplet through t=$(run.final_time) on " *
            "$(nblocks(run.mesh)) blocks."
    )
    result = wait_for_sessile_droplet!(run)
    volume_error = (integrate(run.interface) - run.initial_volume) /
        run.initial_volume
    println(
        "Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates. Relative liquid-volume error=$volume_error. " *
            "Close the window to exit."
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
