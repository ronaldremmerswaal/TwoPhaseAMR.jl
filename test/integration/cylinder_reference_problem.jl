using TwoPhaseAMR

const CYLINDER_REFERENCE = (
    reynolds = 100.0,
    strouhal = 0.165,
    pressure_drag = 0.93,
    viscous_drag = 0.31,
    total_drag = 1.24,
    maximum_lift = 0.3,
    separation_angle_degrees = 117.0,
)

"""Pressure and viscous force coefficients on a PLIC embedded solid."""
function embedded_force_coefficients(
        pressure, velocity, geometry, diffusion;
        density = 1.0, reference_speed = 1.0,
        reference_length = 1.0,
    )
    pressure_force = zeros(2)
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        segment = TwoPhaseAMR._local_plic_segment(geometry.solid, ref)
        isnothing(segment) && continue
        leaf = mesh.blocks[block]
        first, last = segment
        length = hypot(
            (last[1] - first[1]) * leaf.spacing[1],
            (last[2] - first[2]) * leaf.spacing[2],
        )
        normal = (
            TwoPhaseAMR._get(geometry.solid.normal_x, ref) / leaf.spacing[1],
            TwoPhaseAMR._get(geometry.solid.normal_y, ref) / leaf.spacing[2],
        )
        normal_scale = hypot(normal...)
        normal_scale > 0 || continue
        value = TwoPhaseAMR._get(pressure, ref)
        pressure_force[1] -= value * normal[1] / normal_scale * length
        pressure_force[2] -= value * normal[2] / normal_scale * length
    end

    viscous_force = zeros(2)
    for axis in 1:2
        component = axis == 1 ? diffusion.x : diffusion.y
        for connection in component.wall_connections
            iszero(connection.boundary_axis) || continue
            TwoPhaseAMR._include_wall_connection(diffusion.boundary, connection) || continue
            global_index = component.local_to_global[
                connection.degree_of_freedom,
            ]
            viscous_force[axis] += diffusion.viscosity *
                connection.conductance * velocity.interior[global_index]
        end
        for reaction in component.wall_reactions
            global_index = component.local_to_global[reaction.degree_of_freedom]
            viscous_force[axis] += diffusion.viscosity *
                reaction.coefficient * velocity.interior[global_index]
        end
    end
    scale = 2 / (density * reference_speed^2 * reference_length)
    return (
        pressure_drag = scale * pressure_force[1],
        pressure_lift = scale * pressure_force[2],
        viscous_drag = scale * viscous_force[1],
        viscous_lift = scale * viscous_force[2],
        drag = scale * (pressure_force[1] + viscous_force[1]),
        lift = scale * (pressure_force[2] + viscous_force[2]),
    )
end

"""
Estimate the shedding Strouhal number from positive-going lift crossings.
With `minimum_lift > 0`, a crossing counts only after a preceding negative
excursion and a following positive excursion both reach that magnitude.
"""
function cylinder_strouhal(
        times, lifts; start_time = first(times),
        diameter = 1.0, reference_speed = 1.0, minimum_lift = 0.0,
    )
    crossings = Float64[]
    armed = false
    crossing = nothing
    for index in 1:(length(times) - 1)
        times[index + 1] >= start_time || continue
        first, last = lifts[index], lifts[index + 1]
        if first <= -minimum_lift
            armed = true
            crossing = nothing
        end
        if armed && first <= 0 < last
            fraction = iszero(last - first) ? 0.0 : -first / (last - first)
            candidate = times[index] + fraction * (times[index + 1] - times[index])
            candidate >= start_time && (crossing = candidate)
        end
        if armed && !isnothing(crossing) && last >= minimum_lift
            push!(crossings, crossing)
            crossing = nothing
            armed = false
        end
    end
    length(crossings) >= 2 || return NaN
    period = sum(diff(crossings)) / (length(crossings) - 1)
    return diameter / (reference_speed * period)
end

"""
    cylinder_flow_setup(; reynolds=100, base_level=1, max_level=2,
                        geometry_level=max_level, ...)

Construct the cylinder case of Verstappen and Dröge (2005): diameter one,
unit inflow four diameters upstream, an outlet ten diameters downstream, and
lateral boundaries eight diameters apart. The lateral and outlet velocity
conditions are zero normal derivative. `geometry_level > base_level` enables
fixed-level tree refinement around the PLIC cylinder while retaining 14×8
cells per leaf block, which gives square cells in the 14D×8D domain.
"""
function cylinder_flow_setup(;
        reynolds = 100.0, block_size = (14, 8),
        base_level = 2, max_level = 3,
        geometry_level = max_level, adaptation_interval = 5,
        perturbation = 1.0e-3,
    )
    reynolds > 0 || throw(ArgumentError("reynolds must be positive"))
    0 <= base_level <= geometry_level <= max_level || throw(
        ArgumentError("require 0 <= base_level <= geometry_level <= max_level"),
    )
    adaptation = MeshAdaptationOptions(
        update_interval = adaptation_interval,
        coarsen_delay = adaptation_interval,
        minimum_level = base_level,
    )
    mesh = BlockTreeMesh(
        block_size = block_size,
        origin = (-4.0, -4.0), extent = (14.0, 8.0),
        periodic = (false, false), max_level = max_level,
        refine = (key, origin, extent) -> key.level < base_level,
        adaptation = adaptation,
    )
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x, y) - 0.5;
        lipschitz = 1, atol = 1.0e-7, maxdepth = 24,
    )
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    # A transverse perturbation even in y breaks reflection symmetry (v_y is
    # odd in a symmetric wake), allowing the shedding instability to grow.
    initial_velocity = (x, y, time, axis) -> axis == 1 ? 1.0 :
        perturbation * exp(-((x - 0.75)^2 + y^2))
    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    indicator = GeometryRefinementIndicator(
        geometry; level = geometry_level,
    )
    if geometry_level > base_level
        initial_adaptation = AMRWorkspace(mesh)
        register_adaptive_field!(initial_adaptation, geometry)
        for step in 1:(geometry_level - base_level + 1)
            stats = adapt!(
                initial_adaptation, (indicator,); step, force = true,
            )
            stats.changed || break
            # Setup may resample the fixed analytic cylinder.  Subsequent AMR
            # updates use conservative geometric PLIC transfer.
            initialize!(
                geometry.solid, geometry.levelset;
                lipschitz = 1, atol = 1.0e-7, maxdepth = 24,
            )
            refresh_geometry!(geometry)
        end
    end
    flow = FlowSimulation(
        mesh, SinglePhase(
            density = 1.0, dynamic_viscosity = inv(reynolds),
        ); velocity = initial_velocity, geometry, boundary,
        numerics = NavierStokesOptions(
            advection = AdvectionOptions(scheme = Centred()),
        ),
        adaptivity = Adaptivity(
            criteria = geometry_level > base_level ? (indicator,) : (),
        ),
    )
    enforce_velocity_boundary!(
        flow.velocity, boundary; geometry,
    )
    forces() = embedded_force_coefficients(
        flow.pressure, flow.velocity, geometry, flow.diffusion_operator,
    )
    for (name, component) in (
            (:drag, :drag), (:pressure_drag, :pressure_drag),
            (:viscous_drag, :viscous_drag), (:lift, :lift),
        )
        let selected_component = component
            add_diagnostic!(
                flow, name,
                state -> getproperty(forces(), selected_component),
            )
        end
    end
    return (; flow, mesh, geometry, boundary, indicator)
end

function _cylinder_visualization(case; viewer, display, checkpoints, live_dt)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, case.flow)
    register_geometry!(fields, case.geometry)
    register_diagnostic!(
        fields, case.flow, :drag;
        label = "Drag coefficient", color = :darkorange,
    )
    register_diagnostic!(
        fields, case.flow, :pressure_drag;
        label = "Pressure drag coefficient", color = :firebrick,
    )
    register_diagnostic!(
        fields, case.flow, :viscous_drag;
        label = "Viscous drag coefficient", color = :goldenrod,
    )
    register_diagnostic!(
        fields, case.flow, :lift;
        label = "Lift coefficient", color = :dodgerblue,
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :velocity_vorticity, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1100, 720)),
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints, live_dt,
        diagnostic_every = 1, asynchronous = false,
    )
    return fields, session
end

"""Run the paper's Reynolds-100 cylinder case and return force history."""
function cylinder_flow(;
        final_time = 40.0, dt = nothing, cfl = 0.25,
        transient_time = final_time / 2,
        checkpoints = 11, live_dt = final_time / 200,
        viewer = NoVisualization(), display = false,
        tolerance = 1.0e-9, max_cycles = 1000, kwargs...,
    )
    case = cylinder_flow_setup(; kwargs...)
    fields, session = _cylinder_visualization(
        case; viewer, display, checkpoints, live_dt,
    )
    result = simulate!(
        case.flow, final_time; dt, cfl, visualization = session,
        tolerance, max_cycles,
    )
    samples = visualization_diagnostics(session)
    times = [sample.time for sample in samples]
    drag = [sample[:drag] for sample in samples]
    pressure_drag = [sample[:pressure_drag] for sample in samples]
    viscous_drag = [sample[:viscous_drag] for sample in samples]
    lift = [sample[:lift] for sample in samples]
    selected = findall(>=(transient_time), times)
    mean_drag = isempty(selected) ? NaN : sum(drag[selected]) / length(selected)
    mean_pressure_drag = isempty(selected) ? NaN :
        sum(pressure_drag[selected]) / length(selected)
    mean_viscous_drag = isempty(selected) ? NaN :
        sum(viscous_drag[selected]) / length(selected)
    maximum_lift = isempty(selected) ? NaN : maximum(abs, lift[selected])
    strouhal = cylinder_strouhal(
        times, lift; start_time = transient_time,
    )
    return merge(
        case, (;
            result, fields, viewer = session.viewer,
            session, times, drag,
            pressure_drag, viscous_drag, lift, mean_drag,
            mean_pressure_drag, mean_viscous_drag, maximum_lift, strouhal,
        )
    )
end

"""
    cylinder_paper_validation(; cells_per_diameter=(4, 8, 16), ...)

Run a uniform-grid refinement study of the Reynolds-100 cylinder testcase in
Verstappen and Dröge (2005).  The domain and boundary conditions are fixed by
[`cylinder_flow_setup`](@ref).  A root block of `(14n, 8n)` gives `n` cells per
cylinder diameter and square cells.  The returned observed pressure-drag order
can be compared with the approximately second-order result reported in the
paper; the remaining quantities can be compared with `CYLINDER_REFERENCE`.

This is intentionally a long validation run: `final_time` must cover the
initial transient and at least two positive-going lift crossings after
`transient_time` for a Strouhal number to be defined.
"""
function cylinder_paper_validation(;
        cells_per_diameter = (4, 8, 16), final_time = 40.0,
        transient_time = final_time / 2, kwargs...,
    )
    length(cells_per_diameter) >= 3 || throw(
        ArgumentError(
            "at least three resolutions are required to establish convergence",
        )
    )
    all(>(0), cells_per_diameter) || throw(
        ArgumentError(
            "cells_per_diameter entries must be positive",
        )
    )
    studies = map(cells_per_diameter) do resolution
        cylinder_flow(;
            block_size = (14resolution, 8resolution),
            base_level = 0, max_level = 0, geometry_level = 0,
            final_time, transient_time, viewer = NoVisualization(),
            display = false, kwargs...,
        )
    end
    drag = getproperty.(studies, :mean_pressure_drag)
    # The default resolutions double, so the last three values directly give
    # the standard self-convergence estimate.  For a custom sequence report
    # NaN unless its last two refinement ratios agree.
    h = inv.(Float64.(cells_per_diameter))
    first_ratio = h[end - 2] / h[end - 1]
    second_ratio = h[end - 1] / h[end]
    pressure_drag_order = isapprox(
            first_ratio, second_ratio; rtol = 128eps(), atol = 0,
        ) ? log(
            abs(drag[end - 2] - drag[end - 1]) /
            abs(drag[end - 1] - drag[end]),
        ) / log(first_ratio) : NaN
    return (;
        studies, cells_per_diameter, pressure_drag_order,
        reference = CYLINDER_REFERENCE,
    )
end

"""Interactive cylinder visualization with the filled PLIC solid overlay."""
function visualize_cylinder_flow(; kwargs...)
    return cylinder_flow(; display = true, viewer = nothing, kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    case = visualize_cylinder_flow()
    println(
        (
            time = case.result.time, steps = case.result.steps,
            blocks = nblocks(case.mesh), mean_drag = case.mean_drag,
            maximum_lift = case.maximum_lift, strouhal = case.strouhal,
            reference = CYLINDER_REFERENCE,
        )
    )
    scene = visualization_figure(case.viewer).scene
    scene.events.window_open[] && wait(scene)
end
