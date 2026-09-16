using TwoPhaseAMR

"""Third-order finite-depth Stokes-wave initial condition used in the thesis.

The default nondimensional parameters reproduce the two-dimensional case in
Sec. 8.6: steepness 0.55, Re=4e4, Bo=1e3, liquid/gas density ratio 850, and
the reported phase viscosities. The gas is initially at rest, as stated in
the thesis. This is the unprojected analytical input; [`stokes3_wave`](@ref)
projects it onto the discrete one-velocity divergence constraint after the
initial AMR passes.
"""
function stokes3_initial_condition(;
        steepness = 0.55, depth = 0.5, gravity = 1.0
    )
    isfinite(steepness) && steepness >= 0 ||
        throw(ArgumentError("steepness must be finite and nonnegative"))
    isfinite(depth) && depth > 0 ||
        throw(ArgumentError("depth must be finite and positive"))
    isfinite(gravity) && gravity > 0 ||
        throw(ArgumentError("gravity must be finite and positive"))
    wave_number = 2pi
    chi = inv(tanh(wave_number * depth))
    phi1 = inv(cosh(wave_number * depth))
    phi2 = 3(chi^2 - 1)^2 /
        (8chi * cosh(2wave_number * depth))
    phi3 = (chi^2 - 1) * (chi^2 + 3) * (9chi^2 - 13) /
        (64cosh(3wave_number * depth))
    eta2 = chi * (3chi^2 - 1) / 4
    eta31 = -3(chi^4 - 3chi^2 + 3) / 8
    eta33 = 3(8chi^6 + (chi^2 - 1)^2) / 64
    frequency = sqrt(
        gravity * wave_number *
            (1 + steepness^2 * (9(chi^2 - 1)^2 / 8 + chi^2)) / chi
    )
    amplitude = steepness / wave_number

    elevation(x) = amplitude * (
        cos(wave_number * x) +
            steepness * eta2 * cos(2wave_number * x) +
            steepness^2 * (
            eta31 * cos(wave_number * x) +
                eta33 * cos(3wave_number * x)
        )
    )
    levelset(x, y) = y - elevation(x)
    function liquid_velocity(x, y, axis)
        y > elevation(x) && return 0.0
        if axis == 1
            return amplitude * gravity / frequency * (
                wave_number * phi1 * cosh(wave_number * (y + depth)) *
                    cos(wave_number * x) +
                    2wave_number * steepness * phi2 *
                    cosh(2wave_number * (y + depth)) *
                    cos(2wave_number * x) +
                    3wave_number * steepness^2 * phi3 *
                    cosh(3wave_number * (y + depth)) *
                    cos(3wave_number * x)
            )
        end
        return amplitude * gravity / frequency * (
            wave_number * phi1 * sinh(wave_number * (y + depth)) *
                sin(wave_number * x) +
                2wave_number * steepness * phi2 *
                sinh(2wave_number * (y + depth)) *
                sin(2wave_number * x) +
                3wave_number * steepness^2 * phi3 *
                sinh(3wave_number * (y + depth)) *
                sin(3wave_number * x)
        )
    end
    velocity(x, y, time, axis) = liquid_velocity(x, y, axis)
    return (; levelset, elevation, velocity, frequency)
end

"""
    stokes3_wave(; base_cells=8, max_level=5, final_time=5, ...)

Build and optionally advance the thesis Stokes-3 wave. Initial AMR is applied
before constructing the solver, then the exact level set and velocity are
resampled on the selected leaves. During the simulation the mesh combines a
fixed interface band, curvature refinement, and the thesis criterion
`h*abs(vorticity)` with hysteresis.

The thesis sets the gas velocity to zero for its two-velocity formulation.
Sampling that state into a single face velocity does not satisfy the discrete
one-velocity divergence constraint. This example therefore applies
one variable-density pressure projection after the final initial AMR pass and
before any WY transport.

Set `final_time=0` to build the complete initialized problem without taking a
time step. The returned named tuple contains the `FlowSimulation`, AMR
indicators, analytical initial condition, and simulation result.
"""
function stokes3_wave(;
        base_cells = 8, max_level = 5,
        final_time = 5.0,
        steepness = 0.55, depth = 0.5, gravity = 1.0,
        bond = 1.0e3, reynolds = 4.0e4,
        cells_per_radius = 8.0,
        vorticity_refine_tolerance = 0.2,
        vorticity_coarsen_tolerance = vorticity_refine_tolerance / 3,
        density = (1.0, 1 / 850),
        pressure_preconditioner = :tree_geometric,
        pressure_preconditioner_options = (;),
        diffusion_preconditioner = :tree_geometric,
        diffusion_preconditioner_options = (;),
        matrix_free_fine = false,
    )
    all(>(0), density) ||
        throw(ArgumentError("phase densities must be positive"))
    density[1] > density[2] ||
        throw(ArgumentError("Stokes-3 requires liquid density greater than gas density"))
    bond > 0 || throw(ArgumentError("Bond number must be positive"))
    reynolds > 0 || throw(ArgumentError("Reynolds number must be positive"))
    liquid_viscosity = density[1] * sqrt(gravity) / reynolds
    viscosity = (liquid_viscosity, liquid_viscosity / 51.15)
    surface_tension = (density[1] - density[2]) * gravity / (bond * (2pi)^2)
    initial = stokes3_initial_condition(; steepness, depth, gravity)
    mesh = BlockTreeMesh(
        block_size = (base_cells, base_cells),
        origin = (-0.5, -depth), extent = (1.0, 1.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(coarsen_delay = 2)
    )
    interface = InterfaceField(mesh, initial.levelset)
    velocity = FaceVelocity(mesh, initial.velocity)
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = max_level, t_end = 0.01
    )
    curvature_indicator = CurvatureRefinementIndicator(
        interface; cells_per_radius
    )
    vorticity_indicator = VorticityRefinementIndicator(
        velocity;
        refine_tolerance = vorticity_refine_tolerance,
        coarsen_tolerance = vorticity_coarsen_tolerance,
        maximum_level = max_level
    )
    indicators = (
        interface_indicator, curvature_indicator, vorticity_indicator,
    )

    adaptation = AMRWorkspace(mesh, interface, velocity)
    for pass in 1:(max_level + 1)
        stats = adapt!(adaptation, indicators; step = pass, force = true)
        stats.changed || break
        # Initialization may use the analytical geometry. Evolution and later
        # regrids use conservative geometric PLIC transfer.
        initialize!(interface, initial.levelset)
        set_velocities!(velocity, initial.velocity)
    end

    flow = FlowSimulation(
        mesh; interface, velocity, viscosity = viscosity,
        density, surface_tension = surface_tension,
        surface_tension_formulation = GFM(),
        gravity = (0.0, -gravity),
        boundary = FreeSlip(), refinement_indicators = indicators,
        pressure_preconditioner, pressure_preconditioner_options,
        diffusion_preconditioner, diffusion_preconditioner_options,
        matrix_free_fine,
    )
    initial_projection = begin
        stats = project!(
            flow.velocity, flow.pressure, flow.solver.pressure_solver,
            one(eltype(flow.velocity)), flow.workspace.projection
        )
        stats.converged || error(
            "Stokes-3 initial one-velocity projection failed to converge"
        )
        stats
    end
    result = final_time > 0 ? simulate!(
            flow, final_time; rtol = 1.0e-9, atol = 1.0e-11,
            max_cycles = 150
        ) : nothing
    return (; flow, indicators, initial, initial_projection, result)
end

"""
    visualize_stokes3_wave(; final_time=2, checkpoints=21,
                           live_dt=final_time/200, ...)

Run [`stokes3_wave`](@ref) with a lightweight live viewer. Volume fraction,
velocity, pressure, and AMR blocks are available in the field menu. GLMakie
must be loaded by the caller. The returned `simulation` task can be joined
with [`wait_for_stokes3!`](@ref).
"""
function visualize_stokes3_wave(;
        final_time = 2.0, checkpoints = 21,
        live_dt = final_time / 200, display = true,
        asynchronous = true, viewer = nothing, kwargs...
    )
    final_time > 0 ||
        throw(ArgumentError("final_time must be positive"))
    isnothing(viewer) && Threads.nthreads() == 1 && @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI"
    )

    case = stokes3_wave(; final_time = 0.0, kwargs...)
    flow = case.flow
    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields,
        :volume_fraction, flow.interface.fraction,
        :velocity, flow.velocity,
        :pressure, flow.pressure
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1100, 700))
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints, live_dt,
        asynchronous
    )
    simulation = Threads.@spawn simulate!(
        flow, final_time; visualization = session,
        rtol = 1.0e-9, atol = 1.0e-11, max_cycles = 150
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time,
    )
end

function wait_for_stokes3!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_stokes3_wave()
    println(
        "Started the Stokes-3 wave through t=$(run.final_time) on " *
            "$(nblocks(run.flow.solver.mesh)) blocks."
    )
    result = wait_for_stokes3!(run)
    println(
        "Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates on $(nblocks(run.flow.solver.mesh)) blocks. " *
            "Close the window to exit."
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
