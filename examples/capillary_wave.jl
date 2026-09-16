using LinearAlgebra
using SpecialFunctions
using TwoPhaseAMR

"""Cosine-mode amplitude obtained by integrating the geometric VOF field."""
function capillary_wave_amplitude(interface)
    mesh = interface.mesh
    amplitude = zero(eltype(interface))
    for ref in TwoPhaseAMR._cell_references(mesh)
        x, _ = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = mesh.blocks[ref.block].spacing
        cosine_integral = (
            sinpi(2 * (x + hx / 2)) -
                sinpi(2 * (x - hx / 2))
        ) / pi
        amplitude += TwoPhaseAMR._get(
            interface.fraction, ref
        ) * hy * cosine_integral
    end
    return amplitude
end

function _capillary_quartic_roots(coefficients::NTuple{5, Float64})
    coefficients[1] == 1 ||
        throw(ArgumentError("the quartic must be monic"))
    companion = zeros(ComplexF64, 4, 4)
    companion[2, 1] = companion[3, 2] = companion[4, 3] = 1
    companion[:, 4] .= -reverse(collect(coefficients[2:end]))
    return eigvals(companion)
end

"""Prosperetti's finite-viscosity linear capillary-wave amplitude."""
function capillary_linear_amplitude(
        time; liquid_density, gas_density, kinematic_viscosity,
        surface_tension, wavenumber = 2pi
    )
    rho_l, rho_g = liquid_density, gas_density
    viscosity_frequency = kinematic_viscosity * wavenumber^2
    frequency_squared = surface_tension * wavenumber^3 /
        (rho_l + rho_g)
    beta = rho_l * rho_g / (rho_l + rho_g)^2
    roots = _capillary_quartic_roots(
        (
            1.0,
            -4beta * sqrt(viscosity_frequency),
            2(1 - 6beta) * viscosity_frequency,
            4(1 - 3beta) * viscosity_frequency^(3 / 2),
            (1 - 4beta) * viscosity_frequency^2 + frequency_squared,
        )
    )
    products = [
        prod(roots[j] - roots[i] for j in eachindex(roots) if i != j)
            for i in eachindex(roots)
    ]
    result = 4(1 - 4beta) * viscosity_frequency^2 /
        (8(1 - 4beta) * viscosity_frequency^2 + frequency_squared) *
        erfc(sqrt(viscosity_frequency * time))
    for index in eachindex(roots)
        shifted = roots[index]^2 - viscosity_frequency
        result += roots[index] / products[index] * frequency_squared /
            shifted * exp(shifted * time) *
            erfc(roots[index] * sqrt(time))
    end
    return real(result)
end

"""
    capillary_wave_setup(; density_ratio=1e3, base_cells=8,
                         max_level=2, interface_level=max_level, ...)

Build the small-amplitude standing capillary wave from the thesis. The domain
is three wavelengths deep, the initial amplitude is one percent of a
wavelength, and both phases have the same kinematic viscosity at `La=3000`.
"""
function capillary_wave_setup(;
        density_ratio = 1.0e3, base_cells = 8,
        max_level = 2, interface_level = max_level,
        amplitude = 0.01, laplace = 3000.0,
        surface_tension = 1.0, formulation = GFM(),
        adaptation_interval = 5
    )
    density_ratio >= 1 ||
        throw(ArgumentError("density_ratio must be at least one"))
    base_cells >= 2 || throw(ArgumentError("base_cells must be at least two"))
    0 <= interface_level <= max_level ||
        throw(ArgumentError("interface_level must lie within mesh limits"))
    0 < amplitude < 0.1 ||
        throw(ArgumentError("amplitude must lie in (0, 0.1)"))
    laplace > 0 || throw(ArgumentError("laplace must be positive"))
    surface_tension > 0 ||
        throw(ArgumentError("surface_tension must be positive"))

    wavelength = 1.0
    wavenumber = 2pi / wavelength
    mean_height = 0.25
    liquid_density = 1.0
    gas_density = liquid_density / density_ratio
    kinematic_viscosity = sqrt(
        surface_tension * liquid_density * wavelength / laplace
    )
    viscosity = (
        liquid_density * kinematic_viscosity,
        gas_density * kinematic_viscosity,
    )
    levelset = (x, y) ->
    y - mean_height - amplitude * cospi(2x / wavelength)
    mesh = BlockTreeMesh(
        block_size = (base_cells, 3base_cells),
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval, coarsen_delay = 2
        )
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(
        interface; level = interface_level
    )
    if interface_level > 0
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:(interface_level + 1)
            stats = adapt!(adaptation, indicator; step, force = true)
            stats.changed || break
            # Use the analytical geometry only while constructing the mesh.
            initialize!(interface, levelset; lipschitz = 1)
        end
    end
    flow = FlowSimulation(
        mesh; interface, viscosity,
        density = (liquid_density, gas_density),
        surface_tension,
        surface_tension_formulation = formulation,
        boundary = FreeSlip(),
        refinement_indicators = max_level > 0 ? (indicator,) : ()
    )
    omega = sqrt(
        surface_tension * wavenumber^3 /
            (liquid_density + gas_density)
    )
    period = 2pi / omega
    initial_amplitude = capillary_wave_amplitude(interface)
    initial_volume = integrate(interface)
    finest_resolution = base_cells << interface_level
    accurate_dt = period / max(80, finest_resolution^2 ÷ 2)
    return (;
        mesh, interface, indicator, flow, levelset,
        density = (liquid_density, gas_density), viscosity,
        kinematic_viscosity, surface_tension, wavenumber,
        omega, period, initial_amplitude,
        initial_volume, accurate_dt,
    )
end

function _capillary_target(case, periods, final_time)
    periods > 0 || throw(ArgumentError("periods must be positive"))
    target = isnothing(final_time) ? periods * case.period : final_time
    target > case.flow.time ||
        throw(ArgumentError("final_time must exceed the current flow time"))
    return target
end

"""Build and advance a capillary wave, returning the complete numerical case."""
function capillary_wave(;
        periods = 1.0, final_time = nothing, dt = nothing,
        visualization::AbstractVisualization = NoVisualization(), kwargs...
    )
    case = capillary_wave_setup(; kwargs...)
    target = _capillary_target(case, periods, final_time)
    selected_dt = isnothing(dt) ? case.accurate_dt : dt
    result = simulate!(
        case.flow, target; dt = selected_dt, visualization,
        rtol = 1.0e-9, atol = 1.0e-12, max_cycles = 300
    )
    return (; case..., result)
end

"""
    visualize_capillary_wave(; periods=1, checkpoints=21, ...)

Run the thesis standing capillary wave with live interface, velocity, pressure,
and normalized-amplitude views. GLMakie must be loaded by the caller.
"""
function visualize_capillary_wave(;
        periods = 1.0, final_time = nothing, dt = nothing,
        checkpoints = 21, live_dt = nothing, display = true,
        asynchronous = true, viewer = nothing, kwargs...
    )
    case = capillary_wave_setup(; kwargs...)
    target = _capillary_target(case, periods, final_time)
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    selected_dt = isnothing(dt) ? case.accurate_dt : dt
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
        fields, :amplitude_magnitude,
        (velocity, pressure, time, step, solver) ->
        abs(
            capillary_wave_amplitude(case.interface) /
                case.initial_amplitude
        );
        label = "|Interface amplitude / initial|", color = :dodgerblue
    )
    register_diagnostic!(
        fields, :analytical_amplitude_magnitude,
        (velocity, pressure, time, step, solver) -> abs(
            capillary_linear_amplitude(
                time;
                liquid_density = case.density[1],
                gas_density = case.density[2],
                kinematic_viscosity = case.kinematic_viscosity,
                surface_tension = case.surface_tension,
                wavenumber = case.wavenumber
            )
        );
        label = "|Linear-theory amplitude|", color = :black
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1100, 760))
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt, asynchronous
    )
    simulation = Threads.@spawn simulate!(
        case.flow, target; dt = selected_dt, visualization = session,
        rtol = 1.0e-9, atol = 1.0e-12, max_cycles = 300
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time = target,
    )
end

function wait_for_capillary_wave!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_capillary_wave()
    println(
        "Started the capillary wave through t=$(run.final_time) on " *
            "$(nblocks(run.mesh)) blocks."
    )
    result = wait_for_capillary_wave!(run)
    amplitude = capillary_wave_amplitude(run.interface) / run.initial_amplitude
    println(
        "Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates. Final normalized amplitude=$amplitude. " *
            "Close the window to exit."
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
