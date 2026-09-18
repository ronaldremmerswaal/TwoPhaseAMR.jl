using TwoPhaseAMR

const DEFAULT_WAVE_AMPLITUDE = 1.0e-2
const MIXED_LEVEL_INTERFACE_HEIGHT = -0.25

function _free_slip_boundaries()
    return VelocityBoundaryConditions(
        left = FreeSlip(), right = FreeSlip(),
        bottom = FreeSlip(), top = FreeSlip(),
    )
end

function _wave_mesh(
        cells_per_wavelength; mixed_level = false, wavelength = 1.0,
    )
    cells_per_wavelength >= 8 || throw(
        ArgumentError(
            "cells_per_wavelength must be at least 8",
        )
    )
    if !mixed_level
        return BlockTreeMesh(
            block_size = (
                cells_per_wavelength, 3cells_per_wavelength,
            ),
            origin = (0.0, -1.5wavelength),
            extent = (wavelength, 3wavelength),
            periodic = (true, false),
        )
    end
    ratio = cells_per_wavelength / 8
    level = round(Int, log2(ratio))
    8 * 2^level == cells_per_wavelength || throw(
        ArgumentError(
            "mixed-level studies require 8*2^level cells per wavelength",
        )
    )
    interface_height = MIXED_LEVEL_INTERFACE_HEIGHT * wavelength
    return BlockTreeMesh(
        block_size = (8, 24), max_level = level,
        origin = (0.0, -1.5wavelength),
        extent = (wavelength, 3wavelength),
        periodic = (true, false),
        refine = (key, origin, extent) -> key.level < level &&
            origin[2] < interface_height + 0.15wavelength &&
            origin[2] + extent[2] > interface_height - 0.15wavelength,
    )
end

"""Complex first Fourier coefficient of the reconstructed VOF interface."""
function wave_amplitude(
        interface::InterfaceField; wavelength = 1.0,
        liquid_above = false,
    )
    k = 2pi / wavelength
    coefficient = 0.0 + 0.0im
    mesh = interface.mesh
    nx, ny = mesh.block_size
    for (block_index, block) in pairs(mesh.blocks)
        fraction = interior(interface.fraction, block_index)
        for j in 1:ny, i in 1:nx
            x, _ = cellcenter(mesh, block_index, i, j)
            coefficient += block.volume * fraction[i, j] * cis(-k * x)
        end
    end
    # The x-independent mean depth integrates to zero in this Fourier mode.
    result = 2coefficient / wavelength
    return liquid_above ? -result : result
end

"""Normalized trapezoidal `L2` error of a sampled complex wave amplitude."""
function normalized_amplitude_error(times, numerical, analytical)
    length(times) == length(numerical) == length(analytical) || throw(
        DimensionMismatch("times and amplitude histories must have equal length"),
    )
    length(times) >= 2 || throw(
        ArgumentError(
            "at least two amplitude samples are required",
        )
    )
    error_integral = reference_integral = 0.0
    for index in 1:(length(times) - 1)
        width = times[index + 1] - times[index]
        width > 0 || throw(ArgumentError("sample times must increase"))
        error_integral += width * (
            abs2(numerical[index] - analytical[index]) +
                abs2(numerical[index + 1] - analytical[index + 1])
        ) / 2
        reference_integral += width * (
            abs2(analytical[index]) + abs2(analytical[index + 1])
        ) / 2
    end
    reference_integral > 0 || throw(
        ArgumentError(
            "the analytical amplitude history has zero norm",
        )
    )
    return sqrt(error_integral / reference_integral)
end

"""
    instability_dispersion(; density_ratio, weber, bond,
                             mean_interface_velocity, liquid_above, wavelength)

Return the two roots of the inviscid interfacial-wave dispersion relation in
`pt2_appendix.tex`. The capillary frequency is the time scale, so the
square-root terms have magnitudes `1`, `weber`, and `bond`. The optional
`mean_interface_velocity` is the density-weighted velocity multiplying the
non-square-root term. Dimensional coefficients for `FlowSimulation` are
returned as derived quantities.

The definitions are
`density_ratio = rho_liquid/rho_gas`,
`weber = rho_liquid*rho_gas*delta_U^2 /
          (surface_tension*k*(rho_liquid + rho_gas))`, and
`bond = (rho_liquid - rho_gas)*abs(gravity_y) /
        (surface_tension*k^2)`.
"""
function instability_dispersion(;
        density_ratio = 3.0,
        weber = 0.5,
        bond = 0.75,
        mean_interface_velocity = 0.0,
        liquid_above = true,
        wavelength = 1.0,
    )
    isfinite(density_ratio) && density_ratio >= 1 || throw(
        ArgumentError(
            "density_ratio must be finite and at least one",
        )
    )
    isfinite(weber) && weber >= 0 || throw(
        ArgumentError(
            "weber must be finite and nonnegative",
        )
    )
    isfinite(bond) && bond >= 0 || throw(
        ArgumentError(
            "bond must be finite and nonnegative",
        )
    )
    isfinite(mean_interface_velocity) || throw(
        ArgumentError(
            "mean_interface_velocity must be finite",
        )
    )
    isfinite(wavelength) && wavelength > 0 || throw(
        ArgumentError(
            "wavelength must be finite and positive",
        )
    )
    density_ratio == 1 && bond > 0 && throw(
        ArgumentError(
            "nonzero bond cannot be realized at density_ratio = 1",
        )
    )

    rho_liquid = 1.0
    rho_gas = inv(float(density_ratio))
    density_sum = rho_liquid + rho_gas
    k = 2pi / float(wavelength)

    # Set sigma*k^3/(rho_l + rho_g) = 1. The resulting capillary time is the
    # nondimensional time scale, and We and Bo are exact ratios of the shear
    # and gravity terms to this capillary contribution.
    surface_tension = density_sum / k^3
    velocity_jump = density_sum * sqrt(float(weber) / (rho_liquid * rho_gas)) / k
    mean_velocity = float(mean_interface_velocity)
    liquid_velocity = mean_velocity - rho_gas * velocity_jump / density_sum
    gas_velocity = mean_velocity + rho_liquid * velocity_jump / density_sum
    mean_frequency = k * mean_velocity
    gravity_magnitude = bond == 0 ? 0.0 :
        float(bond) * density_sum / (k * (rho_liquid - rho_gas))
    vertical_gravity = -gravity_magnitude

    rho_below, rho_above = liquid_above ?
        (rho_gas, rho_liquid) : (rho_liquid, rho_gas)
    capillary_term = surface_tension * k^3 / density_sum
    shear_term = rho_liquid * rho_gas * k^2 * velocity_jump^2 /
        density_sum^2
    gravity_term = -vertical_gravity * k *
        (rho_below - rho_above) / density_sum
    radicand = capillary_term - shear_term + gravity_term
    offset = sqrt(complex(radicand))
    return (;
        plus = mean_frequency + offset,
        minus = mean_frequency - offset,
        mean_frequency, offset, radicand,
        capillary_term, shear_term, gravity_term,
        density = (rho_liquid, rho_gas),
        tangential_velocity = (liquid_velocity, gas_velocity),
        surface_tension, vertical_gravity,
        density_ratio = float(density_ratio),
        weber = float(weber), bond = float(bond),
        mean_interface_velocity = mean_velocity,
        liquid_above, wavelength = float(wavelength), k,
    )
end

"""Linear-theory complex amplitude for equal excitation of both modes."""
function instability_linear_amplitude(
        time, dispersion; initial_amplitude = DEFAULT_WAVE_AMPLITUDE,
    )
    return initial_amplitude * (
        exp(-im * dispersion.plus * time) +
            exp(-im * dispersion.minus * time)
    ) / 2
end

function _phase_velocity_initializer(
        base_velocity, perturbation, phase_above, interface_height, k,
    )
    if phase_above
        return (x, y, time, axis) -> axis == 1 ?
            base_velocity - k * perturbation * cos(k * x) *
            exp(-k * max(y - interface_height, 0)) :
            k * perturbation * sin(k * x) *
            exp(-k * max(y - interface_height, 0))
    end
    return (x, y, time, axis) -> axis == 1 ?
        base_velocity + k * perturbation * cos(k * x) *
        exp(k * min(y - interface_height, 0)) :
        k * perturbation * sin(k * x) *
        exp(k * min(y - interface_height, 0))
end

function _default_instability_final_time(dispersion)
    growth_rate = abs(imag(dispersion.offset))
    growth_rate > sqrt(eps()) && return log(2) / growth_rate
    oscillation_frequency = abs(real(dispersion.offset))
    oscillation_frequency > sqrt(eps()) &&
        return 2pi / oscillation_frequency
    return one(dispersion.wavelength)
end

function _instability_target(case, final_time)
    target = isnothing(final_time) ? case.default_final_time : final_time
    isfinite(target) && target > case.flow.time || throw(
        ArgumentError(
            "final_time must be finite and exceed the current flow time",
        )
    )
    return target
end

"""Construct the general thesis KH/RT interfacial-wave problem."""
function two_velocity_instability_setup(;
        cells_per_wavelength = 16,
        density_ratio = 3.0,
        weber = 0.5,
        bond = 0.75,
        mean_interface_velocity = 0.0,
        liquid_above = true,
        wavelength = 1.0,
        initial_amplitude = DEFAULT_WAVE_AMPLITUDE,
        mixed_level = false,
    )
    0 < initial_amplitude < 0.1wavelength || throw(
        ArgumentError(
            "initial_amplitude must lie between zero and 0.1 wavelength",
        )
    )
    dispersion = instability_dispersion(;
        density_ratio, weber, bond, mean_interface_velocity,
        liquid_above, wavelength,
    )
    mesh = _wave_mesh(
        cells_per_wavelength; mixed_level, wavelength,
    )
    interface_height = mixed_level ?
        MIXED_LEVEL_INTERFACE_HEIGHT * wavelength : 0.0
    k = dispersion.k
    interface_levelset = if liquid_above
        (x, y) -> interface_height + initial_amplitude * cos(k * x) - y
    else
        (x, y) -> y - interface_height - initial_amplitude * cos(k * x)
    end
    interface = InterfaceField(
        mesh, interface_levelset;
        lipschitz = hypot(1, initial_amplitude * k),
    )

    phase_speed = dispersion.mean_frequency / k
    liquid_base, gas_base = dispersion.tangential_velocity
    liquid_perturbation = initial_amplitude * (phase_speed - liquid_base)
    gas_perturbation = initial_amplitude * (phase_speed - gas_base)
    liquid = _phase_velocity_initializer(
        liquid_base, liquid_perturbation, liquid_above,
        interface_height, k,
    )
    gas = _phase_velocity_initializer(
        gas_base, gas_perturbation, !liquid_above,
        interface_height, k,
    )
    flow = FlowSimulation(
        mesh; density = dispersion.density,
        dynamic_viscosity = (0.0, 0.0), interface,
        velocity = (liquid, gas), boundary = _free_slip_boundaries(),
        surface_tension = dispersion.surface_tension,
        surface_tension_formulation = GFM(),
        gravity = (0.0, dispersion.vertical_gravity),
        interface_method = FrozenInterfaceFluxes(),
    )
    initial_numerical_amplitude = wave_amplitude(
        interface; wavelength, liquid_above,
    )
    return (;
        mesh, interface, flow, dispersion, interface_levelset,
        initial_amplitude = float(initial_amplitude),
        initial_numerical_amplitude,
        default_final_time = _default_instability_final_time(dispersion),
    )
end

function _run_instability!(
        case, final_time; cfl, tolerance, max_cycles,
    )
    flow = case.flow
    for phase in flow.velocity
        enforce_velocity_boundary!(
            phase, flow.solver.boundary; time = flow.time,
            geometry = flow.solver.geometry,
        )
    end
    times = Float64[flow.time]
    amplitudes = ComplexF64[
        wave_amplitude(
            case.interface;
            wavelength = case.dispersion.wavelength,
            liquid_above = case.dispersion.liquid_above,
        ),
    ]
    while flow.time < final_time - 16eps(final_time)
        dt = min(
            stable_timestep(flow; cfl), final_time - flow.time,
        )
        stats = step!(
            flow.velocity, flow.pressure, flow.solver, dt,
            flow.workspace; time = flow.time, tolerance, max_cycles,
        )
        stats.converged || error("mimetic pressure projection failed: $stats")
        flow.time += dt
        flow.steps += 1
        push!(times, flow.time)
        push!(
            amplitudes,
            wave_amplitude(
                case.interface;
                wavelength = case.dispersion.wavelength,
                liquid_above = case.dispersion.liquid_above,
            ),
        )
    end
    return (; times, amplitudes, flow)
end

"""Run the general two-velocity interfacial-wave simulation."""
function two_velocity_instability(;
        final_time = nothing, cfl = 0.35,
        tolerance = 1.0e-7, max_cycles = 3000, kwargs...,
    )
    case = two_velocity_instability_setup(; kwargs...)
    target = _instability_target(case, final_time)
    result = _run_instability!(
        case, target; cfl, tolerance, max_cycles,
    )
    analytical = [
        instability_linear_amplitude(
            time, case.dispersion;
            initial_amplitude = case.initial_amplitude,
        ) for time in result.times
    ]
    return (; case..., result..., analytical, final_time = target)
end

"""Run the general instability with live numerical/theoretical amplitudes."""
function visualize_two_velocity_instability(;
        final_time = nothing, checkpoints = 21, live_dt = nothing,
        display = true, asynchronous = true, viewer = nothing,
        tolerance = 1.0e-7, max_cycles = 3000, kwargs...,
    )
    case = two_velocity_instability_setup(; kwargs...)
    target = _instability_target(case, final_time)
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    isnothing(viewer) && Threads.nthreads() == 1 && @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI",
    )

    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields,
        :volume_fraction, case.interface,
        :pressure, case.flow.pressure,
    )
    register_default_fields!(
        fields, :velocity, case.flow.velocity, case.interface,
    )
    register_diagnostic!(
        fields, :amplitude_magnitude,
        (velocity, pressure, time, step, solver) -> abs(
            wave_amplitude(
                case.interface;
                wavelength = case.dispersion.wavelength,
                liquid_above = case.dispersion.liquid_above,
            ) / case.initial_numerical_amplitude
        );
        label = "|Numerical amplitude / initial|", color = :dodgerblue,
    )
    register_diagnostic!(
        fields, :analytical_amplitude_magnitude,
        (velocity, pressure, time, step, solver) -> abs(
            instability_linear_amplitude(
                time, case.dispersion;
                initial_amplitude = case.initial_amplitude,
            ) / case.initial_amplitude
        );
        label = "|Linear-theory amplitude / initial|", color = :black,
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1100, 760)),
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt, asynchronous,
    )
    simulation = Threads.@spawn simulate!(
        case.flow, target; visualization = session,
        tolerance, max_cycles,
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time = target,
    )
end

function wait_for_two_velocity_instability!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_two_velocity_instability()
    println(
        "Started the combined KH/RT wave through t=$(run.final_time) on " *
            "$(nblocks(run.mesh)) blocks.",
    )
    result = wait_for_two_velocity_instability!(run)
    numerical = wave_amplitude(
        run.interface;
        wavelength = run.dispersion.wavelength,
        liquid_above = run.dispersion.liquid_above,
    )
    analytical = instability_linear_amplitude(
        run.final_time, run.dispersion;
        initial_amplitude = run.initial_amplitude,
    )
    println(
        "Completed $(result.steps) steps. Final numerical amplitude=" *
            "$numerical, linear theory=$analytical. Close the window to exit.",
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
