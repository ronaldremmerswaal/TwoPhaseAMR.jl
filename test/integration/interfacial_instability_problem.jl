using TwoPhaseAMR

const DEFAULT_WAVE_AMPLITUDE = 1.0e-2
const MIXED_LEVEL_INTERFACE_HEIGHT = -0.25

function _free_slip_boundaries()
    return VelocityBoundaryConditions(
        left = FreeSlip(), right = FreeSlip(),
        bottom = FreeSlip(), top = FreeSlip(),
    )
end

function _instability_mesh(
        cells_per_wavelength; mixed_level = false, wavelength = 1.0,
        block_size = (8, 8), adaptation_interval = 5,
    )
    length(block_size) == 2 || throw(
        ArgumentError("block_size must contain two dimensions"),
    )
    all(value -> value isa Integer && !(value isa Bool), block_size) || throw(
        ArgumentError("block_size dimensions must be integers"),
    )
    block = (Int(block_size[1]), Int(block_size[2]))
    cells_per_wavelength isa Integer && !(cells_per_wavelength isa Bool) ||
        throw(ArgumentError("cells_per_wavelength must be an integer"))
    cells_per_wavelength >= block[1] || throw(
        ArgumentError(
            "cells_per_wavelength must be at least block_size[1]",
        )
    )
    if !mixed_level
        iszero(cells_per_wavelength % block[1]) &&
            iszero(3cells_per_wavelength % block[2]) || throw(
            ArgumentError(
                "the uniform wave base mesh must be divisible by block_size",
            )
        )
        return BlockTreeMesh(
            base_mesh = (
                cells_per_wavelength, 3cells_per_wavelength,
            ),
            block_size = block,
            origin = (0.0, -1.5wavelength),
            extent = (wavelength, 3wavelength),
            periodic = (true, false),
        )
    end
    base_cells = lcm(block[1], block[2] ÷ gcd(block[2], 3))
    iszero(cells_per_wavelength % base_cells) || throw(
        ArgumentError(
            "mixed-level studies require a dyadic refinement of the " *
                "coarsest block-compatible base mesh ($base_cells cells " *
                "per wavelength)",
        )
    )
    ratio = cells_per_wavelength ÷ base_cells
    ispow2(ratio) || throw(
        ArgumentError(
            "mixed-level studies require a dyadic refinement of the " *
                "coarsest block-compatible base mesh ($base_cells cells " *
                "per wavelength)",
        )
    )
    level = trailing_zeros(ratio)
    return BlockTreeMesh(
        base_mesh = (base_cells, 3base_cells), block_size = block,
        max_level = level,
        origin = (0.0, -1.5wavelength),
        extent = (wavelength, 3wavelength),
        periodic = (true, false),
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval, coarsen_delay = 2,
        ),
    )
end

"""Complex first Fourier coefficient of the reconstructed VOF interface."""
function _instability_wave_amplitude(
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

"""Aperture-based circulation and renormalised enstrophy of the vortex sheet.

The diagnostic jump is reconstructed at every cut cell with `c(1-c)` weights
by default; this is independent of the projection's jump interpolant.
`enstrophy` is
`1/2 ∫_Γ γ² ds`; it is a sheet diagnostic, not the divergent bulk enstrophy
of an ideal distributional vortex sheet.
"""
function _instability_vortex_sheet_diagnostics(
        interface::InterfaceField, velocity;
        interpolation_weights = :c1mc,
    )
    interpolation_weights in (:c1mc, :geometric) || throw(
        ArgumentError(
            "interpolation_weights must be :c1mc or :geometric",
        ),
    )
    solver = MimeticProjectionSolver(interface)
    workspace = MimeticProjectionWorkspace(solver)
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    aperture_x, aperture_y =
        TwoPhaseAMR._local_integrated_aperture_vectors(interface, data)
    incident = TwoPhaseAMR._cmdgfm_incident_faces(interface.mesh)
    references = TwoPhaseAMR._cell_references(interface.mesh)
    T = eltype(interface)
    circulation = enstrophy = maximum_strength = interface_measure = zero(T)
    for (cell, reference) in pairs(references)
        fraction = TwoPhaseAMR._conditional_fraction(interface, reference)
        interface.fraction_floor < fraction <
            one(T) - interface.fraction_floor || continue
        qx = aperture_x[cell]
        qy = aperture_y[cell]
        measure = hypot(qx, qy)
        measure > T(64) * eps(T) || continue
        jump = zeros(T, 2)
        for axis in 1:2
            faces = filter(incident[cell]) do face
                interface.mesh.faces[face].axis == axis
            end
            weights = T[]
            for face in faces
                weight = if interpolation_weights === :c1mc
                    c = data.liquid_fraction[face]
                    c * (one(T) - c)
                else
                    one(T)
                end
                push!(weights, T(interface.mesh.faces[face].area) * weight)
            end
            weight_sum = sum(weights)
            weight_sum > zero(T) || continue
            jump[axis] = sum(
                weights[index] * (
                    velocity[2].interior[face] -
                        velocity[1].interior[face]
                ) for (index, face) in pairs(faces)
            ) / weight_sum
        end
        strength = (-qy * jump[1] + qx * jump[2]) / measure
        circulation += measure * strength
        enstrophy += measure * strength^2 / 2
        maximum_strength = max(maximum_strength, abs(strength))
        interface_measure += measure
    end
    return (; circulation, enstrophy, maximum_strength, interface_measure)
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
    # Equal excitation of the growing and decaying roots gives
    # a(t)/a(0) = cosh(growth_rate*t), so acosh(2) is the first doubling.
    growth_rate > sqrt(eps()) && return acosh(2) / growth_rate
    oscillation_frequency = abs(real(dispersion.offset))
    oscillation_frequency > sqrt(eps()) &&
        return 2pi / oscillation_frequency
    return one(dispersion.wavelength)
end

function _instability_target(case, final_time)
    target = convert(
        typeof(case.flow.time),
        isnothing(final_time) ? case.default_final_time : final_time,
    )
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
        block_size = (8, 8), adaptation_interval = 5,
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
        velocity_model = nothing,
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
    mesh = _instability_mesh(
        cells_per_wavelength;
        mixed_level, wavelength, block_size, adaptation_interval,
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
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = mesh.max_level,
    )
    if mixed_level && mesh.max_level > 0
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:(mesh.max_level + 1)
            stats = adapt!(
                adaptation, interface_indicator; step, force = true,
            )
            stats.changed || break
            # Re-evaluate the exact initial interface after each topology
            # change; conservative transfer is reserved for time evolution.
            initialize!(
                interface, interface_levelset;
                lipschitz = hypot(1, initial_amplitude * k),
            )
        end
    end

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
    selected_velocity_model = isnothing(velocity_model) ? TwoVelocityModel(
            interface_transport = interface_transport,
            pressure_projection = pressure_projection,
        ) : velocity_model
    selected_velocity_model isa AbstractVelocityModel || throw(
        ArgumentError("velocity_model must be OneVelocityModel() or TwoVelocityModel()"),
    )
    initial_velocity = selected_velocity_model isa TwoVelocityModel ?
        (liquid, gas) :
        (x, y, time, axis) -> interface_levelset(x, y) <= 0 ?
        liquid(x, y, time, axis) : gas(x, y, time, axis)
    physics = TwoPhase(
        interface; density = dispersion.density,
        dynamic_viscosity = (0.0, 0.0),
        surface_tension = dispersion.surface_tension,
        gravity = (0.0, dispersion.vertical_gravity),
    )
    flow = FlowSimulation(
        mesh, physics; velocity_model = selected_velocity_model,
        velocity = initial_velocity, boundary = _free_slip_boundaries(),
        adaptivity = Adaptivity(
            criteria = mixed_level ? (interface_indicator,) : (),
        ),
    )
    initial_numerical_amplitude = _instability_wave_amplitude(
        interface; wavelength, liquid_above,
    )
    amplitude() = _instability_wave_amplitude(
        interface; wavelength, liquid_above,
    )
    add_diagnostic!(flow, :amplitude_real, state -> real(amplitude()))
    add_diagnostic!(flow, :amplitude_imaginary, state -> imag(amplitude()))
    active_interface_transport = selected_velocity_model isa TwoVelocityModel ?
        selected_velocity_model.interface_transport : nothing
    return (;
        mesh, interface, interface_indicator, flow, dispersion,
        interface_levelset,
        initial_amplitude = float(initial_amplitude),
        initial_numerical_amplitude,
        interface_transport = active_interface_transport,
        velocity_model = selected_velocity_model,
        default_final_time = _default_instability_final_time(dispersion),
    )
end

function _run_instability!(
        case, final_time; cfl, tolerance, max_cycles, timestep_factor,
        progress_interval = 0,
    )
    flow = case.flow
    projection_diagnostics = NamedTuple[]
    isfinite(timestep_factor) && 0 < timestep_factor <= 1 || throw(
        ArgumentError("timestep_factor must be finite and in (0, 1]"),
    )
    phase_velocities = flow.velocity isa FaceVelocity ?
        (flow.velocity,) : flow.velocity
    for phase in phase_velocities
        enforce_velocity_boundary!(
            phase, flow.solver.boundary; time = flow.time,
            geometry = flow.solver.geometry,
        )
    end

    time_tolerance = 16eps(final_time)
    while flow.time < final_time - time_tolerance
        dt = min(
            timestep_factor * stable_timestep(flow; cfl),
            final_time - flow.time,
        )
        stats = step!(
            flow.velocity, flow.pressure, flow.solver, dt,
            flow.workspace; time = flow.time, tolerance, max_cycles,
        )
        stats.converged || error(
            TwoPhaseAMR._solver_failure_message(
                "pressure projection", stats, tolerance;
                details = "dt=$dt, time=$(flow.time), " *
                    "max_cycles=$max_cycles",
            )
        )
        if stats isa MimeticProjectionStats
            push!(projection_diagnostics, (;
                time = flow.time + dt,
                candidate_rows = stats.candidate_constraint_rows,
                commuting_rows = stats.commuting_constraint_rows,
                fallback_rows = stats.fallback_constraint_rows,
                commuting_residual = stats.commuting_residual,
                pressure_commuting_rows =
                    stats.pressure_commuting_constraint_rows,
                pressure_fallback_rows =
                    stats.pressure_fallback_constraint_rows,
                pressure_commuting_residual =
                    stats.pressure_commuting_residual,
                variant = stats.variant,
                energy_identity_error = stats.energy_identity_error,
                normal_velocity_defect = stats.normal_velocity_defect,
                momentum_error = stats.momentum_error,
            ))
        end
        flow.time = min(final_time, flow.time + dt)
        flow.steps += 1
        if progress_interval > 0 && iszero(flow.steps % progress_interval)
            println(
                "instability progress: step=$(flow.steps) " *
                    "time=$(flow.time) dt=$dt projection=$(typeof(stats))",
            )
            flush(stdout)
        end
        record_diagnostics!(flow)
    end
    flow.time = final_time
    record_diagnostics!(flow; event = :completed)
    real_history = diagnostic_history(flow, :amplitude_real)
    imaginary_history = diagnostic_history(flow, :amplitude_imaginary)
    times = real_history.times
    amplitudes = ComplexF64.(
        real_history.values .+ im .* imaginary_history.values,
    )
    return (;
        times, amplitudes, flow, diagnostics = flow.diagnostics,
        projection_diagnostics,
    )
end

"""Run the general interfacial-wave simulation (two velocity by default).

`timestep_factor` multiplies the current stability limit at every step. This
supports temporal-refinement studies without freezing a limit that changes as
the instability evolves.
"""
function two_velocity_instability(;
        final_time = nothing, cfl = 0.35,
        tolerance = 1.0e-7, max_cycles = 3000,
        timestep_factor = 1.0, kwargs...,
    )
    case = two_velocity_instability_setup(; kwargs...)
    target = _instability_target(case, final_time)
    result = _run_instability!(
        case, target; cfl, tolerance, max_cycles, timestep_factor,
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
    register_default_fields!(fields, case.flow)
    add_diagnostic!(
        case.flow, :amplitude_magnitude,
        state -> abs(
            _instability_wave_amplitude(
                case.interface;
                wavelength = case.dispersion.wavelength,
                liquid_above = case.dispersion.liquid_above,
            ) / case.initial_numerical_amplitude
        ),
    )
    add_diagnostic!(
        case.flow, :analytical_amplitude_magnitude,
        state -> abs(
            instability_linear_amplitude(
                state.time, case.dispersion;
                initial_amplitude = case.initial_amplitude,
            ) / case.initial_amplitude
        ),
    )
    register_diagnostic!(
        fields, case.flow, :amplitude_magnitude;
        label = "|Numerical amplitude / initial|", color = :dodgerblue,
    )
    register_diagnostic!(
        fields, case.flow, :analytical_amplitude_magnitude;
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

"""Model-neutral entry point for [`two_velocity_instability_setup`](@ref)."""
interfacial_instability_setup(; kwargs...) =
    two_velocity_instability_setup(; kwargs...)

"""Model-neutral entry point for [`two_velocity_instability`](@ref)."""
interfacial_instability(; kwargs...) = two_velocity_instability(; kwargs...)

"""Model-neutral entry point for [`visualize_two_velocity_instability`](@ref)."""
visualize_interfacial_instability(; kwargs...) =
    visualize_two_velocity_instability(; kwargs...)

wait_for_interfacial_instability!(run) =
    wait_for_two_velocity_instability!(run)

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_interfacial_instability()
    println(
        "Started the combined KH/RT wave through t=$(run.final_time) on " *
            "$(nblocks(run.mesh)) blocks.",
    )
    result = wait_for_interfacial_instability!(run)
    numerical = _instability_wave_amplitude(
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
