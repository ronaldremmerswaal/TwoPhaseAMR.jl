using TwoPhaseAMR

const DEFAULT_WAVE_AMPLITUDE = 1.0e-2
const MIXED_LEVEL_INTERFACE_HEIGHT = -0.25

function _free_slip_boundaries()
    return VelocityBoundaryConditions(
        left = FreeSlip(), right = FreeSlip(),
        bottom = FreeSlip(), top = FreeSlip(),
    )
end

function _wave_mesh(cells_per_wavelength; mixed_level = false)
    cells_per_wavelength >= 8 || throw(ArgumentError(
        "cells_per_wavelength must be at least 8",
    ))
    if !mixed_level
        return BlockTreeMesh(
            block_size = (
                cells_per_wavelength, 3cells_per_wavelength,
            ),
            origin = (0.0, -1.5), extent = (1.0, 3.0),
            periodic = (true, false),
        )
    end
    ratio = cells_per_wavelength / 8
    level = round(Int, log2(ratio))
    8 * 2^level == cells_per_wavelength || throw(ArgumentError(
        "mixed-level studies require 8*2^level cells per wavelength",
    ))
    return BlockTreeMesh(
        block_size = (8, 24), max_level = level,
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false),
        refine = (key, origin, extent) -> key.level < level &&
            origin[2] < MIXED_LEVEL_INTERFACE_HEIGHT + 0.15 &&
            origin[2] + extent[2] >
                MIXED_LEVEL_INTERFACE_HEIGHT - 0.15,
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
    length(times) >= 2 || throw(ArgumentError(
        "at least two amplitude samples are required",
    ))
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
    reference_integral > 0 || throw(ArgumentError(
        "the analytical amplitude history has zero norm",
    ))
    return sqrt(error_integral / reference_integral)
end

function kh_frequencies(; density_ratio, weber, velocity_jump = 1.0)
    density_ratio >= 1 || throw(ArgumentError(
        "density_ratio is rho_liquid/rho_gas and must be at least one",
    ))
    rho_liquid = 1.0
    rho_gas = inv(float(density_ratio))
    wavelength = 1.0
    k = 2pi / wavelength
    liquid_velocity = -velocity_jump / 2
    gas_velocity = velocity_jump / 2
    sigma = rho_gas * rho_liquid * velocity_jump^2 * wavelength /
        (2pi * weber * (rho_gas + rho_liquid))
    capillary_time = sqrt(
        (rho_liquid + rho_gas) * wavelength^3 / (2pi * sigma),
    )
    mean_frequency = k * (
        rho_liquid * liquid_velocity + rho_gas * gas_velocity
    ) / (rho_liquid + rho_gas)
    offset = 2pi / capillary_time * sqrt(complex(1 - weber))
    return (;
        plus = mean_frequency + offset,
        minus = mean_frequency - offset,
        sigma, capillary_time, rho_liquid, rho_gas,
        liquid_velocity, gas_velocity,
    )
end

function kh_amplitude(time; density_ratio, weber,
        initial_amplitude = DEFAULT_WAVE_AMPLITUDE, velocity_jump = 1.0)
    frequency = kh_frequencies(; density_ratio, weber, velocity_jump)
    return initial_amplitude * (
        exp(-im * frequency.plus * time) +
            exp(-im * frequency.minus * time)
    ) / 2
end

function rt_frequencies(; density_ratio, bond, gravity = 1.0)
    density_ratio > 1 || throw(ArgumentError(
        "Rayleigh--Taylor requires rho_liquid/rho_gas > 1",
    ))
    rho_liquid = 1.0
    rho_gas = inv(float(density_ratio))
    wavelength = 1.0
    sigma = (rho_liquid - rho_gas) * gravity * wavelength^2 /
        (4pi^2 * bond)
    capillary_time = sqrt(
        (rho_liquid + rho_gas) * wavelength^3 / (2pi * sigma),
    )
    frequency = 2pi / capillary_time * sqrt(complex(1 - bond))
    return (;
        plus = frequency, minus = -frequency, sigma, capillary_time,
        rho_liquid, rho_gas,
    )
end

function rt_amplitude(time; density_ratio, bond,
        initial_amplitude = DEFAULT_WAVE_AMPLITUDE, gravity = 1.0)
    frequency = rt_frequencies(; density_ratio, bond, gravity)
    return initial_amplitude * (
        exp(-im * frequency.plus * time) +
            exp(-im * frequency.minus * time)
    ) / 2
end

function _run_instability!(
        flow, final_time; cfl, tolerance, max_cycles,
        liquid_above = false,
    )
    for phase in flow.velocity
        enforce_velocity_boundary!(
            phase, flow.solver.boundary; time = flow.time,
            geometry = flow.solver.geometry,
        )
    end
    times = Float64[flow.time]
    amplitudes = ComplexF64[
        wave_amplitude(flow.interface; liquid_above)
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
            amplitudes, wave_amplitude(flow.interface; liquid_above),
        )
    end
    return (; times, amplitudes, flow)
end

"""Run the thesis linear Kelvin--Helmholtz wave with the two-velocity solver."""
function kelvin_helmholtz(; cells_per_wavelength = 16,
        density_ratio = 1.0, weber = 0.5, velocity_jump = 1.0,
        initial_amplitude = DEFAULT_WAVE_AMPLITUDE,
        final_time = nothing, mixed_level = false,
        cfl = 0.35, tolerance = 1.0e-7, max_cycles = 3000)
    frequency = kh_frequencies(; density_ratio, weber, velocity_jump)
    duration = isnothing(final_time) ?
        (weber < 1 ? frequency.capillary_time / 4 :
            log(1.5) / abs(imag(frequency.minus))) : final_time
    mesh = _wave_mesh(cells_per_wavelength; mixed_level)
    interface_height = mixed_level ? MIXED_LEVEL_INTERFACE_HEIGHT : 0.0
    k = 2pi
    lipschitz = hypot(1, initial_amplitude * k)
    interface = InterfaceField(
        mesh,
        (x, y) -> y - interface_height - initial_amplitude * cos(k * x);
        lipschitz,
    )
    mean_velocity = (
        frequency.rho_liquid * frequency.liquid_velocity +
            frequency.rho_gas * frequency.gas_velocity
    ) / (frequency.rho_liquid + frequency.rho_gas)
    liquid_perturbation = initial_amplitude * (
        mean_velocity - frequency.liquid_velocity
    )
    gas_perturbation = initial_amplitude * (
        mean_velocity - frequency.gas_velocity
    )
    # Average of the two analytical eigenmodes at t=0.  Including these
    # decaying potential-flow perturbations is essential: piecewise constant
    # base velocities alone violate normal-velocity continuity on the sloped
    # initial interface.
    liquid = (x, y, time, axis) -> axis == 1 ?
        frequency.liquid_velocity +
            k * liquid_perturbation * cos(k * x) *
                exp(k * min(y - interface_height, 0)) :
        k * liquid_perturbation * sin(k * x) *
            exp(k * min(y - interface_height, 0))
    gas = (x, y, time, axis) -> axis == 1 ?
        frequency.gas_velocity -
            k * gas_perturbation * cos(k * x) *
                exp(-k * max(y - interface_height, 0)) :
        k * gas_perturbation * sin(k * x) *
            exp(-k * max(y - interface_height, 0))
    flow = FlowSimulation(
        mesh; density = (frequency.rho_liquid, frequency.rho_gas),
        dynamic_viscosity = (0.0, 0.0), interface,
        velocity = (liquid, gas), boundary = _free_slip_boundaries(),
        surface_tension = frequency.sigma,
        surface_tension_formulation = GFM(),
        interface_method = FrozenInterfaceFluxes(),
    )
    result = _run_instability!(
        flow, duration; cfl, tolerance, max_cycles,
    )
    analytical = [
        kh_amplitude(
            time; density_ratio, weber, initial_amplitude, velocity_jump,
        ) for time in result.times
    ]
    return (; result..., analytical, frequency)
end

"""Run the thesis linear Rayleigh--Taylor wave with the two-velocity solver."""
function rayleigh_taylor(; cells_per_wavelength = 16,
        density_ratio = 3.0, bond = 0.5, gravity = 1.0,
        initial_amplitude = DEFAULT_WAVE_AMPLITUDE,
        final_time = nothing, mixed_level = false,
        cfl = 0.35, tolerance = 1.0e-7, max_cycles = 3000)
    frequency = rt_frequencies(; density_ratio, bond, gravity)
    duration = isnothing(final_time) ?
        (bond < 1 ? frequency.capillary_time / 4 :
            log(1.5) / abs(imag(frequency.minus))) : final_time
    mesh = _wave_mesh(cells_per_wavelength; mixed_level)
    interface_height = mixed_level ? MIXED_LEVEL_INTERFACE_HEIGHT : 0.0
    k = 2pi
    lipschitz = hypot(1, initial_amplitude * k)
    # Liquid is the negative side of the level set and therefore lies above
    # the perturbed interface, as in the thesis RT configuration.
    interface = InterfaceField(
        mesh,
        (x, y) -> interface_height + initial_amplitude * cos(k * x) - y;
        lipschitz,
    )
    zero_velocity = (x, y, time, axis) -> 0.0
    flow = FlowSimulation(
        mesh; density = (frequency.rho_liquid, frequency.rho_gas),
        dynamic_viscosity = (0.0, 0.0), interface,
        velocity = (zero_velocity, zero_velocity),
        boundary = _free_slip_boundaries(),
        surface_tension = frequency.sigma,
        surface_tension_formulation = GFM(), gravity = (0.0, -gravity),
        interface_method = FrozenInterfaceFluxes(),
    )
    result = _run_instability!(
        flow, duration; cfl, tolerance, max_cycles, liquid_above = true,
    )
    analytical = [
        rt_amplitude(
            time; density_ratio, bond, initial_amplitude, gravity,
        ) for time in result.times
    ]
    return (; result..., analytical, frequency)
end

if abspath(PROGRAM_FILE) == @__FILE__
    kh = kelvin_helmholtz()
    println(
        "KH: steps=$(kh.flow.steps), final numerical amplitude=" *
            "$(last(kh.amplitudes)), analytical=$(last(kh.analytical))",
    )
    rt = rayleigh_taylor()
    println(
        "RT: steps=$(rt.flow.steps), final numerical amplitude=" *
            "$(last(rt.amplitudes)), analytical=$(last(rt.analytical))",
    )
end
