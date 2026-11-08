using TwoPhaseAMR

"""
    run_example(; density_ratio=3, weber=0.5, bond=0.75,
                max_level=1, final_time=nothing,
                velocity_model=TwoVelocityModel(...), kwargs...)

Run a combined Kelvin--Helmholtz/Rayleigh--Taylor wave. The physical default
is `TwoVelocityModel(interface_transport=FrozenInterfaceFluxes(),
pressure_projection=CMDGFMProjection())`. Passing `OneVelocityModel()` uses
the density-weighted mean of the phase-specific initial velocities and thus
represents a different closure. Additional keywords are forwarded directly to
[`FlowSimulation`](@ref).
"""
function run_example(;
        density_ratio = 3.0,
        weber = 0.5,
        bond = 0.75,
        max_level = 1,
        final_time = nothing,
        velocity_model = TwoVelocityModel(
            interface_transport = FrozenInterfaceFluxes(),
            pressure_projection = CMDGFMProjection(),
        ),
        kwargs...,
    )
    wavelength = 1.0
    k = 2pi / wavelength
    amplitude = 0.01
    density = (1.0, inv(density_ratio))
    density_sum = sum(density)
    surface_tension = density_sum / k^3
    velocity_jump = density_sum * sqrt(weber / prod(density)) / k
    phase_speed = (
        -density[2] * velocity_jump / density_sum,
        density[1] * velocity_jump / density_sum,
    )
    gravity = bond * density_sum /
        (k * (density[1] - density[2]))
    levelset = (x, y) -> amplitude * cos(k * x) - y

    mesh = BlockTreeMesh(
        base_mesh = (8, 24), block_size = (8, 8),
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2,
        ),
    )
    interface = InterfaceField(mesh, levelset)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset)
    end

    function phase_velocity(base, phase_above)
        perturbation = -amplitude * base
        return (x, y, time, axis) -> if axis == 1
            base + (phase_above ? -1 : 1) * k * perturbation * cos(k * x) *
                exp(-k * abs(y))
        else
            k * perturbation * sin(k * x) * exp(-k * abs(y))
        end
    end
    liquid = phase_velocity(phase_speed[1], true)
    gas = phase_velocity(phase_speed[2], false)
    velocity = if velocity_model isa TwoVelocityModel
        (liquid, gas)
    else
        (x, y, time, axis) -> (
            density[1] * liquid(x, y, time, axis) +
                density[2] * gas(x, y, time, axis)
        ) / density_sum
    end
    physics = TwoPhase(
        interface; density, dynamic_viscosity = (0.0, 0.0),
        surface_tension, gravity = (0.0, -gravity),
    )
    radicand = 1 - weber - bond
    analytical_amplitude(time) = amplitude * if radicand < 0
        cosh(sqrt(-radicand) * time)
    else
        cos(sqrt(radicand) * time)
    end
    flow_options = (;
        velocity_model, velocity, boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(:amplitude, state -> interface_height(state.interface);
                label = "Numerical amplitude", dimension = PhysicalDimension(0, 1, 0)),
            Diagnostic(
                :analytical_amplitude,
                state -> analytical_amplitude(state.time);
                label = "Linear amplitude", dimension = PhysicalDimension(0, 1, 0),
            ),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > 0 ? (refinement,) : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    default_time = radicand < 0 ? acosh(2) / sqrt(-radicand) :
        2pi / sqrt(radicand)
    target_time = isnothing(final_time) ? default_time : final_time
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
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
