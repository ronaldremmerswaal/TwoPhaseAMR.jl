using TwoPhaseAMR

"""Build the geometrically initialized vortex-reversal transport problem."""
function vortex_reverse_setup(;
        base_cells = 8, max_level = 5, interface_level = 1,
        cells_per_radius = 8.0, period = 8.0,
        adaptive = true, adaptation_interval = 5, cfl = 0.4
    )
    base_cells >= 2 || throw(ArgumentError("base_cells must be at least two"))
    0 <= interface_level <= max_level ||
        throw(ArgumentError("interface_level must lie within mesh limits"))
    adaptation_interval >= 1 ||
        throw(ArgumentError("adaptation_interval must be positive"))
    period > 0 || throw(ArgumentError("period must be positive"))
    0 < cfl <= 0.5 || throw(ArgumentError("WY requires 0 < cfl <= 0.5"))

    levelset = (x, y) -> hypot(x - 0.5, y - 0.75) - 0.15
    streamfunction(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
    mesh = BlockTreeMesh(
        block_size = (base_cells, base_cells), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = adaptation_interval, coarsen_delay = 2
        )
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    velocity = FaceVelocity(mesh)
    interface_indicator = InterfaceRefinementIndicator(
        interface; level = interface_level
    )
    curvature_indicator = CurvatureRefinementIndicator(
        interface; cells_per_radius, minimum_level = interface_level
    )
    indicators = (interface_indicator, curvature_indicator)
    adaptation = AMRWorkspace(mesh, interface, velocity)

    if adaptive && max_level > 0
        for step in 1:(max_level + 1)
            stats = adapt!(adaptation, indicators; step, force = true)
            stats.changed || break
            # Analytical resampling is confined to construction. Regridding
            # during transport uses conservative geometric PLIC transfer.
            initialize!(interface, levelset; lipschitz = 1)
        end
    end

    liquid_scalar = (x, y) -> 1 + 0.1sinpi(2x)
    gas_scalar = (x, y) -> 2 + 0.1cospi(2y)
    scalar = PhaseScalar(
        interface; liquid = liquid_scalar, gas = gas_scalar
    )
    unit_scalar = PhaseScalar(interface)
    staggered_scalar = PhaseFaceScalar{2}(
        interface; liquid = liquid_scalar, gas = gas_scalar
    )
    unit_staggered_scalar = PhaseFaceScalar{1}(interface)
    for field in (
            scalar, unit_scalar, staggered_scalar, unit_staggered_scalar,
        )
        register_adaptive_field!(adaptation, field)
    end

    initial_volume = integrate(interface)
    return (;
        mesh, interface, velocity, interface_indicator,
        curvature_indicator, indicators, adaptation, levelset,
        streamfunction, scalar, unit_scalar, staggered_scalar,
        unit_staggered_scalar, liquid_scalar, gas_scalar,
        initial_volume, adaptive, cfl, period,
    )
end

function _set_vortex_velocity!(velocity, streamfunction, time, period)
    mesh = velocity.mesh
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        half_width = face.area / 2
        velocity.interior[index] = cospi(time / period) * if face.axis == 1
            (
                streamfunction(x, y + half_width) -
                    streamfunction(x, y - half_width)
            ) / face.area
        else
            -(
                streamfunction(x + half_width, y) -
                    streamfunction(x - half_width, y)
            ) / face.area
        end
    end
    fill!(velocity.boundary, zero(eltype(velocity)))
    return velocity
end

function _vortex_reverse_errors(case, time)
    isapprox(
        time, case.period;
        atol = 64eps(typeof(time)) * max(1, case.period), rtol = 0
    ) || return (nothing, nothing)
    reference = InterfaceField(case.mesh, case.levelset; lipschitz = 1)
    reference_scalar = PhaseScalar(
        reference;
        liquid = case.liquid_scalar, gas = case.gas_scalar
    )
    interface_error = zero(eltype(case.interface))
    scalar_error = zero(eltype(case.interface))
    for (block_index, block) in pairs(case.mesh.blocks)
        fraction = interior(reference, block_index)
        interface_error += block.volume * sum(
            abs.(
                interior(case.interface, block_index) .-
                    interior(reference, block_index)
            )
        )
        scalar_error += block.volume * sum(
            fraction .* abs.(
                interior(case.scalar.liquid, block_index) .-
                    interior(reference_scalar.liquid, block_index)
            ) +
                (1 .- fraction) .* abs.(
                interior(case.scalar.gas, block_index) .-
                    interior(reference_scalar.gas, block_index)
            )
        )
    end
    return interface_error, scalar_error
end

function _vortex_unit_error(case)
    error = zero(eltype(case.interface))
    for field in (
            case.unit_scalar.liquid, case.unit_scalar.gas,
        )
        for block in eachindex(case.mesh.blocks)
            error = max(
                error,
                maximum(abs.(interior(field, block) .- 1))
            )
        end
    end
    for field in (
            case.unit_staggered_scalar.liquid,
            case.unit_staggered_scalar.gas,
        )
        error = max(
            error,
            maximum(abs.(field.interior .- 1); init = zero(error))
        )
    end
    return error
end

"""
    advance_vortex_reverse!(case, final_time=case.period;
                            visualization=NoVisualization())

Advect the interface with the directionally split Weymouth--Yue method. The
prescribed divergence-free velocity reverses at half the selected period, so
one period returns the continuum interface to its initial position. The WY
incoming-volume CFL is evaluated with the peak velocity and the flow is sampled
at the step midpoint.
"""
function advance_vortex_reverse!(
        case, final_time = case.period;
        visualization::AbstractVisualization = NoVisualization()
    )
    final_time > 0 || throw(ArgumentError("final_time must be positive"))
    time = zero(float(final_time))
    step = 0
    mesh_updates = 0
    work = WYAdvectionWorkspace(case.interface)
    initialize_visualization!(
        visualization; time, step, final_time,
        velocity = case.velocity, pressure = nothing, solver = nothing
    )
    try
        while time < final_time - 16eps(typeof(time)) * max(1, final_time)
            # The peak value bounds |cos(pi*t)| over every step, including
            # steps which straddle the reversal.
            _set_vortex_velocity!(
                case.velocity, case.streamfunction, 0.0, case.period
            )
            dt = min(
                stable_timestep(case.velocity, work; cfl = case.cfl),
                final_time - time
            )
            _set_vortex_velocity!(
                case.velocity, case.streamfunction,
                time + dt / 2, case.period
            )
            advect!(
                case.interface, case.velocity, dt, work;
                scalars = (case.scalar, case.unit_scalar),
                staggered_scalars = (
                    case.staggered_scalar,
                    case.unit_staggered_scalar,
                ),
                sweep_order = iseven(step) ? (1, 2) : (2, 1),
                scheme = Fromm()
            )
            time = min(final_time, time + dt)
            step += 1

            if case.adaptive
                stats = adapt!(
                    case.adaptation, case.indicators; step, time
                )
                if stats.changed
                    mesh_updates += 1
                    work = WYAdvectionWorkspace(case.interface)
                    _set_vortex_velocity!(
                        case.velocity, case.streamfunction,
                        time, case.period
                    )
                end
            end
            update_visualization!(
                visualization; event = :step, time, step, final_time,
                velocity = case.velocity, pressure = nothing, solver = nothing
            )
        end
        update_visualization!(
            visualization; event = :completed, time, step,
            final_time = time, velocity = case.velocity,
            pressure = nothing, solver = nothing
        )
    catch exception
        try
            update_visualization!(
                visualization; event = :failed, time, step, final_time,
                velocity = case.velocity, pressure = nothing,
                solver = nothing, exception
            )
        catch
        end
        rethrow()
    end

    reverse_error, scalar_reverse_error = _vortex_reverse_errors(case, time)
    return (;
        time, steps = step, mesh_updates,
        volume_error = abs(integrate(case.interface) - case.initial_volume),
        reverse_error, scalar_reverse_error,
        unit_error = _vortex_unit_error(case),
    )
end

"""Build and run the vortex-reversal example without opening a viewer."""
function vortex_reverse(; period = 8.0, final_time = nothing, kwargs...)
    case = vortex_reverse_setup(; period, kwargs...)
    target = isnothing(final_time) ? case.period : final_time
    result = advance_vortex_reverse!(case, target)
    return (; case..., result)
end

"""
    visualize_vortex_reverse(; period=8, checkpoints=21, ...)

Run the classical vortex-reversal test with a live volume-fraction and
velocity viewer. GLMakie must be loaded by the caller.
"""
function visualize_vortex_reverse(;
        period = 8.0, final_time = nothing, checkpoints = 21,
        live_dt = nothing, display = true,
        asynchronous = true, viewer = nothing, kwargs...
    )
    case = vortex_reverse_setup(; period, kwargs...)
    target = isnothing(final_time) ? case.period : final_time
    target > 0 || throw(ArgumentError("final_time must be positive"))
    selected_live_dt = isnothing(live_dt) ? target / 200 : live_dt
    isnothing(viewer) && Threads.nthreads() == 1 && @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI"
    )
    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields,
        :volume_fraction, case.interface,
        :curvature, case.interface.curvature,
        :liquid_scalar, case.scalar.liquid,
        :gas_scalar, case.scalar.gas,
        :liquid_staggered_scalar, case.staggered_scalar.liquid,
        :gas_staggered_scalar, case.staggered_scalar.gas,
        :velocity, case.velocity
    )
    register_diagnostic!(
        fields, :block_count,
        (velocity, pressure, time, step, solver) -> nblocks(case.mesh);
        label = "Leaf blocks", color = :darkorange
    )
    selected_viewer = isnothing(viewer) ? visualize(
            fields; field = :volume_fraction, show_blocks = true,
            show_cells = false, display,
            figure = (; size = (1000, 760))
        ) : viewer
    session = VisualizationSession(
        fields; viewer = selected_viewer, checkpoints,
        live_dt = selected_live_dt,
        asynchronous
    )
    simulation = Threads.@spawn advance_vortex_reverse!(
        case, target; visualization = session
    )
    return (;
        case..., fields, viewer = selected_viewer, session, simulation,
        final_time = target,
    )
end

function wait_for_vortex_reverse!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie

    GLMakie.activate!()
    run = visualize_vortex_reverse()
    println(
        "Started vortex reversal through t=$(run.final_time) on " *
            "$(nblocks(run.mesh)) blocks."
    )
    result = wait_for_vortex_reverse!(run)
    println(
        "Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates. Interface/scalar reverse L1 errors=" *
            "$(result.reverse_error)/$(result.scalar_reverse_error). " *
            "Close the window to exit."
    )
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
