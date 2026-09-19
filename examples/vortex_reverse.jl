using TwoPhaseAMR

"""
    run_example(; max_level=5, final_time=8, visualize=false, kwargs...)

Run the classical direction-reversing VOF transport problem. This prescribed
carrier-flow example does not construct a Navier--Stokes `FlowSimulation`.
Additional keywords are passed to [`InterfaceField`](@ref).
"""
function run_example(;
        max_level = 5,
        final_time = 8.0,
        visualize = false,
        kwargs...,
    )
    levelset = (x, y) -> hypot(x - 0.5, y - 0.75) - 0.15
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2,
        ),
    )
    interface_options = (;
        lipschitz = 1, atol = 1.0e-7, maxdepth = 24, kwargs...,
    )
    interface = InterfaceField(mesh, levelset; interface_options...)
    velocity = FaceVelocity(mesh)
    refinement = (
        InterfaceRefinementIndicator(
            interface; level = min(1, max_level),
        ),
        CurvatureRefinementIndicator(
            interface; cells_per_radius = 8,
            minimum_level = min(1, max_level),
        ),
    )
    adaptation = AMRWorkspace(mesh, interface, velocity)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset; interface_options...)
    end
    initial_volume = integrate(interface)
    workspace = WYAdvectionWorkspace(interface)
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(
            fields, :volume_fraction, interface, :velocity, velocity,
        )
        viewer = TwoPhaseAMR.visualize(
            fields; field = :volume_fraction, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 21, live_dt = final_time / 200,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    streamfunction(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
    function set_velocity!(time)
        for (index, face) in pairs(mesh.faces)
            x, y = face.center
            half_width = face.area / 2
            value = if face.axis == 1
                (streamfunction(x, y + half_width) -
                    streamfunction(x, y - half_width)) / face.area
            else
                -(streamfunction(x + half_width, y) -
                    streamfunction(x - half_width, y)) / face.area
            end
            velocity.interior[index] = cospi(time / final_time) * value
        end
        fill!(velocity.boundary, 0)
    end

    time = 0.0
    steps = 0
    mesh_updates = 0
    initialize_visualization!(
        observer; time, step = steps, final_time,
        velocity, pressure = nothing, solver = nothing,
    )
    while time < final_time - 16eps(final_time)
        set_velocity!(0.0)
        dt = min(
            stable_timestep(velocity, workspace; cfl = 0.4),
            final_time - time,
        )
        set_velocity!(time + dt / 2)
        advect!(
            interface, velocity, dt, workspace;
            sweep_order = iseven(steps) ? (1, 2) : (2, 1),
        )
        time += dt
        steps += 1
        if adapt!(adaptation, refinement; step = steps, time).changed
            workspace = WYAdvectionWorkspace(interface)
            mesh_updates += 1
        end
        update_visualization!(
            observer; event = :step, time, step = steps, final_time,
            velocity, pressure = nothing, solver = nothing,
        )
    end
    update_visualization!(
        observer; event = :completed, time, step = steps, final_time,
        velocity, pressure = nothing, solver = nothing,
    )
    isnothing(visualization) || wait_visualization!(visualization.session)
    result = (;
        time, steps, mesh_updates,
        volume_error = abs(integrate(interface) - initial_volume),
    )
    return (; interface, velocity, result, visualization)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    GLMakie.activate!()
    run = run_example(visualize = true)
    println(run.result)
    scene = visualization_figure(run.visualization.viewer).scene
    scene.events.window_open[] && wait(scene)
end
