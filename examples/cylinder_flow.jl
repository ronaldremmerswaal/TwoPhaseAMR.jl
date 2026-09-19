using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=3, final_time=40,
                visualize=false, kwargs...)

Run nondimensional flow around a unit cylinder in a unit free stream.
Additional keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 3,
        final_time = 40.0,
        visualize = false,
        kwargs...,
    )
    base_level = min(1, max_level)
    mesh = BlockTreeMesh(
        block_size = (14, 8), origin = (-4.0, -4.0),
        extent = (14.0, 8.0), periodic = (false, false),
        max_level = max_level,
        refine = (key, origin, extent) -> key.level < base_level,
        adaptation = MeshAdaptationOptions(
            minimum_level = base_level, update_interval = 5,
            coarsen_delay = 5,
        ),
    )
    cylinder = (x, y) -> hypot(x, y) - 0.5
    geometry = GeometryField(
        mesh, cylinder; lipschitz = 1, atol = 1.0e-7, maxdepth = 24,
    )
    refinement = GeometryRefinementIndicator(geometry; level = max_level)
    adaptation = AMRWorkspace(mesh, geometry)
    for pass in 1:(max_level - base_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(geometry.solid, cylinder; lipschitz = 1)
        refresh_geometry!(geometry)
    end

    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = inv(reynolds),
    )
    flow_options = (;
        velocity = inflow, geometry, boundary,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > base_level ? (refinement,) : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    function pressure_drag(state)
        force = 0.0
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
            scale = hypot(normal...)
            iszero(scale) && continue
            force -= TwoPhaseAMR._get(state.pressure, ref) *
                normal[1] / scale * length
        end
        return 2force
    end
    add_diagnostic!(flow, :pressure_drag, pressure_drag)
    visualization = nothing
    observer = NoVisualization()
    if visualize
        fields = VisualizationFieldRegistry()
        register_default_fields!(fields, flow)
        register_geometry!(fields, geometry)
        register_diagnostic!(fields, flow, :pressure_drag)
        viewer = TwoPhaseAMR.visualize(
            fields; field = :velocity_vorticity, show_blocks = true,
        )
        session = VisualizationSession(
            fields; viewer, checkpoints = 21, live_dt = final_time / 200,
            asynchronous = true,
        )
        observer = session
        visualization = (; fields, viewer, session)
    end
    run_simulation() = simulate!(
        flow, final_time; visualization = observer,
    )
    result = isnothing(visualization) ? run_simulation() :
        fetch(Threads.@spawn run_simulation())
    isnothing(visualization) || wait_visualization!(visualization.session)
    return (; flow, result, visualization)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using GLMakie
    GLMakie.activate!()
    run = run_example(visualize = true)
    print_simulation_summary(run.flow, run.result)
    scene = visualization_figure(run.visualization.viewer).scene
    scene.events.window_open[] && wait(scene)
end
