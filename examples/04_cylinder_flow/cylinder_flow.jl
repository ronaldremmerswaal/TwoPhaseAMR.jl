using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=3, final_time=40, kwargs...)

Run nondimensional flow around a unit cylinder in a unit free stream.
Additional keywords are passed directly to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 3,
        final_time = 40.0,
        kwargs...,
    )
    base_level = min(1, max_level)
    mesh = BlockTreeMesh(
        base_mesh = (28, 16),
        block_size = (4, 4), origin = (-4.0, -4.0),
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
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    velocity = FaceVelocity(mesh, inflow)
    refinement = (
        GeometryRefinementIndicator(geometry; level = max_level),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = 0.2,
            coarsen_tolerance = 0.2 / 3, maximum_level = max_level,
        ),
    )
    adaptation = AMRWorkspace(mesh, geometry, velocity)
    for pass in 1:(max_level - base_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(geometry.solid, cylinder; lipschitz = 1)
        refresh_geometry!(geometry)
        set_velocities!(velocity, inflow)
    end


    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = inv(reynolds),
    )
    flow_options = (;
        velocity = velocity, geometry, boundary,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > base_level ? refinement : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    function pressure_drag(state)
        # The impulsive-start sample is not a developed force. At the first
        # t=1e-6 step the smoke test measured raw force 691362 (>100), so a
        # magnitude cutoff would also discard valid completed-step samples.
        iszero(state.time) && return NaN
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
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_geometry!(fields, geometry)
    recorder = Recorder(
        fields; diagnostics = flow, keyframes = EquispacedTimes(21), preview = nothing,
    )
    result = simulate!(flow, final_time; observer = recorder)
    return (; flow, result, recorder)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = run_example()
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
end
