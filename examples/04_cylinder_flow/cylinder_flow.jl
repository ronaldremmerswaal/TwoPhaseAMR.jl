using TwoPhaseAMR

"""
    run_example(; reynolds=100, max_level=3, final_time=40, perturbation=0,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run nondimensional flow around a unit cylinder in a unit free stream.
`domain_origin`, `domain_extent`, and `base_mesh` control the computational
domain and its base grid; `vorticity_tolerance` controls wake refinement.
`perturbation` sets the strength of a localized, divergence-free initial
vortex that breaks wake symmetry without changing the boundary inflow.
Pass `start_studio=studio` after loading TwoPhaseAMRStudio to follow the run
live. `recorder_options` overrides the recording schedule; other keywords are
passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 3,
        final_time = 40.0,
        perturbation = 0.0,
        domain_origin = (-4.0, -4.0),
        domain_extent = (14.0, 8.0),
        base_mesh = (28, 16),
        vorticity_tolerance = 0.2,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    base_level = min(1, max_level)
    mesh = BlockTreeMesh(
        base_mesh = base_mesh,
        block_size = (4, 4), origin = domain_origin,
        extent = domain_extent, periodic = (false, false),
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
            velocity; refine_tolerance = vorticity_tolerance,
            coarsen_tolerance = vorticity_tolerance / 3, maximum_level = max_level,
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
    if !iszero(perturbation)
        # Localized divergence-free initial vortex breaks the symmetric wake.
        # The boundary remains the unperturbed unit inflow.
        vortex(x, y, time, axis) = begin
            dx, dy = x - 1.0, y - 0.5
            radius2 = 0.5^2
            factor = perturbation / radius2 * exp(-(dx^2 + dy^2) / (2radius2))
            axis == 1 ? 1.0 - dy * factor : dx * factor
        end
        set_velocities!(velocity, vortex)
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
    function pressure_force(state, axis)
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
                normal[axis] / scale * length
        end
        return 2force
    end
    add_diagnostic!(flow, :pressure_drag, state -> pressure_force(state, 1))
    add_diagnostic!(flow, :pressure_lift, state -> pressure_force(state, 2))
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_geometry!(fields, geometry)
    recorder = Recorder(
        fields; diagnostics = flow,
        merge(
            (;
                keyframes = EquispacedTimes(21),
                preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2),
            ),
            recorder_options,
        )...,
    )
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    result = try
        simulate!(flow, final_time; observer = recorder)
    catch
        isnothing(app) || close(app)
        rethrow()
    end
    return (; flow, result, recorder, app)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; start_studio = studio)
    else
        run = run_example()
    end
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
    if !isnothing(run.app)
        try
            wait(run.app)
        finally
            close(run.app)
        end
    end
end
