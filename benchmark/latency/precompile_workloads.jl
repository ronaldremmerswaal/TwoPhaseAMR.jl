# Experimental source: include only from a disposable package copy, never src/.
# Set TWOPHASE_LATENCY_WORKLOAD to tg, one, two, all, tg_stable, tg_setup,
# one_setup, two_setup, one_amr, one_amr_setup, one_amr_step or one_amr_rebuild.
# Change this file in the copy between variants to invalidate the old image.
using PrecompileTools: @setup_workload, @compile_workload

function latency_experiment(::Val{mode}) where {mode}
    println("LATENCY_EXPERIMENT begin ", mode)
    single_phase = mode in (:tg, :tg_stable, :tg_setup)
    with_amr = mode in (:one_amr, :one_amr_setup, :one_amr_step, :one_amr_rebuild)
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (8, 8),
        max_level = with_amr ? 1 : 0, periodic = (true, single_phase)
    )
    dt = 0.001
    if single_phase
        physics = SinglePhase(density = 1.0, dynamic_viscosity = 0.02)
        velocity = (x, y, time, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
        flow = FlowSimulation(
            mesh, physics; velocity, boundary = FreeSlip(), dt,
            diagnostics = (KineticEnergy(), DivergenceNorm())
        )
    else
        levelset = (x, y) -> y - 0.5 - 0.01cospi(2x)
        interface = InterfaceField(mesh, levelset; lipschitz = 2)
        physics = TwoPhase(
            interface; density = (1.0, 0.001),
            dynamic_viscosity = (0.01, 0.00001), surface_tension = 0.01
        )
        model = mode in (:two, :two_setup) ?
            TwoVelocityModel(pressure_projection = CMDGFMProjection()) : OneVelocityModel()
        if with_amr
            refinement = InterfaceRefinementIndicator(interface; level = 1)
            adaptation = AMRWorkspace(mesh, interface)
            if adapt!(adaptation, refinement; step = 1, force = true).changed
                initialize!(interface, levelset; lipschitz = 2)
            end
            flow = FlowSimulation(
                mesh, physics; velocity_model = model, boundary = FreeSlip(), dt,
                diagnostics = (PhaseVolume(), DivergenceNorm()),
                adaptivity = Adaptivity(criteria = (refinement,))
            )
        else
            flow = FlowSimulation(
                mesh, physics; velocity_model = model, boundary = FreeSlip(), dt,
                diagnostics = (PhaseVolume(), DivergenceNorm())
            )
        end
    end
    println("LATENCY_EXPERIMENT constructed ", mode)
    if mode === :tg_stable
        stable_timestep(flow; cfl = 0.45)
    elseif !(mode in (:tg_setup, :one_setup, :two_setup, :one_amr_setup, :one_amr_rebuild))
        simulate!(flow, dt; dt)
    end
    if mode in (:one_amr, :one_amr_rebuild)
        _rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt)
    end
    println("LATENCY_EXPERIMENT workload complete ", mode)
    return nothing
end

@setup_workload begin
    mode = Symbol(get(ENV, "TWOPHASE_LATENCY_WORKLOAD", "tg"))
    mode in (
        :tg, :one, :two, :all, :tg_stable, :tg_setup, :one_setup, :two_setup,
        :one_amr, :one_amr_setup, :one_amr_step, :one_amr_rebuild,
    ) ||
        error("unknown latency workload: $mode")
    if mode === :all
        @compile_workload begin
            latency_experiment(Val(:tg))
            latency_experiment(Val(:one))
            latency_experiment(Val(:two))
        end
    else
        @compile_workload latency_experiment(Val(mode))
    end
end
