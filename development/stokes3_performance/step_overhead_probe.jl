# Frozen, read-only diagnosis of the unclassified predictor loop.
# Only workspace scratch is modified. No production method is replaced.
# julia --project=. development/stokes3_performance/step_overhead_probe.jl
using TwoPhaseAMR, Printf
const ProbeAMR = TwoPhaseAMR

function cached_predictor_loop!(phase, model, mesh, dt, midpoint, force)
    state = ProbeAMR._two_velocity_flow_workspace!(phase, model)
    for phase_index in 1:2
        for (index, face) in pairs(mesh.faces)
            isnan(state.convected[phase_index].interior[index]) && continue
            @inbounds state.convected[phase_index].interior[index] +=
                dt * ProbeAMR._body_force(force, face, midpoint)
        end
    end
    return nothing
end

function typed_predictor_loop!(velocity, mesh, dt, midpoint, force)
    for phase_index in 1:2
        for (index, face) in pairs(mesh.faces)
            isnan(velocity[phase_index].interior[index]) && continue
            @inbounds velocity[phase_index].interior[index] +=
                dt * ProbeAMR._body_force(force, face, midpoint)
        end
    end
    return nothing
end

function frozen_overhead_probe()
    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (true, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.47)
    flow = FlowSimulation(mesh, TwoPhase(interface; density = (1000.0, 1.0),
        dynamic_viscosity = (0.0, 0.0));
        velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection()),
        velocity = ((x, y, t, a) -> a == 1 ? 0.05 : 0.0,
            (x, y, t, a) -> a == 1 ? -0.02 : 0.0), boundary = FreeSlip())
    phase, model = flow.workspace.two_phase, flow.solver.two_phase
    state = ProbeAMR._two_velocity_flow_workspace!(phase, model)
    ProbeAMR._copy_velocity!(state.convected, flow.velocity)
    before = map(copy, state.convected)
    # Same arithmetic, missing-value guard, face order and default no-force
    # argument as production. Only the typed function boundary differs.
    cached_predictor_loop!(phase, model, mesh, 0.001, 0.0, nothing)
    typed_predictor_loop!(state.convected, mesh, 0.001, 0.0, nothing)
    for repetition in 1:5
        reference = @timed cached_predictor_loop!(phase, model, mesh, 0.001, 0.0, nothing)
        typed = @timed typed_predictor_loop!(state.convected, mesh, 0.001, 0.0, nothing)
        @printf("PREDICTOR faces=%d repeat=%d cache_loop_seconds=%.9f cache_loop_bytes=%d typed_loop_seconds=%.9f typed_loop_bytes=%d\n",
            length(mesh.faces), repetition, reference.time, reference.bytes, typed.time, typed.bytes)
    end
    @assert all(isequal(a.interior, b.interior) && isequal(a.boundary, b.boundary)
        for (a, b) in zip(before, state.convected))
    println("PREDICTOR frozen_values_and_NaN_masks_unchanged=true")
end

if abspath(PROGRAM_FILE) == @__FILE__
    frozen_overhead_probe()
end
