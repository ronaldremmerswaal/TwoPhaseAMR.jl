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

function production_predictor_measurement(velocity::V, dt, midpoint, force::F) where {V, F}
    # Discard the returned velocity, as production does. Benchmark argument
    # extraction outside this typed boundary is not a per-face kernel cost.
    return @timed begin
        ProbeAMR._body_force_substep!(velocity, dt, midpoint, force)
        nothing
    end
end

function frozen_control_measurements(state::S, physical::V, solver::M,
        interface::I) where {S, V, M, I}
    # These unchanged algorithms are measured behind their actual typed
    # boundary. Only old-velocity scratch is written, never the physical data.
    for repetition in 0:5
        copied = @timed begin
            ProbeAMR._copy_velocity!(state.old_velocity, physical)
            nothing
        end
        masked = @timed begin
            ProbeAMR._mark_inactive_phase_velocities!(state.old_velocity, interface)
            nothing
        end
        bounded = @timed begin
            ProbeAMR._enforce_phase_velocity_boundary!(state.old_velocity, solver, 0.0)
            nothing
        end
        iszero(repetition) && continue # warmup, not a measured repetition
        @printf("STEP_CONTROL repeat=%d copy_seconds=%.9f copy_bytes=%d mask_seconds=%.9f mask_bytes=%d boundary_including_mask_seconds=%.9f boundary_bytes=%d\n",
            repetition, copied.time, copied.bytes, masked.time, masked.bytes,
            bounded.time, bounded.bytes)
    end
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
    physical_before = map(copy, flow.velocity)
    ProbeAMR._copy_velocity!(state.convected, flow.velocity)
    before = map(copy, state.convected)
    # Same arithmetic, missing-value guard, face order and default no-force
    # argument as production. Only the typed function boundary differs.
    cached_predictor_loop!(phase, model, mesh, 0.001, 0.0, nothing)
    typed_predictor_loop!(state.convected, mesh, 0.001, 0.0, nothing)
    production_predictor_measurement(state.convected, 0.001, 0.0, nothing)
    for repetition in 1:5
        reference = @timed cached_predictor_loop!(phase, model, mesh, 0.001, 0.0, nothing)
        typed = @timed typed_predictor_loop!(state.convected, mesh, 0.001, 0.0, nothing)
        production = production_predictor_measurement(state.convected, 0.001, 0.0, nothing)
        @printf("PREDICTOR faces=%d repeat=%d cache_loop_seconds=%.9f cache_loop_bytes=%d typed_loop_seconds=%.9f typed_loop_bytes=%d production_seconds=%.9f production_bytes=%d\n",
            length(mesh.faces), repetition, reference.time, reference.bytes, typed.time, typed.bytes,
            production.time, production.bytes)
    end
    @assert all(isequal(a.interior, b.interior) && isequal(a.boundary, b.boundary)
        for (a, b) in zip(before, state.convected))
    println("PREDICTOR frozen_values_and_NaN_masks_unchanged=true")
    frozen_control_measurements(state, flow.velocity, flow.solver, interface)
    force = (x, y, time, axis) -> 0.1axis + 0.2time
    reference = map(copy, before)
    typed_predictor_loop!(reference, mesh, 0.001, 0.0, force)
    ProbeAMR._body_force_substep!(state.convected, 0.001, 0.0, force)
    @assert all(isequal(a.interior, b.interior) && isequal(a.boundary, b.boundary)
        for (a, b) in zip(reference, state.convected))
    production_predictor_measurement(state.convected, 0.001, 0.0, force)
    for repetition in 1:5
        measured = production_predictor_measurement(state.convected, 0.001, 0.0, force)
        @printf("PREDICTOR_NONZERO_FORCE repeat=%d seconds=%.9f bytes=%d\n",
            repetition, measured.time, measured.bytes)
    end
    @assert all(isequal(a.interior, b.interior) && isequal(a.boundary, b.boundary)
        for (a, b) in zip(physical_before, flow.velocity))
    println("STEP_CONTROL physical_storage_unchanged=true")
end

if abspath(PROGRAM_FILE) == @__FILE__
    frozen_overhead_probe()
end
