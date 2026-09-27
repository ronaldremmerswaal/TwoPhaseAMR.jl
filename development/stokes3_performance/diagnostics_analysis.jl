# Focused diagnostics measurements on Stokes3, before/after production changes.
# From the repository root:
#   julia --project=. development/stokes3_performance/diagnostics_analysis.jl one frozen
#   julia --project=. development/stokes3_performance/diagnostics_analysis.jl cmdgfm frozen
#   julia --project=. development/stokes3_performance/diagnostics_analysis.jl one paired
#   julia --project=. development/stokes3_performance/diagnostics_analysis.jl cmdgfm paired
# `frozen` measures identical initial interfaces, then evolved AMR states.
# `paired` times complete t=0.05 advances with and without diagnostics.
using TwoPhaseAMR
using LinearAlgebra: BLAS
using Printf
using Profile
using Statistics: median, quantile

include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

function model_for(mode)
    mode == :one && return OneVelocityModel()
    mode == :cmdgfm && return TwoVelocityModel(
        pressure_projection = CMDGFMProjection(),
    )
    throw(ArgumentError("mode must be one or cmdgfm"))
end

function timed_calls(label, f; repetitions = 12)
    f() # Compile this specific operation before measuring it.
    GC.gc()
    samples = [@timed f() for _ in 1:repetitions]
    times = [sample.time for sample in samples]
    bytes = [sample.bytes for sample in samples]
    value = samples[end].value
    displayed_value = value isa AbstractArray && length(value) > 8 ?
        summary(value) : repr(value)
    @printf("CALL %-29s median_ms=%9.3f p95_ms=%9.3f median_bytes=%10.0f p95_bytes=%10.0f reps=%d value=%s\n",
        label, 1e3 * median(times), 1e3 * quantile(times, 0.95),
        median(bytes), quantile(bytes, 0.95), repetitions,
        displayed_value)
    return nothing
end

function fraction_checksum(state)
    model = state.solver.two_phase
    total = 0.0
    for face in model.interface.mesh.faces
        total += TwoPhaseAMR._staggered_liquid_fraction(
            model.interface, state.geometry, face,
        )
    end
    return total
end

function fraction_checksum_barrier(field::InterfaceField, geometry)
    total = 0.0
    for face in field.mesh.faces
        total += TwoPhaseAMR._staggered_liquid_fraction(
            field, geometry, face,
        )
    end
    return total
end

function frozen_state(mode, label, flow)
    state = simulation_state(flow)
    mesh = state.solver.mesh
    T = eltype(mesh.origin)
    @printf("STATE mode=%s label=%s time=%.5f steps=%d updates=%d blocks=%d cells=%d faces=%d geometry=%s\n",
        mode, label, flow.time, flow.steps, flow.mesh_updates,
        nblocks(mesh), ncells(mesh), length(mesh.faces),
        typeof(state.geometry))
    diagnostics = (
        KineticEnergy(), LinearMomentum(), PhaseVolume(),
        DivergenceNorm(), LeafBlockCount(),
    )
    for diagnostic in diagnostics
        timed_calls(string(diagnostic_name(diagnostic)),
            () -> diagnostic_value(diagnostic, state))
    end
    timed_calls("face_dual_volumes", () ->
        TwoPhaseAMR._diagnostic_face_volumes(state))
    diffusion = state.workspace.two_phase.diffusion
    if !isnothing(diffusion)
        reference = TwoPhaseAMR._diagnostic_face_volumes(state)
        existing = diffusion.volumes
        length(reference) == length(existing) || error(
            "diffusion and diagnostic volume indexing differs",
        )
        @printf("EXISTING_VOLUMES location=two_phase_diffusion length=%d exact=%s max_abs_difference=%.3e\n",
            length(existing), isequal(existing, reference),
            maximum(abs, existing .- reference))
        for axis in 1:2
            component = axis == 1 ? state.workspace.two_phase.momentum.x :
                state.workspace.two_phase.momentum.y
            schedule = component.advection.schedule
            if hasproperty(schedule, :volumes)
                indices = [index for (index, face) in pairs(mesh.faces)
                    if face.axis == axis]
                interior = view(schedule.volumes, 1:length(indices))
                @printf("EXISTING_VOLUMES location=momentum_schedule axis=%d interior_length=%d exact=%s max_abs_difference=%.3e\n",
                    axis, length(indices),
                    isequal(interior, view(reference, indices)),
                    maximum(abs, interior .- view(reference, indices)))
            end
        end
    end
    if isnothing(state.geometry)
        components = ntuple(axis -> TwoPhaseAMR._amr_dual_schedule(
            mesh, Val(axis), T; build_three_point_stencils = false,
        ).volumes, 2)
        for axis in 1:2
            timed_calls("dual_schedule_axis_$axis", () ->
                TwoPhaseAMR._amr_dual_schedule(
                    mesh, Val(axis), T;
                    build_three_point_stencils = false,
                ).volumes)
        end
        timed_calls("global_face_mapping", () ->
            TwoPhaseAMR._component_volumes_by_global_face(
                mesh, components, T,
            ))
    end
    if mode == :one
        model = state.solver.two_phase
        timed_calls("mixture_density", () ->
            TwoPhaseAMR.mixture_density(model.interface, model.density))
    else
        timed_calls("staggered_fraction_scan", () -> fraction_checksum(state))
        model = state.solver.two_phase
        timed_calls("fraction_scan_barrier", () ->
            fraction_checksum_barrier(model.interface, state.geometry))
    end
    flush(stdout)
    return nothing
end

function frozen(mode; final_time = 0.05)
    model = model_for(mode)
    @printf("ENV mode=%s analysis=frozen julia=%s threads=%d blas_threads=%d\n",
        mode, VERSION, Threads.nthreads(), BLAS.get_num_threads())
    initial = run_example(velocity_model = model, final_time = 0.0).flow
    frozen_state(mode, "initial", initial)
    evolved = run_example(
        velocity_model = model, final_time = final_time,
    ).flow
    evolved.mesh_updates > 0 || error("evolved case must include AMR updates")
    frozen_state(mode, "evolved", evolved)
    return nothing
end

function advance_case(model, with_diagnostics, final_time)
    specs = with_diagnostics ? (
        KineticEnergy(), LinearMomentum(), PhaseVolume(),
        DivergenceNorm(), LeafBlockCount(),
    ) : ()
    flow = run_example(
        velocity_model = model, diagnostics = specs, final_time = 0.0,
    ).flow
    measured = @timed simulate!(flow, final_time)
    @printf("ADVANCE diagnostics=%s wall=%.3f gc=%.3f compile=%.3f bytes=%d steps=%d updates=%d blocks=%d faces=%d series=%d\n",
        with_diagnostics, measured.time, measured.gctime,
        measured.compile_time, measured.bytes, measured.value.steps,
        measured.value.mesh_updates, length(flow.solver.mesh.blocks),
        length(flow.solver.mesh.faces), length(diagnostic_names(flow)))
    flush(stdout)
    return measured
end

function paired(mode; final_time = 0.05)
    model = model_for(mode)
    @printf("ENV mode=%s analysis=paired julia=%s threads=%d blas_threads=%d\n",
        mode, VERSION, Threads.nthreads(), BLAS.get_num_threads())
    for enabled in (true, false)
        advance_case(model, enabled, 0.005) # compilation warmup
    end
    for enabled in (true, false, false, true)
        advance_case(model, enabled, final_time)
    end
    return nothing
end

function allocation_trace(mode)
    model = model_for(mode)
    flow = run_example(velocity_model = model, final_time = 0.0).flow
    state = simulation_state(flow)
    for diagnostic in (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        )
        diagnostic_value(diagnostic, state)
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 begin
            diagnostic_value(diagnostic, state)
        end
        samples = Profile.Allocs.fetch().allocs
        counts = Dict{String, Tuple{Int, Int}}()
        for sample in samples
            relevant = findfirst(frame ->
                occursin("TwoPhaseAMR.jl/src/", string(frame.file)),
                sample.stacktrace)
            frame = isnothing(relevant) ?
                (isempty(sample.stacktrace) ? nothing : last(sample.stacktrace)) :
                sample.stacktrace[relevant]
            location = isnothing(frame) ? "unknown" :
                string(frame.func, "@", basename(string(frame.file)),
                    ":", frame.line)
            calls, bytes = get(counts, location, (0, 0))
            counts[location] = (calls + 1, bytes + sample.size)
        end
        @printf("ALLOC_TRACE mode=%s diagnostic=%s events=%d bytes=%d\n",
            mode, diagnostic_name(diagnostic), length(samples),
            sum(sample.size for sample in samples; init = 0))
        for (location, (calls, bytes)) in
                sort!(collect(counts); by = entry -> entry.second[2],
                    rev = true)
            @printf("ALLOC_SITE calls=%d bytes=%d location=%s\n",
                calls, bytes, location)
        end
    end
    return nothing
end

function cut_allocations(mode)
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
    )
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25;
        lipschitz = 1,
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.5; geometry, lipschitz = 1,
    )
    physics = TwoPhase(
        interface; density = (10.0, 1.0),
        dynamic_viscosity = (0.1, 0.1), surface_tension = 0.0,
    )
    flow = FlowSimulation(
        mesh, physics; geometry, velocity_model = model_for(mode),
        diagnostics = (),
    )
    state = simulation_state(flow)
    for diagnostic in (
            KineticEnergy(), LinearMomentum(), PhaseVolume(),
            DivergenceNorm(), LeafBlockCount(),
        )
        timed_calls(string(diagnostic_name(diagnostic)),
            () -> diagnostic_value(diagnostic, state))
    end
    return nothing
end

function history_allocations(mode)
    flow = run_example(velocity_model = model_for(mode), final_time = 0.0).flow
    for series in flow.diagnostics.series
        sizehint!(series.times, 32)
        sizehint!(series.steps, 32)
        sizehint!(series.values, 32)
    end
    for step in 1:8
        state = SimulationState(
            flow.velocity, flow.pressure, flow.solver,
            flow.time + step * 1.0e-6, step, flow.workspace,
        )
        # The first two events compile the collection's dynamic dispatch.
        if step <= 2
            TwoPhaseAMR._record_diagnostics!(
                flow.diagnostics, state; event = :step,
            )
        else
            bytes = @allocated TwoPhaseAMR._record_diagnostics!(
                flow.diagnostics, state; event = :step,
            )
            @printf("HISTORY mode=%s step=%d bytes=%d\n", mode, step, bytes)
            if step == 8
                next_state = SimulationState(
                    flow.velocity, flow.pressure, flow.solver,
                    flow.time + 9.0e-6, 9, flow.workspace,
                )
                Profile.Allocs.clear()
                Profile.Allocs.@profile sample_rate=1.0 begin
                    TwoPhaseAMR._record_diagnostics!(
                        flow.diagnostics, next_state; event = :step,
                    )
                end
                counts = Dict{String, Tuple{Int, Int}}()
                samples = Profile.Allocs.fetch().allocs
                for (sample_index, sample) in pairs(samples)
                    if sample_index <= 8
                        println("HISTORY_SAMPLE size=", sample.size,
                            " type=", sample.type, " stack=",
                            [string(frame.func, "@", basename(string(frame.file)),
                                ":", frame.line) for frame in
                                    sample.stacktrace[1:min(end, 5)]])
                    end
                    frame = findfirst(item -> occursin(
                        "TwoPhaseAMR.jl/src/", string(item.file),
                    ), sample.stacktrace)
                    relevant = isnothing(frame) ? last(sample.stacktrace) :
                        sample.stacktrace[frame]
                    location = string(relevant.func, "@",
                        basename(string(relevant.file)), ":", relevant.line)
                    calls, bytes_at_location = get(counts, location, (0, 0))
                    counts[location] = (
                        calls + 1, bytes_at_location + sample.size,
                    )
                end
                for (location, (calls, location_bytes)) in
                        sort!(collect(counts); by = x -> x.second[2], rev = true)
                    @printf("HISTORY_ALLOC calls=%d bytes=%d location=%s\n",
                        calls, location_bytes, location)
                end
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(get(ARGS, 1, "one"))
    action = Symbol(get(ARGS, 2, "frozen"))
    if action == :frozen
        frozen(mode)
    elseif action == :paired
        paired(mode)
    elseif action == :trace
        allocation_trace(mode)
    elseif action == :cut
        cut_allocations(mode)
    elseif action == :history
        history_allocations(mode)
    else
        throw(ArgumentError("action must be frozen, paired, trace or cut"))
    end
end
