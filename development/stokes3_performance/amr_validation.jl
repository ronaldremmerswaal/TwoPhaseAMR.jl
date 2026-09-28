# Compare an unmodified-source snapshot with the optimized production advance.
# Use separate environments/processes pointing to the respective sources.
# Snapshot files contain plain arrays, tuples and diagnostic data, not cached
# mesh/workspace objects. Setup/warmup and snapshot I/O are outside the timer.
#
# julia --project=BASELINE_ENV amr_validation.jl capture /tmp/amr-reference 0.08
# julia --project=.            amr_validation.jl compare /tmp/amr-reference 0.08
using TwoPhaseAMR, Serialization, Printf, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

owned_values(field) = reduce(vcat, (vec(collect(interior(field, b)))
    for b in eachindex(field.mesh.blocks)))

function validation_snapshot(flow, result)
    velocities = flow.velocity isa FaceVelocity ? (flow.velocity,) : flow.velocity
    histories = Dict(name => let history = diagnostic_history(flow, name)
        (; times = history.times, steps = history.steps, values = history.values)
    end for name in diagnostic_names(flow))
    return (;
        time = result.time, steps = result.steps, updates = result.mesh_updates,
        blocks = [(b.key.level, b.key.i, b.key.j) for b in flow.solver.mesh.blocks],
        faces = [(f.axis, f.center, f.area) for f in flow.solver.mesh.faces],
        velocities = map(v -> (; interior = copy(v.interior), boundary = copy(v.boundary)), velocities),
        fraction = owned_values(flow.interface.fraction),
        pressure = owned_values(flow.pressure), histories,
        converged = result.pressure.converged,
        iterations = TwoPhaseAMR._projection_iterations(result.pressure),
    )
end

function finite_difference(reference, current)
    length(reference) == length(current) || return Inf
    value = 0.0
    for (a, b) in zip(reference, current)
        isnan(a) == isnan(b) || return Inf
        isnan(a) && continue
        value = max(value, abs(a - b))
    end
    return value
end

function compare_snapshots(reference, current, mode, final_time)
    @assert reference.time == current.time
    @assert reference.steps == current.steps && reference.updates == current.updates
    @assert reference.blocks == current.blocks && reference.faces == current.faces
    errors = Float64[finite_difference(reference.fraction, current.fraction),
        finite_difference(reference.pressure, current.pressure)]
    for (a, b) in zip(reference.velocities, current.velocities)
        push!(errors, finite_difference(a.interior, b.interior))
        push!(errors, finite_difference(a.boundary, b.boundary))
    end
    @printf("STATE_COMPARE mode=%s time=%.3f fraction_Linf=%.3e pressure_Linf=%.3e velocity_Linf=%.3e masks_and_topology_equal=true\n",
        mode, final_time, errors[1], errors[2], maximum(errors[3:end]))
    for name in keys(reference.histories)
        a, b = reference.histories[name], current.histories[name]
        @assert isequal(a, b) "diagnostic history changed: $name"
    end
    @assert all(iszero, errors) "production fields changed"
    @assert reference.iterations == current.iterations && current.converged
    println("STATE_COMPARE diagnostics_equal=true final_pressure_iterations_equal=true")
end

function validate_amr(action, prefix; final_time = 0.08)
    action in (:capture, :compare, :bench) || throw(ArgumentError("capture, compare or bench required"))
    println("ENV source=", pathof(TwoPhaseAMR), " julia=", VERSION,
        " julia_threads=", Threads.nthreads(), " blas_threads=", BLAS.get_num_threads())
    flush(stdout)
    for mode in (:one, :cmdgfm)
        model = mode == :one ? OneVelocityModel() :
            TwoVelocityModel(pressure_projection = CMDGFMProjection())
        run_example(velocity_model = model, final_time = min(final_time, 0.05))
        repetitions = action == :bench ? 3 : 1
        for repetition in 1:repetitions
            flow = run_example(velocity_model = model, final_time = 0.0).flow
            measured = @timed simulate!(flow, final_time)
            snapshot = validation_snapshot(flow, measured.value)
            @printf("ADVANCE mode=%s time=%.3f repeat=%d wall=%.6f bytes=%d gc=%.6f compile=%.6f steps=%d updates=%d faces=%d blocks=%d pressure_converged=%s\n",
                mode, final_time, repetition, measured.time, measured.bytes,
                measured.gctime, measured.compile_time, snapshot.steps, snapshot.updates,
                length(snapshot.faces), length(snapshot.blocks), snapshot.converged)
            filename = string(prefix, "_", mode, "_", final_time, ".bin")
            if action == :capture
                serialize(filename, snapshot)
            elseif action == :compare
                compare_snapshots(deserialize(filename), snapshot, mode, final_time)
            end
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_amr(Symbol(get(ARGS, 1, "bench")), get(ARGS, 2, "/tmp/stokes3_amr_reference");
        final_time = parse(Float64, get(ARGS, 3, "0.08")))
end
