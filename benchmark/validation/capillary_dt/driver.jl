# Issue #178: the two-velocity capillary wave error grows under dt refinement.
#
# Run from a fresh process (one thread):
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 julia --heap-size-hint=3G -t 1 \
#       --project=benchmark/validation/capillary_dt benchmark/validation/capillary_dt/driver.jl
# Writes results.txt next to this file. src/ is not modified. The variants are
# in-script hooks (variants.jl) or in-session method copies
# (inference_override.jl). The copies run in :baseline mode unless a section
# switches them.
using LinearAlgebra
BLAS.set_num_threads(1)
@assert startswith(Base.active_project(), normpath(joinpath(@__DIR__, "..", "..", "..")))
include(joinpath(@__DIR__, "budget.jl"))
include(joinpath(@__DIR__, "variants.jl"))
include(joinpath(@__DIR__, "inference_override.jl"))
include(joinpath(@__DIR__, "faces.jl"))

const OUT = open(joinpath(@__DIR__, "results.txt"), "w")
emit(args...) = (println(args...); println(OUT, args...); flush(OUT))
fmt(x) = @sprintf("%.5f", x)
fmts(x) = @sprintf("%+.5f", x)
fmte(x) = @sprintf("%.2e", x)
trend(e) = all(diff(e) .<= 0) ? "non-increasing" : "INCREASING"

const FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)

function with_mode(f, mode)
    TP.STUDY178_MODE[] = mode
    try
        return f()
    finally
        TP.STUDY178_MODE[] = :baseline
    end
end

"""L2-in-time difference to the finest run of the same series (linear interpolation)."""
function self_errors(times, amplitudes)
    tf, af = times[end], amplitudes[end]
    interp(t) = begin
        k = clamp(searchsortedlast(tf, t), 1, length(tf) - 1)
        w = (t - tf[k]) / (tf[k + 1] - tf[k])
        (1 - w) * af[k] + w * af[k + 1]
    end
    return [
        _trapezoidal_l2(times[i], amplitudes[i] .- interp.(times[i]))
            for i in 1:(length(times) - 1)
    ]
end

function report(label, n, dr, factors, errors, signed_end, times, amplitudes)
    emit(rpad(label, 34), " n=$n DR=$(Int(dr)) dt factors ", join(factors, "/"))
    emit("    test error           ", join(fmt.(errors), " / "), "   ", trend(errors))
    emit("    signed error at T/4  ", join(fmts.(signed_end), " / "),
        "   successive differences ", join(fmts.(diff(signed_end)), " / "))
    s = self_errors(times, amplitudes)
    emit("    |A_dt - A_finest|    ", join(fmt.(s), " / "), "   ", trend(s))
end

function series(label, n, dr, factors; hooks = (;), mode = :baseline)
    results = with_mode(mode) do
        [run_case(n, dr, f; hooks) for f in factors]
    end
    report(
        label, n, dr, factors, [r.error for r in results], [r.errors[end] for r in results],
        [r.times for r in results], [r.amplitudes for r in results],
    )
    return results
end

function one_velocity_series(n, dr, factors)
    rs = [
        wave_case(:uniform, n, dr, :capillary; formulation = CGFM(), timestep_factor = f)
            for f in factors
    ]
    report(
        "one-velocity (test wave_case)", n, dr, factors, [r.error for r in rs],
        [r.amplitudes[end] - r.references[end] for r in rs],
        [r.times for r in rs], [r.amplitudes for r in rs],
    )
    return rs
end

function budget_table(results, factors)
    emit("    Per-step change made by each stage: mean over all steps of the max over faces.")
    emit("    d[u]x, d[u]y = change of u_g - u_l on x-/y-faces carrying both phases; dU = change of the")
    emit("    mass-weighted mixture velocity. The /dt rows are constant over the factors iff the change is O(dt) per step.")
    emit("    ", rpad("stage", 13), rpad("quantity", 10), join([lpad("f=$f", 11) for f in factors]))
    for s in STAGES
        all(r -> iszero(r.sums[s][1]) && iszero(r.sums[s][3]), results) && continue
        means = [r.sums[s] ./ r.nsteps for r in results]
        for (q, label, scaled) in ((1, "d[u]x", false), (1, "d[u]x/dt", true),
                (2, "d[u]y", false), (3, "dU", false), (3, "dU/dt", true))
            values = [scaled ? m[q] / r.dt : m[q] for (m, r) in zip(means, results)]
            emit("    ", rpad(q == 1 && !scaled ? string(s) : "", 13), rpad(label, 10),
                join(lpad.(fmte.(values), 11)))
        end
    end
    emit("    ", rpad("steps", 23), join(lpad.(string.([r.nsteps for r in results]), 11)))
    emit("    ", rpad("end max|[u]| x", 23), join(lpad.(fmt.([r.final_jump.jx for r in results]), 11)))
    emit("    ", rpad("end max|[u]| y", 23), join(lpad.(fmt.([r.final_jump.jy for r in results]), 11)))
end

emit("# Issue #178: capillary standing wave, uniform n x 3n, CGFM, La=3000, quarter period")
emit("# dt factor = multiple of the initial stable step (factor 1 = automatic step, as in the test).")
emit("# test error = L2 in time of the normalized amplitude error (test/integration/two_phase_waves.jl).")
emit("# |A_dt - A_finest| = same L2 norm of the difference to the finest dt of the same row")
emit("# (temporal self-convergence at fixed mesh).")

emit("\n## A. Two-velocity, DR 1, baseline (src unchanged)")
A = series("two-velocity baseline", 16, 1.0, FACTORS)
budget_table(A, FACTORS)

emit("\n## B. One-velocity, DR 1")
one_velocity_series(16, 1.0, FACTORS)

emit("\n## C. Two-velocity, DR 1, same-location inference (V3 of #171), in-session copy")
C = series("two-velocity V3 inference", 16, 1.0, FACTORS; mode = :same_location)
budget_table(C, FACTORS)

emit("\n## C2. Faces with the largest end-time |u_g - u_l| (x: tangential, y: normal for this wave)")
print_jump_faces(emit, "two-velocity baseline, factor 0.25", A[3].flow)
print_jump_faces(emit, "two-velocity V3 inference, factor 1.0", C[1].flow)

emit("\n## D. Two-velocity, DR 1, phase jump removed after every projection (u_l = u_g = aperture mean)")
D = series("equalize all faces", 16, 1.0, FACTORS[1:4]; hooks = (; projection = equalize_hook(:all)))
budget_table(D, FACTORS[1:4])
series("equalize inference-row faces", 16, 1.0, FACTORS[1:4]; hooks = (; projection = equalize_hook(:infer)))
series("equalize faces without a jump row", 16, 1.0, FACTORS[1:4]; hooks = (; projection = equalize_hook(:rowless)))

emit("\n## E. DR 1000")
E = series("two-velocity baseline", 16, 1000.0, FACTORS[1:4])
budget_table(E, FACTORS[1:4])
series("two-velocity V3 inference", 16, 1000.0, FACTORS[1:4]; mode = :same_location)
one_velocity_series(16, 1000.0, FACTORS[1:4])

emit("\n## F. DR 1, n = 32 (the dt -> 0 limit is a spatial error)")
series("two-velocity baseline", 32, 1.0, FACTORS)
series("two-velocity V3 inference", 32, 1.0, FACTORS; mode = :same_location)
one_velocity_series(32, 1.0, FACTORS)
close(OUT)
