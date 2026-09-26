module PoissonJumpAudit

using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "examples",
    "manufactured_poisson.jl"))

"""Locate the face setting the scaled-gradient-jump maximum error.

`pressure_only` is the diagonal-K response to the solved cell pressure on
the physical face. Its sign relative to the exact jump tests whether *any*
positive diagonal rescaling of K could repair that face at fixed pressure.
"""
function audit(n; method = :mimetic, density = (1000.0, 1.0),
        active_mass_fraction = 1.0e-6)
    run = run_example(;
        resolution = n, method,
        density, active_mass_fraction, tolerance = 1e-8,
    )
    data = TwoPhaseAMR._mimetic_face_data(run.solver, run.workspace)
    active, deactivated, _ = TwoPhaseAMR._local_active_faces(
        run.solver, data,
    )
    active_set = Set(active)
    pressure_count = length(run.workspace.pressure_indices)
    incidence = TwoPhaseAMR._mimetic_incidence(run.mesh, Float64)
    pressure_difference = incidence[:, run.workspace.pressure_indices] *
        view(run.workspace.solution, 1:pressure_count)
    dual_volume = TwoPhaseAMR._uncut_face_dual_volumes(
        run.mesh, Float64,
    )
    worst = nothing
    aperture_mixed_linf = 0.0
    aperture_pure_linf = 0.0
    for face_index in data.mixed
        face = run.mesh.faces[face_index]
        exact = manufactured_pressure_gradient(
            face.center..., 2,
        )[face.axis] / density[2] - manufactured_pressure_gradient(
            face.center..., 1,
        )[face.axis] / density[1]
        numerical = run.workspace.jump[face_index]
        defect = abs(numerical - exact)
        if min(data.liquid_aperture[face_index],
                data.gas_aperture[face_index]) > 0
            aperture_mixed_linf = max(aperture_mixed_linf, defect)
        else
            aperture_pure_linf = max(aperture_pure_linf, defect)
        end
        if isnothing(worst) || defect > worst.defect
            pressure_only = data.coupling[face_index] *
                pressure_difference[face_index] /
                (data.relative_mass[face_index] * dual_volume[face_index])
            worst = (;
                face_index, defect, exact, numerical, pressure_only,
                axis = face.axis, center = face.center,
                apertures = (
                    data.liquid_aperture[face_index],
                    data.gas_aperture[face_index],
                ),
                active = face_index in active_set,
            )
        end
    end
    return (;
        n, method, density, mixed_faces = length(data.mixed),
        active_faces = length(active), deactivated,
        aperture_mixed_linf, aperture_pure_linf,
        worst...,
    )
end

function study(; resolutions = (8, 16, 32, 64),
        methods = (:mimetic, :joint_face),
        density = (1000.0, 1.0))
    println("n\tmethod\tmixed\tactive\tdeactivated\tLinf_both_apertures\tLinf_pure_aperture\tworst_face\taxis\tcenter\tapertures\texact_xi\tnumerical_xi\tpressure_only_xi\tpositive_diagonal_K_possible")
    for n in resolutions, method in methods
        result = audit(n; method, density)
        positive_diagonal = sign(result.pressure_only) ==
            sign(result.exact)
        @printf("%d\t%s\t%d\t%d\t%d\t%.6g\t%.6g\t%d\t%d\t%s\t%s\t%.6g\t%.6g\t%.6g\t%s\n",
            n, string(method), result.mixed_faces,
            result.active_faces, result.deactivated,
            result.aperture_mixed_linf, result.aperture_pure_linf,
            result.face_index, result.axis, string(result.center),
            string(result.apertures), result.exact, result.numerical,
            result.pressure_only, string(positive_diagonal),
        )
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module
