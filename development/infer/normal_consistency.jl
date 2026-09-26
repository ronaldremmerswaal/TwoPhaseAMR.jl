module InferNormalConsistency

using Printf
using SparseArrays
using TwoPhaseAMR

"""Forward-only normal test on a circle; no pressure solve is performed."""
function circle_case(n; density = (1000.0, 1.0),
        variant = :joint_smoothed)
    mesh = BlockTreeMesh(
        block_size = (n, n), periodic = (true, true),
    )
    centre = (0.5, 0.5)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - centre[1], y - centre[2]) - 0.3;
        lipschitz = 1,
    )
    solver = MimeticProjectionSolver(
        interface; density, variant,
    )
    workspace = MimeticProjectionWorkspace(solver)
    zero_velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    assembled = TwoPhaseAMR._assemble_mimetic_system!(
        solver, workspace, 1.0, zero_velocity,
    )
    active = assembled.active_faces
    # This smooth field is exactly tangential to the analytical circle.
    tangential = map(active) do face_index
        face = mesh.faces[face_index]
        x, y = face.center
        dx, dy = x - centre[1], y - centre[2]
        radius = hypot(dx, dy)
        vector = (-dy / radius, dx / radius)
        vector[face.axis]
    end
    squared = maximum_error = 0.0
    errors = (assembled.candidate_constraint * tangential) ./
        assembled.candidate_scales
    for error in errors
        squared += abs2(error)
        maximum_error = max(maximum_error, abs(error))
    end
    return (;
        n, density, variant, rows = assembled.candidate_count,
        retained = size(assembled.constraint, 1),
        l2 = sqrt(squared / max(assembled.candidate_count, 1)),
        linf = maximum_error,
    )
end

function study(; resolutions = (8, 16, 32, 64),
        densities = ((1000.0, 1.0), (1.0, 1000.0)),
        variant = :joint_smoothed)
    results = [circle_case(n; density, variant) for density in densities
        for n in resolutions]
    for result in results
        @printf(
            "rho=(%g,%g) n=%d variant=%s rows=%d retained=%d L2=%.6e Linf=%.6e\n",
            result.density..., result.n, string(result.variant),
            result.rows, result.retained, result.l2, result.linf,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module InferNormalConsistency
