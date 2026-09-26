module InferNormalConsistency

using Printf
using SparseArrays
using TwoPhaseAMR

"""Forward-only normal test on a circle; no pressure solve is performed."""
function circle_case(n; density = (1000.0, 1.0))
    mesh = BlockTreeMesh(
        block_size = (n, n), periodic = (true, true),
    )
    centre = (0.5, 0.5)
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - centre[1], y - centre[2]) - 0.3;
        lipschitz = 1,
    )
    solver = MimeticProjectionSolver(
        interface; density, jump_interpolant = :infer,
    )
    workspace = MimeticProjectionWorkspace(solver)
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    active, _, _ = TwoPhaseAMR._local_active_faces(solver, data)
    dual = TwoPhaseAMR._uncut_face_dual_volumes(mesh, Float64)[active]
    kinetic = data.relative_mass[active] .* dual
    space = TwoPhaseAMR._local_infer_jump_space(
        solver, data, active, kinetic, dual,
    )
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
    for (row, column) in pairs(space.normal_columns)
        indices, values = findnz(space.extension[:, column])
        error = sum(
            kinetic[index] * value * tangential[index]
            for (index, value) in zip(indices, values)
        ) / space.normal_scales[row]
        squared += abs2(error)
        maximum_error = max(maximum_error, abs(error))
    end
    return (;
        n, density, rows = length(space.normal_columns),
        paired = space.paired,
        fallback = space.inferred_unpaired + space.imposed_unpaired,
        l2 = sqrt(squared / max(length(space.normal_columns), 1)),
        linf = maximum_error,
    )
end

function study(; resolutions = (8, 16, 32, 64),
        densities = ((1000.0, 1.0), (1.0, 1000.0)))
    results = [circle_case(n; density) for density in densities
        for n in resolutions]
    for result in results
        @printf(
            "rho=(%g,%g) n=%d rows=%d paired=%d fallback=%d L2=%.6e Linf=%.6e\n",
            result.density..., result.n, result.rows,
            result.paired, result.fallback, result.l2, result.linf,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module InferNormalConsistency
