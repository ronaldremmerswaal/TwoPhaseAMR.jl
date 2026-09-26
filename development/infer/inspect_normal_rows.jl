module InspectInferNormalRows

using Printf
using SparseArrays
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))

"""Print the largest *physical* normal-jump defects before projection."""
function inspect(; density_ratio = 1000.0, weber = 0.5,
        cells_per_wavelength = 8, count = 12,
        transport_factor = 0.0, variant = :joint_smoothed,
    )
    case = two_velocity_instability_setup(
        ; density_ratio, weber, cells_per_wavelength,
        bond = 0.0, liquid_above = false,
        pressure_projection = MimeticProjection(; variant),
    )
    if transport_factor > 0
        flow = case.flow
        model = flow.solver.two_phase
        state = TwoPhaseAMR._two_velocity_flow_workspace!(
            flow.workspace.two_phase, model,
        )
        TwoPhaseAMR._enforce_phase_velocity_boundary!(
            flow.velocity, flow.solver, flow.time,
        )
        dt = transport_factor * stable_timestep(flow; cfl = 0.35)
        TwoPhaseAMR._copy_velocity!(state.old_velocity, flow.velocity)
        TwoPhaseAMR._enforce_phase_velocity_boundary!(
            state.old_velocity, flow.solver, flow.time,
        )
        TwoPhaseAMR._transport_substep!(
            flow.velocity, flow.solver, flow.workspace,
            dt, flow.time, model,
        )
    end
    solver = MimeticProjectionSolver(
        case.interface; density = case.dispersion.density,
        variant,
    )
    workspace = MimeticProjectionWorkspace(solver)
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    assembled = TwoPhaseAMR._assemble_mimetic_system!(
        solver, workspace, 1.0, case.flow.velocity,
    )
    active = assembled.active_faces
    jump = case.flow.velocity[2].interior[active] .-
        case.flow.velocity[1].interior[active]
    records = NamedTuple[]
    for row in axes(assembled.candidate_constraint, 1)
        _, indices, values = findnz(assembled.candidate_constraint[row:row, :])
        physical = values ./ assembled.candidate_scales[row]
        defect = sum(physical[k] * jump[indices[k]]
            for k in eachindex(indices))
        push!(records, (; defect, indices, physical))
    end
    sort!(records; by = record -> -abs(record.defect))
    @printf(
        "rho=%g We=%g n=%d transport_factor=%g active=%d candidate_rows=%d retained_rows=%d\n",
        density_ratio, weber, cells_per_wavelength, transport_factor,
        length(active), assembled.candidate_count,
        size(assembled.constraint, 1),
    )
    for record in Iterators.take(records, count)
        @printf("defect=%+.6e", record.defect)
        for (index, coefficient) in zip(record.indices, record.physical)
            face_index = active[index]
            face = case.mesh.faces[face_index]
            @printf(
                " | face=%d axis=%d xy=(%.4f,%.4f) n=%+.4f jump=%+.4f c=%.4f",
                face_index, face.axis, face.center...,
                coefficient, jump[index], data.liquid_fraction[face_index],
            )
        end
        println()
    end
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    inspect()
end

end # module InspectInferNormalRows
