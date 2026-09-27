# Compare force evaluation on one *identical* Stokes3 interface/mesh state.
# Run from the repository root:
#   julia --project=. development/stokes3_performance/force_analysis.jl
# The method comparison uses the same solver, pressure metric, interface and
# force scratch. Only the destination velocity count differs.
using TwoPhaseAMR
using Printf
using Statistics: median

include(joinpath(@__DIR__, "..", "..", "examples", "stokes3_wave.jl"))

function _force_sample(label, f; repetitions = 8)
    f()
    GC.gc()
    samples = [@timed f() for _ in 1:repetitions]
    @printf("FORCE %-22s median_ms=%9.3f median_bytes=%10.0f max_bytes=%d\n",
        label, 1e3 * median([sample.time for sample in samples]),
        median([sample.bytes for sample in samples]),
        maximum(sample.bytes for sample in samples))
    return nothing
end

function _uncached_face_opening_sum(field)
    total = zero(eltype(field))
    for face in field.mesh.faces
        total += (
            TwoPhaseAMR._face_aperture(field, face, face.minus) +
            TwoPhaseAMR._face_aperture(field, face, face.plus)
        ) / 2
    end
    return total
end

function force_state(label, flow)
    solver = flow.solver
    workspace = flow.workspace
    model = solver.two_phase
    field = model.interface
    operator = solver.pressure_solver.operator
    acceleration = workspace.two_phase.acceleration
    elliptic = workspace.projection.elliptic
    one = FaceVelocity(solver.mesh)
    two = (FaceVelocity(solver.mesh), FaceVelocity(solver.mesh))
    mixed = sum(eachindex(solver.mesh.blocks)) do block
        count(c -> 0 < c < 1, interior(field.conditional_fraction, block))
    end
    @printf("STATE label=%s blocks=%d faces=%d mixed_cells=%d steps=%d updates=%d\n",
        label, length(solver.mesh.blocks), length(solver.mesh.faces), mixed,
        flow.steps, flow.mesh_updates)
    _force_sample("old_uncached_openings", () ->
        _uncached_face_opening_sum(field))
    _force_sample("surface_tension", () -> surface_tension!(
        acceleration, field, operator, elliptic;
        sigma = model.sigma, formulation = model.formulation,
    ))
    _force_sample("gravity", () -> gravity!(
        acceleration, field, operator, elliptic;
        density = model.density, gravity = model.gravity,
    ))
    _force_sample("one_complete", () -> TwoPhaseAMR._phase_force_substep!(
        one, solver, workspace, 0.001, model,
    ))
    _force_sample("two_complete", () -> TwoPhaseAMR._phase_force_substep!(
        two, solver, workspace, 0.001, model,
    ))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    initial = run_example(final_time = 0.0).flow
    force_state("initial", initial)
    evolved = run_example(final_time = 0.05).flow
    evolved.mesh_updates > 0 || error("expected AMR updates")
    force_state("post_amr", evolved)
end
