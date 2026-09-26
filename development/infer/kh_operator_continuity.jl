module KHOperatorContinuity

using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "frozen_kh_response.jl"))
using .FrozenKHResponse: frozen_problem, phase_mode

"""Freeze a KH pressure predictor and translate only the interface geometry.

The output separates smooth coefficient changes from changes of the active
jump set and constraint rank.  The interface displacement is measured in
units of the cell width; no transport or surface force is applied here.
"""
function shifted_projection(problem, displacement; selection = :joint_smoothed,
        active_mass_fraction = 1.0e-6)
    mesh = problem.mesh
    state = problem.state
    dispersion = problem.dispersion
    k = dispersion.k
    phi(x, y) = y - real(state.a * cis(k * x)) - displacement
    interface = InterfaceField(mesh, phi;
        lipschitz = hypot(1.0, 0.02k))
    velocity = ntuple(2) do phase
        FaceVelocity(mesh, (x, y, time, axis) -> begin
            mode = phase_mode(x, y, phase, dispersion, state)
            mode.velocity[axis] +
                problem.dt * mode.scaled_gradient[axis]
        end)
    end
    solver = if selection === :cmdgfm
        CMDGFMProjectionSolver(interface; density = dispersion.density)
    else
        MimeticProjectionSolver(interface;
            density = dispersion.density, variant = selection,
            active_mass_fraction,
        )
    end
    workspace = selection === :cmdgfm ?
        CMDGFMProjectionWorkspace(solver) :
        MimeticProjectionWorkspace(solver)
    stats = project!(velocity, CellField(mesh), solver, problem.dt,
        workspace; tolerance = 1.0e-9, max_iterations = 3000)
    stats.converged || error("projection failed at shift $displacement")
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    return (; velocity, stats, fraction = copy(data.liquid_fraction),
        jump = copy(workspace.jump))
end

"""Weighted change on faces mixed in both geometries; all phases are counted."""
function response_change(reference, candidate)
    squared = weight = maximum_change = 0.0
    for face in eachindex(reference.fraction)
        c0, c1 = reference.fraction[face], candidate.fraction[face]
        min(c0, 1 - c0, c1, 1 - c1) > 0.01 || continue
        w = min(c0 * (1 - c0), c1 * (1 - c1))
        for phase in 1:2
            difference = candidate.velocity[phase].interior[face] -
                reference.velocity[phase].interior[face]
            squared += w * abs2(difference)
            weight += w
            maximum_change = max(maximum_change, abs(difference))
        end
    end
    return (; rms = sqrt(squared / max(weight, eps())),
        maximum = maximum_change)
end

function study(; n = 16, density_ratio = 1000.0, weber = 5.0,
        selection = :joint_smoothed, active_mass_fraction = 1.0e-6,
        shifts = (
            -0.04, -0.02, -0.01, -0.001, -0.0001,
            0.0, 0.0001, 0.001, 0.01, 0.02, 0.04,
        ))
    problem = frozen_problem(; n, density_ratio, weber)
    h = 1 / n
    reference = shifted_projection(problem, 0.0;
        selection, active_mass_fraction)
    println("KH frozen geometry rho=$density_ratio We=$weber n=$n ",
        "selection=$selection floor=$active_mass_fraction; ",
        "shift is in cell widths")
    println("shift/h  active  candidates  rows  deactivated  response_rms  response_max ",
        "rms_per_abs_shift  commutator  pressure_error")
    for relative_shift in shifts
        candidate = iszero(relative_shift) ? reference :
            shifted_projection(problem, relative_shift * h;
                selection, active_mass_fraction)
        change = response_change(reference, candidate)
        derivative = iszero(relative_shift) ? 0.0 :
            change.rms / abs(relative_shift * h)
        stats = candidate.stats
        candidates = hasproperty(stats, :candidate_constraint_rows) ?
            stats.candidate_constraint_rows : stats.jump_unknowns
        rows = hasproperty(stats, :independent_constraint_rows) ?
            stats.independent_constraint_rows : stats.jump_unknowns
        deactivated = hasproperty(stats, :mass_deactivated_faces) ?
            stats.mass_deactivated_faces : 0
        commutator = hasproperty(stats, :commuting_residual) ?
            stats.commuting_residual : NaN
        pressure_error =
            hasproperty(stats, :pressure_commuting_residual) ?
                stats.pressure_commuting_residual : NaN
        @printf("%+.6g %d %d %d %d %.6e %.6e %.6e %.6e %.6e\n",
            relative_shift, stats.jump_unknowns,
            candidates, rows, deactivated,
            change.rms, change.maximum, derivative,
            commutator, pressure_error)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    selection = isempty(ARGS) ? :joint_smoothed : Symbol(first(ARGS))
    study(; selection)
end

end # module
