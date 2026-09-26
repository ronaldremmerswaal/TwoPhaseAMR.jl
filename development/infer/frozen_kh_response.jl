module FrozenKHResponse

using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))

const METHODS = (:cmdgfm, :mimetic, :joint_face)

"""The complex interface displacement and its first two time derivatives."""
function wave_state(time, dispersion; amplitude = 0.01)
    modes = (dispersion.plus, dispersion.minus)
    factors = exp.(-im .* modes .* time)
    a = amplitude * sum(factors) / 2
    adot = amplitude * sum((-im .* modes) .* factors) / 2
    addot = amplitude * sum((.-modes .^ 2) .* factors) / 2
    return (; a, adot, addot)
end

"""Linear potential-flow KH mode, evaluated independently in each phase.

The sign `s` is +1 below the sheet and -1 above it.  For a complex
interface displacement a(t) exp(ikx), kinematics gives the vertical
velocity W = a' + ik U a.  Incompressibility and exponential decay give
the horizontal perturbation i*s*W and Bernoulli gives
P = -rho*(W' + ik U W)/(s*k).  This is a *frozen manufactured state*:
it does not integrate transport, curvature, or the moving interface.
"""
function phase_mode(x, y, phase, dispersion, state)
    k = dispersion.k
    rho = dispersion.density[phase]
    U = dispersion.tangential_velocity[phase]
    s = phase == 1 ? 1.0 : -1.0 # this study uses liquid below
    spatial = exp(s * k * y + im * k * x)
    W = state.adot + im * k * U * state.a
    Wdot = state.addot + im * k * U * state.adot
    P = -rho * (Wdot + im * k * U * W) / (s * k)
    velocity = (
        U + real(im * s * W * spatial),
        real(W * spatial),
    )
    scaled_gradient = (
        real(im * k * P * spatial) / rho,
        real(s * k * P * spatial) / rho,
    )
    return (; velocity, scaled_gradient)
end

function frozen_problem(; n = 16, density_ratio = 1000.0,
        weber = 5.0, time_fraction = 0.0)
    dispersion = instability_dispersion(;
        density_ratio, weber, bond = 0.0, liquid_above = false,
    )
    time = time_fraction * _default_instability_final_time(dispersion)
    state = wave_state(time, dispersion)
    mesh = _instability_mesh(n)
    interface = InterfaceField(
        mesh,
        (x, y) -> y - real(state.a * cis(dispersion.k * x));
        lipschitz = hypot(1.0, 0.02 * dispersion.k),
    )
    scale = max(1.0, maximum(abs, dispersion.tangential_velocity))
    dt = 0.35 / (n * scale)
    return (; n, mesh, interface, dispersion, state, dt, time)
end

function projection(method, problem; active_mass_fraction = nothing)
    density = problem.dispersion.density
    method === :cmdgfm && return CMDGFMProjectionSolver(
        problem.interface; density,
    )
    method in (:mimetic, :joint_face) || throw(
        ArgumentError("unknown method: $method"),
    )
    return MimeticProjectionSolver(
        problem.interface; density,
        variant = method === :mimetic ? :joint_smoothed : :joint_face,
        active_mass_fraction,
    )
end

"""Compare a single projection against the analytic frozen KH velocity.

The same geometry, analytic predictor, and time step are used for every
method.  `normal_l2` measures phase-normal velocity on mixed y faces,
normalised by the RMS analytical perturbation there. `jump_l2` measures
the pressure-gradient jump on active mixed faces, with the same geometric
face weighting. These are spatial response diagnostics, not wave errors.
"""
function run_case(problem, method; tolerance = 1e-9)
    mesh = problem.mesh
    density = problem.dispersion.density
    dt = problem.dt
    exact(phase, face) = phase_mode(
        face.center..., phase, problem.dispersion, problem.state,
    )
    velocity = ntuple(2) do phase
        FaceVelocity(
            mesh,
            (x, y, time, axis) -> begin
                mode = phase_mode(
                    x, y, phase, problem.dispersion, problem.state,
                )
                mode.velocity[axis] + dt * mode.scaled_gradient[axis]
            end,
        )
    end
    solver = projection(method, problem)
    workspace = solver isa CMDGFMProjectionSolver ?
        CMDGFMProjectionWorkspace(solver) : MimeticProjectionWorkspace(solver)
    stats = project!(
        velocity, CellField(mesh), solver, dt, workspace;
        tolerance, max_iterations = 3000,
    )
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    normal_error = normal_reference = jump_error = jump_reference = 0.0
    normal_weight = jump_weight = 0.0
    normal_linf = jump_linf = 0.0
    for (i, face) in pairs(mesh.faces)
        c = data.liquid_fraction[i]
        0 < c < 1 || continue
        w = c * (1 - c) * face.area
        w > 0 || continue
        modes = (exact(1, face), exact(2, face))
        exact_jump = modes[2].scaled_gradient[face.axis] -
            modes[1].scaled_gradient[face.axis]
        jump_defect = workspace.jump[i] - exact_jump
        jump_error += w * abs2(jump_defect)
        jump_reference += w * abs2(exact_jump)
        jump_weight += w
        jump_linf = max(jump_linf, abs(jump_defect))
        face.axis == 2 || continue
        for phase in 1:2
            defect = velocity[phase].interior[i] -
                modes[phase].velocity[2]
            perturbation = modes[phase].velocity[2]
            normal_error += w * abs2(defect)
            normal_reference += w * abs2(perturbation)
            normal_weight += w
            normal_linf = max(normal_linf, abs(defect))
        end
    end
    return (;
        method, n = problem.n,
        density_ratio = density[1] / density[2],
        weber = problem.dispersion.weber,
        time_fraction = problem.time /
            _default_instability_final_time(problem.dispersion),
        normal_l2 = sqrt(normal_error / normal_weight),
        normal_relative_l2 = sqrt(normal_error / normal_reference),
        normal_linf,
        jump_l2 = sqrt(jump_error / jump_weight),
        jump_relative_l2 = sqrt(jump_error / jump_reference),
        jump_linf,
        projection_identity_error = stats isa MimeticProjectionStats ?
            stats.energy_identity_error : NaN,
        converged = stats.converged,
    )
end

"""Form one realistic KH predictor, stopping immediately before projection.

This follows the production stepping order. It advances the interface once
with the common transport method and applies the same capillary and gravity
forces. Every candidate subsequently sees these *identical* arrays and
geometry. Only the pressure projection is replayed.
"""
function transported_predictor(; n = 16, density_ratio = 1000.0,
        weber = 5.0, bond = 0.0, liquid_above = false,
        snapshot_steps = 0)
    case = two_velocity_instability_setup(;
        cells_per_wavelength = n, density_ratio, weber, bond,
        liquid_above, pressure_projection = CMDGFMProjection(),
    )
    flow = case.flow
    solver = flow.solver
    workspace = flow.workspace
    model = solver.two_phase
    velocity = flow.velocity
    for phase in velocity
        enforce_velocity_boundary!(
            phase, solver.boundary; time = flow.time,
            geometry = solver.geometry,
        )
    end
    for _ in 1:snapshot_steps
        snapshot_dt = stable_timestep(flow; cfl = 0.35)
        stats = step!(
            velocity, flow.pressure, solver, snapshot_dt, workspace;
            time = flow.time, tolerance = 1e-7, max_cycles = 3000,
        )
        stats.converged || error("CMDGFM snapshot step failed")
        flow.time += snapshot_dt
        flow.steps += 1
    end
    for phase in velocity
        enforce_velocity_boundary!(
            phase, solver.boundary; time = flow.time,
            geometry = solver.geometry,
        )
    end
    dt = stable_timestep(flow; cfl = 0.35)
    state = TwoPhaseAMR._two_velocity_flow_workspace!(
        workspace.two_phase, model,
    )
    TwoPhaseAMR._copy_velocity!(state.old_velocity, velocity)
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        state.old_velocity, solver, flow.time,
    )
    TwoPhaseAMR._transport_substep!(
        velocity, solver, workspace, dt, flow.time, model,
    )
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        state.convected, solver, flow.time + dt / 2,
    )
    TwoPhaseAMR._diffusion_substep!(
        velocity, solver, workspace, dt, flow.time, model;
        tolerance = 1e-9, max_cycles = 3000,
    )
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        velocity, solver, flow.time + dt,
    )
    if model.sigma > 0 || any(!iszero, model.gravity)
        TwoPhaseAMR._refresh_pressure_substep!(
            solver, workspace, model,
        )
        TwoPhaseAMR._phase_force_substep!(
            velocity, solver, workspace, dt, model,
        )
    end
    return (;
        n, mesh = case.mesh, interface = case.interface,
        dispersion = case.dispersion, dt,
        snapshot_steps, snapshot_time = flow.time,
        initial_levelset = case.interface_levelset,
        initial_amplitude = case.initial_amplitude,
        predictor = ntuple(phase -> (
            copy(velocity[phase].interior),
            copy(velocity[phase].boundary),
        ), 2),
    )
end

"""Apply only interface advection to the projected, frozen velocity field.

This measures the first-mode interface speed delivered to the *next* KH
step. It is not an additional momentum or pressure solve.
"""
function transported_mode_speed(problem, candidate)
    field = copy(problem.interface)
    initial = _instability_wave_amplitude(
        field; liquid_above = problem.dispersion.liquid_above,
    )
    work = WYAdvectionWorkspace(
        field, candidate.velocity;
        interface_transport = FrozenInterfaceFluxes(),
    )
    advect!(field, candidate.velocity, problem.dt, work)
    final = _instability_wave_amplitude(
        field; liquid_above = problem.dispersion.liquid_above,
    )
    return (final - initial) / problem.dt
end

function _replay_velocity(problem)
    velocity = ntuple(_ -> FaceVelocity(problem.mesh), 2)
    for phase in 1:2
        copyto!(velocity[phase].interior, problem.predictor[phase][1])
        copyto!(velocity[phase].boundary, problem.predictor[phase][2])
    end
    return velocity
end

function replay_projection(problem, method; tolerance = 1e-9,
        active_mass_fraction = nothing)
    velocity = _replay_velocity(problem)
    solver = projection(method, problem; active_mass_fraction)
    workspace = solver isa CMDGFMProjectionSolver ?
        CMDGFMProjectionWorkspace(solver) : MimeticProjectionWorkspace(solver)
    stats = project!(
        velocity, CellField(problem.mesh), solver, problem.dt, workspace;
        tolerance, max_iterations = 3000,
    )
    return (; velocity, workspace, solver, stats)
end

"""Geometrically weighted frozen-response distance from the CMDGFM result."""
function response_distance(problem, reference, candidate)
    data = TwoPhaseAMR._mimetic_face_data(
        candidate.solver, candidate.workspace,
    )
    normal_error = velocity_error = jump_error = 0.0
    normal_reference = velocity_reference = jump_reference = 0.0
    weight_normal = weight_velocity = weight_jump = 0.0
    for (i, face) in pairs(problem.mesh.faces)
        c = data.liquid_fraction[i]
        0 < c < 1 || continue
        w = c * (1 - c) * face.area
        w > 0 || continue
        jump_candidate = candidate.velocity[2].interior[i] -
            candidate.velocity[1].interior[i]
        jump_cmd = reference.velocity[2].interior[i] -
            reference.velocity[1].interior[i]
        jump_error += w * abs2(jump_candidate - jump_cmd)
        jump_reference += w * abs2(jump_cmd)
        weight_jump += w
        for phase in 1:2
            difference = candidate.velocity[phase].interior[i] -
                reference.velocity[phase].interior[i]
            velocity_error += w * abs2(difference)
            velocity_reference += w * abs2(
                reference.velocity[phase].interior[i] -
                problem.dispersion.tangential_velocity[phase] *
                    (face.axis == 1),
            )
            weight_velocity += w
            if face.axis == 2
                normal_error += w * abs2(difference)
                normal_reference += w * abs2(
                    reference.velocity[phase].interior[i],
                )
                weight_normal += w
            end
        end
    end
    return (;
        normal_l2 = sqrt(normal_error / weight_normal),
        normal_relative_l2 = sqrt(normal_error / normal_reference),
        velocity_l2 = sqrt(velocity_error / weight_velocity),
        velocity_relative_l2 = sqrt(
            velocity_error / velocity_reference,
        ),
        jump_l2 = sqrt(jump_error / weight_jump),
        jump_relative_l2 = sqrt(jump_error / jump_reference),
    )
end

"""Necessary condition for reproducing CMDGFM by a homogeneous energy projection.

For an orthogonal projection in the physical phase kinetic metric, the final
velocity is orthogonal to its correction. A nonzero result means no change
of N or E alone can reproduce this CMDGFM output *exactly* on this state.
"""
function cmdgfm_kinetic_orthogonality(problem, reference)
    data = TwoPhaseAMR._mimetic_face_data(
        reference.solver, reference.workspace,
    )
    volumes = TwoPhaseAMR._uncut_face_dual_volumes(
        problem.mesh, Float64,
    )
    work = final_norm = correction_norm = 0.0
    for i in eachindex(problem.mesh.faces), phase in 1:2
        mass = phase == 1 ? data.liquid_mass[i] : data.gas_mass[i]
        weight = mass * volumes[i]
        final = reference.velocity[phase].interior[i]
        correction = problem.predictor[phase][1][i] - final
        work += weight * final * correction
        final_norm += weight * abs2(final)
        correction_norm += weight * abs2(correction)
    end
    return (;
        work,
        cosine = work / sqrt(final_norm * correction_norm),
        energy_defect = 2work,
    )
end

function study_predictor(; densities = (1.0, 1000.0),
        webers = (0.5, 5.0), resolutions = (8, 16), methods = METHODS,
        snapshot_steps = 0, mass_floors = (nothing,))
    println("rho\tWe\tn\tsnapshot_steps\tmass_floor\tmethod\tnormal_L2\tnormal_rel_L2\tvelocity_L2\tvelocity_rel_L2\tjump_L2\tjump_rel_L2\tmode_speed_difference\tmode_displacement_budget\tmode_speed_analytic_error\tcmd_kinetic_cosine\tenergy_identity\tmass_deactivated\tunderresolved\tconverged")
    flush(stdout)
    results = NamedTuple[]
    for rho in densities, weber in webers, n in resolutions
        problem = transported_predictor(;
            n, density_ratio = rho, weber, snapshot_steps,
        )
        reference = replay_projection(problem, :cmdgfm)
        cmd_orthogonality = cmdgfm_kinetic_orthogonality(
            problem, reference,
        )
        reference_speed = transported_mode_speed(problem, reference)
        analytical_speed = wave_state(
            problem.snapshot_time + problem.dt, problem.dispersion;
            amplitude = problem.initial_amplitude,
        ).adot
        for mass_floor in mass_floors, method in methods
            method === :cmdgfm && !isnothing(mass_floor) && continue
            candidate = method === :cmdgfm ? reference :
                replay_projection(
                    problem, method;
                    active_mass_fraction = mass_floor,
                )
            distance = response_distance(problem, reference, candidate)
            mode_speed = method === :cmdgfm ? reference_speed :
                transported_mode_speed(problem, candidate)
            result = (;
                density_ratio = rho, weber, n, snapshot_steps,
                mass_floor, method, distance...,
                mode_speed_difference = abs(mode_speed - reference_speed),
                # If the instantaneous mode-speed difference persisted for
                # the linear-theory horizon, this would be its displacement
                # relative to the initial wave amplitude. It is a screening
                # scale, not a bound on the evolving nonlinear calculation.
                mode_displacement_budget =
                    abs(mode_speed - reference_speed) *
                    _default_instability_final_time(problem.dispersion) /
                    problem.initial_amplitude,
                mode_speed_analytic_error =
                    abs(mode_speed - analytical_speed),
                cmd_kinetic_cosine = cmd_orthogonality.cosine,
                projection_identity_error =
                    candidate.stats isa MimeticProjectionStats ?
                    candidate.stats.energy_identity_error : NaN,
                mass_deactivated =
                    candidate.stats isa MimeticProjectionStats ?
                    candidate.stats.mass_deactivated_faces : 0,
                underresolved =
                    candidate.stats isa MimeticProjectionStats ?
                    candidate.stats.underresolved_constraint_rows : 0,
                converged = candidate.stats.converged,
            )
            push!(results, result)
            @printf("%g\t%g\t%d\t%d\t%s\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.3g\t%d\t%d\t%s\n",
                rho, weber, n, snapshot_steps,
                isnothing(mass_floor) ? "default" : string(mass_floor),
                string(method),
                result.normal_l2, result.normal_relative_l2,
                result.velocity_l2, result.velocity_relative_l2,
                result.jump_l2, result.jump_relative_l2,
                result.mode_speed_difference,
                result.mode_displacement_budget,
                result.mode_speed_analytic_error,
                result.cmd_kinetic_cosine,
                result.projection_identity_error,
                result.mass_deactivated, result.underresolved,
                string(result.converged),
            )
            flush(stdout)
        end
    end
    return results
end

function study(; densities = (1.0, 1000.0), webers = (0.5, 5.0),
        resolutions = (8, 16), time_fractions = (0.0, 0.25),
        methods = METHODS)
    println("rho\tWe\tn\tt/T\tmethod\tnormal_L2\tnormal_rel_L2\tnormal_Linf\tjump_L2\tjump_rel_L2\tjump_Linf\tenergy_identity\tconverged")
    flush(stdout)
    results = NamedTuple[]
    for rho in densities, weber in webers, n in resolutions,
            fraction in time_fractions
        problem = frozen_problem(;
            n, density_ratio = rho, weber, time_fraction = fraction,
        )
        for method in methods
            result = run_case(problem, method)
            push!(results, result)
            @printf("%g\t%g\t%d\t%g\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.3g\t%s\n",
                rho, weber, n, fraction, string(method),
                result.normal_l2, result.normal_relative_l2,
                result.normal_linf, result.jump_l2,
                result.jump_relative_l2, result.jump_linf,
                result.projection_identity_error, string(result.converged),
            )
            flush(stdout)
        end
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    study_predictor()
end

end # module
