# Frozen physical pressure RHS: test initial guesses without changing any
# production solver, tolerance, pressure metric or jump ILU.
include(joinpath(@__DIR__, "pressure_probe.jl"))

function pressure_predictor!(flow, dt)
    solver, workspace = flow.solver, flow.workspace
    model, velocity = solver.two_phase, flow.velocity
    state = M._two_velocity_flow_workspace!(workspace.two_phase, model)
    time = flow.time
    M._copy_velocity!(state.old_velocity, velocity)
    M._enforce_phase_velocity_boundary!(state.old_velocity, solver, time)
    M._transport_substep!(velocity, solver, workspace, dt, time, model)
    M._enforce_phase_velocity_boundary!(state.convected, solver, time + dt / 2)
    M._diffusion_substep!(velocity, solver, workspace, dt, time, model)
    M._enforce_phase_velocity_boundary!(velocity, solver, time + dt)
    M._refresh_pressure_substep!(solver, workspace, model)
    M._phase_force_substep!(velocity, solver, workspace, dt, model)
    return state
end

function frozen_initial_guesses(flow, assembled, work, preconditioner)
    np = length(assembled.pressure_indices)
    n = size(assembled.matrix, 2)
    previous = zeros(n)
    references = assembled.pressure_operator.references
    for index in 1:np
        cell = assembled.pressure_indices[index]
        anchor = assembled.pressure_anchors[index]
        previous[index] = M._get(flow.pressure, references[cell]) -
            M._get(flow.pressure, references[anchor])
    end
    for (index, face) in pairs(assembled.mixed_faces)
        previous[np + index] = work.jump[face]
    end
    fitted = copy(previous)
    fitted_zero = zeros(n)
    if n > np
        jump_rhs = copy(view(assembled.right_hand_side, (np + 1):n))
        mul!(jump_rhs, assembled.jump_pressure, view(fitted, 1:np), -1.0, 1.0)
        ldiv!(view(fitted, (np + 1):n), preconditioner.jump_preconditioner, jump_rhs)
        ldiv!(view(fitted_zero, (np + 1):n), preconditioner.jump_preconditioner,
            view(assembled.right_hand_side, (np + 1):n))
    end
    return ((:zero, zeros(n)), (:previous, previous),
        (:pressure_and_jump_fit, fitted), (:zero_pressure_and_jump_fit, fitted_zero))
end

function frozen_pressure_solve!(solution, residual, krylov, initial, assembled, preconditioner, dt)
    copyto!(solution, initial)
    mul!(residual, assembled.matrix, solution)
    residual .= assembled.right_hand_side .- residual
    iterations = 0
    tolerance = sqrt(eps(Float64))
    for _ in 1:8
        defects = M._cmdgfm_residual_defects(residual, assembled, dt)
        maximum(defects) <= tolerance && break
        requested = min(tolerance, tolerance / max(maximum(defects), eps(Float64)))
        M.Krylov.bicgstab!(krylov, assembled.matrix, residual;
            N = preconditioner, ldiv = true, atol = 0.0, rtol = requested, itmax = 1000)
        solution .+= M.Krylov.solution(krylov)
        iterations += M.Krylov.iteration_count(krylov)
        mul!(residual, assembled.matrix, solution)
        residual .= assembled.right_hand_side .- residual
    end
    return iterations, M._cmdgfm_residual_defects(residual, assembled, dt)
end

function probe_initial_guesses(; times = (0.03,), repetitions = 3)
    for time in times
        flow = run_example(velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection()), final_time = time).flow
        dt = min(M.stable_timestep(flow.velocity, flow.solver, flow.workspace; cfl = 0.45), 0.002)
        state = pressure_predictor!(flow, dt)
        solver, work = state.projection_solver, state.projection_workspace
        assembled = M._assemble_cmdgfm_system!(solver, work, dt, flow.velocity;
            shared_operator = flow.solver.pressure_solver.operator,
            shared_multigrid = flow.solver.pressure_solver.preconditioner)
        preconditioner = M._cmdgfm_preconditioner(solver, assembled, work)
        solution, residual = zeros(size(assembled.matrix, 1)), similar(assembled.right_hand_side)
        krylov = M.Krylov.BicgstabWorkspace(assembled.matrix, assembled.right_hand_side)
        reference = similar(solution)
        for (label, initial) in frozen_initial_guesses(flow, assembled, work, preconditioner)
            mul!(residual, assembled.matrix, initial)
            residual .= assembled.right_hand_side .- residual
            before = M._cmdgfm_residual_defects(residual, assembled, dt)
            before_l2 = norm(residual)
            frozen_pressure_solve!(solution, residual, krylov, initial, assembled, preconditioner, dt)
            for repetition in 1:repetitions
                result = @timed frozen_pressure_solve!(solution, residual, krylov,
                    initial, assembled, preconditioner, dt)
                iterations, defects = result.value
                maximum(defects) <= sqrt(eps()) || error("unconverged frozen solve")
                label == :zero && copyto!(reference, solution)
                @printf("INITIAL_GUESS time=%.4f faces=%d kind=%s rep=%d initial_pressure=%.8g initial_jump=%.8g initial_l2=%.8g seconds=%.6f bytes=%d iterations=%d final_pressure=%.8g final_jump=%.8g solution_difference=%.8g\n",
                    time, length(flow.solver.mesh.faces), label, repetition,
                    before.pressure, before.jump, before_l2, result.time, result.bytes,
                    iterations, defects.pressure, defects.jump, norm(solution - reference, Inf))
                flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    probe_initial_guesses(times = Tuple(parse.(Float64, isempty(ARGS) ? ["0.03"] : ARGS)))
end
