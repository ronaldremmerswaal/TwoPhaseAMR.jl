# Code check 01: the defects compared with the tolerance (eq:projection:defects) and the projection after a regrid (eq:projection:regrid).
#
# Labels: eq:projection:defects, eq:projection:regrid
# Source: docs/techref/chapters/10_projection.tex:861-990, 995-1031; src/two_phase/projection/*.jl, src/two_phase/two_velocity/flow_workspace.jl
# Checks (4 fixtures; random predictor with phase jump; tolerance 1e-10):
#   D1  CMDGFM: independently evaluated defects (V = dt max|D^mix| over open cells, η = max_{f in F} |N_f(u^{n+1})|, residual defects dt² max_{c in C_r}|r_c|/|c^o| and
#       dt max_f |r_f| of the system residual r = b - M x) equal the code's; `converged` = all four <= ε (also for a run stopped early by max_iterations);
#   D2  mimetic: V, η = max{ max_r |(N_cand [u^{n+1}])_r|/scale_r over ALL candidate rows, max_inactive |[u^{n+1}]| }, tangential defect dt² max_f |r_f|/(m_f V^o_f^{3/2}),
#       multiplier-row defect dt max_r |r_r|/scale_r equal the code's / hold; `converged` = (V, η, τ <= ε);
#   D3  units: the same flow in other velocity units (u -> u/s, dt -> s dt): V is dimensionless, η carries the unit; at the same ε = 1e-3 the relative
#       normal-jump accuracy reached differs by a factor > 10 between the unit factors 1e-6, 1, 1e6 (issue #94: the criterion is not unit-free);
#   R1  regrid scaling: projecting the same velocities with dt and 3 dt gives the same end-of-step velocities (differences ~ solver tolerance) for CMDGFM and mimetic,
#       with and without solid; the assembled matrix does not depend on dt and the right-hand side scales as 1/dt, V scales with dt.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-defects-regrid/scripts/code_01_defects_regrid.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit
include(joinpath(@__DIR__, "..", "..", "eq-projection-cmdgfm-system", "scripts", "code_01_stencils_lib.jl"))   # classify

function normal_jump(s, f, u)
    jump(g) = u[2].interior[g] - u[1].interior[g]
    val = jump(f)
    for (g, w) in zip(s.trans, s.w)
        val += s.beta * s.R * w * jump(g)
    end
    s.ig > 0 && (val += (s.beta - 1) * (u[2].interior[s.ig] - u[1].interior[s.il]))
    return val
end

@testset "defects and regrid" begin
    tol = 1.0e-10; dt = PA.DT
    for (name, field) in PA.fixtures()
        mesh = field.mesh
        vol = PA.open_volumes(field); open_c = findall(>(0), vol)
        d = PA.face_data(field); ind = classify(field, d)
        # ---------------- D1 CMDGFM ----------------
        solver = CMDGFMProjectionSolver(field; density = PA.DENSITY); work = CMDGFMProjectionWorkspace(solver)
        v0 = PA.two_predictor(field, MersenneTwister(12))
        for (label, maxit) in (("converged", 5000), ("stopped early", 3))
            v = PA.copy_velocity(v0)
            st = project!(v, CellField(mesh), solver, dt, work; tolerance = tol, max_iterations = maxit)
            asm = (; matrix = work.matrix, right_hand_side = work.right_hand_side)
            x = work.solution; r = asm.right_hand_side .- asm.matrix * x
            idx = work.pressure_indices; F = work.mixed_faces; np = length(idx)
            res_p = dt^2 * maximum(abs(r[i]) / vol[idx[i]] for i in 1:np if vol[idx[i]] > 0)
            res_j = isempty(F) ? 0.0 : dt * maximum(abs.(r[(np + 1):end]))
            flux = PA.mixture_flux(field, v)
            V = dt * maximum(abs.(flux[open_c]) ./ vol[open_c])
            η = maximum(abs(normal_jump(ind[f], f, v)) for f in F)
            all4 = max(res_p, res_j, V, η) <= tol
            println(
                rpad(name, 17), "D1 ", rpad(label, 14), " V ", round(V, sigdigits = 3), " (code ", round(st.mixture_volume_defect, sigdigits = 3), ")  η ", round(η, sigdigits = 3), " (code ", round(st.normal_velocity_defect, sigdigits = 3),
                ")  residual defects ", round(res_p, sigdigits = 3), " ", round(res_j, sigdigits = 3), "  all <= ε: ", all4, "  code converged: ", st.converged, "  iterations ", st.iterations
            )
            @test abs(V - st.mixture_volume_defect) <= 1.0e-9 * max(V, 1.0e-30) + 1.0e-18
            @test abs(η - st.normal_velocity_defect) <= 1.0e-9 * max(η, 1.0e-30) + 1.0e-18
            @test st.converged == all4
            if label == "converged"
                @test all4
            else
                @test !all4 || st.converged
            end
        end
        # ---------------- D2 mimetic ----------------
        for variant in (:joint_smoothed, :joint_face)
            msolver = MimeticProjectionSolver(field; density = PA.DENSITY, variant); mwork = MimeticProjectionWorkspace(msolver)
            v = PA.copy_velocity(v0)
            asm = AMR._assemble_mimetic_system!(msolver, mwork, dt, PA.copy_velocity(v0))
            st = project!(v, CellField(mesh), msolver, dt, mwork; tolerance = tol, max_iterations = 20000)
            M = mwork.matrix; b = mwork.right_hand_side; x = mwork.solution
            r = M * x - b
            idx = asm.pressure_indices; act = asm.active_faces; np = length(idx); nF = length(act); nl = size(asm.constraint, 1)
            Vo = PA.open_dual_volumes(field)
            τ = dt^2 * maximum(abs(r[np + i]) / (d.m[act[i]] * Vo[act[i]]^1.5) for i in 1:nF)
            mult = dt * maximum(abs(r[np + nF + i]) / asm.constraint_scales[i] for i in 1:nl)
            jump1 = [isnan(v[1].interior[f]) || isnan(v[2].interior[f]) ? 0.0 : v[2].interior[f] - v[1].interior[f] for f in 1:length(d.m)]
            cand = Matrix(asm.candidate_constraint) * jump1[act]
            η = max(maximum(abs.(cand) ./ asm.candidate_scales), isempty(asm.inactive_faces) ? 0.0 : maximum(abs.(jump1[asm.inactive_faces])))
            η_ret = maximum(abs.(Matrix(asm.constraint) * jump1[act]) ./ asm.constraint_scales)
            flux = PA.mixture_flux(field, v); V = dt * maximum(abs.(flux[open_c]) ./ vol[open_c])
            all3 = max(V, η, τ) <= tol
            println(
                rpad(name, 17), "D2 ", rpad(variant, 15), " V ", round(V, sigdigits = 3), " η ", round(η, sigdigits = 3), " (code ", round(st.normal_velocity_defect, sigdigits = 3), "; retained rows only ", round(η_ret, sigdigits = 3),
                ") τ ", round(τ, sigdigits = 3), " (code ", round(st.tangential_defect, sigdigits = 3), ") multiplier rows ", round(mult, sigdigits = 3), " candidates ", asm.candidate_count, " retained ", nl, "  converged ", st.converged
            )
            @test abs(V - st.mixture_volume_defect) <= 1.0e-9 * max(V, 1.0e-30) + 1.0e-18
            @test abs(η - st.normal_velocity_defect) <= 1.0e-9 * max(η, 1.0e-30) + 1.0e-13          # both are round-off-level at the solver tolerance
            @test abs(τ - st.tangential_defect) <= 1.0e-6 * max(τ, 1.0e-30) + 1.0e-18
            @test st.converged == all3
        end
    end
    # ---------------- D3 units ----------------
    field = PA.fixture(:uniform; solid = false)
    solver = CMDGFMProjectionSolver(field; density = PA.DENSITY); work = CMDGFMProjectionWorkspace(solver)
    results = []
    for s in (1.0e-6, 1.0, 1.0e6)
        v = PA.two_predictor(field, MersenneTwister(12))
        for ph in 1:2
            v[ph].interior .*= 1 / s; v[ph].boundary .*= 1 / s
        end        # velocity unit u -> u/s (a problem in other units)
        v0 = PA.copy_velocity(v)
        st = project!(v, CellField(field.mesh), solver, dt * s, work; tolerance = 1.0e-3, max_iterations = 5000)
        jump0 = maximum(abs(v0[2].interior[f] - v0[1].interior[f]) for f in eachindex(v0[1].interior) if !isnan(v0[1].interior[f]) && !isnan(v0[2].interior[f]))
        push!(results, (; s, iterations = st.iterations, converged = st.converged, V = st.mixture_volume_defect, η = st.normal_velocity_defect, jump0))
        println(
            "D3 velocity unit factor ", s, ": iterations ", st.iterations, ", converged ", st.converged, ", V ", st.mixture_volume_defect, ", η ", st.normal_velocity_defect,
            ", |[u*]|max ", jump0, ", η/|[u*]|max ", st.normal_velocity_defect / jump0
        )
    end
    # the predictor's jump scales exactly with the unit; the criterion η <= ε does not: the relative accuracy reached differs (printed, issue #94)
    @test isapprox(results[1].jump0 * 1.0e-6, results[2].jump0; rtol = 1.0e-12)
    @test isapprox(results[3].jump0 * 1.0e6, results[2].jump0; rtol = 1.0e-12)
    @test all(r -> r.converged, results)
    rel = [r.η / r.jump0 for r in results]
    println("D3: relative normal-jump accuracy η/|[u*]|max reached at ε = 1e-3 for the unit factors (1e-6, 1, 1e6): ", rel, "; ratios to unit factor 1: ", rel ./ rel[2])
    @test maximum(rel) / minimum(rel) > 10          # the criterion is not unit-free
    # ---------------- R1 regrid scaling ----------------
    for (name, field) in ["composite" => PA.fixture(:amr; solid = false), "composite+solid" => PA.fixture(:amr; solid = true)]
        mesh = field.mesh
        for method in (:cmdgfm, :mimetic)
            outs = []; mats = []; rhss = []; Vs = []
            for dtx in (dt, 3dt)
                solver = method == :cmdgfm ? CMDGFMProjectionSolver(field; density = PA.DENSITY) : MimeticProjectionSolver(field; density = PA.DENSITY)
                work = method == :cmdgfm ? CMDGFMProjectionWorkspace(solver) : MimeticProjectionWorkspace(solver)
                v = PA.two_predictor(field, MersenneTwister(21)); v0 = PA.copy_velocity(v)
                st = project!(v, CellField(mesh), solver, dtx, work; tolerance = 1.0e-13, max_iterations = 20000)
                @test st.converged
                push!(outs, v); push!(mats, copy(Matrix(work.matrix))); push!(rhss, work.right_hand_side .* dtx)
                vol = PA.open_volumes(field)
                push!(Vs, dtx * maximum(abs.(PA.mixture_flux(field, v0)[vol .> 0]) ./ vol[vol .> 0]))
            end
            dv = maximum(abs(outs[1][ph].interior[f] - outs[2][ph].interior[f]) for ph in 1:2 for f in eachindex(outs[1][1].interior) if !isnan(outs[1][ph].interior[f])) /
                maximum(abs(outs[1][ph].interior[f]) for ph in 1:2 for f in eachindex(outs[1][1].interior) if !isnan(outs[1][ph].interior[f]))
            dM = maximum(abs.(mats[1] .- mats[2])) / maximum(abs.(mats[1]))
            dr = maximum(abs.(rhss[1] .- rhss[2])) / maximum(abs.(rhss[1]))
            println("R1 ", rpad(name, 16), rpad(method, 9), " velocities dt vs 3dt: ", round(dv, sigdigits = 3), "; matrix ", dM, "; dt·rhs ", round(dr, sigdigits = 3), "; V(3dt)/V(dt) = ", Vs[2] / Vs[1])
            @test dv <= 1.0e-7
            @test dM <= 1.0e-15 && dr <= 1.0e-13
            @test Vs[2] / Vs[1] ≈ 3 rtol = 1.0e-12
        end
    end
end
