# Code check 01: the mimetic projection is the constrained kinetic-energy minimiser; the KKT system (eq:projection:kkt) against an independent dense
# assembly; the active/inactive rules; the gauge partition.
#
# Labels: eq:projection:active, eq:projection:inactive, eq:projection:mimetic-min, eq:projection:kkt, eq:projection:gauge (partition)
# Source: docs/techref/chapters/10_projection.tex:469-570
# Checks (4 fixtures x variants :joint_smoothed, :joint_face; random predictor with phase jump; tolerance 1e-10):
#   K1  active set (eq. active) from the text equals the code's `active_faces`; inactive = mixed \ active;
#   K2  dense KKT matrix [[-K_S, C, 0],[C', D, N'],[0, N, 0]] with C = -J_{F,r}' diag(apgfm), D = diag(m^r V^o_F) and the code's fitted rows N, and the right-hand side
#       (1/dt)[ρ; 0; N [u*]_F], ρ = |c^o| D^mix(u*) + J_{inactive,r}' diag(apgfm) [u*]_{inactive}, equal the code's assembled matrix/right-hand side;
#       the matrix is symmetric (exactly), indefinite, nonsingular; inertia (n_ξ positive, n_p + n_N negative);
#   K3  the minimiser of E_def = ½|δū|²_M + ½ Σ_F m^r V^o δ[u]²  over the corrections with D^mix(u^{n+1}) = 0 on the retained cells and
#       N [u^{n+1}] = 0 on F (inactive jumps removed) found by the NULL-SPACE method (independent of the multiplier formulation) equals
#       (δū, δ[u]) = (-dt R_tv J p, -dt ξ) of the code's solution; the code's correction is feasible and the null-space gradient vanishes;
#   K4  eq. inactive: ξ_f = [u*]_f/dt and [u^{n+1}]_f = 0 on the inactive faces.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-kkt/scripts/code_01_kkt_minimisation.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

"""Active faces per eq:projection:active, written from the text."""
function active_faces_text(field, d; ε = 1.0e-6)
    mesh = field.mesh; nx, ny = mesh.block_size; fl = d.floor
    function uncut(ref)
        isnothing(field.geometry) && return true
        return AMR._get(field.geometry.open_fraction, ref) >= 1 - fl
    end
    act = Int[]
    for f in d.mixed
        face = mesh.faces[f]
        uncut(face.minus) && uncut(face.plus) || continue
        min(d.ml[f], d.mg[f]) / d.m[f] >= ε || continue
        d.al[f] + d.ag[f] > fl || continue
        push!(act, f)
    end
    return act
end

@testset "mimetic KKT system and minimisation" begin
    for (name, field) in PA.fixtures(), variant in (:joint_smoothed, :joint_face)
        mesh = field.mesh; nx, ny = mesh.block_size
        dt = PA.DT
        solver = MimeticProjectionSolver(field; density = PA.DENSITY, variant)
        work = MimeticProjectionWorkspace(solver)
        d = PA.face_data(field)
        v0 = PA.two_predictor(field, MersenneTwister(47))
        vstar = PA.copy_velocity(v0)
        assembled = AMR._assemble_mimetic_system!(solver, work, dt, PA.copy_velocity(v0))
        act = assembled.active_faces
        # K1
        @test sort(act) == sort(active_faces_text(field, d))
        inact = sort(setdiff(d.mixed, act))
        @test sort(assembled.inactive_faces) == inact
        idx = assembled.pressure_indices
        nF = length(act); npr = length(idx)
        op = work.cache.pressure.operator
        a = PA.aperture(field)
        γ = isnothing(field.geometry) ? 1 ./ d.m : a ./ d.m
        J = PA.jump_dense(field); K = PA.metric_dense(op)
        R = PA.force_map_dense(field, op, γ)
        S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
        Jr = J[:, idx]
        Vo = PA.open_dual_volumes(field)
        Dm = Diagonal(d.mr[act] .* Vo[act])
        C = -(Jr[act, :]') * Diagonal(d.apgfm[act])
        N = Matrix(assembled.constraint)
        nN = size(N, 1)
        jump_star = [isnan(vstar[1].interior[f]) || isnan(vstar[2].interior[f]) ? 0.0 : vstar[2].interior[f] - vstar[1].interior[f] for f in 1:length(d.m)]
        flux = PA.mixture_flux(field, vstar)
        ρ = flux[idx] .+ (Jr[inact, :]') * (d.apgfm[inact] .* jump_star[inact])
        Mkkt = [-S[idx, idx] C zeros(npr, nN); C' Matrix(Dm) N'; zeros(nN, npr) N zeros(nN, nN)]
        rhs = vcat(ρ, zeros(nF), N * jump_star[act]) ./ dt
        Mc = Matrix(assembled.matrix)
        e_M = maximum(abs.(Mc .- Mkkt)) / maximum(abs.(Mkkt))
        e_b = maximum(abs.(assembled.right_hand_side .- rhs)) / maximum(abs.(rhs))
        sym = maximum(abs.(Mc .- Mc')) / maximum(abs.(Mkkt))          # the code's assembled matrix
        sym_mine = maximum(abs.(Mkkt .- Mkkt')) / maximum(abs.(Mkkt))     # my product J'(√γ K √γ)J: round-off only
        ev = eigvals(Symmetric(Mkkt))
        npos, nneg = count(>(0), ev), count(<(0), ev)
        smin = minimum(abs.(ev)) / maximum(abs.(ev))
        println(
            rpad(name, 18), rpad(variant, 15), " K1 ok (", nF, " active, ", length(inact), " inactive); K2: matrix ", round(e_M, sigdigits = 3), " rhs ", round(e_b, sigdigits = 3),
            " sym ", sym, " inertia (+", npos, ",-", nneg, ") |λ|min/max ", round(smin, sigdigits = 2), "  rows ", nN, " (candidates ", assembled.candidate_count, ")"
        )
        @test e_M <= 1.0e-12 && e_b <= 1.0e-12
        @test sym == 0.0 && sym_mine <= 1.0e-15
        @test npos == nF && nneg == npr + nN
        # project
        velocity = PA.copy_velocity(v0)
        pressure = CellField(mesh)
        stats = project!(velocity, pressure, solver, dt, work; tolerance = 1.0e-10, max_iterations = 20000)
        @test stats.converged
        pv = FA.cell_vector(ref -> AMR._get(pressure, ref), mesh)
        ξ = copy(work.jump)
        # K4
        if !isempty(inact)
            @test maximum(abs.(ξ[inact] .- jump_star[inact] ./ dt)) <= 1.0e-9 * maximum(abs.(jump_star[inact]) / dt)
            jn = [velocity[2].interior[f] - velocity[1].interior[f] for f in inact]
            println("    K4: max|[u^{n+1}]| on the ", length(inact), " inactive faces = ", maximum(abs.(jn)), " (|[u*]| max ", maximum(abs.(jump_star[inact])), ")")
            @test maximum(abs.(jn)) <= 1.0e-9
        end
        # K3: independent minimisation by the null-space method
        P = findall(>(0), γ)
        Mq = (a[P] ./ sqrt.(γ[P])) .* inv(Symmetric(K[P, P])) .* (a[P] ./ sqrt.(γ[P]))'     # kinetic metric on the common velocity (faces P)
        # unknowns z = (δū_P, δ[u]_act); constraints G z = h
        G1 = hcat(-(Jr[P, :]') * Diagonal(a[P]), -(Jr[act, :]') * Diagonal(d.apgfm[act]))
        h1 = -flux[idx] .- (Jr[inact, :]') * (d.apgfm[inact] .* jump_star[inact])
        G2 = hcat(zeros(nN, length(P)), N)
        h2 = -(N * jump_star[act])
        G = vcat(G1, G2); h = vcat(h1, h2)
        W = cat(Mq, Matrix(Dm); dims = (1, 2))
        zc = vcat(-dt .* (R * (J * pv))[P], -dt .* ξ[act])                            # the code's correction
        feas = norm(G * zc .- h) / max(norm(h), 1.0e-300)
        F = svd(G; full = true)
        rk = count(>(1.0e-10 * F.S[1]), F.S)
        Z = F.V[:, (rk + 1):end]
        grad = Z' * (W * zc)                                                                 # stationarity in the null space of the constraints
        zp = pinv(G; rtol = 1.0e-10) * h                                                       # a particular solution
        y = -(Z' * W * Z) \ (Z' * W * zp)
        zopt = zp .+ Z * y
        e_opt = norm(zopt .- zc) / norm(zc)
        Eopt = 0.5 * zopt' * W * zopt; Ec = 0.5 * zc' * W * zc
        println(
            "    K3: constraints ", size(G, 1), " (rank ", rk, ") on ", size(G, 2), " unknowns; feasibility ", round(feas, sigdigits = 3), "; |Zᵀ W z|/|W z| = ",
            round(norm(grad) / norm(W * zc), sigdigits = 3), "; |z_opt - z_code|/|z_code| = ", round(e_opt, sigdigits = 3), "; E_code/E_opt - 1 = ", Ec / Eopt - 1
        )
        @test feas <= 1.0e-8
        @test norm(grad) / norm(W * zc) <= 1.0e-7
        @test e_opt <= 1.0e-6
    end
end
