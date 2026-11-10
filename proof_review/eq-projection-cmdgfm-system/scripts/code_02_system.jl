# Code check 02: the CMDGFM rows, jump rows, interpolant and the two-block system against an independent dense assembly.
#
# Labels: eq:projection:cmdgfm-row, eq:projection:cmdgfm-jump-row, eq:projection:interpolant, eq:projection:cmdgfm-system, eq:projection:gauge (partition),
#         eq:projection:mixture-update (matrix form), eq:projection:phase-gradient/-identities on the solution
# Source: docs/techref/chapters/10_projection.tex:229-260, 328-395, 362-395
# Checks (4 fixtures; random predictor with phase jump; tolerance 1e-10):
#   C1  the gauge partition: anchors = smallest cell index per connected component of (open cells, faces with a_f > 0), retained cells = the rest;
#   C2  dense matrix [[K_S, J_r' diag(apgfm)],[B_p, B_ξ]] and right-hand side [-|c^o|D^mix/dt ; N(v*)/dt] built from the text equal the code's assembled
#       matrix/right-hand side (entry-wise max difference relative to the largest entry);
#   C3  N_f(u) (eq. cmdgfm-row) = (β/n_k) n·J^β(u) with the code's `cmdgfm_jump_interpolant` (β > 0);
#   C4  after project!, the independently evaluated N_f(u^{n+1}) = 0 (f in the resolved set), |c^o| D^mix(u^{n+1}) = 0 on all open cells, the phase
#       identities (phase-identities) hold, and |c^o| D^mix(u^{n+1}) = |c^o| D^mix(u*) + dt (S_tv p + J' diag(apgfm) ξ) cell by cell (mixture-update);
#   C5  the pressure solution agrees with the dense solve of the independent system (relative 1e-6 in the velocity update).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-cmdgfm-system/scripts/code_02_system.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit
isdefined(Main, :classify) || include(joinpath(@__DIR__, "code_01_stencils_lib.jl"))

"""Independent N_f(u) of eq. cmdgfm-row for a resolved stencil `s` (from `classify`)."""
function normal_jump(s, f, u)
    jump(g) = u[2].interior[g] - u[1].interior[g]
    val = jump(f)
    for (g, w) in zip(s.trans, s.w)
        val += s.beta * s.R * w * jump(g)
    end
    s.ig > 0 && (val += (s.beta - 1) * (u[2].interior[s.ig] - u[1].interior[s.il]))
    return val
end

@testset "CMDGFM system against an independent assembly" begin
    for (name, field) in PA.fixtures()
        mesh = field.mesh; nx, ny = mesh.block_size
        dt = PA.DT
        solver = CMDGFMProjectionSolver(field; density = PA.DENSITY)
        work = CMDGFMProjectionWorkspace(solver)
        d = PA.face_data(field)
        ind = classify(field, d)
        resolved = sort([f for (f, s) in ind if s.class == :resolved])
        v0 = PA.two_predictor(field, MersenneTwister(31))
        vstar = PA.copy_velocity(v0)
        assembled = AMR._assemble_cmdgfm_system!(solver, work, dt, PA.copy_velocity(v0))
        F = assembled.mixed_faces
        @test sort(F) == resolved
        idx = assembled.pressure_indices
        # C1 gauge partition
        vol = PA.open_volumes(field); a = PA.aperture(field)
        n = length(vol)
        adj = [Int[] for _ in 1:n]
        for (i, face) in pairs(mesh.faces)
            a[i] > 0 || continue
            m = AMR._cell_index(face.minus, nx, ny); q = AMR._cell_index(face.plus, nx, ny)
            (vol[m] > 0 && vol[q] > 0 && m != q) || continue
            push!(adj[m], q); push!(adj[q], m)
        end
        comp = zeros(Int, n); anchors = Int[]
        for c in 1:n
            (vol[c] > 0 && comp[c] == 0) || continue
            push!(anchors, c); comp[c] = c; st = [c]
            while !isempty(st)
                x = pop!(st)
                for y in adj[x]
                    comp[y] == 0 && (comp[y] = c; push!(st, y))
                end
            end
        end
        retained = [c for c in 1:n if vol[c] > 0 && !(c in anchors)]
        @test idx == retained
        @test assembled.pressure_anchors == [comp[c] for c in retained]
        # operators
        op = AMR._projection_pressure_operator(solver)
        γ = isnothing(field.geometry) ? 1 ./ d.m : a ./ d.m
        pp = findall(>(0), γ); @test maximum(abs.(op.face_coefficient[pp] .- γ[pp]) ./ γ[pp]) <= 1.0e-13; @test all(op.face_coefficient[setdiff(1:length(γ), pp)] .== 0)
        J = PA.jump_dense(field); K = PA.metric_dense(op)
        R = PA.force_map_dense(field, op, γ)
        S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
        Jr = J[:, idx]
        # C2 matrix
        nF = length(F); npress = length(idx)
        pos = Dict(f => i for (i, f) in pairs(F))
        Mtop = hcat(S[idx, idx], Jr[F, :]' * Diagonal(d.apgfm[F]))
        Bp = zeros(nF, npress); Bx = Matrix{Float64}(I, nF, nF); rhs_b = zeros(nF)
        for (row, f) in pairs(F)
            s = ind[f]
            for (g, w) in zip(s.trans, s.w)
                haskey(pos, g) && (Bx[row, pos[g]] += s.beta * s.R * w)
            end
            if s.ig > 0
                haskey(pos, s.ig) && (Bx[row, pos[s.ig]] += (s.beta - 1) * d.ml[s.ig] / d.m[s.ig])
                haskey(pos, s.il) && (Bx[row, pos[s.il]] += (s.beta - 1) * d.mg[s.il] / d.m[s.il])
                Bp[row, :] .= (s.beta - 1) .* ((R[s.ig, :] .- R[s.il, :])' * Jr)[:]
            end
            rhs_b[row] = normal_jump(s, f, vstar) / dt
        end
        M = vcat(Mtop, hcat(Bp, Bx))
        flux = PA.mixture_flux(field, vstar)
        rhs = vcat(-flux[idx] ./ dt, rhs_b)
        Mc = Matrix(assembled.matrix)
        e_M = maximum(abs.(Mc .- M)) / maximum(abs.(M))
        e_b = maximum(abs.(assembled.right_hand_side .- rhs)) / maximum(abs.(rhs))
        println(rpad(name, 18), " C1 ok; C2: matrix ", round(e_M, sigdigits = 3), " rhs ", round(e_b, sigdigits = 3), "  size ", size(M, 1), " (", npress, " p + ", nF, " ξ), anchors ", length(anchors))
        @test e_M <= 1.0e-12
        @test e_b <= 1.0e-12
        # C3 interpolant
        n_checked = 0
        for st in AMR._cmdgfm_jump_stencils(solver, assembled.data)
            st.resolved && st.beta > sqrt(eps()) && !st.full_remove || continue
            s = ind[st.face]
            J1, J2 = AMR.cmdgfm_jump_interpolant(vstar, st)
            axis = mesh.faces[st.face].axis
            Jk, Jt = axis == 1 ? (J1, J2) : (J2, J1)
            nk, nt = s.n[axis], s.n[3 - axis]
            @test abs(s.beta / nk * (nk * Jk + nt * Jt) - normal_jump(s, st.face, vstar)) <= 1.0e-12 * max(1, abs(normal_jump(s, st.face, vstar)))
            n_checked += 1
        end
        # project
        velocity = PA.copy_velocity(v0)
        pressure = CellField(mesh)
        stats = project!(velocity, pressure, solver, dt, work; tolerance = 1.0e-10, max_iterations = 5000)
        @test stats.converged
        pv = FA.cell_vector(ref -> AMR._get(pressure, ref), mesh)
        ξ = copy(work.jump)
        # C4
        N1 = maximum(abs(normal_jump(ind[f], f, velocity)) for f in F)
        flux1 = PA.mixture_flux(field, velocity)
        Sg = R * (J * pv)
        ul_ = [velocity[1].interior[f] for f in 1:length(d.m)]
        # phase gradients from the text: S^φ = S + ζ^φ/m ξ
        Sl = Sg .+ d.zl ./ d.m .* ξ; Sg_ = Sg .+ d.zg ./ d.m .* ξ
        okl = [!isnan(vstar[1].interior[f]) for f in eachindex(Sl)]; okg = [!isnan(vstar[2].interior[f]) for f in eachindex(Sl)]
        e_upd = max(maximum(abs.(velocity[1].interior[okl] .- (vstar[1].interior[okl] .- dt .* Sl[okl]))), maximum(abs.(velocity[2].interior[okg] .- (vstar[2].interior[okg] .- dt .* Sg_[okg]))))
        e_upd /= maximum(abs.(vstar[2].interior[okg]))
        flux0 = flux
        pred = flux0 .+ dt .* (S * pv .+ J' * (d.apgfm .* ξ))
        open_c = findall(>(0), vol)
        e_mix = maximum(abs.(flux1[open_c] .- pred[open_c])) / maximum(abs.(flux0[open_c]))
        defect = dt * maximum(abs.(flux1[open_c]) ./ vol[open_c])
        gj = maximum(abs.((Sg_ .- Sl)[okl .& okg] .- ξ[okl .& okg])); gm = maximum(abs.((d.ml .* Sl .+ d.mg .* Sg_ .- d.m .* Sg)[okl .& okg] ./ d.m[okl .& okg]))
        println(
            "    C4: max|N_f(u^{n+1})| = ", N1, " (code ", stats.normal_velocity_defect, "), dt max|D^mix| = ", defect, " (code ", stats.mixture_volume_defect,
            "), update ", round(e_upd, sigdigits = 3), ", mixture-update ", round(e_mix, sigdigits = 3), ", identities ", gj, " ", gm, ", interpolant checks ", n_checked
        )
        @test N1 <= 1.0e-9 && defect <= 1.0e-9
        @test e_upd <= 1.0e-12
        @test e_mix <= 1.0e-8
        @test gj <= 1.0e-12 && gm <= 1.0e-12
        # C5 dense solve
        x = M \ rhs
        pd = zeros(n); pd[idx] .= x[1:npress]
        ξd = zeros(length(d.m)); ξd[F] .= x[(npress + 1):end]
        e5 = maximum(abs.(dt .* (R * (J * (pd .- pv))))) / maximum(abs.(dt .* (R * (J * pv))))
        e5x = maximum(abs.(ξd .- ξ)) / maximum(abs.(ξ))
        println("    C5: dense solve vs code: pressure-gradient update ", round(e5, sigdigits = 3), ", ξ ", round(e5x, sigdigits = 3), "  cond(M)≈", round(cond(M), sigdigits = 2))
        @test e5 <= 1.0e-6 && e5x <= 1.0e-6
    end
end
