# Code check 02: the stopping rule (eq:projection:one-tolerance) and its round-off floor.
#
# Label:  eq:projection:one-tolerance  (also the sentence "dt max|div v^{n+1}| equals the left-hand side when b̄ = 0")
# Source: docs/techref/chapters/10_projection.tex:97-122; src/solvers/multigrid/{elliptic_solver.jl,cycles.jl,stats.jl}
# Checks (composite fixture with and without the sloped solid; random predictor):
#   T1  with tolerance ε = 1e-9, 1e-12: dt² max_c |r_c|/|c^o| <= max(ε, dt² ε_fl) and the code's `converged` flag agrees; the strong
#       residual is evaluated independently (S, b from the text);
#   T2  floor: with ε = 1e-30 (unreachable), the stop level lies between the
#       initial and independent current-iterate bounds; a raised floor requires progress;
#   T3  dt max_c |div v^{n+1}| = dt² max_c |r_c|/|c^o| when b̄ = 0 (units: dimensionless fraction of cell volume per step);
#   T4  ε must lie in (0,1): the code throws otherwise.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_02_tolerance_floor.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR

const PA = ProjectionAudit

"""128 eps max(1, |terms of S p|/vol, |b|/vol) with the terms bounded face by face (sum of absolute values of the face products), as the text."""
function independent_floor(field, op, p, bint)
    mesh = field.mesh; nx, ny = mesh.block_size
    K = PA.metric_dense(op)
    γ = op.face_coefficient
    vol = PA.open_volumes(field)
    s = zeros(length(mesh.faces))
    for (i, face) in pairs(mesh.faces)
        mi = AMR._cell_index(face.minus, nx, ny); pi_ = AMR._cell_index(face.plus, nx, ny)
        s[i] = face.area * (abs(p[mi]) + abs(p[pi_]))     # |f| (|p-| + |p+|)
    end
    out = abs.(K) * (sqrt.(γ) .* s)                      # |K| (√γ |f|(|p-|+|p+|))
    terms = zeros(length(vol))
    for (i, face) in pairs(mesh.faces)
        w = face.area * sqrt(γ[i]) * out[i]
        terms[AMR._cell_index(face.minus, nx, ny)] += w
        terms[AMR._cell_index(face.plus, nx, ny)] += w
    end
    scale = 1.0
    for c in eachindex(vol)
        vol[c] > 0 || continue
        scale = max(scale, terms[c] / vol[c], abs(bint[c] / vol[c]))
    end
    return 128 * eps() * scale
end

rng = MersenneTwister(20261112)
dt = PA.DT
@testset "tolerance and floor" begin
    for (name, field) in ["composite" => PA.fixture(:amr; solid = false), "composite+solid" => PA.fixture(:amr; solid = true)]
        mesh = field.mesh
        op = FA.one_velocity_operator(field)
        J = PA.jump_dense(field); K = PA.metric_dense(op); γ = op.face_coefficient
        S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
        vol = PA.open_volumes(field); open_cells = findall(>(0), vol)
        for ε in (1.0e-9, 1.0e-12)
            v = PA.one_predictor(field, rng); v0 = PA.copy_velocity(v)
            p, stats, _ = PA.run_projection!(field, op, v; tolerance = ε)
            pv = FA.cell_vector(ref -> AMR._get(p, ref), mesh)
            b = -PA.open_flux(field, v0) ./ dt; b .-= sum(b) / sum(vol) .* vol
            r = b .- S * pv
            lhs = dt^2 * maximum(abs.(r[open_cells]) ./ vol[open_cells])
            initial_floor = independent_floor(field, op, zeros(length(pv)), b)
            final_floor = independent_floor(field, op, pv, b)
            println(rpad(name, 16), " ε=", ε, "  dt² max|r|/|c| = ", round(lhs, sigdigits = 3), "   dt² target = ", round(dt^2 * stats.target_linf, sigdigits = 3), "  converged=", stats.converged)
            @test initial_floor <= stats.target_linf * (1 + 1.0e-6)
            @test stats.target_linf <= max(ε / dt^2, initial_floor, final_floor) * (1 + 1.0e-6)
            @test lhs <= dt^2 * stats.target_linf * (1 + 1.0e-6)
            @test stats.converged
            # T3
            div = PA.open_flux(field, v)
            @test abs(dt * maximum(abs.(div[open_cells] ./ vol[open_cells])) - lhs) <= 1.0e-3 * lhs + 1.0e-14      # equal up to b̄ (≈ 1e-15)
        end
        # T2: unreachable tolerance, the floor decides
        v = PA.one_predictor(field, rng); v0 = PA.copy_velocity(v)
        p, stats, _ = PA.run_projection!(field, op, v; tolerance = 1.0e-30, max_cycles = 400)
        pv = FA.cell_vector(ref -> AMR._get(p, ref), mesh)
        b = -PA.open_flux(field, v0) ./ dt; b .-= sum(b) / sum(vol) .* vol
        initial_floor = independent_floor(field, op, zeros(length(pv)), b)
        final_floor = independent_floor(field, op, pv, b)
        println(rpad(name, 16), " ε=1e-30  code target_linf=", stats.target_linf, "  independent floors=", (initial_floor, final_floor), "  final_linf=", stats.final_linf, " converged=", stats.converged)
        @test initial_floor <= stats.target_linf * (1 + 1.0e-6)
        @test stats.target_linf <= max(initial_floor, final_floor) * (1 + 1.0e-6)
        if stats.target_linf > initial_floor * (1 + 1.0e-6)
            @test stats.final_linf < stats.initial_linf
            @test stats.target_linf < stats.initial_linf / 100
        end
        @test stats.converged == (stats.final_linf <= stats.target_linf)
        # T4
        for bad in (0.0, 1.0, -1.0, NaN)
            @test_throws ArgumentError PA.run_projection!(field, op, PA.one_predictor(field, rng); tolerance = bad)
        end
    end
end
