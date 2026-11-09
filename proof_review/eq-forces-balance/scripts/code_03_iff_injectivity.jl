# Code check 03: "balanced iff J_f = (D a)_f on every face with gamma_f > 0": the converse direction rests on the force map
# being injective on the support {gamma > 0}, i.e. on the metric being positive definite there.
#
# Label:  eq:forces:balance (and eq:forces:hydrostatic, "if and only if", "does not depend on the metric")
# Source: docs/techref/chapters/08_forces.tex:137-212, 440-467; src/embedded/operators.jl (GradientOperator(geometry))
# Checks (composite meshes: without solid, sloped solid, disc; and the uniform disc):
#   C1  the matrix  M = diag(sqrt(gamma)) K diag(sqrt(gamma))  restricted to S = {gamma > 0}  is symmetric positive definite
#       (smallest eigenvalue > 0; K the inverse face metric of the code, including the cut-cell scaling W);
#   C2  consequently R J = S p  with J = D a on S forces J_S = (D p)_S: checked as  min_p |R J - S p| / |R J|  for (i) J = D a
#       (~0), (ii) a generic J (random, O(1)), (iii) J = D a + delta e_f for one face f in S (grows linearly in delta);
#   C3  the lower bound |R(J - D p)| >= lambda_min(M_SS)/max sqrt(gamma)... is not needed; the chain C1 => injective is exact.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-balance/scripts/code_03_iff_injectivity.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

rng = MersenneTwister(20261111)
results = Dict{Symbol, Any}()
for (name, field) in FA.balance_fixtures()
    name in (:uniform,) && continue
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    solid = !isnothing(field.geometry)
    a = FA.apertures(field)
    rho = AMR.mixture_density(field, FA.DENSITY)
    gamma = solid ? a .^ 2 ./ rho : 1 ./ rho
    S = findall(>(0), gamma)
    K = FA.metric_matrix(op)
    M = Symmetric(Matrix(Diagonal(sqrt.(gamma[S])) * K[S, S] * Diagonal(sqrt.(gamma[S]))))
    evals = eigvals(M)
    asym = norm(Matrix(K) - Matrix(K)') / norm(Matrix(K))
    D = Matrix(FA.jump_matrix(mesh))
    # dense weighted gradient S_g = R D
    Rapply(J) = FA.force_map(K, gamma, a, J, solid)
    RD = hcat((Rapply(D[:, k]) for k in 1:size(D, 2))...)
    residual(J) = (r = Rapply(J); norm(r - RD * (RD \ r)) / norm(r))
    r_exact = residual(D * randn(rng, size(D, 2)))
    r_random = residual(randn(rng, size(D, 1)) .* (gamma .> 0))
    base = D * randn(rng, size(D, 2))
    f = S[length(S) ÷ 2]
    deltas = (1.0e-3, 1.0e-2, 1.0e-1)
    r_pert = [(J = copy(base); J[f] += d * maximum(abs, base); residual(J)) for d in deltas]
    results[name] = (; emin = first(evals), emax = last(evals), asym, r_exact, r_random, r_pert)
    println(
        rpad("fixture $name", 14), " |S|=", length(S), " eig(M_SS) in [", round(first(evals), sigdigits = 3), ", ", round(last(evals), sigdigits = 3),
        "], K asym ", round(asym, sigdigits = 2), "; residual: exact ", round(r_exact, sigdigits = 2), " random ", round(r_random, sigdigits = 2),
        " one-face perturbation (1e-3,1e-2,1e-1) ", round.(r_pert, sigdigits = 2)
    )
end
@testset "iff: injectivity of the force map on the support" begin
    for (name, r) in results
        @test r.emin > 0
        @test r.asym <= 1.0e-13
        @test r.r_exact <= 1.0e-10
        @test r.r_random >= 0.1
        @test issorted(r.r_pert) && r.r_pert[1] > 1.0e-6          # a single-face error is seen, proportionally
    end
end
