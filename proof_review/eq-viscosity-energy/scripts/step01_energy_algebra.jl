# Step 01: the algebra behind eq:viscosity:energy and eq:viscosity:momentum, on random symmetric PSD stiffness S and positive
# diagonal mass M (no package code).
#
# Label:  eq:viscosity:energy, eq:viscosity:momentum, eq:viscosity:phase-midpoint
# Source: docs/techref/chapters/09_viscosity.tex:376-407
# Claims: with  A = M + (dt/2) S,  rhs = M u1 - (dt/2) S u1,  u2 = A^{-1}(rhs - s) for a perturbation s,
#   (a) r := A u2 - rhs = M (u2-u1) + (dt/2) S (u1+u2)   (the residual with the sign that makes the printed identities true),
#   (b) (1/2)u2'M u2 - (1/2)u1'M u1 = -(dt/4) w'S w + (1/2) w' r,   w = u1+u2           [eq:viscosity:energy],
#   (c) if S 1 = 0:  1'M(u2-u1) = 1' r                                                    [eq:viscosity:momentum],
#   (d) the solver convention r_solver = rhs - A u2 = -r flips the sign of the right-hand sides of (b), (c).
#   (e) exact solve (r = 0): ||u2||_M <= ||u1||_M for every dt > 0 (A-stability), equality iff S w = 0 for w = u1+u2.
# Run:    julia --project=@claude-symbolics proof_review/eq-viscosity-energy/scripts/step01_energy_algebra.jl
#         (plain Julia stdlib only; any environment works)
using LinearAlgebra, Random, Test

@testset "energy/momentum algebra" begin
    rng = MersenneTwister(20261109)
    worst_b = 0.0; worst_c = 0.0; worst_flip = Inf; worst_e = -Inf
    for trial in 1:200
        n = rand(rng, 5:40)
        B = randn(rng, n, rand(rng, 2:n))
        S = B * B'
        # make the vector 1 a null vector: project out the mean from the rows of B
        P = I - ones(n, n) / n
        S0 = P * S * P
        S0 = (S0 + S0') / 2
        M = Diagonal(exp.(randn(rng, n)))
        dt = 10.0^(4 * rand(rng) - 2)
        u1 = randn(rng, n)
        s = randn(rng, n) * 10.0^(-3 * rand(rng))                       # inexact solve
        A = M + dt / 2 * S0
        rhs = M * u1 - dt / 2 * S0 * u1
        u2 = A \ (rhs - s)
        r = A * u2 - rhs                                                  # = -s
        @test r ≈ -s atol = 1.0e-9 * norm(s)
        w = u1 + u2
        E2 = 0.5 * dot(u2, M * u2); E1 = 0.5 * dot(u1, M * u1)
        lhs = E2 - E1
        rhs_b = -dt / 4 * dot(w, S0 * w) + 0.5 * dot(w, r)
        scale = abs(E1) + abs(E2) + abs(dt / 4 * dot(w, S0 * w))
        worst_b = max(worst_b, abs(lhs - rhs_b) / scale)
        wrong = -dt / 4 * dot(w, S0 * w) - 0.5 * dot(w, r)
        worst_flip = min(worst_flip, abs(lhs - wrong) / scale)             # opposite sign convention: should NOT be ~ 0
        one = ones(n)
        mom = dot(one, M * (u2 - u1))
        worst_c = max(worst_c, abs(mom - dot(one, r)) / (norm(M * abs.(u1), 1) + 1))
        # exact solve: contractive
        u2e = A \ rhs
        worst_e = max(worst_e, 0.5 * dot(u2e, M * u2e) - 0.5 * dot(u1, M * u1))
    end
    println("(b) energy identity: max relative defect ", worst_b)
    println("(d) opposite residual sign: min relative defect ", worst_flip, " (must be far from 0)")
    println("(c) momentum identity: max relative defect ", worst_c)
    println("(e) exact solve: max (E2-E1) over trials ", worst_e, " (must be <= 0 up to round-off)")
    @test worst_b < 1.0e-9
    @test worst_c < 1.0e-9
    @test worst_flip > 1.0e-6
    @test worst_e < 1.0e-9
end
