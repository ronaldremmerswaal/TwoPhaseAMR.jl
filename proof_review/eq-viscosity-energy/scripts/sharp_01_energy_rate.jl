# Sharpness 01: eq:viscosity:energy -- the identity is an equality; the energy factor of a generalised eigenmode is ((1-a l)/(1+a l))^2
# (a = dt/2), so the step is neutrally stable only on the kernel of S and loses no energy (|factor| -> 1) for stiff modes at large dt
# (no L-stability); the dissipation bound is attained on every eigenmode.
#
# Label:  eq:viscosity:energy
# Source: docs/techref/chapters/09_viscosity.tex:376-407
# Checks (periodic composite two-phase fixture; generalised eigenproblem S v = lambda M v, 1240-ish unknowns, dense):
#   X1  E2/E1 = ((1 - a lambda)/(1 + a lambda))^2 for every eigenmode (a = dt/2), dt = 0.05 and 5e3;
#   X2  the largest lambda with dt -> inf: factor -> 1 from below (stiff modes are not damped); eigenmodes with lambda = 0: factor exactly 1
#       (the two constants: energy and momentum conserved);
#   X3  the identity's right-hand side with r = 0 equals the left-hand side on every eigenmode (equality, so no better inequality exists);
#   X4  random vectors: E2/E1 <= 1 with max over 2000 draws approaching the bound only on near-kernel vectors (worst ratio reported).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-energy/scripts/sharp_01_energy_rate.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

@testset "energy rate of eigenmodes" begin
    field = fixture_periodic()
    op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
    density = mixture_density(field, FIX_DENSITY)
    M = density .* op.volumes
    S = Matrix(op.stiffness)
    Mh = Diagonal(1 ./ sqrt.(M))
    lam, W = eigen(Symmetric(Mh * S * Mh))            # S v = lam M v with v = Mh W
    V = Mh * W
    println("generalised eigenvalues: min ", lam[1], ", second ", lam[3], ", max ", lam[end], "; kernel dim ", count(<(1.0e-10 * lam[end]), lam))
    for dt in (0.05, 5.0e3)
        a = dt / 2
        A = Diagonal(M) + a * S
        worst1 = 0.0; worst3 = 0.0
        for k in (1, 2, 3, 10, 100, length(lam) ÷ 2, length(lam) - 1, length(lam))
            u1 = V[:, k]
            u2 = A \ (M .* u1 .- a .* (S * u1))
            ratio = dot(u2, M .* u2) / dot(u1, M .* u1)
            predicted = ((1 - a * lam[k]) / (1 + a * lam[k]))^2
            worst1 = max(worst1, abs(ratio - predicted))
            w = u1 .+ u2
            lhs = (dot(u2, M .* u2) - dot(u1, M .* u1)) / 2
            rhs = -dt / 4 * dot(w, S * w)
            worst3 = max(worst3, abs(lhs - rhs) / (abs(lhs) + 1.0e-300 + dot(u1, M .* u1)))
        end
        top = length(lam)
        a * lam[top] > 1 && println("dt = ", dt, ": largest mode factor ", ((1 - a * lam[top]) / (1 + a * lam[top]))^2, " (a lambda_max = ", a * lam[top], ")")
        println("dt = ", dt, ": max |E2/E1 - ((1-a l)/(1+a l))^2| = ", worst1, ", identity (r = 0) defect ", worst3)
        @test worst1 <= 1.0e-9
        @test worst3 <= 1.0e-9
    end
    # kernel: exactly conserved
    kern = [k for k in eachindex(lam) if lam[k] < 1.0e-10 * lam[end]]
    for k in kern
        u1 = V[:, k]; A = Diagonal(M) + 0.025 * S
        u2 = A \ (M .* u1 .- 0.025 .* (S * u1))
        @test abs(dot(u2, M .* u2) / dot(u1, M .* u1) - 1) <= 1.0e-12
    end
    # random vectors: contractive
    rng = MersenneTwister(7)
    dt = 0.05; A = Diagonal(M) + dt / 2 * S
    worst = 0.0
    for _ in 1:200
        u1 = randn(rng, length(M)); u2 = A \ (M .* u1 .- dt / 2 .* (S * u1))
        worst = max(worst, dot(u2, M .* u2) / dot(u1, M .* u1))
    end
    println("random vectors: max E2/E1 = ", worst)
    @test worst <= 1
end
