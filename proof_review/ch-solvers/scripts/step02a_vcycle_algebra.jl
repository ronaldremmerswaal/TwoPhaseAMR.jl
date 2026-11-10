# Audit ch:solvers -- claim 2 (algebraic part): a projected V-cycle with restriction = prolongation^T, the same smoother
# with the same number of sweeps before and after the coarse correction, a symmetric positive semidefinite coarse
# preconditioner and component projections in transposed pairs is a symmetric positive semidefinite B whose range is
# the volume-mean-free space and which is positive definite there as soon as omega * rho(D^-1 A) < 2 -- for ARBITRARY
# prolongations and ARBITRARY SPD/SPSD coarse operators (no Galerkin product).
#
# Proof sketch (energy inner product <u,v>_A = u'Av on W = {x : v_C' x = 0}): on W the error propagation of one cycle is
#   I - B A = E T E,   E = (Pi S)^nu  (S = I - omega D^-1 A),   T = I - Pi P B_c P' A.
# S is A-selfadjoint, hence so is Pi S on W (A sees no null vectors); A T = A - A P B_c P' A is symmetric, and
# <T x, x>_A = |x|_A^2 - (A x)' P B_c P' (A x) <= |x|_A^2 because B_c is PSD. With 0 < omega lambda < 2 on the nonzero
# spectrum of D^-1 A, |E x|_A < |x|_A on W \ {0}, so <B A e, e>_A >= |e|_A^2 - |E e|_A^2 > 0.
#
# Source: docs/techref/chapters/11_solvers.tex:498-515 (Symmetry and definiteness). Cycle: src/solvers/multigrid/cycles.jl:46-78.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step02a_vcycle_algebra.jl
using Test, Random, Printf, LinearAlgebra

# Random weighted graph Laplacian on n nodes (connected ring + random chords), optionally with a Dirichlet anchor.
function random_spsd(rng, n; anchored = false, scale = 1.0)
    A = zeros(n, n)
    for i in 1:n
        for j in (mod1(i + 1, n), rand(rng, 1:n))
            j == i && continue
            w = scale * 10.0^(2 * (rand(rng) - 0.5))
            A[i, i] += w; A[j, j] += w; A[i, j] -= w; A[j, i] -= w
        end
    end
    anchored && (A[1, 1] += scale)
    return A
end

struct Level
    A::Matrix{Float64}
    V::Vector{Float64}          # volume weights (mean gauge)
    singular::Bool
    P::Matrix{Float64}          # prolongation to this level from the next coarser one (empty on the coarsest)
end

omega_of(A) = (g = maximum(sum(abs, A[i, :]) / A[i, i] for i in axes(A, 1) if A[i, i] > 0); min(1.0, 4 / (3g)))

project_solution(lev, x) = lev.singular ? x .- sum(lev.V .* x) / sum(lev.V) : x      # Pi
project_rhs(lev, r) = lev.singular ? r .- lev.V .* (sum(r) / sum(lev.V)) : r          # Pi'

function cycle(levels, l, r; nu1 = 2, nu2 = 2, weight = omega_of)
    lev = levels[l]
    r = project_rhs(lev, r)
    if l == length(levels)
        n = length(r)
        if lev.singular
            K = [lev.A lev.V; lev.V' 0.0]
            return (K \ [r; 0.0])[1:n]
        else
            return lev.A \ r
        end
    end
    omega = weight(lev.A)
    Dinv = 1 ./ diag(lev.A)
    x = zeros(length(r))
    for _ in 1:nu1
        x = project_solution(lev, x + omega * Dinv .* (r - lev.A * x))
    end
    rc = levels[l].P' * (r - lev.A * x)
    xc = cycle(levels, l + 1, rc; nu1, nu2, weight)
    x = project_solution(lev, x + levels[l].P * xc)
    for _ in 1:nu2
        x = project_solution(lev, x + omega * Dinv .* (r - lev.A * x))
    end
    return x
end

function random_hierarchy(rng; sizes = (30, 12, 5), singular = true, coarse_scale = 1.0, signed_P = true)
    levels = Level[]
    for (l, n) in enumerate(sizes)
        A = random_spsd(rng, n; anchored = !singular, scale = l == 1 ? 1.0 : coarse_scale)
        V = 0.5 .+ rand(rng, n)
        P = l < length(sizes) ? (signed_P ? randn(rng, n, sizes[l + 1]) : rand(rng, n, sizes[l + 1])) : zeros(n, 0)
        push!(levels, Level(A, V, singular, P))
    end
    return levels
end

# B as a dense matrix, plus the spectrum of B A on the mean-free space (Z'AZ = L L', eig of L'(Z'BZ)L).
function analyse(levels; kwargs...)
    n = size(levels[1].A, 1)
    B = zeros(n, n)
    for j in 1:n
        e = zeros(n); e[j] = 1
        B[:, j] = cycle(levels, 1, e; kwargs...)
    end
    A = levels[1].A
    asym = maximum(abs, B - B') / maximum(abs, B)
    if levels[1].singular
        N = nullspace(reshape(levels[1].V, 1, :))          # Z: orthonormal basis of {x: V'x = 0}
    else
        N = Matrix(1.0I, n, n)
    end
    L = cholesky(Symmetric(N' * A * N)).L
    M = L' * (N' * B * N) * L
    lam = eigvals(Symmetric((M + M') / 2))
    rangeerr = levels[1].singular ? maximum(abs, levels[1].V' * B) / maximum(abs, B) : 0.0
    return (; asym, lmin = first(lam), lmax = last(lam), rangeerr)
end

@testset "V-cycle: SPD without Galerkin coarse operators" begin
    rng = MersenneTwister(7)
    worst_asym = 0.0; worst_range = 0.0; lmin = Inf; lmax = 0.0
    for trial in 1:120
        singular = isodd(trial)
        levels = random_hierarchy(rng; singular, signed_P = trial % 3 != 0, sizes = rand(rng, (3, 4)) == 3 ? (30, 12, 5) : (40, 16, 8, 3))
        s = analyse(levels)
        @test s.asym <= 1.0e-12
        @test s.lmin > 0
        @test s.rangeerr <= 1.0e-12
        worst_asym = max(worst_asym, s.asym); worst_range = max(worst_range, s.rangeerr)
        lmin = min(lmin, s.lmin); lmax = max(lmax, s.lmax)
    end
    @printf(
        "  120 random hierarchies, arbitrary P and coarse operators: max asymmetry %.1e, max range error %.1e, spectrum of BA in [%.2e, %.2e]\n",
        worst_asym, worst_range, lmin, lmax
    )
    # lmax of BA can exceed one (the cycle need not converge as a stand-alone iteration) -- but B stays SPD.
    @test lmax > 1
end

@testset "badly scaled coarse operators keep B SPD but not convergent" begin
    rng = MersenneTwister(3)
    for scale in (1.0e-3, 1.0, 1.0e3)
        # a coarse operator scaled by `scale` is B_c / scale: T has eigenvalues 1 - mu, mu up to ~ 1/scale
        levels = random_hierarchy(rng; coarse_scale = scale)
        s = analyse(levels)
        @printf("  coarse scale %.0e: asym %.1e, spectrum [%.2e, %.2e]\n", scale, s.asym, s.lmin, s.lmax)
        @test s.asym <= 1.0e-12 && s.lmin > 0
    end
end

@testset "the hypotheses are operative" begin
    rng = MersenneTwister(5)
    # (a) different numbers of pre- and post-sweeps: B is not symmetric
    levels = random_hierarchy(rng)
    s = analyse(levels; nu1 = 1, nu2 = 2)
    @printf("  nu1=1, nu2=2: asymmetry %.2e\n", s.asym)
    @test s.asym > 1.0e-3
    # (b) a smoother with omega * rho(D^-1 A) > 2 (one sweep): B is indefinite for some draws
    bad = 0
    for trial in 1:40
        levels = random_hierarchy(rng; singular = false)
        s = analyse(levels; nu1 = 1, nu2 = 1, weight = A -> 2.6 / maximum(sum(abs, A[i, :]) / A[i, i] for i in axes(A, 1)))
        bad += s.lmin < 0
    end
    @printf("  omega*g = 2.6, one sweep: B indefinite in %d of 40 draws\n", bad)
    @test bad > 0
    # (c) the threshold is 2, not 4/3: omega*g = 1.95 keeps B SPD with one sweep
    for trial in 1:40
        levels = random_hierarchy(rng; singular = trial % 2 == 0)
        s = analyse(levels; nu1 = 1, nu2 = 1, weight = A -> 1.95 / maximum(sum(abs, A[i, :]) / A[i, i] for i in axes(A, 1)))
        @test s.lmin > 0
    end
    println("  omega*g = 1.95 (< 2): SPD in 40 of 40 draws")
end
