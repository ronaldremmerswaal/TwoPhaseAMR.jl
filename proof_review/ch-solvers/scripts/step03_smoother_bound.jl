# Audit ch:solvers -- claim 3: Jacobi weight omega = min(1, 4/(3 g)) with the Gershgorin bound
# g = max_i sum_j |A_ij| / A_ii  >= max |spec(D^-1 A)|, so omega * lambda <= 4/3 < 2 on the spectrum of D^-1 A
# (energy-norm convergence of the smoother); on the uniform five-point operator g = 2 and omega = 2/3; the transfer
# weight 4/(3g) of eq:solvers:smoothed-transfer is the same expression without the cap.
# Source: docs/techref/chapters/11_solvers.tex:313-322 (eq:solvers:gershgorin), 355-363 (eq:solvers:jacobi),
#   code src/solvers/multigrid/hierarchy.jl:8-30 (_jacobi_weight!).
# Checks per level of the hierarchies of the fixtures: g recomputed from the level matrix, omega of the code = min(1, 4/(3g)),
# spec(D^-1 A) in [0, g], omega lambda_max <= 4/3, symbol of the five-point operator (rho = g = 2 attained at the (pi, pi) mode).
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step03_smoother_bound.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

function level_report(preconditioner, name; dense_limit = 2700)
    worst_ratio = 0.0                  # max omega * lambda_max over levels (<= 4/3)
    gmax = 0.0
    for (k, level) in pairs(preconditioner.levels)
        A = level.matrix
        A isa SparseMatrixCSC || continue
        d = diag(A)
        rows = rowvals(A); vals = nonzeros(A)
        sums = zeros(length(d))
        for j in axes(A, 2), p in nzrange(A, j)
            sums[rows[p]] += abs(vals[p])
        end
        positive = d .> 0
        g = maximum(sums[positive] ./ d[positive])
        gmax = max(gmax, g)
        omega = level.relaxation_weight
        @test omega ≈ min(1.0, 4 / (3g)) rtol = 1.0e-14
        rho = NaN
        if size(A, 1) <= dense_limit
            idx = findall(positive)
            S = Diagonal(1 ./ sqrt.(d[idx])) * Matrix(A[idx, idx]) * Diagonal(1 ./ sqrt.(d[idx]))
            lam = eigvals(Symmetric((S + S') / 2))
            @test first(lam) >= -1.0e-12
            @test last(lam) <= g * (1 + 1.0e-12)         # Gershgorin
            @test omega * last(lam) <= 4 / 3 + 1.0e-12
            rho = last(lam)
            worst_ratio = max(worst_ratio, omega * rho)
        end
        @printf("    %-34s level %d  n=%5d  g=%.4f  omega=%.4f  rho(D^-1A)=%.4f  omega*rho=%.4f\n", name, k, size(A, 1), g, omega, rho, omega * rho)
    end
    return gmax
end

@testset "smoother bound omega*lambda <= 4/3" begin
    gs = Float64[]
    println("== levels of the fixtures ==")
    op = PoissonOperator(solv_uniform(32)); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "uniform 32^2, walls"))
    op = PoissonOperator(solv_uniform(32; periodic = (true, true))); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "periodic 32^2"))
    op = PoissonOperator(solv_amr(16)); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "composite base 16"))
    op = PoissonOperator(solv_drop_field(solv_amr(16)); density = (1.0e6, 1.0)); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "drop 1e6 composite"))
    geometry = GeometryField(solv_uniform(32), solv_band_solid; lipschitz = 1)
    op = PoissonOperator(geometry); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op; geometry), "split cut-cell 32^2"))
    geometry = GeometryField(solv_amr(16), solv_band_solid; lipschitz = 1)
    op = PoissonOperator(geometry); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op; geometry), "composite 16 + solid"))
    for sides in (VelocityBoundaryConditions(left = NoSlip(), right = Outflow(), bottom = NoSlip(), top = NoSlip()),)
        op = PoissonOperator(solv_uniform(32); velocity_boundary = sides); push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "uniform 32^2, Outflow right"))
    end
    # reaction > 0 (shifted cell operator)
    mesh = solv_uniform(32)
    for reaction in (0.5, 50.0, 5.0e4)
        op = EllipticOperator(mesh; reaction)
        push!(gs, level_report(TreeGeometricMultigridPreconditioner(op), "uniform 32^2, reaction $(reaction)"))
    end
    @printf("  largest Gershgorin bound over all levels of all fixtures: %.4f\n", maximum(gs))
    # for a pure Poisson operator (M-matrix rows, nonnegative row sums) g <= 2 always; with reaction g < 2
    @test maximum(gs) <= 2 + 1.0e-12

    println("== uniform five-point operator: symbol and Jacobi smoothing factors ==")
    # D^-1 A has symbol lambda(t1,t2) = 1 - (cos t1 + cos t2)/2 in [0, 2]: rho = g = 2 (attained at (pi,pi))
    thetas = range(0, pi; length = 201)
    lam(t1, t2) = 1 - (cos(t1) + cos(t2)) / 2
    @test maximum(lam(a, b) for a in thetas, b in thetas) ≈ 2
    smoothing(omega) = maximum(abs(1 - omega * lam(a, b)) for a in thetas, b in thetas if max(a, b) >= pi / 2)
    @printf("  smoothing factor mu(omega): mu(2/3) = %.4f, mu(4/5) = %.4f, mu(1) = %.4f\n", smoothing(2 / 3), smoothing(0.8), smoothing(1.0))
    @test smoothing(2 / 3) ≈ 2 / 3 atol = 1.0e-3
    @test smoothing(0.8) ≈ 0.6 atol = 1.0e-3
    # the smoother is energy-convergent for every omega in (0, 2/g) = (0, 1): the 4/3 margin is a safety factor
    @test maximum(abs(1 - 0.999 * lam(a, b)) for a in thetas, b in thetas if lam(a, b) > 0) < 1
    # and the high-frequency damping of the unsmoothed method at omega = 1 is worse than at 2/3 / 4/5
    @test smoothing(1.0) > smoothing(0.8)
end
