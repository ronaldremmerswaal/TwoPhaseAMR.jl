# Audit ch:solvers -- claim 2, necessity of the hypotheses in the code: the V-cycle is symmetric only if the number of pre- and
# post-sweeps agree (MultigridOptions accepts pre_sweeps != post_sweeps; the chapter states "the same number of sweeps ... precede and
# follow the coarse correction" as a design property of the DEFAULT 2+2). With 1+2 sweeps B is nonsymmetric and CG is not justified.
# Also: the number of equal sweeps does not matter for symmetry and definiteness (1+1, 2+2, 3+3).
# Source: docs/techref/chapters/11_solvers.tex:498-515; code src/solvers/multigrid/options.jl, src/solvers/multigrid/cycles.jl:46-78.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step02c_sweep_options.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

@testset "pre/post sweeps" begin
    op = PoissonOperator(solv_amr(8; levels = 2, block = 4))
    n = length(op.volumes)
    A = Matrix(op.matrix)
    for (pre, post) in ((1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (0, 2), (2, 0))
        preconditioner = TreeGeometricMultigridPreconditioner(op; multigrid = MultigridOptions(; pre_sweeps = pre, post_sweeps = post))
        B = vcycle_matrix(preconditioner, n)
        asym = maximum(abs, B - B') / maximum(abs, B)
        # mean-free spectrum of the symmetrised B A
        Z = nullspace(reshape(op.volumes, 1, :))
        L = cholesky(Symmetric(Z' * A * Z)).L
        M = L' * (Z' * ((B + B') / 2) * Z) * L
        lam = eigvals(Symmetric((M + M') / 2))
        @printf("  pre=%d post=%d: asymmetry %.1e, spectrum of sym(B) A on the mean-free space [%.3f, %.3f]\n", pre, post, asym, first(lam), last(lam))
        if pre == post
            @test asym <= 1.0e-12
            @test first(lam) > 0
        else
            @test asym > 1.0e-3
        end
    end
end
