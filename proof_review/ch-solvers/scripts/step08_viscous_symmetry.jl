# Audit ch:solvers -- side question: "the composite-mesh viscous system is nonsymmetric (BiCGSTAB)". Is it, and is it expected after the
# diffusion fixes of #79/#87 (cut-cell/embedded rows without the corrected band)?
# Source: docs/techref/chapters/11_solvers.tex:652-662 (default Krylov method of the shifted one-phase systems), 09_viscosity.tex:117-135,
#   165-172; code src/navier_stokes/implicit_diffusion.jl:58-72, 481-505 (Krylov selection), src/operators/elliptic.jl:1133-1148.
# For every fixture: relative asymmetry of the solved matrices V (I - s L_k) (k = x, y), the stiffness K_k (symmetric by
# construction), the Krylov method the solver selects, and the reason (issymmetric of the matrix, equal volumes).
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step08_viscous_symmetry.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

asymmetry(M) = maximum(abs, M - M') / maximum(abs, M)

function symmetry_row(name, diffusion; s = 1.0e-3)
    operator = StaggeredEllipticOperator(diffusion; diffusion = s)
    solver = StaggeredEllipticSolver(operator)
    rows = map((:x, :y)) do k
        comp = getfield(operator, k).matrix
        dcomp = getfield(diffusion, k)
        V = dcomp.volumes
        stiffness_asym = asymmetry(dcomp.stiffness)
        # V L: conservative rows give -K (symmetric); corrected rows break the symmetry
        VL = Diagonal(V) * dcomp.laplacian
        (;
            matrix = asymmetry(comp), VL = asymmetry(VL), K = stiffness_asym,
            equal_volumes = AMR._staggered_shift_is_symmetric(dcomp), exact_issym = issymmetric(comp),
            distinct_volumes = length(unique(round.(V; sigdigits = 12))),
        )
    end
    println(
        @sprintf(
            "  %-34s x: asym V(I-sL)=%.1e  V*L=%.1e K=%.1e  equal-vol-test=%-5s issymmetric=%-5s | y: asym=%.1e | Krylov: %s",
            name, rows[1].matrix, rows[1].VL, rows[1].K, rows[1].equal_volumes, rows[1].exact_issym, rows[2].matrix, nameof(solver.krylov_factory)
        )
    )
    return (; rows, krylov = solver.krylov_factory)
end

@testset "one-phase viscous matrix: symmetry and method" begin
    println("== no solid ==")
    uniform = symmetry_row("uniform 32^2, free-slip", StaggeredDiffusionOperator(solv_uniform(32); viscosity = 1.0, boundary = FreeSlip()))
    @test uniform.krylov === Krylov.CgWorkspace
    @test all(r.matrix <= 1.0e-14 for r in uniform.rows)
    periodic = symmetry_row("uniform 32^2, periodic", StaggeredDiffusionOperator(solv_uniform(32; periodic = (true, true)); viscosity = 1.0))
    @test periodic.krylov === Krylov.CgWorkspace
    composite = symmetry_row("composite base 16, 2 levels", StaggeredDiffusionOperator(solv_amr(16); viscosity = 1.0, boundary = FreeSlip()))
    @test composite.krylov === Krylov.BicgstabWorkspace
    @test any(r.matrix > 1.0e-4 for r in composite.rows)           # the corrected rows are nonsymmetric (matrix, not roundoff)
    @test all(r.K <= 1.0e-14 for r in composite.rows)              # the conservative stiffness itself is symmetric

    println("== with a solid (embedded constructor: no corrected band) ==")
    geometry = GeometryField(solv_uniform(32), solv_band_solid; lipschitz = 1)
    cut = symmetry_row("cut cells, uniform 32^2", StaggeredDiffusionOperator(geometry; viscosity = 1.0))
    geometry = GeometryField(solv_amr(16), solv_band_solid; lipschitz = 1)
    cutamr = symmetry_row("cut cells, composite base 16", StaggeredDiffusionOperator(geometry; viscosity = 1.0))
    for r in (cut, cutamr)
        # V*L = -K exactly in every row (no band): the matrix V (I - s L) is symmetric up to the roundoff of the product ...
        @test all(row.VL <= 1.0e-12 for row in r.rows)
        @test all(row.matrix <= 1.0e-12 for row in r.rows)
        # ... but the solver still picks BiCGSTAB: the criterion compares volumes entrywise / issymmetric exactly
        @test r.krylov === Krylov.BicgstabWorkspace
        @test !all(row.equal_volumes for row in r.rows)
    end
end
