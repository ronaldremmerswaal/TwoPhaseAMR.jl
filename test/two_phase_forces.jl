using Test, TwoPhaseAMR
@testset "Variable-density balanced CSF and C-GFM projection" begin
    for kind in 1:3,dr in (1.0, 1000.0),formulation in (CSF(), GFM())
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2)
        if kind == 3
            amr = AMRWorkspace(mesh, f)
            adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = PoissonOperator(f; density = (dr, 1.0))
        u = FaceVelocity(mesh)
        surface_tension!(u, f, op, EllipticWorkspace(op); sigma = 1.0, formulation, curvature = 5.0)
        potential = copy(f.fraction)
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(f.fraction, ref)
            TwoPhaseAMR._set!(potential, 5 * (formulation isa CSF ? c : c > 0.5), ref)
        end
        expected = FaceVelocity(mesh)
        weighted_gradient!(expected, potential, op, EllipticWorkspace(op))
        @test u.interior ≈ expected.interior atol = 5.0e-13
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :geometric)
        pressure = CellField(mesh)
        project!(u, pressure, solver, 0.01, ProjectionWorkspace(solver); rtol = 1.0e-11, atol = 1.0e-12)
        @test maximum(abs, u.interior) < 1.0e-9
        @test all(>(0), mixture_density(f, (dr, 1.0)))
    end
end
@testset "Mimetic hydrostatic balance" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2, periodic = (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> y - 0.43)
        if kind == 3
            amr = AMRWorkspace(mesh, f);adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = PoissonOperator(f; density = (1000.0, 1.0))
        u = FaceVelocity(mesh)
        gravity!(u, f, op, EllipticWorkspace(op); density = (1000.0, 1.0), gravity = (0.0, -1.0))
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :geometric)
        p = CellField(mesh)
        project!(u, p, solver, 0.01, ProjectionWorkspace(solver); rtol = 1.0e-11, atol = 1.0e-12)
        @test maximum(abs, u.interior) < 1.0e-9
    end
end
