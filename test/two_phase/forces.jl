using Test, TwoPhaseAMR

@testset "Conservative surface-tension method" begin
    @test CGFM() isa AbstractSurfaceTensionMethod
    @test isconcretetype(CGFM)
    @test fieldcount(CGFM) == 0
    @test SurfaceTension().method isa CGFM
    @test_throws MethodError CGFM(false)
    @test_throws MethodError CGFM(; conservative = false)

    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.37)
    geometry = GeometryField(mesh, (x, y) -> y - 0.12)
    @test_throws MethodError PoissonOperator(interface; formulation = CGFM())
    @test_throws MethodError PoissonOperator(
        interface, geometry; formulation = CGFM(),
    )
    operator = PoissonOperator(interface)
    @test_throws ArgumentError surface_tension!(
        FaceVelocity(mesh), interface, operator, EllipticWorkspace(operator);
        sigma = 1.0, formulation = nothing,
    )
    invalid = try
        surface_tension!(
            FaceVelocity(mesh), interface, operator, EllipticWorkspace(operator);
            sigma = -1.0, formulation = nothing,
        )
        nothing
    catch err
        err
    end
    @test invalid isa ArgumentError
    @test sprint(showerror, invalid) ==
        "ArgumentError: formulation must be CSF() or CGFM()"
end

@testset "Variable-density balanced CSF and CGFM projection" begin
    for kind in 1:3,dr in (1.0, 1000.0),formulation in (CSF(), CGFM())
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
        @test reinterpret(UInt64, op.face_coefficient) ==
            reinterpret(UInt64, inv.(mixture_density(f, (dr, 1.0))))
        u = FaceVelocity(mesh)
        surface_tension!(u, f, op, EllipticWorkspace(op); sigma = 1.0, formulation, curvature = 5.0)
        expected_jumps = map(mesh.faces) do face
            cm = TwoPhaseAMR._conditional_fraction(f, face.minus)
            cp = TwoPhaseAMR._conditional_fraction(f, face.plus)
            jump = formulation isa CSF ? cp - cm : (cp > 0.5) - (cm > 0.5)
            jump == 0 ? 0.0 : 1.0 * 5.0 * jump * face.area
        end
        @test reinterpret(UInt64, op.gradient_operator.pressure_jump) ==
            reinterpret(UInt64, expected_jumps)
        if formulation isa CGFM
            default = FaceVelocity(mesh)
            surface_tension!(
                default, f, op, EllipticWorkspace(op); sigma = 1.0,
                curvature = 5.0,
            )
            @test reinterpret(UInt64, default.interior) ==
                reinterpret(UInt64, u.interior)
        end
        potential = copy(f.fraction)
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(f.fraction, ref)
            TwoPhaseAMR._set!(potential, 5 * (formulation isa CSF ? c : c > 0.5), ref)
        end
        expected = FaceVelocity(mesh)
        weighted_gradient!(expected, potential, op, EllipticWorkspace(op))
        @test u.interior ≈ expected.interior atol = 5.0e-13
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :tree_geometric)
        pressure = CellField(mesh)
        project!(u, pressure, solver, 0.01, ProjectionWorkspace(solver); tolerance = 1.0e-11)
        @test maximum(abs, u.interior) < 1.0e-9
        @test all(>(0), mixture_density(f, (dr, 1.0)))
    end
end
@testset "Surface-tension strength is linear in sigma" begin
    mesh = BlockTreeMesh(
        block_size = (64, 64), origin = (-0.5, 0.0),
        extent = (1.0, 1.0), periodic = (false, false),
    )
    theta = deg2rad(55)
    radius = 0.3
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x, y + radius * cos(theta)) - radius;
        lipschitz = 1,
        contact_angle = (nothing, nothing, theta, nothing),
    )
    operator = PoissonOperator(interface; density = (10.0, 1.0))
    first = FaceVelocity(mesh)
    second = FaceVelocity(mesh)
    surface_tension!(
        first, interface, operator, EllipticWorkspace(operator);
        sigma = 0.25,
    )
    surface_tension!(
        second, interface, operator, EllipticWorkspace(operator);
        sigma = 0.75,
    )
    @test second.interior ≈ 3 .* first.interior rtol = 256eps()
end
@testset "Mimetic hydrostatic balance" begin
    for kind in 1:3, gravity in ((0.0, -1.0), (0.6, -0.8))
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2, periodic = (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        levelset = gravity == (0.0, -1.0) ?
            ((x, y) -> y - 0.43) :
            (
                (x, y) -> gravity[1] * (x - 0.5) +
                gravity[2] * (y - 0.5)
            )
        f = InterfaceField(mesh, levelset; lipschitz = 1)
        if kind == 3
            amr = AMRWorkspace(mesh, f);adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = PoissonOperator(f; density = (1000.0, 1.0))
        u = FaceVelocity(mesh)
        gravity!(
            u, f, op, EllipticWorkspace(op);
            density = (1000.0, 1.0), gravity
        )
        u.interior .*= 0.01
        solver = EllipticSolver(op; preconditioner = :tree_geometric)
        p = CellField(mesh)
        project!(
            u, p, solver, 0.01, ProjectionWorkspace(solver);
            tolerance = 512eps(),
        )
        @test maximum(abs, u.interior) <= 2048eps()
    end
end
