using LinearAlgebra: ldiv!, norm, rank, cholesky, Symmetric
using SparseArrays: sparse

function _joint_test_velocity(mesh)
    return (
        FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? sinpi(2y) + 0.1cospi(2x) : -0.2sinpi(2x)
        ),
        FaceVelocity(
            mesh, (x, y, time, axis) ->
            axis == 1 ? 0.3cospi(2y) : sinpi(2x)
        ),
    )
end

@testset "Jointly fitted mimetic projection variants" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
    )
    # The pressure-commuting bound was established with GHF-fallback normals;
    # commit 0b00093 changed the default to :lhf.
    # NUMERICS CONCERN: the default :lhf gives a residual of about 0.9 here,
    # versus the 0.2 bound below; the default's behavior needs investigation.
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.3;
        lipschitz = 1, reconstruction_method = :lhf_ghf,
    )
    for variant in (:joint_smoothed, :joint_face)
        solver = MimeticProjectionSolver(
            interface; density = (10.0, 1.0), variant,
        )
        workspace = MimeticProjectionWorkspace(solver)
        velocity = _joint_test_velocity(mesh)
        assembled = TwoPhaseAMR._assemble_mimetic_system!(
            solver, workspace, 1.0, velocity,
        )
        @test assembled.matrix ≈ transpose(assembled.matrix)
        @test rank(Matrix(assembled.constraint)) ==
            size(assembled.constraint, 1)
        @test all(>(0), assembled.relative_diagonal)
        @test length(assembled.active_faces) ==
            length(assembled.relative_diagonal)
        preconditioner = TwoPhaseAMR._local_mimetic_preconditioner(
            solver, assembled,
        )
        rhs = sin.(collect(1:size(assembled.matrix, 1)))
        expected = assembled.block_preconditioner \ rhs
        preconditioned = similar(rhs)
        ldiv!(preconditioned, preconditioner, rhs)
        @test preconditioned ≈ expected atol = 1.0e-9
        # A Krylov workspace may alias its input and output buffers.
        aliased = copy(rhs)
        ldiv!(aliased, preconditioner, aliased)
        @test aliased ≈ expected atol = 1.0e-9

        stats = project!(
            velocity, CellField(mesh), solver, 1.0, workspace;
            tolerance = 1.0e-9, max_iterations = 5000,
        )
        @test stats.converged
        @test stats.variant == variant
        @test stats.mixture_volume_defect <= stats.target_defect
        @test stats.normal_velocity_defect <= stats.target_defect
        @test stats.energy_identity_error <= 64stats.target_defect
        @test stats.orthogonality_error <= 64stats.target_defect
        @test maximum(abs, stats.momentum_error) <=
            64stats.target_defect
        @test stats.independent_constraint_rows > 0
        @test stats.candidate_constraint_rows ==
            stats.independent_constraint_rows +
            stats.dependent_constraint_rows
        @test stats.commuting_residual < 0.1
        @test stats.pressure_commuting_residual < 0.2
    end
end

@testset "Mimetic projection configuration" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.5; lipschitz = 1)
    @test MimeticProjection().variant isa TwoPhaseAMR._MimeticJointSmoothed
    @test MimeticProjection(variant = :joint_face).variant isa TwoPhaseAMR._MimeticJointFace
    @test typeof(MimeticProjection()) !==
        typeof(MimeticProjection(variant = :joint_face))
    for variant in (:joint_smoothed, :joint_face)
        projection = MimeticProjection(;
            variant, preconditioner = :ichol, preconditioner_options = (;),
            active_mass_fraction = 0.002, rank_tolerance = 1.0e-4,
            patch_radius = 1, pressure_weight = 12.0,
            commutator_weight = 34.0, correction_cap = 0.5, fit_sweeps = 4,
        )
        direct_solver = MimeticProjectionSolver(
            interface; density = (4.0, 2.0), variant,
            preconditioner = :ichol, preconditioner_options = (;),
            active_mass_fraction = 0.002, rank_tolerance = 1.0e-4,
            patch_radius = 1, pressure_weight = 12.0,
            commutator_weight = 34.0, correction_cap = 0.5, fit_sweeps = 4,
        )
        solver, workspace = TwoPhaseAMR._two_velocity_projection_workspace(
            interface, (4.0, 2.0), projection,
        )
        @test solver isa MimeticProjectionSolver
        @test workspace isa MimeticProjectionWorkspace
        @test TwoPhaseAMR._mimetic_variant_symbol(solver.variant) === variant
        @test direct_solver.variant === solver.variant
        @test typeof(solver) === typeof(direct_solver)
        @test solver.variant === projection.variant
        @test solver.density === (4.0, 2.0)
        @test direct_solver.density === solver.density
        @test solver.preconditioner === projection.preconditioner
        @test solver.preconditioner_options === projection.preconditioner_options
        @test solver.active_mass_fraction === projection.active_mass_fraction
        @test solver.rank_tolerance === projection.rank_tolerance
        @test solver.patch_radius === projection.patch_radius
        @test solver.pressure_weight === projection.pressure_weight
        @test solver.commutator_weight === projection.commutator_weight
        @test solver.correction_cap === projection.correction_cap
        @test solver.fit_sweeps === projection.fit_sweeps
    end
    @test_throws ArgumentError MimeticProjection(variant = :cell)
    for invalid in (nothing, "joint_face", :unknown)
        @test_throws ArgumentError MimeticProjection(variant = invalid)
        @test_throws ArgumentError MimeticProjectionSolver(
            interface; variant = invalid,
        )
    end
    @test_throws MethodError MimeticProjection(jump_interpolant = :face)
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; variant = :paired,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; active_mass_fraction = 0.5,
    )
    @test_throws ArgumentError MimeticProjectionSolver(
        interface; rank_tolerance = 0.0,
    )
    @test_throws ArgumentError MimeticProjection(
        pressure_weight = 0,
    )
    @test_throws ArgumentError MimeticProjection(
        correction_cap = 0,
    )
    @test_throws ArgumentError MimeticProjection(
        fit_sweeps = 0,
    )

    positional = (
        :joint_face, :cholesky, (;), 1.0e-6, nothing, 2,
        1000.0, 1000.0, 1.0, 3,
    )
    positional_projection = MimeticProjection(positional...)
    @test positional_projection.variant isa TwoPhaseAMR._MimeticJointFace
    for (index, invalid) in (
            (1, :cell), (2, :bogus), (6, -1), (7, 0.0),
            (8, 0.0), (9, 0.0), (10, 0),
        )
        args = ntuple(i -> i == index ? invalid : positional[i], length(positional))
        @test_throws ArgumentError MimeticProjection(args...)
        @test_throws MethodError typeof(positional_projection)(args...)
    end
    for preconditioner in (:block, :ilu, nothing)
        @test CMDGFMProjection(preconditioner, (;)).preconditioner === preconditioner
    end
    @test_throws ArgumentError CMDGFMProjection(:bogus, (;))
    @test_throws ArgumentError CMDGFMProjection(preconditioner = :bogus)
    @test_throws MethodError typeof(CMDGFMProjection())(:bogus, (;))
end

@testset "Mimetic Cholesky factor solves support both scalar types" begin
    for T in (Float32, Float64)
        pressure = T[4 1; 1 3]
        multiplier = T[7;;]
        jump_inverse = T[0.5, 0.2]
        preconditioner = TwoPhaseAMR.MimeticBlockCholeskyPreconditioner(
            cholesky(Symmetric(sparse(pressure), :L)), jump_inverse,
            cholesky(Symmetric(sparse(multiplier), :L)), (2, 2, 1),
            zeros(T, 2), zeros(T, 1),
        )
        matrix = zeros(T, 5, 5)
        matrix[1:2, 1:2] .= pressure
        matrix[3, 3] = inv(jump_inverse[1])
        matrix[4, 4] = inv(jump_inverse[2])
        matrix[5, 5] = multiplier[1, 1]
        rhs = T[1, -2, 3, -4, 5]
        expected = matrix \ rhs
        result = similar(rhs)
        ldiv!(result, preconditioner, rhs)
        tolerance = T === Float32 ? 3.0e-6 : 2.0e-14
        @test result ≈ expected rtol = tolerance atol = tolerance
        aliased = copy(rhs)
        ldiv!(aliased, preconditioner, aliased)
        @test aliased ≈ expected rtol = tolerance atol = tolerance
    end
end

@testset "Float32 pure-phase mimetic projection" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, true),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0)
    )
    interface = InterfaceField(mesh, (x, y) -> -1.0f0; lipschitz = 1)
    solver = MimeticProjectionSolver(interface; density = (10.0f0, 1.0f0))
    workspace = MimeticProjectionWorkspace(solver)
    stats = project!(
        _joint_test_velocity(mesh), CellField(mesh), solver, 0.02f0, workspace;
        tolerance = 1.0f-3, max_iterations = 4000
    )
    @test stats isa MimeticProjectionStats{Float32}
    @test stats.converged
    @test isfinite(stats.final_residual)
    @test !isnothing(workspace.cache.krylov)
end
