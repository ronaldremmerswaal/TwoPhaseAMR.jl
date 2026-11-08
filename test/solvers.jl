isdefined(@__MODULE__, :test_cell_values) || include("test_utils.jl")

import LinearAlgebra: ldiv!
using LinearAlgebra: Diagonal, lu
using SparseArrays: sparse, spzeros

struct _InPlaceOnlyPreconditioner end

function ldiv!(::_InPlaceOnlyPreconditioner, values::AbstractVector)
    values .*= 0.5
    return values
end

@testset "external preconditioners support in-place-only ldiv!" begin
    wrapped = TwoPhaseAMR.ExternalPreconditioner(
        :in_place_only, _InPlaceOnlyPreconditioner(), zeros(3, 3),
    )
    right_hand_side = [2.0, 4.0, 6.0]
    destination = zeros(3)
    @test ldiv!(destination, wrapped, right_hand_side) === destination
    @test destination == [1.0, 2.0, 3.0]
    @test right_hand_side == [2.0, 4.0, 6.0]
end

@testset "Coarse sparse LU symbolic reuse" begin
    for T in (Float32, Float64)
        diagonal = sparse([1, 2], [1, 2], T[2, 3], 2, 2)
        changed_values = sparse([1, 2], [1, 2], T[4, 5], 2, 2)
        upper = sparse([1, 1, 2], [1, 2, 2], T[4, 1, 5], 2, 2)
        lower = sparse([1, 2, 2], [1, 1, 2], T[4, 1, 5], 2, 2)
        rhs = [0.5, -0.75]
        factorization = lu(diagonal)
        symbolic = factorization.symbolic
        reused = TwoPhaseAMR._refactorize_coarse!(factorization, changed_values)
        @test reused === factorization
        @test reused.symbolic === symbolic
        @test reused \ rhs ≈ Matrix{Float64}(changed_values) \ rhs

        # Different nnz, then the same nnz but a different row/column pattern.
        fallback = TwoPhaseAMR._refactorize_coarse!(reused, upper)
        @test fallback !== reused
        @test fallback \ rhs ≈ Matrix{Float64}(upper) \ rhs
        different_rows = TwoPhaseAMR._refactorize_coarse!(fallback, lower)
        @test different_rows !== fallback
        @test different_rows \ rhs ≈ Matrix{Float64}(lower) \ rhs
        larger = sparse([1, 2, 3], [1, 2, 3], T[2, 3, 4], 3, 3)
        resized = TwoPhaseAMR._refactorize_coarse!(different_rows, larger)
        @test resized !== different_rows
        @test resized \ ones(3) ≈ Matrix{Float64}(larger) \ ones(3)
        pattern_a = sparse([1, 2, 2, 3], [1, 1, 2, 3], T[4, 1, 5, 6], 3, 3)
        pattern_b = sparse([1, 3, 2, 3], [1, 1, 2, 3], T[4, 1, 5, 6], 3, 3)
        @test pattern_a.colptr == pattern_b.colptr
        first_pattern = TwoPhaseAMR._refactorize_coarse!(resized, pattern_a)
        second_pattern = TwoPhaseAMR._refactorize_coarse!(first_pattern, pattern_b)
        @test second_pattern !== first_pattern
        @test second_pattern \ ones(3) ≈ Matrix{Float64}(pattern_b) \ ones(3)

        # Cover both the unaugmented shifted case and the actual singular
        # pressure path, including changing which volume entries are zero.
        mesh = BlockTreeMesh(block_size = (4, 4))
        shifted = EllipticOperator(mesh; T, reaction = 1)
        shifted_hierarchy = TreeGeometricMultigridPreconditioner(
            shifted; coarse_size = 1, max_levels = 1,
        )
        old_shifted = shifted_hierarchy.coarse_factorization
        old_shifted_symbolic = old_shifted.symbolic
        coefficient = T[1 + 0.05sin(index) for index in eachindex(mesh.faces)]
        TwoPhaseAMR._replace_face_coefficient(shifted, coefficient)
        TwoPhaseAMR._refresh_geometric_multigrid!(shifted_hierarchy, shifted)
        @test shifted_hierarchy.coarse_factorization === old_shifted
        @test old_shifted.symbolic === old_shifted_symbolic
        shifted_rhs = Float64[sin(index) for index in 1:ncells(mesh)]
        @test shifted.matrix * (old_shifted \ shifted_rhs) ≈ shifted_rhs rtol = 1.0e-10

        pressure = EllipticOperator(mesh; T, inactive_condition = :decoupled)
        hierarchy = TreeGeometricMultigridPreconditioner(
            pressure; coarse_size = 1, max_levels = 1,
        )
        old_factorization = hierarchy.coarse_factorization
        old_symbolic = old_factorization.symbolic
        TwoPhaseAMR._replace_face_coefficient(pressure, coefficient)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, pressure)
        @test hierarchy.coarse_factorization === old_factorization
        @test old_factorization.symbolic === old_symbolic

        active = trues(ncells(mesh))
        active[1] = false
        volumes = TwoPhaseAMR._cell_volumes(mesh, T)
        volumes[1] = zero(T)
        covered = EllipticOperator(
            mesh; T, face_coefficient = coefficient, active_cells = active, volumes,
            inactive_condition = :decoupled,
            gradient_operator = pressure.gradient_operator,
        )
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, covered)
        replacement = hierarchy.coarse_factorization
        @test replacement !== old_factorization
        coarse = only(hierarchy.levels)
        augmented = sparse([
            coarse.matrix sparse(coarse.volumes);
            sparse(transpose(coarse.volumes)) spzeros(T, 1, 1)
        ])
        augmented_rhs = Float64[cos(index) for index in 1:(ncells(mesh) + 1)]
        @test augmented * (replacement \ augmented_rhs) ≈ augmented_rhs rtol = 1.0e-10
        symbolic = replacement.symbolic
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, covered)
        @test hierarchy.coarse_factorization === replacement
        @test replacement.symbolic === symbolic
    end
end

import IncompleteLU
using Krylov

_optional_linear_solver_dependencies = all(
    package -> !isnothing(Base.find_package(package)),
    ("AlgebraicMultigrid", "LimitedLDLFactorizations"),
)

if _optional_linear_solver_dependencies
    import AlgebraicMultigrid
    import LimitedLDLFactorizations

    @testset "optional linear-solver extensions load on demand" begin
        @test !isnothing(
            Base.get_extension(
                TwoPhaseAMR, :TwoPhaseAMRAlgebraicMultigridExt,
            )
        )
        @test !isnothing(
            Base.get_extension(
                TwoPhaseAMR, :TwoPhaseAMRLimitedLDLFactorizationsExt,
            )
        )
    end

    @testset "optional public preconditioners" begin
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 1,
            refine = (key, origin, extent) -> true,
        )
        pressure_operator = PoissonOperator(mesh)
        pressure_rhs = CellField(mesh)
        initialize!(
            pressure_rhs, (x, y) ->
            8pi^2 * sinpi(2x) * cospi(2y)
        )

        pressure_cases = (
            (
                name = :algebraic_multigrid, krylov = Krylov.CgWorkspace,
                preconditioner = :amg, options = (;),
            ),
            (
                name = :incomplete_cholesky, krylov = Krylov.CgWorkspace,
                preconditioner = :ichol,
                options = (; memory = 4, shift = 1.0e-8),
            ),
            # ILU is not guaranteed symmetric positive definite, so pair it with
            # BiCGSTAB instead of violating CG's preconditioner requirements.
            (
                name = :ilu, krylov = Krylov.BicgstabWorkspace,
                preconditioner = :ilu,
                options = (; drop_tolerance = 1.0e-3),
            ),
        )
        for case in pressure_cases
            solver = EllipticSolver(
                pressure_operator; krylov = case.krylov,
                preconditioner = case.preconditioner,
                preconditioner_options = case.options
            )
            pressure = CellField(mesh)
            stats = solve!(
                pressure, solver, pressure_rhs; rtol = 1.0e-9, max_cycles = 200
            )
            @test stats.converged
            @test stats.preconditioner === case.name
            @test size(solver.preconditioner) == size(pressure_operator.matrix)
            @test eltype(solver.preconditioner) === Float64
            @test all(isfinite, test_cell_values(pressure))
        end

        diffusion = StaggeredDiffusionOperator(
            mesh; viscosity = 0.1, boundary = FreeSlip()
        )
        diffusion_operator = StaggeredEllipticOperator(
            diffusion; diffusion = 0.05
        )
        velocity_rhs = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)
        )
        for (name, specification, options) in (
                (:algebraic_multigrid, :amg, (;)),
                (
                    :incomplete_cholesky, :ichol,
                    (; memory = 4, shift = 1.0e-8),
                ),
                (:ilu, :ilu, (; drop_tolerance = 1.0e-3)),
            )
            solver = StaggeredEllipticSolver(
                diffusion_operator; preconditioner = specification,
                preconditioner_options = options
            )
            velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
            stats = solve!(
                velocity, solver, velocity_rhs;
                tolerance = 1.0e-9, max_cycles = 200
            )
            @test stats.converged
            @test stats.x.preconditioner === name
            @test stats.y.preconditioner === name
            @test all(isfinite, velocity.interior)
        end

        navier_mesh = BlockTreeMesh(block_size = (4, 4))
        navier_solver = IncompressibleNavierStokesSolver(
            navier_mesh; density = 1.0, dynamic_viscosity = 0.01,
            boundary = FreeSlip(),
            pressure_preconditioner = :amg,
            diffusion_preconditioner = :ilu,
            diffusion_preconditioner_options = (; drop_tolerance = 1.0e-3),
            coarse_size = 8
        )
        navier_workspace = NavierStokesWorkspace(navier_solver)
        velocity = FaceVelocity(navier_mesh, (x, y, time, axis) -> 0.0)
        pressure = CellField(navier_mesh)
        stats = step!(
            velocity, pressure, navier_solver, 0.1, navier_workspace;
            force = (x, y, time, axis) ->
            axis == 1 ? 0.1sinpi(2y) : 0.0,
            tolerance = 1.0e-9, max_cycles = 100
        )
        @test stats.converged
        @test stats.preconditioner === :algebraic_multigrid
        @test navier_workspace.elliptic.x.stats.preconditioner === :ilu
    end
end

function _krylov_resize_test_system(n)
    diagonal = [1.0 + 0.3 * mod(index - 1, 8) for index in 1:n]
    right_hand_side = [1.0 + sin(0.7 * index) for index in 1:n]
    return Diagonal(diagonal), right_hand_side
end

function _test_resized_krylov_solve!(workspace, n; preconditioned)
    matrix, right_hand_side = _krylov_resize_test_system(n)
    preconditioner = preconditioned ? Diagonal(fill(1.3, n)) : nothing
    options = preconditioned && !(workspace isa Krylov.CgWorkspace) ?
        (; M = preconditioner) : (;)
    TwoPhaseAMR._run_krylov!(
        workspace, matrix, right_hand_side, zeros(n), preconditioner,
        1.0e-10, ones(n), 100, options,
    )
    # This tolerance is looser than the Krylov recurrence threshold, allowing
    # for its roundoff and for the left-preconditioned BiCGSTAB callback.
    @test maximum(abs, matrix * Krylov.solution(workspace) - right_hand_side) <
        1.0e-8
    return nothing
end

@testset "Krylov workspace resize and re-solve" begin
    cases = (
        (;
            method = :cg, factory = Krylov.CgWorkspace,
            required = (:x, :r, :p, :Ap), optional = (:Δx, :npc_dir, :z),
            bases = (),
        ),
        (;
            method = :bicgstab, factory = Krylov.BicgstabWorkspace,
            required = (:x, :r, :p, :v, :s, :qd),
            optional = (:Δx, :yz, :t), bases = (),
        ),
        (;
            method = :gmres, factory = Krylov.GmresWorkspace,
            required = (:x, :w), optional = (:Δx, :p, :q), bases = (:V,),
        ),
        (;
            method = :fgmres, factory = Krylov.FgmresWorkspace,
            required = (:x, :w), optional = (:Δx, :q), bases = (:V, :Z),
        ),
    )
    for case in cases
        @testset "$(case.method)" begin
            matrix, right_hand_side = _krylov_resize_test_system(12)
            workspace = isempty(case.bases) ?
                case.factory(matrix, right_hand_side) :
                case.factory(matrix, right_hand_side; memory = 2)
            @test TwoPhaseAMR._krylov_method(workspace) === case.method
            fields = (case.required..., case.optional...)
            initial_vectors = [getproperty(workspace, field) for field in fields]
            @test all(field -> isempty(getproperty(workspace, field)), case.optional)
            @test TwoPhaseAMR._resize_krylov_workspace!(workspace, 16) === workspace
            @test (workspace.m, workspace.n) == (16, 16)
            @test all(
                getproperty(workspace, field) === vector
                    for (field, vector) in zip(fields, initial_vectors)
            )
            @test all(field -> length(getproperty(workspace, field)) == 16, case.required)
            @test all(field -> isempty(getproperty(workspace, field)), case.optional)
            @test all(
                length(vector) == 16
                    for field in case.bases for vector in getproperty(workspace, field)
            )

            _test_resized_krylov_solve!(workspace, 16; preconditioned = true)
            if !isempty(case.bases)
                # With restart=false Krylov 0.10 appends basis vectors on demand.
                @test length(workspace.V) > 2
            end
            # Include optional storage not required by this particular solve,
            # e.g. CG's negative-curvature direction, in the next resize.
            for field in case.optional
                vector = getproperty(workspace, field)
                isempty(vector) && resize!(vector, 16)
            end
            vectors = [getproperty(workspace, field) for field in fields]
            basis_groups = [getproperty(workspace, field) for field in case.bases]
            basis_vectors = [copy(group) for group in basis_groups]
            auxiliary_fields = isempty(case.bases) ? () : (:c, :s, :z, :R)
            auxiliary_vectors = [getproperty(workspace, field) for field in auxiliary_fields]
            auxiliary_values = copy.(auxiliary_vectors)
            stats = workspace.stats
            workspace.warm_start = true
            isempty(case.bases) || (workspace.inner_iter = 7)
            @test TwoPhaseAMR._resize_krylov_workspace!(workspace, 24) === workspace
            @test (workspace.m, workspace.n) == (24, 24)
            @test !workspace.warm_start
            @test workspace.stats === stats
            @test all(
                getproperty(workspace, field) === vector && length(vector) == 24
                    for (field, vector) in zip(fields, vectors)
            )
            @test all(
                getproperty(workspace, field) === group &&
                    length(group) == length(saved) &&
                    all(group[i] === saved[i] && length(group[i]) == 24 for i in eachindex(group))
                    for (field, group, saved) in zip(case.bases, basis_groups, basis_vectors)
            )
            @test all(
                getproperty(workspace, field) === vector && vector == values
                    for (field, vector, values) in
                    zip(auxiliary_fields, auxiliary_vectors, auxiliary_values)
            )
            if !isempty(case.bases)
                @test workspace.inner_iter == 0
            end

            _test_resized_krylov_solve!(workspace, 24; preconditioned = false)
            shrink_to = 8
            if !isempty(case.bases)
                # The current, possibly grown V determines the failure bound;
                # rejected resizes must not mutate any part of the workspace.
                shrink_to = length(workspace.V)
                @test shrink_to < 24
                vectors = [getproperty(workspace, field) for field in fields]
                basis_groups = [getproperty(workspace, field) for field in case.bases]
                basis_vectors = [copy(group) for group in basis_groups]
                auxiliary_values = copy.(auxiliary_vectors)
                old_solution = copy(workspace.x)
                workspace.warm_start = true
                workspace.inner_iter = 7
                @test isnothing(
                    TwoPhaseAMR._resize_krylov_workspace!(
                        workspace, length(workspace.V) - 1,
                    )
                )
                @test (workspace.m, workspace.n) == (24, 24)
                @test workspace.warm_start
                @test workspace.inner_iter == 7
                @test workspace.x == old_solution
                @test all(
                    getproperty(workspace, field) === vector && length(vector) == 24
                        for (field, vector) in zip(fields, vectors)
                )
                @test all(
                    getproperty(workspace, field) === group &&
                        length(group) == length(saved) &&
                        all(group[i] === saved[i] && length(group[i]) == 24 for i in eachindex(group))
                        for (field, group, saved) in zip(case.bases, basis_groups, basis_vectors)
                )
                @test all(
                    getproperty(workspace, field) === vector && vector == values
                        for (field, vector, values) in
                        zip(auxiliary_fields, auxiliary_vectors, auxiliary_values)
                )
                smaller = shrink_to - 1
                small_matrix, small_rhs = _krylov_resize_test_system(smaller)
                replacement = TwoPhaseAMR._construct_krylov(
                    (matrix, rhs) -> case.factory(matrix, rhs; memory = 2),
                    small_matrix, small_rhs,
                )
                @test replacement !== workspace
                @test length(replacement.V) == min(2, smaller)
                _test_resized_krylov_solve!(
                    replacement, smaller; preconditioned = true,
                )
            end
            @test TwoPhaseAMR._resize_krylov_workspace!(workspace, shrink_to) === workspace
            @test (workspace.m, workspace.n) == (shrink_to, shrink_to)
            @test !workspace.warm_start
            @test all(field -> length(getproperty(workspace, field)) == shrink_to, fields)
            @test all(
                length(vector) == shrink_to
                    for field in case.bases for vector in getproperty(workspace, field)
            )
            if !isempty(case.bases)
                @test workspace.inner_iter == 0
            end
            _test_resized_krylov_solve!(workspace, shrink_to; preconditioned = true)
        end
    end

    matrix, right_hand_side = _krylov_resize_test_system(12)
    unsupported = Krylov.MinresWorkspace(matrix, right_hand_side)
    @test TwoPhaseAMR._krylov_method(unsupported) === :MinresWorkspace
    @test isnothing(TwoPhaseAMR._resize_krylov_workspace!(unsupported, 16))
    @test (unsupported.m, unsupported.n) == (12, 12)
    @test_throws ArgumentError TwoPhaseAMR._run_krylov!(
        unsupported, matrix, right_hand_side, zeros(12), nothing,
        1.0e-10, ones(12), 100, (;),
    )
end

@testset "shared Krylov residual scales and copy order" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (true, true))
    diffusion = StaggeredDiffusionOperator(
        mesh; viscosity = 0.1, boundary = FreeSlip(),
    )
    component = StaggeredEllipticOperator(
        diffusion; reaction = 1, diffusion = 0.1,
    ).x
    volumes = component.volumes
    workspace = Krylov.CgWorkspace(component.matrix, zeros(length(volumes)))
    solver = EllipticSolver(component, workspace, nothing)
    fill!(solver.solution, 0.9)
    solver.right_hand_side .= 4.0 .* volumes
    TwoPhaseAMR._true_residual!(solver)
    initial_l2, initial_linf =
        TwoPhaseAMR._integrated_residual_norms(solver.residual, volumes)
    floor = TwoPhaseAMR._roundoff_residual_floor!(
        component.matrix, solver.solution, solver.right_hand_side,
        volumes, solver.product,
    )
    initial_scale = max(initial_linf, eps(Float64))
    rhs_scale = TwoPhaseAMR._strong_maximum(solver.right_hand_side, volumes)
    @test initial_scale != rhs_scale

    for relative_to_rhs in (false, true)
        copied = similar(solver.solution)
        prepared = Ref(false)
        prepare! = () -> begin
            prepared[] = true
            return 0.25
        end
        solver.stats.cycles = -1
        atol = relative_to_rhs ? 1000.0 : 0.01
        stats = TwoPhaseAMR._solve_krylov_with_stats!(
            prepare!, solver, component.matrix, volumes, 0.1, 0;
            atol, relative_to_rhs,
        ) do values
            @test prepared[]
            # Copying occurs after the final norm and floor, before stats.
            @test solver.stats.cycles == -1
            copyto!(copied, values)
        end
        scale = relative_to_rhs ? rhs_scale : initial_scale
        target = relative_to_rhs ?
            max(0.1 * scale, floor) : max(atol, 0.1 * scale, floor)
        @test copied == solver.solution
        @test stats === solver.stats
        @test stats.cycles == 0
        @test stats.initial_l2 === initial_l2
        @test stats.initial_linf === initial_linf
        @test stats.final_l2 === initial_l2
        @test stats.final_linf === initial_linf
        @test stats.residual_scale === scale
        @test stats.target_linf === target
        @test stats.converged == (initial_linf <= target)
        @test stats.removed_rhs_mean === 0.25
        @test stats.method === :cg
        @test stats.preconditioner === :none
    end
end

@testset "Krylov trait dispatch preserves direct solver results" begin
    cases = (
        (;
            factory = Krylov.CgWorkspace, solve = Krylov.cg!,
            placement = :M, strong_callback = true,
        ),
        (;
            factory = Krylov.BicgstabWorkspace, solve = Krylov.bicgstab!,
            placement = :N, strong_callback = true,
        ),
        (;
            factory = Krylov.GmresWorkspace, solve = Krylov.gmres!,
            placement = :N, strong_callback = false,
        ),
        (;
            factory = Krylov.FgmresWorkspace, solve = Krylov.fgmres!,
            placement = :N, strong_callback = false,
        ),
    )
    matrix, right_hand_side = _krylov_resize_test_system(12)
    volumes = [0.5 + 0.01 * index for index in 1:12]
    target = 1.0e-10
    # The driver must override the tolerances and iteration limit for every
    # method, and override this callback only for CG and BiCGSTAB.
    options = (; atol = 100.0, rtol = 0.5, itmax = 1, callback = _ -> true)
    for case in cases, preconditioned in (false, true)
        actual = case.factory(matrix, right_hand_side)
        reference = case.factory(matrix, right_hand_side)
        preconditioner = preconditioned ? Diagonal(fill(1.3, 12)) : nothing
        TwoPhaseAMR._run_krylov!(
            actual, matrix, right_hand_side, zeros(12), preconditioner,
            target, volumes, 100, options,
        )
        settings = if case.strong_callback
            callback = TwoPhaseAMR.StrongResidualCallback(target, volumes)
            merge(options, (; atol = 0.0, rtol = 0.0, itmax = 100, callback))
        else
            atol = target * minimum(volumes)
            merge(options, (; atol, rtol = 0.0, itmax = 100))
        end
        if preconditioned
            settings = case.placement === :M ?
                merge(settings, (; M = preconditioner, ldiv = true)) :
                merge(settings, (; N = preconditioner, ldiv = true))
        end
        case.solve(reference, matrix, right_hand_side, zeros(12); settings...)
        @test all(isfinite, Krylov.solution(actual))
        @test isequal(Krylov.solution(actual), Krylov.solution(reference))
        @test Krylov.iteration_count(actual) == Krylov.iteration_count(reference)
        if case.strong_callback
            @test Krylov.iteration_count(actual) > 1
        else
            @test Krylov.iteration_count(actual) == 1
        end
    end
end

@testset "cut-cell root coarsening and AMR mesh reuse" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        periodic = (false, false),
    )
    levelset(x, y) = hypot(x - 0.52, y - 0.46) - 0.19
    geometry = GeometryField(mesh, levelset; lipschitz = 1)
    old = TreeGeometricMultigridPreconditioner(
        PoissonOperator(geometry); geometry, coarse_size = 16,
    )
    @test [op.mesh.base_mesh for op in old.operators] ==
        [(16, 16), (8, 8), (4, 4)]
    old_coarse_mesh = last(old.operators).mesh

    adaptation = AMRWorkspace(mesh)
    register_adaptive_field!(adaptation, geometry)
    labels = [
        block.key == BlockKey(0, 0, 0) ? :refine : :keep
            for block in mesh.blocks
    ]
    @test adapt!(adaptation, labels; step = 1, force = true).changed
    initialize!(geometry.solid, levelset; lipschitz = 1)
    refresh_geometry!(geometry)
    operator = PoissonOperator(geometry)
    rebuilt = TreeGeometricMultigridPreconditioner(
        operator; geometry, coarse_size = 16, reuse = old,
    )
    fresh = TreeGeometricMultigridPreconditioner(
        operator; geometry, coarse_size = 16,
    )
    @test last(rebuilt.operators).mesh === old_coarse_mesh
    @test length(rebuilt.operators) == length(fresh.operators)
    for (a, b) in zip(rebuilt.operators, fresh.operators)
        @test a.volumes ≈ b.volumes
        @test a.face_coefficient ≈ b.face_coefficient
        @test a.matrix ≈ b.matrix
    end
    for index in eachindex(rebuilt.transfer_refreshes)
        fine_active = Float64.(rebuilt.geometry_levels[index].active_cells)
        coarse_active = Float64.(rebuilt.geometry_levels[index + 1].active_cells)
        @test rebuilt.levels[index].prolongation * coarse_active ≈
            fine_active atol = 128eps()
    end
    exact = CellField(mesh, (x, y) -> sinpi(x) * cospi(y))
    rhs = CellField(mesh)
    apply!(rhs, operator, exact, EllipticWorkspace(operator))
    solver = EllipticSolver(
        operator; preconditioner_options = (; geometry), coarse_size = 16,
    )
    stats = solve!(
        CellField(mesh), solver, rhs; rtol = 1.0e-9, max_cycles = 300,
    )
    @test stats.converged
end
