using LinearAlgebra: Symmetric, dot, eigvals, ldiv!, mul!, opnorm
using SparseArrays: dropzeros!, nnz, nonzeros, nzrange, rowvals, sparse
using Krylov

isdefined(@__MODULE__, :test_cell_values) ||
    include(joinpath(@__DIR__, "..", "test_utils.jl"))

@testset "Sorted vertex patches match dictionary assembly" begin
    function dictionary_metric(mesh, ::Type{T}) where {T}
        rows, columns, values = Int[], Int[], T[]
        patches = TwoPhaseAMR._vertex_patches(mesh, T)
        cache = TwoPhaseAMR.GradientConstructionCache(T)
        for key in sort!(collect(keys(patches)))
            faces = patches[key]
            if TwoPhaseAMR._physical_vertex(mesh, key) || length(faces) <= 2
                TwoPhaseAMR._append_boundary_patch_metric!(rows, columns, values, faces)
                continue
            end
            local_metric = TwoPhaseAMR._cached_local_inverse_metric!(cache, faces)
            for (local_column, column_face) in pairs(faces),
                    (local_row, row_face) in pairs(faces)
                push!(rows, row_face.index)
                push!(columns, column_face.index)
                push!(values, local_metric[local_row, local_column])
            end
        end
        result = sparse(rows, columns, values, length(mesh.faces), length(mesh.faces))
        dropzeros!(result)
        return result
    end

    for T in (Float32, Float64), mixed in (false, true),
            periodic in ((true, true), (false, false), (true, false))
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 1, periodic = periodic,
            refine = (key, origin, extent) ->
            mixed && key.level == 0 && key.i == 0 && key.j == 0,
        )
        expected = dictionary_metric(mesh, T)
        actual = TwoPhaseAMR._assemble_inverse_metric(
            mesh, T, TwoPhaseAMR.GradientConstructionCache(T),
        )
        @test actual.colptr == expected.colptr
        @test rowvals(actual) == rowvals(expected)
        @test isequal(nonzeros(actual), nonzeros(expected))
    end
end

@testset "Fixed-pattern elliptic refresh and matrix-free storage" begin
    function reference_refresh(operator::TwoPhaseAMR.EllipticOperator{T}) where {T}
        mesh = operator.mesh
        stiffness = operator.stiffness
        nx, ny = mesh.block_size
        lookup = Dict(
            (rowvals(stiffness)[pointer], column) => pointer
                for column in axes(stiffness, 2) for pointer in nzrange(stiffness, column)
        )
        pointers, left_faces, right_faces, factors = Int[], Int[], Int[], T[]
        function add(left_face, right_face, metric_value)
            left, right = mesh.faces[left_face], mesh.faces[right_face]
            left_cells = (
                (TwoPhaseAMR._cell_index(left.minus, nx, ny), -one(T)),
                (TwoPhaseAMR._cell_index(left.plus, nx, ny), one(T)),
            )
            right_cells = (
                (TwoPhaseAMR._cell_index(right.minus, nx, ny), -one(T)),
                (TwoPhaseAMR._cell_index(right.plus, nx, ny), one(T)),
            )
            for (row, left_sign) in left_cells, (column, right_sign) in right_cells
                push!(pointers, lookup[(row, column)])
                push!(left_faces, left_face)
                push!(right_faces, right_face)
                push!(
                    factors, left_sign * T(left.area) * metric_value *
                        right_sign * T(right.area),
                )
            end
        end
        metric = operator.gradient_operator.inverse_metric
        if metric isa TwoPhaseAMR.DiagonalInverseFaceMetric
            for face in eachindex(metric.diagonal)
                add(face, face, metric.diagonal[face])
            end
        else
            for column in axes(metric.matrix, 2), pointer in nzrange(metric.matrix, column)
                add(rowvals(metric.matrix)[pointer], column, nonzeros(metric.matrix)[pointer])
            end
        end
        offsets, permutation = TwoPhaseAMR._grouped_output_schedule(pointers, nnz(stiffness))
        return (;
            pointers = pointers[permutation], left_faces = left_faces[permutation],
            right_faces = right_faces[permutation], factors = factors[permutation],
            offsets, lookup,
        )
    end

    for T in (Float32, Float64), mixed in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 1,
            refine = (key, origin, extent) ->
            mixed && key.level == 0 && key.i == 0 && key.j == 0,
        )
        active = trues(ncells(mesh))
        active[1] = false
        full_volumes = TwoPhaseAMR._cell_volumes(mesh, T)
        volumes = ifelse.(active, full_volumes, zero(T))
        coefficients = zeros(T, length(mesh.faces))
        options = (;
            T, reaction = T(0.75), diffusion = T(1.3),
            active_cells = active, volumes, inactive_condition = :dirichlet,
        )
        assembled = EllipticOperator(mesh; face_coefficient = coefficients, options...)
        matrix_free = EllipticOperator(
            mesh; face_coefficient = coefficients, assemble_matrix = false, options...,
        )
        @test matrix_free.matrix === nothing
        @test matrix_free.refresh === nothing
        @test assembled.refresh isa TwoPhaseAMR.StiffnessRefresh{T}
        refresh = assembled.refresh
        expected = reference_refresh(assembled)
        @test refresh.stiffness_pointers == expected.pointers
        @test refresh.left_faces == expected.left_faces
        @test refresh.right_faces == expected.right_faces
        @test isequal(refresh.stiffness_factors, expected.factors)
        @test refresh.offsets == expected.offsets
        for column in axes(assembled.stiffness, 2),
                pointer in nzrange(assembled.stiffness, column)
            row = rowvals(assembled.stiffness)[pointer]
            @test refresh.transpose_pointers[pointer] == expected.lookup[(column, row)]
        end
        matrix_pattern = (copy(assembled.matrix.colptr), copy(rowvals(assembled.matrix)))
        stiffness_pattern = (copy(assembled.stiffness.colptr), copy(rowvals(assembled.stiffness)))
        source = T[sin(0.3index) for index in 1:ncells(mesh)]

        for updated in (
                T[
                    index % 7 == 0 ? 0 : 0.5 + 0.01index
                        for index in eachindex(mesh.faces)
                ],
                zeros(T, length(mesh.faces)),
                T[1 + 0.1cos(index) for index in eachindex(mesh.faces)],
            )
            TwoPhaseAMR._replace_face_coefficient(assembled, updated)
            TwoPhaseAMR._replace_face_coefficient(matrix_free, updated)
            fresh = EllipticOperator(mesh; face_coefficient = updated, options...)
            @test assembled.refresh === refresh
            @test (assembled.matrix.colptr, rowvals(assembled.matrix)) == matrix_pattern
            @test (assembled.stiffness.colptr, rowvals(assembled.stiffness)) ==
                stiffness_pattern
            # Sparse products and the fixed refresh sum the same contributions
            # in different orders; the allowance is proportional to roundoff.
            @test assembled.matrix ≈ fresh.matrix rtol = 512eps(T) atol = 512eps(T)
            actual = similar(source)
            mul!(actual, TwoPhaseAMR.FactorizedEllipticOperator(matrix_free), source)
            @test actual ≈ fresh.matrix * source rtol = 1024eps(T) atol = 1024eps(T)
        end

        changed_active = copy(active)
        changed_active[1] = true
        changed_active[end] = false
        changed_volumes = ifelse.(changed_active, full_volumes, zero(T))
        nonzero_coefficients = T[0.7 + 0.01index for index in eachindex(mesh.faces)]
        for operator in (assembled, matrix_free)
            TwoPhaseAMR._update_elliptic_activity!(operator, changed_active, changed_volumes)
            TwoPhaseAMR._replace_face_coefficient(operator, nonzero_coefficients)
        end
        fresh = EllipticOperator(
            mesh; T, reaction = T(0.75), diffusion = T(1.3),
            face_coefficient = nonzero_coefficients, active_cells = changed_active,
            volumes = changed_volumes, inactive_condition = :dirichlet,
        )
        @test assembled.matrix ≈ fresh.matrix rtol = 512eps(T) atol = 512eps(T)
        actual = similar(source)
        mul!(actual, TwoPhaseAMR.FactorizedEllipticOperator(matrix_free), source)
        @test actual ≈ fresh.matrix * source rtol = 1024eps(T) atol = 1024eps(T)
    end
end

@testset "CSC transfer schedules preserve stored entries and order" begin
    # An explicit zero is a stored coupling and must remain in the map.
    stiffness = sparse(
        [1, 2, 1, 2, 3], [1, 1, 2, 2, 3], [2.0, -1.0, -1.0, 2.0, 0.0], 3, 3,
    )
    tentative = sparse([1, 2, 3], [1, 1, 2], ones(3), 3, 2)
    transfer = TwoPhaseAMR._smoothed_transfer_refresh(stiffness, tentative)
    expected_offsets, expected_indices = TwoPhaseAMR._grouped_output_schedule(
        transfer.stiffness_to_transfer, nnz(transfer.prolongation),
    )
    @test transfer.transfer_offsets == expected_offsets
    @test transfer.stiffness_indices == expected_indices
    @test nnz(stiffness) == 5
    @test all(
        rowvals(transfer.prolongation)[transfer.stiffness_to_transfer[pointer]] ==
            rowvals(stiffness)[pointer]
            for pointer in eachindex(nonzeros(stiffness))
    )
    @test_throws ErrorException TwoPhaseAMR._csc_stored_pointer(stiffness, 3, 1)

    for mixed in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 1,
            refine = (key, origin, extent) ->
            mixed && key.level == 0 && key.i == 0 && key.j == 0,
        )
        coefficients = ones(length(mesh.faces))
        coefficients[1] = 0
        assembled = PoissonOperator(mesh; face_coefficient = coefficients)
        matrix_free = PoissonOperator(
            mesh; face_coefficient = coefficients, assemble_matrix = false,
        )
        a = TreeGeometricMultigridPreconditioner(assembled; coarse_size = 16)
        b = TreeGeometricMultigridPreconditioner(matrix_free; coarse_size = 16)
        factorized = first(b.transfer_refreshes)
        offsets, indices = TwoPhaseAMR._grouped_output_schedule(
            factorized.entry_to_transfer, nnz(factorized.prolongation),
        )
        @test factorized.transfer_offsets == offsets
        @test factorized.entry_indices == indices
        # The two assembly paths differ only in sparse summation roundoff.
        @test first(a.levels).prolongation ≈ first(b.levels).prolongation rtol =
            1024eps() atol = 1024eps()
        TwoPhaseAMR._replace_face_coefficient(matrix_free, 1 .+ coefficients)
        old_factorization = b.coarse_factorization
        TwoPhaseAMR._refresh_geometric_multigrid!(b, matrix_free)
        @test b.coarse_factorization === old_factorization
        exact = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
        rhs = CellField(mesh)
        apply!(rhs, matrix_free, exact, EllipticWorkspace(matrix_free))
        solver = EllipticSolver(matrix_free; coarse_size = 16)
        solution = CellField(mesh)
        stats = solve!(solution, solver, rhs; rtol = 1.0e-9, max_cycles = 100)
        @test stats.converged
        reference = test_cell_values(exact)
        reference .-= dot(matrix_free.volumes, reference) / sum(matrix_free.volumes)
        @test test_cell_values(solution) ≈ reference rtol = 1.0e-7 atol = 1.0e-7
    end
end

@testset "Factorized pressure action matches assembled operator" begin
    for topology in (:uniform, :mixed, :adapted)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) -> topology == :mixed &&
                (
                key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0)
            ),
        )
        if topology == :adapted
            marker = CellField(mesh, (x, y) -> sinpi(2x) * sinpi(2y))
            adaptation = AMRWorkspace(mesh, marker)
            adapt!(adaptation, [:refine]; step = 1, force = true)
            labels = [
                block.key.level == 1 && block.key.i == 0 &&
                    block.key.j == 0 ? :refine : :keep
                    for block in mesh.blocks
            ]
            adapt!(adaptation, labels; step = 2, force = true)
        end
        coefficient = [
            1 + 0.3sin(index) for index in eachindex(mesh.faces)
        ]
        operator = EllipticOperator(
            mesh; reaction = 0.2, diffusion = 0.7,
            face_coefficient = coefficient,
        )
        action = TwoPhaseAMR.FactorizedEllipticOperator(operator)
        x = [sin(0.7index) for index in eachindex(operator.volumes)]
        y = [cos(0.3index) for index in eachindex(operator.volumes)]
        action_x = similar(x)
        action_y = similar(y)
        mul!(action_x, action, x)
        @test action_x ≈ operator.matrix * x rtol = 64eps() atol = 64eps()
        mul!(action_y, action, y)
        @test dot(x, action_y) ≈ dot(action_x, y) rtol = 128eps()
        @test dot(x, action_x) > 0
        poisson = PoissonOperator(mesh; face_coefficient = coefficient)
        poisson_action = TwoPhaseAMR.FactorizedEllipticOperator(poisson)
        mul!(action_x, poisson_action, ones(length(x)))
        @test maximum(abs, action_x) <= 64eps()
    end
end

@testset "Jacobi policies retain smoothing semantics" begin
    diagonal = [2.0, 0.0, -3.0]
    weight = TwoPhaseAMR._jacobi_weight!(
        diagonal, [1.0, 0.0, 0.0];
        skip_nonpositive = true, clamp_weight = false, zero_weight = 0.0,
    )
    @test diagonal == [0.5, 0.0, -3.0]
    @test weight == 4.0 / (3.0 * 0.5)
    @test_throws ArgumentError TwoPhaseAMR._jacobi_weight!(
        [2.0, 0.0], [1.0, 0.0],
    )
    @test TwoPhaseAMR._jacobi_weight!([2.0], [1.0]) == 1.0
    @test TwoPhaseAMR._jacobi_weight!([2.0], [0.0]) == 1.0
    @test TwoPhaseAMR._jacobi_weight!(
        [2.0], [0.0];
        skip_nonpositive = true, clamp_weight = false, zero_weight = 0.0,
    ) == 0.0
end

@testset "Elliptic zero-Dirichlet inactive cells" begin
    mesh = BlockTreeMesh(block_size = (6, 6))
    references = TwoPhaseAMR._cell_references(mesh)
    active = BitVector(
        cellcenter(mesh, ref.block, ref.i, ref.j)[1] < 0.75
            for ref in references
    )
    volumes = [
        active[index] ? mesh.blocks[ref.block].volume : 0.0
            for (index, ref) in pairs(references)
    ]
    operator = EllipticOperator(
        mesh; active_cells = active, volumes,
        inactive_condition = :dirichlet,
    )
    @test !operator.singular
    action = TwoPhaseAMR.FactorizedEllipticOperator(operator)
    source = [
        active[index] ? sin(0.3index) : 0.0
            for index in eachindex(active)
    ]
    factorized = similar(source)
    mul!(factorized, action, source)
    @test factorized ≈ operator.matrix * source
    @test dot(source, factorized) > 0

    rhs = CellField(mesh)
    for index in eachindex(references)
        TwoPhaseAMR._set!(
            rhs,
            active[index] ? factorized[index] / volumes[index] : 0.0,
            references[index],
        )
    end
    solution = CellField(mesh)
    stats = solve!(
        solution, EllipticSolver(operator; preconditioner = nothing), rhs;
        rtol = 0.0, atol = 1.0e-11, max_cycles = 200,
        compatibility = :error,
    )
    @test stats.converged
    @test test_cell_values(solution) ≈ source atol = 1.0e-11
end

@testset "Rediscretized pressure hierarchy on pruned trees" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 3,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < 3 && origin[1] < 0.5 && origin[2] < 0.5),
    )
    coefficient = [
        exp(0.4sin(index)) for index in eachindex(mesh.faces)
    ]
    operator = PoissonOperator(mesh; face_coefficient = coefficient)
    all_terminal = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        operator; multigrid = MultigridOptions(
            coarse_size = 64, pre_sweeps = 1, post_sweeps = 3,
        ), pre_sweeps = 2, pruning = :all,
    )
    deepest = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 64, pruning = :deepest,
    )
    @test all_terminal.pre_sweeps == 2
    @test all_terminal.post_sweeps == 3
    @test length(all_terminal.levels) > 2
    @test length(all_terminal.operators[2].volumes) <
        length(deepest.operators[2].volumes)
    @test all(diff(length.(getproperty.(all_terminal.operators, :volumes))) .< 0)

    for index in 1:(length(all_terminal.operators) - 1)
        fine = all_terminal.operators[index]
        coarse = all_terminal.operators[index + 1]
        expected_coefficient = similar(coarse.face_coefficient)
        TwoPhaseAMR._restrict_face_coefficients!(
            expected_coefficient, fine.face_coefficient,
            all_terminal.coefficient_restrictions[index],
        )
        @test coarse.face_coefficient ≈ expected_coefficient rtol = 16eps()
        reference = PoissonOperator(
            coarse.mesh; face_coefficient = expected_coefficient,
        )
        @test coarse.matrix ≈ reference.matrix rtol = 64eps()
        @test maximum(abs, coarse.matrix * ones(length(coarse.volumes))) <=
            1024eps() * max(opnorm(Matrix(coarse.matrix), Inf), 1.0)
    end

    refreshed = TwoPhaseAMR._replace_face_coefficient(
        operator,
        [exp(0.6cos(index)) for index in eachindex(mesh.faces)],
    )
    identity = all_terminal
    TwoPhaseAMR._refresh_geometric_multigrid!(all_terminal, refreshed)
    @test all_terminal === identity
    large_block = BlockTreeMesh(block_size = (8, 8))
    within_block = TwoPhaseAMR.TreeGeometricMultigridPreconditioner(
        PoissonOperator(large_block); coarse_size = 16,
    )
    @test within_block isa TreeGeometricMultigridPreconditioner
    @test length(within_block.levels) == 1
    @test first(within_block.operators).mesh === large_block
end

@testset "Root-block Poisson hierarchy and stopping" begin
    # The face coefficient is affine along each face. Its exact area average
    # on every coarser face is therefore its value at the coarse face center.
    coefficient(face) =
        3.0 + 0.1face.center[1] + 0.2face.center[2] + 0.3face.axis
    expected_bases = [(16, 16), (8, 8), (4, 4)]
    expected_sizes = [256, 64, 16]
    for periodic in ((true, true), (true, false), (false, true), (false, false))
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4),
            origin = (-1.0, 0.25), extent = (2.0, 1.5), periodic = periodic,
        )
        operator = PoissonOperator(
            mesh; face_coefficient = coefficient.(mesh.faces),
        )
        hierarchy = TreeGeometricMultigridPreconditioner(
            operator; multigrid = MultigridOptions(coarse_size = 16),
        )
        @test [level.mesh.base_mesh for level in hierarchy.operators] ==
            expected_bases
        @test length.(getproperty.(hierarchy.operators, :volumes)) ==
            expected_sizes
        @test length(hierarchy.levels) == length(expected_sizes)
        @test first(hierarchy.operators).mesh === mesh
        for (index, coarse_operator) in pairs(hierarchy.operators)
            coarse = coarse_operator.mesh
            @test coarse.block_size == mesh.block_size
            @test coarse.origin == mesh.origin
            @test coarse.extent == mesh.extent
            @test coarse.periodic == periodic
            @test all(block -> iszero(block.key.level), coarse.blocks)
            @test nblocks(coarse) == prod(coarse.base_mesh .÷ coarse.block_size)
            @test ncells(coarse) == prod(coarse.base_mesh)
            @test size(hierarchy.levels[index].matrix) ==
                (ncells(coarse), ncells(coarse))
            # Summation of cell volumes and affine face averages is accurate
            # to round-off on these small, dyadically nested meshes.
            @test sum(coarse_operator.volumes) ≈ prod(mesh.extent) rtol = 128eps()
            @test coarse_operator.face_coefficient ≈
                coefficient.(coarse.faces) rtol = 128eps()
            @test maximum(
                abs, coarse_operator.matrix * ones(ncells(coarse)),
            ) <= 1024eps() * max(opnorm(Matrix(coarse_operator.matrix), Inf), 1.0)
            for axis in 1:2
                tangential_cells = coarse.base_mesh[3 - axis]
                @test count(
                    face -> face.axis == axis, coarse.boundary_faces,
                ) == (periodic[axis] ? 0 : 2tangential_cells)
                @test count(
                    face -> face.axis == axis &&
                        face.center[axis] == coarse.origin[axis],
                    coarse.faces,
                ) == (periodic[axis] ? tangential_cells : 0)
            end
            if index < length(hierarchy.operators)
                next_size = length(hierarchy.operators[index + 1].volumes)
                prolongation = hierarchy.levels[index].prolongation
                @test size(prolongation) == (ncells(coarse), next_size)
                # Smoothed interpolation must preserve the constant
                # nullspace, up to a few sparse floating-point sums per row.
                @test prolongation * ones(next_size) ≈
                    ones(ncells(coarse)) atol = 128eps() rtol = 0
            end
        end
    end

    # Agglomeration is isotropic: an odd or unit root count stops both axes.
    for (layout, expected) in (
            ((6, 4), ((6, 4), (3, 2))),
            ((4, 6), ((4, 6), (2, 3))),
            ((3, 4), ((3, 4),)),
            ((4, 3), ((4, 3),)),
            ((1, 4), ((1, 4),)),
            ((4, 1), ((4, 1),)),
            ((1, 1), ((1, 1),)),
        )
        mesh = BlockTreeMesh(
            base_mesh = layout .* (4, 4), block_size = (4, 4),
            periodic = (true, false),
        )
        hierarchy = TreeGeometricMultigridPreconditioner(
            PoissonOperator(mesh); multigrid = MultigridOptions(coarse_size = 1),
        )
        @test [level.mesh.root.layout for level in hierarchy.operators] ==
            collect(expected)
        @test length.(getproperty.(hierarchy.operators, :volumes)) ==
            [16prod(roots) for roots in expected]
    end

    mesh = BlockTreeMesh(base_mesh = (16, 16), block_size = (4, 4))
    operator = PoissonOperator(mesh)
    for (coarse_size, max_levels, expected) in (
            (64, 32, [256, 64]),
            (16, 2, [256, 64]),
            (16, 1, [256]),
            (256, 32, [256]),
        )
        hierarchy = TreeGeometricMultigridPreconditioner(
            operator; multigrid = MultigridOptions(; coarse_size, max_levels),
        )
        @test length.(getproperty.(hierarchy.operators, :volumes)) == expected
    end

    # The old flat hierarchy is a direct augmented LU, not a comparable
    # multilevel cycle: it is an inverse on compatible mean-zero data and CG
    # can already converge in one iteration. Strict improvement is impossible.
    old_flat = TreeGeometricMultigridPreconditioner(
        operator; multigrid = MultigridOptions(coarse_size = 16, max_levels = 1),
    )
    exact = [sin(0.17index) + 0.3cos(0.31index) for index in 1:ncells(mesh)]
    exact .-= dot(operator.volumes, exact) / sum(operator.volumes)
    rhs = operator.matrix * exact
    recovered = similar(rhs)
    ldiv!(recovered, old_flat, rhs)
    @test length(old_flat.levels) == 1
    # Allow for the conditioning of the singular solve and sparse LU round-off.
    @test recovered ≈ exact rtol = 1.0e-11 atol = 1.0e-11

    # Both the old direct hierarchy and the new V-cycle converge to the same
    # requested tolerance on the flat multiroot mesh. Their iteration counts
    # need not agree: the old bottom solve was already exact.
    new_flat = TreeGeometricMultigridPreconditioner(
        operator; multigrid = MultigridOptions(coarse_size = 16),
    )
    @test length(new_flat.levels) == 3
    exact_field = CellField(
        mesh, (x, y) -> sinpi(2x) * cospi(2y) + 0.2cospi(4x - 2y),
    )
    rhs_field = CellField(mesh)
    apply!(rhs_field, operator, exact_field, EllipticWorkspace(operator))
    for preconditioner in (old_flat, new_flat)
        krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
        solver = EllipticSolver(operator, krylov, preconditioner)
        stats = solve!(
            CellField(mesh), solver, rhs_field; rtol = 1.0e-8, max_cycles = 100,
        )
        @test stats.converged
    end

    # Both pruning policies must continue into the same base-mesh hierarchy.
    mixed = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && key.i == 0 && key.j == 0,
    )
    for pruning in (:all, :deepest)
        hierarchy = TreeGeometricMultigridPreconditioner(
            PoissonOperator(mixed); coarse_size = 16, pruning,
        )
        @test length.(getproperty.(hierarchy.operators, :volumes)) ==
            [112, 64, 16]
    end
end

@testset "Periodic base meshes retain four cells per period" begin
    for periodic in ((true, true), (true, false), (false, true))
        mesh = BlockTreeMesh(
            base_mesh = (4, 4), block_size = (2, 2), periodic = periodic,
        )
        operator = PoissonOperator(mesh)
        hierarchy = TreeGeometricMultigridPreconditioner(
            operator; coarse_size = 1,
        )
        @test length(hierarchy.levels) == 1
        @test only(hierarchy.operators).mesh === mesh
        rhs = CellField(mesh, (x, y) -> sinpi(2x) + cospi(2y))
        solver = EllipticSolver(
            operator, Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh))),
            hierarchy,
        )
        @test solve!(
            CellField(mesh), solver, rhs; rtol = 1.0e-8, max_cycles = 100,
        ).converged
    end
    # The bound applies only to periodic directions.
    for (base, periodic, expected) in (
            ((8, 8), (true, true), [(8, 8), (4, 4)]),
            ((4, 4), (false, false), [(4, 4), (2, 2)]),
            ((8, 4), (true, false), [(8, 4), (4, 2)]),
            ((4, 8), (false, true), [(4, 8), (2, 4)]),
        )
        mesh = BlockTreeMesh(
            base_mesh = base, block_size = (2, 2), periodic = periodic,
        )
        hierarchy = TreeGeometricMultigridPreconditioner(
            PoissonOperator(mesh); coarse_size = 1,
        )
        @test [op.mesh.base_mesh for op in hierarchy.operators] == expected
    end
end

@testset "Root-block hierarchy reuse after AMR" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && key.i == 0 && key.j == 0,
    )
    options = MultigridOptions(coarse_size = 16)
    old = TreeGeometricMultigridPreconditioner(
        PoissonOperator(mesh); multigrid = options,
    )
    @test [level.mesh.base_mesh for level in old.operators] ==
        [(16, 16), (16, 16), (8, 8), (4, 4)]
    old_factorization = old.coarse_factorization
    old_symbolic = old_factorization.symbolic

    marker = CellField(mesh; init = 1.0)
    adaptation = AMRWorkspace(mesh, marker)
    labels = [
        block.key == BlockKey(0, 1, 1) ? :refine : :keep
            for block in mesh.blocks
    ]
    @test adapt!(adaptation, labels; step = 1, force = true).changed
    coefficient = [
        2.0 + 0.25face.center[1] + 0.15face.center[2] + 0.1face.axis
            for face in mesh.faces
    ]
    operator = PoissonOperator(mesh; face_coefficient = coefficient)
    reused = TreeGeometricMultigridPreconditioner(
        operator; multigrid = options, reuse = old,
    )
    fresh = TreeGeometricMultigridPreconditioner(
        operator; multigrid = options,
    )
    @test length(reused.operators) == length(old.operators) == length(fresh.operators)
    @test reused.reused_levels == length(old.operators) - 1
    @test all(
        reused.operators[index] === old.operators[index]
            for index in 2:length(old.operators)
    )
    @test last(reused.levels) === last(old.levels)
    @test reused.coarse_factorization === old_factorization
    @test reused.coarse_factorization.symbolic === old_symbolic
    for (reused_operator, fresh_operator) in zip(reused.operators, fresh.operators)
        # Refreshing an unchanged coarse topology must give the same
        # coefficients and matrix as rebuilding it, up to sparse summation.
        @test reused_operator.face_coefficient ≈
            fresh_operator.face_coefficient rtol = 1.0e-12 atol = 1.0e-12
        @test reused_operator.matrix ≈
            fresh_operator.matrix rtol = 1.0e-12 atol = 1.0e-12
    end
    rhs = [sin(0.2index) + cos(0.7index) for index in 1:ncells(mesh)]
    rhs .-= (sum(rhs) / sum(operator.volumes)) .* operator.volumes
    reused_result, fresh_result = similar(rhs), similar(rhs)
    ldiv!(reused_result, reused, rhs)
    ldiv!(fresh_result, fresh, rhs)
    # The two V-cycles differ only by accumulated floating-point round-off.
    @test reused_result ≈ fresh_result rtol = 1.0e-11 atol = 1.0e-11
end

@testset "Root-block V-cycle Poisson CG comparison" begin
    # Capping at four levels reproduces the tree-only hierarchy on this
    # mixed-level mesh; unlike the flat case it is not a direct fine solve.
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), max_level = 3,
        refine = (key, origin, extent) ->
        (key.level == 0 && key.i == 0 && key.j == 0) ||
            (key.level == 1 && key.i == 1 && key.j == 1) ||
            (key.level == 2 && key.i == 2 && key.j == 2),
    )
    operator = PoissonOperator(mesh)
    old = TreeGeometricMultigridPreconditioner(
        operator; multigrid = MultigridOptions(
            coarse_size = 16, max_levels = 4, pre_sweeps = 4, post_sweeps = 4,
        ),
    )
    hierarchy = TreeGeometricMultigridPreconditioner(
        operator; multigrid = MultigridOptions(
            coarse_size = 16, pre_sweeps = 4, post_sweeps = 4,
        ),
    )
    @test length(old.levels) == 4
    @test length(hierarchy.levels) == 5
    @test last(old.operators).mesh.base_mesh == (8, 8)
    @test last(hierarchy.operators).mesh.base_mesh == (4, 4)

    exact = CellField(mesh)
    initialize!(
        exact, (x, y) -> sinpi(2x) * cospi(2y) +
            0.2cospi(4x - 2y) + 0.1sinpi(4y),
    )
    rhs = CellField(mesh)
    apply!(rhs, operator, exact, EllipticWorkspace(operator))
    exact_values = test_cell_values(exact)
    exact_values .-= dot(operator.volumes, exact_values) / sum(operator.volumes)
    function solve_with(preconditioner)
        krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
        solver = EllipticSolver(operator, krylov, preconditioner)
        solution = CellField(mesh)
        stats = solve!(
            solution, solver, rhs; rtol = 1.0e-8, max_cycles = 100,
        )
        return stats, test_cell_values(solution)
    end
    old_stats, old_values = solve_with(old)
    stats, values = solve_with(hierarchy)
    @test old_stats.converged && stats.converged
    @test old_stats.method === :cg && stats.method === :cg
    @test old_stats.cycles > 1
    @test stats.cycles <= old_stats.cycles
    # The absolute error budget allows conditioning of the inverse Poisson
    # operator at a 1e-8 relative strong-residual tolerance.
    @test old_values ≈ exact_values rtol = 0 atol = 1.0e-6
    @test values ≈ exact_values rtol = 0 atol = 1.0e-6
end

function _poisson_error(block_size; mixed = true, rtol = 1.0e-10)
    mesh = mixed ? BlockTreeMesh(
            block_size = (block_size, block_size), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ) : BlockTreeMesh(block_size = (block_size, block_size))
    exact_function = (x, y) -> sinpi(2x) * cospi(2y)
    rhs = CellField(mesh)
    initialize!(rhs, (x, y) -> 8pi^2 * exact_function(x, y))
    pressure = CellField(mesh)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
    stats = solve!(pressure, solver, rhs; rtol, max_cycles = 60)

    squared_error = 0.0
    maximum_error = 0.0
    total_volume = 0.0
    nx, ny = mesh.block_size
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(mesh, block_index, i, j)
        error = abs(interior(pressure, block_index)[i, j] - exact_function(x, y))
        squared_error += block.volume * error^2
        maximum_error = max(maximum_error, error)
        total_volume += block.volume
    end
    return (
        l2 = sqrt(squared_error / total_volume), linf = maximum_error,
        stats, levels = length(solver.preconditioner.levels),
    )
end

@testset "elliptic operator preserves gradient-divergence duality" begin
    for mesh in (
            BlockTreeMesh(block_size = (8, 8)),
            BlockTreeMesh(
                block_size = (4, 4), max_level = 2,
                refine = (key, origin, extent) ->
                key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
            ),
        )
        pressure = CellField(mesh)
        initialize!(
            pressure,
            (x, y) -> sinpi(2x) * cospi(2y) + 0.1cospi(4x - 2y)
        )
        operator = PoissonOperator(mesh)
        workspace = EllipticWorkspace(operator)
        applied = CellField(mesh)
        apply!(applied, operator, pressure, workspace)

        face_gradient = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        gradient!(face_gradient, pressure, operator.gradient_operator)
        divergence = CellField(mesh)
        divergence!(divergence, face_gradient)
        @test test_cell_values(applied) ≈ -test_cell_values(divergence) atol = 8.0e-13 rtol = 8.0e-13

        @test operator.matrix ≈ transpose(operator.matrix) atol = 4.0e-14 rtol = 4.0e-14
        @test maximum(abs, operator.matrix * ones(ncells(mesh))) <= 2.0e-13
    end
end

@testset "variable-coefficient elliptic operator" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    cell_coefficient = CellField(mesh)
    initialize!(cell_coefficient, (x, y) -> x < 0.5 ? 1.0 : 25.0)
    face_coefficient = harmonic_face_coefficients(cell_coefficient)
    @test all(>(0), face_coefficient)
    @test minimum(face_coefficient) >= 1
    @test maximum(face_coefficient) <= 25

    operator = PoissonOperator(mesh; face_coefficient)
    pressure = CellField(mesh)
    initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
    workspace = EllipticWorkspace(operator)
    applied = CellField(mesh)
    apply!(applied, operator, pressure, workspace)
    weighted = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    weighted_gradient!(weighted, pressure, operator, workspace)
    divergence = CellField(mesh)
    divergence!(divergence, weighted)
    @test test_cell_values(applied) ≈ -test_cell_values(divergence) atol = 2.0e-11 rtol = 2.0e-13
    @test operator.matrix ≈ transpose(operator.matrix) atol = 2.0e-13 rtol = 2.0e-13

    eigenvalues = eigvals(Symmetric(Matrix(operator.matrix)))
    tolerance = 5.0e-12 * maximum(eigenvalues)
    @test first(eigenvalues) >= -tolerance
    @test count(value -> abs(value) <= tolerance, eigenvalues) == 1
end

@testset "geometric multigrid Poisson convergence" begin
    results = _poisson_error.((4, 8, 16))
    @test all(result -> result.stats.converged, results)
    @test maximum(getproperty.(getproperty.(results, :stats), :cycles)) -
        minimum(getproperty.(getproperty.(results, :stats), :cycles)) <= 3
    for norm_name in (:l2, :linf)
        errors = collect(getproperty.(results, norm_name))
        rates = log2.(errors[1:2] ./ errors[2:3])
        @test all(diff(errors) .< 0)
        @test minimum(rates) > 1.7
    end
end

@testset "external Krylov solver and independent preconditioner" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 4,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < 4 && origin[1] < 0.5 && origin[2] < 0.5),
    )
    rhs = CellField(mesh)
    initialize!(
        rhs, (x, y) ->
        8pi^2 * sinpi(2x) * cospi(2y) +
            2pi^2 * sinpi(2x + 2y)
    )
    operator = PoissonOperator(mesh)

    preconditioner = TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 32
    )
    krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
    solver = EllipticSolver(operator, krylov, preconditioner)
    solution = CellField(mesh)
    stats = solve!(solution, solver, rhs; rtol = 1.0e-10, max_cycles = 100)

    @test stats.converged
    @test stats.method === :cg
    @test stats.preconditioner === :tree_geometric_multigrid
    @test solver.krylov === krylov
    @test solver.preconditioner === preconditioner

    default_solver = EllipticSolver(operator; coarse_size = 32)
    default_solution = CellField(mesh)
    default_stats = solve!(
        default_solution, default_solver, rhs; rtol = 1.0e-10, max_cycles = 100
    )
    @test default_stats.converged
    @test default_stats.method === :cg
    @test maximum(
        abs,
        test_cell_values(default_solution) - test_cell_values(solution)
    ) < 2.0e-9

    no_preconditioner = EllipticSolver(operator; preconditioner = nothing)
    plain_solution = CellField(mesh)
    plain_stats = solve!(
        plain_solution, no_preconditioner, rhs; rtol = 1.0e-8, max_cycles = 500
    )
    @test plain_stats.converged
    @test plain_stats.preconditioner === :none
    @test stats.cycles < plain_stats.cycles
end

@testset "Poisson solve is independent of uniform block partitioning" begin
    one_block = BlockTreeMesh(block_size = (16, 16))
    four_blocks = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    four_roots = BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8))
    function solve_partition(mesh)
        rhs = CellField(mesh)
        initialize!(rhs, (x, y) -> 8pi^2 * sinpi(2x) * cospi(2y))
        pressure = CellField(mesh)
        solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
        stats = solve!(pressure, solver, rhs; rtol = 2.0e-12)
        values = Dict{Tuple{Float64, Float64}, Float64}()
        nx, ny = mesh.block_size
        for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
            values[cellcenter(mesh, block, i, j)] = interior(pressure, block)[i, j]
        end
        return values, stats
    end
    first_values, first_stats = solve_partition(one_block)
    second_values, second_stats = solve_partition(four_blocks)
    third_values, third_stats = solve_partition(four_roots)
    @test first_stats.converged && second_stats.converged &&
        third_stats.converged
    @test keys(first_values) == keys(second_values)
    @test maximum(
        abs(first_values[key] - second_values[key])
            for key in keys(first_values)
    ) <= 2.0e-12
    @test keys(first_values) == keys(third_values)
    @test maximum(
        abs(first_values[key] - third_values[key])
            for key in keys(first_values)
    ) <= 2.0e-12
end

@testset "periodic nullspace convention" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    rhs = CellField(mesh; init = 3.0)
    pressure = CellField(mesh; init = 7.0)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 4)
    @test_throws ArgumentError solve!(pressure, solver, rhs; compatibility = :error)
    @test_throws ArgumentError solve!(pressure, solver, rhs; compatibility = :invalid)
    stats = solve!(pressure, solver, rhs; compatibility = :project)
    @test stats.converged
    @test stats.removed_rhs_mean ≈ 3
    @test abs(integrate(pressure)) <= 16eps()
    @test maximum(abs, test_cell_values(pressure)) <= 16eps()
end

@testset "open pressure boundary operators and solvers" begin
    for mixed in (false, true), periodic in ((false, false), (false, true))
        mesh = BlockTreeMesh(
            base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0),
            periodic = periodic, max_level = 1,
            refine = (key, origin, extent) -> mixed && key.level == 0 && key.i == 3,
        )
        boundary = VelocityBoundaryConditions(
            left = ZeroGradientVelocity(), right = Outflow(),
            bottom = NoSlip(), top = FreeSlip(),
        )
        material = [
            0.8 + 0.2face.center[2] for face in mesh.boundary_faces
        ]
        options = (;
            face_coefficient = 1.2, velocity_boundary = boundary,
            boundary_face_coefficient = material,
        )
        assembled = PoissonOperator(mesh; options...)
        factorized = PoissonOperator(mesh; options..., assemble_matrix = false)
        @test !assembled.singular && !factorized.singular
        pressure = CellField(mesh, (x, y) -> 0.7 + sinpi(x / 2) * cospi(y))
        vector = test_cell_values(pressure)
        output = similar(vector)
        mul!(output, TwoPhaseAMR.FactorizedEllipticOperator(factorized), vector)
        @test output ≈ assembled.matrix * vector atol = 1.0e-12 rtol = 1.0e-12
        shifted = EllipticOperator(
            mesh; options..., reaction = 0.4, diffusion = 0.3,
        )
        mul!(output, TwoPhaseAMR.FactorizedEllipticOperator(shifted), vector)
        @test output ≈ shifted.matrix * vector atol = 1.0e-12 rtol = 1.0e-12
        extended = PoissonOperator(
            mesh; face_coefficient = fill(1.2, length(mesh.faces)),
            velocity_boundary = boundary,
        )
        @test all(
            isapprox(extended.boundary_face_coefficient[index], 1.2)
                for (index, face) in pairs(mesh.boundary_faces) if face.axis == 1
        )

        gradient = FaceVelocity(mesh)
        work = EllipticWorkspace(assembled)
        weighted_gradient!(gradient, pressure, assembled, work)
        strong = CellField(mesh)
        divergence = CellField(mesh)
        apply!(strong, assembled, pressure, work)
        divergence!(divergence, gradient)
        @test test_cell_values(strong) ≈ -test_cell_values(divergence) atol = 2.0e-12
        for (index, face) in pairs(mesh.boundary_faces)
            if face.axis == 1
                spacing = mesh.blocks[face.cell.block].spacing[face.axis]
                expected = -face.side * material[index] *
                    TwoPhaseAMR._get(pressure, face.cell) / (spacing / 2)
                @test gradient.boundary[index] ≈ expected atol = 1.0e-12
            else
                @test iszero(gradient.boundary[index])
            end
        end

        # Both matrix representations and the rediscretized hierarchy must
        # solve a nonzero-mean RHS without a Neumann compatibility projection.
        for operator in (assembled, factorized)
            solver = EllipticSolver(operator; coarse_size = 8)
            numerical = CellField(mesh)
            stats = solve!(
                numerical, solver, strong;
                compatibility = :error, rtol = 1.0e-11, max_cycles = 300,
            )
            @test stats.converged
            @test iszero(stats.removed_rhs_mean)
            @test test_cell_values(numerical) ≈ vector atol = 1.0e-9 rtol = 1.0e-9
            @test !solver.preconditioner.coarse_augmented
            @test all(!level_operator.singular for level_operator in solver.preconditioner.operators)
        end

        # Coefficient refresh must retain the boundary diagonal in both paths.
        refreshed_solver = EllipticSolver(assembled; coarse_size = 8)
        updated = fill(1.7, length(mesh.faces))
        updated_boundary = fill(0.6, length(mesh.boundary_faces))
        TwoPhaseAMR._replace_face_coefficient(
            assembled, updated; boundary_face_coefficient = updated_boundary,
        )
        TwoPhaseAMR._replace_face_coefficient(
            factorized, updated; boundary_face_coefficient = updated_boundary,
        )
        mul!(output, TwoPhaseAMR.FactorizedEllipticOperator(factorized), vector)
        @test output ≈ assembled.matrix * vector atol = 1.0e-12 rtol = 1.0e-12
        TwoPhaseAMR._refresh_geometric_multigrid!(refreshed_solver.preconditioner, assembled)
        @test all(!level_operator.singular for level_operator in refreshed_solver.preconditioner.operators)
        for level_operator in refreshed_solver.preconditioner.operators
            @test all(
                isapprox(level_operator.boundary_face_coefficient[index], 0.6)
                    for (index, face) in pairs(level_operator.mesh.boundary_faces)
                    if face.axis == 1
            )
        end
        apply!(strong, assembled, pressure, work)
        numerical = CellField(mesh)
        refreshed = solve!(
            numerical, refreshed_solver, strong;
            compatibility = :error, rtol = 1.0e-11, max_cycles = 300,
        )
        @test refreshed.converged
        @test test_cell_values(numerical) ≈ vector atol = 1.0e-9 rtol = 1.0e-9

        closed_solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 8)
        velocity = FaceVelocity(mesh)
        before = copy(velocity)
        @test_throws ArgumentError project!(
            velocity, CellField(mesh), closed_solver, 0.1, ProjectionWorkspace(closed_solver);
            velocity_boundary = boundary,
        )
        @test velocity.interior == before.interior
        @test velocity.boundary == before.boundary
    end
end

@testset "vertical open pressure faces with periodic transverse direction" begin
    mesh = BlockTreeMesh(
        block_size = (8, 4), extent = (3.0, 1.0), periodic = (true, false),
    )
    boundary = VelocityBoundaryConditions(
        bottom = ZeroGradientVelocity(), top = Outflow(),
    )
    operator = PoissonOperator(mesh; face_coefficient = 2.0, velocity_boundary = boundary)
    pressure = CellField(mesh; init = 1.0)
    gradient = FaceVelocity(mesh)
    work = EllipticWorkspace(operator)
    weighted_gradient!(gradient, pressure, operator, work)
    @test all(iszero, gradient.interior)
    for (index, face) in pairs(mesh.boundary_faces)
        h = mesh.blocks[face.cell.block].spacing[2]
        @test face.axis == 2
        @test gradient.boundary[index] ≈ -face.side * 2 / (h / 2)
    end
    strong = CellField(mesh)
    divergence = CellField(mesh)
    apply!(strong, operator, pressure, work)
    divergence!(divergence, gradient)
    @test test_cell_values(strong) ≈ -test_cell_values(divergence) atol = 1.0e-12
end

@testset "cut-cell open pressure boundary flux" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i == 3,
    )
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 2.0, y - 0.5) - 0.26; lipschitz = 1)
    boundary = VelocityBoundaryConditions(right = Outflow())
    operator = PoissonOperator(geometry; face_coefficient = 0.75, velocity_boundary = boundary)
    pressure = CellField(mesh, (x, y) -> 0.8 + x / 3 + y / 5)
    gradient = FaceVelocity(mesh)
    work = EllipticWorkspace(operator)
    TwoPhaseAMR._cut_cell_weighted_gradient!(gradient, pressure, operator, work, geometry)
    strong = CellField(mesh)
    divergence = CellField(mesh)
    apply!(strong, operator, pressure, work)
    divergence!(divergence, gradient, geometry)
    for (index, ref) in pairs(operator.references)
        operator.volumes[index] > 0 || continue
        @test TwoPhaseAMR._get(strong, ref) ≈ -TwoPhaseAMR._get(divergence, ref) atol = 2.0e-11
    end
    @test any(0 < a < 1 for a in geometry.boundary_aperture)
    for (index, face) in pairs(mesh.boundary_faces)
        if face.axis == 1 && face.side > 0 && geometry.boundary_aperture[index] > 0
            spacing = mesh.blocks[face.cell.block].spacing[face.axis]
            expected = -0.75 * TwoPhaseAMR._get(pressure, face.cell) / (spacing / 2)
            @test gradient.boundary[index] ≈ expected atol = 1.0e-12
        else
            @test iszero(gradient.boundary[index])
        end
    end

    velocity = FaceVelocity(
        mesh, (x, y, t, axis) -> axis == 1 ?
            sinpi(x / 2) * cospi(y) : cospi(x / 2) * sinpi(y),
    )
    enforce_velocity_boundary!(velocity, boundary; geometry)
    enforce_geometry_no_slip!(velocity, geometry)
    predictor = copy(velocity)
    solver = EllipticSolver(
        operator; coarse_size = 8, preconditioner_options = (; geometry),
    )
    projection = ProjectionWorkspace(solver)
    dt = 0.02
    stats = project!(
        velocity, CellField(mesh), solver, dt, projection, geometry;
        velocity_boundary = boundary, tolerance = 1.0e-11, max_cycles = 1000,
        compatibility = :error,
    )
    @test stats.converged
    @test iszero(stats.removed_rhs_mean)
    divergence!(divergence, velocity, geometry)
    for (index, ref) in pairs(operator.references)
        volume = operator.volumes[index]
        volume > 0 || continue
        @test TwoPhaseAMR._get(divergence, ref) ≈
            -dt * solver.residual[index] / volume atol = 1.0e-11
    end
    for (index, face) in pairs(mesh.boundary_faces)
        if face.axis == 1 && face.side > 0 && geometry.boundary_aperture[index] > 0
            @test velocity.boundary[index] ≈ predictor.boundary[index] -
                dt * projection.weighted_gradient.boundary[index] atol = 1.0e-12
        end
    end
end

@testset "shifted operator supports implicit diffusion" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    operator = EllipticOperator(mesh; reaction = 1, diffusion = 0.02)
    workspace = EllipticWorkspace(operator)
    exact = CellField(mesh)
    initialize!(exact, (x, y) -> 0.4 + sinpi(2x) * cospi(2y))
    rhs = CellField(mesh)
    apply!(rhs, operator, exact, workspace)
    numerical = CellField(mesh)
    solver = EllipticSolver(operator; coarse_size = 8)
    stats = solve!(numerical, solver, rhs; rtol = 2.0e-12, max_cycles = 60)
    @test stats.converged
    @test maximum(abs, test_cell_values(numerical) - test_cell_values(exact)) <= 2.0e-11
end

@testset "pressure projection" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    initial_velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y),
    )
    divergence = CellField(mesh)
    divergence!(divergence, initial_velocity)
    initial_divergence = maximum(abs, test_cell_values(divergence))

    coefficient = CellField(mesh)
    initialize!(coefficient, (x, y) -> x < 0.5 ? 1.0 : 100.0)
    for face_coefficient in (
            ones(length(mesh.faces)),
            harmonic_face_coefficients(coefficient),
        )
        velocity = copy(initial_velocity)
        pressure = CellField(mesh)
        solver = EllipticSolver(
            PoissonOperator(mesh; face_coefficient); coarse_size = 8
        )
        projection_workspace = ProjectionWorkspace(solver)
        stats = project!(
            velocity, pressure, solver, 0.1,
            projection_workspace; tolerance = 1.0e-11
        )
        divergence!(divergence, velocity)
        final_divergence = maximum(abs, test_cell_values(divergence))
        @test stats.converged
        @test stats.initial_linf >= stats.final_linf
        @test stats.residual_scale ≈ 100
        @test stats.final_linf / stats.residual_scale <= 1.0e-11
        @test 0.1 * final_divergence ≈
            stats.final_linf / stats.residual_scale atol =
            8eps(eltype(velocity)) rtol = 0
        @test final_divergence <= 2.0e-11 * initial_divergence
        @test abs(integrate(pressure)) <= 2.0e-12
    end
end

@testset "incremental pressure projection" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    operator = PoissonOperator(
        mesh; face_coefficient = [
            1 + 0.25cospi(2face.center[1]) for face in mesh.faces
        ],
    )
    solver = EllipticSolver(operator; coarse_size = 8)
    workspace = ProjectionWorkspace(solver)
    pressure = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    gradient = FaceVelocity(mesh)
    weighted_gradient!(gradient, pressure, operator, workspace.elliptic)
    dt = 0.1
    velocity = FaceVelocity(mesh)
    velocity.interior .= -dt .* gradient.interior
    stats = project!(
        velocity, pressure, solver, dt, workspace;
        tolerance = 1.0e-13, max_cycles = 100,
    )
    @test stats.converged
    @test maximum(abs, velocity.interior) < 1.0e-10
    @test maximum(abs, test_cell_values(pressure)) < 1.0e-9
end

@testset "dimensionless pressure tolerance validation" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    velocity = FaceVelocity(mesh)
    pressure = CellField(mesh)
    solver = EllipticSolver(PoissonOperator(mesh); coarse_size = 4)
    workspace = ProjectionWorkspace(solver)
    velocity.interior .= 1.0
    original = copy(velocity.interior)
    @test_throws ArgumentError project!(
        velocity, pressure, solver, 0.1, workspace; compatibility = :invalid,
    )
    @test velocity.interior == original
    for invalid in (0.05, Inf, NaN)
        @test_throws ArgumentError project!(
            velocity, pressure, solver, 0.1, workspace; _reference_dt = invalid,
        )
        @test velocity.interior == original
    end
    for invalid in (0.0, -eps(), 1.0, Inf, NaN)
        @test_throws ArgumentError project!(
            velocity, pressure, solver, 0.1, workspace;
            tolerance = invalid,
        )
    end
end

@testset "mixed-level no-permeation projection" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (false, false), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            0.3 + sinpi(2x) * cospi(2y) :
            -0.2 + cospi(2x) * sinpi(2y)
    )
    pressure = CellField(mesh)
    solver = EllipticSolver(
        PoissonOperator(mesh); coarse_size = 8
    )
    workspace = ProjectionWorkspace(solver)
    stats = project!(
        velocity, pressure, solver, 0.05, workspace;
        tolerance = 1.0e-10, max_cycles = 80
    )
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test stats.converged
    @test maximum(abs, test_cell_values(divergence)) <= 1.0e-8
    @test all(iszero, velocity.boundary)
    @test abs(integrate(pressure)) <= 2.0e-11
end
