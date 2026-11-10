isdefined(@__MODULE__, :test_cell_values) || include("test_utils.jl")

import LinearAlgebra: ldiv!
using LinearAlgebra: Diagonal, dot, lu
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
        # Nonzero faces touching the imposed inactive cell anchor the fluid
        # component even with the default :decoupled inactive condition.
        @test !covered.singular
        @test !hierarchy.coarse_augmented
        coarse = only(hierarchy.levels)
        covered_rhs = Float64[cos(index) for index in 1:ncells(mesh)]
        @test coarse.matrix * (replacement \ covered_rhs) ≈ covered_rhs rtol = 1.0e-10
        symbolic = replacement.symbolic
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, covered)
        @test hierarchy.coarse_factorization === replacement
        @test replacement.symbolic === symbolic
    end
end

@testset "multigrid coarse gauges follow coefficient connectivity" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    coefficients = [
        face.axis == 1 && face.center[1] == 0.5 ? 0.0 : 1.0
            for face in mesh.faces
    ]
    operator = PoissonOperator(mesh; face_coefficient = coefficients)
    references = operator.references
    left = findall(
        ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[1] < 0.5, references,
    )
    right = setdiff(collect(eachindex(references)), left)
    split = (left, right)
    merged = (collect(eachindex(references)),)
    function check_coarse_solution(hierarchy, matrix, groups)
        reference = [sin(i) + 0.1cos(2i) for i in eachindex(references)]
        for cells in groups
            mean = sum(operator.volumes[cells] .* reference[cells]) /
                sum(operator.volumes[cells])
            reference[cells] .-= mean
        end
        rhs = matrix * reference
        actual = zeros(length(reference))
        ldiv!(actual, hierarchy, rhs)
        # A single V-cycle is the augmented direct solve on this one-level mesh.
        @test actual ≈ reference atol = 1.0e-10 rtol = 1.0e-10
        @test matrix * actual ≈ rhs atol = 1.0e-10 rtol = 1.0e-10
    end
    hierarchy = TreeGeometricMultigridPreconditioner(
        operator; coarse_size = 1, max_levels = 1,
    )
    @test length(hierarchy.coarse_rhs) == ncells(mesh) + 2
    shifted = EllipticOperator(mesh; reaction = 1)
    @test_throws ArgumentError TwoPhaseAMR._refresh_geometric_multigrid!(
        hierarchy, shifted,
    )
    @test hierarchy.operator === operator
    check_coarse_solution(hierarchy, operator.matrix, split)
    for (weights, groups) in ((ones(length(coefficients)), merged), (coefficients, split))
        TwoPhaseAMR._replace_face_coefficient(operator, weights)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        @test length(hierarchy.coarse_rhs) == ncells(mesh) + length(groups)
        check_coarse_solution(hierarchy, operator.matrix, groups)
    end

    matrix_free = PoissonOperator(
        mesh; face_coefficient = coefficients, assemble_matrix = false,
    )
    matrix_free_hierarchy = TreeGeometricMultigridPreconditioner(
        matrix_free; coarse_size = 1, max_levels = 1,
    )
    check_coarse_solution(matrix_free_hierarchy, operator.matrix, split)
end

@testset "mixed component anchors follow activity and coefficient updates" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    boundary = VelocityBoundaryConditions(right = Outflow())
    split = [
        face.axis == 1 && face.center[1] == 0.5 ? 0.0 : 1.0
            for face in mesh.faces
    ]
    full_volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
    refs = TwoPhaseAMR._cell_references(mesh)
    left = findall(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[1] < 0.5, refs)
    right = setdiff(collect(eachindex(refs)), left)
    for assembled in (false, true)
        active = trues(ncells(mesh))
        active[first(left)] = false
        volumes = ifelse.(active, full_volumes, 0.0)
        operator = EllipticOperator(
            mesh; face_coefficient = split, velocity_boundary = boundary,
            active_cells = active, volumes, inactive_condition = :dirichlet,
            assemble_matrix = assembled,
        )
        solver = EllipticSolver(operator; coarse_size = 1, max_levels = 1)
        @test solver.components !== operator.components
        hierarchy = solver.preconditioner
        closed = EllipticOperator(
            mesh; face_coefficient = split, active_cells = active, volumes,
            inactive_condition = :dirichlet, assemble_matrix = assembled,
        )
        @test_throws ArgumentError TwoPhaseAMR._refresh_geometric_multigrid!(
            hierarchy, closed,
        )
        geometry = GeometryField(mesh, (x, y) -> x - 0.25; lipschitz = 1)
        @test_throws ArgumentError TreeGeometricMultigridPreconditioner(
            operator; geometry, coarse_size = 1,
        )

        function check_state(groups)
            anchors = sort([minimum(cells) for cells in groups])
            @test operator.singular == !isempty(groups)
            @test operator.components.anchors == anchors
            @test hierarchy.coarse_augmented == !isempty(groups)
            @test length(hierarchy.coarse_rhs) == ncells(mesh) + length(groups)
            fresh = EllipticOperator(
                mesh; face_coefficient = copy(operator.face_coefficient),
                velocity_boundary = boundary,
                boundary_face_coefficient = copy(operator.boundary_face_coefficient),
                active_cells = operator.volumes .> 0, volumes = copy(operator.volumes),
                inactive_condition = :dirichlet,
            )
            if assembled
                @test operator.matrix ≈ fresh.matrix atol = 1.0e-12
            end
            exact = [
                operator.volumes[index] > 0 ? 2 + sin(0.4index) : 0.0
                    for index in eachindex(refs)
            ]
            integrated_rhs = fresh.matrix * exact
            rhs = CellField(mesh)
            for index in eachindex(refs)
                volume = operator.volumes[index]
                TwoPhaseAMR._set!(
                    rhs, volume > 0 ? integrated_rhs[index] / volume : 0.0,
                    refs[index],
                )
            end
            expected = copy(exact)
            for cells in groups
                expected[cells] .-= dot(operator.volumes[cells], expected[cells]) /
                    sum(operator.volumes[cells])
            end
            numerical = CellField(mesh; init = -3.0)
            stats = solve!(
                numerical, solver, rhs; compatibility = :error,
                rtol = 0.0, atol = 1.0e-10, max_cycles = 200,
            )
            actual = test_cell_values(numerical)
            @test stats.converged
            @test stats.removed_rhs_component_anchors == anchors
            @test actual ≈ expected atol = 1.0e-8 rtol = 0
            residual = solver.right_hand_side - fresh.matrix * actual
            present = findall(>(0), operator.volumes)
            true_linf = maximum(abs.(residual[present] ./ operator.volumes[present]))
            @test true_linf < 1.0e-9
            @test true_linf ≈ stats.final_linf atol = 1.0e-10 rtol = 0
        end

        # The inactive cell anchors the left component, and the physical
        # outlet independently anchors the right component.
        check_state(())
        # An inactive cell separated by zero-coefficient faces does not
        # anchor the remaining left component.
        isolated = copy(split)
        for (index, face) in pairs(mesh.faces)
            minus = TwoPhaseAMR._cell_index(face.minus, mesh.block_size...)
            plus = TwoPhaseAMR._cell_index(face.plus, mesh.block_size...)
            (minus == first(left) || plus == first(left)) && (isolated[index] = 0)
        end
        TwoPhaseAMR._replace_face_coefficient(operator, isolated)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state((setdiff(left, [first(left)]),))
        TwoPhaseAMR._replace_face_coefficient(operator, split)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state(())
        active .= true
        TwoPhaseAMR._update_elliptic_activity!(operator, active, full_volumes)
        TwoPhaseAMR._replace_face_coefficient(operator, split)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state((left,))

        # Turning off the physical Dirichlet conductance adds a second gauge.
        material = copy(operator.boundary_face_coefficient)
        TwoPhaseAMR._replace_face_coefficient(
            operator, split; boundary_face_coefficient = zeros(length(material)),
        )
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state((left, right))
        # With both components initially ungauged, anchoring only the left
        # changes the gauged component while the global flag stays true.
        active[first(left)] = false
        volumes = ifelse.(active, full_volumes, 0.0)
        TwoPhaseAMR._update_elliptic_activity!(operator, active, volumes)
        TwoPhaseAMR._replace_face_coefficient(operator, split)
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state((right,))
        active .= true
        TwoPhaseAMR._update_elliptic_activity!(operator, active, full_volumes)

        # Restoring the outlet and joining the components removes all gauges.
        TwoPhaseAMR._replace_face_coefficient(
            operator, ones(length(split)); boundary_face_coefficient = material,
        )
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state(())
        # A fully active but open operator is nonsingular: this update used
        # to reject that valid combination of activity and nullspace.
        TwoPhaseAMR._update_elliptic_activity!(operator, active, full_volumes)
        TwoPhaseAMR._replace_face_coefficient(operator, ones(length(split)))
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state(())

        # Changing activity at an open cell updates its boundary diagonal.
        open_cell = first(right)
        for face in mesh.boundary_faces
            if face.axis == 1 && face.side > 0
                open_cell = TwoPhaseAMR._cell_index(face.cell, mesh.block_size...)
                break
            end
        end
        active[open_cell] = false
        volumes = ifelse.(active, full_volumes, 0.0)
        TwoPhaseAMR._update_elliptic_activity!(operator, active, volumes)
        @test iszero(operator.boundary_diagonal[open_cell])
        TwoPhaseAMR._replace_face_coefficient(operator, ones(length(split)))
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        check_state(())
    end
end

@testset "multilevel mixed gauges follow activity changes" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (2, 2), periodic = (false, false),
    )
    boundary = VelocityBoundaryConditions(right = Outflow())
    coefficients = [
        face.axis == 1 && face.center[1] == 0.5 ? 0.0 : 1.0
            for face in mesh.faces
    ]
    full_volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
    refs = TwoPhaseAMR._cell_references(mesh)
    left = findall(ref -> cellcenter(mesh, ref.block, ref.i, ref.j)[1] < 0.5, refs)
    outlet = TwoPhaseAMR._cell_index(
        first(face.cell for face in mesh.boundary_faces if face.axis == 1 && face.side > 0),
        mesh.block_size...,
    )
    for assembled in (false, true)
        active = trues(ncells(mesh))
        operator = EllipticOperator(
            mesh; face_coefficient = coefficients, velocity_boundary = boundary,
            active_cells = active, volumes = full_volumes,
            inactive_condition = :dirichlet, assemble_matrix = assembled,
        )
        hierarchy = TreeGeometricMultigridPreconditioner(
            operator; coarse_size = 1, max_levels = 2,
        )
        @test length(hierarchy.levels) == 2
        function compare_fresh()
            fresh = TreeGeometricMultigridPreconditioner(
                operator; coarse_size = 1, max_levels = 2,
            )
            @test hierarchy.coarse_augmented == fresh.coarse_augmented
            @test length(hierarchy.operators) == length(fresh.operators) == 2
            for (old, new) in zip(hierarchy.operators, fresh.operators)
                @test old.constraints == new.constraints
                @test old.components.anchors == new.components.anchors
                @test old.boundary_diagonal ≈ new.boundary_diagonal atol = 1.0e-12
            end
            rhs = [sin(0.2i) + cos(0.3i) for i in eachindex(full_volumes)]
            actual, expected = similar(rhs), similar(rhs)
            ldiv!(actual, hierarchy, rhs)
            ldiv!(expected, fresh, rhs)
            @test actual ≈ expected atol = 1.0e-10 rtol = 1.0e-10
        end
        compare_fresh()
        for index in (first(left), outlet)
            active[index] = false
            volumes = ifelse.(active, full_volumes, 0.0)
            TwoPhaseAMR._update_elliptic_activity!(operator, active, volumes)
            TwoPhaseAMR._replace_face_coefficient(operator, coefficients)
            TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
            compare_fresh()
            active[index] = true
            TwoPhaseAMR._update_elliptic_activity!(operator, active, full_volumes)
            TwoPhaseAMR._replace_face_coefficient(operator, coefficients)
            TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
            compare_fresh()
        end
    end
end

@testset "coarse augmentation follows coarse connectivity on hierarchy reuse" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (2, 2), periodic = (false, false),
    )
    boundary = VelocityBoundaryConditions(right = Outflow())
    coefficients(position) = [
        face.axis == 1 && face.center[1] == position ? 0.0 : 1.0
            for face in mesh.faces
    ]
    fine_split = PoissonOperator(
        mesh; face_coefficient = coefficients(0.375), velocity_boundary = boundary,
    )
    old = TreeGeometricMultigridPreconditioner(
        fine_split; coarse_size = 1, max_levels = 2,
    )
    @test fine_split.singular
    @test length(old.levels) == 2
    @test !old.coarse_augmented
    @test isempty(last(old.levels).component_volumes)

    coarse_split = PoissonOperator(
        mesh; face_coefficient = coefficients(0.5), velocity_boundary = boundary,
    )
    reused = TreeGeometricMultigridPreconditioner(
        coarse_split; coarse_size = 1, max_levels = 2, reuse = old,
    )
    fresh = TreeGeometricMultigridPreconditioner(
        coarse_split; coarse_size = 1, max_levels = 2,
    )
    # The finest operator is the new input; only the matching coarse level
    # is reused from the previous hierarchy.
    @test reused.reused_levels == length(old.levels) - 1
    @test last(reused.levels) === last(old.levels)
    @test reused.coarse_augmented
    @test length(last(reused.levels).component_volumes) == 1
    @test length(reused.coarse_rhs) == length(last(reused.levels).volumes) + 1
    rhs = [sin(0.2index) + cos(0.7index) for index in eachindex(coarse_split.volumes)]
    TwoPhaseAMR._multigrid_project_component_rhs!(rhs, first(reused.levels))
    reused_result, fresh_result = similar(rhs), similar(rhs)
    ldiv!(reused_result, reused, rhs)
    ldiv!(fresh_result, fresh, rhs)
    @test reused_result ≈ fresh_result atol = 1.0e-10 rtol = 1.0e-10
end

@testset "elliptic component cache tracks topology identity and generation" begin
    first_mesh = BlockTreeMesh(block_size = (4, 4), periodic = (true, false))
    second_mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, true))
    first = PoissonOperator(first_mesh)
    second = PoissonOperator(second_mesh)
    components = TwoPhaseAMR.EllipticComponentWorkspace(first)
    first_incidence = copy(components.incidence)
    @test length(first.face_coefficient) == length(second.face_coefficient)

    TwoPhaseAMR._refresh_elliptic_components!(components, second)
    reference = TwoPhaseAMR.EllipticComponentWorkspace(second)
    @test components.mesh_identity === second_mesh.topology_cache.identity
    @test components.incidence == reference.incidence
    @test components.incidence != first_incidence
    @test components.labels == reference.labels

    previous_incidence = copy(components.incidence)
    reverse!(second_mesh.faces)
    TwoPhaseAMR._invalidate_mesh_topology_cache!(second_mesh)
    TwoPhaseAMR._refresh_elliptic_components!(components, second)
    reference = TwoPhaseAMR.EllipticComponentWorkspace(second)
    @test components.generation == second_mesh.topology_cache.generation
    @test components.incidence == reference.incidence
    @test components.incidence != previous_incidence
    @test components.labels == reference.labels
end

@testset "elliptic cache retains type when nullspace changes" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    pressure = EllipticSolver(PoissonOperator(mesh); preconditioner = nothing)
    shifted = EllipticSolver(EllipticOperator(mesh; reaction = 1); preconditioner = nothing)
    @test pressure.operator.singular
    @test !shifted.operator.singular
    @test typeof(pressure) === typeof(shifted)
    cache = Ref(pressure)
    cache[] = shifted
    @test cache[] === shifted
    cache[] = pressure
    @test cache[] === pressure
end

@testset "multigrid component merge preserves symmetric positive action" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), periodic = (false, false),
    )
    # A band cuts the fine mesh but lies inside a cell on the 4x4 coarse mesh.
    geometry = GeometryField(mesh, (x, y) -> abs(y - 0.35) - 0.04; lipschitz = 1)
    for matrix_free in (false, true), use_geometry in (false, true)
        operator = PoissonOperator(geometry; assemble_matrix = !matrix_free)
        hierarchy = TreeGeometricMultigridPreconditioner(
            operator; geometry = use_geometry ? geometry : nothing, coarse_size = 16,
        )
        @test length(first(hierarchy.levels).component_volumes) == 2
        @test length(last(hierarchy.levels).component_volumes) == 1
        volumes = operator.volumes
        refs = operator.references
        groups = (
            findall(i -> volumes[i] > 0 && cellcenter(mesh, refs[i].block, refs[i].i, refs[i].j)[2] < 0.35, eachindex(volumes)),
            findall(i -> volumes[i] > 0 && cellcenter(mesh, refs[i].block, refs[i].i, refs[i].j)[2] > 0.35, eachindex(volumes)),
        )
        x = zeros(length(volumes))
        y = similar(x)
        fill!(y, 0)
        for cells in groups
            for cell in cells
                x[cell] = sin(0.37cell)
                y[cell] = cos(0.61cell)
            end
            x[cells] .-= (sum(x[cells]) / sum(volumes[cells])) .* volumes[cells]
            y[cells] .-= (sum(y[cells]) / sum(volumes[cells])) .* volumes[cells]
        end
        bx, by = similar(x), similar(y)
        ldiv!(bx, hierarchy, x)
        ldiv!(by, hierarchy, y)
        @test dot(x, by) ≈ dot(bx, y) atol = 1.0e-10 rtol = 1.0e-10
        @test dot(x, bx) > 0
        @test dot(y, by) > 0
        TwoPhaseAMR._refresh_geometric_multigrid!(hierarchy, operator)
        refreshed = similar(x)
        ldiv!(refreshed, hierarchy, x)
        @test refreshed ≈ bx atol = 1.0e-10 rtol = 1.0e-10
    end
end

import IncompleteLU
using Krylov

# A deterministic external Krylov method for the stopping-floor regression.
# Its returned iterate has a huge nullspace part but has not reduced the residual.
struct _InflatedIterateWorkspace
    x::Vector{Float64}
end

Krylov.solution(workspace::_InflatedIterateWorkspace) = workspace.x
Krylov.iteration_count(::_InflatedIterateWorkspace) = 1

function TwoPhaseAMR._run_krylov!(
        workspace::_InflatedIterateWorkspace, matrix, right_hand_side,
        initial_solution, preconditioner, target, volumes, max_iterations, options,
    )
    fill!(workspace.x, 1.0e15)
    return workspace
end

# A large nullspace drift can accompany a modest residual decrease.
struct _PartiallyInflatedWorkspace
    x::Vector{Float64}
end

Krylov.solution(workspace::_PartiallyInflatedWorkspace) = workspace.x
Krylov.iteration_count(::_PartiallyInflatedWorkspace) = 1

function TwoPhaseAMR._run_krylov!(
        workspace::_PartiallyInflatedWorkspace, matrix, right_hand_side,
        initial_solution, preconditioner, target, volumes, max_iterations, options,
    )
    workspace.x .= 1.0e15
    workspace.x[1] += 0.5
    return workspace
end

mutable struct _ControlledFloorWorkspace{T}
    x::Vector{T}
    offset::T
    corrections::Vector{T}
    pass::Int
end

Krylov.solution(workspace::_ControlledFloorWorkspace) = workspace.x
Krylov.iteration_count(::_ControlledFloorWorkspace) = 1

function TwoPhaseAMR._run_krylov!(
        workspace::_ControlledFloorWorkspace, matrix, right_hand_side,
        initial_solution, preconditioner, target, volumes, max_iterations, options,
    )
    workspace.pass += 1
    correction = workspace.corrections[min(workspace.pass, length(workspace.corrections))]
    workspace.x[1] = workspace.offset + correction
    workspace.x[2] = workspace.offset
    return workspace
end

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
    unsupported = Krylov.CrWorkspace(matrix, right_hand_side)
    @test TwoPhaseAMR._krylov_method(unsupported) === :CrWorkspace
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
        solver.stats.compatible = false
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
        @test stats.compatible
        @test stats.removed_rhs_mean === 0.25
        @test stats.method === :cg
        @test stats.preconditioner === :none
    end
end

@testset "final iterate cannot inflate the original stopping target" begin
    # Issue #97: the final iterate can grow in a nullspace direction without
    # reducing b-A*x. A floor recomputed from |A|*|x| would hide this residual.
    matrix = sparse([1, 2, 1, 2], [1, 1, 2, 2], [1.0, -1.0, -1.0, 1.0])
    volumes = ones(2)
    for relative_to_rhs in (false, true)
        workspace = _InflatedIterateWorkspace(zeros(2))
        solver = (;
            action = matrix, krylov = workspace, preconditioner = nothing,
            krylov_options = (;), solution = zeros(2),
            right_hand_side = [1.0, -1.0], residual = zeros(2), product = zeros(2),
            stats = TwoPhaseAMR.LinearSolverStats(Float64),
        )
        initial_floor = TwoPhaseAMR._roundoff_residual_floor!(
            matrix, solver.solution, solver.right_hand_side, volumes, solver.product,
        )
        copied = zeros(2)
        stats = TwoPhaseAMR._solve_krylov_with_stats!(
            () -> 0.0, solver, matrix, volumes, 0.1, 1; relative_to_rhs,
        ) do values
            copyto!(copied, values)
        end
        final_floor = TwoPhaseAMR._roundoff_residual_floor!(
            matrix, copied, solver.right_hand_side, volumes, solver.product,
        )
        actual = maximum(abs, solver.right_hand_side - matrix * copied)
        @test initial_floor < 0.1
        @test final_floor > actual
        @test copied == fill(1.0e15, 2)
        @test stats.cycles == 1
        @test stats.initial_linf == 1.0
        @test stats.target_linf == 0.1
        @test stats.final_linf == actual == 1.0
        @test actual > stats.target_linf
        @test !stats.converged
    end
    for relative_to_rhs in (false, true)
        workspace = _PartiallyInflatedWorkspace(zeros(2))
        solver = (;
            action = matrix, krylov = workspace, preconditioner = nothing,
            krylov_options = (;), solution = zeros(2),
            right_hand_side = [1.0, -1.0], residual = zeros(2), product = zeros(2),
            stats = TwoPhaseAMR.LinearSolverStats(Float64),
        )
        stats = TwoPhaseAMR._solve_krylov_with_stats!(
            values -> nothing, () -> 0.0, solver, matrix, volumes, 0.1, 1;
            relative_to_rhs,
        )
        @test stats.initial_linf == 1.0
        @test stats.final_linf == 0.5
        @test stats.target_linf == 0.1
        @test !stats.converged
    end
end

@testset "guarded iterate roundoff floor" begin
    function controlled_solve(T, offset, corrections; relative_to_rhs, initial = zeros(T, 2))
        matrix = sparse([1, 2, 1, 2], [1, 1, 2, 2], T[1, -1, -1, 1])
        volumes = ones(T, 2)
        workspace = _ControlledFloorWorkspace(zeros(T, 2), offset, corrections, 0)
        solver = (;
            action = matrix, krylov = workspace, preconditioner = nothing,
            krylov_options = (;), solution = copy(initial),
            right_hand_side = T[1, -1], residual = zeros(T, 2), product = zeros(T, 2),
            stats = TwoPhaseAMR.LinearSolverStats(T),
        )
        stats = TwoPhaseAMR._solve_krylov_with_stats!(
            values -> nothing, () -> zero(T), solver, matrix, volumes, T(1.0e-4),
            length(corrections); relative_to_rhs,
        )
        floor = TwoPhaseAMR._roundoff_residual_floor!(
            matrix, solver.solution, solver.right_hand_side, volumes, solver.product,
        )
        return (; stats, workspace, floor)
    end

    for T in (Float32, Float64), relative_to_rhs in (false, true)
        precision = T === Float32 ? 23 : 52
        small_offset = ldexp(one(T), precision - 16)
        large_offset = ldexp(one(T), precision - 12)
        correction = T(511) / T(512)

        accepted = controlled_solve(T, small_offset, T[correction]; relative_to_rhs)
        @test accepted.workspace.pass == accepted.stats.cycles == 1
        @test accepted.stats.initial_linf == one(T)
        @test accepted.stats.final_linf == T(1) / T(512)
        @test accepted.floor < accepted.stats.initial_linf / 100
        @test accepted.stats.final_linf > T(1.0e-4)
        @test accepted.stats.target_linf == accepted.floor
        @test accepted.stats.converged

        rejected = controlled_solve(T, large_offset, T[correction]; relative_to_rhs)
        @test rejected.workspace.pass == rejected.stats.cycles == 1
        @test rejected.stats.final_linf == T(1) / T(512)
        @test rejected.floor > rejected.stats.initial_linf / 100
        @test rejected.stats.final_linf < rejected.floor
        @test rejected.stats.target_linf == T(1.0e-4)
        @test !rejected.stats.converged

        # A temporary residual increase at roundoff must not prevent a later
        # pass from satisfying the guarded floor.
        corrections = T[509 / 512, 508 / 512, 511 / 512]
        stagnated = controlled_solve(T, small_offset, corrections; relative_to_rhs)
        @test stagnated.workspace.pass == stagnated.stats.cycles == 3
        @test stagnated.stats.converged
        @test stagnated.stats.final_linf == T(1) / T(512)
        @test stagnated.stats.target_linf == stagnated.floor

        initial = T[small_offset + correction, small_offset]
        warm = controlled_solve(T, small_offset, T[correction]; relative_to_rhs, initial)
        @test warm.workspace.pass == warm.stats.cycles == 0
        @test warm.stats.converged
        @test warm.stats.initial_linf == warm.stats.final_linf == T(1) / T(512)
        @test warm.stats.target_linf == warm.floor
    end
end

@testset "roundoff-limited heterogeneous Poisson solve" begin
    mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (4, 4))
    # A 1000:1 density contrast and one small open volume model the
    # roundoff amplification in a tiny cut cell.
    coefficient = [
        face.center[1] < 0.5 ? 1.0 : 1.0e-3 for face in mesh.faces
    ]
    volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
    volumes[1] *= 1.0e-3
    operator = EllipticOperator(mesh; face_coefficient = coefficient, volumes)
    rhs = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    initial_floor = TwoPhaseAMR._roundoff_residual_floor!(
        operator.matrix, zeros(ncells(mesh)),
        test_cell_values(rhs) .* operator.volumes, operator.volumes,
        zeros(ncells(mesh)),
    )
    pressure = CellField(mesh)
    solver = EllipticSolver(operator)
    stats = solve!(pressure, solver, rhs; rtol = 1.0e-15, max_cycles = 500)
    final_floor = TwoPhaseAMR._roundoff_residual_floor!(
        operator.matrix, test_cell_values(pressure), solver.right_hand_side,
        operator.volumes, zeros(ncells(mesh)),
    )
    @test stats.converged
    @test stats.final_linf <= stats.target_linf
    @test stats.final_linf > initial_floor
    @test stats.target_linf > initial_floor
    @test stats.target_linf ≈ final_floor rtol = 128eps()
    @test stats.target_linf < stats.initial_linf / 100
    @test stats.final_linf < 1.0e-8
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
