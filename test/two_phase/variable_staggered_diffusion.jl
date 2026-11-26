using Test, TwoPhaseAMR, LinearAlgebra, SparseArrays, Krylov
@testset "Factorized variable-stress action matches assembled operator" begin
    for topology in (:uniform, :mixed, :adapted)
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) -> topology == :mixed &&
                (
                key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0)
            ),
        )
        marker = CellField(mesh, (x, y) -> sinpi(2x) * sinpi(2y))
        if topology == :adapted
            adaptation = AMRWorkspace(mesh, marker)
            adapt!(adaptation, [:refine]; step = 1, force = true)
            labels = [
                block.key.level == 1 && block.key.i == 0 &&
                    block.key.j == 0 ? :refine : :keep
                    for block in mesh.blocks
            ]
            adapt!(adaptation, labels; step = 2, force = true)
        end
        viscosity = CellField(mesh, (x, y) -> exp(0.4sinpi(2x)))
        operator = StaggeredDiffusionOperator(mesh; viscosity)
        density = [
            1 + 0.2cos(index) for index in eachindex(operator.volumes)
        ]
        inertia = density .* operator.volumes
        dt = 0.03
        action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
            operator, inertia, dt,
        )
        matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
        x = [sin(0.7index) for index in eachindex(operator.volumes)]
        y = [cos(0.3index) for index in eachindex(operator.volumes)]
        action_x = similar(x)
        action_y = similar(y)
        mul!(action_x, action, x)
        @test action_x ≈ matrix * x rtol = 128eps() atol = 128eps()
        mul!(action_y, action, y)
        symmetry_scale = max(
            norm(x) * norm(action_y), norm(action_x) * norm(y), 1.0,
        )
        @test abs(dot(x, action_y) - dot(action_x, y)) <=
            256eps() * symmetry_scale
        @test dot(x, action_x) > 0
    end
end

@testset "Reused stress construction matches a fresh operator across AMR" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 3,
        refine = (key, origin, extent) -> key.level < 2,
    )
    viscosity_function = (x, y) -> 1 + 0.2x + 0.1y
    viscosity = CellField(mesh, viscosity_function)
    cache = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    original = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, construction_cache = cache,
    )

    adaptation = AMRWorkspace(mesh, viscosity)
    labels = [
        block.key == BlockKey(2, 0, 0) ? :refine : :keep
            for block in mesh.blocks
    ]
    stats = adapt!(adaptation, labels; step = 1, force = true)
    @test stats.changed
    initialize!(viscosity, viscosity_function)

    cached = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, construction_cache = cache, reuse = original,
    )
    fresh = StaggeredDiffusionOperator(mesh; viscosity)
    @test cached.strain == fresh.strain
    @test cached.stress_interpolation == fresh.stress_interpolation
    @test cached.stress_measures == fresh.stress_measures
    @test cached.stress_weights == fresh.stress_weights
    @test cached.stiffness == fresh.stiffness

    repeated = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, construction_cache = cache, reuse = cached,
    )
    @test repeated.strain == fresh.strain
    @test repeated.stiffness == fresh.stiffness
end

@testset "CG requires symmetric variable-diffusion V-cycle sweeps" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), periodic = (true, true),
    )
    viscosity = CellField(mesh, (x, y) -> 0.1 * (1 + 0.2sinpi(2x)))
    operator = StaggeredDiffusionOperator(mesh; viscosity, boundary = FreeSlip())
    density = ones(length(operator.volumes))
    dt = 0.01
    symmetric = MultigridOptions(coarse_size = 1, max_levels = 2)
    asymmetric = MultigridOptions(
        coarse_size = 1, max_levels = 2, pre_sweeps = 1, post_sweeps = 3,
    )
    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = symmetric,
    )
    @test solver.krylov isa Krylov.CgWorkspace
    @test length(solver.preconditioner.levels) == 2
    @test solver.preconditioner.pre_sweeps == solver.preconditioner.post_sweeps == 2
    @test_throws ArgumentError TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = asymmetric,
    )
    @test_throws ArgumentError TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; krylov = Krylov.CgWorkspace, multigrid = asymmetric,
    )

    nonsymmetric = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; krylov = Krylov.BicgstabWorkspace, multigrid = asymmetric,
    )
    hierarchy = nonsymmetric.preconditioner
    @test hierarchy isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    workspace = Krylov.CgWorkspace(operator.stiffness, zeros(length(operator.volumes)))
    for krylov in (
            Krylov.CgWorkspace, workspace, (matrix, rhs) -> Krylov.CgWorkspace(matrix, rhs),
        )
        @test_throws ArgumentError TwoPhaseAMR.VariableStaggeredDiffusionSolver(
            operator, density, dt; krylov, preconditioner = hierarchy,
        )
    end
    @test_throws ArgumentError TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; preconditioner = nothing,
        krylov_options = (; M = hierarchy, ldiv = true),
    )
    # Reusing a CG solver must check its actual workspace, not only the
    # nonsymmetric factory passed for the newly requested solver.
    @test_throws ArgumentError TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; krylov = Krylov.BicgstabWorkspace,
        preconditioner = hierarchy, reuse_solver = solver,
    )

    source = FaceVelocity(
        mesh, (x, y, time, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    destination = FaceVelocity(mesh)
    @test implicit_midpoint_diffusion!(
        destination, source, operator, dt, nonsymmetric;
        density, tolerance = 1.0e-9, max_cycles = 300,
    ) === destination
    @test nonsymmetric.stats.converged
    @test nonsymmetric.stats.method === :bicgstab

    # The solve-time guard also covers a hierarchy mutated after construction,
    # even when an already-converged RHS would otherwise skip all iterations.
    solver.preconditioner.post_sweeps = 3
    @test_throws ArgumentError TwoPhaseAMR._solve_variable_diffusion!(
        FaceVelocity(mesh), solver, operator; tolerance = 1.0e-9, max_cycles = 0,
    )
end

@testset "Rediscretized variable diffusion on pruned trees" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 3,
        periodic = (true, false),
        refine = (key, origin, extent) ->
        key.level == 0 ||
            (key.level < 3 && origin[1] < 0.5 && origin[2] < 0.5),
    )
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.55; lipschitz = hypot(1, 0.2),
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (1.0e-2, 1.0e-5), boundary = FreeSlip(),
    )
    density = TwoPhaseAMR.mixture_density(interface, (1.0, 1.0e-3))
    dt = 0.01
    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt;
        multigrid = MultigridOptions(coarse_size = 64),
    )
    preconditioner = solver.preconditioner
    @test preconditioner isa
        TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test length(preconditioner.levels) > 2
    @test solver.matrix isa SparseMatrixCSC{Float64, Int}
    @test preconditioner.levels[1].matrix === solver.matrix
    @test all(
        level -> level.matrix isa SparseMatrixCSC{Float64, Int},
        preconditioner.levels,
    )
    @test all(
        diff(length.(getproperty.(preconditioner.operators, :volumes))) .< 0,
    )

    for index in 1:(length(preconditioner.levels) - 1)
        fine_operator = preconditioner.operators[index]
        coarse_operator = preconditioner.operators[index + 1]
        prolongation = preconditioner.levels[index].prolongation
        @test transpose(prolongation) * fine_operator.volumes ≈
            coarse_operator.volumes rtol = 64eps()

        expected_log_viscosity = similar(coarse_operator.cell_work)
        TwoPhaseAMR._restrict_cell_coefficient!(
            expected_log_viscosity, fine_operator.cell_work,
            preconditioner.cell_restrictions[index],
        )
        @test coarse_operator.cell_work ≈ expected_log_viscosity rtol = 16eps()
        inertia = preconditioner.levels[index + 1].volumes
        reference = Diagonal(inertia) +
            (dt / 2) * coarse_operator.stiffness
        @test preconditioner.levels[index + 1].matrix ≈
            reference rtol = 64eps()
    end

    source = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    destination = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(
        destination, source, operator, dt, solver;
        density, tolerance = 1.0e-9, max_cycles = 300,
    )
    @test solver.stats.converged
    @test solver.stats.residual_scale ≈ maximum(
        abs.(solver.right_hand_side ./ solver.inertia),
    )
    @test solver.stats.final_linf <= solver.stats.target_linf

    moved = InterfaceField(
        mesh, (x, y) -> x - 0.15y - 0.43;
        lipschitz = hypot(1, 0.15),
    )
    coarse_factorization = preconditioner.coarse_factorization
    TwoPhaseAMR._update_dynamic_viscosity!(
        operator, moved, (2.0e-2, 2.0e-5),
    )
    TwoPhaseAMR._update_variable_diffusion_solver!(
        solver, operator, reverse(density), 1.2dt,
    )
    @test solver.preconditioner === preconditioner
    @test preconditioner.coarse_factorization === coarse_factorization
    coarse_matrix = last(preconditioner.levels).matrix
    coarse_rhs = [sin(0.3index) for index in axes(coarse_matrix, 1)]
    @test coarse_matrix * (coarse_factorization \ coarse_rhs) ≈
        coarse_rhs rtol = 1.0e-10 atol = 1.0e-10
    @test preconditioner.levels[1].matrix === solver.matrix
    for (level, level_operator) in zip(
            preconditioner.levels, preconditioner.operators,
        )
        reference = Diagonal(level.volumes) +
            (1.2dt / 2) * level_operator.stiffness
        @test level.matrix ≈ reference rtol = 64eps()
    end
    reused_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt;
        multigrid = MultigridOptions(coarse_size = 64),
        reuse_preconditioner = preconditioner,
        reuse_solver = solver,
    )
    @test reused_solver === solver
    reused_hierarchy = solver.preconditioner
    @test 0 < reused_hierarchy.reused_levels < length(reused_hierarchy.levels)
    @test reused_hierarchy.levels[1].matrix === solver.matrix
    for (level, level_operator) in zip(
            reused_hierarchy.levels, reused_hierarchy.operators,
        )
        @test level.matrix isa SparseMatrixCSC{Float64, Int}
        reference = Diagonal(level.volumes) +
            (dt / 2) * level_operator.stiffness
        @test level.matrix ≈ reference rtol = 64eps()
    end
    uniform_mesh = BlockTreeMesh(block_size = (8, 8))
    uniform_interface = InterfaceField(
        uniform_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    uniform_operator = StaggeredDiffusionOperator(
        uniform_interface; viscosity = (1.0e-2, 1.0e-5),
    )
    within_block = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        uniform_operator, ones(length(uniform_mesh.faces)), dt;
        multigrid = MultigridOptions(coarse_size = 16),
    )
    @test within_block.preconditioner isa
        TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test within_block.krylov isa Krylov.CgWorkspace
    @test length(within_block.preconditioner.levels) == 1
end

@testset "Variable diffusion refactorization with Float32 coefficients" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    viscosity = CellField(mesh, Float32; init = 0.1f0)
    operator = StaggeredDiffusionOperator(
        mesh; viscosity, boundary = FreeSlip(), T = Float32,
    )
    density = ones(Float32, length(mesh.faces))
    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, 0.01f0;
        multigrid = MultigridOptions(coarse_size = 16),
    )
    preconditioner = solver.preconditioner
    factorization = preconditioner.coarse_factorization
    TwoPhaseAMR._update_variable_diffusion_solver!(
        solver, operator, 2 .* density, 0.02f0,
    )
    @test preconditioner.coarse_factorization === factorization
    coarse = last(preconditioner.levels).matrix
    rhs = [sin(0.3index) for index in axes(coarse, 1)]
    @test coarse * (factorization \ rhs) ≈ rhs rtol = 1.0e-10 atol = 1.0e-10
end

@testset "variable diffusion base-mesh multigrid" begin
    for periodic in ((true, true), (true, false))
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4),
            periodic = periodic, max_level = 1,
        )
        viscosity = CellField(mesh, (x, y) -> exp(0.2sinpi(2x)))
        operator = StaggeredDiffusionOperator(mesh; viscosity, boundary = FreeSlip())
        density = [1 + 0.2cospi(2face.center[2]) for face in mesh.faces]
        solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
            operator, density, 0.01;
            multigrid = MultigridOptions(coarse_size = 16),
        )
        hierarchy = solver.preconditioner
        @test [op.mesh.base_mesh for op in hierarchy.operators] ==
            [(16, 16), (8, 8), (4, 4)]
        @test all(op -> op.mesh.periodic == periodic, hierarchy.operators)
        for index in 1:2
            fine = hierarchy.operators[index]
            coarse = hierarchy.operators[index + 1]
            expected_density = similar(hierarchy.densities[index + 1])
            TwoPhaseAMR._restrict_face_coefficients!(
                expected_density, hierarchy.densities[index],
                hierarchy.density_restrictions[index],
            )
            @test hierarchy.densities[index + 1] ≈ expected_density rtol = 64eps()
            expected_log_viscosity = similar(coarse.cell_work)
            TwoPhaseAMR._restrict_cell_coefficient!(
                expected_log_viscosity, fine.cell_work,
                hierarchy.cell_restrictions[index],
            )
            @test coarse.cell_work ≈ expected_log_viscosity rtol = 64eps()
        end
        source = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        )
        destination = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(
            destination, source, operator, 0.01, solver;
            density, tolerance = 1.0e-9, max_cycles = 300,
        )
        @test solver.stats.converged
        if periodic == (true, true)
            old_mesh = last(hierarchy.operators).mesh
            adaptation = AMRWorkspace(mesh, viscosity)
            labels = [
                block.key == BlockKey(0, 0, 0) ? :refine : :keep
                    for block in mesh.blocks
            ]
            @test adapt!(adaptation, labels; step = 1, force = true).changed
            initialize!(viscosity, (x, y) -> exp(0.2sinpi(2x)))
            new_operator = StaggeredDiffusionOperator(
                mesh; viscosity, boundary = FreeSlip(),
            )
            new_density = [1 + 0.2cospi(2face.center[2]) for face in mesh.faces]
            rebuilt = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
                new_operator, new_density, 0.01;
                multigrid = MultigridOptions(coarse_size = 16),
                reuse_preconditioner = hierarchy,
            )
            @test rebuilt.preconditioner.reused_levels > 0
            @test last(rebuilt.preconditioner.operators).mesh === old_mesh
        end
    end
    # The coupled face count is below 2coarse_size at physical walls even
    # though the underlying base mesh still has more than coarse_size cells.
    wall_mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), periodic = (false, false),
    )
    wall_viscosity = CellField(wall_mesh, (x, y) -> 1.0)
    wall_operator = StaggeredDiffusionOperator(
        wall_mesh; viscosity = wall_viscosity, boundary = FreeSlip(),
    )
    @test length(wall_mesh.faces) < 2 * 60 < 2ncells(wall_mesh)
    wall_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        wall_operator, ones(length(wall_mesh.faces)), 0.01;
        multigrid = MultigridOptions(coarse_size = 60),
    )
    @test [op.mesh.base_mesh for op in wall_solver.preconditioner.operators] ==
        [(8, 8), (4, 4)]

    periodic_mesh = BlockTreeMesh(
        base_mesh = (4, 4), block_size = (2, 2), periodic = (true, true),
    )
    periodic_viscosity = CellField(periodic_mesh, (x, y) -> 1.0)
    periodic_operator = StaggeredDiffusionOperator(
        periodic_mesh; viscosity = periodic_viscosity, boundary = FreeSlip(),
    )
    periodic_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        periodic_operator, ones(length(periodic_mesh.faces)), 0.01;
        multigrid = MultigridOptions(coarse_size = 1),
    )
    @test length(periodic_solver.preconditioner.levels) == 1
end

@testset "Phase-viscosity averaging rules (#113)" begin
    for (rule, cell_value, row_value) in (
                (
                    HarmonicViscosityAveraging(),
                    (c, liquid, gas) -> 1 / (c / liquid + (1 - c) / gas),
                    inverse -> iszero(inverse) ? 1.0 : inv(inverse),
                ),
                (
                    GeometricViscosityAveraging(),
                    (c, liquid, gas) -> liquid^c * gas^(1 - c),
                    _ -> nothing,
                ),
            ), kind in 1:2
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (key, origin, extent) -> kind == 2 &&
                (
                key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0)
            )
        )
        interface = InterfaceField(
            mesh, (x, y) -> x + 0.3y - 0.57; lipschitz = hypot(1, 0.3)
        )
        phases = (1.0e-2, 1.0e-5)
        viscosity = dynamic_viscosity(
            interface; viscosity = phases, averaging = rule,
        )
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(interface.fraction, ref)
            expected = cell_value(c, phases[1], phases[2])
            @test TwoPhaseAMR._get(viscosity, ref) ≈ expected rtol = 16eps()
        end
        from_interface = StaggeredDiffusionOperator(
            interface; viscosity = phases, averaging = rule,
        )
        stiffness_storage = from_interface.stiffness
        TwoPhaseAMR._update_dynamic_viscosity!(
            from_interface, interface, phases
        )
        @test from_interface.stiffness === stiffness_storage
        solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
            from_interface, ones(length(from_interface.volumes)), 0.01;
            preconditioner = nothing,
        )
        @test solver.matrix.colptr == from_interface.stiffness.colptr
        @test rowvals(solver.matrix) == rowvals(from_interface.stiffness)
        matrix_storage = solver.matrix.nzval
        TwoPhaseAMR._update_variable_diffusion_solver!(
            solver, from_interface,
            ones(length(from_interface.volumes)), 0.02,
        )
        @test solver.matrix.nzval === matrix_storage
        from_field = StaggeredDiffusionOperator(
            mesh; viscosity, averaging = rule,
        )
        @test from_interface.stiffness ≈ from_field.stiffness rtol = 16eps()
        cells = [
            TwoPhaseAMR._get(viscosity, ref)
                for ref in TwoPhaseAMR._cell_references(mesh)
        ]
        observed = from_interface.stress_weights ./
            from_interface.stress_measures
        if rule isa HarmonicViscosityAveraging
            # The series conductance of the cell quarters the row spans.
            row_inverse = from_interface.stress_interpolation *
                (1 ./ cells)
            expected = row_value.(row_inverse)
        else
            # The thesis' geometric mixture of the row's cell quarters.
            logs = [
                TwoPhaseAMR._get(interface.fraction, ref)
                    for ref in TwoPhaseAMR._cell_references(mesh)
            ]
            row_fraction = from_interface.stress_interpolation * logs
            expected = phases[1] .^ row_fraction .* phases[2] .^
                (1 .- row_fraction)
        end
        @test observed ≈ expected rtol = 32eps()

        # A VOF update refreshes the cell mixture through the same rule; the
        # refreshed operator equals one built from the new mixture field.
        moved = InterfaceField(
            mesh, (x, y) -> x + 0.3y - 0.51; lipschitz = hypot(1, 0.3),
        )
        TwoPhaseAMR._update_dynamic_viscosity!(from_interface, moved, phases)
        refreshed = dynamic_viscosity(
            moved; viscosity = phases, averaging = rule,
        )
        @test from_interface.cell_work ≈ [
            log(TwoPhaseAMR._get(refreshed, ref))
                for ref in TwoPhaseAMR._cell_references(mesh)
        ] rtol = 64eps()
        rebuilt = StaggeredDiffusionOperator(
            mesh; viscosity = refreshed, averaging = rule,
        )
        @test from_interface.stiffness ≈ rebuilt.stiffness rtol = 64eps()
    end
end

"""Shifted-solve error for the 1-D shear `mu(y) u'(y) = -cos(2 pi y)`.

The tangential traction is continuous across the interface at `y = 1/2`, so
the exact velocity is smooth in traction but has a derivative jump of the
viscosity ratio there. A stress vertex that spans the jump must carry the
series conductance of its two phases for the discrete flux to reproduce the
traction; with any other average the local flux defect does not vanish under
refinement.
"""
function _aligned_jump_shear_error(
        n; liquid = 2.0, gas = 1.0,
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    )
    mesh = BlockTreeMesh(block_size = (4, n), periodic = (true, true))
    viscosity = CellField(mesh, (x, y) -> y < 0.5 ? liquid : gas)
    operator = StaggeredDiffusionOperator(mesh; viscosity, averaging)
    volumes = operator.volumes
    exact_value(y) = -sinpi(2y) / (2pi * (y < 0.5 ? liquid : gas))
    # d/dy(mu u') = 2 pi sin(2 pi y)
    divergence(y) = 2pi * sinpi(2y)
    exact = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? exact_value(y) : 0.0,
    )
    shift = 1.0
    rhs = similar(volumes)
    for (index, face) in pairs(mesh.faces)
        rhs[index] = volumes[index] * (
            exact.interior[index] -
                (face.axis == 1 ? shift * divergence(face.center[2]) : 0.0)
        )
    end
    matrix = spdiagm(0 => volumes) + shift * operator.stiffness
    solution = matrix \ rhs
    error = sqrt(
        sum(volumes .* (solution .- exact.interior) .^ 2) / sum(volumes)
    )
    return error, operator
end

@testset "Aligned viscosity jump transmits the tangential traction (#113)" begin
    # The series conductance 2 mu_liquid mu_gas / (mu_liquid + mu_gas) = 4/3 at
    # the interface vertices, whose row viscosity the cell rows keep at their
    # cell value, gives second-order shear across the jump.
    harmonic_errors = Float64[]
    for n in (16, 32, 64, 128)
        error, operator = _aligned_jump_shear_error(n)
        push!(harmonic_errors, error)
        row_viscosity = sort(
            unique(
                round.(
                    operator.stress_weights ./ operator.stress_measures,
                    digits = 12,
                ),
            ),
        )
        @test row_viscosity ≈ [1.0, 4 / 3, 2.0]
    end
    harmonic_rates = log2.(
        harmonic_errors[1:(end - 1)] ./ harmonic_errors[2:end],
    )
    @test all(harmonic_rates .>= 1.9)

    # Equal viscosities are unaffected by the averaging rule: every row takes
    # the common value and the same solve keeps its second order, with or
    # without the geometric mixture.
    for rule in (
            HarmonicViscosityAveraging(), GeometricViscosityAveraging(),
        )
        equal_errors = Float64[]
        for n in (16, 32, 64, 128)
            error, operator = _aligned_jump_shear_error(
                n; liquid = 1.0, gas = 1.0, averaging = rule,
            )
            push!(equal_errors, error)
            @test all(
                value == 1.0 for value in
                    operator.stress_weights ./ operator.stress_measures
            )
        end
        equal_rates = log2.(equal_errors[1:(end - 1)] ./ equal_errors[2:end])
        @test all(equal_rates .>= 1.9)
    end
end

@testset "Geometric mixture keeps the first-order aligned shear (#113)" begin
    # The thesis' diffuse mixture is preserved as an option, including its
    # measured behaviour: the geometric vertex coefficient sqrt(2) instead of
    # 4/3 overestimates the traction flux by 6.07 %, and the aligned shear does
    # not converge at second order.
    errors = Float64[]
    for n in (16, 32, 64, 128)
        error, operator = _aligned_jump_shear_error(
            n; averaging = GeometricViscosityAveraging(),
        )
        push!(errors, error)
        row_viscosity = sort(
            unique(
                round.(
                    operator.stress_weights ./ operator.stress_measures,
                    digits = 12,
                ),
            ),
        )
        @test row_viscosity ≈ [1.0, sqrt(2.0), 2.0]
    end
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(rates .< 1.2)
end

@testset "Harmonic vertex viscosity stays dissipative (#113)" begin
    mesh = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && origin[1] < 0.5,
    )
    viscosity = CellField(mesh, (x, y) -> y < 0.5 ? 2.0 : 1.0)
    for rule in (HarmonicViscosityAveraging(), GeometricViscosityAveraging())
        operator = StaggeredDiffusionOperator(mesh; viscosity, averaging = rule)
        volumes = operator.volumes
        @test operator.stiffness ≈ transpose(operator.stiffness) atol =
            1.0e-14 * norm(operator.stiffness)
        constant = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : -2.0)
        constant_force = FaceVelocity(mesh)
        diffusion!(constant_force, constant, operator)
        # A constant velocity has no strain, so it is in the kernel up to
        # round-off relative to the matrix magnitude (the product is zero).
        stiffness_scale = maximum(abs, operator.stiffness) *
            maximum(abs, constant.interior)
        @test maximum(abs, operator.stiffness * constant.interior) <=
            1.0e-12 * stiffness_scale
        @test maximum(abs, constant_force.interior) <=
            1.0e-12 * stiffness_scale / minimum(volumes)
        velocity = FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ? sinpi(2y) + 3 : sinpi(2x) - 1,
        )
        force = FaceVelocity(mesh)
        diffusion!(force, velocity, operator)
        @test dot(velocity.interior, volumes .* force.interior) < 0
        # The stiffness stays positive semidefinite for every velocity; the
        # midpoint step is energy contractive with it.
        for seed in 1:4
            probe = [sin(seed * 1.7index) + cos(index / seed) for index in eachindex(volumes)]
            @test dot(probe, operator.stiffness * probe) >=
                -1.0e-12 * norm(probe)^2 * norm(operator.stiffness)
        end
        old_energy = sum(volumes .* velocity.interior .^ 2)
        result = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(result, velocity, operator, 0.01)
        @test sum(volumes .* result.interior .^ 2) < old_energy
    end
end

function _variable_stress_shifted_error(n, kind)
    refinement = (key, origin, extent) ->
    kind == :fixed &&
        (
        key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0)
    )
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2,
        periodic = (false, false), refine = refinement
    )
    viscosity_function = (x, y) -> 1 + 0.25cospi(2x)
    viscosity = CellField(mesh, viscosity_function)
    if kind == :adaptive
        adaptation = AMRWorkspace(mesh, viscosity)
        adapt!(adaptation, [:refine]; step = 1, force = true)
        labels = [
            block.key.level == 1 && block.key.i == 0 &&
                block.key.j == 0 ? :refine : :keep
                for block in mesh.blocks
        ]
        adapt!(adaptation, labels; step = 2, force = true)
        initialize!(viscosity, viscosity_function)
    end
    operator = StaggeredDiffusionOperator(
        mesh; viscosity, boundary = NoSlip()
    )
    exact_velocity(x, y, axis) = axis == 1 ?
        sinpi(x)^2 * sinpi(2y) :
        -sinpi(2x) * sinpi(y)^2
    function exact_diffusion(x, y, axis)
        cosx, cosy = cospi(2x), cospi(2y)
        sinx, siny = sinpi(2x), sinpi(2y)
        mu = viscosity_function(x, y)
        mux = -pi * sinx / 2
        return axis == 1 ?
            mu * pi^2 * (4cosx - 2) * siny +
            2mux * pi * sinx * siny :
            mu * 2pi^2 * sinx * (1 - 2cosy) +
            mux * 2pi * (
                sinpi(x)^2 * cosy - cosx * sinpi(y)^2
            )
    end
    exact = FaceVelocity(
        mesh,
        (x, y, time, axis) -> exact_velocity(x, y, axis)
    )
    shift = 0.01
    rhs = similar(operator.volumes)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        rhs[index] = operator.volumes[index] * (
            exact.interior[index] -
                shift * exact_diffusion(x, y, face.axis)
        )
    end
    matrix = spdiagm(0 => operator.volumes) +
        shift * operator.stiffness
    solution = matrix \ rhs
    error = solution .- exact.interior
    return (
        sqrt(sum(operator.volumes .* error .^ 2) / sum(operator.volumes)),
        maximum(abs, error),
    )
end


@testset "variable diffusion is independent of base-root partitioning" begin
    one_root = BlockTreeMesh(base_mesh = (16, 16), block_size = (16, 16))
    four_roots = BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8))

    function diffusion_values(mesh)
        viscosity = CellField(mesh, (x, y) -> 1 + 0.2cospi(2x) * sinpi(2y))
        operator = StaggeredDiffusionOperator(mesh; viscosity)
        velocity = FaceVelocity(
            mesh,
            (x, y, time, axis) -> axis == 1 ?
                sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
        )
        tendency = FaceVelocity(mesh)
        diffusion!(tendency, velocity, operator)
        return Dict(
            (face.center[1], face.center[2], face.axis) =>
                tendency.interior[index]
                for (index, face) in pairs(mesh.faces)
        )
    end

    first = diffusion_values(one_root)
    second = diffusion_values(four_roots)
    @test keys(first) == keys(second)
    @test all(
        isapprox(
            first[key], second[key]; atol = 2.0e-12, rtol = 2.0e-12,
        )
            for key in keys(first)
    )
end


@testset "variable diffusion crosses adaptive base-root seams" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        (key.level == 0 && key.i == 0 && key.j == 0) ||
            (key.level == 1 && key.i == 1 && key.j == 0),
    )
    viscosity = CellField(mesh, (x, y) -> 1 + 0.2cospi(2x) * sinpi(2y))
    operator = StaggeredDiffusionOperator(mesh; viscosity)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y),
    )
    tendency = FaceVelocity(mesh)
    diffusion!(tendency, velocity, operator)
    @test isbalanced(mesh)
    @test operator.stiffness ≈ transpose(operator.stiffness) atol = 2.0e-13
    @test dot(
        velocity.interior,
        operator.volumes .* tendency.interior,
    ) < 0
end

@testset "Variable stress shifted solve on fixed and adapted trees" begin
    # The shear rows give every staggered control volume the flux of its own
    # contact lengths, also at the level transitions (#112), so the composite
    # solve converges at second order in L2 and in the maximum norm.
    for kind in (:fixed, :adaptive)
        errors = [
            _variable_stress_shifted_error(n, kind)
                for n in (8, 16, 32)
        ]
        for norm_index in 1:2
            norm_errors = getindex.(errors, norm_index)
            rates = log2.(norm_errors[1:(end - 1)] ./ norm_errors[2:end])
            @test all(rates .>= 1.8)
        end
    end
end

function _transition_meshes()
    meshes = BlockTreeMesh[]
    for periodic in ((false, false), (true, false), (true, true))
        push!(
            meshes, BlockTreeMesh(;
                block_size = (6, 6), max_level = 2, periodic,
                refine = (key, origin, extent) -> key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0),
            ),
        )
        push!(
            meshes, BlockTreeMesh(;
                base_mesh = (16, 16), block_size = (4, 4), max_level = 2,
                periodic, refine = (key, origin, extent) ->
                (key.level == 0 && 0.25 <= origin[1] < 0.75 && 0.25 <= origin[2] < 0.75) ||
                    (key.level == 1 && 0.375 <= origin[1] < 0.625 && 0.375 <= origin[2] < 0.5),
            ),
        )
    end
    return meshes
end

@testset "Variable stress shear rows are flux-consistent at level transitions (#112)" begin
    for mesh in _transition_meshes(), boundary in (NoSlip(), FreeSlip())
        unit = StaggeredDiffusionOperator(
            mesh; viscosity = CellField(mesh; init = 1.0), boundary,
        )
        cells = ncells(mesh)
        shear_rows = (2cells + 1):length(unit.stress_measures)
        # The shear rows partition the periodic domain.
        all(mesh.periodic) &&
            @test sum(unit.stress_measures[shear_rows]) ≈ prod(mesh.extent) rtol = 1.0e-14
        # Away from walls and periodic seams, an affine velocity has exact
        # shear rates, and its uniform stress is in equilibrium on every
        # staggered control volume, including those at the transitions.
        spacing = maximum(maximum(block.spacing) for block in mesh.blocks)
        inner = [
            all(
                axis -> 2spacing < face.center[axis] - mesh.origin[axis] <
                    mesh.extent[axis] - 2spacing, 1:2,
            ) for face in mesh.faces
        ]
        inner_rows = [
            row for row in shear_rows if
                all(inner[column] for column in findall(!iszero, unit.strain[row, :]))
        ]
        @test length(inner_rows) > 50
        for (a, b, c, d) in (
                (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0, -1.0), (0.3, -0.7, 1.1, 0.4),
            )
            velocity = [
                face.axis == 1 ? a * face.center[1] + b * face.center[2] :
                    c * face.center[1] + d * face.center[2]
                    for face in mesh.faces
            ]
            @test unit.strain[inner_rows, :] * velocity ≈
                fill(b + c, length(inner_rows)) atol = 1.0e-12
            force = unit.stiffness * velocity
            @test maximum(abs, force[inner] ./ unit.volumes[inner]) <= 1.0e-9
        end

        viscosity = CellField(mesh, (x, y) -> 1 + 0.3sinpi(2x) * cospi(2y))
        operator = StaggeredDiffusionOperator(mesh; viscosity, boundary)
        stiffness = operator.stiffness
        @test stiffness == transpose(stiffness)
        @test eigmin(Symmetric(Matrix(stiffness))) >= -1.0e-12 * opnorm(Matrix(stiffness))
        # Momentum: a constant component has no strain when it is not
        # prescribed by walls (periodic along it and periodic or free slip
        # across it), so the constant is in the kernel.
        for component in 1:2
            mesh.periodic[component] || continue
            mesh.periodic[3 - component] || boundary isa FreeSlip || continue
            constant = [face.axis == component ? 1.0 : 0.0 for face in mesh.faces]
            @test maximum(abs, stiffness * constant) <= 1.0e-12 * opnorm(Matrix(stiffness), 1)
        end
    end
end

@testset "Variable stress rows are independent of the mesh origin" begin
    # The shear partition lives on the integer lattice, so a translated mesh
    # (walls in x, a periodic seam in y, two level transitions) gives the
    # same strain rows and stiffness bit for bit.
    function stress_operator(origin)
        mesh = BlockTreeMesh(;
            block_size = (4, 4), origin, extent = (0.7, 3.3),
            periodic = (false, true), max_level = 2,
            refine = (key, o, e) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        )
        return StaggeredDiffusionOperator(mesh; viscosity = CellField(mesh; init = 1.3))
    end
    reference = stress_operator((0.0, 0.0))
    translated = stress_operator((1.0e12, -1.0e12))
    @test translated.strain == reference.strain
    @test translated.stress_measures == reference.stress_measures
    @test translated.stiffness == reference.stiffness
end

@testset "Variable coupled stress diffusion" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        mu = CellField(mesh, (x, y) -> 1 + 0.25cospi(2x))
        if kind == 3
            amr = AMRWorkspace(mesh, mu);adapt!(amr, [:refine]; step = 1, force = true)
        end
        op = StaggeredDiffusionOperator(mesh; viscosity = mu)
        u = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? sinpi(2y) : sinpi(2x))
        force = FaceVelocity(mesh)
        diffusion!(force, u, op)
        @test dot(u.interior, op.volumes .* force.interior) < 0
        @test op.stiffness ≈ transpose(op.stiffness) atol = 1.0e-13
        constant = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 1.0 : -2.0)
        diffusion!(force, constant, op)
        @test maximum(abs, force.interior) < 1.0e-10
        oldenergy = sum(op.volumes .* u.interior .^ 2)
        result = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(result, u, op, 0.01)
        @test sum(op.volumes .* result.interior .^ 2) < oldenergy
        @test_throws ArgumentError StaggeredDiffusionOperator(mesh; viscosity = CellField(mesh; init = -1.0))
    end
end
@testset "Variable stress manufactured bulk convergence" begin
    errors = Float64[]
    for n in (12, 24, 48)
        mesh = BlockTreeMesh(block_size = (n, n))
        mu = CellField(mesh, (x, y) -> 1 + 0.25cospi(2x))
        op = StaggeredDiffusionOperator(mesh; viscosity = mu)
        u = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? sinpi(2y) : sinpi(2x))
        force = FaceVelocity(mesh);diffusion!(force, u, op)
        error = 0.0
        for (i, f) in pairs(mesh.faces)
            x, y = f.center
            viscosity = 1 + 0.25cospi(2x)
            expected = f.axis == 1 ? -4pi^2 * viscosity * sinpi(2y) :
                -4pi^2 * viscosity * sinpi(2x) - 0.5pi * sinpi(2x) * 2pi * (cospi(2y) + cospi(2x))
            error += op.volumes[i] * (force.interior[i] - expected)^2
        end
        push!(errors, sqrt(error))
    end
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(rates .>= 1.8)
end

@testset "Grid-aligned embedded walls have the cut-wall limit" begin
    # Use a floor much smaller than the offset: otherwise the interface cleanup
    # turns the 1e-9 cut into a pure cell and the comparison is vacuous.
    wall = 4.0
    offsets = (0.0, 1.0e-9)
    viscosity_value(x, y) = 1.0 + 0.02x + 0.03y

    function wall_mesh(normal, topology)
        tangent = 3 - normal
        refinement_axis = topology == :tangent_transition ? tangent : normal
        return BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
            periodic = normal == 1 ? (false, true) : (true, false),
            max_level = 1,
            refine = (key, origin, extent) ->
            topology != :uniform && key.level == 0 &&
                origin[refinement_axis] >= wall,
        )
    end

    function first_open_tangent_face(mesh, geometry, normal, side, coordinate)
        tangent = 3 - normal
        candidates = [
            index for (index, face) in pairs(mesh.faces)
                if face.axis == tangent &&
                abs(face.center[tangent] - coordinate) < 1.0e-12 &&
                side * (face.center[normal] - wall) > 0 &&
                geometry.face_dual_volumes[index] > 0
        ]
        sort!(
            candidates;
            by = index -> side * (mesh.faces[index].center[normal] - wall),
        )
        @test !isempty(candidates)
        return first(candidates)
    end

    for normal in 1:2, side in (-1, 1),
            topology in (:uniform, :tangent_transition, :normal_transition)
        @testset "normal=$normal side=$side topology=$topology" begin
            mesh = wall_mesh(normal, topology)
            tangent = 3 - normal
            viscosity = CellField(mesh, viscosity_value)
            refs = TwoPhaseAMR._cell_references(mesh)
            scalar_rates = Vector{Vector{Float64}}()
            variable_rates = Vector{Vector{Float64}}()
            phase_rates = Vector{Vector{Float64}}()
            selected_faces = Vector{Vector{Int}}()

            for offset in offsets
                solid = InterfaceField(mesh; fraction_floor = 1.0e-14)
                normal_vector = normal == 1 ? (side, 0) : (0, side)
                initialize!(solid, normal_vector, side * wall + offset)
                geometry = GeometryField(solid)
                fractions = [
                    TwoPhaseAMR._get(geometry.solid.fraction, ref)
                        for ref in refs
                ]
                if iszero(offset)
                    @test all(c -> iszero(c) || c == 1, fractions)
                else
                    @test any(c -> 0 < c < 1, fractions)
                end
                faces = [
                    first_open_tangent_face(mesh, geometry, normal, side, s)
                        for s in (2.0, 4.0, 6.0)
                ]
                push!(selected_faces, faces)
                velocity = FaceVelocity(
                    mesh, (x, y, time, axis) -> axis == tangent ? 1.0 : 0.0,
                )
                for index in eachindex(velocity.interior)
                    iszero(geometry.face_dual_volumes[index]) &&
                        (velocity.interior[index] = 0.0)
                end
                # This test calibrates the legacy closure's cut-wall limit;
                # pin it so a default change cannot alter the reference.
                scalar_no_slip = StaggeredDiffusionOperator(
                    geometry; viscosity = 1.0, boundary = NoSlip(),
                    closure = WallDistanceClosure(),
                )
                scalar_free_slip = StaggeredDiffusionOperator(
                    geometry; viscosity = 1.0, boundary = FreeSlip(),
                    closure = WallDistanceClosure(),
                )
                variable_no_slip = TwoPhaseAMR._staggered_diffusion_operator(
                    mesh; viscosity, boundary = NoSlip(), geometry,
                    closure = WallDistanceClosure(),
                )
                variable_free_slip = TwoPhaseAMR._staggered_diffusion_operator(
                    mesh; viscosity, boundary = FreeSlip(), geometry,
                    closure = WallDistanceClosure(),
                )
                scalar_no_rate = FaceVelocity(mesh)
                scalar_free_rate = FaceVelocity(mesh)
                variable_no_rate = FaceVelocity(mesh)
                variable_free_rate = FaceVelocity(mesh)
                diffusion!(scalar_no_rate, velocity, scalar_no_slip)
                diffusion!(scalar_free_rate, velocity, scalar_free_slip)
                diffusion!(variable_no_rate, velocity, variable_no_slip)
                diffusion!(variable_free_rate, velocity, variable_free_slip)
                push!(scalar_rates, scalar_no_rate.interior[faces])
                push!(variable_rates, variable_no_rate.interior[faces])
                if topology == :normal_transition
                    # Exercise the public two-phase viscosity construction as
                    # well as the spatially varying CellField dispatch.
                    interface = InterfaceField(
                        mesh; geometry, fraction_floor = 1.0e-14,
                    )
                    phase_normal = tangent == 1 ? (1, 0) : (0, 1)
                    initialize!(interface, phase_normal, 4.25)
                    phase_operator = StaggeredDiffusionOperator(
                        interface; viscosity = (1.0, 0.2), boundary = NoSlip(),
                        closure = WallDistanceClosure(),
                    )
                    phase_rate = FaceVelocity(mesh)
                    diffusion!(phase_rate, velocity, phase_operator)
                    push!(phase_rates, phase_rate.interior[faces])
                end

                component = tangent == 1 ? scalar_no_slip.x : scalar_no_slip.y
                scalar_indices = Dict(
                    global_index => local_index
                        for (local_index, global_index) in
                        pairs(component.local_to_global)
                )
                stresses = TwoPhaseAMR._embedded_variable_wall_stresses(geometry)
                for face_index in faces
                    local_index = scalar_indices[face_index]
                    wall_conductance = sum(
                        connection.conductance
                            for connection in component.wall_connections
                            if connection.degree_of_freedom == local_index &&
                            iszero(connection.boundary_axis) &&
                            connection.tangential;
                        init = 0.0,
                    )
                    wall_stress = sum(
                        conductance * TwoPhaseAMR._get(viscosity, refs[cell])
                            for (index, conductance, cell) in stresses
                            if index == face_index;
                        init = 0.0,
                    )
                    @test wall_conductance > 0
                    @test wall_stress > 0
                    # No-slip adds exactly the stationary embedded-wall row;
                    # FreeSlip omits it, without modifying other stresses.
                    @test isapprox(
                        scalar_no_rate.interior[face_index] -
                            scalar_free_rate.interior[face_index],
                        -wall_conductance / component.volumes[local_index];
                        rtol = 1.0e-11, atol = 1.0e-11,
                    )
                    @test isapprox(
                        variable_no_rate.interior[face_index] -
                            variable_free_rate.interior[face_index],
                        -wall_stress / variable_no_slip.volumes[face_index];
                        rtol = 1.0e-11, atol = 1.0e-11,
                    )
                end

                @test scalar_no_slip.x.stiffness ≈
                    transpose(scalar_no_slip.x.stiffness) atol = 1.0e-12
                @test scalar_no_slip.y.stiffness ≈
                    transpose(scalar_no_slip.y.stiffness) atol = 1.0e-12
                @test variable_no_slip.stiffness ≈
                    transpose(variable_no_slip.stiffness) atol = 1.0e-12
                @test dot(
                    velocity.interior,
                    variable_no_slip.stiffness * velocity.interior,
                ) >= 0

                if iszero(offset) && topology == :uniform
                    moving = StaggeredDiffusionOperator(
                        geometry;
                        viscosity = 1.0,
                        boundary = NoSlip((x, y, time, axis) -> 3.0),
                        closure = WallDistanceClosure(),
                    )
                    moving_rate = FaceVelocity(mesh)
                    diffusion!(moving_rate, velocity, moving)
                    # Only the remote domain walls move; the aligned embedded
                    # wall remains stationary even with a moving callback.
                    @test moving_rate.interior[faces] ≈
                        scalar_no_rate.interior[faces] atol = 1.0e-12
                end

                if iszero(offset)
                    # A covered face gets its algebraic stationary-wall row,
                    # not an additional aligned physical-wall connection.
                    covered = findall(iszero, geometry.face_dual_volumes)
                    @test !isempty(covered)
                    for axis in 1:2
                        part = axis == 1 ? scalar_no_slip.x : scalar_no_slip.y
                        for connection in part.wall_connections
                            global_index = part.local_to_global[
                                connection.degree_of_freedom,
                            ]
                            if iszero(geometry.face_dual_volumes[global_index]) &&
                                    iszero(connection.boundary_axis)
                                @test !connection.tangential
                            end
                        end
                    end
                    @test all(
                        index -> count(row -> row[1] == index, stresses) == 1,
                        covered,
                    )
                end
            end

            @test selected_faces[1] == selected_faces[2]
            for rates in (scalar_rates, variable_rates), point in 1:3
                aligned, shifted = rates[1][point], rates[2][point]
                @test aligned < 0
                @test shifted < 0
                # The 1e-9 displacement is several orders below this 1e-6
                # relative tolerance on both the coarse and fine meshes.
                @test abs(aligned - shifted) <=
                    1.0e-6 * max(abs(aligned), abs(shifted))
            end
            if topology == :normal_transition
                for point in 1:3
                    aligned, shifted = phase_rates[1][point], phase_rates[2][point]
                    @test aligned < 0
                    @test shifted < 0
                    @test abs(aligned - shifted) <=
                        1.0e-6 * max(abs(aligned), abs(shifted))
                end
            end
        end
    end
end

@testset "aligned hanging wall segments use exact lattice intervals" begin
    n = 128
    wall = Float64(π)
    block_width = 8wall / n
    mesh = BlockTreeMesh(
        base_mesh = (n, n), block_size = (4, 4), extent = (2wall, 2wall),
        periodic = (false, true), max_level = 1,
        refine = (key, origin, extent) ->
        key.level == 0 && wall <= origin[1] < wall + block_width,
    )
    solid = InterfaceField(mesh)
    initialize!(solid, (-1.0, 0.0), -wall)
    geometry = GeometryField(solid)
    hanging_faces = count(mesh.faces) do face
        minus_open = TwoPhaseAMR._get(geometry.open_fraction, face.minus)
        plus_open = TwoPhaseAMR._get(geometry.open_fraction, face.plus)
        mesh.blocks[face.minus.block].key.level != mesh.blocks[face.plus.block].key.level &&
            (
            (minus_open == 1 && iszero(plus_open)) ||
                (plus_open == 1 && iszero(minus_open))
        )
    end
    @test hanging_faces == 2n
    segments = TwoPhaseAMR._embedded_aligned_wall_segments(geometry)
    @test length(segments) == n
    @test all(length(pieces) == 1 for pieces in values(segments))
    @test all(
        only(pieces) == ((0.5, -0.5), (0.5, 0.5))
            for pieces in values(segments) if length(pieces) == 1
    )
end

@testset "geometry-aware variable diffusion multigrid is mesh independent around a solid" begin
    L = 8.0
    mesh_for(n, refined) = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), extent = (L, L),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) ->
        refined && key.level == 0 && key.i >= div(n, 8),
    )
    solid(x, y) = hypot(x - 4, y - 4) - 1.3
    # The interface crosses the solid; viscosity ratio 100.
    interface_levelset(x, y) = y - 4.3 - 0.1 * (x - 4)
    function iterations(n, refined)
        mesh = mesh_for(n, refined)
        geometry = GeometryField(mesh, solid; lipschitz = 1.2)
        interface = InterfaceField(
            mesh, interface_levelset; geometry, lipschitz = hypot(1, 0.1),
        )
        operator = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity = (1.0, 0.01), boundary = NoSlip(),
        )
        density = ones(length(operator.volumes))
        dt = 2 * (L / n)^2
        solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
            operator, density, dt; preconditioner_options = (; geometry),
        )
        unknowns = length(operator.volumes)
        # deterministic, rough
        solver.right_hand_side .= solver.inertia .* sinpi.(0.37 .* (1:unknowns))
        stats = TwoPhaseAMR._solve_variable_diffusion!(
            FaceVelocity(mesh), solver, operator;
            tolerance = 1.0e-10, max_cycles = 500,
        )
        @test stats.converged
        preconditioner = solver.preconditioner
        if n == 32
            # Covered faces neither receive nor restrict coarse corrections.
            covered = iszero.(geometry.face_dual_volumes)
            @test any(covered)
            prolongation = first(preconditioner.levels).prolongation
            @test all(iszero, sum(abs, prolongation[covered, :]; dims = 2))
            # Per-row Jacobi weights, folded into the inverse diagonal.
            @test preconditioner.local_relaxation
            @test all(level -> level.relaxation_weight == 1, preconditioner.levels)
            # The V-cycle stays a symmetric positive preconditioner for CG.
            first_residual = cospi.(0.21 .* (1:unknowns))
            second_residual = sinpi.(0.13 .* (1:unknowns) .+ 0.1)
            first_correction = similar(first_residual)
            second_correction = similar(second_residual)
            TwoPhaseAMR.ldiv!(first_correction, preconditioner, first_residual)
            TwoPhaseAMR.ldiv!(second_correction, preconditioner, second_residual)
            @test dot(second_residual, first_correction) ≈
                dot(first_residual, second_correction) rtol = 1.0e-10
            @test dot(first_residual, first_correction) > 0
        end
        return stats.cycles
    end
    # Before the coarse levels saw the solid, dt/2 = h² took 20 → 141 (uniform)
    # and 71 → 291 (composite) CG iterations from n = 32 to n = 128. For
    # dt/2 = 1 even the solid-free hierarchy grows (25 → 44, uniform), so that
    # regime cannot carry this bound.
    for refined in (false, true)
        coarse = iterations(32, refined)
        fine = iterations(128, refined)
        @test fine <= 1.5 * coarse
    end

    # A two-phase flow passes its solid to the viscous hierarchy itself.
    mesh = mesh_for(32, false)
    geometry = GeometryField(mesh, solid; lipschitz = 1.2)
    interface = InterfaceField(
        mesh, interface_levelset; geometry, lipschitz = hypot(1, 0.1),
    )
    flow = FlowSimulation(
        mesh, TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (1.0, 0.01));
        geometry,
    )
    hierarchy = flow.workspace.two_phase.diffusion_solver.preconditioner
    @test hierarchy isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test length(hierarchy.levels) > 1
    @test hierarchy.local_relaxation
    prolongation = first(hierarchy.levels).prolongation
    @test all(iszero, sum(abs, prolongation[iszero.(geometry.face_dual_volumes), :]; dims = 2))
end

# --- LS-STAG two-phase analytic and structural tests -------------------------
#
# The fixture puts a straight solid wall of slope 1/2 inside the unit square,
# with the two-phase interface above it. An affine velocity that vanishes on
# the wall has a constant strain tensor, so every LS-STAG stress row next to
# the wall must reproduce one tensor component exactly. LS-STAG places a cut
# face's value at the midpoint of its open part, not at the face centre, so the
# tests sample the analytic field at that midpoint. The open part is computed
# here from the level set (an independent geometric expectation), not from the
# closure's PLIC tables. The PLIC reconstruction of the straight line agrees
# with the analytic trace to its quadrature tolerance; `rtol = 1e-7` on the
# row residuals is far above that and far below an O(1) sign or factor error.

const _LS2P_WALL_INTERCEPT = 0.25
const _LS2P_WALL_SLOPE = 0.5

function _ls2p_mesh(n = 16; T = Float64, periodic = (false, false))
    return BlockTreeMesh(;
        block_size = (n, n),
        origin = (zero(T), zero(T)), extent = (one(T), one(T)),
        periodic,
    )
end

_ls2p_wall_levelset(x, y) = y - (_LS2P_WALL_INTERCEPT + _LS2P_WALL_SLOPE * x)
_ls2p_phase_levelset(x, y) = y - 0.85

function _ls2p_geometry(mesh; T = eltype(mesh.origin))
    return GeometryField(
        mesh,
        (x, y) -> y - (T(_LS2P_WALL_INTERCEPT) + T(_LS2P_WALL_SLOPE) * x);
        lipschitz = hypot(one(T), T(_LS2P_WALL_SLOPE)),
    )
end

"""Open part of an axis-aligned face against a level set linear along it.
Returns `(length, midpoint)`. A face lying in the solid returns length zero."""
function _ls2p_open_segment(face, levelset)
    center = face.center
    half = face.area / 2
    tangent = 3 - face.axis
    lower = center[tangent] - half
    upper = center[tangent] + half
    value(t) = face.axis == 1 ? levelset(center[1], t) : levelset(t, center[2])
    flow = value(lower)
    fhigh = value(upper)
    if flow >= 0 && fhigh >= 0
        return face.area, center
    elseif flow < 0 && fhigh < 0
        return zero(face.area), center
    end
    crossing = lower + (upper - lower) * flow / (flow - fhigh)
    open_lower, open_upper = flow >= 0 ? (lower, crossing) : (crossing, upper)
    midpoint = face.axis == 1 ?
        (center[1], (open_lower + open_upper) / 2) :
        ((open_lower + open_upper) / 2, center[2])
    return open_upper - open_lower, midpoint
end

"""Sample a component of `component[axis] * scalar` at the LS-STAG value
point of every active face; constraints and covered faces are zero."""
function _ls2p_sample(
        mesh, geometry, operator, component, scalar, ::Type{T},
    ) where {T}
    values = zeros(T, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        open_length, midpoint = _ls2p_open_segment(face, scalar)
        active = open_length > T(64) * eps(T) * T(face.area) &&
            geometry.face_dual_volumes[index] > zero(T)
        active || continue
        @inbounds values[index] =
            T(component[face.axis]) * T(scalar(midpoint...))
    end
    for index in operator.dirichlet_faces
        values[index] = zero(T)
    end
    return values
end

@inline function _ls2p_cell_index(ref, mesh)
    nx, ny = mesh.block_size
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
end

"""Rows of the strain map that couple both velocity components: the shear
rows of the stress representation, as opposed to normal and wall rows."""
function _ls2p_shear_rows(mesh, operator)
    number_of_faces = length(mesh.faces)
    xface = falses(number_of_faces)
    yface = falses(number_of_faces)
    for (index, face) in pairs(mesh.faces)
        face.axis == 1 ? (xface[index] = true) : (yface[index] = true)
    end
    rows = Int[]
    for row in 1:length(operator.stress_measures)
        columns = findall(!iszero, operator.strain[row, :])
        isempty(columns) && continue
        any(xface[columns]) && any(yface[columns]) && push!(rows, row)
    end
    return rows
end

"""The normalized wall normal, tangent and signed distance field of the
sloping-wall fixture."""
function _ls2p_affine_fields()
    weight = hypot(1.0, _LS2P_WALL_SLOPE)
    normal = (-_LS2P_WALL_SLOPE / weight, 1.0 / weight)
    tangent = (1.0 / weight, _LS2P_WALL_SLOPE / weight)
    scalar(x, y) = normal[1] * x + normal[2] * (y - _LS2P_WALL_INTERCEPT)
    return normal, tangent, scalar
end

"""Check the analytic normal and mixed-shear rows of an affine velocity that
vanishes on the sloping wall. `inside(center)` decides whether a row's cell
support is far enough from the domain walls. Returns the checked row counts."""
function _ls2p_check_affine_rows(
        mesh, geometry, operator;
        inside, rtol = 1.0e-7, atol = 1.0e-8,
    )
    normal, tangent, scalar = _ls2p_affine_fields()
    refs = TwoPhaseAMR._cell_references(mesh)
    shear_rows = _ls2p_shear_rows(mesh, operator)
    checked_cut = 0
    checked_shear = 0
    for component in (normal, tangent)
        values = _ls2p_sample(
            mesh, geometry, operator, component, scalar, Float64,
        )
        stresses = operator.strain * values
        expected_normal = (
            component[1] * normal[1], component[2] * normal[2],
        )
        expected_shear =
            component[1] * normal[2] + component[2] * normal[1]
        values_x = copy(values)
        values_y = copy(values)
        for (index, face) in pairs(mesh.faces)
            face.axis == 1 ? (values_y[index] = 0.0) : (values_x[index] = 0.0)
        end
        strain_x = operator.strain * values_x
        strain_y = operator.strain * values_y

        for ref in refs
            inside(cellcenter(mesh, ref.block, ref.i, ref.j)) || continue
            fraction = TwoPhaseAMR._get(geometry.open_fraction, ref)
            # Skip measure-zero slivers: their PLIC reconstruction error is
            # not the signal this analytic-row test measures.
            1.0e-4 < fraction < 1 - 1.0e-4 || continue
            cell = _ls2p_cell_index(ref, mesh)
            @test isapprox(
                stresses[2 * (cell - 1) + 1], expected_normal[1];
                rtol, atol,
            )
            @test isapprox(
                stresses[2 * cell], expected_normal[2]; rtol, atol,
            )
            checked_cut += 1
        end

        for row in shear_rows
            cells = findall(!iszero, operator.stress_interpolation[row, :])
            isempty(cells) && continue
            near = any(
                cell -> begin
                    fraction = TwoPhaseAMR._get(
                        geometry.open_fraction, refs[cell],
                    )
                    1.0e-4 < fraction < 1 - 1.0e-4
                end, cells,
            )
            near || continue
            all(
                cell -> begin
                    ref = refs[cell]
                    inside(cellcenter(mesh, ref.block, ref.i, ref.j))
                end, cells,
            ) || continue
            # A shear row is the sum of both mixed derivatives. The split
            # checks that neither component is lost and that their signs add.
            @test isapprox(stresses[row], expected_shear; rtol, atol)
            @test isapprox(
                strain_x[row], component[1] * normal[2]; rtol, atol,
            )
            @test isapprox(
                strain_y[row], component[2] * normal[1]; rtol, atol,
            )
            checked_shear += 1
        end
    end
    return checked_cut, checked_shear
end

@testset "LS-STAG two-phase stress rows are exact at a sloping solid" begin
    mesh = _ls2p_mesh()
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    # Uniform phase viscosity: the affine field has a constant stress
    # tensor, so its interior viscous force is exactly zero. Variable phase
    # viscosity is covered by the assembly and refresh tests below.
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (1.0, 1.0), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test operator isa TwoPhaseAMR.VariableStaggeredDiffusionOperator
    @test operator.closure isa LSStagClosure
    @test operator.mesh === mesh

    _, tangent, scalar = _ls2p_affine_fields()
    h = 1.0 / mesh.block_size[1]
    checked_cut, checked_shear = _ls2p_check_affine_rows(
        mesh, geometry, operator;
        inside = center ->
        1.5h < center[1] < 1 - 1.5h &&
            1.5h < center[2] < 1 - 1.5h,
    )
    @test checked_cut > 2
    @test checked_shear > 2

    # The independent open-part geometry agrees with the closure's apertures.
    validated = 0
    for (index, face) in pairs(mesh.faces)
        open_length, _ = _ls2p_open_segment(face, _ls2p_wall_levelset)
        aperture = geometry.aperture[index]
        if aperture > 64eps()
            @test isapprox(
                open_length, aperture * face.area;
                rtol = 1.0e-7, atol = 1.0e-8 * face.area,
            )
            validated += 1
        else
            @test open_length <= 1.0e-6 * face.area
        end
    end
    @test validated > 10

    stiffness = operator.stiffness
    scale = maximum(abs, nonzeros(stiffness))
    @test stiffness ≈ transpose(stiffness) rtol = 256eps() atol = 256eps() * scale
    @test operator.strain_transpose == copy(transpose(operator.strain))
    # E'WE is a sum of squares; only the accumulation of its rows separates
    # the smallest eigenvalue from zero.
    @test eigmin(Symmetric(Matrix(stiffness))) >= -1.0e-10 * scale
    for probe_index in 1:4
        probe = [
            sin(0.73 * probe_index * index) + cos(0.31 * index^2)
                for index in eachindex(operator.volumes)
        ]
        @test dot(probe, stiffness * probe) >=
            -1.0e-10 * scale * dot(probe, probe)
        # The LS rows must stay inside the symmetric E'WE representation, not
        # be added as an unrelated non-conservative correction.
        stress = operator.strain * probe
        @test isapprox(
            dot(probe, stiffness * probe),
            dot(stress, operator.stress_weights .* stress);
            rtol = 4096eps(),
            atol = 4096eps() * scale * dot(probe, probe),
        )
    end

    # A linear field with constant stress has zero viscous force. At the solid
    # this is a nontrivial balance of the Green rows and the physical mixed
    # shear rows, including their one-sided stationary-wall partners; it
    # catches a missing mixed component, a wrong wall quotient or a lost
    # factor of two near the cut cells.
    force = stiffness * _ls2p_sample(
        mesh, geometry, operator, tangent, scalar, Float64,
    )
    row_magnitude = abs.(transpose(stiffness)) *
        abs.(_ls2p_sample(mesh, geometry, operator, tangent, scalar, Float64))
    largest = maximum(row_magnitude)
    nx, ny = mesh.block_size
    checked_force = 0
    for index in eachindex(force)
        geometry.face_dual_volumes[index] > 0 || continue
        index in operator.dirichlet_faces && continue
        face = mesh.faces[index]
        x, y = face.center
        2h < x < 1 - 2h && 2h < y < 1 - 2h || continue
        minus_ref, plus_ref = face.minus, face.plus
        (
            1 < minus_ref.i < nx && 1 < minus_ref.j < ny &&
                1 < plus_ref.i < nx && 1 < plus_ref.j < ny
        ) || continue
        scale_force = max(row_magnitude[index], 1.0e-12 * largest)
        @test abs(force[index]) <= 1.0e-7 * scale_force
        checked_force += 1
    end
    @test checked_force > 20
end

@testset "LS-STAG two-phase affine rows on a fixed composite mesh" begin
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (8, 8), max_level = 1,
        periodic = (false, false),
        refine = (key, origin, extent) ->
        key.level == 0 && origin[1] >= 0.5,
    )
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (2.5, 0.4), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test operator.closure isa LSStagClosure
    # Skip rows whose support reaches the domain boundary; the coarse block
    # width bounds the distance on both levels.
    spacing = maximum(maximum(block.spacing) for block in mesh.blocks)
    checked_cut, checked_shear = _ls2p_check_affine_rows(
        mesh, geometry, operator;
        inside = center ->
        2spacing < center[1] < 1 - 2spacing &&
            2spacing < center[2] < 1 - 2spacing,
        rtol = 1.0e-6, atol = 1.0e-7,
    )
    @test checked_cut > 2
    @test checked_shear > 2
end

@testset "LS-STAG factorized two-phase action matches the assembled operator" begin
    mesh = _ls2p_mesh()
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (2.5, 0.4), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    density = TwoPhaseAMR.mixture_density(interface, (1.0, 0.25))
    inertia = density .* operator.volumes
    dt = 0.05
    action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
        operator, inertia, dt,
    )
    matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
    x = [sin(0.7index) for index in eachindex(operator.volumes)]
    y = [cos(0.3index) for index in eachindex(operator.volumes)]
    action_x = similar(x)
    action_y = similar(y)
    mul!(action_x, action, x)
    mul!(action_y, action, y)
    @test action_x ≈ matrix * x rtol = 256eps() atol = 256eps() * norm(matrix * x)
    symmetry_scale = max(
        norm(x) * norm(action_y), norm(action_x) * norm(y), 1.0,
    )
    @test abs(dot(x, action_y) - dot(action_x, y)) <=
        512eps() * symmetry_scale
    @test dot(x, action_x) > 0
end

@testset "LS-STAG phase relabel and viscosity refresh around a solid" begin
    mesh = _ls2p_mesh()
    geometry = _ls2p_geometry(mesh)
    liquid_below = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    liquid_above = InterfaceField(
        mesh, (x, y) -> -_ls2p_phase_levelset(x, y);
        geometry, lipschitz = 1.0,
    )
    phases = (2.5, 0.4)
    original = StaggeredDiffusionOperator(
        liquid_below; viscosity = phases, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    swapped = StaggeredDiffusionOperator(
        liquid_above; viscosity = reverse(phases), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test original.closure isa LSStagClosure
    @test swapped.closure isa LSStagClosure
    # Relabelling the phases and exchanging their viscosities is a symmetry of
    # the assembly; the harmonic mixture is invariant under `c -> 1-c`. The two
    # fractions come from independent level-set quadratures (atol 1e-10), so
    # the comparison is quadrature-limited rather than round-off-limited.
    @test original.strain == swapped.strain
    @test original.stress_measures ≈ swapped.stress_measures rtol = 64eps()
    @test original.stress_weights ≈ swapped.stress_weights rtol = 1.0e-8
    @test original.stiffness ≈ swapped.stiffness rtol = 1.0e-8

    # Moving the interface then refreshing the mixture must be identical to a
    # fresh assembly, including the LS wall rows in cut cells.
    initialize!(liquid_below, (x, y) -> y - 0.84; lipschitz = 1.0)
    TwoPhaseAMR._update_dynamic_viscosity!(original, liquid_below, phases)
    fresh = StaggeredDiffusionOperator(
        liquid_below; viscosity = phases, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test original.strain == fresh.strain
    @test original.stress_measures == fresh.stress_measures
    # The refresh stores log-viscosity and recomputes the weights in place;
    # a fresh assembly takes the same logarithms through the public mixture.
    @test original.cell_work ≈ fresh.cell_work rtol = 1.0e-10
    @test original.stress_weights ≈ fresh.stress_weights rtol = 1.0e-10
    @test original.stiffness ≈ fresh.stiffness rtol = 1.0e-10
    @test original.closure isa LSStagClosure

    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) :
            -cospi(2x) * sinpi(2y),
    )
    refreshed_force = FaceVelocity(mesh)
    fresh_force = FaceVelocity(mesh)
    diffusion!(refreshed_force, velocity, original)
    diffusion!(fresh_force, velocity, fresh)
    @test refreshed_force.interior ≈ fresh_force.interior rtol = 1.0e-9 atol = 1.0e-10

    # The solver matrix and the factorized action are refreshed from the same
    # in-place stress weights.
    density = TwoPhaseAMR.mixture_density(liquid_below, (1.0, 0.25))
    dt = 0.02
    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        original, density, dt; preconditioner = nothing,
    )
    TwoPhaseAMR._update_variable_diffusion_solver!(
        solver, original, density, dt,
    )
    reference = Diagonal(solver.inertia) + (dt / 2) * original.stiffness
    @test solver.matrix ≈ reference rtol = 64eps()
end

@testset "LS-STAG two-phase variable stress in Float32" begin
    T = Float32
    mesh = _ls2p_mesh(12; T)
    geometry = GeometryField(
        mesh,
        (x, y) -> y - (T(_LS2P_WALL_INTERCEPT) + T(_LS2P_WALL_SLOPE) * x);
        lipschitz = hypot(one(T), T(_LS2P_WALL_SLOPE)),
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - T(0.85); geometry, lipschitz = one(T),
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (T(2.5), T(0.4)), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test operator.closure isa LSStagClosure
    @test eltype(operator.stiffness) == T
    @test eltype(operator.volumes) == T
    @test eltype(operator.stress_weights) == T
    @test isapprox(
        operator.stiffness, transpose(operator.stiffness);
        rtol = T(32) * eps(T),
    )
    scale = maximum(abs, nonzeros(operator.stiffness))
    x = T[sin(T(0.7) * index) for index in eachindex(operator.volumes)]
    stress = operator.strain * x
    @test isapprox(
        dot(x, operator.stiffness * x),
        dot(stress, operator.stress_weights .* stress);
        rtol = T(1024) * eps(T),
        atol = T(1024) * eps(T) * scale * dot(x, x),
    )
    @test eigmin(Symmetric(Matrix(operator.stiffness))) >=
        -T(4096) * eps(T) * scale

    density = TwoPhaseAMR.mixture_density(interface, (one(T), T(0.25)))
    inertia = density .* operator.volumes
    dt = T(0.05)
    action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
        operator, inertia, dt,
    )
    matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
    action_x = similar(x)
    mul!(action_x, action, x)
    @test isapprox(
        action_x, matrix * x;
        rtol = T(512) * eps(T),
        atol = T(512) * eps(T) * norm(matrix * x),
    )

    # The Green normal row is the analytic derivative for the affine field,
    # sampled at the open midpoints. Float32 reconstruction and round-off
    # broaden the tolerance, but an O(1) sign or factor error is still caught.
    weight = hypot(one(T), T(_LS2P_WALL_SLOPE))
    normal = (-T(_LS2P_WALL_SLOPE) / weight, one(T) / weight)
    scalar(x, y) = normal[1] * x +
        normal[2] * (y - T(_LS2P_WALL_INTERCEPT))
    values = _ls2p_sample(mesh, geometry, operator, normal, scalar, T)
    stresses = operator.strain * values
    refs = TwoPhaseAMR._cell_references(mesh)
    nx, ny = mesh.block_size
    checked = 0
    for ref in refs
        1 < ref.i < nx && 1 < ref.j < ny || continue
        fraction = TwoPhaseAMR._get(geometry.open_fraction, ref)
        T(1.0e-4) < fraction < 1 - T(1.0e-4) || continue
        cell = _ls2p_cell_index(ref, mesh)
        @test isapprox(
            stresses[2 * (cell - 1) + 1], normal[1]^2;
            rtol = T(2) * T(1.0e-3), atol = T(1.0e-3),
        )
        @test isapprox(
            stresses[2 * cell], normal[2]^2;
            rtol = T(2) * T(1.0e-3), atol = T(1.0e-3),
        )
        checked += 1
    end
    @test checked > 2
end

@testset "LS-STAG coarse hierarchy keeps the closure and matches a rebuild" begin
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 && origin[1] >= 0.5,
    )
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (2.5, 0.4), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    density = TwoPhaseAMR.mixture_density(interface, (1.0, 0.25))
    dt = 0.02
    multigrid = MultigridOptions(coarse_size = 32)
    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt;
        multigrid, preconditioner_options = (; geometry),
    )
    hierarchy = solver.preconditioner
    @test hierarchy isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test length(hierarchy.operators) > 1
    @test hierarchy.local_relaxation
    # Every rediscretized level consumes the fine operator's closure, not the
    # legacy default.
    @test all(
        level_operator -> level_operator.closure isa LSStagClosure,
        hierarchy.operators,
    )
    @test all(
        level_operator -> isapprox(
            level_operator.stiffness,
            transpose(level_operator.stiffness);
            rtol = 256eps(),
        ),
        hierarchy.operators,
    )
    # The solid is present on the coarse levels: their open volumes differ
    # from the uncut mesh cache.
    @test any(
        level_operator ->
        level_operator.volumes != level_operator.mesh.face_dual_volumes,
        hierarchy.operators[2:end],
    )

    fresh = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt;
        multigrid, preconditioner_options = (; geometry),
    )
    rebuilt = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt;
        multigrid, preconditioner_options = (; geometry),
        reuse_preconditioner = hierarchy,
    )
    @test length(rebuilt.preconditioner.operators) ==
        length(fresh.preconditioner.operators)
    for (reused, reference) in zip(
            rebuilt.preconditioner.operators, fresh.preconditioner.operators,
        )
        @test reused.closure isa LSStagClosure
        @test reused.volumes == reference.volumes
        @test reused.stiffness ≈ reference.stiffness rtol = 64eps()
    end
    for (reused, reference) in zip(
            rebuilt.preconditioner.levels, fresh.preconditioner.levels,
        )
        @test reused.matrix ≈ reference.matrix rtol = 64eps()
    end

    # The hierarchy result equals the direct shifted solve, so the closure is
    # exercised end to end and not only during construction.
    source = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) :
            -cospi(2x) * sinpi(2y),
    )
    source.interior[operator.dirichlet_faces] .= 0.0
    reference_rhs = solver.inertia .* source.interior -
        (dt / 2) * (operator.stiffness * source.interior)
    TwoPhaseAMR._zero_variable_diffusion_constraints!(reference_rhs, operator)
    reference = (
        Diagonal(solver.inertia) + (dt / 2) * operator.stiffness
    ) \ reference_rhs
    for current in (rebuilt, fresh)
        destination = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(
            destination, source, operator, dt, current;
            density, tolerance = 1.0e-10, max_cycles = 500,
        )
        @test current.stats.converged
        @test destination.interior ≈ reference rtol = 1.0e-7 atol = 1.0e-9
    end

    # A cached hierarchy of the other closure must not be reused. Without a
    # solid the coarse sparsity is closure independent, so only the explicit
    # closure key prevents an LS solve from running on WallDistance levels.
    fluid = InterfaceField(mesh, _ls2p_phase_levelset)
    fluid_density = TwoPhaseAMR.mixture_density(fluid, (1.0, 0.25))
    legacy = StaggeredDiffusionOperator(
        fluid; viscosity = (2.5, 0.4), boundary = NoSlip(),
        closure = WallDistanceClosure(),
    )
    legacy_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        legacy, fluid_density, dt; multigrid,
    )
    ls_fluid = StaggeredDiffusionOperator(
        fluid; viscosity = (2.5, 0.4), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    ls_solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        ls_fluid, fluid_density, dt;
        multigrid, reuse_preconditioner = legacy_solver.preconditioner,
    )
    @test typeof(legacy_solver.preconditioner.operators[1].closure) ===
        WallDistanceClosure
    @test all(
        level_operator -> level_operator.closure isa LSStagClosure,
        ls_solver.preconditioner.operators,
    )
    # A refresh cannot silently retarget an LS hierarchy to the other closure.
    @test_throws ArgumentError TwoPhaseAMR._update_variable_diffusion_solver!(
        ls_solver, legacy, fluid_density, dt,
    )
end

@testset "LS-STAG two-phase slip at an embedded wall" begin
    mesh = _ls2p_mesh(16; periodic = (true, false))
    wall = 4.0 / 16.0
    geometry = GeometryField(mesh, (x, y) -> y - wall; lipschitz = 1.0)
    viscosity = CellField(mesh; init = 1.0)
    no_slip = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary = NoSlip(), geometry,
        closure = LSStagClosure(),
    )
    free_slip = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, boundary = FreeSlip(), geometry,
        closure = LSStagClosure(),
    )
    @test no_slip.closure isa LSStagClosure
    @test free_slip.closure isa LSStagClosure
    @test no_slip.volumes == free_slip.volumes
    @test no_slip.dirichlet_faces == free_slip.dirichlet_faces

    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0,
    )
    for index in eachindex(velocity.interior)
        geometry.face_dual_volumes[index] > 0 ||
            (velocity.interior[index] = 0.0)
    end
    velocity.interior[no_slip.dirichlet_faces] .= 0.0

    no_tendency = FaceVelocity(mesh)
    free_tendency = FaceVelocity(mesh)
    diffusion!(no_tendency, velocity, no_slip)
    diffusion!(free_tendency, velocity, free_slip)
    # A constant field tangential to a free-slip wall has no strain at all.
    @test maximum(abs, free_tendency.interior) <= 256eps()
    @test any(!iszero, no_tendency.interior)
    difference = no_slip.stiffness - free_slip.stiffness
    @test difference ≈ transpose(difference) rtol = 256eps()
    @test eigmin(Symmetric(Matrix(difference))) >=
        -1024eps() * maximum(abs, nonzeros(difference))
    @test dot(velocity.interior, no_slip.stiffness * velocity.interior) > 0
    @test dot(velocity.interior, free_slip.stiffness * velocity.interior) <=
        256eps()
    @test dot(velocity.interior, difference * velocity.interior) > 0

    h = 1.0 / 16.0
    drag_face = findfirst(
        face -> face.axis == 1 &&
            isapprox(face.center[2], wall + h / 2; atol = 64eps()),
        mesh.faces,
    )
    @test !isnothing(drag_face)
    @test no_tendency.interior[drag_face] < 0
end

@testset "LS-STAG tilted FreeSlip keeps a rigid tangent local null mode" begin
    mesh = _ls2p_mesh()
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(
        mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0,
    )
    normal, tangent, _ = _ls2p_affine_fields()
    @test isapprox(
        normal[1] * tangent[1] + normal[2] * tangent[2], 0.0;
        atol = 128eps(),
    )
    free_slip = StaggeredDiffusionOperator(
        interface; viscosity = (1.0, 1.0), boundary = FreeSlip(),
        closure = LSStagClosure(),
    )
    no_slip = StaggeredDiffusionOperator(
        interface; viscosity = (1.0, 1.0), boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test free_slip.closure isa LSStagClosure
    @test no_slip.closure isa LSStagClosure
    @test free_slip.volumes == no_slip.volumes
    @test free_slip.dirichlet_faces == no_slip.dirichlet_faces

    # A constant velocity tangent to the planar wall, hence a steady rigid
    # tangential motion. A free tangential wall trace leaves it strain-free;
    # only the domain walls it crosses can sense it, so the local check stays
    # two cells inside the box.
    velocity = zeros(length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        # Positive dual volume does not imply an open face: a velocity
        # degree at a zero-length aperture is only the isolated algebraic
        # extension, not a sample of the fluid's rigid translation.
        geometry.aperture[index] > 0 && (velocity[index] = tangent[face.axis])
    end
    velocity[free_slip.dirichlet_faces] .= 0.0
    h = 1.0 / mesh.block_size[1]
    inside(center) =
        1.5h < center[1] < 1 - 1.5h && 1.5h < center[2] < 1 - 1.5h

    free_force = free_slip.stiffness * velocity
    free_magnitude = abs.(transpose(free_slip.stiffness)) * abs.(velocity)
    free_largest = maximum(free_magnitude)
    closed = findall(iszero, geometry.aperture)
    @test !isempty(closed)
    @test all(iszero, free_force[closed])
    checked = 0
    for index in eachindex(free_force)
        geometry.face_dual_volumes[index] > 0 || continue
        index in free_slip.dirichlet_faces && continue
        face = mesh.faces[index]
        inside(face.center) || continue
        scale = max(free_magnitude[index], 1.0e-12 * free_largest)
        @test abs(free_force[index]) <= 1.0e-9 * scale
        checked += 1
    end
    @test checked > 20

    # The same rigid tangent motion is damped by the no-slip tilted wall, so
    # the free-slip elimination is not vacuous.
    no_velocity = copy(velocity)
    no_velocity[no_slip.dirichlet_faces] .= 0.0
    no_force = no_slip.stiffness * no_velocity
    no_magnitude = abs.(transpose(no_slip.stiffness)) * abs.(no_velocity)
    no_largest = maximum(no_magnitude)
    damped = 0
    for index in eachindex(no_force)
        geometry.face_dual_volumes[index] > 0 || continue
        index in no_slip.dirichlet_faces && continue
        inside(mesh.faces[index].center) || continue
        abs(no_force[index]) >
            1.0e-6 * max(no_magnitude[index], 1.0e-12 * no_largest) ||
            continue
        damped += 1
    end
    @test damped > 0
end

# Manufacture the exact volume fractions AND exact PLICs of a periodic
# planar band. This isolates the viscous identity from the error in estimating
# a normal from neighboring fractions across a 2:1 transition.
function _ls2p_exact_slip_band(mesh, slope, offset)
    refs = TwoPhaseAMR._cached_cell_references!(TwoPhaseAMR.CellRef[], mesh)
    function plane(ref)
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        d = mod(y - slope * x - offset + 0.25, 0.5) - 0.25
        dx, dy = mesh.blocks[ref.block].spacing
        side = copysign(1.0, d)
        return -side * slope * dx, side * dy, 0.125 - abs(d)
    end
    fractions = CellField(mesh)
    for ref in refs
        TwoPhaseAMR._set!(fractions, TwoPhaseAMR._plane_fraction(plane(ref)...), ref)
    end
    solid = InterfaceField(fractions)
    for ref in refs
        0 < TwoPhaseAMR._get(fractions, ref) < 1 || continue
        nx, ny, a = plane(ref)
        norm = hypot(nx, ny)
        TwoPhaseAMR._set!(solid.normal_x, nx / norm, ref)
        TwoPhaseAMR._set!(solid.normal_y, ny / norm, ref)
        TwoPhaseAMR._set!(solid.intercept, a / norm, ref)
    end
    return GeometryField(solid)
end

@testset "LS-STAG free-slip energy nests at terminating grid lines" begin
    for composite in (false, true), (slope, offset) in ((0.5, 0.0), (-0.5, 1 / 128))
        mesh = BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (4, 4), max_level = 1, periodic = (true, true),
            refine = (key, origin, _) -> composite && key.level == 0 && origin[1] >= 0.5,
        )
        # The first band has walls through vertices. The second misses the
        # vertices, but cuts the coarse partner of a hanging grid line.
        geometry = _ls2p_exact_slip_band(mesh, slope, offset)
        phase = InterfaceField(mesh, (x, _) -> abs(x - 0.5) - 0.25; geometry, lipschitz = 1.0)
        no_slip = StaggeredDiffusionOperator(phase; viscosity = (0.2, 0.03), boundary = NoSlip())
        free_slip = StaggeredDiffusionOperator(phase; viscosity = (0.2, 0.03), boundary = FreeSlip())
        loss = no_slip.stiffness - free_slip.stiffness
        @test eigmin(Symmetric(Matrix(loss))) >= -1024eps()
        # Unlike in a closed box, this rigid translation is admissible on
        # the whole periodic band. Releasing the trace must remove its work.
        u = [geometry.aperture[i] > 0 ? (1.0, slope)[f.axis] : 0.0 for (i, f) in pairs(mesh.faces)]
        @test dot(u, loss * u) > 0
        @test maximum(abs, free_slip.strain * u) <= 1024eps()
    end
end

@testset "Uncondensed LS-STAG FreeSlip rows reproduce the PLIC tangent" begin
    for composite in (false, true), (slope, offset) in ((0.5, 0.0), (-0.5, 1 / 128))
        # Straight parallel walls on a torus: every cut cell, including ones
        # at the box seam and at hanging vertices, admits a rigid tangent.
        mesh = BlockTreeMesh(;
            base_mesh = (32, 32), block_size = (8, 8), max_level = 1,
            periodic = (true, true),
            refine = (key, origin, _) -> composite && key.level == 0 && origin[1] >= 0.5,
        )
        geometry = _ls2p_exact_slip_band(mesh, slope, offset)
        phase = InterfaceField(mesh, (x, _) -> abs(x - 0.5) - 0.25; geometry, lipschitz = 1.0)
        noslip = StaggeredDiffusionOperator(
            phase; viscosity = (1.0, 0.3), boundary = NoSlip(), closure = LSStagClosure(),
        )

        # Both boundary types start with the same unprojected normal rows.
        # Assemble the actual slip shear/vertex rows, but do not apply the
        # local wall-velocity elimination: its input b is what we check.
        n = ncells(mesh)
        measures = noslip.stress_measures[1:(2n)]
        construction = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
        refs = TwoPhaseAMR._cached_cell_references!(construction.cell_references, mesh)
        topology = TwoPhaseAMR._ls_stress_topology!(
            TwoPhaseAMR._LSStressCache(Float64), geometry,
        )
        sites = Tuple{Int, Int, NTuple{3, Float64}}[]
        for (c, ref) in pairs(refs)
            slip = TwoPhaseAMR._ls_stress_cell!(topology, geometry, ref, c, measures, construction, FreeSlip())
            isnothing(slip) || push!(sites, (c, slip...))
        end
        @test length(sites) == count(ref -> 0 < TwoPhaseAMR._get(geometry.open_fraction, ref) < 1, refs)
        @test !isempty(sites)
        shear = sparse(
            construction.strain_rows, construction.strain_columns, construction.strain_values,
            length(measures), length(mesh.faces),
        )
        strain = vcat(noslip.strain[1:(2n), :], shear[(2n + 1):end, :])
        basis = zeros(length(mesh.faces), 2)
        for (i, f) in pairs(mesh.faces)
            geometry.aperture[i] > 0 && (basis[i, f.axis] = 1)
        end
        response = strain * basis
        magnitude = abs.(strain) * basis
        row_max = zeros(3)
        max_scaled = 0.0
        tangent = (1.0, slope) ./ hypot(1.0, slope)
        for (c, wall, b) in sites
            ref = refs[c]
            # One constant tangent throughout the mesh. The scalar wall
            # velocity (and hence its column b) points the other way on
            # the opposite side of the band.
            orient = -sign(TwoPhaseAMR._get(geometry.solid.normal_y, ref))
            for (j, r) in pairs((2c - 1, 2c, wall))
                residual = abs(tangent[1] * response[r, 1] + tangent[2] * response[r, 2] - orient * b[j])
                scale = max(1.0, abs(b[j]), abs(tangent[1]) * magnitude[r, 1] + abs(tangent[2]) * magnitude[r, 2])
                row_max[j] = max(row_max[j], residual)
                max_scaled = max(max_scaled, residual / scale)
                @test residual <= 128eps() * scale
            end
        end
        # The shared shear derivative of either Cartesian constant must be
        # zero before any cut-cell condensation or cancellation in E'WE.
        shared = filter(!iszero, topology.vertex_rows)
        @test !isempty(shared)
        for r in shared, axis in 1:2
            @test abs(response[r, axis]) <= 128eps() * max(1.0, magnitude[r, axis])
        end
        println(
            "Uncondensed E*u_t=b: composite=$composite slope=$slope cut_cells=$(length(sites)) ",
            "shared_vertices=$(length(shared)) max_abs_normal_shear=$(Tuple(row_max)) ",
            "max_scaled=$max_scaled",
        )
    end
end

@testset "LS-STAG NoSlip Helmholtz convergence at box-boundary T-junction" begin
    # psi = [x(1-x)y(1-y)(y-c)]^2, u = (psi_y, -psi_x).
    # Its gradient vanishes on the box and on the stationary solid y = c,
    # and div(u) = 0. With constant viscosity, div(2D(u)) = Laplacian(u).
    c = 0.25
    levelset(x, y) = y - c
    function manufactured(x, y, axis)
        g, gx = x * (1 - x), 1 - 2x
        p, px = g^2, 2g * gx
        pxx, pxxx = 2(gx^2 - 2g), -12gx
        h, hy = y * (1 - y), 1 - 2y
        q = h * (y - c)
        qy = hy * (y - c) + h
        qyy = -2(y - c) + 2hy
        qyyy = -6
        Q, Qy = q^2, 2q * qy
        Qyy = 2(qy^2 + q * qyy)
        Qyyy = 2(3qy * qyy + q * qyyy)
        return axis == 1 ? (p * Qy, pxx * Qy + p * Qyyy) :
            (-px * Q, -pxxx * Q - px * Qyy)
    end

    errors = Float64[]
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(;
            base_mesh = (n, n), block_size = (4, 4), max_level = 1,
            periodic = (false, false),
            refine = (key, origin, _) -> key.level == 0 && origin[1] >= 0.5,
        )
        geometry = GeometryField(mesh, levelset; lipschitz = 1.0)
        interface = InterfaceField(mesh, (x, y) -> y - 5; geometry, lipschitz = 1.0)
        operator = StaggeredDiffusionOperator(
            interface; viscosity = (1.0, 1.0), boundary = NoSlip(),
            closure = LSStagClosure(),
        )
        exact = zeros(length(mesh.faces))
        rhs = similar(exact)
        fill!(rhs, 0)
        active = falses(length(mesh.faces))
        for (index, face) in pairs(mesh.faces)
            open_length, midpoint = _ls2p_open_segment(face, levelset)
            open_length > 64eps() * face.area || continue
            geometry.face_dual_volumes[index] > 0 || continue
            index in operator.dirichlet_faces && continue
            active[index] = true
            u, laplacian = manufactured(midpoint..., face.axis)
            exact[index] = u
            # The discrete mass is the operator's dual volume, including
            # the algebraic treatment of covered and constrained faces.
            rhs[index] = operator.volumes[index] * (u - laplacian)
        end
        @test any(active)
        solution = (spdiagm(0 => operator.volumes) + operator.stiffness) \ rhs
        push!(errors, maximum(abs, (solution - exact)[active]))
    end
    # The coarsest pair is slightly pre-asymptotic (reviewed EOC ~1.77);
    # the two-level slope and final pair both exceed 1.8 for second order,
    # while a first-order T-junction error fails either criterion.
    @test log2(errors[1] / errors[3]) / 2 >= 1.8
    @test log2(errors[2] / errors[3]) >= 1.8
end

@testset "LS-STAG thin-gap FreeSlip box traces are compact and refresh exactly" begin
    n = 64
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), periodic = (false, false),
    )
    # A 1.5-cell fluid gap separates x=0 from the parallel embedded solid.
    geometry = GeometryField(mesh, (x, y) -> 1.5 / n - x; lipschitz = 1.0)
    lower = InterfaceField(mesh, (x, y) -> y - 0.5; geometry, lipschitz = 1.0)
    crossing = InterfaceField(
        mesh, (x, y) -> y - 0.25 + 0.05x; geometry, lipschitz = hypot(1.0, 0.05),
    )
    initial = (lower, (2.5, 0.4))
    states = ((crossing, (0.35, 1.9)), initial)
    construct(field, viscosity, boundary) =
        StaggeredDiffusionOperator(field; viscosity, boundary, closure = LSStagClosure())
    free = construct(initial..., FreeSlip())
    patches = free.box_traces
    @test patches isa Vector
    @test any(p -> min(length(p.rows), length(p.faces)) >= n ÷ 2, patches)
    patch_rows = falses(length(free.stress_weights))
    for patch in patches
        @test size(patch.basis, 1) == length(patch.rows)
        @test 0 < size(patch.basis, 2) <= length(patch.rows)
        patch_rows[patch.rows] .= true
    end
    # Count all retained arrays, including coefficient/CSC-slot matrices and
    # both trace bases. The old 24-byte tuples per (row,face,face) exceed
    # this quadratic envelope by more than an order of magnitude at n=64.
    entries = sum(p -> length(p.rows) * length(p.faces) + length(p.faces)^2, patches)
    @test Base.summarysize(patches) <= 16 * sizeof(Int) * entries + 4096 * (1 + length(patches))
    @test !any(row -> patch_rows[row], free.stiffness_stresses)

    peak(A) = maximum(abs, nonzeros(A); init = 0.0)
    for (field, viscosity) in states
        old_weights = free.stress_weights[patch_rows]
        old_strain, old_stiffness = copy(free.strain), copy(free.stiffness)
        TwoPhaseAMR._update_dynamic_viscosity!(free, field, viscosity)
        fresh = construct(field, viscosity, FreeSlip())
        @test free.stress_weights[patch_rows] != old_weights
        @test peak(free.strain - old_strain) > 0
        @test peak(free.stiffness - old_stiffness) > 1.0e-6 * peak(fresh.stiffness)
        for name in (:strain, :stiffness, :wall_reaction_strain)
            actual, expected = getproperty(free, name), getproperty(fresh, name)
            @test actual.colptr == expected.colptr
            @test rowvals(actual) == rowvals(expected)
            @test peak(actual - expected) <= 512eps() * max(1.0, peak(expected))
        end
        @test free.dirichlet_faces == fresh.dirichlet_faces
        @test free.stress_weights ≈ fresh.stress_weights rtol = 64eps()
        @test free.stiffness_factors == fresh.stiffness_factors
        @test free.stiffness_stresses == fresh.stiffness_stresses
        @test free.stiffness_pointers == fresh.stiffness_pointers
        @test free.stiffness_refresh_offsets == fresh.stiffness_refresh_offsets
        for op in (free, fresh)
            E, K = op.strain, op.stiffness
            @test op.strain_transpose == transpose(E)
            @test issymmetric(K)
            @test peak(K - transpose(E) * (Diagonal(op.stress_weights) * E)) <=
                1024eps() * peak(K)
        end

        no_slip = construct(field, viscosity, NoSlip())
        @test isnothing(no_slip.box_traces)
        loss = dropzeros!(no_slip.stiffness - free.stiffness)
        @test issymmetric(loss)
        @test peak(loss) > 1.0e-6 * peak(no_slip.stiffness)
        # All other columns (and by symmetry rows) vanish exactly. This
        # principal block has the full nonzero spectrum; no mode is sampled
        # or discarded by a numerical support threshold.
        support = findall(c -> !isempty(nzrange(loss, c)), axes(loss, 2))
        @test 0 < length(support) <= 16n
        if 0 < length(support) <= 16n
            @test eigmin(Symmetric(Matrix(loss[support, support]))) >=
                -1024eps() * peak(no_slip.stiffness)
        end
    end
end

# Per-side walls (`VelocityBoundaryConditions` with stationary `NoSlip()` and
# `FreeSlip()` sides). A box vertex frees its tangential trace only on a
# FreeSlip side; at a box corner each velocity component is the normal
# velocity of a side, so the corner trace stays zero for every combination.
# The trace sets nest from all no-slip sides through every mixed
# configuration to all free-slip sides. An embedded solid takes the
# `embedded` condition, a stationary no-slip wall by default, as in the
# single-phase operator.
_pw_sides(left, right, bottom, top; embedded = NoSlip()) =
    VelocityBoundaryConditions(; left, right, bottom, top, embedded)
_pw_all_sides(wall; embedded = NoSlip()) = _pw_sides(wall, wall, wall, wall; embedded)
_pw_combinations() = vec(
    [
        _pw_sides(left, right, bottom, top)
            for left in (NoSlip(), FreeSlip()), right in (NoSlip(), FreeSlip()),
            bottom in (NoSlip(), FreeSlip()), top in (NoSlip(), FreeSlip())
    ],
)
_pw_eigmin(matrix) = eigmin(Symmetric(Matrix(matrix)))

"""Composite fixture with the right (`axis = 1`) or top (`axis = 2`) half
refined, without and with the sloping solid, and a phase interface that
touches no wall."""
function _pw_fields(; axis = 1, n = 8, T = Float64)
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (4, 4), max_level = 1,
        origin = (zero(T), zero(T)), extent = (one(T), one(T)),
        periodic = (false, false),
        refine = (key, origin, _) -> key.level == 0 && origin[axis] >= T(0.5),
    )
    phase(x, y) = hypot(x - T(0.45), y - T(0.62)) - T(0.21)
    fluid = InterfaceField(mesh, phase; lipschitz = one(T))
    solid = InterfaceField(mesh, phase; geometry = _ls2p_geometry(mesh; T), lipschitz = one(T))
    return mesh, (fluid, solid)
end

@testset "Per-side variable-stress walls match uniform walls and reject open sides" begin
    mesh, fields = _pw_fields()
    viscosity = (0.2, 0.03)
    for field in fields, closure in (LSStagClosure(), WallDistanceClosure())
        build(boundary) = StaggeredDiffusionOperator(field; viscosity, boundary, closure)
        # Equal sides and solid reproduce the uniform operator bitwise.
        for wall in (NoSlip(), FreeSlip())
            sides = _pw_all_sides(wall; embedded = wall)
            reference, operator = build(wall), build(sides)
            @test operator.boundary === sides
            @test operator.strain == reference.strain
            @test operator.stress_weights == reference.stress_weights
            @test operator.stiffness == reference.stiffness
            @test operator.dirichlet_faces == reference.dirichlet_faces
        end
        for invalid in (
                _pw_sides(Inflow((x, y, time, axis) -> 0.0), NoSlip(), NoSlip(), NoSlip()),
                _pw_sides(NoSlip(), Outflow(), NoSlip(), NoSlip()),
                _pw_sides(NoSlip(), NoSlip(), ZeroGradientVelocity(), NoSlip()),
                _pw_sides(NoSlip(), NoSlip(), NoSlip(), NoSlip((x, y, time, axis) -> 1.0)),
                NoSlip((x, y, time, axis) -> 1.0),
            )
            @test_throws ArgumentError build(invalid)
        end
    end
    for embedded in (NoSlip((x, y, time, axis) -> 1.0), Outflow())
        @test_throws ArgumentError VelocityBoundaryConditions(; embedded)
    end
    # The default solid is a no-slip wall: free-slip sides then release only
    # box traces, so the released stiffness stays within two coarse cells of
    # the box, and uniform FreeSlip(), which also releases the solid, lies
    # below it.
    field = last(fields)
    no_slip = StaggeredDiffusionOperator(field; viscosity, boundary = NoSlip())
    sides = StaggeredDiffusionOperator(field; viscosity, boundary = _pw_all_sides(FreeSlip()))
    free_slip = StaggeredDiffusionOperator(field; viscosity, boundary = FreeSlip())
    released = dropzeros!(no_slip.stiffness - sides.stiffness)
    support = [mesh.faces[c].center for c in axes(released, 2) if !isempty(nzrange(released, c))]
    @test !isempty(support)
    @test all(x -> min(x[1], 1 - x[1], x[2], 1 - x[2]) < 2 / 8, support)
    scale = maximum(abs, nonzeros(no_slip.stiffness))
    solid_release = sides.stiffness - free_slip.stiffness
    @test maximum(abs, nonzeros(solid_release)) > 1.0e-6 * scale
    @test _pw_eigmin(solid_release) >= -1024eps() * scale
end

@testset "Mixed per-side walls nest between no-slip and free-slip walls" begin
    _, fields = _pw_fields()
    # A solid disk next to the bottom-left corner cuts the corner cell, whose
    # Green completion then sees the traces of both sides at the corner.
    corner_mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), periodic = (false, false))
    corner_geometry = GeometryField(corner_mesh, (x, y) -> hypot(x - 0.12, y - 0.12) - 0.1; lipschitz = 1.0)
    @test 0 < TwoPhaseAMR._get(corner_geometry.open_fraction, TwoPhaseAMR.CellRef(1, 1, 1)) < 1
    corner = InterfaceField(
        corner_mesh, (x, y) -> hypot(x - 0.6, y - 0.55) - 0.2;
        geometry = corner_geometry, lipschitz = 1.0,
    )
    for field in (fields..., corner), closure in (LSStagClosure(), WallDistanceClosure()),
            viscosity in ((0.2, 0.03), (1.0, 1.0e-3))
        build(boundary) = StaggeredDiffusionOperator(field; viscosity, boundary, closure)
        no_slip = build(NoSlip())
        free = (build(FreeSlip()), build(_pw_all_sides(FreeSlip())))
        scale = maximum(abs, nonzeros(no_slip.stiffness))
        for sides in _pw_combinations()
            mixed = build(sides)
            @test issymmetric(mixed.stiffness)
            @test _pw_eigmin(mixed.stiffness) >= -1024eps() * scale
            @test _pw_eigmin(no_slip.stiffness - mixed.stiffness) >= -1024eps() * scale
            for lower in free
                @test _pw_eigmin(mixed.stiffness - lower.stiffness) >= -1024eps() * scale
            end
        end
    end
end

@testset "LS-STAG per-side box traces follow their side and never a corner" begin
    mesh, fields = _pw_fields(; axis = 2)
    lattice = TwoPhaseAMR._finest_lattice_size(mesh)
    construction = TwoPhaseAMR.VariableDiffusionConstructionCache(Float64)
    for sides in _pw_combinations()
        TwoPhaseAMR._staggered_diffusion_operator(
            last(fields); viscosity = (0.2, 0.03), boundary = sides,
            closure = LSStagClosure(), construction_cache = construction,
        )
        cache = construction.ls_stag
        corners = 0
        counts = Dict((axis, side) => 0 for axis in 1:2, side in (-1, 1))
        for vertex in cache.box_vertices
            key = cache.keys[vertex]
            on = map(d -> key[d] == 0 || key[d] == lattice[d], (1, 2))
            if all(on)
                corners += 1
            else
                axis = on[1] ? 1 : 2
                counts[(axis, iszero(key[axis]) ? -1 : 1)] += 1
            end
        end
        @test corners == 0
        for ((axis, side), count) in counts
            # The solid covers the bottom side; every other free side has
            # released fluid traces, and a no-slip side has none.
            free = TwoPhaseAMR._side_boundary(sides, axis, side) isa FreeSlip
            @test (count > 0) == (free && (axis, side) != (2, -1))
        end
    end
end

@testset "Mixed per-side walls keep the rigid tangent of a free-slip side" begin
    # Channel periodic along x with hanging vertices on the floor (right half
    # refined), viscosity jumps that meet the floor and a no-slip solid next
    # to the no-slip lid. A constant tangential velocity has no strain at a
    # free-slip floor; a no-slip floor damps it.
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        periodic = (true, false),
        refine = (key, origin, _) -> key.level == 0 && origin[1] >= 0.5,
    )
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 0.5, y - 0.75) - 0.12; lipschitz = 1.0)
    interface = InterfaceField(mesh, (x, y) -> abs(x - 0.5) - 0.2; geometry, lipschitz = 1.0)
    floor = [i for (i, face) in pairs(mesh.faces) if face.center[2] < 0.4]
    for closure in (LSStagClosure(), WallDistanceClosure())
        build(bottom) = StaggeredDiffusionOperator(
            interface; viscosity = (2.5, 0.4), closure,
            boundary = _pw_sides(NoSlip(), NoSlip(), bottom, NoSlip()),
        )
        free_floor, no_slip_floor = build(FreeSlip()), build(NoSlip())
        @test free_floor.dirichlet_faces == no_slip_floor.dirichlet_faces
        closure isa LSStagClosure && @test !isnothing(free_floor.box_traces)
        u = [geometry.aperture[i] > 0 && face.axis == 1 ? 1.0 : 0.0 for (i, face) in pairs(mesh.faces)]
        u[free_floor.dirichlet_faces] .= 0.0
        force = free_floor.stiffness * u
        magnitude = abs.(free_floor.stiffness) * abs.(u)
        largest = maximum(magnitude)
        @test maximum(i -> abs(force[i]) / max(magnitude[i], 1.0e-12 * largest), floor) <= 1.0e-9
        damping = no_slip_floor.stiffness * u
        @test count(i -> abs(damping[i]) > 1.0e-6 * largest, floor) > 0
        @test dot(u, no_slip_floor.stiffness * u) > dot(u, free_floor.stiffness * u) > 0
    end
end

@testset "Per-side walls with a free-slip solid nest and keep its rigid tangent" begin
    # At fixed sides, releasing the solid only removes dissipation; with a
    # free-slip solid the side traces still nest down to uniform FreeSlip().
    _, fields = _pw_fields()
    field = last(fields)
    for closure in (LSStagClosure(), WallDistanceClosure()), viscosity in ((0.2, 0.03), (1.0, 1.0e-3))
        build(boundary) = StaggeredDiffusionOperator(field; viscosity, boundary, closure)
        scale = maximum(abs, nonzeros(build(NoSlip()).stiffness))
        no_slip_sides = build(_pw_all_sides(NoSlip(); embedded = FreeSlip()))
        free_slip = build(FreeSlip())
        for sides in _pw_combinations()
            no_slip_solid = build(sides)
            free_solid = build(
                _pw_sides(sides.left, sides.right, sides.bottom, sides.top; embedded = FreeSlip()),
            )
            K = free_solid.stiffness
            @test issymmetric(K)
            @test _pw_eigmin(K) >= -1024eps() * scale
            @test _pw_eigmin(no_slip_solid.stiffness - K) >= -1024eps() * scale
            @test _pw_eigmin(no_slip_sides.stiffness - K) >= -1024eps() * scale
            @test _pw_eigmin(K - free_slip.stiffness) >= -1024eps() * scale
        end
    end
    # A constant velocity tangent to the sloping free-slip solid has no
    # strain away from the box, whatever the sides; a no-slip solid damps it.
    mesh = _ls2p_mesh()
    geometry = _ls2p_geometry(mesh)
    interface = InterfaceField(mesh, _ls2p_phase_levelset; geometry, lipschitz = 1.0)
    _, tangent, _ = _ls2p_affine_fields()
    h = 1.0 / mesh.block_size[1]
    inside(center) = 1.5h < center[1] < 1 - 1.5h && 1.5h < center[2] < 1 - 1.5h
    for embedded in (FreeSlip(), NoSlip())
        operator = StaggeredDiffusionOperator(
            interface; viscosity = (1.0, 1.0), closure = LSStagClosure(),
            boundary = _pw_sides(NoSlip(), FreeSlip(), FreeSlip(), NoSlip(); embedded),
        )
        u = [geometry.aperture[i] > 0 ? tangent[face.axis] : 0.0 for (i, face) in pairs(mesh.faces)]
        u[operator.dirichlet_faces] .= 0.0
        force = operator.stiffness * u
        magnitude = abs.(operator.stiffness) * abs.(u)
        largest = maximum(magnitude)
        inner = [
            i for (i, face) in pairs(mesh.faces) if geometry.face_dual_volumes[i] > 0 &&
                !(i in operator.dirichlet_faces) && inside(face.center)
        ]
        @test length(inner) > 20
        ratio = maximum(i -> abs(force[i]) / max(magnitude[i], 1.0e-12 * largest), inner)
        @test embedded isa FreeSlip ? ratio <= 1.0e-9 : ratio > 1.0e-6
    end
end

# Bivariate polynomials, `P[i + 1, j + 1]` the coefficient of `x^i y^j`, for
# exact manufactured streamfunctions with mixed side conditions.
function _pw_poly_mul(A, B)
    C = zeros(size(A, 1) + size(B, 1) - 1, size(A, 2) + size(B, 2) - 1)
    for i in axes(A, 1), j in axes(A, 2), k in axes(B, 1), l in axes(B, 2)
        C[i + k - 1, j + l - 1] += A[i, j] * B[k, l]
    end
    return C
end
_pw_poly_mul(A, B, C...) = _pw_poly_mul(_pw_poly_mul(A, B), C...)
function _pw_poly_add(A, B)
    C = zeros(max(size(A, 1), size(B, 1)), max(size(A, 2), size(B, 2)))
    C[axes(A)...] .+= A
    C[axes(B)...] .+= B
    return C
end
_pw_poly_add(A, B, C...) = _pw_poly_add(_pw_poly_add(A, B), C...)
_pw_poly_dx(A) = size(A, 1) == 1 ? zero(A) : [i * A[i + 1, j] for i in 1:(size(A, 1) - 1), j in axes(A, 2)]
_pw_poly_dy(A) = size(A, 2) == 1 ? zero(A) : [j * A[i, j + 1] for i in axes(A, 1), j in 1:(size(A, 2) - 1)]
_pw_poly(A, x, y) = sum(A[i, j] * x^(i - 1) * y^(j - 1) for i in axes(A, 1), j in axes(A, 2))
_pw_x(c...) = reshape(collect(Float64, c), :, 1)
_pw_y(c...) = reshape(collect(Float64, c), 1, :)

"""Velocity `(psi_y, -psi_x)` and its Laplacian for the streamfunction `psi`."""
function _pw_manufactured(psi)
    velocity = (_pw_poly_dy(psi), -_pw_poly_dx(psi))
    laplacian = map(u -> _pw_poly_add(_pw_poly_dx(_pw_poly_dx(u)), _pw_poly_dy(_pw_poly_dy(u))), velocity)
    return velocity, laplacian
end

@testset "Mixed per-side walls keep second-order manufactured solutions" begin
    # psi vanishes with its normal derivative on no-slip walls and with its
    # second normal derivative on free-slip sides, whose tangential velocity
    # is not zero. Free side walls: `x - 2x^3 + x^4`; no-slip floor and lid:
    # `y^2 (1 - y)^2`. With the solid y = c + s x (no slip, fluid above),
    # psi = l^2 x(1 - x)(1 - y)^2 H, l = y - c - s x, with H the cubic in x
    # that makes psi_xx vanish on x = 0 and 1 (free sides meeting the solid);
    # psi = l^2 x^2 (1 - x)^2 (1 - y)(3 - c - s x - 2y) has a free lid
    # between no-slip sides, with mixed corners. For c = 0 the solid passes
    # through the corner (0, 0), whose cell is cut on every mesh.
    s = 0.5
    level(c) = _pw_poly_add(_pw_x(-c, -s), _pw_y(0, 1))
    free_sides(c) = _pw_poly_mul(
        level(c), level(c), _pw_x(0, 1, -1), _pw_y(1, -2, 1),
        _pw_poly_add(
            _pw_poly_mul(_pw_x(1, 0, -3, 2), _pw_y(-c, 1)),
            _pw_poly_mul(_pw_x(0, 1, -2, 1), _pw_y(2s - c, 1)),
            _pw_poly_mul(_pw_x(0, 0, 3, -2), _pw_y(-c - s, 1)),
            _pw_poly_mul(_pw_x(0, 0, -1, 1), _pw_y(c + 3s, -1)),
        ),
    )
    free_lid(c) = _pw_poly_mul(
        level(c), level(c), _pw_x(0, 0, 1, -2, 1), _pw_y(1, -1),
        _pw_poly_add(_pw_x(3 - c, -s), _pw_y(0, -2)),
    )
    wall(c) = (x, y) -> y - c - s * x
    F, N = FreeSlip(), NoSlip()
    cases = (
        # (streamfunction, sides, solid level set or nothing, refined axis)
        (_pw_poly_mul(_pw_x(0, 1, 0, -2, 1), _pw_y(0, 0, 1, -2, 1)), _pw_sides(F, F, N, N), nothing, 2),
        (free_sides(0.25), _pw_sides(F, F, N, N), wall(0.25), 1),
        (free_lid(0.25), _pw_sides(N, N, N, F), wall(0.25), 1),
        (free_sides(0.0), _pw_sides(F, F, N, N), wall(0.0), 2),
    )
    for (psi, sides, solid, axis) in cases
        velocity, laplacian = _pw_manufactured(psi)
        inside(x, y) = isnothing(solid) || solid(x, y) >= 0
        # The fixture satisfies its own wall conditions.
        residual = 0.0
        for t in range(0.0, 1.0; length = 33), (wall_axis, side) in ((1, -1), (1, 1), (2, -1), (2, 1))
            x, y = wall_axis == 1 ? (side < 0 ? 0.0 : 1.0, t) : (t, side < 0 ? 0.0 : 1.0)
            inside(x, y) || continue
            tangent = 3 - wall_axis
            normal_velocity = _pw_poly(velocity[wall_axis], x, y)
            slip = TwoPhaseAMR._side_boundary(sides, wall_axis, side) isa FreeSlip
            shear = _pw_poly(wall_axis == 1 ? _pw_poly_dx(velocity[tangent]) : _pw_poly_dy(velocity[tangent]), x, y)
            residual = max(residual, abs(normal_velocity), abs(slip ? shear : _pw_poly(velocity[tangent], x, y)))
        end
        @test residual <= 1.0e-12
        l2 = Float64[]
        peak = Float64[]
        for n in (16, 32, 64)
            mesh = BlockTreeMesh(;
                base_mesh = (n, n), block_size = (4, 4), max_level = 1,
                periodic = (false, false),
                refine = (key, origin, _) -> key.level == 0 && origin[axis] >= 0.5,
            )
            geometry = isnothing(solid) ? nothing : GeometryField(mesh, solid; lipschitz = hypot(1.0, s))
            interface = isnothing(geometry) ? InterfaceField(mesh, (x, y) -> y - 5) :
                InterfaceField(mesh, (x, y) -> y - 5; geometry, lipschitz = 1.0)
            operator = StaggeredDiffusionOperator(
                interface; viscosity = (1.0, 1.0), boundary = sides, closure = LSStagClosure(),
            )
            weights = isnothing(geometry) ? operator.volumes : geometry.face_dual_volumes
            exact = zeros(length(mesh.faces))
            rhs = zeros(length(mesh.faces))
            active = falses(length(mesh.faces))
            for (index, face) in pairs(mesh.faces)
                open_length, midpoint = isnothing(solid) ? (face.area, face.center) :
                    _ls2p_open_segment(face, solid)
                open_length > 64eps() * face.area || continue
                weights[index] > 0 || continue
                index in operator.dirichlet_faces && continue
                active[index] = true
                exact[index] = _pw_poly(velocity[face.axis], midpoint...)
                rhs[index] = operator.volumes[index] *
                    (exact[index] - _pw_poly(laplacian[face.axis], midpoint...))
            end
            solution = (spdiagm(0 => operator.volumes) + operator.stiffness) \ rhs
            misfit = (solution - exact)[active]
            push!(l2, sqrt(sum(weights[active] .* misfit .^ 2) / sum(weights[active])))
            push!(peak, maximum(abs, misfit))
        end
        # Two-level slope and final pair, as for the no-slip T-junction: a
        # first-order wall, junction or corner error fails either criterion.
        for errors in (l2, peak)
            @test log2(errors[1] / errors[3]) / 2 >= 1.8
            @test log2(errors[2] / errors[3]) >= 1.8
        end
    end
end

@testset "LS-STAG per-side free-slip traces in Float32" begin
    T = Float32
    _, fields = _pw_fields(; T)
    field = last(fields)
    viscosity = (T(2.5), T(0.4))
    build(boundary) = StaggeredDiffusionOperator(field; viscosity, boundary, closure = LSStagClosure())
    mixed = build(_pw_sides(FreeSlip(), NoSlip(), NoSlip(), FreeSlip()))
    @test eltype(mixed.stiffness) == T
    @test eltype(mixed.stress_weights) == T
    @test !isnothing(mixed.box_traces)
    @test all(patch -> eltype(patch.gram) == T, mixed.box_traces)
    scale = maximum(abs, nonzeros(build(NoSlip()).stiffness))
    @test isapprox(mixed.stiffness, transpose(mixed.stiffness); rtol = T(32) * eps(T))
    @test _pw_eigmin(Float64.(mixed.stiffness)) >= -4096eps(T) * scale
    @test _pw_eigmin(Float64.(build(NoSlip()).stiffness - mixed.stiffness)) >= -4096eps(T) * scale
    @test _pw_eigmin(Float64.(mixed.stiffness - build(FreeSlip()).stiffness)) >= -4096eps(T) * scale
end

# Keep the allocation check in compiled code, warming each launch path.
function _pw_thin_gap_refresh_allocations(operator, lower, crossing, selected_grain)
    grain = TwoPhaseAMR._PARALLEL_GRAIN
    previous = grain[]
    try
        grain[] = selected_grain
        TwoPhaseAMR._update_dynamic_viscosity!(operator, lower, (2.5, 0.4))
        TwoPhaseAMR._update_dynamic_viscosity!(operator, crossing, (0.35, 1.9))
        return @allocated TwoPhaseAMR._update_dynamic_viscosity!(operator, lower, (2.5, 0.4))
    finally
        grain[] = previous
    end
end

@testset "LS-STAG per-side thin-gap traces refresh exactly with bounded allocations" begin
    n = 32
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), periodic = (false, false))
    # A 1.5-cell fluid gap runs along the left side.
    geometry = GeometryField(mesh, (x, y) -> 1.5 / n - x; lipschitz = 1.0)
    lower = InterfaceField(mesh, (x, y) -> y - 0.5; geometry, lipschitz = 1.0)
    crossing = InterfaceField(
        mesh, (x, y) -> y - 0.25 + 0.05x; geometry, lipschitz = hypot(1.0, 0.05),
    )
    construct(field, viscosity, boundary) =
        StaggeredDiffusionOperator(field; viscosity, boundary, closure = LSStagClosure())
    gap_free = _pw_sides(FreeSlip(), NoSlip(), NoSlip(), NoSlip())
    ends_free = _pw_sides(NoSlip(), NoSlip(), FreeSlip(), FreeSlip())
    # Only a free gap side couples its traces along the whole side.
    long_patch(operator) = !isnothing(operator.box_traces) && any(
        patch -> min(length(patch.rows), length(patch.faces)) >= n ÷ 2, operator.box_traces,
    )
    operator = construct(lower, (2.5, 0.4), gap_free)
    @test long_patch(operator)
    @test !long_patch(construct(lower, (2.5, 0.4), ends_free))
    peak(A) = maximum(abs, nonzeros(A); init = 0.0)
    for (field, viscosity) in ((crossing, (0.35, 1.9)), (lower, (2.5, 0.4)))
        TwoPhaseAMR._update_dynamic_viscosity!(operator, field, viscosity)
        fresh = construct(field, viscosity, gap_free)
        for name in (:strain, :stiffness, :wall_reaction_strain)
            actual, expected = getproperty(operator, name), getproperty(fresh, name)
            @test actual.colptr == expected.colptr
            @test rowvals(actual) == rowvals(expected)
            @test peak(actual - expected) <= 512eps() * max(1.0, peak(expected))
        end
        no_slip = construct(field, viscosity, NoSlip())
        loss = dropzeros!(no_slip.stiffness - operator.stiffness)
        support = findall(c -> !isempty(nzrange(loss, c)), axes(loss, 2))
        @test 0 < length(support) <= 16n
        @test _pw_eigmin(loss[support, support]) >= -1024eps() * peak(no_slip.stiffness)
    end
    # Two threads, bounds checking: 0 B serial; 48 B on Julia 1.11 and
    # 0 B on 1.12 for both default and forced threaded grains. Allow the
    # same 64 B launcher bound as the threaded maximum test on Julia 1.11.
    default_grain = TwoPhaseAMR._PARALLEL_GRAIN[]
    for grain in (typemax(Int), default_grain, 1)
        allocated = _pw_thin_gap_refresh_allocations(operator, lower, crossing, grain)
        if grain == typemax(Int) || Threads.nthreads(:default) == 1 || VERSION >= v"1.12"
            @test allocated == 0
        else
            @test allocated <= 64
        end
    end
end

@testset "Per-side walls key the variable diffusion hierarchy" begin
    _, fields = _pw_fields()
    fluid = first(fields)
    density = TwoPhaseAMR.mixture_density(fluid, (1.0, 0.25))
    multigrid = MultigridOptions(coarse_size = 32)
    solver(boundary; reuse = nothing) = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        StaggeredDiffusionOperator(fluid; viscosity = (2.5, 0.4), boundary), density, 0.02;
        multigrid, reuse_preconditioner = reuse,
    )
    sides = _pw_sides(FreeSlip(), FreeSlip(), NoSlip(), NoSlip())
    transposed = _pw_sides(NoSlip(), NoSlip(), FreeSlip(), FreeSlip())
    first_solver = solver(sides)
    @test length(first_solver.preconditioner.operators) > 1
    @test all(level -> level.boundary === sides, first_solver.preconditioner.operators)
    # Another side configuration rebuilds every coarse level for its sides.
    other = solver(transposed; reuse = first_solver.preconditioner)
    @test all(level -> level.boundary === transposed, other.preconditioner.operators)
    reference = solver(transposed)
    for (level, expected) in zip(other.preconditioner.operators, reference.preconditioner.operators)
        @test level.stiffness == expected.stiffness
    end
end

@testset "Two-phase flows with per-side walls for every velocity model" begin
    # A bubble above a sloping solid between free-slip side walls, with a
    # no-slip floor and lid and a no-slip or free-slip solid; the first step
    # refines the mesh around the bubble and rebuilds the operator on the new
    # mesh.
    sides = _pw_sides(FreeSlip(), FreeSlip(), NoSlip(), NoSlip())
    free_solid = _pw_sides(FreeSlip(), FreeSlip(), NoSlip(), NoSlip(); embedded = FreeSlip())
    energies = Dict{Tuple{Int, Symbol}, Float64}()
    models = (
        OneVelocityModel(),
        TwoVelocityModel(pressure_projection = CMDGFMProjection()),
        TwoVelocityModel(pressure_projection = MimeticProjection()),
    )
    walls = ((:sides, sides), (:free_solid, free_solid), (:no_slip, NoSlip()), (:free_slip, FreeSlip()))
    for (index, velocity_model) in pairs(models), (name, boundary) in walls
        mesh = BlockTreeMesh(;
            base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
            periodic = (false, false),
        )
        geometry = GeometryField(mesh, (x, y) -> y - 0.12 - 0.1x; lipschitz = hypot(1, 0.1))
        interface = InterfaceField(mesh, (x, y) -> 0.2 - hypot(x - 0.5, y - 0.45); geometry, lipschitz = 1)
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (10.0, 1.0), dynamic_viscosity = (0.1, 0.01),
                surface_tension = 0.05, gravity = (0.0, -1.0),
            );
            geometry, velocity_model, boundary,
            adaptivity = Adaptivity(criteria = (InterfaceRefinementIndicator(interface; level = 1),)),
        )
        for _ in 1:3
            # The first step refines the mesh, which lowers the stable step.
            dt = 0.5 * stable_timestep(flow)
            simulate!(flow, flow.time + dt; dt, tolerance = 1.0e-10)
        end
        values = flow.velocity isa TwoPhaseVelocity ?
            [value for phase in flow.velocity for value in phase.interior if !isnan(value)] :
            flow.velocity.interior
        @test all(isfinite, values)
        energies[(index, name)] = sum(abs2, values)
        boundary isa VelocityBoundaryConditions || continue
        @test flow.mesh_updates > 0
        operator = flow.diffusion_operator
        @test operator.boundary === boundary
        hierarchy = flow.workspace.two_phase.diffusion_solver.preconditioner
        @test hierarchy isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
        @test length(hierarchy.operators) > 1
        @test all(level -> level.boundary === boundary, hierarchy.operators)
        fresh = StaggeredDiffusionOperator(flow.interface; viscosity = (0.1, 0.01), boundary)
        @test maximum(abs, nonzeros(operator.stiffness - fresh.stiffness)) <=
            64eps() * maximum(abs, nonzeros(fresh.stiffness))
    end
    # The side and solid conditions are in effect: every configuration
    # gives a different flow.
    for index in eachindex(models)
        @test allunique(energies[(index, name)] for (name, _) in walls)
    end
end

@testset "Per-side walls with a contact line on the no-slip floor" begin
    # The wall contact angles are curvature conditions per side and do not
    # read the velocity conditions; the drop spreads on a no-slip floor.
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), periodic = (false, false))
    sides = _pw_sides(FreeSlip(), FreeSlip(), NoSlip(), NoSlip())
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y + 0.1) - 0.3;
        contact_angle = (pi / 2, pi / 2, pi / 3, pi / 2), lipschitz = 1,
    )
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (10.0, 1.0), dynamic_viscosity = (0.1, 0.01),
            surface_tension = 0.05,
        );
        boundary = sides,
    )
    for _ in 1:3
        dt = 0.5 * stable_timestep(flow)
        simulate!(flow, flow.time + dt; dt, tolerance = 1.0e-10)
    end
    @test flow.diffusion_operator.boundary === sides
    @test all(isfinite, flow.velocity.interior)
    @test maximum(abs, flow.velocity.interior) > 0
end
