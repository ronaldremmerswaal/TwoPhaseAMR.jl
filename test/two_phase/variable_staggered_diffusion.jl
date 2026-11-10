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

@testset "Local shear patches survive remote AMR changes" begin
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
    @test keytype(cache.derivative_templates) ===
        TwoPhaseAMR.ShearTemplateKey{Float64}
    initial_misses = cache.patch_misses
    @test initial_misses > 0
    @test cache.patch_hits == 0

    adaptation = AMRWorkspace(mesh, viscosity)
    labels = [
        block.key == BlockKey(2, 0, 0) ? :refine : :keep
            for block in mesh.blocks
    ]
    stats = adapt!(adaptation, labels; step = 1, force = true)
    @test stats.changed
    initialize!(viscosity, viscosity_function)

    old_hits = cache.patch_hits
    old_misses = cache.patch_misses
    old_invalidations = cache.patch_invalidations
    cached = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, construction_cache = cache, reuse = original,
    )
    hits = cache.patch_hits - old_hits
    misses = cache.patch_misses - old_misses
    @test hits > 0
    @test misses > 0
    @test hits + misses > initial_misses
    @test cache.patch_invalidations > old_invalidations
    @test cache.patch_remap_failures == 0

    fresh = StaggeredDiffusionOperator(mesh; viscosity)
    @test cached.strain == fresh.strain
    @test cached.stress_interpolation == fresh.stress_interpolation
    @test cached.stress_measures == fresh.stress_measures
    @test cached.stress_weights == fresh.stress_weights
    @test cached.stiffness == fresh.stiffness

    # A second construction on the identical tree must reuse every active
    # vertex/component patch, including empty free-slip wall patches.
    same_hits = cache.patch_hits
    same_misses = cache.patch_misses
    repeated = TwoPhaseAMR._staggered_diffusion_operator(
        mesh; viscosity, construction_cache = cache, reuse = cached,
    )
    @test cache.patch_hits > same_hits
    @test cache.patch_misses == same_misses
    @test repeated.strain == fresh.strain
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

@testset "Thesis geometric dynamic-viscosity mixture" begin
    for kind in 1:2
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
        viscosity = dynamic_viscosity(interface; viscosity = phases)
        for ref in TwoPhaseAMR._cell_references(mesh)
            c = TwoPhaseAMR._get(interface.fraction, ref)
            expected = phases[1]^c * phases[2]^(1 - c)
            @test TwoPhaseAMR._get(viscosity, ref) ≈ expected rtol = 16eps()
        end
        from_interface = StaggeredDiffusionOperator(
            interface; viscosity = phases
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
            mesh; viscosity
        )
        @test from_interface.stiffness ≈ from_field.stiffness rtol = 16eps()
        fractions = [
            TwoPhaseAMR._get(interface.fraction, ref)
                for ref in TwoPhaseAMR._cell_references(mesh)
        ]
        stress_fractions = from_interface.stress_interpolation * fractions
        observed = from_interface.stress_weights ./
            from_interface.stress_measures
        expected = phases[1] .^ stress_fractions .* phases[2] .^
            (1 .- stress_fractions)
        @test observed ≈ expected rtol = 32eps()
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
    return sqrt(
        sum(operator.volumes .* (solution .- exact.interior) .^ 2) /
            sum(operator.volumes)
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
    # The conforming bulk discretization is second order (tested below). A
    # fixed nonconforming transition occupies O(h) area and gives first-order
    # global velocity convergence for this weak stress discretization.
    for kind in (:fixed, :adaptive)
        errors = [
            _variable_stress_shifted_error(n, kind)
                for n in (6, 12, 24)
        ]
        rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
        @test all(diff(errors) .< 0)
        @test all(rates .>= 0.8)
    end
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
                scalar_no_slip = StaggeredDiffusionOperator(
                    geometry; viscosity = 1.0, boundary = NoSlip(),
                )
                scalar_free_slip = StaggeredDiffusionOperator(
                    geometry; viscosity = 1.0, boundary = FreeSlip(),
                )
                variable_no_slip = TwoPhaseAMR._staggered_diffusion_operator(
                    mesh; viscosity, boundary = NoSlip(), geometry,
                )
                variable_free_slip = TwoPhaseAMR._staggered_diffusion_operator(
                    mesh; viscosity, boundary = FreeSlip(), geometry,
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
