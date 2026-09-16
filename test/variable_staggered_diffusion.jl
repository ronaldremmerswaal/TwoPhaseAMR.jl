using Test, TwoPhaseAMR, LinearAlgebra, SparseArrays
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
        mul!(action_x, action, x)
        @test @allocated(mul!(action_x, action, x)) == 0
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
        coarse_size = 64,
    )
    preconditioner = solver.preconditioner
    @test preconditioner isa
        TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test length(preconditioner.levels) > 2
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
        density, rtol = 1.0e-9, max_cycles = 300,
    )
    @test solver.stats.converged

    moved = InterfaceField(
        mesh, (x, y) -> x - 0.15y - 0.43;
        lipschitz = hypot(1, 0.15),
    )
    TwoPhaseAMR._update_dynamic_viscosity!(
        operator, moved, (2.0e-2, 2.0e-5),
    )
    TwoPhaseAMR._update_variable_diffusion_solver!(
        solver, operator, reverse(density), 1.2dt,
    )
    @test solver.preconditioner === preconditioner
    right_hand_side = copy(solver.right_hand_side)
    solution = similar(right_hand_side)
    ldiv!(solution, preconditioner, right_hand_side)
    @test @allocated(
        ldiv!(solution, preconditioner, right_hand_side)
    ) == 0

    uniform_mesh = BlockTreeMesh(block_size = (8, 8))
    uniform_interface = InterfaceField(
        uniform_mesh, (x, y) -> x - 0.5; lipschitz = 1,
    )
    uniform_operator = StaggeredDiffusionOperator(
        uniform_interface; viscosity = (1.0e-2, 1.0e-5),
    )
    within_block = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        uniform_operator, ones(length(uniform_mesh.faces)), dt;
        coarse_size = 16,
    )
    @test within_block.preconditioner isa
        TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    @test length(within_block.preconditioner.levels) == 1
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
