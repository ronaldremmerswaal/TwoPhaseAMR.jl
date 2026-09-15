function _maximum_divergence(mesh, velocity)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    return maximum(
        maximum(abs, interior(divergence, block))
            for block in eachindex(mesh.blocks)
    )
end

function _component_momenta(velocity)
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return (
        integrate(x, StaggeredAdvectionWorkspace(x)),
        integrate(y, StaggeredAdvectionWorkspace(y)),
    )
end

function _face_momentum(field::FaceField)
    return integrate(field, StaggeredAdvectionWorkspace(field))
end

@testset "interpolation-error refinement indicators" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2)
    constant = CellField(mesh; init = 2.0)
    oscillatory = CellField(mesh)
    initialize!(oscillatory, (x, y) -> 1 + 0.5sinpi(4x) * cospi(4y))
    workspace = AMRWorkspace(mesh, constant, oscillatory)

    coarsen = InterpolationErrorIndicator(
        constant; E_refine = 0.2, E_coarsen = 0.1
    )
    keep = InterpolationErrorIndicator(
        constant; E_refine = 0.2, E_coarsen = 0.0
    )
    refine = InterpolationErrorIndicator(
        oscillatory; E_refine = 0.01, E_coarsen = 0.005
    )
    @test refinement_labels!(workspace, coarsen) == [:coarsen]
    @test refinement_labels!(workspace, (coarsen, keep)) == [:keep]
    @test refinement_labels!(workspace, (coarsen, keep, refine)) == [:refine]

    @test_throws ArgumentError InterpolationErrorIndicator(
        constant; E_refine = 0.1, E_coarsen = 0.1
    )
    @test_throws ArgumentError InterpolationErrorIndicator(
        constant; E_refine = 0.1, E_coarsen = -0.1
    )
end

@testset "vorticity refinement indicator" begin
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) : -sinpi(2x)
    )
    indicator = VorticityRefinementIndicator(
        velocity; refine_tolerance = 0.5,
        coarsen_tolerance = 0.1, maximum_level = 1
    )
    workspace = AMRWorkspace(mesh, velocity)
    @test refinement_labels!(workspace, indicator) == [:refine]
    @test adapt!(workspace, indicator; step = 1, force = true).changed
    @test all(block -> block.key.level == 1, mesh.blocks)
    fill!(velocity.interior, 0)
    fill!(velocity.boundary, 0)
    @test refinement_labels!(workspace, indicator) ==
        fill(:coarsen, nblocks(mesh))
    @test_throws ArgumentError VorticityRefinementIndicator(
        velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.1
    )
end

@testset "in-place conservative adaptive field transfer" begin
    options = MeshAdaptationOptions(
        update_interval = 2, coarsen_delay = 3, minimum_level = 0
    )
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2, adaptation = options
    )
    density = CellField(mesh)
    tracer = CellField(mesh)
    initialize!(density, (x, y) -> 2 + 0.2sinpi(2x) * cospi(2y))
    initialize!(tracer, (x, y) -> 1 + x + y)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    )
    face_x = FaceField{1}(
        mesh, (x, y) -> 0.5 + 0.1sinpi(2x) * cospi(2y)
    )
    face_y = FaceField{2}(
        mesh, (x, y) -> -0.25 + 0.2cospi(2x) * sinpi(2y)
    )
    workspace = AMRWorkspace(
        mesh, density, tracer, velocity, face_x, face_y
    )

    cell_integrals = (integrate(density), integrate(tracer))
    velocity_momentum = _component_momenta(velocity)
    face_momentum = (_face_momentum(face_x), _face_momentum(face_y))
    initial_divergence = _maximum_divergence(mesh, velocity)

    refined = adapt!(workspace, [:refine]; step = 0)
    @test refined == AMRStats(true, true, 1, 4, 1, 0)
    @test all(block -> block.key.level == 1, mesh.blocks)
    @test isbalanced(mesh)
    @test integrate(density) ≈ cell_integrals[1] atol = 8.0e-14
    @test integrate(tracer) ≈ cell_integrals[2] atol = 8.0e-14
    @test all(
        isapprox.(
            _component_momenta(velocity), velocity_momentum; atol = 2.0e-13
        )
    )
    @test all(
        isapprox.(
            (
                _face_momentum(face_x), _face_momentum(face_y),
            ),
            face_momentum; atol = 2.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <=
        initial_divergence + 2.0e-13
    first_child_storage = Set(objectid.(density.blocks))

    skipped = adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 1)
    @test !skipped.attempted
    cooling_down = adapt!(
        workspace, fill(:coarsen, nblocks(mesh)); step = 2
    )
    @test cooling_down.attempted
    @test !cooling_down.changed
    skipped_again = adapt!(
        workspace, fill(:coarsen, nblocks(mesh)); step = 3
    )
    @test !skipped_again.attempted
    coarsened = adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 4)
    @test coarsened == AMRStats(true, true, 4, 1, 0, 1)
    @test integrate(density) ≈ cell_integrals[1] atol = 8.0e-14
    @test integrate(tracer) ≈ cell_integrals[2] atol = 8.0e-14
    @test all(
        isapprox.(
            _component_momenta(velocity), velocity_momentum; atol = 2.0e-13
        )
    )
    @test all(
        isapprox.(
            (
                _face_momentum(face_x), _face_momentum(face_y),
            ),
            face_momentum; atol = 2.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <=
        initial_divergence + 3.0e-13

    # The next refinement reuses the four child matrices retained by the
    # field buffer instead of allocating another set of block payloads.
    rerefined = adapt!(workspace, [:refine]; step = 6)
    @test rerefined.changed
    @test Set(objectid.(density.blocks)) == first_child_storage
end


@testset "face prolongation is affine exact" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    xface = FaceField{1}(mesh, (x, y) -> 1 + 2x - 3y)
    yface = FaceField{2}(mesh, (x, y) -> -2 + 4x + y)
    workspace = AMRWorkspace(mesh, xface, yface)
    stats = adapt!(workspace, [:refine]; step = 0)
    @test stats.changed

    xerror = maximum(
        abs(
            xface.interior[index] - (1 + 2face.center[1] - 3face.center[2])
        )
            for (index, face) in enumerate(filter(f -> f.axis == 1, mesh.faces))
    )
    yerror = maximum(
        abs(
            yface.interior[index] - (-2 + 4face.center[1] + face.center[2])
        )
            for (index, face) in enumerate(filter(f -> f.axis == 2, mesh.faces))
    )
    @test xerror <= 5.0e-14
    @test yerror <= 5.0e-14
end


@testset "Navier--Stokes optionally adapts and rebuilds its operators" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0)
    )
    marker = CellField(mesh)
    initialize!(marker, (x, y) -> 1 + 0.2sinpi(4x) * sinpi(4y))
    passive_face = FaceField{1}(mesh, (x, y) -> x - y)
    pressure = CellField(mesh)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    indicator = InterpolationErrorIndicator(
        marker; E_refine = 1.0e-3, E_coarsen = 1.0e-4
    )
    solver = NavierStokesSolver(
        mesh; viscosity = 0.01, boundary = FreeSlip(), coarse_size = 8,
        refinement_indicators = indicator,
        adaptive_fields = (passive_face,)
    )
    workspace = NavierStokesWorkspace(solver)

    refined = simulate!(
        velocity, pressure, solver, 0.01, workspace;
        dt = 0.01, rtol = 1.0e-10, max_cycles = 30
    )
    @test refined.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test length(pressure.blocks) == nblocks(mesh)
    @test length(passive_face.interior) ==
        count(face -> face.axis == 1, mesh.faces)
    @test length(workspace.projection.elliptic.input) == ncells(mesh)
    @test length(solver.pressure_solver.operator.volumes) == ncells(mesh)

    initialize!(marker, (x, y) -> 1.0)
    coarsened = simulate!(
        velocity, pressure, solver, 0.02, workspace;
        time = 0.01, dt = 0.01, rtol = 1.0e-10, max_cycles = 30
    )
    @test coarsened.mesh_updates == 1
    @test nblocks(mesh) == 1
    @test mesh.adaptation.completed_steps == 2

    registry = VisualizationFieldRegistry()
    register_field!(registry, :pressure, pressure)
    exact = VisualizationSession(
        registry; checkpoint_times = [0.025], exact_checkpoints = true
    )
    @test_throws ArgumentError simulate!(
        velocity, pressure, solver, 0.03, workspace;
        time = 0.02, dt = 0.01, visualization = exact,
        rtol = 1.0e-10, max_cycles = 30
    )
end

@testset "adaptive refinement remains 2:1 graded" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 3,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    )
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> 1 + 0.1cospi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    )
    workspace = AMRWorkspace(mesh, scalar, velocity)
    mass = integrate(scalar)
    momentum = _component_momenta(velocity)
    divergence = _maximum_divergence(mesh, velocity)

    labels = fill(:keep, nblocks(mesh))
    selected = findfirst(block -> block.key == BlockKey(2, 0, 0), mesh.blocks)
    labels[selected] = :refine
    stats = adapt!(workspace, labels; step = 0)
    @test stats.changed
    @test stats.refined_blocks >= 1
    @test maximum(block.key.level for block in mesh.blocks) == 3
    @test isbalanced(mesh)
    @test integrate(scalar) ≈ mass atol = 2.0e-13
    @test all(
        isapprox.(
            _component_momenta(velocity), momentum; atol = 3.0e-13
        )
    )
    @test _maximum_divergence(mesh, velocity) <= divergence + 5.0e-12

    # A keep request in a sibling family prevents the other three coarsening
    # requests from removing that family.
    family = BlockKey(2, 0, 0)
    children = TwoPhaseAMR._child_keys(family)
    labels = fill(:keep, nblocks(mesh))
    for (index, block) in pairs(mesh.blocks)
        block.key in children && (labels[index] = :coarsen)
    end
    kept_child = findfirst(block -> block.key == children[1], mesh.blocks)
    labels[kept_child] = :keep
    kept = adapt!(workspace, labels; step = 1)
    @test !kept.changed
    @test all(child -> any(block -> block.key == child, mesh.blocks), children)
end

@testset "mesh adaptation option validation" begin
    @test_throws ArgumentError MeshAdaptationOptions(update_interval = 0)
    @test_throws ArgumentError MeshAdaptationOptions(coarsen_delay = -1)
    @test_throws ArgumentError MeshAdaptationOptions(minimum_level = -1)
    @test_throws ArgumentError BlockTreeMesh(
        max_level = 1,
        adaptation = MeshAdaptationOptions(minimum_level = 2)
    )
end
