using Test, TwoPhaseAMR, Krylov
using LinearAlgebra: Diagonal, mul!, norm
using Random: MersenneTwister, randn
@testset "One Navier-Stokes API, transported density jump" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> x - 0.43)
        indicator = InterfaceRefinementIndicator(f; level = 1)
        q = PhaseScalar(f; liquid = 1.0, gas = 1.0)
        flow = FlowSimulation(
            mesh; interface = f, dynamic_viscosity = (0.0, 0.0),
            density = (1000.0, 1.0),
            velocity = (0.1, -0.05), phase_scalars = (q,),
            refinement_indicators = kind == 3 ? (indicator,) : ()
        )
        @test flow.interface === f
        @test :interface in propertynames(flow)
        @test isnothing(flow.diffusion_operator)
        volume = integrate(f)
        simulate!(flow, 0.02; dt = 0.002, tolerance = 1.0e-11)
        @test integrate(f) ≈ volume atol = 5.0e-13
        for (i, face) in pairs(mesh.faces)
            @test flow.velocity.interior[i] ≈ (face.axis == 1 ? 0.1 : -0.05) atol = 2.0e-10
        end
        for b in eachindex(mesh.blocks)
            @test interior(q.liquid, b) ≈ ones(size(interior(q.liquid, b))) atol = 2.0e-13
            @test interior(q.gas, b) ≈ ones(size(interior(q.gas, b))) atol = 2.0e-13
        end
        @test kind != 3 || nblocks(mesh) > 1
    end
end

@testset "Two-phase gravity-wave timestep safety factor" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), periodic = (true, false),
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.43; lipschitz = 1,
    )
    density = (1000.0, 1.0)
    gravity = 9.81
    flow = FlowSimulation(
        mesh; interface, density,
        dynamic_viscosity = (0.0, 0.0),
        gravity = (0.0, -gravity), boundary = FreeSlip(),
    )
    h = minimum(minimum(block.spacing) for block in mesh.blocks)
    expected = sqrt(
        sum(density) / abs(density[1] - density[2]) * h / (2gravity),
    )
    @test stable_timestep(flow; cfl = 0.35) ≈ expected rtol = 8eps()
    # Both solver-level paths should reuse the completed interface PLIC and
    # phase apertures without allocating on an unchanged topology.
    stable_timestep(flow; cfl = 0.35)
    @test @allocated(stable_timestep(flow; cfl = 0.35)) == 0
    intrinsic = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density, dynamic_viscosity = (0.0, 0.0),
            gravity = (0.0, -gravity),
        );
        boundary = FreeSlip(),
        velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection(),
        ),
    )
    stable_timestep(intrinsic; cfl = 0.35)
    @test stable_timestep(intrinsic; cfl = 0.35) ≈ expected rtol = 8eps()
    @test @allocated(stable_timestep(intrinsic; cfl = 0.35)) == 0
end

@testset "Intrinsic phase apertures are updated once per step" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    interface = InterfaceField(mesh)
    initialize!(interface, (1.0, 0.0), 0.47)
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (2.0, 1.0),
            dynamic_viscosity = (0.0, 0.0),
        );
        velocity = (0.05, 0.0), boundary = FreeSlip(),
        velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection(),
        ),
    )
    before = interface.phase_aperture_updates[]
    stable_timestep(flow)
    @test interface.phase_aperture_updates[] == before + 1
    simulate!(flow, 0.001; dt = 0.001)
    @test interface.phase_aperture_updates[] == before + 2
    stable_timestep(flow)
    @test interface.phase_aperture_updates[] == before + 2
    simulate!(flow, 0.002; dt = 0.001)
    @test interface.phase_aperture_updates[] == before + 3
end

@testset "Two-phase diffusion uses the thesis split state" begin
    mesh = BlockTreeMesh(block_size = (6, 6))
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.53; lipschitz = hypot(1, 0.2),
    )
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = (0.02, 0.001), boundary = FreeSlip(),
        diffusion_preconditioner = nothing,
    )
    phase = flow.workspace.two_phase
    @test isnothing(flow.solver.diffusion)
    @test flow.diffusion_operator === phase.diffusion
    @test phase.diffusion isa VariableStaggeredDiffusionOperator

    old_values = similar(flow.velocity.interior)
    split_values = similar(flow.velocity.interior)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        old_values[index] = face.axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
        split_values[index] = old_values[index] +
            0.2sinpi(2x + y + face.axis / 3)
    end
    copyto!(flow.workspace.old_velocity.interior, old_values)
    copyto!(flow.workspace.convected.interior, split_values)

    dt = 0.01
    operator = phase.diffusion
    TwoPhaseAMR._update_dynamic_viscosity!(
        operator, interface, flow.solver.two_phase.dynamic_viscosity,
    )
    density = mixture_density(interface, flow.solver.two_phase.density)
    inertia = density .* operator.volumes
    matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
    expected = matrix \ (
        inertia .* split_values -
            (dt / 2) * (operator.stiffness * split_values)
    )
    @test norm(operator.stiffness * (old_values - split_values)) > 0

    TwoPhaseAMR._diffusion_substep!(
        flow.velocity, flow.solver, flow.workspace, dt, 0.0,
        flow.solver.two_phase; tolerance = 1.0e-12, max_cycles = 500,
    )
    @test flow.velocity.interior ≈ expected rtol = 4096eps()
end

@testset "Live visualization follows two-phase AMR topology" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (true, true)
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 1)
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), velocity = (0.1, 0.0),
        refinement_indicators = (indicator,)
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :fraction, interface.fraction)
    session = VisualizationSession(registry; checkpoints = 2)
    result = simulate!(
        flow, 0.002; dt = 0.001, visualization = session,
        tolerance = 1.0e-10, max_cycles = 100
    )
    @test result.mesh_updates == 1
    @test nblocks(mesh) == 4
    @test isnothing(session.step_procedure)
    @test length(latest_frame(session).topology.cell_geometry.cells) ==
        ncells(mesh)
end

@testset "Topology rebuild reuses variable-diffusion construction storage" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (true, true),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 1)
    phase_viscosity = (0.02, 0.001)
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = phase_viscosity, velocity = (0.0, 0.0),
        refinement_indicators = (indicator,),
    )
    phase_workspace = flow.workspace.two_phase
    cache = phase_workspace.diffusion_cache
    strain_rows = cache.strain_rows
    stiffness_pointers = phase_workspace.diffusion.stiffness_pointers
    result = simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 100,
    )
    @test result.mesh_updates == 1
    @test flow.workspace.two_phase.diffusion_cache === cache
    @test flow.workspace.two_phase.diffusion_cache.strain_rows === strain_rows
    @test flow.workspace.two_phase.diffusion.stiffness_pointers ===
        stiffness_pointers
    cached = flow.workspace.two_phase.diffusion
    fresh = StaggeredDiffusionOperator(
        flow.interface; viscosity = phase_viscosity,
        boundary = flow.solver.boundary,
    )
    @test cached.stiffness == fresh.stiffness
    @test cached.strain == fresh.strain
    @test cached.stress_interpolation == fresh.stress_interpolation
    @test cached.stress_weights == fresh.stress_weights
end

@testset "Tree solvers retain matching hierarchy levels across AMR" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        periodic = (true, true),
        refine = (key, origin, extent) -> key.level == 0,
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 2)
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = (0.02, 0.001), velocity = (0.0, 0.0),
        refinement_indicators = (indicator,),
        coarse_size = 16,
    )
    @test flow.solver.pressure_preconditioner === :tree_geometric
    @test flow.solver.diffusion_preconditioner === :tree_geometric
    pressure_solver = flow.solver.pressure_solver
    phase_workspace = flow.workspace.two_phase
    diffusion_solver = phase_workspace.diffusion_solver
    density = phase_workspace.density
    pressure_preconditioner = pressure_solver.preconditioner
    diffusion_preconditioner = diffusion_solver.preconditioner
    diffusion_hierarchy_cache = diffusion_preconditioner.construction_cache
    pressure_krylov = pressure_solver.krylov
    diffusion_krylov = diffusion_solver.krylov
    wy_workspace = phase_workspace.advection
    momentum_x_values = phase_workspace.momentum.x.values
    pressure_geometry_cache =
        pressure_solver.operator.gradient_operator.construction_cache
    pressure_cache_hits = pressure_geometry_cache.hits
    diffusion_template_hits =
        phase_workspace.diffusion_cache.derivative_template_hits
    pressure_bottom = pressure_solver.preconditioner.operators[end]
    diffusion_bottom = diffusion_solver.preconditioner.operators[end]

    result = simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 100,
    )
    @test result.mesh_updates == 1
    @test flow.solver.pressure_solver === pressure_solver
    @test flow.workspace.two_phase.diffusion_solver === diffusion_solver
    @test flow.workspace.two_phase.density === density
    refreshed_pressure_preconditioner =
        flow.solver.pressure_solver.preconditioner
    refreshed_diffusion_preconditioner =
        flow.workspace.two_phase.diffusion_solver.preconditioner
    @test refreshed_pressure_preconditioner !== pressure_preconditioner
    @test refreshed_diffusion_preconditioner !== diffusion_preconditioner
    @test refreshed_pressure_preconditioner.reused_levels > 0
    @test refreshed_diffusion_preconditioner.reused_levels > 0
    @test refreshed_diffusion_preconditioner.construction_cache ===
        diffusion_hierarchy_cache
    @test flow.solver.pressure_solver.krylov === pressure_krylov
    @test flow.workspace.two_phase.diffusion_solver.krylov === diffusion_krylov
    @test flow.workspace.two_phase.advection === wy_workspace
    @test flow.workspace.two_phase.momentum.x.values === momentum_x_values
    @test flow.solver.pressure_solver.operator.gradient_operator.construction_cache ===
        pressure_geometry_cache
    @test pressure_geometry_cache.hits > pressure_cache_hits
    @test flow.workspace.two_phase.diffusion_cache.derivative_template_hits >=
        diffusion_template_hits
    @test refreshed_pressure_preconditioner.operators[end] ===
        pressure_bottom
    @test refreshed_diffusion_preconditioner.operators[end] ===
        diffusion_bottom

    fresh_pressure = PoissonOperator(
        interface; density = flow.solver.two_phase.density,
        formulation = flow.solver.two_phase.formulation,
    )
    @test size(flow.solver.pressure_solver.operator.matrix) ==
        (ncells(mesh), ncells(mesh))
    @test all(
        !iszero(size(level.matrix, 1))
            for level in refreshed_pressure_preconditioner.levels[2:end]
    )
    rng = MersenneTwister(0x2a17)
    pressure_source = randn(rng, length(fresh_pressure.volumes))
    cached_pressure_product = similar(pressure_source)
    fresh_pressure_product = similar(pressure_source)
    mul!(
        cached_pressure_product, flow.solver.pressure_solver.action,
        pressure_source,
    )
    mul!(
        fresh_pressure_product,
        TwoPhaseAMR.FactorizedEllipticOperator(fresh_pressure),
        pressure_source,
    )
    @test cached_pressure_product ≈ fresh_pressure_product rtol = 64eps()
    fresh_diffusion = StaggeredDiffusionOperator(
        interface; viscosity = flow.solver.two_phase.dynamic_viscosity,
        boundary = flow.solver.boundary,
    )
    @test size(flow.workspace.two_phase.diffusion.stiffness) ==
        (length(mesh.faces), length(mesh.faces))
    @test !isnothing(flow.workspace.two_phase.diffusion_solver.matrix)
    diffusion_source = randn(rng, length(fresh_diffusion.volumes))
    cached_diffusion_product = similar(diffusion_source)
    fresh_diffusion_product = similar(diffusion_source)
    cached_action = flow.workspace.two_phase.diffusion_solver.action
    fresh_action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
        fresh_diffusion, cached_action.inertia, 2cached_action.half_step,
    )
    mul!(cached_diffusion_product, cached_action, diffusion_source)
    mul!(fresh_diffusion_product, fresh_action, diffusion_source)
    @test cached_diffusion_product ≈ fresh_diffusion_product rtol = 64eps()
end


@testset "Two-phase pressure and diffusion solver configuration" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface = InterfaceField(mesh, (x, y) -> x - 0.47; lipschitz = 1)
    flow = FlowSimulation(
        mesh; interface, density = (1000.0, 1.0),
        dynamic_viscosity = (0.02, 0.001), boundary = FreeSlip(),
        velocity = (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) : sinpi(2x),
        diffusion_krylov = Krylov.GmresWorkspace,
        diffusion_preconditioner = nothing
    )
    @test flow.solver.pressure_krylov === Krylov.CgWorkspace
    @test flow.diffusion_operator isa VariableStaggeredDiffusionOperator
    old_pressure_preconditioner =
        flow.solver.pressure_solver.preconditioner
    old_diffusion_preconditioner =
        flow.workspace.two_phase.diffusion_solver.preconditioner
    old_diffusion_krylov =
        flow.workspace.two_phase.diffusion_solver.krylov
    simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 100
    )
    @test flow.solver.pressure_solver.preconditioner ===
        old_pressure_preconditioner
    @test flow.solver.pressure_solver.preconditioner.operator ===
        flow.solver.pressure_solver.operator
    diffusion_solver = flow.workspace.two_phase.diffusion_solver
    @test diffusion_solver.preconditioner === old_diffusion_preconditioner
    @test diffusion_solver.krylov === old_diffusion_krylov
    @test diffusion_solver.stats.converged
    @test diffusion_solver.stats.method === :gmres
    @test diffusion_solver.stats.preconditioner === :none

    single_phase = NavierStokesSolver(
        mesh; density = 1.0, dynamic_viscosity = 0.02,
        boundary = FreeSlip()
    )
    @test single_phase.pressure_krylov === Krylov.CgWorkspace
    @test isnothing(single_phase.diffusion_krylov)
    single_phase_workspace = NavierStokesWorkspace(single_phase)
    @test single_phase_workspace.elliptic.x.krylov isa Krylov.CgWorkspace
end

function _visualization_transport_flow(final_time; visualization = NoVisualization())
    mesh = BlockTreeMesh(block_size = (6, 6))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.35, y - 0.55) - 0.18;
        lipschitz = 1
    )
    scalar = PhaseScalar(
        interface; liquid = (x, y) -> 1 + x + 0.2y,
        gas = (x, y) -> 2 - 0.3x + y
    )
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), velocity = (0.08, -0.03),
        phase_scalars = (scalar,)
    )
    simulate!(
        flow, final_time; dt = 0.01, visualization,
        tolerance = 1.0e-10, max_cycles = 100
    )
    return flow, interface, scalar
end

@testset "Exact checkpoints isolate all two-phase state" begin
    baseline, baseline_interface, baseline_scalar =
        _visualization_transport_flow(0.02)

    mesh = BlockTreeMesh(block_size = (6, 6))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.35, y - 0.55) - 0.18;
        lipschitz = 1
    )
    scalar = PhaseScalar(
        interface; liquid = (x, y) -> 1 + x + 0.2y,
        gas = (x, y) -> 2 - 0.3x + y
    )
    flow = FlowSimulation(
        mesh; interface, density = (4.0, 1.0),
        dynamic_viscosity = (0.0, 0.0), velocity = (0.08, -0.03),
        phase_scalars = (scalar,)
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :fraction, interface.fraction)
    register_field!(registry, :liquid_scalar, scalar.liquid)
    session = VisualizationSession(
        registry; checkpoint_times = [0.005], exact_checkpoints = true
    )
    simulate!(
        flow, 0.02; dt = 0.01, visualization = session,
        tolerance = 1.0e-10, max_cycles = 100
    )
    reference, reference_interface, reference_scalar =
        _visualization_transport_flow(0.005)

    @test visualization_values(interface.fraction) ==
        visualization_values(baseline_interface.fraction)
    @test visualization_values(scalar.liquid) ==
        visualization_values(baseline_scalar.liquid)
    @test flow.velocity.interior == baseline.velocity.interior
    checkpoint = only(visualization_checkpoints(session)).frame
    @test checkpoint[:fraction] ==
        visualization_values(reference_interface.fraction)
    @test checkpoint[:liquid_scalar] ==
        visualization_values(reference_scalar.liquid)
end
@testset "Flat heavy interface remains hydrostatic" begin
    for kind in 1:3, gravity in ((0.0, -1.0), (0.6, -0.8))
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            periodic = gravity == (0.0, -1.0) ?
                (true, false) : (false, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        levelset = gravity == (0.0, -1.0) ?
            ((x, y) -> y - 0.43) :
            (
                (x, y) -> gravity[1] * (x - 0.5) +
                gravity[2] * (y - 0.5)
            )
        f = InterfaceField(mesh, levelset; lipschitz = 1)
        indicator = InterfaceRefinementIndicator(f; level = 1)
        flow = FlowSimulation(
            mesh; interface = f, dynamic_viscosity = (0.0, 0.0),
            density = (1000.0, 1.0),
            gravity, refinement_indicators = kind == 3 ? (indicator,) : ()
        )
        mass = integrate(f)
        simulate!(flow, 0.02; dt = 0.002, tolerance = 1.0e-11)
        @test maximum(abs, flow.velocity.interior) < 1.0e-9
        @test integrate(f) ≈ mass atol = 2.0e-13
    end
end
