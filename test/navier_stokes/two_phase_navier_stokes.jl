using Test, TwoPhaseAMR, Krylov
using LinearAlgebra: Diagonal, dot, mul!, norm
using Random: MersenneTwister, randn
isdefined(@__MODULE__, :test_allocated_call) ||
    include(joinpath(@__DIR__, "..", "test_utils.jl"))
isdefined(@__MODULE__, :_phase_storage_contract) ||
    include(joinpath(@__DIR__, "..", "two_phase", "two_velocity", "helpers.jl"))
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
            mesh,
            TwoPhase(
                f; dynamic_viscosity = (0.0, 0.0),
                density = (1000.0, 1.0), scalars = (q,),
            );
            velocity = (0.1, -0.05),
            adaptivity = Adaptivity(criteria = kind == 3 ? (indicator,) : ()),
        )
        @test flow.interface === f
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
        mesh,
        TwoPhase(
            interface; density, dynamic_viscosity = (0.0, 0.0),
            gravity = (0.0, -gravity),
        );
        boundary = FreeSlip(),
    )
    h = minimum(minimum(block.spacing) for block in mesh.blocks)
    expected = sqrt(
        sum(density) / abs(density[1] - density[2]) * h / (2gravity),
    )
    @test stable_timestep(flow; cfl = 0.35) ≈ expected rtol = 8eps()
    # Both solver-level paths should reuse the completed interface PLIC and
    # phase apertures without allocating on an unchanged topology.
    stable_timestep(flow; cfl = 0.35)
    @test test_allocated_call(() -> stable_timestep(flow; cfl = 0.35)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(flow; cfl = 0.35)) == 0
    end
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
    @test test_allocated_call(() -> stable_timestep(intrinsic; cfl = 0.35)) == 0
    if VERSION >= v"1.12"
        @test @allocated(stable_timestep(intrinsic; cfl = 0.35)) == 0
    end
end

@testset "Two-phase capillary and gravity bounds agree across velocity models" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    h = minimum(minimum(block.spacing) for block in mesh.blocks)
    cases = (
        ((8.0, 2.0), 0.7, (0.0, 0.0)),
        ((8.0, 2.0), 0.0, (3.0, -4.0)),
        ((8.0, 2.0), 0.7, (3.0, -4.0)),
        ((2.0, 8.0), 0.7, (3.0, -4.0)),
        ((2.0, 2.0), 0.7, (3.0, -4.0)),
        ((2.0, 2.0), 0.0, (3.0, -4.0)),
    )
    for (density, sigma, gravity) in cases
        interface = InterfaceField(mesh, (x, y) -> y - 0.43; lipschitz = 1)
        flows = map(
            (
                OneVelocityModel(),
                TwoVelocityModel(pressure_projection = CMDGFMProjection()),
            )
        ) do velocity_model
            FlowSimulation(
                mesh,
                TwoPhase(
                    interface; density, dynamic_viscosity = (0.0, 0.0),
                    surface_tension = sigma, gravity,
                );
                velocity_model, boundary = FreeSlip(),
            )
        end
        bounds = map(flow -> stable_timestep(flow; cfl = 0.35), flows)
        expected = Inf
        if sigma > 0
            expected = min(expected, sqrt(sum(density) * h^3 / (4pi * sigma)))
        end
        g = hypot(gravity...)
        if g > 0 && density[1] != density[2]
            expected = min(
                expected,
                sqrt(sum(density) / abs(density[1] - density[2]) * h / (2g)),
            )
        end
        @test all(bound -> isequal(bound, expected), bounds)
        if isfinite(expected)
            capillary = sigma > 0 ? sqrt(sum(density) * h^3 / (4pi * sigma)) : Inf
            name = expected == capillary ? "capillary" : "gravity"
            for flow in flows
                error = try
                    step!(
                        flow.velocity, flow.pressure, flow.system, 1.01expected,
                        flow.workspace,
                    )
                    nothing
                catch exception
                    exception
                end
                @test error isa ArgumentError
                if error isa ArgumentError
                    @test occursin("exceeds $name stability bound", sprint(showerror, error))
                end
            end
        end
    end
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
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.02, 0.001),
        );
        boundary = FreeSlip(),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(preconditioner = nothing),
        ),
    )
    phase = flow.workspace.two_phase
    @test isnothing(flow.system.diffusion)
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
        operator, interface, flow.system.two_phase.dynamic_viscosity,
    )
    density = mixture_density(interface, flow.system.two_phase.density)
    inertia = density .* operator.volumes
    matrix = Diagonal(inertia) + (dt / 2) * operator.stiffness
    expected = matrix \ (
        inertia .* split_values -
            (dt / 2) * (operator.stiffness * split_values)
    )
    @test norm(operator.stiffness * (old_values - split_values)) > 0

    TwoPhaseAMR._diffusion_substep!(
        flow.velocity, flow.system, flow.workspace, dt, 0.0,
        flow.system.two_phase; tolerance = 1.0e-12, max_cycles = 500,
    )
    @test flow.velocity.interior ≈ expected rtol = 4096eps()
end

@testset "Recording follows two-phase AMR topology" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        periodic = (true, true)
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 1)
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.0, 0.0),
        );
        velocity = (0.1, 0.0), adaptivity = Adaptivity(criteria = (indicator,)),
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :fraction, interface.fraction)
    recorder = Recorder(registry; keyframes = EquispacedTimes(2), preview = nothing)
    result = simulate!(
        flow, 0.002; dt = 0.001, observer = recorder,
        tolerance = 1.0e-10, max_cycles = 100
    )
    @test result.mesh_updates == 1
    @test nblocks(mesh) == 4
    store = snapshot_store(recorder)
    @test length(newest_frame(store)[:fraction]) == ncells(mesh)
    @test store_usage(store).topologies == 2
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
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = phase_viscosity,
        );
        velocity = (0.0, 0.0), adaptivity = Adaptivity(criteria = (indicator,)),
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
        boundary = flow.system.boundary,
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
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.02, 0.001),
        );
        velocity = (0.0, 0.0), adaptivity = Adaptivity(criteria = (indicator,)),
        numerics = NavierStokesOptions(multigrid = MultigridOptions(coarse_size = 16)),
    )
    @test flow.system.pressure_solver.preconditioner isa TreeGeometricMultigridPreconditioner
    @test flow.workspace.two_phase.diffusion_solver.preconditioner isa TwoPhaseAMR.TreeVariableDiffusionPreconditioner
    pressure_solver = flow.system.pressure_solver
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
    pressure_bottom = pressure_solver.preconditioner.operators[end]
    diffusion_bottom = diffusion_solver.preconditioner.operators[end]

    result = simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 100,
    )
    @test result.mesh_updates == 1
    @test flow.system.pressure_solver === pressure_solver
    @test flow.workspace.two_phase.diffusion_solver === diffusion_solver
    @test flow.workspace.two_phase.density === density
    refreshed_pressure_preconditioner =
        flow.system.pressure_solver.preconditioner
    refreshed_diffusion_preconditioner =
        flow.workspace.two_phase.diffusion_solver.preconditioner
    @test refreshed_pressure_preconditioner !== pressure_preconditioner
    @test refreshed_diffusion_preconditioner !== diffusion_preconditioner
    @test refreshed_pressure_preconditioner.reused_levels > 0
    @test refreshed_diffusion_preconditioner.reused_levels > 0
    @test refreshed_diffusion_preconditioner.construction_cache ===
        diffusion_hierarchy_cache
    @test flow.system.pressure_solver.krylov === pressure_krylov
    @test flow.workspace.two_phase.diffusion_solver.krylov === diffusion_krylov
    @test flow.workspace.two_phase.advection === wy_workspace
    @test flow.workspace.two_phase.momentum.x.values === momentum_x_values
    @test flow.system.pressure_solver.operator.gradient_operator.construction_cache ===
        pressure_geometry_cache
    @test pressure_geometry_cache.hits > pressure_cache_hits
    @test refreshed_pressure_preconditioner.operators[end] ===
        pressure_bottom
    @test refreshed_diffusion_preconditioner.operators[end] ===
        diffusion_bottom

    fresh_pressure = PoissonOperator(
        interface; density = flow.system.two_phase.density,
    )
    @test size(flow.system.pressure_solver.operator.matrix) ==
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
        cached_pressure_product, flow.system.pressure_solver.action,
        pressure_source,
    )
    mul!(
        fresh_pressure_product,
        TwoPhaseAMR.FactorizedEllipticOperator(fresh_pressure),
        pressure_source,
    )
    @test cached_pressure_product ≈ fresh_pressure_product rtol = 64eps()
    fresh_diffusion = StaggeredDiffusionOperator(
        interface; viscosity = flow.system.two_phase.dynamic_viscosity,
        boundary = flow.system.boundary,
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
        mesh,
        TwoPhase(
            interface; density = (1000.0, 1.0),
            dynamic_viscosity = (0.02, 0.001),
        );
        boundary = FreeSlip(),
        velocity = (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) : sinpi(2x),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(
                krylov = Krylov.GmresWorkspace, preconditioner = nothing,
            ),
        ),
    )
    @test flow.system.pressure_solver.krylov isa Krylov.CgWorkspace
    @test flow.diffusion_operator isa VariableStaggeredDiffusionOperator
    old_pressure_preconditioner =
        flow.system.pressure_solver.preconditioner
    old_diffusion_preconditioner =
        flow.workspace.two_phase.diffusion_solver.preconditioner
    old_diffusion_krylov =
        flow.workspace.two_phase.diffusion_solver.krylov
    simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 100
    )
    @test flow.system.pressure_solver.preconditioner ===
        old_pressure_preconditioner
    @test flow.system.pressure_solver.preconditioner.operator ===
        flow.system.pressure_solver.operator
    diffusion_solver = flow.workspace.two_phase.diffusion_solver
    @test diffusion_solver.preconditioner === old_diffusion_preconditioner
    @test diffusion_solver.krylov === old_diffusion_krylov
    @test diffusion_solver.stats.converged
    @test diffusion_solver.stats.method === :gmres
    @test diffusion_solver.stats.preconditioner === :none

    single_phase = NavierStokesSystem(
        mesh; density = 1.0, dynamic_viscosity = 0.02,
        boundary = FreeSlip()
    )
    @test single_phase.pressure_solver.krylov isa Krylov.CgWorkspace
    @test isnothing(single_phase.options.diffusion.krylov)
    single_phase_workspace = NavierStokesWorkspace(single_phase)
    @test single_phase_workspace.elliptic.x.krylov isa Krylov.CgWorkspace
end

function _recorded_transport_flow(final_time; observer = NoObserver())
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
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.0, 0.0), scalars = (scalar,),
        );
        velocity = (0.08, -0.03),
    )
    simulate!(
        flow, final_time; dt = 0.01, observer,
        tolerance = 1.0e-10, max_cycles = 100
    )
    return flow, interface, scalar
end

@testset "Exact keyframes isolate all two-phase state" begin
    baseline, baseline_interface, baseline_scalar =
        _recorded_transport_flow(0.02)

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
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.0, 0.0), scalars = (scalar,),
        );
        velocity = (0.08, -0.03),
    )
    registry = VisualizationFieldRegistry()
    register_field!(registry, :fraction, interface.fraction)
    register_field!(registry, :liquid_scalar, scalar.liquid)
    recorder = Recorder(
        registry; keyframes = AtTimes([0.005]), exact_keyframes = true,
        preview = nothing,
    )
    simulate!(
        flow, 0.02; dt = 0.01, observer = recorder,
        tolerance = 1.0e-10, max_cycles = 100
    )
    reference, reference_interface, reference_scalar =
        _recorded_transport_flow(0.005)

    @test visualization_values(interface.fraction) ==
        visualization_values(baseline_interface.fraction)
    @test visualization_values(scalar.liquid) ==
        visualization_values(baseline_scalar.liquid)
    @test flow.velocity.interior == baseline.velocity.interior
    store = snapshot_store(recorder)
    keyframe = store[only(filter(info -> info.time == 0.005, frames(store)))]
    @test keyframe.requested_times == [0.005]
    @test keyframe[:fraction] ==
        visualization_values(reference_interface.fraction)
    @test keyframe[:liquid_scalar] ==
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
            mesh,
            TwoPhase(
                f; dynamic_viscosity = (0.0, 0.0),
                density = (1000.0, 1.0), gravity,
            );
            adaptivity = Adaptivity(criteria = kind == 3 ? (indicator,) : ()),
        )
        mass = integrate(f)
        simulate!(flow, 0.02; dt = 0.002, tolerance = 1.0e-11)
        @test maximum(abs, flow.velocity.interior) < 1.0e-9
        @test integrate(f) ≈ mass atol = 2.0e-13
    end
end

function _two_velocity_resume_flow()
    mesh = BlockTreeMesh(
        block_size = (4, 4), periodic = (true, true), max_level = 1,
        adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = 0),
    )
    interface = InterfaceField(mesh, (x, y) -> y - 0.43; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 1, t_init = 0.3)
    return FlowSimulation(
        mesh, TwoPhase(
            interface; density = (2.0, 1.0), dynamic_viscosity = (0.0, 0.0),
        );
        velocity_model = TwoVelocityModel(), velocity = (1.0, 0.0),
        adaptivity = Adaptivity(criteria = (indicator,)),
        diagnostics = (KineticEnergy(), PhaseVolume()),
    )
end

function _check_two_velocity_resume_fields(reference, resumed)
    @test [b.key for b in resumed.system.mesh.blocks] ==
        [b.key for b in reference.system.mesh.blocks]
    for phase in 1:2
        # Absent-phase velocities are NaNs and must remain absent on both runs.
        @test isequal(
            resumed.velocity[phase].interior, reference.velocity[phase].interior,
        )
    end
    for b in eachindex(reference.pressure.blocks)
        @test interior(resumed.pressure, b) == interior(reference.pressure, b)
        @test interior(resumed.interface.fraction, b) ==
            interior(reference.interface.fraction, b)
    end
    return nothing
end

@testset "two-velocity automatic timestep survives resumption and regrid" begin
    force = (x, y, time, axis) -> axis == 1 ? -0.4 : 0.0
    options = (; cfl = 0.25, force, tolerance = 1.0e-9, max_cycles = 1000)
    reference = _two_velocity_resume_flow()
    simulate!(reference, 0.5; options...)
    resumed = _two_velocity_resume_flow()
    first_run = simulate!(resumed, 0.25; options...)
    second_run = simulate!(resumed, 0.5; options...)

    @test first_run.mesh_updates == 0
    @test second_run.mesh_updates == reference.mesh_updates == 1
    @test first_run.steps + second_run.steps == reference.steps == resumed.steps
    for name in (:kinetic_energy, :phase_volume)
        expected = diagnostic_history(reference, name)
        actual = diagnostic_history(resumed, name)
        @test actual.times == expected.times
        @test actual.steps == expected.steps
        @test actual.values == expected.values
    end
    _check_two_velocity_resume_fields(reference, resumed)

    # This shortened physical step also triggers a regrid. Its geometric
    # projection must use the retained proposal, not the shortened step.
    automatic = _two_velocity_resume_flow()
    fixed = _two_velocity_resume_flow()
    simulate!(automatic, 0.25; options...)
    simulate!(fixed, 0.25; options...)
    automatic_regrid = simulate!(automatic, 0.300001; options...)
    fixed_regrid = simulate!(fixed, 0.300001; dt = 0.0625, options...)
    @test automatic_regrid.mesh_updates == fixed_regrid.mesh_updates == 1
    _check_two_velocity_resume_fields(fixed, automatic)
end

# ---------------------------------------------------------------------------
# Stationary embedded solids with an explicitly selected viscous closure.
# The two-phase stress operator accepts the same `closure` as the single-phase
# one. `WallDistanceClosure()` is the legacy/thesis assembly; `LSStagClosure()`
# replaces the cut-cell rows. Every fixture below names its closure, because
# the `NavierStokesOptions` default is deliberately not assumed.

"""Two-phase fixture on (0,8)²: a stationary embedded `solid`, a flat phase
interface above it, and a 100-fold dynamic-viscosity jump."""
function _two_phase_embedded_fixture(
        solid; viscosity = (1.0, 0.01), density = (2.0, 1.0)
    )
    mesh = BlockTreeMesh(
        block_size = (8, 8), extent = (8.0, 8.0),
        periodic = (false, false),
    )
    geometry = GeometryField(mesh, solid; lipschitz = 1.0)
    interface = InterfaceField(
        mesh, (x, y) -> y - 6.2; geometry, lipschitz = 1.0,
    )
    return (; mesh, geometry, interface, viscosity, density)
end

@testset "two-phase stress operator consumes the selected closure" begin
    solid(x, y) = hypot(x - 4.0, y - 4.0) - 1.6
    fixture = _two_phase_embedded_fixture(solid)
    (; mesh, interface, viscosity) = fixture
    wall = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary = NoSlip(),
        closure = WallDistanceClosure(),
    )
    lsstag = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    public_operator = StaggeredDiffusionOperator(
        interface; viscosity, boundary = NoSlip(),
        closure = LSStagClosure(),
    )
    @test wall.closure isa WallDistanceClosure
    @test lsstag.closure isa LSStagClosure
    @test public_operator.closure isa LSStagClosure
    @test public_operator.stiffness ≈ lsstag.stiffness rtol = 16eps()
    mu = dynamic_viscosity(interface; viscosity)
    from_cells = StaggeredDiffusionOperator(
        fixture.geometry; viscosity = mu, closure = LSStagClosure(),
    )
    @test from_cells.strain == lsstag.strain
    @test from_cells.stiffness == lsstag.stiffness
    # The closure replaces cut-cell rows only: the unknown set and the mass
    # (the dual volumes) are those of the legacy assembly.
    @test lsstag.dirichlet_faces == wall.dirichlet_faces
    @test lsstag.volumes == wall.volumes
    @test any(>(0), lsstag.volumes)
    covered = findall(iszero, fixture.geometry.face_dual_volumes)
    @test !isempty(covered)
    # A covered face carries no physical dual volume; its algebraic row must
    # keep the stiffness column positive under either closure.
    @test all(index -> lsstag.stiffness[index, index] > 0, covered)
    @test all(index -> wall.stiffness[index, index] > 0, covered)
    # A distinct closure must change the assembled values, not just the label.
    @test Matrix(lsstag.stiffness) != Matrix(wall.stiffness)
    @test lsstag.stiffness ≈ transpose(lsstag.stiffness) rtol = 1.0e-14
    wall_stiffness = transpose(wall.stiffness)
    @test wall.stiffness ≈ wall_stiffness rtol = 1.0e-14
    # `E'WE` with nonnegative stress weights, so the quadratic form is the
    # discrete dissipation. Both orders of evaluation agree to round-off.
    rng = MersenneTwister(0x1e55)
    velocity = randn(rng, length(mesh.faces))
    velocity[lsstag.dirichlet_faces] .= 0.0
    strain = lsstag.strain * velocity
    quadratic = dot(velocity, lsstag.stiffness * velocity)
    @test quadratic > 0
    @test quadratic ≈ dot(strain, lsstag.stress_weights .* strain) rtol = 1.0e-12
    # FreeSlip condenses the tangential wall trace in the cut-cell stress
    # rows; NoSlip fixes it at zero. Releasing that constraint cannot add
    # dissipation, and must change a motion with nonzero wall-tangential strain.
    free = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity, boundary = FreeSlip(),
        closure = LSStagClosure(),
    )
    difference = lsstag.stiffness - free.stiffness
    @test difference ≈ transpose(difference) atol = 1.0e-12
    @test dot(velocity, difference * velocity) >= -1.0e-10 * max(quadratic, 1.0)
    uniform = ones(length(mesh.faces))
    uniform[lsstag.dirichlet_faces] .= 0.0
    @test dot(uniform, difference * uniform) > 0
    # LS-STAG replaces the values on and next to cut cells; rows far from the
    # solid keep the legacy assembly.
    far = [
        index for (index, face) in pairs(mesh.faces)
            if solid(face.center...) > 2.5
    ]
    @test !isempty(far)
    @test lsstag.stiffness[far, :] ≈ wall.stiffness[far, :] atol = 1.0e-12
end

@testset "two-phase embedded wall rows drag the tangential velocity" begin
    wall = 4.0
    mesh = BlockTreeMesh(
        base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0),
        periodic = (true, false), max_level = 1,
    )
    solid = InterfaceField(mesh; fraction_floor = 1.0e-14)
    initialize!(solid, (0.0, 1.0), wall)
    geometry = GeometryField(solid)
    interface = InterfaceField(
        mesh, (x, y) -> x - 3.5; geometry,
        fraction_floor = 1.0e-14, lipschitz = 1.0,
    )
    viscosity = (1.0, 0.1)
    tangent = 1
    normal = 2
    # The first open tangential face above the wall, at three streamwise
    # positions, so one phase and then the other sees the wall.
    faces = map((2.0, 4.0, 6.0)) do coordinate
        candidates = [
            index for (index, face) in pairs(mesh.faces)
                if face.axis == tangent &&
                abs(face.center[tangent] - coordinate) < 1.0e-12 &&
                face.center[normal] > wall &&
                geometry.face_dual_volumes[index] > 0
        ]
        @test !isempty(candidates)
        sort!(candidates; by = index -> mesh.faces[index].center[normal])
        return first(candidates)
    end
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == tangent ? 1.0 : 0.0,
    )
    for index in eachindex(velocity.interior)
        iszero(geometry.face_dual_volumes[index]) &&
            (velocity.interior[index] = 0.0)
    end
    for closure in (WallDistanceClosure(), LSStagClosure())
        no_slip = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity, boundary = NoSlip(), closure,
        )
        free_slip = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity, boundary = FreeSlip(), closure,
        )
        @test no_slip.closure isa typeof(closure)
        @test free_slip.closure isa typeof(closure)
        no_slip_rate, free_slip_rate = FaceVelocity(mesh), FaceVelocity(mesh)
        diffusion!(no_slip_rate, velocity, no_slip)
        diffusion!(free_slip_rate, velocity, free_slip)
        for face in faces
            # A uniform tangential field has no interior strain; the stationary
            # wall can only slow it, and only NoSlip does.
            @test no_slip_rate.interior[face] < 0
            @test no_slip_rate.interior[face] <
                free_slip_rate.interior[face]
        end
    end
end

@testset "two-phase closure is dissipative in an implicit midpoint step" begin
    solid(x, y) = hypot(x - 4.0, y - 4.0) - 1.8
    fixture = _two_phase_embedded_fixture(solid; viscosity = (0.2, 0.002))
    (; mesh, interface, viscosity) = fixture
    source = FaceVelocity(
        mesh,
        (x, y, time, axis) -> 0.1 * (
            axis == 1 ? sinpi(x / 8) * cospi(y / 8) :
                -cospi(x / 8) * sinpi(y / 8)
        ),
    )
    density = mixture_density(interface, fixture.density)
    for closure in (WallDistanceClosure(), LSStagClosure())
        operator = TwoPhaseAMR._staggered_diffusion_operator(
            interface; viscosity, boundary = NoSlip(), closure,
        )
        TwoPhaseAMR._update_dynamic_viscosity!(
            operator, interface, viscosity,
        )
        @test operator.closure isa typeof(closure)
        source.interior[operator.dirichlet_faces] .= 0.0
        result = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(result, source, operator, 0.5; density)
        inertia = density .* operator.volumes
        before = sum(inertia .* source.interior .^ 2)
        after = sum(inertia .* result.interior .^ 2)
        @test before > 0
        # The implicit midpoint step is contractive in this inertia; the
        # inequality is strict because the source carries interior strain.
        @test after < before
        @test after >= 0
        @test all(isfinite, result.interior)
        @test all(iszero, result.interior[operator.dirichlet_faces])
    end
end

@testset "two-phase cut-cell closure reaches the viscous multigrid levels" begin
    for closure in (WallDistanceClosure(), LSStagClosure())
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4),
            periodic = (false, false),
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.22;
            lipschitz = 1.0,
        )
        interface = InterfaceField(
            mesh, (x, y) -> x - 0.43; geometry, lipschitz = 1.0,
        )
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (4.0, 1.0),
                dynamic_viscosity = (0.2, 0.002),
            );
            geometry,
            numerics = NavierStokesOptions(
                embedded_viscous_closure = closure,
                multigrid = MultigridOptions(coarse_size = 8),
            ),
        )
        operator = flow.workspace.two_phase.diffusion
        @test operator.closure isa typeof(closure)
        preconditioner =
            flow.workspace.two_phase.diffusion_solver.preconditioner
        @test preconditioner isa
            TwoPhaseAMR.TreeVariableDiffusionPreconditioner
        @test length(preconditioner.operators) > 1
        @test preconditioner.operators[1] === operator
        @test all(
            level -> level.closure isa typeof(closure),
            preconditioner.operators,
        )
    end
end

@testset "two-phase diffusion reuse is keyed to the closure" begin
    solid(x, y) = hypot(x - 4.0, y - 4.0) - 1.6
    fixture = _two_phase_embedded_fixture(solid)
    physics = TwoPhase(
        fixture.interface; density = fixture.density,
        dynamic_viscosity = fixture.viscosity,
    )
    model = TwoPhaseAMR.TwoPhaseModel(physics)
    legacy = TwoPhaseAMR.TwoPhaseFlowWorkspace(
        model; boundary = NoSlip(), closure = WallDistanceClosure(),
        diffusion_preconditioner = nothing,
    )
    @test legacy.diffusion.closure isa WallDistanceClosure
    switched = TwoPhaseAMR.TwoPhaseFlowWorkspace(
        model; boundary = NoSlip(), closure = LSStagClosure(),
        diffusion_preconditioner = nothing,
        reuse_diffusion = legacy.diffusion,
    )
    @test switched.diffusion.closure isa LSStagClosure
    @test switched.diffusion !== legacy.diffusion
    @test Matrix(switched.diffusion.stiffness) !=
        Matrix(legacy.diffusion.stiffness)
    repeated = TwoPhaseAMR.TwoPhaseFlowWorkspace(
        model; boundary = NoSlip(), closure = LSStagClosure(),
        diffusion_preconditioner = nothing,
        reuse_diffusion = switched.diffusion,
    )
    @test repeated.diffusion.closure isa LSStagClosure
    @test repeated.diffusion.stiffness_pointers ===
        switched.diffusion.stiffness_pointers
end

@testset "two-phase stationary solid runs with the selected closure" begin
    models = (
        (:one, OneVelocityModel()),
        (:cmdgfm, TwoVelocityModel(pressure_projection = CMDGFMProjection())),
        (:mimetic, TwoVelocityModel(pressure_projection = MimeticProjection())),
    )
    for (name, model) in models, boundary in (NoSlip(), FreeSlip()),
            closure in (WallDistanceClosure(), LSStagClosure())
        mesh = BlockTreeMesh(
            block_size = (6, 6), periodic = (false, false),
        )
        geometry = GeometryField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.45) - 0.22;
            lipschitz = 1.0,
        )
        interface = InterfaceField(
            mesh, (x, y) -> y - 0.8; geometry,
            embedded_contact_angle = pi / 3, lipschitz = 1.0,
        )
        physics = TwoPhase(
            interface; density = (2.0, 1.0),
            dynamic_viscosity = (0.2, 0.002),
        )
        shared = (x, y, time, axis) -> 0.05 * (
            axis == 1 ? sinpi(x) * cospi(y) :
                -cospi(x) * sinpi(y)
        )
        initial = model isa OneVelocityModel ? shared : (shared, shared)
        flow = FlowSimulation(
            mesh, physics; geometry, velocity_model = model,
            velocity = initial, boundary,
            numerics = NavierStokesOptions(
                embedded_viscous_closure = closure,
            ),
        )
        operator = flow.workspace.two_phase.diffusion
        @test flow.diffusion_operator === operator
        @test operator.closure isa typeof(closure)
        @test flow.system.options.embedded_viscous_closure === closure
        covered = findall(iszero, geometry.face_dual_volumes)
        @test !isempty(covered)
        @test all(index -> operator.stiffness[index, index] > 0, covered)
        bound = stable_timestep(flow)
        @test isfinite(bound) && bound > 0
        volume = integrate(interface)
        dt = 0.25bound
        result = simulate!(
            flow, 2dt; dt,
            tolerance = 1.0e-9, max_cycles = 200,
        )
        @test result.steps == 2
        @test flow.workspace.two_phase.diffusion_solver.stats.converged
        for block in flow.pressure.blocks
            @test all(isfinite, block)
        end
        for block in interface.fraction.blocks
            @test all(isfinite, block)
            @test all(value -> -1.0e-12 <= value <= 1 + 1.0e-12, block)
        end
        # The flux-form transport conserves the phase volume; the fraction
        # floor of the PLIC cleanup may move a sliver per mixed cell.
        @test integrate(interface) ≈ volume atol =
            1.0e-8 * max(1, abs(volume))
        if flow.velocity isa TwoPhaseVelocity
            @test _phase_storage_contract(flow.velocity, interface)
            for phase in 1:2
                @test !isempty(_phase_interior_indices(interface, phase))
            end
        else
            @test all(isfinite, flow.velocity.interior)
        end
    end
end

@testset "two-phase embedded wetting angle with LS-STAG" begin
    contact_angle = pi / 3
    for projection in (CMDGFMProjection(), MimeticProjection())
        mesh = BlockTreeMesh(
            block_size = (16, 16), periodic = (false, false),
        )
        geometry = GeometryField(
            mesh, (x, y) -> y - 0.3; lipschitz = 1.0,
        )
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.42) - 0.2;
            geometry, embedded_contact_angle = contact_angle,
            contact_angle = pi / 2, lipschitz = 1.0,
            atol = 1.0e-8, maxdepth = 20,
        )
        @test interface.embedded_contact_angle == Float64(contact_angle)
        physics = TwoPhase(
            interface; density = (2.0, 1.0),
            dynamic_viscosity = (0.2, 0.002),
            surface_tension = 0.05, gravity = (0.0, -0.2),
        )
        flow = FlowSimulation(
            mesh, physics; geometry, boundary = FreeSlip(),
            velocity_model = TwoVelocityModel(
                pressure_projection = projection,
            ),
            numerics = NavierStokesOptions(
                embedded_viscous_closure = LSStagClosure(),
            ),
        )
        @test flow.workspace.two_phase.diffusion.closure isa LSStagClosure
        bound = stable_timestep(flow)
        @test isfinite(bound) && bound > 0
        volume = integrate(interface)
        dt = 0.1bound
        result = simulate!(
            flow, 2dt; dt,
            tolerance = 1.0e-9, max_cycles = 200,
        )
        @test result.steps == 2
        @test flow.workspace.two_phase.diffusion_solver.stats.converged
        @test _phase_storage_contract(flow.velocity, interface)
        for phase in 1:2
            @test !isempty(_phase_interior_indices(interface, phase))
        end
        for block in interface.fraction.blocks
            @test all(value -> -1.0e-12 <= value <= 1 + 1.0e-12, block)
        end
        @test integrate(interface) ≈
            volume atol = 1.0e-8 * max(1, abs(volume))
    end
end

@testset "two-phase LS-STAG dynamic regrid keeps closure and constraints" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 1, periodic = (false, false),
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.25; lipschitz = 1.0,
    )
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.7; geometry, lipschitz = 1.0,
    )
    indicator = InterfaceRefinementIndicator(interface; level = 1)
    physics = TwoPhase(
        interface; density = (4.0, 1.0),
        dynamic_viscosity = (0.02, 0.001),
    )
    flow = FlowSimulation(
        mesh, physics; geometry, velocity = (0.0, 0.0),
        adaptivity = Adaptivity(criteria = (indicator,)),
        numerics = NavierStokesOptions(
            embedded_viscous_closure = LSStagClosure(),
        ),
    )
    before = flow.workspace.two_phase.diffusion
    @test before.closure isa LSStagClosure
    @test !isempty(before.dirichlet_faces)
    covered = findall(iszero, geometry.face_dual_volumes)
    @test !isempty(covered)
    # A covered face carries no physical dual volume; its algebraic row must
    # keep the stiffness column positive.
    @test all(index -> before.stiffness[index, index] > 0, covered)
    result = simulate!(
        flow, 1.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 200,
    )
    @test result.mesh_updates > 0
    @test result.steps > 0
    # Adaptation and rebuilding follow the first physical step. Exercise the
    # rebuilt variable-stress operator in a second step before checking state.
    after_regrid = simulate!(
        flow, 2.0e-4; dt = 1.0e-4,
        tolerance = 1.0e-9, max_cycles = 200,
    )
    @test after_regrid.steps == 1
    @test flow.mesh_updates == result.mesh_updates + after_regrid.mesh_updates
    @test nblocks(mesh) > 1
    operator = flow.workspace.two_phase.diffusion
    @test operator !== before
    @test operator.closure isa LSStagClosure
    @test flow.system.options.embedded_viscous_closure isa LSStagClosure
    @test !isempty(operator.dirichlet_faces)
    fresh = TwoPhaseAMR._staggered_diffusion_operator(
        interface; viscosity = flow.system.two_phase.dynamic_viscosity,
        boundary = flow.system.boundary, closure = LSStagClosure(),
    )
    @test operator.closure isa LSStagClosure
    @test operator.dirichlet_faces == fresh.dirichlet_faces
    @test operator.strain == fresh.strain
    @test operator.volumes == fresh.volumes
    # The face indices changed with the topology: recompute the covered set
    # on the rebuilt mesh before checking its algebraic columns.
    covered = findall(iszero, geometry.face_dual_volumes)
    @test !isempty(covered)
    @test all(index -> operator.stiffness[index, index] > 0, covered)
    @test all(isfinite, flow.velocity.interior)
    @test all(isfinite, flow.velocity.boundary)
    for block in flow.pressure.blocks
        @test all(isfinite, block)
    end
    for block in interface.fraction.blocks
        @test all(isfinite, block)
        @test all(value -> -1.0e-12 <= value <= 1 + 1.0e-12, block)
    end
    @test all(
        index -> iszero(flow.velocity.interior[index]),
        operator.dirichlet_faces,
    )
    println(
        "two-phase LS-STAG regrid witness: mesh_updates=",
        result.mesh_updates + after_regrid.mesh_updates,
        " steps=", result.steps + after_regrid.steps,
        " blocks=", nblocks(mesh), " covered_faces=", length(covered),
        " constrained_faces=", length(operator.dirichlet_faces),
    )
end
