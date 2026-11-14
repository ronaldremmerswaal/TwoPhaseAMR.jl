using Test, TwoPhaseAMR, LinearAlgebra
include("helpers.jl")

const TVD = TwoPhaseAMR

function _two_velocity_diffusion_case(kind)
    geometry_case = kind in (:geometry, :mixed_geometry)
    mixed_case = kind in (:mixed, :mixed_geometry)
    mesh = BlockTreeMesh(
        block_size = geometry_case ? (6, 6) : (4, 4),
        max_level = mixed_case ? 2 : 0,
        periodic = geometry_case ? (false, false) : (true, true),
        refine = (key, origin, extent) -> mixed_case &&
            (
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0)
        ),
    )
    geometry = geometry_case ? GeometryField(
            mesh, (x, y) -> y - 0.16 - 0.1x; lipschitz = hypot(1, 0.1),
        ) : nothing
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.17y - 0.53;
        geometry, lipschitz = hypot(1, 0.17),
    )
    liquid = (x, y, time, axis) ->
    sinpi(2x + y + axis / 3) + 0.1axis
    gas = (x, y, time, axis) ->
    0.4 + cospi(x - 2y + axis / 5)
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (5.0, 1.0),
            dynamic_viscosity = (0.03, 0.002),
        );
        velocity_model = TwoVelocityModel(),
        geometry, velocity = (liquid, gas), boundary = FreeSlip(),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(preconditioner = nothing),
        ),
    )
    state = TVD._two_velocity_flow_workspace!(
        flow.workspace.two_phase, flow.system.two_phase,
    )
    TVD._copy_velocity!(state.convected, flow.velocity)
    return (; flow, state, geometry_case)
end

@testset "Two-velocity implicit diffusion identities" begin
    for kind in (:uniform, :mixed, :geometry, :mixed_geometry)
        case = _two_velocity_diffusion_case(kind)
        flow = case.flow
        state = case.state
        model = flow.system.two_phase
        phase = flow.workspace.two_phase
        dt = 0.002

        liquid_before = copy(state.convected[1].interior)
        gas_before = copy(state.convected[2].interior)
        mixed = findall(
            index -> isfinite(liquid_before[index]) && isfinite(gas_before[index]),
            eachindex(liquid_before)
        )
        jump_before = gas_before[mixed] .- liquid_before[mixed]
        TVD._mass_weighted_one_velocity!(
            state.mixture_velocity, phase.density, state.convected, model,
        )
        mixture_before = copy(state.mixture_velocity.interior)
        density_before = copy(phase.density)
        inertia = density_before .* phase.diffusion.volumes

        if case.geometry_case
            fraction_x = TVD._phase_face_average!(
                FaceField{1}(flow.system.mesh),
                model.interface.conditional_fraction, model.interface.geometry,
            )
            fraction_y = TVD._phase_face_average!(
                FaceField{2}(flow.system.mesh),
                model.interface.conditional_fraction, model.interface.geometry,
            )
            local_x = local_y = 0
            for face in flow.system.mesh.faces
                averaged = if face.axis == 1
                    local_x += 1
                    fraction_x.interior[local_x]
                else
                    local_y += 1
                    fraction_y.interior[local_y]
                end
                @test TVD._staggered_liquid_fraction(
                    model.interface, model.interface.geometry, face,
                ) ≈ averaged atol = 8eps()
            end
        end

        TVD._diffusion_substep!(
            flow.velocity, flow.system, flow.workspace, dt, 0.0, model;
            tolerance = 1.0e-11, max_cycles = 1000,
        )
        @test phase.diffusion_solver.stats.converged
        @test norm(
            state.diffused_mixture.interior - mixture_before,
        ) > 100eps()

        jump_after = flow.velocity[2].interior[mixed] .-
            flow.velocity[1].interior[mixed]
        @test _phase_storage_contract(flow.velocity, model.interface)
        jump_scale = max(norm(jump_before, Inf), 1.0)
        @test norm(jump_after - jump_before, Inf) <=
            16eps() * jump_scale

        recovered = FaceVelocity(flow.system.mesh)
        recovered_density = similar(phase.density)
        TVD._mass_weighted_one_velocity!(
            recovered, recovered_density, flow.velocity, model,
        )
        @test recovered_density ≈ density_before rtol = 16eps()
        # The diffusion algebra protects zero-volume scratch divisions with
        # a positive placeholder. Compare physical open faces, not that
        # protected volume vector: fully solid intrinsic velocities are NaN.
        geometry = model.interface.geometry
        open_faces = isnothing(geometry) ? eachindex(recovered.interior) :
            findall(>(0), geometry.face_dual_volumes)
        @test recovered.interior[open_faces] ≈
            state.diffused_mixture.interior[open_faces] rtol = 64eps()

        if kind === :uniform
            stiffness = phase.diffusion.stiffness
            matrix = Diagonal(inertia) + (dt / 2) * stiffness
            expected = matrix \ (
                inertia .* mixture_before -
                    (dt / 2) * (stiffness * mixture_before)
            )
            @test state.diffused_mixture.interior ≈ expected rtol = 2.0e-10
        end

        if !case.geometry_case
            for axis in 1:2
                indices = findall(
                    face -> face.axis == axis, flow.system.mesh.faces,
                )
                momentum_before = sum(
                    inertia[index] * mixture_before[index] for index in indices
                )
                momentum_after = sum(
                    inertia[index] * recovered.interior[index]
                        for index in indices
                )
                scale = max(abs(momentum_before), norm(inertia), 1.0)
                @test abs(momentum_after - momentum_before) <= 2.0e-10scale
            end
        end
    end
end

@testset "Two-velocity diffusion recovers the continuous-velocity limit" begin
    mesh = BlockTreeMesh(block_size = (6, 6), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> x - 0.47; lipschitz = 1,
    )
    initial = (x, y, time, axis) -> axis == 1 ?
        sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    physics = TwoPhase(
        interface; density = (7.0, 1.0),
        dynamic_viscosity = (0.04, 0.003),
    )
    options = (;
        boundary = FreeSlip(),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(preconditioner = nothing),
        ),
    )
    one_velocity = FlowSimulation(mesh, physics; options..., velocity = initial)
    two_velocity = FlowSimulation(
        mesh, physics; options..., velocity = (initial, initial),
        velocity_model = TwoVelocityModel(),
    )
    TVD._copy_velocity!(
        one_velocity.workspace.convected, one_velocity.velocity,
    )
    state = TVD._two_velocity_flow_workspace!(
        two_velocity.workspace.two_phase, two_velocity.system.two_phase,
    )
    TVD._copy_velocity!(state.convected, two_velocity.velocity)

    dt = 0.003
    TVD._diffusion_substep!(
        one_velocity.velocity, one_velocity.system,
        one_velocity.workspace, dt, 0.0, one_velocity.system.two_phase;
        tolerance = 1.0e-12, max_cycles = 1000,
    )
    TVD._diffusion_substep!(
        two_velocity.velocity, two_velocity.system,
        two_velocity.workspace, dt, 0.0, two_velocity.system.two_phase;
        tolerance = 1.0e-12, max_cycles = 1000,
    )
    for phase in 1:2
        indices = _phase_interior_indices(interface, phase)
        @test two_velocity.velocity[phase].interior[indices] ≈
            one_velocity.velocity.interior[indices] rtol = 2.0e-11
    end
end

@testset "High-level two-velocity step accepts dynamic viscosity" begin
    mesh = BlockTreeMesh(block_size = (6, 6), periodic = (true, true))
    interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
    velocity = (
        (x, y, time, axis) -> axis == 1 ? 0.05 : 0.0,
        (x, y, time, axis) -> axis == 1 ? 0.05 : 0.0,
    )
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (4.0, 1.0),
            dynamic_viscosity = (0.02, 0.001),
        );
        velocity_model = TwoVelocityModel(), velocity, boundary = FreeSlip(),
        numerics = NavierStokesOptions(
            diffusion = LinearSolverOptions(preconditioner = nothing),
        ),
    )
    stats = step!(
        flow.velocity, flow.pressure, flow.system, 1.0e-3,
        flow.workspace; tolerance = 1.0e-10, max_cycles = 1000,
    )
    @test stats.converged
    @test flow.workspace.two_phase.diffusion_solver.stats.converged
    @test _phase_storage_contract(flow.velocity, interface)
    for phase in 1:2, index in _phase_interior_indices(interface, phase)
        @test flow.velocity[phase].interior[index] ≈
            (mesh.faces[index].axis == 1 ? 0.05 : 0.0) atol = 2.0e-10
    end
end
