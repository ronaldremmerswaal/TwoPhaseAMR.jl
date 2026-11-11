using Test, TwoPhaseAMR
using LinearAlgebra: ldiv!, mul!

function two_phase_kernel_allocations(kind, n)
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2,
        refine = (key, origin, extent) -> kind == :mixed &&
            (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0))
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5, y - 0.65) - 0.18; lipschitz = 1
    )
    scalar = PhaseScalar(
        interface;
        liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y)
    )
    staggered = PhaseFaceScalar{1}(
        interface;
        liquid = (x, y) -> 1 + 0.1sinpi(2x), gas = (x, y) -> 2 + 0.1cospi(2y)
    )
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : -0.05)
    destination = FaceVelocity(mesh)
    advection = WYAdvectionWorkspace(interface)
    momentum = MomentumAdvectionWorkspace(interface)
    dt = stable_timestep(velocity, advection; cfl = 0.1)
    advect!(
        interface, velocity, dt, advection; scalars = (scalar,),
        staggered_scalars = (staggered,), momentum = destination,
        density = (1000.0, 1.0), momentum_workspace = momentum,
        scheme = MonotonizedCentral()
    )
    advection_bytes = @allocated advect!(
        interface, velocity, dt, advection;
        scalars = (scalar,), staggered_scalars = (staggered,), momentum = destination,
        density = (1000.0, 1.0), momentum_workspace = momentum,
        scheme = MonotonizedCentral()
    )

    poisson = PoissonOperator(interface; density = (1000.0, 1.0))
    elliptic_work = EllipticWorkspace(poisson)
    pressure = CellField(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    cell_result = CellField(mesh)
    apply!(cell_result, poisson, pressure, elliptic_work)
    weighted_gradient!(destination, pressure, poisson, elliptic_work)
    surface_tension!(
        destination, interface, poisson, elliptic_work;
        sigma = 1.0, formulation = CSF(), curvature = inv(0.18)
    )
    elliptic_bytes = @allocated begin
        apply!(cell_result, poisson, pressure, elliptic_work)
        weighted_gradient!(destination, pressure, poisson, elliptic_work)
        surface_tension!(
            destination, interface, poisson, elliptic_work;
            sigma = 1.0, formulation = CSF(), curvature = inv(0.18)
        )
    end

    diffusion = StaggeredDiffusionOperator(interface; viscosity = (2.0, 0.1))
    diffusion!(destination, velocity, diffusion)
    diffusion_bytes = @allocated diffusion!(destination, velocity, diffusion)
    return (;
        advection = advection_bytes, elliptic = elliptic_bytes,
        diffusion = diffusion_bytes, dofs = ncells(mesh) + length(mesh.faces),
    )
end

function momentum_workspace_overhead(kind)
    mesh = BlockTreeMesh(
        block_size = (12, 12), max_level = 2,
        refine = (key, origin, extent) -> kind == :mixed &&
            (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0))
    )
    template = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.5, y - 0.65) - 0.18; lipschitz = 1
    )
    interface_only = copy(template)
    with_momentum = copy(template)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : -0.05
    )
    destination = FaceVelocity(mesh)
    interface_work = WYAdvectionWorkspace(interface_only)
    momentum_work = WYAdvectionWorkspace(with_momentum)
    momentum = MomentumAdvectionWorkspace(with_momentum)
    dt = stable_timestep(velocity, interface_work; cfl = 0.1)

    advect!(interface_only, velocity, dt, interface_work)
    advect!(
        with_momentum, velocity, dt, momentum_work;
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = MonotonizedCentral()
    )
    interface_bytes = @allocated advect!(
        interface_only, velocity, dt, interface_work
    )
    momentum_bytes = @allocated advect!(
        with_momentum, velocity, dt, momentum_work;
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = MonotonizedCentral()
    )
    return interface_bytes, momentum_bytes
end

function two_velocity_momentum_allocations(kind, n)
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2,
        periodic = (true, false),
        refine = (key, origin, extent) -> kind == :mixed &&
            (
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0)
        ),
    )
    geometry = kind == :mixed ? GeometryField(
            mesh, (x, y) -> 0.08 - y; lipschitz = 1,
        ) : nothing
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.52; geometry, lipschitz = 1,
    )
    velocity = (
        FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : 0.0,
        ),
        FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? -0.05 : 0.0,
        ),
    )
    staggered = (
        PhaseFaceScalar{1}(
            interface; liquid = (x, y) -> 1 + x,
            gas = (x, y) -> 2 - y,
        ),
        PhaseFaceScalar{2}(
            interface; liquid = (x, y) -> 1 + x,
            gas = (x, y) -> 2 - y,
        ),
    )
    destination = (FaceVelocity(mesh), FaceVelocity(mesh))
    advection = WYAdvectionWorkspace(interface, velocity)
    momentum = MomentumAdvectionWorkspace(interface, velocity)
    dt = stable_timestep(velocity, advection; cfl = 0.1)
    advance!() = advect!(
        interface, velocity, dt, advection;
        staggered_scalars = staggered,
        momentum = destination, density = (1000.0, 1.0),
        momentum_workspace = momentum, scheme = MonotonizedCentral(),
    )
    advance!()
    bytes = @allocated advance!()
    return (; bytes, dofs = ncells(mesh) + length(mesh.faces))
end

function two_velocity_diffusion_coupling_allocations(kind, n)
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2,
        periodic = (true, true),
        refine = (key, origin, extent) -> kind == :mixed &&
            (
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0)
        ),
    )
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.53; lipschitz = hypot(1, 0.2),
    )
    velocity = (
        FaceVelocity(mesh, (x, y, time, axis) -> sinpi(x + y + axis)),
        FaceVelocity(mesh, (x, y, time, axis) -> cospi(x - y + axis)),
    )
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (1000.0, 1.0),
            dynamic_viscosity = (0.01, 1.0e-4),
        );
        velocity_model = TwoVelocityModel(), velocity, boundary = FreeSlip(),
    )
    state = TwoPhaseAMR._two_velocity_flow_workspace!(
        flow.workspace.two_phase, flow.system.two_phase,
    )
    TwoPhaseAMR._mass_weighted_one_velocity!(
        state.mixture_velocity, flow.workspace.two_phase.density,
        velocity, flow.system.two_phase,
    )
    TwoPhaseAMR._apply_two_velocity_diffusion_update!(
        velocity, velocity, state.mixture_velocity, state.mixture_velocity,
    )
    bytes = @allocated begin
        TwoPhaseAMR._mass_weighted_one_velocity!(
            state.mixture_velocity, flow.workspace.two_phase.density,
            velocity, flow.system.two_phase,
        )
        TwoPhaseAMR._apply_two_velocity_diffusion_update!(
            velocity, velocity,
            state.mixture_velocity, state.mixture_velocity,
        )
    end
    return (; bytes, faces = length(mesh.faces))
end

@testset "Two-phase kernel allocation scaling" begin
    for kind in (:uniform, :mixed)
        small = two_phase_kernel_allocations(kind, 4)
        large = two_phase_kernel_allocations(kind, 8)
        growth = large.dofs / small.dofs
        @info "two-phase kernel allocations" kind small large growth
        # Hot kernels may own size-dependent temporary storage, but allocation
        # must remain linear in the number of active cell and face unknowns.
        @test large.advection <= 1.5growth * small.advection
        @test large.elliptic <= 1.5growth * small.elliptic
        @test large.diffusion <= 1.5growth * small.diffusion
    end
end


@testset "Momentum transport reuses phase-face scratch" begin
    for kind in (:uniform, :mixed)
        interface_bytes, momentum_bytes = momentum_workspace_overhead(kind)
        @test momentum_bytes <= interface_bytes + 4096
    end
end

@testset "Two-velocity staggered transport allocation scaling" begin
    for kind in (:uniform, :mixed)
        small = two_velocity_momentum_allocations(kind, 4)
        large = two_velocity_momentum_allocations(kind, 8)
        growth = large.dofs / small.dofs
        @info "two-velocity staggered allocations" kind small large growth
        @test large.bytes <= 1.5growth * small.bytes
    end
end

@testset "Two-velocity diffusion coupling has bounded scratch allocation" begin
    for kind in (:uniform, :mixed)
        small = two_velocity_diffusion_coupling_allocations(kind, 12)
        large = two_velocity_diffusion_coupling_allocations(kind, 24)
        @info "two-velocity diffusion coupling allocations" kind small large
        # Both kernels write package-owned arrays. Parallel scheduler metadata
        # may allocate a fixed amount, but storage must not grow per face.
        @test large.bytes <= small.bytes + 16_384
    end
end

@testset "Two-phase geometry query allocation contracts" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.12 - 0.18x; lipschitz = 1,
    )
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.48, y - 0.46) - 0.28;
        geometry, lipschitz = 1,
    )
    reconstruct!(interface)
    ref = CellRef(1, 1, 1)
    fraction = TwoPhaseAMR._phase_face_average!(
        FaceField{1}(mesh), interface.conditional_fraction, geometry,
    )

    TwoPhaseAMR._phase_cell_fractions(interface, geometry, ref)
    TwoPhaseAMR._phase_staggered_fractions(
        interface, geometry, fraction, 1,
    )
    TwoPhaseAMR._phase_face_apertures(interface, geometry, 1)
    @test @allocated(
        TwoPhaseAMR._phase_cell_fractions(interface, geometry, ref)
    ) == 0
    @test @allocated(
        TwoPhaseAMR._phase_staggered_fractions(
            interface, geometry, fraction, 1,
        )
    ) == 0
    @test @allocated(
        TwoPhaseAMR._phase_face_apertures(interface, geometry, 1)
    ) == 0
end

@testset "Variable-stress solver allocation contracts" begin
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 2,
        refine = (key, origin, extent) -> key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    )
    interface = InterfaceField(
        mesh, (x, y) -> x + 0.2y - 0.55; lipschitz = hypot(1, 0.2),
    )
    operator = StaggeredDiffusionOperator(
        interface; viscosity = (1.0e-2, 1.0e-5), boundary = FreeSlip(),
    )
    density = TwoPhaseAMR.mixture_density(interface, (1.0, 1.0e-3))
    dt = 0.01
    inertia = density .* operator.volumes
    action = TwoPhaseAMR.FactorizedVariableDiffusionOperator(
        operator, inertia, dt,
    )
    vector = [sin(0.7index) for index in eachindex(operator.volumes)]
    action_vector = similar(vector)
    mul!(action_vector, action, vector)
    @test @allocated(mul!(action_vector, action, vector)) == 0

    solver = TwoPhaseAMR.VariableStaggeredDiffusionSolver(
        operator, density, dt; multigrid = MultigridOptions(coarse_size = 16),
    )
    right_hand_side = copy(solver.right_hand_side)
    for index in eachindex(right_hand_side)
        right_hand_side[index] = sin(0.2index)
    end
    solution = similar(right_hand_side)
    ldiv!(solution, solver.preconditioner, right_hand_side)
    @test @allocated(
        ldiv!(solution, solver.preconditioner, right_hand_side)
    ) == 0
end
