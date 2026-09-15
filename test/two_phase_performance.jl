using Test, TwoPhaseAMR

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
        density = (1000.0, 1.0), momentum_workspace = momentum, scheme = MC()
    )
    advection_bytes = @allocated advect!(
        interface, velocity, dt, advection;
        scalars = (scalar,), staggered_scalars = (staggered,), momentum = destination,
        density = (1000.0, 1.0), momentum_workspace = momentum, scheme = MC()
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
