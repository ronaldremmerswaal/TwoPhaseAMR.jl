using Test, TwoPhaseAMR
const TP = TwoPhaseAMR
vof_values(f) = reduce(vcat, [vec(collect(interior(f, b))) for b in 1:nblocks(f.mesh)])
function phase_momentum(interface, velocity, ::Val{Axis}, density) where {Axis}
    fraction = TP._phase_face_average!(
        FaceField{Axis}(interface.mesh),
        interface.fraction
    )
    values = FaceField{Axis}(velocity)
    schedule = StaggeredAdvectionWorkspace(values).schedule
    return sum(eachindex(values.interior)) do i
        volume = TP._phase_dual_volume(schedule, i)
        rho = density[1] * fraction.interior[i] + density[2] * (1 - fraction.interior[i])
        volume * rho * values.interior[i]
    end
end
function transport_meshes()
    return (
        BlockTreeMesh(block_size = (16, 16)),
        BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0)
        ),
        BlockTreeMesh(block_size = (8, 8), max_level = 2),
        BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8)),
    )
end
@testset "PLIC geometry" begin
    for n in ((1.0, 0.0), (0.0, 1.0), (0.3, -0.7), (-0.5, -0.5)), c in (0.0, 1.0e-10, 0.1, 0.5, 0.9, 1 - 1.0e-10, 1.0)
        a = TP._plane_intercept(c, n...)
        @test TP._plane_fraction(n..., a) ≈ c atol = 5.0e-15
        children = [TP._plane_fraction(n..., a, x, x + 0.5, y, y + 0.5) for x in (-0.5, 0.0), y in (-0.5, 0.0)]
        @test sum(children) / 4 ≈ c atol = 5.0e-15
        @test all(0 .<= children .<= 1)
    end
end
@testset "Global fraction floor" begin
    mesh = BlockTreeMesh(block_size = (4, 4))
    interface64 = InterfaceField(mesh)
    interface32 = InterfaceField(mesh; T = Float32)
    @test interface64.fraction_floor === sqrt(eps(Float64))
    @test interface32.fraction_floor === sqrt(eps(Float32))
    @test WYAdvectionWorkspace(interface64).ctu_threshold === 0.5
    floor = 1.0e-6
    cleaned = InterfaceField(mesh; fraction_floor = floor)
    @test cleaned.geometry_is_valid[]
    reconstruct!(cleaned)
    @test @allocated(reconstruct!(cleaned)) == 0
    values = interior(cleaned.fraction, 1)
    values[1, 1] = floor / 2
    values[2, 1] = 2floor
    values[3, 1] = 1 - floor / 2
    values[4, 1] = 1 - 2floor
    reconstruct!(cleaned)
    @test cleaned.geometry_is_valid[]
    @test interior(cleaned.reconstruction_fraction, 1) == values
    @test values[1, 1] == 0
    @test values[2, 1] == 2floor
    @test values[3, 1] == 1
    @test values[4, 1] == 1 - 2floor

    geometry = GeometryField(
        mesh, (x, y) -> x - 0.4; lipschitz = 1,
    )
    cut = InterfaceField(mesh; geometry, fraction_floor = floor)
    open = interior(geometry.open_fraction, 1)
    cut_values = interior(cut.fraction, 1)
    open_indices = findall(>(0), open)
    cut_values[open_indices[1]] = open[open_indices[1]] * floor / 2
    cut_values[open_indices[2]] = open[open_indices[2]] * (1 - floor / 2)
    reconstruct!(cut)
    @test cut_values[open_indices[1]] == 0
    @test cut_values[open_indices[2]] == open[open_indices[2]]
    previous_revision = cut.reconstructed_geometry_revision[]
    refresh_geometry!(geometry)
    @test geometry.revision > previous_revision
    @test !TP._reconstruction_is_current(cut)
    reconstruct!(cut)
    @test TP._reconstruction_is_current(cut)

    moving = InterfaceField(
        mesh, (x, y) -> x - 0.5;
        lipschitz = 1, fraction_floor = floor,
    )
    constant = PhaseScalar(moving)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.25 : 0.0,
    )
    before = vof_values(moving)
    displacement = mesh.blocks[1].spacing[1] * floor / 2
    advect!(
        moving, velocity, displacement / 0.25,
        WYAdvectionWorkspace(moving); scalars = (constant,),
    )
    @test vof_values(moving) == before
    @test vof_values(constant.liquid) == ones(length(before))
    @test vof_values(constant.gas) == ones(length(before))

    @test WYAdvectionWorkspace(interface64; ctu_threshold = 0.25).ctu_threshold === 0.25
    @test_throws ArgumentError InterfaceField(mesh; fraction_floor = 0)
    @test_throws ArgumentError InterfaceField(mesh; fraction_floor = 0.5)
    @test_throws ArgumentError WYAdvectionWorkspace(interface64; ctu_threshold = 0)
end


@testset "WY transport is independent of base-root partitioning" begin
    one_root = BlockTreeMesh(base_mesh = (16, 16), block_size = (16, 16))
    four_roots = BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8))

    function transported_values(mesh)
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.43, y - 0.57) - 0.23;
            lipschitz = 1,
        )
        scalar = PhaseScalar(
            interface;
            liquid = (x, y) -> 1 + 0.2sinpi(2y),
            gas = (x, y) -> 2 + 0.1cospi(2x),
        )
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? 0.2 : -0.15,
        )
        workspace = WYAdvectionWorkspace(interface)
        advect!(interface, velocity, 0.02, workspace; scalars = (scalar,))
        nx, ny = mesh.block_size
        return Dict(
            cellcenter(mesh, block, i, j) => (
                interior(interface, block)[i, j],
                interior(scalar.liquid, block)[i, j],
                interior(scalar.gas, block)[i, j],
            )
                for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        )
    end

    first = transported_values(one_root)
    second = transported_values(four_roots)
    @test keys(first) == keys(second)
    @test all(
        all(
            isapprox(a, b; atol = 4.0e-14, rtol = 4.0e-14)
                for (a, b) in zip(first[key], second[key])
        )
            for key in keys(first)
    )
end

@testset "WY volume and phase constants on uniform, mixed and regridded meshes" begin
    for (kind, mesh) in enumerate(transport_meshes())
        f = InterfaceField(mesh, (x, y) -> x - 0.41; lipschitz = 1)
        mass = integrate(f)
        if kind == 3
            amr = AMRWorkspace(mesh, f)
            @test adapt!(amr, [:refine]; step = 1, force = true).changed
            @test integrate(f) ≈ mass atol = 2.0e-15
            @test adapt!(amr, [:refine, :keep, :keep, :keep]; step = 2, force = true).changed
            @test integrate(f) ≈ mass atol = 2.0e-15
        end
        q = PhaseScalar(f; liquid = 1.0, gas = 1.0)
        density = PhaseScalar(f; liquid = 1000.0, gas = 1.0)
        tracer = PhaseScalar(f; liquid = (x, y) -> 1 + 0.2sinpi(2y), gas = (x, y) -> 2 + 0.3cospi(2y))
        tracer_mass = (integrate(tracer, :liquid), integrate(tracer, :gas))
        u = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 0.3 : -0.2)
        w = WYAdvectionWorkspace(f)
        dt = stable_timestep(u, w; cfl = 0.45)
        for step in 1:20
            advect!(f, u, dt, w; scalars = (q, density, tracer), sweep_order = isodd(step) ? (1, 2) : (2, 1))
            c = vof_values(f)
            @test minimum(c) >= 0
            @test maximum(c) <= 1
            @test vof_values(q.liquid) ≈ ones(length(c)) atol = 2.0e-14
            @test vof_values(q.gas) ≈ ones(length(c)) atol = 2.0e-14
            @test vof_values(density.liquid) ≈ fill(1000.0, length(c)) atol = 2.0e-11
            @test all(isfinite, vof_values(tracer.liquid))
            @test integrate(f) ≈ mass atol = 2.0e-14
        end
        @test integrate(tracer, :liquid) ≈ tracer_mass[1] atol = 2.0e-14
        @test integrate(tracer, :gas) ≈ tracer_mass[2] atol = 2.0e-14
        @test w.liquid_flux + w.gas_flux ≈ [dt * u.interior[i] * face.area for (i, face) in pairs(mesh.faces)]
        @test_throws ArgumentError advect!(f, u, 2stable_timestep(u, w), w)
        @test_throws ArgumentError stable_timestep(u, w; cfl = 0.6)
    end
end
@testset "WY dimensional dilation preserves phase constants" begin
    # Streamfunction differences give exactly solenoidal composite face fluxes.
    for mesh in transport_meshes()
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.7) - 0.15; lipschitz = 1)
        q = PhaseScalar(f)
        w = WYAdvectionWorkspace(f)
        u = FaceVelocity(mesh)
        psi(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
        for (i, face) in pairs(mesh.faces)
            x, y = face.center; h = face.area / 2
            u.interior[i] = face.axis == 1 ? (psi(x, y + h) - psi(x, y - h)) / face.area :
                -(psi(x + h, y) - psi(x - h, y)) / face.area
        end
        original = vof_values(f); mass = integrate(f)
        speed = copy(u.interior)
        time = 0.0; finish = 0.2; step = 0
        while time < finish - 1.0e-14
            u.interior .= speed .* cospi(time / finish)
            dt = min(0.005, stable_timestep(u, w; cfl = 0.4), finish - time)
            u.interior .= speed .* cospi((time + dt / 2) / finish)
            advect!(f, u, dt, w; scalars = (q,), sweep_order = iseven(step) ? (1, 2) : (2, 1))
            time += dt; step += 1
        end
        @test integrate(f) ≈ mass atol = 3.0e-14
        @test vof_values(q.liquid) ≈ ones(ncells(mesh)) atol = 3.0e-14
        @test vof_values(q.gas) ≈ ones(ncells(mesh)) atol = 3.0e-14
        # Reversal accuracy is checked by the refinement study in integration/.
    end
end
@testset "Compatible staggered phase constants" begin
    for mesh in transport_meshes()
        f = InterfaceField(mesh, (x, y) -> x - 0.41; lipschitz = 1)
        qx, qy = PhaseFaceScalar{1}(f), PhaseFaceScalar{2}(f; liquid = 1000, gas = 2)
        u = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 0.3 : -0.2)
        w = WYAdvectionWorkspace(f)
        for step in 1:20
            advect!(f, u, stable_timestep(u, w; cfl = 0.3), w; staggered_scalars = (qx, qy))
            for (q, l, g) in ((qx, 1.0, 1.0), (qy, 1000.0, 2.0))
                TP._phase_face_average!(q.fraction, f.fraction)
                @test q.liquid.interior ≈ fill(l, length(q.liquid.interior)) atol = 2.0e-11
                @test q.gas.interior ≈ fill(g, length(q.gas.interior)) atol = 2.0e-11
                @test all(isfinite, q.liquid.interior)
            end
        end
    end
end
@testset "All staggered flux interpolants in two-phase flow" begin
    schemes = (LW(), MC())
    for mesh in transport_meshes()[1:2], Axis in (1, 2), scheme in schemes
        shape = (x, y) -> x - 0.43
        f = InterfaceField(mesh, shape; lipschitz = 1)
        liquid = (x, y) -> 1 + 0.1sinpi(2y)
        gas = (x, y) -> 2 + 0.1cospi(2y)
        q = PhaseFaceScalar{Axis}(f; liquid, gas)
        one = PhaseFaceScalar{Axis}(f)
        liquid_reference = FaceField{Axis}(mesh, liquid)
        gas_reference = FaceField{Axis}(mesh, gas)
        velocity = FaceVelocity(mesh, (x, y, t, a) -> a == 2 ? 0.2 : 0.0)
        work = WYAdvectionWorkspace(f)
        dt = stable_timestep(velocity, work; cfl = 0.3)
        interface_scheme = default_amr_interface_scheme(scheme)
        liquid_work = StaggeredAdvectionWorkspace(liquid_reference)
        gas_work = StaggeredAdvectionWorkspace(gas_reference)
        advect!(
            liquid_reference, velocity, dt, liquid_work;
            scheme, amr_interface_scheme = interface_scheme
        )
        advect!(
            gas_reference, velocity, dt, gas_work;
            scheme, amr_interface_scheme = interface_scheme
        )
        advect!(
            f, velocity, dt, work; staggered_scalars = (q, one), scheme,
            amr_interface_scheme = interface_scheme
        )
        TP._phase_face_average!(q.fraction, f.fraction)
        liquid_bulk = findall(==(1), q.fraction.interior)
        gas_bulk = findall(iszero, q.fraction.interior)
        @test !isempty(liquid_bulk)
        @test !isempty(gas_bulk)
        @test q.liquid.interior[liquid_bulk] ≈
            liquid_reference.interior[liquid_bulk] atol = 5.0e-14
        @test q.gas.interior[gas_bulk] ≈
            gas_reference.interior[gas_bulk] atol = 5.0e-14
        @test one.liquid.interior ≈ ones(length(one.liquid.interior)) atol = 5.0e-14
        @test one.gas.interior ≈ ones(length(one.gas.interior)) atol = 5.0e-14
        @test all(isfinite, q.liquid.interior)
        @test all(isfinite, q.gas.interior)
        if scheme isa Union{Upwind, Minmod, MonotonizedCentral, VanLeer, Superbee}
            @test minimum(q.liquid.interior) >= 0.9 - 32eps()
            @test maximum(q.liquid.interior) <= 1.1 + 32eps()
            @test minimum(q.gas.interior) >= 1.9 - 32eps()
            @test maximum(q.gas.interior) <= 2.1 + 32eps()
        end
    end
end
@testset "All centred flux interpolants in two-phase flow" begin
    schemes = (LW(), MC())
    for mesh_template in (transport_meshes()[index] for index in (1, 2, 4))
        results = Vector{Vector{Float64}}()
        for scheme in schemes
            # Each scheme mutates the interface, so use the template only to
            # select the same uniform or mixed topology.
            multiple_roots = mesh_template.root.layout != (1, 1)
            mixed = !multiple_roots && nblocks(mesh_template) > 1
            mesh = multiple_roots ? BlockTreeMesh(
                    base_mesh = mesh_template.base_mesh,
                    block_size = mesh_template.block_size,
                ) : mixed ? BlockTreeMesh(
                    block_size = mesh_template.block_size, max_level = 2,
                    refine = (key, origin, extent) -> key.level == 0 ||
                    (key.level == 1 && key.i == 0 && key.j == 0)
                ) : BlockTreeMesh(block_size = mesh_template.block_size)
            interface = InterfaceField(
                mesh, (x, y) -> x - 0.43; lipschitz = 1
            )
            scalar = PhaseScalar(
                interface;
                liquid = (x, y) -> 1 + 0.1sinpi(2y),
                gas = (x, y) -> 2 + 0.1cospi(2y)
            )
            one = PhaseScalar(interface)
            initial = (
                integrate(scalar, :liquid), integrate(scalar, :gas),
            )
            velocity = FaceVelocity(
                mesh, (x, y, time, axis) -> axis == 2 ? 0.2 : 0.0
            )
            work = WYAdvectionWorkspace(interface)
            dt = stable_timestep(velocity, work; cfl = 0.3)
            advect!(
                interface, velocity, dt, work;
                scalars = (scalar, one), scheme,
                amr_interface_scheme = default_amr_interface_scheme(scheme)
            )
            @test integrate(scalar, :liquid) ≈ initial[1] atol = 128eps()
            @test integrate(scalar, :gas) ≈ initial[2] atol = 128eps()
            for b in eachindex(mesh.blocks)
                @test interior(one.liquid, b) ≈
                    ones(size(interior(one.liquid, b))) atol = 32eps()
                @test interior(one.gas, b) ≈
                    ones(size(interior(one.gas, b))) atol = 32eps()
                @test all(isfinite, interior(scalar.liquid, b))
                @test all(isfinite, interior(scalar.gas, b))
            end
            push!(
                results,
                reduce(
                    vcat, vec(copy(interior(scalar.liquid, b)))
                        for b in eachindex(mesh.blocks)
                )
            )
        end
        # The public scheme option must reach the centred phase-scalar flux.
        @test results[1] != results[2]
    end
end
@testset "All staggered flux interpolants transport one-velocity momentum" begin
    schemes = (LW(), MC())
    for mesh in transport_meshes()[1:2], scheme in schemes
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.7) - 0.18; lipschitz = 1)
        velocity = FaceVelocity(mesh)
        psi(x, y) = sinpi(x)^2 * sinpi(y)^2 / pi
        for (i, face) in pairs(mesh.faces)
            x, y = face.center
            h = face.area / 2
            velocity.interior[i] = face.axis == 1 ?
                (psi(x, y + h) - psi(x, y - h)) / face.area :
                -(psi(x + h, y) - psi(x - h, y)) / face.area
        end
        destination = FaceVelocity(mesh)
        work = WYAdvectionWorkspace(f)
        momentum = MomentumAdvectionWorkspace(f)
        density = (1000.0, 1.0)
        initial_momentum = (
            phase_momentum(f, velocity, Val(1), density),
            phase_momentum(f, velocity, Val(2), density),
        )
        dt = stable_timestep(velocity, work; cfl = 0.2)
        advect!(
            f, velocity, dt, work; momentum = destination, density,
            momentum_workspace = momentum, scheme,
            amr_interface_scheme = default_amr_interface_scheme(scheme)
        )
        final_momentum = (
            phase_momentum(f, destination, Val(1), density),
            phase_momentum(f, destination, Val(2), density),
        )
        @test all(isfinite, destination.interior)
        @test all(isfinite, destination.boundary)
        @test maximum(abs, destination.interior) > 0
        @test final_momentum[1] ≈ initial_momentum[1] atol = 512eps()
        @test final_momentum[2] ≈ initial_momentum[2] atol = 512eps()
    end
end

@testset "Remaining phase-transport schemes dispatch" begin
    for scheme in (
            Upwind(), Centred(), Fromm(), Minmod(), VanLeer(), Superbee(),
        )
        mesh = BlockTreeMesh(block_size = (4, 4))
        interface = InterfaceField(mesh, (x, y) -> x - 0.43; lipschitz = 1)
        scalar = PhaseScalar(interface; liquid = 3.0, gas = 7.0)
        velocity = FaceVelocity(
            mesh, (x, y, time, axis) -> axis == 1 ? 0.1 : 0.0,
        )
        workspace = WYAdvectionWorkspace(interface)
        initial = (
            integrate(scalar, :liquid), integrate(scalar, :gas),
        )
        advect!(
            interface, velocity,
            stable_timestep(velocity, workspace; cfl = 0.2), workspace;
            scalars = (scalar,), scheme,
            amr_interface_scheme = default_amr_interface_scheme(scheme),
        )
        @test integrate(scalar, :liquid) ≈ initial[1] atol = 128eps()
        @test integrate(scalar, :gas) ≈ initial[2] atol = 128eps()
        @test all(isfinite, vof_values(scalar.liquid))
        @test all(isfinite, vof_values(scalar.gas))
    end
end
@testset "Geometric phase transfer through dynamic refinement and coarsening" begin
    for block_size in ((8, 8), (4, 4))
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = block_size, max_level = 2,
            adaptation = MeshAdaptationOptions(coarsen_delay = 0)
        )
        f = InterfaceField(
            mesh, (x, y) -> 0.7x - 0.3y - 0.21;
            lipschitz = hypot(0.7, 0.3)
        )
        q = PhaseScalar(f; liquid = 1000, gas = 1)
        amr = AMRWorkspace(mesh, f, q)
        volume = integrate(f)
        for (step, action) in enumerate((:refine, :refine, :coarsen, :coarsen))
            @test adapt!(amr, fill(action, nblocks(mesh)); step, force = true).changed
            reconstruct!(f)
            @test integrate(f) ≈ volume atol = 3.0e-14
            @test vof_values(q.liquid) ≈
                fill(1000.0, ncells(mesh)) atol = 3.0e-11
            @test vof_values(q.gas) ≈ ones(ncells(mesh)) atol = 3.0e-14
        end
    end
end
@testset "Intrinsic phase scalar AMR transfer convergence" begin
    errors = Float64[]
    for n in (8, 16, 32)
        mesh = BlockTreeMesh(
            block_size = (n, n), max_level = 1,
            adaptation = MeshAdaptationOptions(coarsen_delay = 0)
        )
        f = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.6) - 0.2; lipschitz = 1)
        liquid = (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y)
        gas = (x, y) -> 2 + 0.1cospi(2x) * sinpi(2y)
        q = PhaseScalar(f; liquid, gas)
        initial = (integrate(q, :liquid), integrate(q, :gas))
        amr = AMRWorkspace(mesh, f, q)
        @test adapt!(amr, fill(:refine, nblocks(mesh)); step = 1, force = true).changed
        @test integrate(q, :liquid) ≈ initial[1] atol = 128eps()
        @test integrate(q, :gas) ≈ initial[2] atol = 128eps()
        error = 0.0
        for (b, block) in pairs(mesh.blocks),j in 1:n,i in 1:n
            x, y = cellcenter(mesh, b, i, j)
            c = interior(f, b)[i, j]
            error += block.volume * (
                c * abs(interior(q.liquid, b)[i, j] - liquid(x, y)) +
                    (1 - c) * abs(interior(q.gas, b)[i, j] - gas(x, y))
            )
        end
        push!(errors, error)
    end
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(rates .>= 1.8)
end
