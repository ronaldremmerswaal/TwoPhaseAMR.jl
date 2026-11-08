using Test, TwoPhaseAMR
isdefined(@__MODULE__, :test_allocated_call) ||
    include(joinpath(@__DIR__, "..", "test_utils.jl"))
const TP = TwoPhaseAMR
vof_values(f) = reduce(vcat, [vec(collect(interior(f, b))) for b in 1:nblocks(f.mesh)])
_transport_test_geometry(field) = field.geometry
_transport_test_interface(work) = work.interface
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
@testset "Concrete interface geometry and WY ownership" begin
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> x - 0.25; lipschitz = 1)
    @test fieldtype(typeof(geometry), :solid) === typeof(geometry.solid)
    @test typeof(geometry.solid).parameters[3] === Nothing
    @test (@inferred _transport_test_geometry(geometry.solid)) === nothing
    geometry_copy = copy(geometry)
    @test geometry_copy !== geometry
    @test geometry_copy.solid.fraction !== geometry.solid.fraction
    @test typeof(geometry_copy.solid).parameters[3] === Nothing

    polygon = PolygonIndicator(
        [
            (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9),
        ]
    )
    velocity = FaceVelocity(mesh)
    for solid in (nothing, geometry)
        uniform = InterfaceField(mesh; init = 0.375, geometry = solid)
        interfaces = (
            uniform,
            InterfaceField(copy(uniform.fraction); geometry = solid),
            InterfaceField(mesh, (x, y) -> y - 0.45; geometry = solid, lipschitz = 1),
            InterfaceField(mesh, polygon; geometry = solid),
        )
        for interface in interfaces
            @test interface isa InterfaceField{Float64, typeof(mesh)}
            @test typeof(interface).parameters[3] === typeof(solid)
            @test fieldtype(typeof(interface), :geometry) === typeof(solid)
            @test (@inferred _transport_test_geometry(interface)) === solid

            copied = copy(interface)
            @test typeof(copied) === typeof(interface)
            @test copied.geometry === solid
            @test copied.fraction !== interface.fraction
            @test vof_values(copied) == vof_values(interface)
            @test copied.contact_angles == interface.contact_angles
            @test copied.embedded_contact_angle == interface.embedded_contact_angle

            scalar = PhaseScalar(interface)
            @test fieldtype(typeof(scalar), :interface) === typeof(interface)
            @test (@inferred _transport_test_interface(scalar)) === interface
            work = WYAdvectionWorkspace(interface)
            @test !hasproperty(work, :geometry)
            @test (@inferred _transport_test_interface(work)) === interface
            @test fieldtype(typeof(work), :method_workspace) === Base.RefValue{Nothing}
            @test (@inferred getindex(work.method_workspace)) === nothing
            @test (@inferred stable_timestep(velocity, work)) == Inf
            before = vof_values(interface)
            advect!(interface, velocity, 0.001, work)
            @test vof_values(interface) ≈ before atol = 128eps(Float64)
        end
    end
    @test_throws ArgumentError GeometryField(InterfaceField(mesh; geometry))
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
    @test test_allocated_call(() -> reconstruct!(cleaned)) == 0
    if VERSION >= v"1.12"
        @test @allocated(reconstruct!(cleaned)) == 0
    end
    # The fast path must skip reconstruction unless explicitly forced, even
    # when derived fields have been edited directly.
    normals = interior(cleaned.normal_x, 1)
    normals[1, 1] = 10
    @test reconstruct!(cleaned) === cleaned
    @test normals[1, 1] == 10
    @test reconstruct!(cleaned; force = true) === cleaned
    @test normals[1, 1] != 10
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
        sources = increments = nothing
        for step in 1:20
            advect!(f, u, dt, w; scalars = (q, density, tracer), sweep_order = isodd(step) ? (1, 2) : (2, 1))
            if step == 1
                sources = copy(w.scalar_sources)
                increments = copy(w.scalar_increments)
            else
                @test all(w.scalar_sources[k] === sources[k] for k in eachindex(sources))
                @test all(w.scalar_increments[k] === increments[k] for k in eachindex(increments))
            end
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
        @test_throws ArgumentError advect!(f, u, dt, w; scalars = (q, q))
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
    schemes = (LaxWendroff(), MonotonizedCentral())
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
    schemes = (LaxWendroff(), MonotonizedCentral())
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
    schemes = (LaxWendroff(), MonotonizedCentral())
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
        work = WYAdvectionWorkspace(f)
        advect!(f, FaceVelocity(mesh), 0.01, work; scalars = (q,))
        source = work.scalar_sources[1]
        increment = work.scalar_increments[1]
        volume = integrate(f)
        for (step, action) in enumerate((:refine, :refine, :coarsen, :coarsen))
            @test adapt!(amr, fill(action, nblocks(mesh)); step, force = true).changed
            reconstruct!(f)
            @test TwoPhaseAMR._reconfigure_wy_workspace!(work, f) === work
            @test work.scalar_sources[1] === source
            @test work.scalar_increments[1] === increment
            @test all(length(buffer.blocks) == nblocks(mesh) for buffer in (source..., increment...))
            advect!(f, FaceVelocity(mesh), 0.01, work; scalars = (q,))
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

# The audit fixture places a cut coarse cell with two fine segments on a side.
function cut_momentum_fixture()
    mesh = BlockTreeMesh(
        base_mesh = (12, 12), block_size = (4, 4), extent = (12.0, 12.0),
        periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
    solid(x, y) = hypot(x - 8.07, y - 6.33) - 1.55
    bubble(x, y) = 1.62 - hypot(x - 2.55, y - 5.95)
    geometry = GeometryField(mesh, solid; lipschitz = 1)
    interface = InterfaceField(mesh, bubble; geometry, lipschitz = 1)
    psi(x, y) = solid(x, y) <= 0 ? 0.0 :
        12 / (2pi) * sinpi(2x / 12) * sinpi(2y / 12)
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        h = face.area / 2
        flux = face.axis == 1 ? psi(x, y + h) - psi(x, y - h) :
            -(psi(x + h, y) - psi(x - h, y))
        aperture = geometry.aperture[index]
        @test aperture > 0 || abs(flux) <= 1.0e-14
        velocity.interior[index] = aperture > 0 ?
            flux / (face.area * aperture) : 0.0
    end
    return interface, velocity
end

function cut_momentum_identity_defect(interface, schedule, ::Val{Axis}, axis) where {Axis}
    mesh, geometry = interface.mesh, interface.geometry
    flux = [
        face.axis == axis ?
            face.area * geometry.aperture[index] *
            sin(0.37face.center[1] + 0.61face.center[2]) : 0.0
            for (index, face) in pairs(mesh.faces)
    ]
    cell_divergence = CellField(mesh)
    for (index, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            open = TP._get(geometry.open_fraction, ref)
            open > 0 || continue
            TP._add!(
                cell_divergence, sign * flux[index] /
                    (mesh.blocks[ref.block].volume * open), ref,
            )
        end
    end
    interpolated = TP._phase_face_average!(
        FaceField{Axis}(mesh), cell_divergence, geometry,
    )
    values = vcat(interpolated.interior, interpolated.boundary)
    divergence = zeros(length(values))
    for face in schedule.faces
        face.axis == axis || continue
        q = TP._phase_dual_flux(face, schedule, flux, mesh)
        divergence[face.minus] += q
        divergence[face.plus] -= q
    end
    return maximum(abs, divergence .- schedule.volumes .* values)
end

function cut_momentum_amount(
        interface, velocity, momentum, density, ::Val{Axis}; absolute = false,
    ) where {Axis}
    component = Axis == 1 ? momentum.x : momentum.y
    fraction = TP._phase_face_average!(
        FaceField{Axis}(interface.mesh), interface.conditional_fraction,
        interface.geometry,
    )
    values = FaceField{Axis}(velocity)
    c = vcat(fraction.interior, fraction.boundary)
    u = vcat(values.interior, values.boundary)
    volumes = component.advection.schedule.volumes
    amounts = volumes .* (density[1] .* c .+ density[2] .* (1 .- c)) .* u
    return absolute ? sum(abs, amounts) : sum(amounts)
end

@testset "Cut coarse-fine mass-momentum consistency" begin
    interface, carrier = cut_momentum_fixture()
    mesh = interface.mesh
    momentum = MomentumAdvectionWorkspace(interface)
    # Test the actual operator identity independently of the WY solution, in
    # each direction. The former geometric-side factorization must fail here.
    old_defect = 0.0
    for component in (momentum.x, momentum.y), axis in 1:2
        k = component === momentum.x ? Val(1) : Val(2)
        new = cut_momentum_identity_defect(interface, component.advection.schedule, k, axis)
        old_schedule = TP._geometry_phase_schedule(interface.geometry, k)
        old_defect = max(old_defect, cut_momentum_identity_defect(interface, old_schedule, k, axis))
        @test new <= 2.0e-13
    end
    @test old_defect > 1.0e-4

    density = (1000.0, 1.0)
    for order in ((1, 2), (2, 1)), scheme in (Upwind(), LaxWendroff())
        field = copy(interface)
        work = WYAdvectionWorkspace(field)
        momentum = MomentumAdvectionWorkspace(field)
        destination = FaceVelocity(mesh)
        before = ntuple(k -> cut_momentum_amount(field, carrier, momentum, density, Val(k)), 2)
        scale = ntuple(
            k -> cut_momentum_amount(field, carrier, momentum, density, Val(k); absolute = true),
            2,
        )
        dt = stable_timestep(carrier, work; cfl = 0.2)
        advect!(
            field, carrier, dt, work; momentum = destination, density,
            momentum_workspace = momentum, sweep_order = order, scheme,
        )
        after = ntuple(k -> cut_momentum_amount(field, destination, momentum, density, Val(k)), 2)
        @test all(abs(after[k] - before[k]) <= 128eps() * max(scale[k], 1) for k in 1:2)

        field = copy(interface)
        work = WYAdvectionWorkspace(field)
        constant = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.3 : -0.2)
        advect!(
            field, carrier, dt, work; momentum = destination, density,
            momentum_workspace = MomentumAdvectionWorkspace(field),
            momentum_source = constant, sweep_order = order, scheme,
        )
        @test destination.interior == constant.interior
        @test destination.boundary == constant.boundary
    end
end

function wy_cut_hanging_counterexample(parameters)
    mesh = BlockTreeMesh(
        base_mesh = (4, 4), block_size = (2, 2), extent = (4.0, 4.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
    )
    centre = (1.5, 2.5)
    donor = only(
        ref for ref in TP._cell_references(mesh)
            if cellcenter(mesh, ref.block, ref.i, ref.j) == centre
    )
    geometry = GeometryField(mesh, (x, y) -> y + 100.0; lipschitz = 1)
    nw = (cos(parameters.tw), sin(parameters.tw))
    solid = TP._plane_fraction(nw..., parameters.ow)
    TP._set!(geometry.solid.fraction, solid, donor)
    TP._set!(geometry.solid.normal_x, nw[1], donor)
    TP._set!(geometry.solid.normal_y, nw[2], donor)
    TP._set!(geometry.solid.intercept, parameters.ow, donor)
    TP._set!(geometry.open_fraction, 1 - solid, donor)
    for (index, face) in pairs(mesh.faces)
        geometry.aperture[index] = TP._interior_open_aperture(geometry.solid, face)
    end
    nl = (cos(parameters.tl), sin(parameters.tl))
    liquid(x, y) = nl[1] * (x - centre[1]) + nl[2] * (y - centre[2]) - parameters.ol
    field = InterfaceField(mesh, liquid; geometry, lipschitz = 1)
    reconstruct!(field)
    right = [i for (i, face) in pairs(mesh.faces) if face.axis == 1 && face.minus == donor]
    left = [i for (i, face) in pairs(mesh.faces) if face.axis == 1 && face.plus == donor]
    @test length(right) == 2 && length(left) == 1
    return field, right, left
end

@testset "WY sliver-cell bound distinguishes round-off from violations" begin
    for T in (Float32, Float64)
        open = T(4.7e-4)
        tolerance = T(128) * eps(T)
        roundoff = tolerance / (T(4) * open)
        @test roundoff > tolerance
        @test TP._wy_fraction_in_bounds(one(T) + roundoff, open)
        @test TP._wy_fraction_in_bounds(-roundoff, open)
        @test !TP._wy_fraction_in_bounds(one(T) + T(4) * tolerance / open, open)
        @test !TP._wy_fraction_in_bounds(-T(4) * tolerance / open, open)
        @test !TP._wy_fraction_in_bounds(T(NaN), open)
        @test !TP._wy_fraction_in_bounds(T(Inf), open)
        @test !TP._wy_fraction_in_bounds(T(2), zero(T))
        @test TP._wy_fraction_in_bounds(one(T) + tolerance / 2, one(T))
        @test !TP._wy_fraction_in_bounds(one(T) + 2tolerance, one(T))
        @test !TP._wy_negative_fraction_out_of_bounds(-roundoff, open)
        @test TP._wy_negative_fraction_out_of_bounds(-T(4) * tolerance / open, open)
        @test !TP._wy_negative_fraction_out_of_bounds(T(2), open)
        @test TP._wy_negative_fraction_out_of_bounds(-2tolerance, zero(T))
    end
    @test TP._wy_fraction_in_bounds(1.000000000000035, 4.7e-4)
    @test TP._wy_fraction_in_bounds(1.000000000000035, Float32(4.7e-4))
    @test !TP._wy_negative_fraction_out_of_bounds(-4.79159355422365e-14, 4.7e-4)

    # The audited #56 counterexamples are real overshoots, not round-off.
    # The stable step must still be rejected for both phase carriers.
    for frozen in (false, true)
        parameters = frozen ?
            (
                tw = 6.047103058103431, ow = 0.3519469972142466,
                tl = 1.263045452558202, ol = 0.16999975582492988,
            ) :
            (
                tw = 3.642192944522876, ow = 0.23059525565923286,
                tl = 5.777533614551216, ol = -0.39261586027343004,
            )
        field, right, left = wy_cut_hanging_counterexample(parameters)
        mesh = field.mesh
        liquid = FaceVelocity(mesh)
        if frozen
            gas = FaceVelocity(mesh)
            liquid.interior[right[1]] = -0.10191943684044136
            liquid.interior[left[1]] = -0.19799910106953192
            gas.interior[right[1]] = -0.10848167605356168
            gas.interior[right[2]] = 0.179524103894174
            velocity = (liquid, gas)
            work = WYAdvectionWorkspace(field, velocity; interface_transport = FrozenInterfaceFluxes())
        else
            liquid.interior[right[2]] = 0.9368430659316056
            liquid.interior[left[1]] = -0.35686834133268
            velocity = liquid
            work = WYAdvectionWorkspace(field)
        end
        dt = stable_timestep(velocity, work)
        err = try
            advect!(field, velocity, dt, work; check_timestep = false)
            nothing
        catch exception
            exception
        end
        @test err isa ErrorException &&
            occursin("WY conditional fraction out of bounds", sprint(showerror, err))
    end
end
