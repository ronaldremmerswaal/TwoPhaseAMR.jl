function cell_advection_interior_values(field)
    return reduce(vcat, (vec(collect(interior(field, b))) for b in 1:nblocks(field.mesh)))
end

function uniform_reference_step(values, ux, uy, dt, dx, dy)
    nx, ny = size(values)
    result = similar(values)
    for j in 1:ny, i in 1:nx
        im, ip = mod1(i - 1, nx), mod1(i + 1, nx)
        jm, jp = mod1(j - 1, ny), mod1(j + 1, ny)
        fx_right = ux >= 0 ? ux * values[i, j] : ux * values[ip, j]
        fx_left = ux >= 0 ? ux * values[im, j] : ux * values[i, j]
        fy_top = uy >= 0 ? uy * values[i, j] : uy * values[i, jp]
        fy_bottom = uy >= 0 ? uy * values[i, jm] : uy * values[i, j]
        result[i, j] = values[i, j] - dt * (
            (fx_right - fx_left) / dx +
                (fy_top - fy_bottom) / dy
        )
    end
    return result
end


@testset "centred transport is independent of base-root partitioning" begin
    one_root = BlockTreeMesh(base_mesh = (8, 8), block_size = (8, 8))
    four_roots = BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4))
    velocity_function = (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2
    initial = (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)

    function advected_values(mesh)
        scalar = CellField(mesh, initial)
        velocity = FaceVelocity(mesh, velocity_function)
        workspace = AdvectionWorkspace(scalar)
        advect!(scalar, velocity, 0.02, workspace)
        nx, ny = mesh.block_size
        return Dict(
            cellcenter(mesh, block, i, j) => interior(scalar, block)[i, j]
                for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        )
    end

    @test advected_values(one_root) == advected_values(four_roots)
end

@testset "conservative centred advection" begin
    configurations = (
        BlockTreeMesh(block_size = (8, 8)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 1,
            refine = (key, origin, extent) -> true
        ),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    velocity_function = (x, y, t, axis) -> axis == 1 ? 0.3 : -0.2

    for mesh in configurations
        scalar = CellField(mesh)
        initialize!(scalar, (x, y) -> 2.0)
        velocity = FaceVelocity(mesh, velocity_function)
        workspace = AdvectionWorkspace(scalar)
        dt = stable_timestep(velocity, workspace; cfl = 0.5)
        initial_mass = integrate(scalar)
        advect!(scalar, velocity, dt, workspace)
        @test all(==(2.0), cell_advection_interior_values(scalar))
        @test integrate(scalar) == initial_mass

        initialize!(scalar, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
        initial_mass = integrate(scalar)
        for _ in 1:12
            advect!(scalar, velocity, dt / 12, workspace)
        end
        @test integrate(scalar) ≈ initial_mass rtol = 2.0e-14
        @test minimum(cell_advection_interior_values(scalar)) >= 0.8
        @test maximum(cell_advection_interior_values(scalar)) <= 1.2
    end

    # The one-block result matches the uniform upwind finite-volume reference
    # used by MimeticFV's `:upwind` face-flux mode.
    mesh = BlockTreeMesh(block_size = (8, 6))
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> exp(sinpi(2x) + 0.5cospi(2y)))
    old = collect(interior(scalar, 1))
    ux, uy, dt = 0.25, -0.1, 0.01
    expected = uniform_reference_step(old, ux, uy, dt, 1 / 8, 1 / 6)
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? ux : uy)
    workspace = AdvectionWorkspace(scalar)
    advect!(scalar, velocity, dt, workspace)
    @test collect(interior(scalar, 1)) ≈ expected rtol = 2.0e-15 atol = 2.0e-15

    physical = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    scalar = CellField(physical)
    initialize!(scalar, (x, y) -> 0.0)
    velocity = FaceVelocity(physical, (x, y, t, axis) -> axis == 1 ? 1.0 : 0.0)
    workspace = AdvectionWorkspace(scalar)
    dt = stable_timestep(velocity, workspace; cfl = 0.5)
    advect!(
        scalar, velocity, dt, workspace;
        boundary = Dirichlet((x, y, t) -> 1.0)
    )
    @test integrate(scalar) ≈ dt
end

function cut_cell_reference_timestep(velocity, geometry; cfl)
    mesh = velocity.mesh
    rates = CellField(mesh)
    for (index, face) in pairs(mesh.faces)
        flux = abs(velocity.interior[index]) * face.area * geometry.aperture[index]
        for ref in (face.minus, face.plus)
            open = TwoPhaseAMR._get(geometry.open_fraction, ref)
            open > 0 || continue
            TwoPhaseAMR._add!(rates, flux / (mesh.blocks[ref.block].volume * open), ref)
        end
    end
    for (index, face) in pairs(mesh.boundary_faces)
        ref = face.cell
        open = TwoPhaseAMR._get(geometry.open_fraction, ref)
        open > 0 || continue
        flux = abs(velocity.boundary[index]) * face.area * geometry.boundary_aperture[index]
        TwoPhaseAMR._add!(rates, flux / (mesh.blocks[ref.block].volume * open), ref)
    end
    maxrate = maximum(cell_advection_interior_values(rates))
    return iszero(maxrate) ? Inf : cfl / maxrate
end

function cut_cell_streamfunction_velocity(mesh, geometry, wall)
    psi(x, y) = max(wall(x, y), 0) * sinpi(x)^2 * sinpi(y)^2
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
    return velocity
end

@testset "Cut-cell scalar CFL uses open face and cell measures" begin
    meshes = (
        BlockTreeMesh(block_size = (8, 8), periodic = (false, false)),
        BlockTreeMesh(
            block_size = (4, 4), max_level = 2, periodic = (false, false),
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        ),
    )
    demonstrated_old_bound_failure = false
    for mesh in meshes, slope in (-0.37, 0.23), height in (0.19, 0.313, 0.387)
        wall(x, y) = y - height - slope * x
        geometry = GeometryField(mesh, wall; lipschitz = hypot(1, slope))
        velocity = cut_cell_streamfunction_velocity(mesh, geometry, wall)
        initial(x, y) = sinpi(8x) * sinpi(8y) > 0 ? 1.0 : 0.0
        scalar = CellField(mesh, initial)
        original = copy(scalar)
        workspace = AdvectionWorkspace(scalar, geometry)
        unbound = AdvectionWorkspace(original)
        dt = stable_timestep(velocity, workspace; cfl = 0.9)
        @test dt ≈ cut_cell_reference_timestep(velocity, geometry; cfl = 0.9) rtol = 5eps()
        @test dt == stable_timestep(velocity, unbound, geometry; cfl = 0.9)
        @test isfinite(dt) && dt > 0
        advect!(scalar, velocity, dt, workspace)
        advect!(original, velocity, dt, unbound, geometry)
        @test cell_advection_interior_values(scalar) == cell_advection_interior_values(original)
        values = cell_advection_interior_values(scalar)
        @test minimum(values) >= -2.0e-13
        @test maximum(values) <= 1 + 2.0e-13

        old_scalar = CellField(mesh, initial)
        old_step = stable_timestep(velocity, unbound; cfl = 0.9)
        advect!(old_scalar, velocity, old_step, unbound, geometry)
        old_values = cell_advection_interior_values(old_scalar)
        demonstrated_old_bound_failure |= minimum(old_values) < -1.0e-10 ||
            maximum(old_values) > 1 + 1.0e-10

        # Include nonzero physical boundary fluxes in the independent rate sum.
        arbitrary = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.8 : -0.6)
        @test stable_timestep(arbitrary, workspace; cfl = 0.7) ≈
            cut_cell_reference_timestep(arbitrary, geometry; cfl = 0.7) rtol = 5eps()
    end
    @test demonstrated_old_bound_failure

    mesh = first(meshes)
    scalar = CellField(mesh; init = 0.25)
    velocity = FaceVelocity(mesh; init = 1.0)
    covered = GeometryField(mesh, (x, y) -> -1.0; lipschitz = 1)
    workspace = AdvectionWorkspace(scalar, covered)
    @test stable_timestep(velocity, workspace) == Inf
    advect!(scalar, velocity, 1.0, workspace)
    @test all(==(0.25), cell_advection_interior_values(scalar))
    @test stable_timestep(FaceVelocity(mesh), workspace) == Inf
    @test_throws ArgumentError stable_timestep(velocity, workspace; cfl = 0.0)
    @test_throws ArgumentError AdvectionWorkspace(scalar, GeometryField(last(meshes), (x, y) -> -1.0; lipschitz = 1))

    mesh32 = BlockTreeMesh(
        block_size = (8, 8), periodic = (false, false),
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
    )
    solid32(x, y) = y - 0.31f0 - 0.23f0 * x
    geometry32 = GeometryField(mesh32, solid32; lipschitz = 1.1f0)
    scalar32 = CellField(mesh32; init = 0.5f0)
    velocity32 = FaceVelocity(mesh32; init = 0.25f0)
    workspace32 = AdvectionWorkspace(scalar32, geometry32)
    dt32 = stable_timestep(velocity32, workspace32; cfl = 0.5f0)
    @test dt32 isa Float32
    @test isfinite(dt32) && dt32 > 0
    @test dt32 ≈ cut_cell_reference_timestep(velocity32, geometry32; cfl = 0.5f0) rtol = 8eps(Float32)
end
