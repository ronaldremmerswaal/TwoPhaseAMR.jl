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
        @test all(==(2.0), all_interior_values(scalar))
        @test integrate(scalar) == initial_mass

        initialize!(scalar, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))
        initial_mass = integrate(scalar)
        for _ in 1:12
            advect!(scalar, velocity, dt / 12, workspace)
        end
        @test integrate(scalar) ≈ initial_mass rtol = 2.0e-14
        @test minimum(all_interior_values(scalar)) >= 0.8
        @test maximum(all_interior_values(scalar)) <= 1.2
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
