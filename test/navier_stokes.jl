function _velocity_l2_error(velocity, exact, time, solver)
    squared = 0.0
    total_volume = 0.0
    for component in (solver.diffusion.x, solver.diffusion.y)
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            volume = component.volumes[local_index]
            face = velocity.mesh.faces[global_index]
            value = exact(face.center[1], face.center[2], time, face.axis)
            squared += volume * (velocity.interior[global_index] - value)^2
            total_volume += volume
        end
    end
    return sqrt(squared / total_volume)
end

function _taylor_green_error(nx; mixed=false)
    mesh = mixed ? BlockTreeMesh(
        block_size=(nx, nx), extent=(2pi, 2pi), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    ) : BlockTreeMesh(block_size=(nx, nx), extent=(2pi, 2pi))
    viscosity = 0.02
    exact = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    velocity = FaceVelocity(mesh, exact)
    pressure = CellField(mesh)
    solver = IncompressibleNavierStokesSolver(
        mesh; viscosity, boundary=FreeSlip(), coarse_size=16)
    workspace = NavierStokesWorkspace(solver)
    final_time = 0.02
    result = simulate!(velocity, pressure, solver, final_time, workspace;
                       dt=0.02 / nx, rtol=2e-10, max_cycles=80)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    maximum_divergence = maximum(maximum(abs, interior(divergence, block))
                                 for block in eachindex(mesh.blocks))
    return (_velocity_l2_error(velocity, exact, final_time, solver),
            maximum_divergence, result)
end

@testset "Taylor--Green Navier--Stokes integration" begin
    coarse = _taylor_green_error(12)
    fine = _taylor_green_error(24)
    @test fine[1] < coarse[1]
    @test log2(coarse[1] / fine[1]) > 1.8
    @test coarse[2] <= 2e-8
    @test fine[2] <= 2e-8
    @test coarse[3].pressure.converged
    @test fine[3].pressure.converged

    mixed = _taylor_green_error(8; mixed=true)
    @test mixed[1] < 0.08
    @test mixed[2] <= 2e-8
end

@testset "Navier--Stokes timestep is convective only" begin
    mesh = BlockTreeMesh(block_size=(4, 4))
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.25 : -0.125)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; viscosity=100.0, boundary=FreeSlip(), coarse_size=8)
    workspace = NavierStokesWorkspace(solver)

    expected = stable_timestep(velocity, workspace.advection; cfl=0.35)
    @test stable_timestep(velocity, solver, workspace; cfl=0.35) == expected

    fill!(velocity.interior, 0.0)
    result = simulate!(
        velocity, pressure, solver, 0.2, workspace;
        rtol=1e-10, max_cycles=20)
    @test result.steps == 1
    @test result.time == 0.2
    @test result.pressure.converged
end

function _cavity_centerline(velocity, axis, coordinates)
    points = Tuple{Float64,Float64}[]
    for (index, face) in pairs(velocity.mesh.faces)
        face.axis == axis || continue
        abs(face.center[axis] - 0.5) <= 32eps() || continue
        push!(points, (face.center[3 - axis], velocity.interior[index]))
    end
    if axis == 1
        push!(points, (0.0, 0.0))
        push!(points, (1.0, 1.0))
    else
        push!(points, (0.0, 0.0))
        push!(points, (1.0, 0.0))
    end
    sort!(points; by=first)
    abscissae = first.(points)
    return map(coordinates) do coordinate
        left_index = clamp(
            searchsortedlast(abscissae, coordinate), 1, length(points) - 1)
        left = points[left_index]
        right = points[left_index + 1]
        fraction = (coordinate - left[1]) / (right[1] - left[1])
        return left[2] + fraction * (right[2] - left[2])
    end
end

@testset "Ghia Re=100 lid-driven cavity" begin
    # Tables I and II of Ghia, Ghia, and Shin, JCP 48 (1982), pp. 398--399,
    # DOI 10.1016/0021-9991(82)90058-4. End points are omitted because they
    # are prescribed wall values rather than independent validation samples.
    ghia_y = [
        0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
        0.7344, 0.6172, 0.5000, 0.4531, 0.2813,
        0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
    ]
    ghia_u = [
         0.84123,  0.78871,  0.73722,  0.68717,  0.23151,
         0.00332, -0.13641, -0.20581, -0.21090, -0.15662,
        -0.10150, -0.06434, -0.04775, -0.04192, -0.03717,
    ]
    ghia_x = [
        0.9688, 0.9609, 0.9531, 0.9453, 0.9063,
        0.8594, 0.8047, 0.5000, 0.2344, 0.2266,
        0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
    ]
    ghia_v = [
        -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
        -0.22445, -0.24533,  0.05454,  0.17527,  0.17507,
         0.16077,  0.12317,  0.10890,  0.10091,  0.09233,
    ]

    mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
    lid = NoSlip((x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0)
    solver = IncompressibleNavierStokesSolver(
        mesh; viscosity=0.01, boundary=lid, coarse_size=32)
    workspace = NavierStokesWorkspace(solver)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    result = simulate!(velocity, pressure, solver, 15.0, workspace;
                       dt=0.005, rtol=1e-8, max_cycles=100)
    @test result.pressure.converged
    @test all(iszero, velocity.boundary)

    numerical_u = _cavity_centerline(velocity, 1, ghia_y)
    numerical_v = _cavity_centerline(velocity, 2, ghia_x)
    u_error = numerical_u - ghia_u
    v_error = numerical_v - ghia_v
    @test sqrt(sum(abs2, u_error) / length(u_error)) < 0.004
    @test sqrt(sum(abs2, v_error) / length(v_error)) < 0.006
    @test maximum(abs, u_error) < 0.006
    @test maximum(abs, v_error) < 0.012

    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test maximum(abs, interior(divergence, 1)) <= 2e-8

    velocity_at_15 = copy(velocity.interior)
    settled = simulate!(velocity, pressure, solver, 16.0, workspace;
                        time=15.0, dt=0.005, rtol=1e-8, max_cycles=100)
    @test settled.pressure.converged
    @test maximum(abs, velocity.interior - velocity_at_15) < 1e-4
end
