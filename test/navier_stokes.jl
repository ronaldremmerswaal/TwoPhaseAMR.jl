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

@testset "lid-driven cavity wall and projection" begin
    mesh = BlockTreeMesh(block_size=(16, 16), periodic=(false, false))
    lid = NoSlip((x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0)
    solver = IncompressibleNavierStokesSolver(
        mesh; viscosity=0.01, boundary=lid, coarse_size=16)
    workspace = NavierStokesWorkspace(solver)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    result = simulate!(velocity, pressure, solver, 0.1, workspace;
                       dt=0.01, rtol=2e-10, max_cycles=80)
    @test result.steps == 10
    @test result.pressure.converged
    @test all(iszero, velocity.boundary)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    @test maximum(abs, interior(divergence, 1)) <= 2e-8
    upper_horizontal = [velocity.interior[index]
        for (index, face) in pairs(mesh.faces)
        if face.axis == 1 && face.center[2] > 0.75]
    @test sum(upper_horizontal) > 0
end
