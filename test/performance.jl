@testset "kernel allocation contract" begin
    mesh = BlockTreeMesh(block_size=(16, 16))
    scalar = CellField(mesh)
    initialize!(scalar, (x, y) -> 1 + 0.1sinpi(2x) * sinpi(2y))
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.2 : -0.1)
    workspace = AdvectionWorkspace(scalar)

    advect!(scalar, velocity, 0.001, workspace)
    fill_halo!(scalar)
    stable_timestep(velocity, workspace)

    @test @allocated(advect!(scalar, velocity, 0.001, workspace)) == 0
    @test @allocated(fill_halo!(scalar)) == 0
    @test @allocated(stable_timestep(velocity, workspace)) == 0
end
