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

    face = FaceField{1}(mesh, (x, y) -> 1 + 0.1sinpi(2x) * sinpi(2y))
    staggered_workspace = StaggeredAdvectionWorkspace(face)
    face_rate = FaceField{1}(mesh)
    convection!(face_rate, face, velocity, 0.001, staggered_workspace;
                scheme=Centred())
    advect!(face, velocity, 0.001, staggered_workspace; scheme=LW())
    stable_timestep(velocity, staggered_workspace)
    divergence = CellField(mesh)
    divergence!(divergence, velocity)

    vector_workspace = StaggeredAdvectionWorkspace(velocity)
    vector_tendency = copy(velocity)
    convection!(vector_tendency, velocity, velocity, 0.001, vector_workspace;
                scheme=Centred())

    @test @allocated(convection!(face_rate, face, velocity, 0.001,
                                 staggered_workspace; scheme=Centred())) == 0
    @test @allocated(advect!(face, velocity, 0.001, staggered_workspace;
                             scheme=LW())) == 0
    @test @allocated(stable_timestep(velocity, staggered_workspace)) == 0
    @test @allocated(divergence!(divergence, velocity)) == 0
    @test @allocated(convection!(vector_tendency, velocity, velocity, 0.001,
                                 vector_workspace; scheme=Centred())) == 0
    @test @allocated(advect!(velocity, 0.001, vector_workspace;
                             scheme=Upwind())) == 0
end
