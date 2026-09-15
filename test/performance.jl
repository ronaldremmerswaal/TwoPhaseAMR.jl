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
    pressure_gradient = copy(velocity)
    gradient_operator = GradientOperator(scalar)
    gradient!(pressure_gradient, scalar, gradient_operator)

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
    @test @allocated(gradient!(pressure_gradient, scalar, gradient_operator)) == 0
    @test @allocated(convection!(vector_tendency, velocity, velocity, 0.001,
                                 vector_workspace; scheme=Centred())) == 0
    @test @allocated(advect!(velocity, 0.001, vector_workspace;
                             scheme=Upwind())) == 0

    adaptive_mesh = BlockTreeMesh(
        block_size=(4, 4), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    adaptive_pressure = CellField(adaptive_mesh)
    initialize!(adaptive_pressure, (x, y) -> sinpi(2x) * cospi(2y))
    adaptive_gradient = FaceVelocity(
        adaptive_mesh, (x, y, time, axis) -> 0.0)
    adaptive_operator = GradientOperator(adaptive_pressure)
    gradient!(adaptive_gradient, adaptive_pressure, adaptive_operator)
    @test @allocated(gradient!(adaptive_gradient, adaptive_pressure,
                               adaptive_operator)) == 0

    adaptive_velocity = FaceVelocity(
        adaptive_mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
    adaptive_face = FaceField{1}(
        adaptive_mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    adaptive_destination = FaceField{1}(adaptive_mesh)
    adaptive_workspace = StaggeredAdvectionWorkspace(adaptive_face)
    adaptive_dt = stable_timestep(
        adaptive_velocity, adaptive_workspace; cfl=0.5)
    convection!(adaptive_destination, adaptive_face, adaptive_velocity,
                adaptive_dt, adaptive_workspace; scheme=Upwind())
    convection!(adaptive_destination, adaptive_face, adaptive_velocity,
                adaptive_dt, adaptive_workspace; scheme=LW(),
                amr_interface_scheme=LW())
    convection!(adaptive_destination, adaptive_face, adaptive_velocity,
                adaptive_dt, adaptive_workspace; scheme=LW(),
                amr_interface_scheme=MC())
    @test @allocated(convection!(
        adaptive_destination, adaptive_face, adaptive_velocity,
        adaptive_dt, adaptive_workspace; scheme=Upwind())) == 0
    @test @allocated(convection!(
        adaptive_destination, adaptive_face, adaptive_velocity,
        adaptive_dt, adaptive_workspace; scheme=LW(),
        amr_interface_scheme=LW())) == 0
    @test @allocated(convection!(
        adaptive_destination, adaptive_face, adaptive_velocity,
        adaptive_dt, adaptive_workspace; scheme=LW(),
        amr_interface_scheme=MC())) == 0
    @test @allocated(stable_timestep(
        adaptive_velocity, adaptive_workspace; cfl=0.5)) == 0
end
