@testset "shared-memory kernel scheduling" begin
    @test TwoPhaseAMR._PARALLEL_GRAIN < 4096
    @test TwoPhaseAMR._parallel_batch_size(16_384, 1024) <=
        cld(16_384, 4Threads.nthreads(:default))

    mesh = BlockTreeMesh(block_size = (64, 64))
    field = CellField(mesh)
    used_threads = [Threads.Atomic{Int}(0) for _ in 1:Threads.maxthreadid()]
    initialize!(
        field, (x, y) -> begin
            used_threads[Threads.threadid()][] = 1
            x + 2y
        end
    )
    field_sum = integrate(field)
    @test field_sum ≈ 1.5 atol = 32eps()
    if Threads.nthreads(:default) > 1
        @test count(thread -> thread[] == 1, used_threads) > 1
    end

    scalar = CellField(mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            0.1sinpi(2x)^2 * sinpi(2y) :
            -0.1sinpi(2x) * sinpi(2y)^2,
    )
    workspace = AdvectionWorkspace(scalar)
    initial_mass = integrate(scalar)
    advect!(
        scalar, velocity,
        stable_timestep(velocity, workspace; cfl = 0.2), workspace,
    )
    @test integrate(scalar) ≈ initial_mass atol = 64eps()
end
