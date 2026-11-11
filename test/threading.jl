using TwoPhaseAMR: @_parallel, _parallel_for, _parallel_block_columns, _PARALLEL_GRAIN

# Run `f` with every loop above `grain` work units threaded.
function with_parallel_grain(f, grain)
    previous = _PARALLEL_GRAIN[]
    _PARALLEL_GRAIN[] = grain
    try
        return f()
    finally
        _PARALLEL_GRAIN[] = previous
    end
end

function parallel_loop_features(values)
    @_parallel for index in eachindex(values)
        iseven(index) && continue
        total = 0
        for term in 1:index
            term > 5 && break
            isodd(term) || continue
            total += term
        end
        values[index] = total
    end
    return values
end

function scaled_copy!(destination, source)
    @_parallel for index in eachindex(destination)
        @inbounds destination[index] = 2source[index]
    end
    return destination
end

expand_parallel(loop) = macroexpand(TwoPhaseAMR, :(@_parallel $loop))

@testset "parallel loop launch" begin
    @test _PARALLEL_GRAIN[] == 1024
    expected = [isodd(index) ? sum(filter(isodd, 1:min(index, 5))) : -1 for index in 1:37]
    for grain in (1, _PARALLEL_GRAIN[])
        @test with_parallel_grain(() -> parallel_loop_features(fill(-1, 37)), grain) == expected
    end

    @test_throws "cannot break" expand_parallel(
        :(
            for index in 1:3
                break
            end
        )
    )
    @test_throws "cannot return" expand_parallel(
        :(
            for index in 1:3
                for inner in 1:2
                    return inner
                end
            end
        )
    )
    @test_throws "only `work = expression`" macroexpand(
        TwoPhaseAMR, Meta.parse("@_parallel grain = 4 for index in 1:3\nend"),
    )

    visited = zeros(Int, 100)
    with_parallel_grain(1) do
        _parallel_for(length(visited); work = 3) do index
            visited[index] += index
        end
    end
    @test visited == 1:100
    @test_throws ArgumentError _parallel_for(identity, -1)

    # The loop form launches its body without a closure argument, so a warm
    # launch allocates nothing on either path.
    destination, source = zeros(4096), rand(4096)
    for grain in (1, _PARALLEL_GRAIN[])
        with_parallel_grain(grain) do
            scaled_copy!(destination, source)
            @test @allocated(scaled_copy!(destination, source)) == 0
        end
    end
    @test destination == 2source

    for mesh in (BlockTreeMesh(block_size = (8, 6)), BlockTreeMesh(base_mesh = (32, 16), block_size = (8, 8)))
        nx, ny = mesh.block_size
        owners = zeros(Int, nx * ny * nblocks(mesh))
        with_parallel_grain(1) do
            _parallel_block_columns(mesh) do block, first_j, last_j
                for j in first_j:last_j, i in 1:nx
                    owners[(block - 1) * nx * ny + (j - 1) * nx + i] += 1
                end
            end
        end
        @test all(==(1), owners)
    end
end

@testset "parallel failures and user callbacks" begin
    # Inputs are validated before threaded loops, which therefore never
    # surface a TaskFailedException from a Polyester worker.
    mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.5; lipschitz = 1)
    interface.fraction.blocks[end][3, 3] = 2.0
    with_parallel_grain(1) do
        @test_throws ArgumentError TwoPhaseAMR._refresh_conditional_fraction!(interface)
    end

    active = trues(ncells(mesh))
    volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
    operator = EllipticOperator(
        mesh; face_coefficient = ones(length(mesh.faces)), reaction = 0.75,
        active_cells = active, volumes, inactive_condition = :dirichlet,
    )
    constraints = copy(operator.constraints)
    operator_volumes = copy(operator.volumes)
    inconsistent = copy(volumes)
    inconsistent[end] = 0.0
    with_parallel_grain(1) do
        @test_throws ArgumentError TwoPhaseAMR._update_elliptic_activity!(
            operator, active, inconsistent,
        )
    end
    @test operator.constraints == constraints
    @test operator.volumes == operator_volumes

    # Initialization stays transactional now that the level set runs serially.
    fraction = copy.(interface.fraction.blocks)
    failing_levelset(x, y) = x > 0.7 ? error("level set failed") : y - 0.5
    @test_throws "level set failed" initialize!(interface, failing_levelset; lipschitz = 1)
    @test interface.fraction.blocks == fraction

    # User callbacks run serially on the calling task, so a callback that is
    # not thread-safe (here: push! to a shared vector) is still correct.
    calls = Int[]
    field = CellField(mesh)
    velocity = FaceVelocity(mesh)
    with_parallel_grain(1) do
        initialize!(field, (x, y) -> (push!(calls, Threads.threadid()); x))
        set_velocities!(velocity, (x, y, time, axis) -> (push!(calls, Threads.threadid()); y))
        fill_halo!(field, Dirichlet((x, y, time) -> (push!(calls, Threads.threadid()); 0.0)))
        @test_throws ArgumentError PhaseScalar(interface; liquid = (x, y) -> x > 0.9 ? NaN : 1.0)
    end
    physical_halos = count(entry -> entry.physical_boundary, mesh.halo_entries)
    @test physical_halos > 0
    @test length(calls) ==
        ncells(mesh) + length(mesh.faces) + length(mesh.boundary_faces) + physical_halos
    @test all(==(Threads.threadid()), calls)
end

@testset "shared-memory kernel scheduling" begin
    mesh = BlockTreeMesh(block_size = (64, 64))
    field = CellField(mesh)
    initialize!(field, (x, y) -> x + 2y)
    field_sum = integrate(field)
    @test field_sum ≈ 1.5 atol = 32eps()

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
