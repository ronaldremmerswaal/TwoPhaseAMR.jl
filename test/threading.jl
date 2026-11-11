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
