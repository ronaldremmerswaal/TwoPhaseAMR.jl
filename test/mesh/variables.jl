function all_interior_values(field)
    return reduce(vcat, (vec(collect(interior(field, b))) for b in 1:nblocks(field.mesh)))
end

@testset "cell fields, halos, restriction, and prolongation" begin
    meshes = (
        BlockTreeMesh(block_size = (4, 4)),
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

    for mesh in meshes
        field = CellField(mesh, Float64)
        @test placement(field) == Cell()
        initialize!(field, (x, y) -> 3.25)
        fill_halo!(field)
        @test integrate(field) ≈ 3.25 * prod(mesh.extent)
        for block in field.blocks
            @test all(==(3.25), block)
        end
    end

    # A coarse ghost cell adjacent to finer leaves receives the conservative
    # average of the four fine cells occupying its footprint.
    twolevel = meshes[3]
    linear = CellField(twolevel)
    initialize!(linear, (x, y) -> x)
    fill_halo!(linear)
    coarse_index = findfirst(block -> block.key == BlockKey(1, 1, 0), twolevel.blocks)
    coarse = twolevel.blocks[coarse_index]
    for j in 1:twolevel.block_size[2]
        expected_x = coarse.origin[1] - coarse.spacing[1] / 2
        @test blockdata(linear, coarse_index)[1, j + twolevel.halo] ≈ expected_x
    end

    physical = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    field = CellField(physical)
    initialize!(field, (x, y) -> x + 2y)
    fill_halo!(field, Dirichlet((x, y, t) -> t); time = 7.0)
    h = physical.halo
    @test all(field.blocks[1][1, :] .== 7.0)
    @test all(field.blocks[1][end, :] .== 7.0)
    @test all(field.blocks[1][:, 1] .== 7.0)
    @test all(field.blocks[1][:, end] .== 7.0)

    h = 1
    parent = zeros(6, 6)
    for j in 1:4, i in 1:4
        parent[i + h, j + h] = i + 10j
    end
    children = ntuple(_ -> fill(NaN, 6, 6), 4)
    prolong!(children, parent, h)
    reconstructed = zeros(6, 6)
    restrict!(reconstructed, children, h)
    @test reconstructed[2:5, 2:5] == parent[2:5, 2:5]

    # Limited linear prolongation is affine-exact when a valid one-cell parent
    # halo is available, while its four subcell averages remain conservative.
    for j in 0:5, i in 0:5
        parent[i + 1, j + 1] = 1 + 0.3(i - 1.5) - 0.2(j - 1.5)
    end
    prolong!(children, parent, h)
    for (child_index, child) in pairs(children), j in 1:4, i in 1:4
        child_x = mod(child_index - 1, 2)
        child_y = (child_index - 1) ÷ 2
        fine_i = child_x * 4 + i
        fine_j = child_y * 4 + j
        x = (fine_i - 0.5) / 2 - 1
        y = (fine_j - 0.5) / 2 - 1
        @test child[i + h, j + h] ≈ 1 + 0.3x - 0.2y atol = 4eps()
    end
    restrict!(reconstructed, children, h)
    @test reconstructed[2:5, 2:5] ≈ parent[2:5, 2:5] atol = 4eps()

    for (child_index, child) in pairs(children), j in 1:4, i in 1:4
        child[i + h, j + h] = child_index + i / 10 + j / 100
    end
    restrict!(reconstructed, children, h)
    parent_mass = sum(@view reconstructed[2:5, 2:5])
    fine_mass = sum(sum(@view child[2:5, 2:5]) for child in children) / 4
    @test parent_mass ≈ fine_mass rtol = 8eps()

    mesh32 = BlockTreeMesh(
        origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
        block_size = (4, 4)
    )
    @test eltype(CellField(mesh32)) == Float32
    @test eltype(FaceVelocity(mesh32, (x, y, t, axis) -> 0.0f0).interior) == Float32
end
