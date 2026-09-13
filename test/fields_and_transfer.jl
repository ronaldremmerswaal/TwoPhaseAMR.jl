function all_interior_values(field)
    reduce(vcat, (vec(collect(interior(field, b))) for b in 1:nblocks(field.mesh)))
end

@testset "cell fields, halos, restriction, and prolongation" begin
    meshes = (
        BlockTreeMesh(block_size=(4, 4)),
        BlockTreeMesh(block_size=(4, 4), max_level=1,
                      refine=(key, origin, extent) -> true),
        BlockTreeMesh(
            block_size=(4, 4), max_level=2,
            refine=(key, origin, extent) ->
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

    physical = BlockTreeMesh(block_size=(4, 4), periodic=(false, false))
    field = CellField(physical)
    initialize!(field, (x, y) -> x + 2y)
    fill_halo!(field, Dirichlet((x, y, t) -> t); time=7.0)
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

    for (child_index, child) in pairs(children), j in 1:4, i in 1:4
        child[i + h, j + h] = child_index + i / 10 + j / 100
    end
    restrict!(reconstructed, children, h)
    parent_mass = sum(@view reconstructed[2:5, 2:5])
    fine_mass = sum(sum(@view child[2:5, 2:5]) for child in children) / 4
    @test parent_mass ≈ fine_mass rtol=8eps()
end
