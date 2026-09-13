@testset "placements and block tree" begin
    @test Cell() isa AbstractPlacement
    @test Face{1}() isa AbstractPlacement
    @test Face{2}() isa AbstractPlacement
    @test_throws ArgumentError Face{3}()

    oneblock = BlockTreeMesh(block_size=(8, 6))
    @test nblocks(oneblock) == 1
    @test ncells(oneblock) == 48
    @test isbalanced(oneblock)
    @test length(oneblock.faces) == 2ncells(oneblock)
    @test isempty(oneblock.boundary_faces)

    uniform = BlockTreeMesh(block_size=(4, 4), max_level=1,
                            refine=(key, origin, extent) -> true)
    @test nblocks(uniform) == 4
    @test all(block -> block.key.level == 1, uniform.blocks)
    @test isbalanced(uniform)

    twolevel = BlockTreeMesh(
        block_size=(4, 4), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    @test nblocks(twolevel) == 7
    @test count(block -> block.key.level == 1, twolevel.blocks) == 3
    @test count(block -> block.key.level == 2, twolevel.blocks) == 4
    @test isbalanced(twolevel)

    # A sharply localized level-three request is expanded to a fully 2:1
    # balanced leaf set, including diagonal neighbors.
    balanced = BlockTreeMesh(
        block_size=(2, 2), max_level=3,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.i == 0 && key.j == 0),
    )
    @test isbalanced(balanced)
    @test maximum(block.key.level for block in balanced.blocks) == 3
    right_edge_levels = [block.key.level for block in balanced.blocks
                         if block.origin[1] + block.extent[1] ≈ 1.0]
    @test maximum(right_edge_levels) >= 2

    physical = BlockTreeMesh(block_size=(4, 4), periodic=(false, false))
    @test length(physical.faces) == 24
    @test length(physical.boundary_faces) == 16

    @test_throws ArgumentError BlockTreeMesh(block_size=(3, 4))
    @test_throws ArgumentError BlockTreeMesh(halo=0)
    @test_throws ArgumentError BlockTreeMesh(max_level=-1)
end
