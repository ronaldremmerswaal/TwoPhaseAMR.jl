@testset "placements and block tree" begin
    @test Cell() isa AbstractPlacement
    @test Face{1}() isa AbstractPlacement
    @test Face{2}() isa AbstractPlacement
    @test_throws ArgumentError Face{3}()

    oneblock = BlockTreeMesh(block_size = (8, 6))
    @test nblocks(oneblock) == 1
    @test ncells(oneblock) == 48
    @test isbalanced(oneblock)
    @test length(oneblock.faces) == 2ncells(oneblock)
    @test isempty(oneblock.boundary_faces)

    uniform = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> true
    )
    @test nblocks(uniform) == 4
    @test all(block -> block.key.level == 1, uniform.blocks)
    @test isbalanced(uniform)

    twolevel = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    @test nblocks(twolevel) == 7
    @test count(block -> block.key.level == 1, twolevel.blocks) == 3
    @test count(block -> block.key.level == 2, twolevel.blocks) == 4
    @test isbalanced(twolevel)

    # A sharply localized level-three request is expanded to a fully 2:1
    # balanced leaf set, including diagonal neighbors.
    balanced = BlockTreeMesh(
        block_size = (2, 2), max_level = 3,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.i == 0 && key.j == 0),
    )
    @test isbalanced(balanced)
    @test maximum(block.key.level for block in balanced.blocks) == 3
    right_edge_levels = [
        block.key.level for block in balanced.blocks
            if block.origin[1] + block.extent[1] ≈ 1.0
    ]
    @test maximum(right_edge_levels) >= 2

    physical = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    @test length(physical.faces) == 24
    @test length(physical.boundary_faces) == 16

    @test_throws ArgumentError BlockTreeMesh(block_size = (3, 4))
    @test_throws ArgumentError BlockTreeMesh(halo = 0)
    @test_throws ArgumentError BlockTreeMesh(max_level = -1)
end

@testset "indexed face-line neighbourhood search" begin
    lines = Dict(
        0 => [(lo = 0, hi = 3), (lo = 3, hi = 8)],
        2 => [(lo = 0, hi = 2), (lo = 4, hi = 8)],
        4 => [(lo = 1, hi = 5), (lo = 5, hi = 8)],
        6 => [(lo = 0, hi = 4), (lo = 6, hi = 8)],
    )
    coordinates = sort!(collect(keys(lines)))
    period = 8

    function exhaustive(normal, tangent, periodic)
        lower_distance = typemin(Int)
        upper_distance = typemax(Int)
        lower_coordinate = lower_segment = 0
        upper_coordinate = upper_segment = 0
        for coordinate in coordinates
            delta = coordinate - normal
            periodic && (delta -= round(Int, delta / period) * period)
            iszero(delta) && continue
            segment = findfirst(
                entry -> entry.lo <= tangent <= entry.hi,
                lines[coordinate],
            )
            isnothing(segment) && continue
            if delta < 0 && delta > lower_distance
                lower_distance = delta
                lower_coordinate = coordinate
                lower_segment = segment
            elseif delta > 0 && delta < upper_distance
                upper_distance = delta
                upper_coordinate = coordinate
                upper_segment = segment
            end
        end
        return (
            lower_distance, lower_coordinate, lower_segment,
            upper_distance, upper_coordinate, upper_segment,
        )
    end

    for periodic in (false, true), normal in 0:period,
            tangent in (0.5, 2.0, 3.5, 5.0, 6.5, 7.5)
        indexed = TwoPhaseAMR._bracketing_line_segments(
            lines, coordinates, normal, tangent, period, periodic,
        )
        @test indexed == exhaustive(normal, tangent, periodic)
    end
    line = lines[0]
    for tangent in (0, 2, 3, 4, 8)
        expected = findfirst(entry -> entry.lo <= tangent <= entry.hi, line)
        @test TwoPhaseAMR._line_segment_at(line, tangent) == expected
    end
end
