@testset "Visualization geometry" begin
    mesh = BlockTreeMesh(
        block_size=(4, 2), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0))
    geometry = visualization_geometry(mesh)

    @test length(geometry.cells) == ncells(mesh)
    @test length(geometry.points) == 4ncells(mesh)
    @test length(geometry.triangles) == 2ncells(mesh)
    @test length(geometry.block_segments) == 8nblocks(mesh)
    @test geometry.cells[1] == CellRef(1, 1, 1)
    @test cell_for_vertex(geometry, 1) == geometry.cells[1]
    @test cell_for_vertex(geometry, 4) == geometry.cells[1]
    @test cell_for_vertex(geometry, 5) == geometry.cells[2]
    @test isnothing(cell_for_vertex(geometry, 0))
    @test isnothing(cell_for_vertex(geometry, length(geometry.points) + 1))

    for (cell_index, cell) in pairs(geometry.cells)
        block = mesh.blocks[cell.block]
        first_vertex = 4cell_index - 3
        @test geometry.points[first_vertex] ==
            (block.origin[1] + (cell.i - 1) * block.spacing[1],
             block.origin[2] + (cell.j - 1) * block.spacing[2])
        @test geometry.triangles[2cell_index - 1] ==
            (first_vertex, first_vertex + 1, first_vertex + 2)
        @test geometry.triangles[2cell_index] ==
            (first_vertex, first_vertex + 2, first_vertex + 3)
    end

    field = CellField(mesh)
    initialize!(field, (x, y) -> 2x - 3y)
    values = visualization_values(field)
    @test length(values) == ncells(mesh)
    for (index, cell) in pairs(geometry.cells)
        x, y = cellcenter(mesh, cell.block, cell.i, cell.j)
        @test values[index] == 2x - 3y
    end

    @test update_visualization!(NoVisualization(), field) isa NoVisualization
    @test isnothing(selected_cell(NoVisualization()))
end
