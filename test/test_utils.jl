function test_cell_values(field::CellField)
    nx, ny = field.mesh.block_size
    return [
        interior(field, block)[i, j]
            for block in eachindex(field.mesh.blocks)
            for j in 1:ny for i in 1:nx
    ]
end
