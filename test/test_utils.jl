function test_cell_values(field::CellField)
    nx, ny = field.mesh.block_size
    return [
        interior(field, block)[i, j]
            for block in eachindex(field.mesh.blocks)
            for j in 1:ny for i in 1:nx
    ]
end

# Julia 1.11's @allocated evaluates in the surrounding @testset scope, where
# call-site boxing can be counted; 1.12 adds a function barrier for calls.
# Measure the specialized call itself and retain the newer direct checks.
test_allocated_call(f::F) where {F} = @allocated f()
