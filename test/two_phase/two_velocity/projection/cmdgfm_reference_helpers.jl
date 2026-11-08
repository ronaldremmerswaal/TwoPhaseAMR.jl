using LinearAlgebra, SparseArrays, TwoPhaseAMR

function _cmdgfm_metric_matrix(metric::TwoPhaseAMR.DiagonalInverseFaceMetric{T}) where {T}
    return spdiagm(0 => metric.diagonal)
end

_cmdgfm_metric_matrix(metric::TwoPhaseAMR.SparseInverseFaceMetric) = metric.matrix

function _cmdgfm_common_gradient_matrix(
        operator::TwoPhaseAMR.EllipticOperator{T}, incidence,
        geometry,
    ) where {T}
    scale = operator.square_root_face_coefficient
    matrix = spdiagm(0 => scale) *
        _cmdgfm_metric_matrix(operator.gradient_operator.inverse_metric) *
        spdiagm(0 => scale) * incidence
    if !isnothing(geometry)
        inverse_aperture = map(geometry.aperture) do aperture
            aperture > zero(T) ? inv(aperture) : zero(T)
        end
        matrix = spdiagm(0 => inverse_aperture) * matrix
    end
    dropzeros!(matrix)
    return matrix
end

TwoPhaseAMR._cmdgfm_gradient_rows(matrix::SparseMatrixCSC) = copy(transpose(matrix))

function TwoPhaseAMR._cmdgfm_append_gradient_row!(
        rows, columns, values,
        gradient::SparseMatrixCSC, pressure_lookup, row, face, factor
    )
    for pointer in nzrange(gradient, face)
        cell = rowvals(gradient)[pointer]
        local_pressure = get(pressure_lookup, cell, 0)
        iszero(local_pressure) && continue
        push!(rows, row)
        push!(columns, local_pressure)
        push!(values, factor * nonzeros(gradient)[pointer])
    end
    return nothing
end

function _cmdgfm_incidence(mesh, ::Type{T}) where {T}
    return TwoPhaseAMR._projection_incidence!(spzeros(T, 0, 0), Int[], mesh)
end

# Independent triplet oracle for the compact cached incidence builder.
function _reference_projection_incidence(mesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    rows, columns, values = Int[], Int[], T[]
    for (index, face) in pairs(mesh.faces)
        push!(rows, index, index)
        push!(
            columns, TwoPhaseAMR._cell_index(face.minus, nx, ny),
            TwoPhaseAMR._cell_index(face.plus, nx, ny)
        )
        push!(values, -T(face.area), T(face.area))
    end
    return sparse(rows, columns, values, length(mesh.faces), ncells(mesh))
end

# Independent reference for the production incidence-based gauge traversal.
function _reference_gauge_partition(mesh, eligible, face_weights)
    nx, ny = mesh.block_size
    adjacency = [Int[] for _ in 1:ncells(mesh)]
    for (index, face) in pairs(mesh.faces)
        iszero(face_weights[index]) && continue
        minus = TwoPhaseAMR._cell_index(face.minus, nx, ny)
        plus = TwoPhaseAMR._cell_index(face.plus, nx, ny)
        eligible[minus] && eligible[plus] || continue
        push!(adjacency[minus], plus)
        push!(adjacency[plus], minus)
    end
    retained, anchors = Int[], Int[]
    visited = falses(length(eligible))
    stack = Int[]
    for root in eachindex(eligible)
        eligible[root] && !visited[root] || continue
        empty!(stack)
        push!(stack, root)
        visited[root] = true
        component = Int[]
        while !isempty(stack)
            cell = pop!(stack)
            push!(component, cell)
            for neighbour in adjacency[cell]
                visited[neighbour] && continue
                visited[neighbour] = true
                push!(stack, neighbour)
            end
        end
        length(component) > 1 || continue
        anchor = minimum(component)
        for cell in component
            cell == anchor && continue
            push!(retained, cell)
            push!(anchors, anchor)
        end
    end
    permutation = sortperm(retained)
    return retained[permutation], anchors[permutation]
end
