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

function TwoPhaseAMR._cmdgfm_append_gradient_row!(rows, columns, values,
        gradient::SparseMatrixCSC, pressure_lookup, row, face, factor)
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
    return TwoPhaseAMR._cmdgfm_incidence!(spzeros(T, 0, 0), Int[], mesh)
end
