"""One local CMDGFM normal-jump equation associated with a mixed face."""
struct CMDGFMJumpStencil{T}
    face::Int
    transverse_faces::Vector{Int}
    transverse_weights::Vector{T}
    ratio::T
    gas_inference_face::Int
    liquid_inference_face::Int
    beta::T
    resolved::Bool
    full_remove::Bool
end

# Connectivity needs face indices only, not the geometric MetricFace objects
# used to construct K. Store the same endpoint patches in compact arrays.
mutable struct CMDGFMVertexConnectivity
    lookup::Dict{Tuple{Int, Int}, Int}
    endpoints::Vector{NTuple{2, Int}}
    offsets::Vector{Int}
    faces::Vector{Int}
    counts::Vector{Int}
end
CMDGFMVertexConnectivity() = CMDGFMVertexConnectivity(
    Dict{Tuple{Int, Int}, Int}(), NTuple{2, Int}[], Int[], Int[], Int[])

mutable struct CMDGFMStencilStorage{T}
    all::Vector{CMDGFMJumpStencil{T}}
    resolved::Vector{CMDGFMJumpStencil{T}}
    mixed::Vector{Int}
    faces::Vector{Vector{Int}}
    weights::Vector{Vector{T}}
    parallel::Vector{Int}
    normal::Vector{Int}
end
CMDGFMStencilStorage(::Type{T}) where {T} = CMDGFMStencilStorage(
    CMDGFMJumpStencil{T}[], CMDGFMJumpStencil{T}[], Int[], Vector{Int}[],
    Vector{T}[], Int[], Int[])

mutable struct CMDGFMTriplets{T}
    rows::Vector{Int}
    columns::Vector{Int}
    values::Vector{T}
    cursors::Vector{Int}
    sorted_rows::Vector{Int}
    sorted_values::Vector{T}
end
CMDGFMTriplets(::Type{T}) where {T} = CMDGFMTriplets(
    Int[], Int[], T[], Int[], Int[], T[])

const _CMDGFMDiagonalOperator{T, M} = EllipticOperator{
    T, M, GradientOperator{T, M, DiagonalInverseFaceMetric{T}, GradientConstructionCache{T}},
}
const _CMDGFMSparseOperator{T, M} = EllipticOperator{
    T, M, GradientOperator{T, M, SparseInverseFaceMetric{T, Int}, GradientConstructionCache{T}},
}

"""Fixed-topology pressure data; moving-interface coefficients are not frozen.

The incidence, pressure gauge and vertex connectivity depend on the mesh and
fixed solid only. The pressure operator retains its inverse metric and
coefficient-refresh map, but its density coefficient is updated for every
projection. Block keys and the solid revision, not array lengths alone,
determine whether this data may be reused after AMR.
"""
mutable struct CMDGFMPressureCache{T, M <: BlockTreeMesh}
    keys::Vector{BlockKey}
    geometry_revision::UInt
    initialized::Bool
    operator::Union{Nothing, _CMDGFMDiagonalOperator{T, M}, _CMDGFMSparseOperator{T, M}}
    incidence::SparseMatrixCSC{T, Int}
    pressure_indices::Vector{Int}
    pressure_anchors::Vector{Int}
    incidence_counts::Vector{Int}
    component_stack::Vector{Int}
    vertex_patches::CMDGFMVertexConnectivity
    elliptic::EllipticWorkspace{T}
    construction_cache::GradientConstructionCache{T}
    face_data::Union{Nothing, _ProjectionFaceData{T}}
    pressure_lookup::Vector{Int}
    metric_source::Union{Nothing, DiagonalInverseFaceMetric{T}, SparseInverseFaceMetric{T, Int}}
    gradient_metric_rows::Union{Nothing, DiagonalInverseFaceMetric{T}, SparseMatrixCSC{T, Int}}
    stiffness_source::Union{Nothing, SparseMatrixCSC{T, Int}}
    reduced_pressure::SparseMatrixCSC{T, Int}
    reduced_pointers::Vector{Int}
    cell_flux::Vector{T}
    coupled_matrix::SparseMatrixCSC{T, Int}
    initial_guess::Vector{T}
    stencil_storage::CMDGFMStencilStorage{T}
    jump_lookup::Vector{Int}
    pressure_jump::SparseMatrixCSC{T, Int}
    jump_pressure::SparseMatrixCSC{T, Int}
    jump_block::SparseMatrixCSC{T, Int}
    jump_rhs::Vector{T}
    pressure_triplets::CMDGFMTriplets{T}
    jump_triplets::CMDGFMTriplets{T}
end

CMDGFMPressureCache(::Type{T}, ::Type{M}) where {T, M <: BlockTreeMesh} =
    CMDGFMPressureCache{T, M}(
    BlockKey[], UInt(0), false, nothing, spzeros(T, 0, 0),
    Int[], Int[], Int[], Int[], CMDGFMVertexConnectivity(),
    EllipticWorkspace(T[], T[], T[], T[]), GradientConstructionCache(T),
    nothing, Int[], nothing, nothing, nothing, spzeros(T, 0, 0), Int[], T[], spzeros(T, 0, 0), T[],
    CMDGFMStencilStorage(T), Int[], spzeros(T, 0, 0), spzeros(T, 0, 0),
    spzeros(T, 0, 0), T[], CMDGFMTriplets(T), CMDGFMTriplets(T),
)

function _cmdgfm_vertex_connectivity!(graph::CMDGFMVertexConnectivity, mesh)
    empty!(graph.lookup)
    resize!(graph.endpoints, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        vertices = ntuple(2) do endpoint
            key = _vertex_key(mesh, face, endpoint)
            get!(graph.lookup, key) do
                length(graph.lookup) + 1
            end
        end
        graph.endpoints[index] = vertices
    end
    nv = length(graph.lookup)
    resize!(graph.counts, nv)
    fill!(graph.counts, 0)
    for vertices in graph.endpoints, vertex in vertices
        graph.counts[vertex] += 1
    end
    _csr_offsets(graph.counts, graph.offsets)
    copyto!(graph.counts, 1, graph.offsets, 1, nv)
    resize!(graph.faces, 2length(mesh.faces))
    for (face, vertices) in pairs(graph.endpoints), vertex in vertices
        graph.faces[graph.counts[vertex]] = face
        graph.counts[vertex] += 1
    end
    return graph
end

function _cmdgfm_gauge_partition!(cache, mesh, geometry::G) where {G}
    eligible = isnothing(geometry) ? nothing : geometry.active_cells
    face_weights = isnothing(geometry) ? nothing : geometry.aperture
    _projection_gauge_partition!(
        cache.pressure_indices, cache.pressure_anchors,
        cache.pressure_lookup, cache.component_stack, cache.incidence,
        mesh, eligible, face_weights,
    )
    return nothing
end

function _cmdgfm_same_topology(keys, mesh)
    length(keys) == nblocks(mesh) || return false
    for index in eachindex(keys)
        keys[index] == mesh.blocks[index].key || return false
    end
    return true
end


function _cmdgfm_check_shared_operator(solver, operator, data)
    interface = solver.interface
    isnothing(interface.geometry) || throw(ArgumentError(
        "a cut-cell CMDGFM operator cannot share the one-velocity A²/rho operator"))
    operator.mesh === interface.mesh &&
        size(operator.stiffness) == (ncells(interface.mesh), ncells(interface.mesh)) &&
        length(operator.face_coefficient) == length(interface.mesh.faces) &&
        iszero(operator.reaction) && isone(operator.diffusion) ||
        throw(ArgumentError("shared CMDGFM pressure operator has an incompatible layout"))
    T = eltype(interface)
    for index in eachindex(operator.face_coefficient)
        expected = inv(data.total_mass[index])
        actual = operator.face_coefficient[index]
        abs(actual - expected) <= 128eps(T) * max(abs(actual), abs(expected)) ||
            throw(ArgumentError("shared CMDGFM pressure operator has stale/different density"))
    end
    return operator
end

function _cmdgfm_pressure_cache!(cache::CMDGFMPressureCache{T}, solver, data;
        shared_operator = nothing) where {T}
    interface = solver.interface
    mesh = interface.mesh
    revision = _interface_geometry_revision(interface.geometry)
    current = cache.initialized && cache.geometry_revision == revision &&
        _cmdgfm_same_topology(cache.keys, mesh)
    if !current
        cache.incidence = _projection_incidence!(cache.incidence, cache.incidence_counts, mesh)
        _cmdgfm_gauge_partition!(cache, mesh, interface.geometry)
        _cmdgfm_vertex_connectivity!(cache.vertex_patches, mesh)
        cache.operator = nothing
        cache.metric_source = nothing
        cache.stiffness_source = nothing
        resize!(cache.keys, nblocks(mesh))
        for index in eachindex(cache.keys)
            cache.keys[index] = mesh.blocks[index].key
        end
        cache.geometry_revision = revision
        cache.initialized = true
        resize!(cache.elliptic.input, ncells(mesh))
        resize!(cache.elliptic.output, ncells(mesh))
        resize!(cache.elliptic.face_input, length(mesh.faces))
        resize!(cache.elliptic.face_output, length(mesh.faces))
        resize!(cache.cell_flux, ncells(mesh))
    end
    cache.operator = if isnothing(shared_operator)
        _projection_pressure_operator(solver; template = cache.operator,
            construction_cache = cache.construction_cache)
    else
        _cmdgfm_check_shared_operator(solver, shared_operator, data)
    end
    metric = cache.operator.gradient_operator.inverse_metric
    if cache.metric_source !== metric
        cache.gradient_metric_rows = metric isa DiagonalInverseFaceMetric ? metric :
            _cmdgfm_transpose!(cache.gradient_metric_rows, metric.matrix, cache.incidence_counts)
        cache.metric_source = metric
    end
    stiffness = cache.operator.stiffness::SparseMatrixCSC{T, Int}
    if cache.stiffness_source !== stiffness
        # Select the same gauge rows/columns, recording source CSC positions.
        # Retain the slice's buffers even when AMR changes its dimensions.
        reduced = cache.reduced_pressure
        columns, rows, values = reduced.colptr, rowvals(reduced), nonzeros(reduced)
        np = length(cache.pressure_indices)
        resize!(columns, np + 1)
        entries = 0
        for column in cache.pressure_indices, pointer in nzrange(stiffness, column)
            cache.pressure_lookup[rowvals(stiffness)[pointer]] > 0 && (entries += 1)
        end
        resize!(rows, entries)
        resize!(values, entries)
        resize!(cache.reduced_pointers, entries)
        cursor = 0
        for (local_column, column) in pairs(cache.pressure_indices)
            columns[local_column] = cursor + 1
            for pointer in nzrange(stiffness, column)
                local_row = cache.pressure_lookup[rowvals(stiffness)[pointer]]
                local_row > 0 || continue
                cursor += 1
                rows[cursor] = local_row
                cache.reduced_pointers[cursor] = pointer
            end
        end
        columns[end] = cursor + 1
        if size(reduced) != (np, np)
            cache.reduced_pressure = SparseMatrixCSC(np, np, columns, rows, values)
        end
        cache.stiffness_source = stiffness
    end
    for pointer in eachindex(cache.reduced_pointers)
        nonzeros(cache.reduced_pressure)[pointer] =
            nonzeros(stiffness)[cache.reduced_pointers[pointer]]
    end
    return cache
end

function _cmdgfm_transpose!(previous, source::SparseMatrixCSC{T}, cursors) where {T}
    matrix = previous isa SparseMatrixCSC{T, Int} ? previous : spzeros(T, 0, 0)
    columns, rows, values = matrix.colptr, rowvals(matrix), nonzeros(matrix)
    resize!(cursors, size(source, 1))
    fill!(cursors, 0)
    for row in rowvals(source)
        cursors[row] += 1
    end
    resize!(columns, size(source, 1) + 1)
    columns[1] = 1
    for column in eachindex(cursors)
        columns[column + 1] = columns[column] + cursors[column]
        cursors[column] = columns[column]
    end
    resize!(rows, nnz(source))
    resize!(values, nnz(source))
    for column in 1:size(source, 2), pointer in nzrange(source, column)
        row = rowvals(source)[pointer]
        target = cursors[row]
        rows[target] = column
        values[target] = nonzeros(source)[pointer]
        cursors[row] += 1
    end
    return size(matrix) == reverse(size(source)) ? matrix :
        SparseMatrixCSC(size(source, 2), size(source, 1), columns, rows, values)
end

struct CMDGFMGradientRows{T, R, G, M}
    metric_rows::R
    scale::Vector{T}
    geometry::G
    mesh::M
end

function _cmdgfm_gradient_rows(cache::CMDGFMPressureCache, geometry)
    return CMDGFMGradientRows(cache.gradient_metric_rows,
        cache.operator.square_root_face_coefficient, geometry, cache.operator.mesh)
end

_cmdgfm_gradient_rows(rows::CMDGFMGradientRows) = rows

@inline _cmdgfm_pressure_index(lookup::Vector{Int}, cell) = lookup[cell]
@inline _cmdgfm_pressure_index(lookup::AbstractDict, cell) = get(lookup, cell, 0)

function _cmdgfm_append_incidence!(rows, columns, values, gradient::CMDGFMGradientRows,
        pressure_lookup, row, incident_face, factor)
    nx, ny = gradient.mesh.block_size
    face = gradient.mesh.faces[incident_face]
    coefficient = factor * face.area
    for (reference, sign) in ((face.minus, -1), (face.plus, 1))
        cell = _cell_index(reference, nx, ny)
        local_pressure = _cmdgfm_pressure_index(pressure_lookup, cell)
        iszero(local_pressure) && continue
        push!(rows, row)
        push!(columns, local_pressure)
        push!(values, sign * coefficient)
    end
    return nothing
end

function _cmdgfm_append_gradient_row!(rows, columns, values, gradient::CMDGFMGradientRows,
        pressure_lookup, row, face, factor)
    # Exactly the same common-gradient row as S K S I (and A^-1 for a
    # cut cell), evaluated only at CMDGFM's inference donors. K is not changed.
    aperture = isnothing(gradient.geometry) ? one(eltype(gradient.scale)) :
        gradient.geometry.aperture[face]
    iszero(aperture) && return nothing
    metric = gradient.metric_rows
    if metric isa DiagonalInverseFaceMetric
        coefficient = gradient.scale[face] * metric.diagonal[face] * gradient.scale[face]
        _cmdgfm_append_incidence!(rows, columns, values, gradient, pressure_lookup,
            row, face, factor * coefficient / aperture)
    else
        for pointer in nzrange(metric, face)
            incident_face = rowvals(metric)[pointer]
            coefficient = gradient.scale[face] * nonzeros(metric)[pointer] *
                gradient.scale[incident_face]
            _cmdgfm_append_incidence!(rows, columns, values, gradient, pressure_lookup,
                row, incident_face, factor * coefficient / aperture)
        end
    end
    return nothing
end

function _cmdgfm_pressure_jump(cache::CMDGFMPressureCache{T}, data, mixed_faces, mesh) where {T}
    # Each xi column touches at most its two adjacent retained pressure cells.
    # Write that CSC column directly rather than convert general triplets.
    matrix = cache.pressure_jump
    columns, rows, values = matrix.colptr, rowvals(matrix), nonzeros(matrix)
    resize!(columns, length(mixed_faces) + 1)
    resize!(rows, 2length(mixed_faces))
    resize!(values, length(rows))
    cursor = 1
    nx, ny = mesh.block_size
    for (column, index) in pairs(mixed_faces)
        columns[column] = cursor
        face = mesh.faces[index]
        coefficient = data.coupling[index] * T(face.area)
        minus = cache.pressure_lookup[_cell_index(face.minus, nx, ny)]
        plus = cache.pressure_lookup[_cell_index(face.plus, nx, ny)]
        if minus == plus
            if minus > 0
                rows[cursor], values[cursor] = minus, zero(T)
                cursor += 1
            end
        else
            for (row, value) in (minus < plus ?
                    ((minus, -coefficient), (plus, coefficient)) :
                    ((plus, coefficient), (minus, -coefficient)))
                iszero(row) && continue
                rows[cursor], values[cursor] = row, value
                cursor += 1
            end
        end
    end
    columns[end] = cursor
    resize!(rows, cursor - 1)
    resize!(values, cursor - 1)
    dimensions = (length(cache.pressure_indices), length(mixed_faces))
    if size(matrix) != dimensions
        cache.pressure_jump = SparseMatrixCSC(dimensions..., columns, rows, values)
    end
    return cache.pressure_jump
end

function _cmdgfm_clear_triplets!(scratch::CMDGFMTriplets)
    empty!(scratch.rows)
    empty!(scratch.columns)
    empty!(scratch.values)
    return scratch.rows, scratch.columns, scratch.values
end

function _cmdgfm_sorted_rows_csc!(matrix::SparseMatrixCSC{T}, scratch, m, n) where {T}
    # Jump assembly emits rows in increasing order. Counting-sort by column
    # therefore gives sorted CSC rows without a general sparse conversion.
    # Contributions at an identical (row,column) are summed in emission order,
    # just as the reference sparse(rows, columns, values, ...). Explicit zeros
    # are retained: neither the operator nor ILU's input pattern is changed.
    columns, rows, values = matrix.colptr, rowvals(matrix), nonzeros(matrix)
    cursors = scratch.cursors
    resize!(cursors, n)
    fill!(cursors, 0)
    for column in scratch.columns
        cursors[column] += 1
    end
    resize!(columns, n + 1)
    columns[1] = 1
    for column in 1:n
        columns[column + 1] = columns[column] + cursors[column]
        cursors[column] = columns[column]
    end
    sorted_rows, sorted_values = scratch.sorted_rows, scratch.sorted_values
    resize!(sorted_rows, length(scratch.rows))
    resize!(sorted_values, length(scratch.values))
    for index in eachindex(scratch.rows)
        column = scratch.columns[index]
        target = cursors[column]
        sorted_rows[target] = scratch.rows[index]
        sorted_values[target] = scratch.values[index]
        cursors[column] += 1
    end
    resize!(rows, length(sorted_rows))
    resize!(values, length(sorted_values))
    cursor = 1
    for column in 1:n
        first, last = columns[column], columns[column + 1] - 1
        columns[column] = cursor
        for pointer in first:last
            row, value = sorted_rows[pointer], sorted_values[pointer]
            if pointer > first && row == rows[cursor - 1]
                values[cursor - 1] += value
            else
                rows[cursor], values[cursor] = row, value
                cursor += 1
            end
        end
    end
    columns[end] = cursor
    resize!(rows, cursor - 1)
    resize!(values, cursor - 1)
    return size(matrix) == (m, n) ? matrix : SparseMatrixCSC(m, n, columns, rows, values)
end

function _cmdgfm_cell_flux!(result, data, interface, velocity)
    geometry = interface.geometry
    # I'*(a_l*u_l + a_g*u_g) - boundary flux. No absent-phase value is
    # evaluated and no unused centred normal constraint is constructed.
    fill!(result, zero(eltype(result)))
    mesh = interface.mesh
    nx, ny = mesh.block_size
    for (index, face) in pairs(mesh.faces)
        mixture = _projection_phase_sum(
            data.liquid_aperture[index], velocity[1].interior[index],
            data.gas_aperture[index], velocity[2].interior[index],
        )
        flux = face.area * mixture
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        result[minus] -= flux
        result[plus] += flux
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid, gas = _phase_boundary_face_apertures(interface, geometry, index)
        flux = face.side * face.area * _projection_phase_sum(
            liquid, velocity[1].boundary[index], gas, velocity[2].boundary[index],
        )
        cell = _cell_index(face.cell, nx, ny)
        result[cell] -= flux
    end
    return result
end

function _cmdgfm_matrix_asymmetry(matrix::SparseMatrixCSC{T}) where {T}
    # Sorted CSC lookup avoids allocating A-A' just for a scalar diagnostic.
    result = zero(T)
    for column in 1:size(matrix, 2)
        for pointer in nzrange(matrix, column)
            row = rowvals(matrix)[pointer]
            result = max(result, abs(nonzeros(matrix)[pointer] - matrix[column, row]))
        end
    end
    return result
end

function _cmdgfm_block_matrix!(cache::CMDGFMPressureCache{T}, PP, PX, XP, XX) where {T}
    # The jump set changes almost every step. Refill CSC structure as well as
    # values, retaining buffer capacity rather than assuming a fixed pattern.
    # Like all projection scratch, an assembly is valid until the next call;
    # callers needing a persistent snapshot must copy it.
    np, nx = size(PP, 2), size(XX, 2)
    number_of_columns = np + nx
    matrix = cache.coupled_matrix
    columns, rows, values = matrix.colptr, rowvals(matrix), nonzeros(matrix)
    resize!(columns, number_of_columns + 1)
    number_of_entries = nnz(PP) + nnz(PX) + nnz(XP) + nnz(XX)
    resize!(rows, number_of_entries)
    resize!(values, number_of_entries)
    cursor = 1
    for (offset, top, bottom) in ((0, PP, XP), (np, PX, XX))
        for column in 1:size(top, 2)
            columns[offset + column] = cursor
            for (row_offset, block) in ((0, top), (np, bottom))
                for pointer in nzrange(block, column)
                    rows[cursor] = row_offset + rowvals(block)[pointer]
                    values[cursor] = nonzeros(block)[pointer]
                    cursor += 1
                end
            end
        end
    end
    columns[end] = cursor
    if size(matrix) != (number_of_columns, number_of_columns)
        matrix = SparseMatrixCSC{T, Int}(number_of_columns, number_of_columns,
            columns, rows, values)
        cache.coupled_matrix = matrix
    end
    return matrix
end
