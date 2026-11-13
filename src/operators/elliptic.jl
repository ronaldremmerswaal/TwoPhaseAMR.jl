"""
    harmonic_face_coefficients(coefficient::CellField)

Interpolate a positive cell-centred diffusion coefficient to owned interior
faces with the distance-weighted harmonic mean.  This is the appropriate
default for discontinuous material coefficients: the normal flux is continuous
and the two half-cell resistances are added in series.
"""
function harmonic_face_coefficients(coefficient::CellField{T}) where {T}
    mesh = coefficient.mesh
    result = Vector{T}(undef, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        minus_value = _get(coefficient, face.minus)
        plus_value = _get(coefficient, face.plus)
        minus_value > zero(T) && plus_value > zero(T) ||
            throw(ArgumentError("cell coefficients must be strictly positive"))
        minus_distance = mesh.blocks[face.minus.block].spacing[face.axis] / 2
        plus_distance = mesh.blocks[face.plus.block].spacing[face.axis] / 2
        @inbounds result[index] = (minus_distance + plus_distance) /
            (minus_distance / minus_value + plus_distance / plus_value)
    end
    return result
end

function _cell_references(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    return [
        CellRef(block, i, j)
            for block in eachindex(mesh.blocks) for j in 1:ny for i in 1:nx
    ]
end

function _cell_volumes(mesh::BlockTreeMesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    result = Vector{T}(undef, ncells(mesh))
    for (block_index, block) in pairs(mesh.blocks)
        offset = (block_index - 1) * nx * ny
        fill!(@view(result[(offset + 1):(offset + nx * ny)]), T(block.volume))
    end
    return result
end

function _face_coefficients(mesh::BlockTreeMesh, coefficient, ::Type{T}) where {T}
    number_of_faces = length(mesh.faces)
    values = if coefficient isa Number
        fill(T(coefficient), number_of_faces)
    elseif coefficient isa AbstractVector
        length(coefficient) == number_of_faces ||
            throw(DimensionMismatch("one coefficient is required per interior face"))
        T.(coefficient)
    elseif coefficient isa FaceVelocity
        coefficient.mesh === mesh ||
            throw(ArgumentError("face coefficient and operator must share a mesh"))
        T.(coefficient.interior)
    elseif applicable(
            coefficient, mesh.faces[1].center[1],
            mesh.faces[1].center[2], mesh.faces[1].axis
        )
        [
            T(coefficient(face.center[1], face.center[2], face.axis))
                for face in mesh.faces
        ]
    else
        throw(
            ArgumentError(
                "face_coefficient must be a number, vector, FaceVelocity, or (x, y, axis) function"
            )
        )
    end
    all(isfinite, values) || throw(ArgumentError("face coefficients must be finite"))
    all(>=(zero(T)), values) ||
        throw(ArgumentError("face coefficients must be nonnegative"))
    return values
end

_pressure_boundary_sides(::Nothing) = (false, false, false, false)
_pressure_boundary_sides(::Union{NoSlip, FreeSlip}) = (false, false, false, false)
_pressure_boundary_sides(sides::NTuple{4, Bool}) = sides
function _pressure_boundary_sides(boundary::VelocityBoundaryConditions)
    return map(
        condition -> condition isa Union{Outflow, ZeroGradientVelocity},
        (boundary.left, boundary.right, boundary.bottom, boundary.top),
    )
end

@inline _pressure_open_face(sides, face) =
    sides[2 * face.axis - (face.side < 0 ? 1 : 0)]

# A vector of interior coefficients has no boundary entries. Extend it from
# the opposite normal faces, with area weights at coarse/fine interfaces.
function _boundary_material(mesh, coefficient, sides, ::Type{T}) where {T}
    values = zeros(T, length(mesh.boundary_faces))
    any(sides) || return values
    if coefficient isa AbstractVector
        length(coefficient) == length(mesh.faces) || throw(
            DimensionMismatch("one coefficient is required per interior face"),
        )
        indices = Dict{Tuple{CellRef, Int, Int}, Int}()
        for (index, face) in pairs(mesh.boundary_faces)
            _pressure_open_face(sides, face) || continue
            indices[(face.cell, face.axis, face.side)] = index
        end
        weights = zeros(T, length(values))
        for (index, face) in pairs(mesh.faces)
            for (cell, side) in ((face.minus, -1), (face.plus, 1))
                boundary_index = get(indices, (cell, face.axis, side), 0)
                iszero(boundary_index) && continue
                values[boundary_index] += T(face.area) * T(coefficient[index])
                weights[boundary_index] += T(face.area)
            end
        end
        for (index, face) in pairs(mesh.boundary_faces)
            _pressure_open_face(sides, face) || continue
            weights[index] > zero(T) || throw(
                ArgumentError(
                    "cannot extend an interior coefficient to an open face without an opposite interior face; supply boundary_face_coefficient",
                ),
            )
            values[index] /= weights[index]
        end
    else
        for (index, face) in pairs(mesh.boundary_faces)
            _pressure_open_face(sides, face) || continue
            values[index] = if coefficient isa Number
                T(coefficient)
            elseif coefficient isa FaceVelocity
                T(coefficient.boundary[index])
            else
                T(coefficient(face.center[1], face.center[2], face.axis))
            end
        end
    end
    return values
end

function _elliptic_boundary_data!(
        diagonal::Vector{T}, gradient::Vector{T}, mesh, sides,
        material::Vector{T}, aperture::Vector{T}, constraints,
    ) where {T}
    fill!(diagonal, zero(T))
    fill!(gradient, zero(T))
    nx, ny = mesh.block_size
    for (index, face) in pairs(mesh.boundary_faces)
        _pressure_open_face(sides, face) || continue
        cell = _cell_index(face.cell, nx, ny)
        iszero(constraints[cell]) && aperture[index] > zero(T) || continue
        factor = 2 * material[index] / T(mesh.blocks[face.cell.block].spacing[face.axis])
        gradient[index] = factor
        diagonal[cell] += T(face.area) * aperture[index] * factor
    end
    return diagonal
end

"""Columns per fixed chunk of `_integrated_stiffness`."""
const _ELLIPTIC_COLUMN_CHUNK = 256

"""
    _integrated_stiffness(mesh, gradient_operator, coefficient)

Symmetric integrated stiffness `(A + A') / 2` of `A = J' M J`, where `J` is
the area-weighted face jump (`-|f|` at the minus cell, `|f|` at the plus cell)
and `M` the coefficient-weighted inverse face metric: `κ K` for a diagonal
metric `K`, `sqrt(κ) K sqrt(κ)` otherwise. The pattern holds every pair of
cells that share a metric entry and the diagonal; an entry whose value
cancels is stored as zero, because a coefficient refresh can make it nonzero.

The values are bitwise those of SparseArrays' `transpose(J) * M * J` (left to
right), `(A + transpose(A)) / 2` and `dropzeros!`: each product entry is the
left-to-right sum in increasing inner index of its terms, the first term
assigned. `P = J'M` is built by face columns and `A = PJ` by cell columns,
each by a count pass, a prefix sum and a fill pass threaded over fixed chunks
of `_ELLIPTIC_COLUMN_CHUNK` columns with chunk-owned merge scratch, so the
result is the same for every thread count.
"""
function _integrated_stiffness(
        mesh::BlockTreeMesh,
        gradient_operator::GradientOperator{T},
        coefficient::Vector{T}
    ) where {T}
    nx, ny = mesh.block_size
    faces = mesh.faces
    number_of_faces = length(faces)
    number_of_cells = ncells(mesh)
    metric = gradient_operator.inverse_metric
    # Weighted metric entries, in stored order.
    weighted = Vector{T}(undef, _metric_entry_count(metric))
    square_root = metric isa DiagonalInverseFaceMetric ? coefficient : sqrt.(coefficient)
    _parallel_for(number_of_faces) do right_face
        for entry in _metric_column_entries(metric, right_face)
            left_face, value = _metric_entry(metric, entry)
            weighted[entry] = metric isa DiagonalInverseFaceMetric ?
                coefficient[right_face] * value :
                (square_root[left_face] * value) * square_root[right_face]
        end
    end
    # P = J'M by face columns; scratch bound: two cells per metric entry. Each
    # pass runs behind a function barrier: its closures capture arguments
    # only, never a binding this function reassigns (that would be boxed).
    face_width = _elliptic_face_width(metric, number_of_faces)
    face_chunks = cld(number_of_faces, _ELLIPTIC_COLUMN_CHUNK)
    scratch_rows = Vector{Int}(undef, face_chunks * face_width)
    scratch_values = Vector{T}(undef, face_chunks * face_width)
    product_counts = Vector{Int}(undef, number_of_faces)
    _elliptic_product_pass!(
        nothing, product_counts, metric, weighted, faces, nx, ny,
        scratch_rows, scratch_values, face_width,
    )
    product_offsets = _csr_offsets(product_counts)
    product_rows = Vector{Int}(undef, product_offsets[end] - 1)
    product_values = Vector{T}(undef, length(product_rows))
    _elliptic_product_pass!(
        (product_offsets, product_rows, product_values), product_counts, metric,
        weighted, faces, nx, ny, scratch_rows, scratch_values, face_width,
    )
    # A = PJ by cell columns; scratch bound: the product columns of the cell's
    # faces and the diagonal.
    incidence_offsets, incidence_faces, incidence_signs = _cell_face_incidence(mesh)
    cell_width = _elliptic_cell_width(incidence_offsets, incidence_faces, product_counts)
    cell_chunks = cld(number_of_cells, _ELLIPTIC_COLUMN_CHUNK)
    resize!(scratch_rows, cell_chunks * cell_width)
    resize!(scratch_values, cell_chunks * cell_width)
    # A diagonal entry without a term is marked by `has_term = false`.
    scratch_terms = Vector{Bool}(undef, cell_chunks * cell_width)
    column_counts = Vector{Int}(undef, number_of_cells)
    product = (product_offsets, product_rows, product_values)
    incidence = (incidence_offsets, incidence_faces, incidence_signs)
    scratch = (scratch_rows, scratch_values, scratch_terms)
    _elliptic_cell_pass!(nothing, column_counts, product, incidence, faces, scratch, cell_width)
    column_pointers = _csr_offsets(column_counts)
    rows = Vector{Int}(undef, column_pointers[end] - 1)
    products = Vector{T}(undef, length(rows))
    has_term = Vector{Bool}(undef, length(rows))
    _elliptic_cell_pass!(
        (column_pointers, rows, products, has_term), column_counts, product,
        incidence, faces, scratch, cell_width,
    )
    # Symmetrise: dropped (zero) sums become stored zeros of the pattern. A
    # missing mirror slot is recorded per column and thrown after the launch.
    stiffness_values = Vector{T}(undef, length(rows))
    missing_mirror = zeros(Bool, number_of_cells)
    _parallel_for(number_of_cells) do column
        for slot in column_pointers[column]:(column_pointers[column + 1] - 1)
            row = rows[slot]
            mirror = _csc_stored_pointer_or_zero(rows, column_pointers, column, row)
            if iszero(mirror)
                missing_mirror[column] = true
                continue
            end
            value = has_term[slot] ? (products[slot] + products[mirror]) / 2 : zero(T)
            stiffness_values[slot] = iszero(value) ? zero(T) : value
        end
    end
    any(missing_mirror) && error("sparse topology is missing a required entry")
    return SparseMatrixCSC(
        number_of_cells, number_of_cells, column_pointers, rows, stiffness_values,
    )
end

_elliptic_face_width(metric, number_of_faces) = maximum(
    right_face -> 2length(_metric_column_entries(metric, right_face)),
    1:number_of_faces; init = 0,
)

function _elliptic_cell_width(incidence_offsets, incidence_faces, product_counts)
    width = 0
    for cell in 1:(length(incidence_offsets) - 1)
        cell_width = 1
        for incidence in incidence_offsets[cell]:(incidence_offsets[cell + 1] - 1)
            cell_width += product_counts[incidence_faces[incidence]]
        end
        width = max(width, cell_width)
    end
    return width
end

"""Count (`outputs === nothing`) or fill the columns of `P = J'M` of
`_integrated_stiffness`, threaded over fixed chunks of face columns."""
function _elliptic_product_pass!(
        outputs, product_counts, metric, weighted::Vector{T}, faces, nx, ny,
        scratch_rows, scratch_values, width,
    ) where {T}
    number_of_faces = length(faces)
    chunks = cld(number_of_faces, _ELLIPTIC_COLUMN_CHUNK)
    _parallel_for(chunks; work = _ELLIPTIC_COLUMN_CHUNK) do chunk
        offset = (chunk - 1) * width
        for right_face in _chunk_columns(chunk, number_of_faces)
            stored = 0
            for entry in _metric_column_entries(metric, right_face)
                left_face, _ = _metric_entry(metric, entry)
                left = faces[left_face]
                value = weighted[entry]
                stored = _accumulate_column_term!(
                    scratch_rows, scratch_values, offset, stored,
                    _cell_index(left.minus, nx, ny), -T(left.area) * value,
                )
                stored = _accumulate_column_term!(
                    scratch_rows, scratch_values, offset, stored,
                    _cell_index(left.plus, nx, ny), T(left.area) * value,
                )
            end
            if outputs === nothing
                product_counts[right_face] = stored
            else
                product_offsets, product_rows, product_values = outputs
                _store_sorted_column!(
                    product_rows, product_values, product_offsets[right_face],
                    scratch_rows, scratch_values, offset, stored,
                )
            end
        end
    end
    return nothing
end

"""Count (`outputs === nothing`) or fill the columns of `A = PJ` of
`_integrated_stiffness`, threaded over fixed chunks of cell columns."""
function _elliptic_cell_pass!(
        outputs, column_counts, product::Tuple{Vector{Int}, Vector{Int}, Vector{T}},
        incidence, faces, scratch, width,
    ) where {T}
    product_offsets, product_rows, product_values = product
    incidence_offsets, incidence_faces, incidence_signs = incidence
    scratch_rows, scratch_values, scratch_terms = scratch
    number_of_cells = length(column_counts)
    chunks = cld(number_of_cells, _ELLIPTIC_COLUMN_CHUNK)
    _parallel_for(chunks; work = _ELLIPTIC_COLUMN_CHUNK) do chunk
        offset = (chunk - 1) * width
        for cell in _chunk_columns(chunk, number_of_cells)
            stored = 0
            for incidence_index in incidence_offsets[cell]:(incidence_offsets[cell + 1] - 1)
                face_index = incidence_faces[incidence_index]
                area = T(faces[face_index].area)
                jump = incidence_signs[incidence_index] < 0 ? -area : area
                for slot in product_offsets[face_index]:(product_offsets[face_index + 1] - 1)
                    stored = _accumulate_column_term!(
                        scratch_rows, scratch_values, offset, stored,
                        product_rows[slot], product_values[slot] * jump,
                    )
                end
            end
            diagonal_present = false
            for slot in (offset + 1):(offset + stored)
                scratch_terms[slot] = true
                diagonal_present |= scratch_rows[slot] == cell
            end
            if !diagonal_present
                stored += 1
                scratch_rows[offset + stored] = cell
                scratch_values[offset + stored] = zero(T)
                scratch_terms[offset + stored] = false
            end
            if outputs === nothing
                column_counts[cell] = stored
            else
                column_pointers, rows, products, has_term = outputs
                _store_sorted_column!(
                    rows, products, column_pointers[cell],
                    scratch_rows, scratch_values, offset, stored,
                    has_term, scratch_terms,
                )
            end
        end
    end
    return nothing
end

@inline _chunk_columns(chunk, number_of_columns) =
    ((chunk - 1) * _ELLIPTIC_COLUMN_CHUNK + 1):min(number_of_columns, chunk * _ELLIPTIC_COLUMN_CHUNK)

"""Add `term` to row `row` of the column held in `rows`/`column_values` at
`offset .+ (1:stored)` (assign it to a new row) and return the new count."""
@inline function _accumulate_column_term!(rows, column_values, offset, stored, row, term)
    for slot in (offset + 1):(offset + stored)
        if rows[slot] == row
            column_values[slot] += term
            return stored
        end
    end
    rows[offset + stored + 1] = row
    column_values[offset + stored + 1] = term
    return stored + 1
end

"""Store the scratch column `offset .+ (1:stored)` sorted by row from
`start` on (insertion sort in place; columns are short), with an optional
flag per entry."""
function _store_sorted_column!(
        rows, column_values, start, scratch_rows, scratch_values, offset, stored,
        flags = nothing, scratch_flags = nothing,
    )
    for position in (offset + 2):(offset + stored)
        row = scratch_rows[position]
        value = scratch_values[position]
        flag = flags === nothing ? false : scratch_flags[position]
        target = position
        while target > offset + 1 && scratch_rows[target - 1] > row
            scratch_rows[target] = scratch_rows[target - 1]
            scratch_values[target] = scratch_values[target - 1]
            flags === nothing || (scratch_flags[target] = scratch_flags[target - 1])
            target -= 1
        end
        scratch_rows[target] = row
        scratch_values[target] = value
        flags === nothing || (scratch_flags[target] = flag)
    end
    for local_index in 1:stored
        rows[start + local_index - 1] = scratch_rows[offset + local_index]
        column_values[start + local_index - 1] = scratch_values[offset + local_index]
        flags === nothing ||
            (flags[start + local_index - 1] = scratch_flags[offset + local_index])
    end
    return nothing
end

# CSC columns are sorted by row. Return the stored slot even when its current
# numerical value is zero: these searches build fixed-topology refresh maps.
@inline function _csc_stored_pointer(matrix::SparseMatrixCSC, row::Int, column::Int)
    rows = rowvals(matrix)
    lo = matrix.colptr[column]
    hi = matrix.colptr[column + 1] - 1
    while lo <= hi
        middle = (lo + hi) >>> 1
        candidate = @inbounds rows[middle]
        if candidate < row
            lo = middle + 1
        elseif candidate > row
            hi = middle - 1
        else
            return middle
        end
    end
    error("sparse topology is missing a required entry")
end

"""Stored CSC slot of every diagonal entry of a fixed-pattern `matrix`.

An assembled elliptic matrix keeps its sparsity for the lifetime of the
operator, so the tree's diagonal is located once, here; coefficient refreshes
index these slots directly and their threaded bodies cannot throw.
"""
function _elliptic_diagonal_pointers(matrix::SparseMatrixCSC)
    pointers = Vector{Int}(undef, size(matrix, 1))
    for index in eachindex(pointers)
        pointers[index] = _csc_stored_pointer(matrix, index, index)
    end
    return pointers
end

struct StiffnessRefresh{T}
    stiffness_pointers::Vector{Int}
    left_faces::Vector{Int}
    right_faces::Vector{Int}
    stiffness_factors::Vector{T}
    transpose_pointers::Vector{Int}
    offsets::Vector{Int}
end

"""Reusable open-face connectivity and component reductions for an elliptic operator."""
mutable struct EllipticComponentWorkspace{T}
    incidence::SparseMatrixCSC{T, Int}
    counts::Vector{Int}
    labels::Vector{Int}
    stack::Vector{Int}
    active::BitVector
    open_faces::BitVector
    mesh_identity::Union{Nothing, Base.RefValue{Nothing}}
    generation::UInt
    anchors::Vector{Int}
    volumes::Vector{T}
    sums::Vector{T}
    ungauged::BitVector
    count::Int
    # Per cell: index of its ungauged component in `anchors`, or zero. The
    # mean removal sums per slot in fixed reduction chunks (`partials`).
    slots::Vector{Int}
    slot_sums::Vector{T}
    slot_scales::Vector{T}
    partials::Vector{T}
end

"""
    EllipticOperator(mesh; reaction=0, diffusion=1, face_coefficient=1,
                     velocity_boundary=nothing, boundary_face_coefficient=nothing,
                     assemble_matrix=true)

Construct the positive cell-centred operator

`L(u) = reaction*u - diffusion*div(face_coefficient*grad(u))`.

The matrix is stored in integrated finite-volume form,
`reaction*M + diffusion*S`, where `M` is the diagonal cell-volume matrix and
`S` is symmetric positive semidefinite.  Keeping the reaction and diffusion
parts separate makes the same object usable for periodic pressure Poisson
problems (`reaction=0`) and later implicit diffusion solves (`reaction>0`).

At a mixed-level interface the inverse face metric is non-diagonal.  Variable
coefficients therefore enter through the symmetric product
`sqrt(kappa)*K*sqrt(kappa)`; this preserves gradient/divergence duality and
reduces to ordinary face multiplication wherever `K` is diagonal.

`velocity_boundary` selects homogeneous Dirichlet data on `Outflow()` and
`ZeroGradientVelocity()` sides; all other physical sides are homogeneous
Neumann. A boundary coefficient may be supplied per physical face with
`boundary_face_coefficient`; otherwise scalars and functions are evaluated at
the boundary, and interior coefficient vectors are area-weighted from the
opposite normal faces. Supply `boundary_face_coefficient` if an open face has
no opposite interior face. The boundary distance is half the local normal cell
spacing. Open boundary conductances participate in the same matrix, factorized
action, and weighted gradient.

A zero-reaction operator is singular if at least one active connected component
has neither an open Dirichlet conductance nor a nonzero connection to an
imposed inactive cell. Gauges are applied only to those components.

Pass `assemble_matrix = false` for a factorized operator without an assembled
matrix; then `matrix` is `nothing`. The diagonal slots of an assembled matrix
are located once, at construction; a coefficient refresh reuses them so its
threaded body cannot throw.
"""
mutable struct EllipticOperator{T, M <: BlockTreeMesh, G <: GradientOperator{T}}
    mesh::M
    gradient_operator::G
    face_coefficient::Vector{T}
    square_root_face_coefficient::Vector{T}
    boundary_sides::NTuple{4, Bool}
    boundary_face_coefficient::Vector{T}
    boundary_aperture::Vector{T}
    boundary_gradient::Vector{T}
    boundary_diagonal::Vector{T}
    stiffness::SparseMatrixCSC{T, Int}
    matrix::Union{Nothing, SparseMatrixCSC{T, Int}}
    # Stored CSC slot of every diagonal entry, in cell order, or empty for a
    # matrix-free operator. The sparsity is fixed at assembly, so the search
    # runs once here rather than in every coefficient refresh.
    diagonal_pointers::Vector{Int}
    volumes::Vector{T}
    constraints::Vector{T}
    reaction::T
    diffusion::T
    references::Vector{CellRef}
    refresh::Union{Nothing, StiffnessRefresh{T}}
    inactive_condition::Symbol
    components::Union{Nothing, EllipticComponentWorkspace{T}}
    singular::Bool
end

function _impose_inactive_constraints!(
        matrix::SparseMatrixCSC{T, Int}, constraints::Vector{T},
    ) where {T}
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2)
        column_inactive = !iszero(@inbounds constraints[column])
        for pointer in nzrange(matrix, column)
            row = rows[pointer]
            row_inactive = !iszero(@inbounds constraints[row])
            if row_inactive || column_inactive
                @inbounds values[pointer] =
                    row == column && column_inactive ? one(T) : zero(T)
            end
        end
    end
    return matrix
end

# Stored entries of an inverse face metric, by face column: the left faces and
# values of column `right_face`, in stored order.
@inline _metric_column_entries(metric::DiagonalInverseFaceMetric, right_face) =
    right_face:right_face
@inline _metric_column_entries(metric::SparseInverseFaceMetric, right_face) =
    nzrange(metric.matrix, right_face)
@inline _metric_entry(metric::DiagonalInverseFaceMetric, entry) =
    (entry, @inbounds metric.diagonal[entry])
@inline _metric_entry(metric::SparseInverseFaceMetric, entry) =
    (@inbounds(rowvals(metric.matrix)[entry]), @inbounds(nonzeros(metric.matrix)[entry]))
@inline _metric_entry_count(metric::DiagonalInverseFaceMetric) = length(metric.diagonal)
@inline _metric_entry_count(metric::SparseInverseFaceMetric) = nnz(metric.matrix)

# The stored slot of `(row, column)`, or zero when the pattern lacks it.
@inline function _csc_stored_pointer_or_zero(rows, column_pointers, row::Int, column::Int)
    lo = @inbounds column_pointers[column]
    hi = (@inbounds column_pointers[column + 1]) - 1
    while lo <= hi
        middle = (lo + hi) >>> 1
        candidate = @inbounds rows[middle]
        if candidate < row
            lo = middle + 1
        elseif candidate > row
            hi = middle - 1
        else
            return middle
        end
    end
    return 0
end

@inline function _metric_entry_cells(faces, nx, ny, face_index)
    face = @inbounds faces[face_index]
    return _cell_index(face.minus, nx, ny), _cell_index(face.plus, nx, ny)
end

"""
Refresh map of the elliptic stiffness: for stored entry `p`, the contributions
`left_sign * area(left) * K[left, right] * right_sign * area(right)` of every
metric entry `K[left, right]` whose cells meet in `p`, in metric-entry order
(face columns, then stored order; per entry the minus/plus cells of the left
face, then of the right face). The four stiffness slots of each metric entry
are located in a threaded pass over face columns; a stable counting sort by
slot then places every contribution once.
"""
function _elliptic_coefficient_refresh_map(
        mesh::BlockTreeMesh,
        gradient_operator::GradientOperator{T},
        stiffness::SparseMatrixCSC{T, Int},
    ) where {T}
    stiffness_rows = rowvals(stiffness)
    column_pointers = stiffness.colptr
    metric = gradient_operator.inverse_metric
    faces = mesh.faces
    nx, ny = mesh.block_size
    number_of_contributions = 4 * _metric_entry_count(metric)
    contribution_pointers = Vector{Int}(undef, number_of_contributions)
    # A missing slot is stored as zero; the caller throws after the launch.
    _parallel_for(length(faces)) do right_face
        right_minus, right_plus = _metric_entry_cells(faces, nx, ny, right_face)
        for entry in _metric_column_entries(metric, right_face)
            left_face, _ = _metric_entry(metric, entry)
            left_minus, left_plus = _metric_entry_cells(faces, nx, ny, left_face)
            contribution = 4entry - 3
            for left_cell in (left_minus, left_plus), right_cell in (right_minus, right_plus)
                contribution_pointers[contribution] = _csc_stored_pointer_or_zero(
                    stiffness_rows, column_pointers, left_cell, right_cell,
                )
                contribution += 1
            end
        end
    end
    transpose_pointers = Vector{Int}(undef, nnz(stiffness))
    @_parallel for column in 1:size(stiffness, 2)
        for slot in column_pointers[column]:(column_pointers[column + 1] - 1)
            @inbounds transpose_pointers[slot] = _csc_stored_pointer_or_zero(
                stiffness_rows, column_pointers, column, stiffness_rows[slot],
            )
        end
    end
    (any(iszero, contribution_pointers) || any(iszero, transpose_pointers)) &&
        error("sparse topology is missing a required entry")
    # Stable counting by target preserves the original per-entry summation
    # order without sorting the potentially much larger contribution list.
    offsets, order = _grouped_output_schedule(contribution_pointers, nnz(stiffness))
    ordered_pointers = contribution_pointers[order]
    ordered_left = Vector{Int}(undef, number_of_contributions)
    ordered_right = Vector{Int}(undef, number_of_contributions)
    ordered_factors = Vector{T}(undef, number_of_contributions)
    # Every contribution of entry `k` lies in `4k - 3:4k`; recover its faces,
    # metric value and signs from that index.
    entry_faces = Vector{Int}(undef, _metric_entry_count(metric))
    _parallel_for(length(faces)) do right_face
        for entry in _metric_column_entries(metric, right_face)
            entry_faces[entry] = right_face
        end
    end
    @_parallel for target in 1:number_of_contributions
        contribution = @inbounds order[target]
        entry, combination = divrem(contribution - 1, 4)
        entry += 1
        right_face = @inbounds entry_faces[entry]
        left_face, metric_value = _metric_entry(metric, entry)
        left_sign = combination < 2 ? -one(T) : one(T)
        right_sign = iseven(combination) ? -one(T) : one(T)
        @inbounds ordered_left[target] = left_face
        @inbounds ordered_right[target] = right_face
        @inbounds ordered_factors[target] =
            left_sign * T(faces[left_face].area) * metric_value *
            right_sign * T(faces[right_face].area)
    end
    return StiffnessRefresh(
        ordered_pointers, ordered_left, ordered_right, ordered_factors,
        transpose_pointers, offsets,
    )
end

function EllipticOperator(
        mesh::M; reaction = 0, diffusion = 1,
        face_coefficient = 1,
        velocity_boundary = nothing, boundary_face_coefficient = nothing,
        boundary_aperture = nothing,
        T = eltype(mesh.origin),
        construction_cache = GradientConstructionCache(T),
        gradient_operator = nothing,
        volumes = nothing,
        active_cells = nothing,
        inactive_condition = :decoupled,
        assemble_matrix = true,
    ) where {M <: BlockTreeMesh}
    _check_real_type(T, "elliptic operators require Float32 or Float64")
    reaction_value = T(reaction)
    diffusion_value = T(diffusion)
    reaction_value >= zero(T) ||
        throw(ArgumentError("reaction must be nonnegative"))
    diffusion_value > zero(T) ||
        throw(ArgumentError("diffusion must be strictly positive"))
    inactive_condition in (:decoupled, :dirichlet) || throw(
        ArgumentError(
            "inactive_condition must be :decoupled or :dirichlet",
        ),
    )
    gradient_operator = isnothing(gradient_operator) ? GradientOperator(
            mesh, T; construction_cache,
        ) : gradient_operator
    gradient_operator isa GradientOperator{T} &&
        gradient_operator.mesh === mesh || throw(
        ArgumentError(
            "gradient_operator must use the operator mesh and scalar type"
        ),
    )
    coefficients = _face_coefficients(mesh, face_coefficient, T)
    volumes = isnothing(volumes) ? _cell_volumes(mesh, T) : T.(volumes)
    length(volumes) == ncells(mesh) || throw(
        DimensionMismatch("one volume is required per cell"),
    )
    all(isfinite, volumes) && all(>=(zero(T)), volumes) || throw(
        ArgumentError("cell volumes must be finite and nonnegative"),
    )
    active = isnothing(active_cells) ? trues(ncells(mesh)) :
        BitVector(active_cells)
    length(active) == ncells(mesh) || throw(
        DimensionMismatch("one active flag is required per cell"),
    )
    all(
        index -> active[index] == (volumes[index] > zero(T)),
        eachindex(active),
    ) || throw(
        ArgumentError(
            "active cells must have positive volume and covered cells zero volume"
        ),
    )
    constraints = T.(.!active)
    sides = _pressure_boundary_sides(velocity_boundary)
    boundary_material = isnothing(boundary_face_coefficient) ?
        _boundary_material(mesh, face_coefficient, sides, T) :
        T.(boundary_face_coefficient)
    apertures = isnothing(boundary_aperture) ?
        ones(T, length(mesh.boundary_faces)) : T.(boundary_aperture)
    length(boundary_material) == length(apertures) == length(mesh.boundary_faces) ||
        throw(DimensionMismatch("one coefficient and aperture are required per boundary face"))
    all(isfinite, boundary_material) && all(>=(zero(T)), boundary_material) ||
        throw(ArgumentError("boundary coefficients must be finite and nonnegative"))
    all(isfinite, apertures) && all(a -> zero(T) <= a <= one(T), apertures) ||
        throw(ArgumentError("boundary apertures must lie in [0, 1]"))
    boundary_gradient = zeros(T, length(apertures))
    boundary_diagonal = zeros(T, ncells(mesh))
    _elliptic_boundary_data!(
        boundary_diagonal, boundary_gradient, mesh, sides,
        boundary_material, apertures, constraints,
    )
    if assemble_matrix
        stiffness = _integrated_stiffness(mesh, gradient_operator, coefficients)
        diagonal_pointers = _elliptic_diagonal_pointers(stiffness)
        stiffness_values = nonzeros(stiffness)
        for cell in eachindex(boundary_diagonal)
            @inbounds stiffness_values[diagonal_pointers[cell]] +=
                boundary_diagonal[cell]
        end
        # The matrix is a value-scaled copy of the same sparsity, so it uses
        # the same diagonal slots.
        matrix = copy(stiffness)
        matrix_values = nonzeros(matrix)
        matrix_values .*= diffusion_value
        if !iszero(reaction_value)
            for column in axes(matrix, 2)
                @inbounds matrix_values[diagonal_pointers[column]] +=
                    reaction_value * volumes[column]
            end
        end
        refresh_map = _elliptic_coefficient_refresh_map(
            mesh, gradient_operator, stiffness,
        )
        _impose_inactive_constraints!(stiffness, constraints)
        _impose_inactive_constraints!(matrix, constraints)
    else
        # Only the factorized action is available on a matrix-free level.
        stiffness = spzeros(T, 0, 0)
        matrix = nothing
        diagonal_pointers = Int[]
        refresh_map = nothing
    end
    operator = EllipticOperator{T, M, typeof(gradient_operator)}(
        mesh, gradient_operator, coefficients, sqrt.(coefficients),
        sides, boundary_material, apertures, boundary_gradient, boundary_diagonal,
        stiffness, matrix, diagonal_pointers, volumes, constraints,
        reaction_value, diffusion_value, _cell_references(mesh),
        refresh_map, inactive_condition, nothing, false,
    )
    if iszero(reaction_value)
        operator.components = EllipticComponentWorkspace(operator)
        operator.singular = !isempty(operator.components.anchors)
    end
    return operator
end

"""Reassemble only the coefficient-dependent part of an elliptic operator.

The mesh, cell volumes, and compatible inverse face metric are retained. This
is the fixed-topology path used by an advected two-phase pressure coefficient;
rebuilding the vertex-patch metric and its geometric search every time step is
both unnecessary and substantially more expensive than the coefficient update.
"""
function _replace_face_coefficient(
        operator::EllipticOperator{T}, face_coefficient;
        boundary_face_coefficient = nothing,
    ) where {T}
    coefficients = _face_coefficients(
        operator.mesh, face_coefficient, T
    )
    isnothing(boundary_face_coefficient) || _update_elliptic_boundary_coefficient!(
        operator, boundary_face_coefficient,
    )
    return _update_elliptic_face_coefficient!(operator, coefficients)
end

"""Update boundary coefficients and nullspace metadata.

Follow with `_update_elliptic_face_coefficient!` to refresh assembled matrices,
then refresh any preconditioner hierarchy that uses this operator.
"""
function _update_elliptic_boundary_coefficient!(
        operator::EllipticOperator{T}, material,
    ) where {T}
    length(material) == length(operator.boundary_face_coefficient) ||
        throw(DimensionMismatch("one coefficient is required per boundary face"))
    all(isfinite, material) && all(>=(zero(T)), material) ||
        throw(ArgumentError("boundary coefficients must be finite and nonnegative"))
    # Boundary anchors may change independently of interior connectivity.
    copyto!(operator.boundary_face_coefficient, material)
    _elliptic_boundary_data!(
        operator.boundary_diagonal, operator.boundary_gradient,
        operator.mesh, operator.boundary_sides, operator.boundary_face_coefficient,
        operator.boundary_aperture, operator.constraints,
    )
    isnothing(operator.components) ||
        _refresh_elliptic_components!(operator.components, operator)
    return operator
end


function _update_elliptic_face_coefficient!(
        operator::EllipticOperator{T}, coefficients::Vector{T}
    ) where {T}
    @_parallel for index in 1:length(coefficients)
        @inbounds operator.face_coefficient[index] = coefficients[index]
        @inbounds operator.square_root_face_coefficient[index] =
            sqrt(coefficients[index])
    end
    isnothing(operator.components) ||
        _refresh_elliptic_components!(operator.components, operator)
    refresh = operator.refresh
    isnothing(refresh) && return operator
    stiffness_values = nonzeros(operator.stiffness)
    square_root = operator.square_root_face_coefficient
    offsets = refresh.offsets
    @_parallel for pointer in 1:length(stiffness_values)
        value = zero(T)
        for index in offsets[pointer]:(offsets[pointer + 1] - 1)
            left_face = @inbounds refresh.left_faces[index]
            right_face = @inbounds refresh.right_faces[index]
            factor = @inbounds refresh.stiffness_factors[index]
            @inbounds value +=
                square_root[left_face] * square_root[right_face] * factor
        end
        @inbounds stiffness_values[pointer] = value
    end
    stiffness_rows = rowvals(operator.stiffness)
    @_parallel for column in 1:size(operator.stiffness, 2)
        for pointer in nzrange(operator.stiffness, column)
            row = stiffness_rows[pointer]
            row < column || continue
            transpose_pointer = @inbounds refresh.transpose_pointers[pointer]
            average = (
                stiffness_values[pointer] +
                    stiffness_values[transpose_pointer]
            ) / 2
            @inbounds stiffness_values[pointer] = average
            @inbounds stiffness_values[transpose_pointer] = average
        end
    end
    stiffness_diagonal = operator.diagonal_pointers
    for cell in eachindex(operator.boundary_diagonal)
        @inbounds stiffness_values[stiffness_diagonal[cell]] +=
            operator.boundary_diagonal[cell]
    end
    _impose_inactive_constraints!(operator.stiffness, operator.constraints)

    matrix = operator.matrix::SparseMatrixCSC{T, Int}
    matrix_values = nonzeros(matrix)
    length(matrix_values) == length(stiffness_values) || error(
        "elliptic matrix and stiffness sparsity differ",
    )
    @_parallel for index in 1:length(matrix_values)
        @inbounds matrix_values[index] =
            operator.diffusion * stiffness_values[index]
    end
    if !iszero(operator.reaction)
        # The diagonal slots were located at assembly, so this body cannot
        # throw and needs no serial validation pass.
        diagonal_pointers = operator.diagonal_pointers
        volumes = operator.volumes
        reaction = operator.reaction
        @_parallel for index in eachindex(diagonal_pointers)
            pointer = @inbounds diagonal_pointers[index]
            @inbounds matrix_values[pointer] += reaction * volumes[index]
        end
    end
    _impose_inactive_constraints!(matrix, operator.constraints)
    return operator
end

"""Refresh a fixed-topology operator's active-cell Dirichlet mask.

Call `_update_elliptic_face_coefficient!` afterwards to reassemble the stored
matrix with the updated constraints, then refresh any dependent hierarchy.
"""
function _update_elliptic_activity!(
        operator::EllipticOperator{T}, active_cells, volumes,
    ) where {T}
    operator.inactive_condition === :dirichlet || throw(
        ArgumentError(
            "only a Dirichlet-inactive operator may change its active mask",
        )
    )
    length(active_cells) == length(operator.constraints) == length(volumes) ||
        throw(DimensionMismatch("elliptic activity arrays have the wrong size"))
    # Validate before writing: a rejected mask leaves the operator unchanged,
    # and the threaded update must not throw.
    for index in eachindex(operator.constraints)
        active_cells[index] == (T(volumes[index]) > zero(T)) || throw(
            ArgumentError(
                "active elliptic cells must have positive volume: cell " *
                    "$(operator.references[index]) is " *
                    "active=$(active_cells[index]) with " *
                    "volume=$(volumes[index])",
            )
        )
    end
    @_parallel for index in eachindex(operator.constraints)
        active = @inbounds active_cells[index]
        @inbounds begin
            operator.constraints[index] = active ? zero(T) : one(T)
            operator.volumes[index] = T(volumes[index])
        end
    end
    _elliptic_boundary_data!(
        operator.boundary_diagonal, operator.boundary_gradient,
        operator.mesh, operator.boundary_sides, operator.boundary_face_coefficient,
        operator.boundary_aperture, operator.constraints,
    )
    isnothing(operator.components) ||
        _refresh_elliptic_components!(operator.components, operator)
    return operator
end

"""
    PoissonOperator(mesh; face_coefficient=1, velocity_boundary=nothing,
                    boundary_face_coefficient=nothing, ...)

Construct the positive Poisson operator `-div(kappa*grad(u))`. Pass the same
`velocity_boundary` to this constructor and to a standalone `project!` call.
"""
PoissonOperator(
    mesh::BlockTreeMesh; face_coefficient = 1, velocity_boundary = nothing,
    boundary_face_coefficient = nothing,
    T = eltype(mesh.origin),
    construction_cache = GradientConstructionCache(T),
    assemble_matrix = true,
) =
    EllipticOperator(
    mesh; reaction = 0, diffusion = 1,
    face_coefficient, velocity_boundary, boundary_face_coefficient,
    construction_cache, assemble_matrix, T
)

"""Reusable vector scratch for applying an `EllipticOperator` to fields."""
struct EllipticWorkspace{T}
    input::Vector{T}
    output::Vector{T}
    face_input::Vector{T}
    face_output::Vector{T}
end

EllipticWorkspace(operator::EllipticOperator{T}) where {T} =
    EllipticWorkspace(
    zeros(T, length(operator.volumes)),
    zeros(T, length(operator.volumes)),
    zeros(T, length(operator.face_coefficient)),
    zeros(T, length(operator.face_coefficient))
)

"""Allocation-free factorized action of an [`EllipticOperator`](@ref).

The action keeps the AMR face Riesz map in its native form and evaluates

`reaction*M*x + diffusion*J'*sqrt(kappa)*K*sqrt(kappa)*J*x`.

It deliberately owns its two face work vectors: `GradientOperator` scratch is
used by field kernels and must not be aliased by a Krylov iteration.  One
instance may therefore be reused by a solver, but must not be called
concurrently.
"""
mutable struct FactorizedEllipticOperator{
        T, O <: EllipticOperator{T}, V <: Vector{T},
    }
    operator::O
    face_input::V
    face_output::V
    cell_face_offsets::Vector{Int}
    cell_faces::Vector{Int}
    cell_face_signs::Vector{Int8}
end

function _cell_face_incidence(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    counts = zeros(Int, ncells(mesh))
    for face in mesh.faces
        counts[_cell_index(face.minus, nx, ny)] += 1
        counts[_cell_index(face.plus, nx, ny)] += 1
    end
    offsets = _csr_offsets(counts)
    copyto!(counts, 1, offsets, 1, length(counts))
    faces = Vector{Int}(undef, offsets[end] - 1)
    signs = Vector{Int8}(undef, length(faces))
    for (face_index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        faces[counts[minus]] = face_index
        signs[counts[minus]] = -1
        counts[minus] += 1
        faces[counts[plus]] = face_index
        signs[counts[plus]] = 1
        counts[plus] += 1
    end
    return offsets, faces, signs
end

function FactorizedEllipticOperator(operator::O) where
    {T, O <: EllipticOperator{T}}
    number_of_faces = length(operator.face_coefficient)
    offsets, faces, signs = _cell_face_incidence(operator.mesh)
    return FactorizedEllipticOperator{
        T, O, Vector{T},
    }(
        operator,
        zeros(T, number_of_faces),
        zeros(T, number_of_faces),
        offsets, faces, signs,
    )
end

Base.size(action::FactorizedEllipticOperator) =
    (length(action.operator.volumes), length(action.operator.volumes))
Base.size(action::FactorizedEllipticOperator, dimension::Integer) =
    size(action)[dimension]
Base.eltype(::FactorizedEllipticOperator{T}) where {T} = T
issymmetric(::FactorizedEllipticOperator) = true

function _factorized_elliptic_mul!(
        destination::AbstractVector{T},
        operator::EllipticOperator{T},
        source::AbstractVector{T},
        face_input::AbstractVector{T},
        face_output::AbstractVector{T},
    ) where {T}
    number_of_cells = length(operator.volumes)
    number_of_faces = length(operator.face_coefficient)
    length(destination) == length(source) == number_of_cells ||
        throw(DimensionMismatch("elliptic vector sizes differ"))
    length(face_input) == length(face_output) == number_of_faces ||
        throw(DimensionMismatch("elliptic face-work sizes differ"))
    destination === source && throw(
        ArgumentError("factorized elliptic output must not alias its input"),
    )

    nx, ny = operator.mesh.block_size
    square_root = operator.square_root_face_coefficient
    faces = operator.mesh.faces
    @_parallel for face_index in 1:length(faces)
        face = faces[face_index]
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        minus_value = iszero(@inbounds operator.constraints[minus]) ?
            source[minus] : zero(T)
        plus_value = iszero(@inbounds operator.constraints[plus]) ?
            source[plus] : zero(T)
        @inbounds face_input[face_index] =
            square_root[face_index] * face.area *
            (plus_value - minus_value)
    end
    _mul_inverse_metric!(
        face_output, operator.gradient_operator.inverse_metric, face_input,
    )

    reaction = operator.reaction
    diffusion = operator.diffusion
    # Field applications do not own the Krylov action's cell-incidence cache.
    for index in eachindex(destination)
        @inbounds destination[index] = iszero(operator.constraints[index]) ?
            (
                reaction * operator.volumes[index] +
                diffusion * operator.boundary_diagonal[index]
            ) * source[index] :
            source[index]
    end
    for (face_index, face) in pairs(faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        @inbounds flux = diffusion * face.area * square_root[face_index] *
            face_output[face_index]
        iszero(@inbounds operator.constraints[minus]) &&
            (@inbounds destination[minus] -= flux)
        iszero(@inbounds operator.constraints[plus]) &&
            (@inbounds destination[plus] += flux)
    end
    return destination
end

function mul!(
        destination::AbstractVector{T},
        action::FactorizedEllipticOperator{T},
        source::AbstractVector{T},
    ) where {T}
    operator = action.operator
    number_of_cells = length(operator.volumes)
    number_of_faces = length(operator.face_coefficient)
    length(destination) == length(source) == number_of_cells ||
        throw(DimensionMismatch("elliptic vector sizes differ"))
    destination === source && throw(
        ArgumentError("factorized elliptic output must not alias its input"),
    )
    mesh = operator.mesh
    nx, ny = mesh.block_size
    square_root = operator.square_root_face_coefficient
    @_parallel for face_index in 1:number_of_faces
        face = mesh.faces[face_index]
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        minus_value = iszero(@inbounds operator.constraints[minus]) ?
            source[minus] : zero(T)
        plus_value = iszero(@inbounds operator.constraints[plus]) ?
            source[plus] : zero(T)
        @inbounds action.face_input[face_index] =
            square_root[face_index] * face.area *
            (plus_value - minus_value)
    end
    _mul_inverse_metric!(
        action.face_output, operator.gradient_operator.inverse_metric,
        action.face_input,
    )
    reaction = operator.reaction
    diffusion = operator.diffusion
    @_parallel for cell in 1:number_of_cells
        if !iszero(@inbounds operator.constraints[cell])
            @inbounds destination[cell] = source[cell]
            continue
        end
        value = @inbounds (
            reaction * operator.volumes[cell] +
                diffusion * operator.boundary_diagonal[cell]
        ) * source[cell]
        for pointer in action.cell_face_offsets[cell]:(
                action.cell_face_offsets[cell + 1] - 1
            )
            face_index = @inbounds action.cell_faces[pointer]
            face = mesh.faces[face_index]
            @inbounds value += action.cell_face_signs[pointer] * diffusion *
                face.area * square_root[face_index] *
                action.face_output[face_index]
        end
        @inbounds destination[cell] = value
    end
    return destination
end

function mul!(
        destination::AbstractVector{T},
        action::FactorizedEllipticOperator{T},
        source::AbstractVector{T}, alpha::Number, beta::Number,
    ) where {T}
    # Krylov.jl uses the three-argument method.  Keep the standard five-
    # argument contract for callers without allocating a third cell vector.
    iszero(beta) || throw(
        ArgumentError(
            "factorized elliptic five-argument mul! currently requires beta=0",
        ),
    )
    mul!(destination, action, source)
    alpha_value = T(alpha)
    if !isone(alpha_value)
        for index in eachindex(destination)
            @inbounds destination[index] *= alpha_value
        end
    end
    return destination
end

function _copy_from_field!(
        destination::AbstractVector, source::CellField,
        references::Vector{CellRef}
    )
    length(destination) == length(references) ||
        throw(DimensionMismatch("field and vector sizes differ"))
    for (index, reference) in pairs(references)
        @inbounds destination[index] = _get(source, reference)
    end
    return destination
end

function _copy_to_field!(
        destination::CellField, source::AbstractVector,
        references::Vector{CellRef}
    )
    length(source) == length(references) ||
        throw(DimensionMismatch("field and vector sizes differ"))
    for (index, reference) in pairs(references)
        _set!(destination, @inbounds(source[index]), reference)
    end
    return destination
end

"""
    apply!(destination, operator, source, workspace)

Apply the strong-form elliptic operator to cell values without allocating.
"""
function apply!(
        destination::CellField{T}, operator::EllipticOperator{T},
        source::CellField{T}, workspace::EllipticWorkspace{T}
    ) where {T}
    destination.mesh === source.mesh === operator.mesh ||
        throw(ArgumentError("fields and elliptic operator must share a mesh"))
    _copy_from_field!(workspace.input, source, operator.references)
    _factorized_elliptic_mul!(
        workspace.output, operator, workspace.input,
        workspace.face_input, workspace.face_output,
    )
    for index in eachindex(workspace.output)
        @inbounds workspace.output[index] =
            operator.volumes[index] > zero(T) ?
            workspace.output[index] / operator.volumes[index] :
            workspace.input[index]
    end
    _copy_to_field!(destination, workspace.output, operator.references)
    return destination
end

function _elliptic_boundary_gradient!(
        destination::FaceVelocity{T}, pressure::CellField{T},
        operator::EllipticOperator{T},
    ) where {T}
    for (index, face) in pairs(operator.mesh.boundary_faces)
        @inbounds destination.boundary[index] =
            -T(face.side) * operator.boundary_gradient[index] * _get(pressure, face.cell)
    end
    return destination
end

"""
    weighted_gradient!(destination, pressure, operator, workspace)

Apply the coefficient-weighted face gradient used inside `operator`.  Calling
this routine followed by `-divergence!` is algebraically identical to the
diffusion part of `apply!`, including the non-diagonal AMR interface metric.
"""
function weighted_gradient!(
        destination::FaceVelocity{T},
        pressure::CellField{T},
        operator::EllipticOperator{T},
        workspace::EllipticWorkspace{T}
    ) where {T}
    destination.mesh === pressure.mesh === operator.mesh ||
        throw(ArgumentError("fields and elliptic operator must share a mesh"))
    _pressure_jump!(operator.gradient_operator.pressure_jump, pressure)
    for index in eachindex(workspace.face_input)
        @inbounds workspace.face_input[index] =
            operator.square_root_face_coefficient[index] *
            operator.gradient_operator.pressure_jump[index]
    end
    _mul_inverse_metric!(
        workspace.face_output,
        operator.gradient_operator.inverse_metric,
        workspace.face_input
    )
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            operator.square_root_face_coefficient[index] * workspace.face_output[index]
    end
    _elliptic_boundary_gradient!(destination, pressure, operator)
    return destination
end

struct StaggeredEllipticComponent{Axis, T}
    matrix::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    local_to_global::Vector{Int}
end

"""
    StaggeredEllipticOperator(diffusion_operator;
                              reaction=1, diffusion=1)

Construct the integrated staggered elliptic problem

`reaction*u - diffusion*Δ_h*u`.

The operator uses dual volumes and one matrix per velocity component. For an
implicit midpoint viscous step, `diffusion = dt*viscosity/2` and `reaction=1`.
Nonhomogeneous wall data belongs on the right-hand side and is not embedded in
these homogeneous matrices.
"""
struct StaggeredEllipticOperator{T, M <: BlockTreeMesh, X, Y, D}
    mesh::M
    x::X
    y::Y
    diffusion_operator::D
    reaction::T
    diffusion::T
end

function _staggered_elliptic_component(
        component::StaggeredDiffusionComponent{Axis, T}, reaction,
        diffusion
    ) where {Axis, T}
    mass = spdiagm(0 => component.volumes)
    matrix = mass * (
        reaction *
            spdiagm(0 => ones(T, length(component.volumes))) -
            diffusion * component.laplacian
    )
    dropzeros!(matrix)
    return StaggeredEllipticComponent{Axis, T}(
        sparse(matrix), component.volumes, component.local_to_global
    )
end

"""Construct the symmetric conservative shifted matrix used for preconditioning."""
function _staggered_stable_matrix(
        component::StaggeredDiffusionComponent{Axis, T}, reaction,
        diffusion
    ) where {Axis, T}
    matrix = diffusion * component.stiffness +
        spdiagm(0 => reaction .* component.volumes)
    dropzeros!(matrix)
    return matrix
end

function StaggeredEllipticOperator(
        diffusion_operator::StaggeredDiffusionOperator{T};
        reaction = 1, diffusion = 1
    ) where {T}
    reaction_value = T(reaction)
    diffusion_value = T(diffusion)
    reaction_value > zero(T) ||
        throw(ArgumentError("staggered elliptic reaction must be positive"))
    diffusion_value >= zero(T) ||
        throw(ArgumentError("staggered elliptic diffusion must be nonnegative"))
    x = _staggered_elliptic_component(
        diffusion_operator.x, reaction_value, diffusion_value
    )
    y = _staggered_elliptic_component(
        diffusion_operator.y, reaction_value, diffusion_value
    )
    return StaggeredEllipticOperator{
        T, typeof(diffusion_operator.mesh), typeof(x), typeof(y),
        typeof(diffusion_operator),
    }(
        diffusion_operator.mesh, x, y, diffusion_operator,
        reaction_value, diffusion_value
    )
end
