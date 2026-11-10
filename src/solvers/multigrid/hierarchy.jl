"""Compute a Jacobi weight and replace positive diagonals by their reciprocals.

Multigrid smoothers reject negative diagonals and cap the weight at one.
Only singular levels allow zero diagonals for isolated nullspace cells. The
assembled smoothed transfer instead skips non-positive diagonals, uses an
uncapped weight, and returns zero for a zero spectral bound.
"""
function _jacobi_weight!(
        inverse_diagonal::AbstractVector{T}, row_sums;
        skip_nonpositive = false, allow_zero_diagonal = false,
        clamp_weight = true, zero_weight = one(T),
        error_message = "elliptic matrix has a non-positive diagonal",
    ) where {T}
    spectral_bound = zero(T)
    for index in eachindex(inverse_diagonal)
        diagonal = @inbounds inverse_diagonal[index]
        iszero(diagonal) && allow_zero_diagonal && continue
        if !(diagonal > zero(T))
            skip_nonpositive && continue
            throw(ArgumentError(error_message))
        end
        @inbounds spectral_bound = max(
            spectral_bound, row_sums[index] / diagonal,
        )
        @inbounds inverse_diagonal[index] = inv(diagonal)
    end
    iszero(spectral_bound) && return zero_weight
    weight = T(4) / (T(3) * spectral_bound)
    return clamp_weight ? min(one(T), weight) : weight
end

function _prolongate!(destination, prolongation::SparseMatrixCSC, source)
    mul!(destination, prolongation, source)
    return destination
end

function _restrict!(destination, prolongation::SparseMatrixCSC, source)
    mul!(destination, transpose(prolongation), source)
    return destination
end

mutable struct MultigridLevel{T, P, M, R}
    # Coarse levels use SparseMatrixCSC. The finest tree level may instead use
    # an exact factorized action so its global matrix is never materialised.
    matrix::M
    # CSC stores columns.  This row-form mirror lets output rows be owned by
    # parallel workers for general (including nonsymmetric) sparse operators.
    matrix_transpose::R
    volumes::Vector{T}
    prolongation::P
    inverse_diagonal::Vector{T}
    relaxation_weight::T
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    temporary::Vector{T}
    # Component indices are compact (zero denotes a constrained cell). The
    # component sums are scratch shared by the dual RHS and solution gauges.
    # Connectivity is reused until activity, face openness, or mesh topology changes.
    component_indices::Vector{Int}
    component_volumes::Vector{T}
    component_sums::Vector{T}
    component_open_faces::BitVector
    component_mesh_identity::Union{Nothing, Base.RefValue{Nothing}}
    component_generation::UInt
end

function _refresh_multigrid_components!(
        level::MultigridLevel{T}, operator::EllipticOperator{T},
    ) where {T}
    indices = level.component_indices
    volumes = level.component_volumes
    sums = level.component_sums
    open_faces = level.component_open_faces
    if !operator.singular
        empty!(indices)
        empty!(volumes)
        empty!(sums)
        empty!(open_faces)
        level.component_mesh_identity = nothing
        level.component_generation = zero(UInt)
        return level
    end

    mesh_identity = operator.mesh.topology_cache.identity
    generation = operator.mesh.topology_cache.generation
    changed = level.component_mesh_identity !== mesh_identity ||
        level.component_generation != generation ||
        length(indices) != length(operator.volumes) ||
        length(open_faces) != length(operator.face_coefficient)
    if !changed
        for cell in eachindex(indices)
            if @inbounds((indices[cell] > 0) != (operator.volumes[cell] > zero(T)))
                changed = true
                break
            end
        end
    end
    if !changed
        for face in eachindex(open_faces)
            if @inbounds(open_faces[face] != !iszero(operator.face_coefficient[face]))
                changed = true
                break
            end
        end
    end
    if changed
        labels = _elliptic_component_labels(operator)
        resize!(indices, length(labels))
        resize!(open_faces, length(operator.face_coefficient))
        for face in eachindex(open_faces)
            @inbounds open_faces[face] = !iszero(operator.face_coefficient[face])
        end
        empty!(volumes)
        root_indices = zeros(Int, length(labels))
        for cell in eachindex(labels)
            root = @inbounds labels[cell]
            if iszero(root)
                @inbounds indices[cell] = 0
                continue
            end
            component = @inbounds root_indices[root]
            if iszero(component)
                push!(volumes, zero(T))
                component = length(volumes)
                @inbounds root_indices[root] = component
            end
            @inbounds indices[cell] = component
        end
        resize!(sums, length(volumes))
        level.component_mesh_identity = mesh_identity
        level.component_generation = generation
    end
    fill!(volumes, zero(T))
    for cell in eachindex(indices)
        component = @inbounds indices[cell]
        iszero(component) && continue
        @inbounds volumes[component] += operator.volumes[cell]
    end
    return level
end

# These two projections are transposes: Q removes each component's
# volume-weighted solution mean and Q' projects an integrated RHS.  Keeping
# both sides of a coarse correction projected preserves symmetry on the
# compatible subspace even when coarsening merges or splits components.
function _multigrid_remove_component_means!(values, level::MultigridLevel)
    sums = level.component_sums
    fill!(sums, zero(eltype(sums)))
    for cell in eachindex(values)
        component = @inbounds level.component_indices[cell]
        iszero(component) && continue
        @inbounds sums[component] += level.volumes[cell] * values[cell]
    end
    for component in eachindex(sums)
        @inbounds sums[component] /= level.component_volumes[component]
    end
    for cell in eachindex(values)
        component = @inbounds level.component_indices[cell]
        iszero(component) && continue
        @inbounds values[cell] -= sums[component]
    end
    return values
end

function _multigrid_project_component_rhs!(values, level::MultigridLevel)
    sums = level.component_sums
    fill!(sums, zero(eltype(sums)))
    for cell in eachindex(values)
        component = @inbounds level.component_indices[cell]
        iszero(component) && continue
        @inbounds sums[component] += values[cell]
    end
    for component in eachindex(sums)
        @inbounds sums[component] /= level.component_volumes[component]
    end
    for cell in eachindex(values)
        component = @inbounds level.component_indices[cell]
        iszero(component) && continue
        @inbounds values[cell] -= level.volumes[cell] * sums[component]
    end
    return values
end

function _multigrid_levels(::Type{T}, matrix_free_fine::Bool) where {T}
    Sparse = SparseMatrixCSC{T, Int}
    return matrix_free_fine ?
        MultigridLevel{T, Sparse}[] :
        MultigridLevel{T, Sparse, Sparse, Sparse}[]
end

"""Unique cell couplings of a factorized compatible elliptic operator.

The contribution arrays retain the face-local factorization. `entry_pointers`
groups contributions which belong to the same `(row,column)` entry, permitting
exact Jacobi diagonals and row sums without constructing a CSC matrix.
"""
mutable struct FactorizedEllipticStencil{T}
    entry_rows::Vector{Int}
    entry_columns::Vector{Int}
    entry_pointers::Vector{Int}
    left_faces::Vector{Int}
    right_faces::Vector{Int}
    factors::Vector{T}
    values::Vector{T}
    inverse_diagonal::Vector{T}
    row_sums::Vector{T}
    relaxation_weight::T
end

struct FactorizedEllipticLevelAction{T, A}
    action::A
    stencil::FactorizedEllipticStencil{T}
end

Base.size(action::FactorizedEllipticLevelAction) = size(action.action)
Base.size(action::FactorizedEllipticLevelAction, dimension::Integer) =
    size(action.action, dimension)
Base.eltype(::FactorizedEllipticLevelAction{T}) where {T} = T
issymmetric(::FactorizedEllipticLevelAction) = true

mul!(destination, action::FactorizedEllipticLevelAction, source) =
    mul!(destination, action.action, source)

function _factorized_elliptic_stencil(operator::EllipticOperator{T}) where {T}
    mesh = operator.mesh
    nx, ny = mesh.block_size
    contributions = Tuple{Int, Int, Int, Int, T}[]
    has_diagonal = falses(ncells(mesh))
    function add_metric_entry(left_face, right_face, metric_value)
        left = mesh.faces[left_face]
        right = mesh.faces[right_face]
        left_cells = (
            (_cell_index(left.minus, nx, ny), -one(T)),
            (_cell_index(left.plus, nx, ny), one(T)),
        )
        right_cells = (
            (_cell_index(right.minus, nx, ny), -one(T)),
            (_cell_index(right.plus, nx, ny), one(T)),
        )
        for (row, left_sign) in left_cells, (column, right_sign) in right_cells
            row == column && (has_diagonal[row] = true)
            push!(
                contributions,
                (
                    row, column, left_face, right_face,
                    left_sign * T(left.area) * metric_value *
                        right_sign * T(right.area),
                ),
            )
        end
        return nothing
    end
    metric = operator.gradient_operator.inverse_metric
    if metric isa DiagonalInverseFaceMetric
        for face in eachindex(metric.diagonal)
            add_metric_entry(face, face, metric.diagonal[face])
        end
    else
        rows = rowvals(metric.matrix)
        values = nonzeros(metric.matrix)
        for right_face in axes(metric.matrix, 2), pointer in nzrange(metric.matrix, right_face)
            add_metric_entry(rows[pointer], right_face, values[pointer])
        end
    end
    # An entirely isolated or covered cell may have no metric contribution.
    for cell in eachindex(has_diagonal)
        has_diagonal[cell] ||
            push!(contributions, (cell, cell, 0, 0, zero(T)))
    end
    sort!(contributions; by = contribution -> (contribution[1], contribution[2]))
    entry_rows = Int[]
    entry_columns = Int[]
    entry_pointers = Int[1]
    left_faces = Int[]
    right_faces = Int[]
    factors = T[]
    previous = (0, 0)
    for contribution in contributions
        entry = (contribution[1], contribution[2])
        if entry != previous
            previous == (0, 0) || push!(entry_pointers, length(factors) + 1)
            push!(entry_rows, entry[1])
            push!(entry_columns, entry[2])
            previous = entry
        end
        push!(left_faces, contribution[3])
        push!(right_faces, contribution[4])
        push!(factors, contribution[5])
    end
    push!(entry_pointers, length(factors) + 1)
    number_of_cells = length(operator.volumes)
    stencil = FactorizedEllipticStencil(
        entry_rows, entry_columns, entry_pointers,
        left_faces, right_faces, factors,
        zeros(T, length(entry_rows)), zeros(T, number_of_cells),
        zeros(T, number_of_cells), one(T),
    )
    return _refresh_factorized_elliptic_stencil!(stencil, operator)
end

function _refresh_factorized_elliptic_stencil!(
        stencil::FactorizedEllipticStencil{T},
        operator::EllipticOperator{T},
    ) where {T}
    square_root = operator.square_root_face_coefficient
    fill!(stencil.inverse_diagonal, zero(T))
    fill!(stencil.row_sums, zero(T))
    for entry in eachindex(stencil.entry_rows)
        row = @inbounds stencil.entry_rows[entry]
        column = @inbounds stencil.entry_columns[entry]
        row_constrained = !iszero(@inbounds operator.constraints[row])
        column_constrained = !iszero(@inbounds operator.constraints[column])
        value = zero(T)
        if row_constrained || column_constrained
            row == column && row_constrained && (value = one(T))
        else
            for pointer in stencil.entry_pointers[entry]:(stencil.entry_pointers[entry + 1] - 1)
                left_face = @inbounds stencil.left_faces[pointer]
                iszero(left_face) && continue
                @inbounds value += square_root[left_face] *
                    square_root[stencil.right_faces[pointer]] *
                    stencil.factors[pointer]
            end
            value *= operator.diffusion
            row == column && (value += operator.reaction * operator.volumes[row])
        end
        row == column && (@inbounds stencil.inverse_diagonal[row] = value)
        @inbounds stencil.values[entry] = value
        @inbounds stencil.row_sums[row] += abs(value)
    end
    stencil.relaxation_weight = _jacobi_weight!(
        stencil.inverse_diagonal, stencil.row_sums;
        allow_zero_diagonal = operator.singular,
        error_message = "factorized elliptic operator has a non-positive diagonal",
    )
    return stencil
end

function _factorized_elliptic_level(operator, prolongation, stencil)
    T = eltype(operator.volumes)
    action = FactorizedEllipticLevelAction(
        FactorizedEllipticOperator(operator), stencil,
    )
    number_of_cells = length(operator.volumes)
    return MultigridLevel(
        action, nothing, operator.volumes, prolongation,
        copy(stencil.inverse_diagonal), stencil.relaxation_weight,
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        Int[], T[], T[], falses(0), nothing, zero(UInt),
    )
end

"""Fixed sparsity for one Jacobi-smoothed geometric transfer.

`stiffness_to_transfer[p]` identifies the interpolation entry receiving the
contribution of stiffness entry `p`.  Only the values are refreshed when the
two-phase coefficient changes; all graph searches belong to topology setup.
"""
struct SmoothedTransferRefresh{T}
    prolongation::SparseMatrixCSC{T, Int}
    tentative_pointers::Vector{Int}
    stiffness_to_transfer::Vector{Int}
    transfer_offsets::Vector{Int}
    stiffness_indices::Vector{Int}
    tentative_entries::BitVector
    inverse_diagonal::Vector{T}
    row_sums::Vector{T}
end

struct FactorizedSmoothedTransferRefresh{T}
    prolongation::SparseMatrixCSC{T, Int}
    tentative_pointers::Vector{Int}
    entry_to_transfer::Vector{Int}
    transfer_offsets::Vector{Int}
    entry_indices::Vector{Int}
    tentative_entries::BitVector
    stencil::FactorizedEllipticStencil{T}
end

"""Restrict a cell transfer to the active fluid spaces.

Embedded geometry is rediscretized independently on every tree level, so the
sets of active background cells are not nested.  The pressure null vector on a
cut-cell level is one in active cells and zero in covered cells.  A transfer
which instead preserves the constant background vector sends coarse pressure
corrections into constrained solid cells and ceases to represent this null
vector.  The error is particularly damaging for large density ratios.

Remove couplings to either covered space and normalize every active fine row,
thereby enforcing `P * coarse_active == fine_active`. Activity may be supplied
as Boolean masks or cell volumes (zero denotes a covered cell). Zeros are
retained in the sparse structure because coefficient refreshes reuse pointer
schedules.
"""
function _restrict_transfer_to_active_domain!(
        prolongation::SparseMatrixCSC{T, Int},
        fine_active::AbstractVector,
        coarse_active::AbstractVector,
        row_sums::AbstractVector{T},
    ) where {T}
    size(prolongation) == (length(fine_active), length(coarse_active)) ||
        throw(
        DimensionMismatch(
            "geometry activity masks do not match pressure transfer",
        )
    )
    length(row_sums) == length(fine_active) || throw(
        DimensionMismatch(
            "geometry-aware transfer scratch has the wrong length",
        )
    )
    rows = rowvals(prolongation)
    values = nonzeros(prolongation)
    fill!(row_sums, zero(T))
    for column in axes(prolongation, 2)
        column_active = !iszero(@inbounds coarse_active[column])
        for pointer in nzrange(prolongation, column)
            row = @inbounds rows[pointer]
            row_active = !iszero(@inbounds fine_active[row])
            if column_active && row_active
                @inbounds row_sums[row] += values[pointer]
            else
                @inbounds values[pointer] = zero(T)
            end
        end
    end
    for row in eachindex(fine_active)
        row_active = !iszero(@inbounds fine_active[row])
        row_active || continue
        row_sum = @inbounds row_sums[row]
        isfinite(row_sum) && !iszero(row_sum) || error(
            "geometry-aware pressure transfer leaves an active cell unowned",
        )
    end
    for column in axes(prolongation, 2)
        column_active = !iszero(@inbounds coarse_active[column])
        column_active || continue
        for pointer in nzrange(prolongation, column)
            row = @inbounds rows[pointer]
            row_active = !iszero(@inbounds fine_active[row])
            row_active || continue
            @inbounds values[pointer] /= row_sums[row]
        end
    end
    return prolongation
end

# Stable grouping keeps each transfer's accumulation order identical to
# sortperm while avoiding a comparison sort over every fine-grid coupling.
function _transfer_output_schedule(targets::Vector{Int}, outputs::Int)
    offsets = zeros(Int, outputs + 1)
    for target in targets
        @inbounds offsets[target + 1] += 1
    end
    for target in 1:outputs
        @inbounds offsets[target + 1] += offsets[target]
    end
    offsets .+= 1
    next = copy(offsets)
    indices = Vector{Int}(undef, length(targets))
    for index in eachindex(targets)
        target = @inbounds targets[index]
        pointer = @inbounds next[target]
        @inbounds begin
            indices[pointer] = index
            next[target] += 1
        end
    end
    return offsets, indices
end
function _pressure_transfer_owners(tentative::SparseMatrixCSC)
    owners = zeros(Int, size(tentative, 1))
    tentative_rows = rowvals(tentative)
    for column in axes(tentative, 2), pointer in nzrange(tentative, column)
        row = tentative_rows[pointer]
        iszero(owners[row]) || throw(
            ArgumentError("tentative pressure transfer must have one owner"),
        )
        owners[row] = column
    end
    all(!iszero, owners) || throw(
        ArgumentError("tentative pressure transfer leaves an unknown unowned"),
    )
    return owners
end

function _smoothed_transfer_refresh(
        stiffness::SparseMatrixCSC{T, Int},
        tentative::SparseMatrixCSC{T, Int},
    ) where {T}
    number_of_fine, number_of_coarse = size(tentative)
    owners = _pressure_transfer_owners(tentative)

    rows = Int[]
    columns = Int[]
    sizehint!(rows, nnz(stiffness) + number_of_fine)
    sizehint!(columns, nnz(stiffness) + number_of_fine)
    for row in 1:number_of_fine
        push!(rows, row)
        push!(columns, owners[row])
    end
    stiffness_rows = rowvals(stiffness)
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        push!(rows, stiffness_rows[pointer])
        push!(columns, owners[column])
    end
    prolongation = sparse(
        rows, columns, ones(T, length(rows)),
        number_of_fine, number_of_coarse,
    )
    fill!(nonzeros(prolongation), zero(T))
    tentative_pointers = [
        _csc_stored_pointer(prolongation, row, owners[row])
            for row in 1:number_of_fine
    ]
    stiffness_to_transfer = Vector{Int}(undef, nnz(stiffness))
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        @inbounds stiffness_to_transfer[pointer] =
            _csc_stored_pointer(prolongation, stiffness_rows[pointer], owners[column])
    end
    transfer_offsets, stiffness_indices = _transfer_output_schedule(
        stiffness_to_transfer, nnz(prolongation),
    )
    tentative_entries = falses(nnz(prolongation))
    tentative_entries[tentative_pointers] .= true
    refresh = SmoothedTransferRefresh(
        prolongation, tentative_pointers, stiffness_to_transfer,
        transfer_offsets, stiffness_indices, tentative_entries,
        zeros(T, number_of_fine), zeros(T, number_of_fine),
    )
    _refresh_smoothed_transfer!(refresh, stiffness)
    return refresh
end

function _refresh_smoothed_transfer!(
        refresh::SmoothedTransferRefresh{T},
        stiffness::SparseMatrixCSC{T, Int},
    ) where {T}
    length(refresh.stiffness_to_transfer) == nnz(stiffness) || throw(
        ArgumentError("pressure stiffness sparsity changed on fixed topology"),
    )
    inverse_diagonal = refresh.inverse_diagonal
    row_sums = refresh.row_sums
    rows = rowvals(stiffness)
    stiffness_values = nonzeros(stiffness)
    @_parallel for column in 1:size(stiffness, 2)
        row_sum = zero(T)
        diagonal = zero(T)
        for pointer in nzrange(stiffness, column)
            value = @inbounds stiffness_values[pointer]
            row_sum += abs(value)
            @inbounds rows[pointer] == column && (diagonal = value)
        end
        @inbounds row_sums[column] = row_sum
        @inbounds inverse_diagonal[column] = diagonal
    end
    weight = _jacobi_weight!(
        inverse_diagonal, row_sums;
        skip_nonpositive = true, clamp_weight = false, zero_weight = zero(T),
    )

    prolongation_values = nonzeros(refresh.prolongation)
    @_parallel for transfer_pointer in 1:length(prolongation_values)
        value = refresh.tentative_entries[transfer_pointer] ? one(T) : zero(T)
        for schedule_index in refresh.transfer_offsets[transfer_pointer]:(
                refresh.transfer_offsets[transfer_pointer + 1] - 1
            )
            stiffness_pointer =
                @inbounds refresh.stiffness_indices[schedule_index]
            row = @inbounds rows[stiffness_pointer]
            @inbounds value -= weight * inverse_diagonal[row] *
                stiffness_values[stiffness_pointer]
        end
        @inbounds prolongation_values[transfer_pointer] = value
    end
    return refresh.prolongation
end

function _factorized_smoothed_transfer_refresh(
        operator::EllipticOperator{T},
        tentative::SparseMatrixCSC{T, Int},
        stencil::FactorizedEllipticStencil{T},
    ) where {T}
    number_of_fine, number_of_coarse = size(tentative)
    owners = _pressure_transfer_owners(tentative)
    rows = collect(1:number_of_fine)
    columns = copy(owners)
    for entry in eachindex(stencil.entry_rows)
        push!(rows, stencil.entry_rows[entry])
        push!(columns, owners[stencil.entry_columns[entry]])
    end
    prolongation = sparse(
        rows, columns, ones(T, length(rows)), number_of_fine, number_of_coarse,
    )
    fill!(nonzeros(prolongation), zero(T))
    tentative_pointers = [
        _csc_stored_pointer(prolongation, row, owners[row])
            for row in 1:number_of_fine
    ]
    entry_to_transfer = [
        _csc_stored_pointer(
            prolongation, stencil.entry_rows[entry],
            owners[stencil.entry_columns[entry]],
        ) for entry in eachindex(stencil.entry_rows)
    ]
    transfer_offsets, entry_indices = _transfer_output_schedule(
        entry_to_transfer, nnz(prolongation),
    )
    tentative_entries = falses(nnz(prolongation))
    tentative_entries[tentative_pointers] .= true
    refresh = FactorizedSmoothedTransferRefresh(
        prolongation, tentative_pointers, entry_to_transfer,
        transfer_offsets, entry_indices, tentative_entries, stencil,
    )
    _refresh_smoothed_transfer!(refresh, operator)
    return refresh
end

function _refresh_smoothed_transfer!(
        refresh::FactorizedSmoothedTransferRefresh{T},
        operator::EllipticOperator{T},
    ) where {T}
    stencil = _refresh_factorized_elliptic_stencil!(refresh.stencil, operator)
    prolongation_values = nonzeros(refresh.prolongation)
    weight = stencil.relaxation_weight
    inverse_diagonal = stencil.inverse_diagonal
    @_parallel for pointer in 1:length(prolongation_values)
        value = refresh.tentative_entries[pointer] ? one(T) : zero(T)
        for schedule_index in refresh.transfer_offsets[pointer]:(
                refresh.transfer_offsets[pointer + 1] - 1
            )
            entry = @inbounds refresh.entry_indices[schedule_index]
            row = @inbounds stencil.entry_rows[entry]
            @inbounds value -= weight * inverse_diagonal[row] *
                stencil.values[entry]
        end
        @inbounds prolongation_values[pointer] = value
    end
    return refresh.prolongation
end

function _multigrid_level(
        matrix::SparseMatrixCSC{T, Int},
        volumes::Vector{T},
        prolongation; singular = false,
    ) where {T}
    matrix_transpose = copy(transpose(matrix))
    inverse_diagonal = collect(diag(matrix))
    row_sums = zeros(T, size(matrix, 1))
    column_pointers = matrix_transpose.colptr
    matrix_values = nonzeros(matrix_transpose)
    @_parallel for column in 1:size(matrix_transpose, 2)
        value = zero(eltype(row_sums))
        for pointer in column_pointers[column]:(column_pointers[column + 1] - 1)
            @inbounds value += abs(matrix_values[pointer])
        end
        @inbounds row_sums[column] = value
    end
    relaxation_weight = _jacobi_weight!(
        inverse_diagonal, row_sums; allow_zero_diagonal = singular,
    )
    number_of_cells = size(matrix, 1)
    return MultigridLevel(
        matrix, matrix_transpose, volumes, prolongation,
        inverse_diagonal, relaxation_weight,
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        Int[], T[], T[], falses(0), nothing, zero(UInt),
    )
end
