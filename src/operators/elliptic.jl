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

@inline function _elliptic_cell_index(ref::CellRef, nx::Int, ny::Int)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
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

function _weighted_inverse_metric(
        metric::DiagonalInverseFaceMetric{T},
        coefficient::Vector{T}
    ) where {T}
    return spdiagm(0 => coefficient .* metric.diagonal)
end

function _weighted_inverse_metric(
        metric::SparseInverseFaceMetric{T},
        coefficient::Vector{T}
    ) where {T}
    square_root = spdiagm(0 => sqrt.(coefficient))
    return square_root * metric.matrix * square_root
end

function _integrated_stiffness(
        mesh::BlockTreeMesh,
        gradient_operator::GradientOperator{T},
        coefficient::Vector{T}
    ) where {T}
    nx, ny = mesh.block_size
    number_of_faces = length(mesh.faces)
    rows = Vector{Int}(undef, 2number_of_faces)
    columns = Vector{Int}(undef, 2number_of_faces)
    values = Vector{T}(undef, 2number_of_faces)
    for (face_index, face) in pairs(mesh.faces)
        entry = 2face_index - 1
        rows[entry] = face_index
        columns[entry] = _elliptic_cell_index(face.minus, nx, ny)
        values[entry] = -T(face.area)
        rows[entry + 1] = face_index
        columns[entry + 1] = _elliptic_cell_index(face.plus, nx, ny)
        values[entry + 1] = T(face.area)
    end
    jump = sparse(rows, columns, values, number_of_faces, ncells(mesh))
    face_metric = _weighted_inverse_metric(
        gradient_operator.inverse_metric, coefficient
    )
    stiffness = transpose(jump) * face_metric * jump
    # Sparse products can accumulate the two triangles in a different order.
    # Explicit symmetrisation makes the algebraic contract exact to roundoff.
    stiffness = (stiffness + transpose(stiffness)) / 2
    dropzeros!(stiffness)
    return stiffness
end

function _complete_elliptic_stiffness_pattern(
        mesh::BlockTreeMesh,
        gradient_operator::GradientOperator{T},
        stiffness::SparseMatrixCSC{T, Int},
    ) where {T}
    nx, ny = mesh.block_size
    rows = Int[]
    columns = Int[]
    append!(rows, 1:ncells(mesh))
    append!(columns, 1:ncells(mesh))
    function add_face_pair(left_face, right_face)
        left = mesh.faces[left_face]
        right = mesh.faces[right_face]
        left_cells = (
            _elliptic_cell_index(left.minus, nx, ny),
            _elliptic_cell_index(left.plus, nx, ny),
        )
        right_cells = (
            _elliptic_cell_index(right.minus, nx, ny),
            _elliptic_cell_index(right.plus, nx, ny),
        )
        for left_cell in left_cells, right_cell in right_cells
            push!(rows, left_cell)
            push!(columns, right_cell)
        end
        return nothing
    end
    metric = gradient_operator.inverse_metric
    if metric isa DiagonalInverseFaceMetric
        for face in eachindex(metric.diagonal)
            add_face_pair(face, face)
        end
    else
        metric_rows = rowvals(metric.matrix)
        for right_face in axes(metric.matrix, 2),
                pointer in nzrange(metric.matrix, right_face)
            add_face_pair(metric_rows[pointer], right_face)
        end
    end
    completed = sparse(
        rows, columns, ones(T, length(rows)), ncells(mesh), ncells(mesh),
    )
    fill!(nonzeros(completed), zero(T))
    lookup = Dict{Tuple{Int, Int}, Int}()
    completed_rows = rowvals(completed)
    for column in axes(completed, 2), pointer in nzrange(completed, column)
        lookup[(completed_rows[pointer], column)] = pointer
    end
    stiffness_rows = rowvals(stiffness)
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        @inbounds nonzeros(completed)[
            lookup[(stiffness_rows[pointer], column)],
        ] = nonzeros(stiffness)[pointer]
    end
    return completed
end

"""
    EllipticOperator(mesh; reaction=0, diffusion=1, face_coefficient=1)

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
"""
struct EllipticOperator{T, M <: BlockTreeMesh, G <: GradientOperator{T}}
    mesh::M
    gradient_operator::G
    face_coefficient::Vector{T}
    square_root_face_coefficient::Vector{T}
    stiffness::SparseMatrixCSC{T, Int}
    matrix::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    constraints::Vector{T}
    reaction::T
    diffusion::T
    references::Vector{CellRef}
    stiffness_pointers::Vector{Int}
    left_faces::Vector{Int}
    right_faces::Vector{Int}
    stiffness_factors::Vector{T}
    transpose_pointers::Vector{Int}
    stiffness_refresh_offsets::Vector{Int}
    inactive_condition::Symbol
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

function _elliptic_coefficient_refresh_map(
        mesh::BlockTreeMesh,
        gradient_operator::GradientOperator{T},
        stiffness::SparseMatrixCSC{T, Int},
    ) where {T}
    target_lookup = Dict{Tuple{Int, Int}, Int}()
    stiffness_rows = rowvals(stiffness)
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        target_lookup[(stiffness_rows[pointer], column)] = pointer
    end
    pointers = Int[]
    left_faces = Int[]
    right_faces = Int[]
    factors = T[]
    nx, ny = mesh.block_size
    function add_metric_entry(left_face, right_face, metric_value)
        left = mesh.faces[left_face]
        right = mesh.faces[right_face]
        left_cells = (
            (_elliptic_cell_index(left.minus, nx, ny), -one(T)),
            (_elliptic_cell_index(left.plus, nx, ny), one(T)),
        )
        right_cells = (
            (_elliptic_cell_index(right.minus, nx, ny), -one(T)),
            (_elliptic_cell_index(right.plus, nx, ny), one(T)),
        )
        for (left_cell, left_sign) in left_cells
            for (right_cell, right_sign) in right_cells
                push!(pointers, target_lookup[(left_cell, right_cell)])
                push!(left_faces, left_face)
                push!(right_faces, right_face)
                push!(
                    factors,
                    left_sign * T(left.area) * metric_value *
                        right_sign * T(right.area),
                )
            end
        end
        return nothing
    end
    metric = gradient_operator.inverse_metric
    if metric isa DiagonalInverseFaceMetric
        for face in eachindex(metric.diagonal)
            add_metric_entry(face, face, metric.diagonal[face])
        end
    else
        metric_rows = rowvals(metric.matrix)
        metric_values = nonzeros(metric.matrix)
        for right_face in axes(metric.matrix, 2),
                pointer in nzrange(metric.matrix, right_face)
            add_metric_entry(
                metric_rows[pointer], right_face, metric_values[pointer],
            )
        end
    end
    transpose_pointers = Vector{Int}(undef, nnz(stiffness))
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        @inbounds transpose_pointers[pointer] =
            target_lookup[(column, stiffness_rows[pointer])]
    end
    permutation = sortperm(pointers)
    pointers[:] = pointers[permutation]
    left_faces[:] = left_faces[permutation]
    right_faces[:] = right_faces[permutation]
    factors[:] = factors[permutation]
    offsets = Vector{Int}(undef, nnz(stiffness) + 1)
    contribution = 1
    for pointer in 1:nnz(stiffness)
        offsets[pointer] = contribution
        while contribution <= length(pointers) &&
                pointers[contribution] == pointer
            contribution += 1
        end
    end
    offsets[end] = contribution
    contribution == length(pointers) + 1 || error(
        "elliptic coefficient refresh contains an invalid sparse pointer",
    )
    return pointers, left_faces, right_faces, factors,
        transpose_pointers, offsets
end

function EllipticOperator(
        mesh::M; reaction = 0, diffusion = 1,
        face_coefficient = 1,
        T = eltype(mesh.origin),
        construction_cache = GradientConstructionCache(T),
        gradient_operator = nothing,
        volumes = nothing,
        active_cells = nothing,
        inactive_condition = :decoupled,
        assemble_matrix = true,
    ) where {M <: BlockTreeMesh}
    T <: Union{Float32, Float64} ||
        throw(ArgumentError("elliptic operators require Float32 or Float64"))
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
    if assemble_matrix
        stiffness = _complete_elliptic_stiffness_pattern(
            mesh, gradient_operator,
            _integrated_stiffness(mesh, gradient_operator, coefficients),
        )
        matrix = copy(stiffness)
        nonzeros(matrix) .*= diffusion_value
        if !iszero(reaction_value)
            matrix_rows = rowvals(matrix)
            for column in axes(matrix, 2)
                for pointer in nzrange(matrix, column)
                    if matrix_rows[pointer] == column
                        @inbounds nonzeros(matrix)[pointer] +=
                            reaction_value * volumes[column]
                        break
                    end
                end
            end
        end
        refresh_map = _elliptic_coefficient_refresh_map(
            mesh, gradient_operator, stiffness,
        )
        _impose_inactive_constraints!(stiffness, constraints)
        _impose_inactive_constraints!(matrix, constraints)
    else
        # A zero-dimensional sentinel catches accidental matrix use. The exact
        # Krylov and finest multigrid actions use the compatible factorization.
        stiffness = spzeros(T, 0, 0)
        matrix = spzeros(T, 0, 0)
        refresh_map = (Int[], Int[], Int[], T[], Int[], Int[1])
    end
    singular = iszero(reaction_value) && !(
        inactive_condition === :dirichlet && any(!iszero, constraints)
    )
    return EllipticOperator{T, M, typeof(gradient_operator)}(
        mesh, gradient_operator, coefficients, sqrt.(coefficients),
        stiffness, matrix, volumes, constraints,
        reaction_value, diffusion_value, _cell_references(mesh),
        refresh_map..., inactive_condition, singular,
    )
end

"""Reassemble only the coefficient-dependent part of an elliptic operator.

The mesh, cell volumes, and compatible inverse face metric are retained. This
is the fixed-topology path used by an advected two-phase pressure coefficient;
rebuilding the vertex-patch metric and its geometric search every time step is
both unnecessary and substantially more expensive than the coefficient update.
"""
function _replace_face_coefficient(
        operator::EllipticOperator{T}, face_coefficient
    ) where {T}
    coefficients = _face_coefficients(
        operator.mesh, face_coefficient, T
    )
    return _update_elliptic_face_coefficient!(operator, coefficients)
end

@inline function _add_existing_sparse!(
        matrix::SparseMatrixCSC, row::Int, column::Int, value
    )
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for index in nzrange(matrix, column)
        if rows[index] == row
            @inbounds values[index] += value
            return matrix
        end
    end
    error("coefficient refresh changed elliptic sparsity")
end


function _update_elliptic_face_coefficient!(
        operator::EllipticOperator{T}, coefficients::Vector{T}
    ) where {T}
    @_parallel for index in 1:length(coefficients)
        @inbounds operator.face_coefficient[index] = coefficients[index]
        @inbounds operator.square_root_face_coefficient[index] =
            sqrt(coefficients[index])
    end
    isempty(operator.stiffness_pointers) && return operator
    stiffness_values = nonzeros(operator.stiffness)
    square_root = operator.square_root_face_coefficient
    offsets = operator.stiffness_refresh_offsets
    @_parallel for pointer in 1:length(stiffness_values)
        value = zero(T)
        for index in offsets[pointer]:(offsets[pointer + 1] - 1)
            left_face = @inbounds operator.left_faces[index]
            right_face = @inbounds operator.right_faces[index]
            factor = @inbounds operator.stiffness_factors[index]
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
            transpose_pointer = @inbounds operator.transpose_pointers[pointer]
            average = (
                stiffness_values[pointer] +
                    stiffness_values[transpose_pointer]
            ) / 2
            @inbounds stiffness_values[pointer] = average
            @inbounds stiffness_values[transpose_pointer] = average
        end
    end
    _impose_inactive_constraints!(operator.stiffness, operator.constraints)

    matrix_values = nonzeros(operator.matrix)
    length(matrix_values) == length(stiffness_values) || error(
        "elliptic matrix and stiffness sparsity differ",
    )
    @_parallel for index in 1:length(matrix_values)
        @inbounds matrix_values[index] =
            operator.diffusion * stiffness_values[index]
    end
    if !iszero(operator.reaction)
        @_parallel for index in 1:length(operator.volumes)
            _add_existing_sparse!(
                operator.matrix, index, index,
                operator.reaction * operator.volumes[index]
            )
        end
    end
    _impose_inactive_constraints!(operator.matrix, operator.constraints)
    return operator
end

"""Refresh a fixed-topology operator's active-cell Dirichlet mask."""
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
    has_inactive = any(!, active_cells)
    has_inactive == !operator.singular || throw(
        ArgumentError(
            "active-mask nullspace changed; rebuild the elliptic operator",
        )
    )
    @_parallel for index in eachindex(operator.constraints)
        active = @inbounds active_cells[index]
        volume = T(@inbounds volumes[index])
        active == (volume > zero(T)) || throw(
            ArgumentError(
                "active elliptic cells must have positive volume",
            )
        )
        @inbounds begin
            operator.constraints[index] = active ? zero(T) : one(T)
            operator.volumes[index] = volume
        end
    end
    return operator
end

"""Construct the periodic positive Poisson operator `-div(kappa*grad(u))`."""
PoissonOperator(
    mesh::BlockTreeMesh; face_coefficient = 1,
    T = eltype(mesh.origin),
    construction_cache = GradientConstructionCache(T),
    assemble_matrix = true,
) =
    EllipticOperator(
    mesh; reaction = 0, diffusion = 1,
    face_coefficient, construction_cache, assemble_matrix, T
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
        counts[_elliptic_cell_index(face.minus, nx, ny)] += 1
        counts[_elliptic_cell_index(face.plus, nx, ny)] += 1
    end
    offsets = Vector{Int}(undef, length(counts) + 1)
    offsets[1] = 1
    for cell in eachindex(counts)
        offsets[cell + 1] = offsets[cell] + counts[cell]
        counts[cell] = offsets[cell]
    end
    faces = Vector{Int}(undef, offsets[end] - 1)
    signs = Vector{Int8}(undef, length(faces))
    for (face_index, face) in pairs(mesh.faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
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
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
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
            reaction * operator.volumes[index] * source[index] : source[index]
    end
    for (face_index, face) in pairs(faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
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
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
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
            reaction * operator.volumes[cell]
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
    fill!(destination.boundary, zero(T))
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
