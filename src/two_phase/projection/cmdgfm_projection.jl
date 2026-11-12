"""Configuration for the nonsymmetric two-block CMDGFM projection."""
struct CMDGFMProjectionSolver{T, I, P, PO}
    interface::I
    density::NTuple{2, T}
    preconditioner::P
    preconditioner_options::PO
end

function CMDGFMProjectionSolver(
        interface::InterfaceField{T}; density = (one(T), one(T)),
        preconditioner = :block,
        preconditioner_options = (;),
    ) where {T}
    rho = (T(density[1]), T(density[2]))
    all(value -> isfinite(value) && value > zero(T), rho) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    preconditioner in (:block, :ilu, nothing) || throw(
        ArgumentError(
            "CMDGFM preconditioner must be :block, :ilu, or nothing",
        ),
    )
    return CMDGFMProjectionSolver{
        T, typeof(interface), typeof(preconditioner),
        typeof(preconditioner_options),
    }(interface, rho, preconditioner, preconditioner_options)
end

"""Gauge-reduced pressure action backed by one tree-geometric GMG V-cycle."""
struct CMDGFMReducedPressurePreconditioner{T, P}
    multigrid::P
    pressure_indices::Vector{Int}
    pressure_anchors::Vector{Int}
    full_right_hand_side::Vector{T}
    full_solution::Vector{T}
    # Coefficients must be snapshotted: the fine operator now updates in place,
    # so comparing it to the hierarchy's aliased fine operator is insufficient.
    fine_coefficients::Vector{T}
    mesh_keys::Vector{BlockKey}
    geometry_revision::UInt
end

"""Lower CMDGFM block preconditioner using pressure GMG and jump ILU."""
struct CMDGFMBlockPreconditioner{T, P, J, C}
    scalar_type::Type{T}
    pressure_preconditioner::P
    jump_preconditioner::Union{Nothing, J}
    jump_pressure::C
    jump_work::Vector{T}
    jump_column_pointers::Vector{Int}
    jump_row_values::Vector{Int}
    jump_values::Vector{T}
    mixed_faces::Vector{Int}
    pressure_size::Int
    dimensions::Tuple{Int, Int}
    pressure_hierarchy_reused::Bool
    jump_factor_reused::Bool

    function CMDGFMBlockPreconditioner{T, P, J, C}(
            scalar_type, pressure_preconditioner, jump_preconditioner,
            jump_pressure, jump_work, jump_column_pointers, jump_row_values,
            jump_values, mixed_faces, pressure_size, dimensions,
            pressure_hierarchy_reused, jump_factor_reused,
        ) where {T, P, J, C}
        return new{T, P, J, C}(
            scalar_type, pressure_preconditioner, jump_preconditioner,
            jump_pressure, jump_work, jump_column_pointers, jump_row_values,
            jump_values, mixed_faces, pressure_size, dimensions,
            pressure_hierarchy_reused, jump_factor_reused,
        )
    end
end

# The pressure operator can switch between these two concrete metrics after
# regridding. Both hierarchies use assembled CSC levels and SuiteSparse LU.
import SparseArrays: UMFPACK
const _CMDGFMLevel{T} = MultigridLevel{
    T, SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int},
}
const _CMDGFMHierarchy{T, O} = TreeGeometricMultigridPreconditioner{
    T, O, UMFPACK.UmfpackLU{Float64, Int}, _CMDGFMLevel{T}, FaceCoefficientRestriction{T},
}
const _CMDGFMJumpILU{T} = ExternalPreconditioner{T, IncompleteLU.ILUFactorization{T, Int}}
const _CMDGFMDiagonalBlock{T, M} = CMDGFMBlockPreconditioner{
    T, CMDGFMReducedPressurePreconditioner{T, _CMDGFMHierarchy{T, _CMDGFMDiagonalOperator{T, M}}},
    _CMDGFMJumpILU{T}, SparseMatrixCSC{T, Int},
}
const _CMDGFMSparseBlock{T, M} = CMDGFMBlockPreconditioner{
    T, CMDGFMReducedPressurePreconditioner{T, _CMDGFMHierarchy{T, _CMDGFMSparseOperator{T, M}}},
    _CMDGFMJumpILU{T}, SparseMatrixCSC{T, Int},
}

"""Reusable diagonal scaling for the two-velocity Krylov systems."""
struct _ProjectionKrylovScaling{T}
    rows::Vector{T}
    columns::Vector{T}
    rhs::Vector{T}
    scratch::Vector{T}
end

_ProjectionKrylovScaling(::Type{T}) where {T} =
    _ProjectionKrylovScaling(T[], T[], T[], T[])

function _resize_projection_scaling!(scaling, n)
    for values in (scaling.rows, scaling.columns, scaling.rhs, scaling.scratch)
        resize!(values, n)
    end
    return scaling
end

# If x = D_c*x̂ and A*x = b, Krylov sees (D_r*A*D_c)*x̂ = D_r*b.
# The physical residual and solution remain available in their original units.
# `matrix` is the system's `_RowForm`, so the product threads over its rows;
# the diagonal scalings thread per entry. Both are bitwise the serial result.
struct _ScaledProjectionOperator{T, A}
    matrix::A
    scaling::_ProjectionKrylovScaling{T}
end

Base.size(op::_ScaledProjectionOperator) = size(op.matrix)
Base.size(op::_ScaledProjectionOperator, dimension::Integer) = size(op.matrix, dimension)
Base.eltype(::_ScaledProjectionOperator{T}) where {T} = T

function mul!(destination, op::_ScaledProjectionOperator, source)
    scaling = op.scaling
    scratch, columns, rows = scaling.scratch, scaling.columns, scaling.rows
    length(scratch) == length(source) == length(destination) ||
        throw(DimensionMismatch("scaled projection vector sizes differ"))
    @_parallel for index in 1:length(scratch)
        @inbounds scratch[index] = columns[index] * source[index]
    end
    mul!(destination, op.matrix, scratch)
    @_parallel for index in 1:length(destination)
        @inbounds destination[index] = rows[index] * destination[index]
    end
    return destination
end

struct _ScaledProjectionPreconditioner{T, P}
    preconditioner::P
    scaling::_ProjectionKrylovScaling{T}
end

Base.size(pc::_ScaledProjectionPreconditioner) = size(pc.preconditioner)
Base.eltype(::_ScaledProjectionPreconditioner{T}) where {T} = T

function ldiv!(destination, pc::_ScaledProjectionPreconditioner, source)
    scaling = pc.scaling
    scratch, columns, rows = scaling.scratch, scaling.columns, scaling.rows
    length(scratch) == length(source) == length(destination) ||
        throw(DimensionMismatch("scaled projection vector sizes differ"))
    @_parallel for index in 1:length(scratch)
        @inbounds scratch[index] = source[index] / rows[index]
    end
    ldiv!(destination, pc.preconditioner, scratch)
    @_parallel for index in 1:length(destination)
        @inbounds destination[index] = destination[index] / columns[index]
    end
    return destination
end

function _scaled_projection_matrix(matrix::SparseMatrixCSC, scaling)
    result = copy(matrix)
    values = nonzeros(result)
    rows = rowvals(result)
    for column in axes(result, 2)
        column_scale = scaling.columns[column]
        for pointer in nzrange(result, column)
            @inbounds values[pointer] *= scaling.rows[rows[pointer]] * column_scale
        end
    end
    return result
end

"""Smallest cell edge in a block, used as the local projection defect length."""
@inline _projection_block_spacing(mesh, block) = min(mesh.blocks[block].spacing...)

"""Smallest edge of either cell adjacent to a projection face.

This is the fine-cell minimum edge on a coarse--fine face, including on
anisotropic meshes; it is a defect length, not the center-to-center distance.
"""
@inline function _projection_face_spacing(mesh, face_index)
    face = @inbounds mesh.faces[face_index]
    return min(
        _projection_block_spacing(mesh, face.minus.block),
        _projection_block_spacing(mesh, face.plus.block),
    )
end

@inline _projection_reference_length(mesh) = max(mesh.extent...)

# With L the domain length, pressure has scale rho*(L/dt)^2 and
# the gradient-jump unknown has scale L/dt^2. The integrated pressure
# and jump equations have scales L^2/dt^2 and L/dt^2, respectively.
function _cmdgfm_krylov_scaling!(scaling, assembled, density, dt)
    T = eltype(scaling.rows)
    np, nx = length(assembled.pressure_indices), length(assembled.mixed_faces)
    _resize_projection_scaling!(scaling, np + nx)
    h = T(_projection_reference_length(assembled.pressure_operator.mesh))
    time = T(dt)
    pressure_scale = T(density) * (h / time)^2
    jump_scale = h / time^2
    pressure_row = (time / h)^2
    jump_row = time^2 / h
    for index in 1:np
        scaling.rows[index] = pressure_row
        scaling.columns[index] = pressure_scale
    end
    for index in (np + 1):(np + nx)
        scaling.rows[index] = jump_row
        scaling.columns[index] = jump_scale
    end
    return scaling
end

"""Reusable fields and the most recently assembled CMDGFM system."""
mutable struct CMDGFMProjectionWorkspace{T, M <: BlockTreeMesh}
    pressure_increment::CellField{T, M}
    increment_keys::Vector{BlockKey}
    pressure_gradient::FaceVelocity{T, M}
    base_gradient::FaceVelocity{T, M}
    liquid_gradient::FaceVelocity{T, M}
    gas_gradient::FaceVelocity{T, M}
    jump::Vector{T}
    liquid_fraction_x::FaceField{1, T, M}
    liquid_fraction_y::FaceField{2, T, M}
    matrix::SparseMatrixCSC{T, Int}
    jump_block::SparseMatrixCSC{T, Int}
    right_hand_side::Vector{T}
    solution::Vector{T}
    pressure_indices::Vector{Int}
    mixed_faces::Vector{Int}
    stencils::Vector{CMDGFMJumpStencil{T}}
    block_preconditioner::Union{Nothing, _CMDGFMDiagonalBlock{T, M}, _CMDGFMSparseBlock{T, M}}
    krylov::Union{Nothing, Krylov.BicgstabWorkspace{T, T, Vector{T}}}
    correction_rhs::Vector{T}
    scaling::_ProjectionKrylovScaling{T}
    pressure_cache::CMDGFMPressureCache{T, M}
    initial_guess_used::Bool
    # Per-chunk flags of `_mark_inactive_phase_velocities!` (resized on use).
    inactive_failures::Vector{Int}
    # Stored transpose of the assembled system (`_cmdgfm_row_form!`) with its
    # pattern-checked value map, and the chunk maxima and chunk change flags
    # of the threaded asymmetry and solution update.
    matrix_transpose::SparseMatrixCSC{T, Int}
    transpose_map::SparseTransposeMap
    reduction_partials::Vector{T}
    chunk_flags::Vector{Bool}
end

function CMDGFMProjectionWorkspace(
        solver::CMDGFMProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return CMDGFMProjectionWorkspace{T, typeof(mesh)}(
        CellField(mesh, T), BlockKey[],
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        T[], T[], Int[], Int[], CMDGFMJumpStencil{T}[], nothing,
        nothing, T[], _ProjectionKrylovScaling(T), CMDGFMPressureCache(T, typeof(mesh)), false,
        Int[], spzeros(T, 0, 0), SparseTransposeMap(), T[], Bool[],
    )
end

"""Diagnostics for one conservative multidimensional ghost-fluid projection.

The mixture and normal-velocity defects are dimensionless and report the
actual projected state: `dt * max(abs(div(u_mix)))` and the maximum of
`dt * abs(jump_row) / h_face`, respectively. Here `h_face` is the smallest
cell edge among the two adjacent cells, so a coarse--fine face uses the
fine-cell minimum edge. The algebraic convergence check also bounds
`dt^2 * max(abs(r_pressure) / cell_volume)`, including the pressure rows
omitted to fix the gauge, and `dt^2 * max(abs(r_jump) / h_face)`. For a
shortened step retaining an inherited mixture-flux defect, convergence
measures the mixture deviation from that target but still requires the
actual normal defect to be within tolerance.
"""
struct CMDGFMProjectionStats{T}
    converged::Bool
    iterations::Int
    initial_residual::T
    final_residual::T
    matrix_asymmetry::T
    mixture_divergence::T
    mixture_volume_defect::T
    normal_velocity_defect::T
    target_defect::T
    momentum_error::NTuple{2, T}
    momentum_identity_error::T
    minimum_jump_diagonal_margin::T
    pressure_unknowns::Int
    jump_unknowns::Int
    underresolved_jump_rows::Int
    preconditioner::Symbol
    pressure_hierarchy_reused::Bool
    jump_factor_reused::Bool
end

_dimensionless_residual(stats::CMDGFMProjectionStats) = max(
    stats.mixture_volume_defect,
    stats.normal_velocity_defect,
)

function Base.show(io::IO, stats::CMDGFMProjectionStats)
    return print(
        io, "CMDGFMProjectionStats(converged=", stats.converged,
        ", iterations=", stats.iterations,
        ", initial_residual=", stats.initial_residual,
        ", final_residual=", stats.final_residual,
        ", matrix_asymmetry=", stats.matrix_asymmetry,
        ", mixture_divergence=", stats.mixture_divergence,
        ", mixture_volume_defect=", stats.mixture_volume_defect,
        ", normal_velocity_defect=", stats.normal_velocity_defect,
        ", target_defect=", stats.target_defect,
        ", momentum_error=", stats.momentum_error,
        ", momentum_identity_error=", stats.momentum_identity_error,
        ", minimum_jump_diagonal_margin=",
        stats.minimum_jump_diagonal_margin,
        ", unknowns=(`p`=", stats.pressure_unknowns,
        ", `xi`=", stats.jump_unknowns, ")",
        ", underresolved_jump_rows=", stats.underresolved_jump_rows,
        ", preconditioner=", stats.preconditioner,
        ", pressure_hierarchy_reused=", stats.pressure_hierarchy_reused,
        ", jump_factor_reused=", stats.jump_factor_reused, ")",
    )
end


Base.size(preconditioner::CMDGFMReducedPressurePreconditioner) = (
    length(preconditioner.pressure_indices),
    length(preconditioner.pressure_indices),
)
Base.eltype(::CMDGFMReducedPressurePreconditioner{T}) where {T} = T

function ldiv!(
        destination::AbstractVector{T},
        preconditioner::CMDGFMReducedPressurePreconditioner{T},
        right_hand_side::AbstractVector{T},
    ) where {T}
    fill!(preconditioner.full_right_hand_side, zero(T))
    for local_index in eachindex(preconditioner.pressure_indices)
        cell = @inbounds preconditioner.pressure_indices[local_index]
        anchor = @inbounds preconditioner.pressure_anchors[local_index]
        value = @inbounds right_hand_side[local_index]
        @inbounds begin
            preconditioner.full_right_hand_side[cell] += value
            preconditioner.full_right_hand_side[anchor] -= value
        end
    end
    ldiv!(
        preconditioner.full_solution, preconditioner.multigrid,
        preconditioner.full_right_hand_side,
    )
    for local_index in eachindex(preconditioner.pressure_indices)
        cell = @inbounds preconditioner.pressure_indices[local_index]
        anchor = @inbounds preconditioner.pressure_anchors[local_index]
        @inbounds destination[local_index] =
            preconditioner.full_solution[cell] -
            preconditioner.full_solution[anchor]
    end
    return destination
end


Base.size(preconditioner::CMDGFMBlockPreconditioner) = preconditioner.dimensions
Base.eltype(::CMDGFMBlockPreconditioner{T}) where {T} = T
_preconditioner_name(::CMDGFMBlockPreconditioner) = :cmdgfm_block

function ldiv!(
        destination::AbstractVector{T},
        preconditioner::CMDGFMBlockPreconditioner{T},
        right_hand_side::AbstractVector{T},
    ) where {T}
    np = preconditioner.pressure_size
    if np > 0
        pressure_destination = view(destination, 1:np)
        ldiv!(
            pressure_destination, preconditioner.pressure_preconditioner,
            view(right_hand_side, 1:np),
        )
    end
    nx = length(right_hand_side) - np
    if nx > 0
        jump_rhs = preconditioner.jump_work
        copyto!(jump_rhs, view(right_hand_side, (np + 1):(np + nx)))
        if np > 0
            mul!(
                jump_rhs, preconditioner.jump_pressure,
                view(destination, 1:np), -one(T), one(T),
            )
        end
        ldiv!(
            view(destination, (np + 1):(np + nx)),
            preconditioner.jump_preconditioner, jump_rhs,
        )
    end
    return destination
end

function _cmdgfm_incident_faces(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    result = [Int[] for _ in 1:ncells(mesh)]
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        push!(result[minus], index)
        push!(result[plus], index)
    end
    return result
end

@inline function _cmdgfm_physical_cell_normal(
        interface::InterfaceField{T}, reference::CellRef,
    ) where {T}
    spacing = interface.mesh.blocks[reference.block].spacing
    reference_nx = _get(interface.normal_x, reference)
    reference_ny = _get(interface.normal_y, reference)
    hypot(reference_nx, reference_ny) > sqrt(eps(T)) || return (zero(T), zero(T))
    nx = reference_nx / spacing[1]
    ny = reference_ny / spacing[2]
    magnitude = hypot(nx, ny)
    return (nx / magnitude, ny / magnitude)
end

function _cmdgfm_face_normal(
        interface::InterfaceField{T}, face::InteriorFace,
    ) where {T}
    minus_fraction = _conditional_fraction(interface, face.minus)
    plus_fraction = _conditional_fraction(interface, face.plus)
    minus_weight = minus_fraction * (one(T) - minus_fraction)
    plus_weight = plus_fraction * (one(T) - plus_fraction)
    minus = _cmdgfm_physical_cell_normal(interface, face.minus)
    plus = _cmdgfm_physical_cell_normal(interface, face.plus)
    nx = minus_weight * minus[1] + plus_weight * plus[1]
    ny = minus_weight * minus[2] + plus_weight * plus[2]
    magnitude = hypot(nx, ny)
    weight = minus_weight + plus_weight
    if iszero(weight) || magnitude <= sqrt(eps(T)) * weight
        # An interface aligned with this face can have pure cells on both
        # sides. The fraction difference then supplies its oriented normal.
        # The c(1-c) weights suppress pure-cell normals: those are local
        # extrapolations rather than PLIC normals and need not change sign
        # exactly when the two phase labels are exchanged. The weights vary
        # smoothly through mixed cells and are invariant under c -> 1-c.
        difference = minus_fraction - plus_fraction
        nx = face.axis == 1 ? difference : zero(T)
        ny = face.axis == 2 ? difference : zero(T)
        magnitude = hypot(nx, ny)
    end
    magnitude > sqrt(eps(T)) || return nothing
    return (nx / magnitude, ny / magnitude)
end

function _cmdgfm_unique_incident_candidates(
        incident, face::InteriorFace, face_index, candidate_axis, axis_of,
        result = Int[],
    )
    empty!(result)
    for cell in (face.minus, face.plus)
        # `_cell_index` needs the block dimensions, but the incident
        # table is indexed by global cell number. Recover it from any face is
        # deliberately avoided; callers pass an incident accessor closure.
        for candidate in incident(cell)
            candidate == face_index && continue
            axis_of(candidate) == candidate_axis || continue
            candidate in result || push!(result, candidate)
        end
    end
    return result
end

_cmdgfm_cell_faces(table, cell) = table[cell]
_cmdgfm_cell_faces(table::SparseMatrixCSC, cell) = view(rowvals(table), nzrange(table, cell))

_cmdgfm_vertex_patch(patches, mesh, face, face_index, endpoint) =
    get(patches, _vertex_key(mesh, face, endpoint), nothing)
_cmdgfm_patch_face(candidate::MetricFace) = candidate.index
_cmdgfm_patch_face(candidate::Int) = candidate

function _cmdgfm_vertex_candidates(
        patches, mesh::BlockTreeMesh, face::InteriorFace,
        face_index::Int, candidate_axis::Int, result = Int[],
    )
    empty!(result)
    for endpoint in 1:2
        patch = _cmdgfm_vertex_patch(patches, mesh, face, face_index, endpoint)
        isnothing(patch) && continue
        for candidate in patch
            index = _cmdgfm_patch_face(candidate)
            index == face_index && continue
            mesh.faces[index].axis == candidate_axis || continue
            index in result || push!(result, index)
        end
    end
    sort!(result)
    return result
end

function _cmdgfm_vertex_candidates(
        graph::CMDGFMVertexConnectivity, mesh::BlockTreeMesh, face::InteriorFace,
        face_index::Int, candidate_axis::Int, result = Int[],
    )
    # Read the compact patch ranges directly. Returning a patch view to a
    # generic traversal made that view escape once per endpoint/stencil.
    empty!(result)
    for vertex in graph.endpoints[face_index]
        for pointer in graph.offsets[vertex]:(graph.offsets[vertex + 1] - 1)
            index = graph.faces[pointer]
            index == face_index && continue
            mesh.faces[index].axis == candidate_axis || continue
            index in result || push!(result, index)
        end
    end
    sort!(result)
    return result
end

function _cmdgfm_stencil_buffers!(storage::CMDGFMStencilStorage{T}, index) where {T}
    if index > length(storage.faces)
        push!(storage.faces, Int[])
        push!(storage.weights, T[])
    end
    return empty!(storage.faces[index]), empty!(storage.weights[index])
end

function _cmdgfm_unresolved_stencil(face, transverse, weights, ::Type{T}) where {T}
    return CMDGFMJumpStencil(
        face, empty!(transverse), empty!(weights), zero(T),
        0, 0, one(T), false, false
    )
end

@inline function _cmdgfm_aperture_mixed(data, index, floor)
    return data.liquid_aperture[index] > floor &&
        data.gas_aperture[index] > floor
end

@inline function _cmdgfm_jump_active(data, index, floor)
    fraction = @inbounds data.liquid_fraction[index]
    return fraction > floor && fraction < one(fraction) - floor
end

function _cmdgfm_jump_stencils(
        solver::CMDGFMProjectionSolver{T}, data;
        incident_faces = nothing, vertex_patches = nothing, storage = nothing,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    nx, ny = mesh.block_size
    # Do not reassign the argument captured by `incident`: Julia would box
    # that binding, making every candidate an abstract value. On Stokes3
    # this alone allocated >1 MB per projection despite reusing the vectors.
    # Bind each finalized table once, so traversal has concrete index types.
    incident_table = isnothing(incident_faces) ? _cmdgfm_incident_faces(mesh) : incident_faces
    vertex_table = isnothing(vertex_patches) ? _vertex_patches(mesh, T) : vertex_patches
    incident(reference) = _cmdgfm_cell_faces(
        incident_table,
        _cell_index(reference, nx, ny)
    )
    face_axis(index) = mesh.faces[index].axis
    floor = interface.fraction_floor
    stencils = isnothing(storage) ? CMDGFMJumpStencil{T}[] : empty!(storage.all)
    for face_index in data.mixed
        _cmdgfm_jump_active(data, face_index, floor) || continue
        transverse, weights = isnothing(storage) ? (Int[], T[]) :
            _cmdgfm_stencil_buffers!(storage, length(stencils) + 1)
        face = mesh.faces[face_index]
        open_aperture = data.liquid_aperture[face_index] +
            data.gas_aperture[face_index]
        if open_aperture <= floor
            push!(
                stencils,
                _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
            )
            continue
        end
        if !_cmdgfm_aperture_mixed(data, face_index, floor)
            normal_neighbours = _cmdgfm_unique_incident_candidates(
                incident, face, face_index, face.axis, face_axis,
                isnothing(storage) ? Int[] : storage.normal,
            )
            if any(
                    index -> _cmdgfm_aperture_mixed(data, index, floor),
                    normal_neighbours,
                )
                # The Fortran reference suppresses a fraction-mixed face when
                # its phase aperture is closed but a neighbouring pressure
                # gradient jump is active in the divergence operator.
                push!(
                    stencils,
                    _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
                )
                continue
            end
        end
        normal = _cmdgfm_face_normal(interface, face)
        if isnothing(normal) || abs(normal[face.axis]) <= sqrt(eps(T))
            push!(
                stencils,
                _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
            )
            continue
        end
        tangent = 3 - face.axis
        ratio = normal[tangent] / normal[face.axis]
        if abs(normal[face.axis]) >= one(T) - T(64) * eps(T)
            # With an axis-aligned normal the Fortran implementation bypasses
            # interpolation and removes the face-normal velocity jump exactly.
            push!(
                stencils,
                CMDGFMJumpStencil(
                    face_index, transverse, weights, ratio, 0, 0, one(T), true, true,
                ),
            )
            continue
        end
        _cmdgfm_vertex_candidates(vertex_table, mesh, face, face_index, tangent, transverse)
        filter!(transverse) do index
            _cmdgfm_jump_active(data, index, floor) &&
                data.liquid_aperture[index] + data.gas_aperture[index] > floor
        end
        resize!(weights, length(transverse))
        for (local_index, index) in pairs(transverse)
            weights[local_index] = data.liquid_fraction[index] *
                (one(T) - data.liquid_fraction[index])
        end
        weight_sum = sum(weights)
        touches_boundary = any(
            endpoint -> _physical_vertex(
                mesh, _vertex_key(mesh, face, endpoint),
            ),
            1:2,
        )
        crosses_level = any(
            index -> mesh.faces[index].area != face.area, transverse,
        )
        required_transverse = touches_boundary || crosses_level ? 1 : 2
        if length(transverse) < required_transverse ||
                weight_sum <= zero(T)
            push!(
                stencils,
                _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
            )
            continue
        end
        weights ./= weight_sum
        ratio_sum = abs(ratio)
        beta = one(T)
        gas_face = liquid_face = 0
        if ratio_sum > one(T)
            parallel = _cmdgfm_vertex_candidates(
                vertex_table, mesh, face, face_index, face.axis,
                isnothing(storage) ? Int[] : storage.parallel,
            )
            if !isnothing(interface.geometry)
                # A fully solid dual volume has neither intrinsic velocity.
                # It cannot supply a phase-side inference value, even though
                # the protected algebraic mass used by the pressure metric
                # may be positive. Retain only physical donor volumes.
                filter!(
                    index -> interface.geometry.face_dual_volumes[index] >
                        zero(T), parallel
                )
            end
            if length(parallel) < 2
                push!(
                    stencils,
                    _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
                )
                continue
            end
            sort!(
                parallel; by = index -> (
                    data.liquid_mass[index] / data.total_mass[index], index,
                )
            )
            gas_face = first(parallel)
            liquid_face = last(parallel)
            if data.gas_mass[gas_face] < data.liquid_mass[gas_face] ||
                    data.liquid_mass[liquid_face] < data.gas_mass[liquid_face]
                # `include_infer_contribution` in the reference requires the
                # gas inference face to be gas dominated and the liquid face
                # to be liquid dominated. Without both, the row is explicitly
                # classified as underresolved and xi is extended by zero.
                push!(
                    stencils,
                    _cmdgfm_unresolved_stencil(face_index, transverse, weights, T),
                )
                continue
            end
            gamma_liquid_gas =
                data.liquid_mass[gas_face] / data.total_mass[gas_face]
            gamma_gas_liquid =
                data.gas_mass[liquid_face] / data.total_mass[liquid_face]
            inferred_sum = gamma_liquid_gas + gamma_gas_liquid
            denominator = ratio_sum - inferred_sum
            denominator > zero(T) || error(
                "CMDGFM beta denominator is not positive on face $face_index",
            )
            beta = clamp(
                (one(T) - inferred_sum) / denominator, zero(T), one(T),
            )
        end
        push!(
            stencils,
            CMDGFMJumpStencil(
                face_index, transverse, weights, ratio,
                gas_face, liquid_face, beta, true, false,
            ),
        )
    end
    return stencils
end

@inline function _cmdgfm_normal_velocity_row(velocity::TwoPhaseVelocity, stencil)
    T = eltype(velocity[1])
    liquid, gas = velocity
    face = stencil.face
    value = @inbounds gas.interior[face] - liquid.interior[face]
    stencil.resolved || return value
    factor = stencil.beta * stencil.ratio
    for (candidate, weight) in zip(stencil.transverse_faces, stencil.transverse_weights)
        @inbounds value += factor * weight *
            (gas.interior[candidate] - liquid.interior[candidate])
    end
    if stencil.gas_inference_face > 0
        @inbounds value += (stencil.beta - one(T)) * (
            gas.interior[stencil.gas_inference_face] -
                liquid.interior[stencil.liquid_inference_face]
        )
    end
    return value
end

function _cmdgfm_jump_blocks(
        stencils, mixed_faces, data, common_gradient, pressure_indices,
        velocity::TwoPhaseVelocity, dt, ::Type{T}; pressure_lookup = nothing,
        storage = nothing,
    ) where {T}
    jump_index = if isnothing(storage)
        Dict(face => index for (index, face) in pairs(mixed_faces))
    else
        resize!(storage.jump_lookup, length(data.liquid_fraction))
        fill!(storage.jump_lookup, 0)
        for (index, face) in pairs(mixed_faces)
            storage.jump_lookup[face] = index
        end
        storage.jump_lookup
    end
    rows, columns, values = isnothing(storage) ? (Int[], Int[], T[]) :
        _cmdgfm_clear_triplets!(storage.jump_triplets)
    pressure_rows, pressure_columns, pressure_values = isnothing(storage) ?
        (Int[], Int[], T[]) : _cmdgfm_clear_triplets!(storage.pressure_triplets)
    right_hand_side = isnothing(storage) ? zeros(T, length(stencils)) : storage.jump_rhs
    resize!(right_hand_side, length(stencils))
    underresolved = 0
    liquid = velocity[1].interior
    gas = velocity[2].interior
    isnothing(pressure_lookup) && (
        pressure_lookup = Dict(
            cell => local_index for (local_index, cell) in pairs(pressure_indices)
        )
    )
    gradient_rows = _cmdgfm_gradient_rows(common_gradient)
    for (row, stencil) in pairs(stencils)
        face = stencil.face
        push!(rows, row)
        push!(columns, jump_index[face])
        push!(values, one(T))
        if !stencil.resolved
            underresolved += 1
            @inbounds right_hand_side[row] =
                (gas[face] - liquid[face]) / T(dt)
            continue
        end
        rhs = @inbounds(gas[face] - liquid[face]) / T(dt)
        transverse_factor = stencil.beta * stencil.ratio
        for (candidate, weight) in zip(
                stencil.transverse_faces, stencil.transverse_weights,
            )
            coefficient = transverse_factor * weight
            local_index = _cmdgfm_pressure_index(jump_index, candidate)
            iszero(local_index) || begin
                push!(rows, row)
                push!(columns, local_index)
                push!(values, coefficient)
            end
            @inbounds rhs += coefficient *
                (gas[candidate] - liquid[candidate]) / T(dt)
        end
        if stencil.gas_inference_face > 0
            gas_face = stencil.gas_inference_face
            liquid_face = stencil.liquid_inference_face
            inference_factor = stencil.beta - one(T)
            gamma_liquid_gas =
                data.liquid_mass[gas_face] / data.total_mass[gas_face]
            gamma_gas_liquid =
                data.gas_mass[liquid_face] / data.total_mass[liquid_face]
            for (candidate, gamma) in (
                    (gas_face, gamma_liquid_gas),
                    (liquid_face, gamma_gas_liquid),
                )
                local_index = _cmdgfm_pressure_index(jump_index, candidate)
                iszero(local_index) || begin
                    push!(rows, row)
                    push!(columns, local_index)
                    push!(values, inference_factor * gamma)
                end
            end
            # Add (beta-1)*(Gbar_hg-Gbar_hl), evaluating only these rows
            # of the unchanged pressure gradient at the inference donors.
            for (gradient_face, sign) in (
                    (gas_face, one(T)),
                    (liquid_face, -one(T)),
                )
                _cmdgfm_append_gradient_row!(
                    pressure_rows, pressure_columns,
                    pressure_values, gradient_rows, pressure_lookup, row,
                    gradient_face, inference_factor * sign
                )
            end
            @inbounds rhs += inference_factor *
                (gas[gas_face] - liquid[liquid_face]) / T(dt)
        end
        @inbounds right_hand_side[row] = rhs
    end
    jump_block, pressure_block = if isnothing(storage)
        (
            sparse(rows, columns, values, length(stencils), length(mixed_faces)),
            sparse(
                pressure_rows, pressure_columns, pressure_values,
                length(stencils), length(pressure_indices)
            ),
        )
    else
        storage.jump_block = _cmdgfm_sorted_rows_csc!(
            storage.jump_block,
            storage.jump_triplets, length(stencils), length(mixed_faces)
        )
        storage.jump_pressure = _cmdgfm_sorted_rows_csc!(
            storage.jump_pressure,
            storage.pressure_triplets, length(stencils), length(pressure_indices)
        )
        (storage.jump_block, storage.jump_pressure)
    end
    return pressure_block, jump_block, right_hand_side, underresolved
end

function _assemble_cmdgfm_system!(
        solver::CMDGFMProjectionSolver{T},
        workspace::CMDGFMProjectionWorkspace{T}, dt,
        velocity::TwoPhaseVelocity; shared_operator = nothing,
        shared_multigrid = nothing, _inherited_cell_flux = nothing,
        _reference_dt = dt,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    isnothing(_inherited_cell_flux) ||
        length(_inherited_cell_flux) == ncells(mesh) || throw(
        ArgumentError("inherited CMDGFM cell flux must match the current mesh"),
    )
    reference_dt = T(_reference_dt)
    isfinite(reference_dt) && reference_dt > zero(T) || throw(
        ArgumentError("reference dt must be finite and strictly positive"),
    )
    _reference_dt >= dt || throw(
        ArgumentError("reference dt must be greater than or equal to dt"),
    )
    inherited_weight = if isnothing(_inherited_cell_flux) || reference_dt == T(dt)
        zero(T)
    else
        (reference_dt - T(dt)) / (T(dt) * reference_dt)
    end
    data = _projection_face_data(solver, workspace; storage = workspace.pressure_cache.face_data)
    workspace.pressure_cache.face_data = data
    cache = _cmdgfm_pressure_cache!(
        workspace.pressure_cache, solver, data;
        shared_operator
    )
    # The topology cache deliberately permits different concrete elliptic
    # operator/metric types. Specialize the numerical assembly after fetching
    # them, rather than boxing every face coefficient in the hot loops.
    return _assemble_cmdgfm_cached!(
        solver, workspace, dt, velocity, cache, cache.operator, data,
        shared_multigrid, _inherited_cell_flux, inherited_weight,
    )
end

function _assemble_cmdgfm_cached!(
        solver::CMDGFMProjectionSolver{T}, workspace,
        dt, velocity, cache, pressure_operator, data, shared_multigrid,
        inherited_cell_flux, inherited_weight,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
    incidence = cache.incidence
    pressure_indices, pressure_anchors = cache.pressure_indices, cache.pressure_anchors
    all_stencils = _cmdgfm_jump_stencils(
        solver, data;
        incident_faces = cache.incidence, vertex_patches = cache.vertex_patches,
        storage = cache.stencil_storage
    )
    underresolved = count(stencil -> !stencil.resolved, all_stencils)
    # The CMDGFM note explicitly says not to impose a jump condition when the
    # transverse interface stencil is underresolved. Such a face therefore has
    # no retained xi unknown; its gradient jump is extended by zero, exactly as
    # on every other non-retained face.
    stencils, mixed_faces = cache.stencil_storage.resolved, cache.stencil_storage.mixed
    empty!(stencils)
    empty!(mixed_faces)
    for stencil in all_stencils
        stencil.resolved || continue
        push!(stencils, stencil)
        push!(mixed_faces, stencil.face)
    end
    pressure_positive = cache.reduced_pressure
    pressure_jump = _cmdgfm_pressure_jump(cache, data, mixed_faces, mesh)
    common_gradient = _cmdgfm_gradient_rows(cache, interface.geometry)
    jump_pressure, jump_block, jump_rhs, retained_underresolved =
        _cmdgfm_jump_blocks(
        stencils, mixed_faces, data, common_gradient,
        pressure_indices, velocity, dt, T; pressure_lookup = cache.pressure_lookup,
        storage = cache,
    )
    iszero(retained_underresolved) || error(
        "an underresolved CMDGFM row was unexpectedly retained",
    )
    matrix = _cmdgfm_block_matrix!(cache, pressure_positive, pressure_jump, jump_pressure, jump_block)
    _cmdgfm_cell_flux!(cache.cell_flux, data, interface, velocity)
    right_hand_side = workspace.right_hand_side
    np = length(pressure_indices)
    resize!(right_hand_side, np + length(jump_rhs))
    if iszero(inherited_weight) || isnothing(inherited_cell_flux)
        for (local_index, cell) in pairs(pressure_indices)
            right_hand_side[local_index] = cache.cell_flux[cell] / T(dt)
        end
    else
        # cell_flux is the negative integrated mixture divergence. Remove
        # only the short-step excess of the defect inherited before transport.
        for (local_index, cell) in pairs(pressure_indices)
            right_hand_side[local_index] = cache.cell_flux[cell] / T(dt) -
                inherited_weight * inherited_cell_flux[cell]
        end
    end
    copyto!(right_hand_side, np + 1, jump_rhs, 1, length(jump_rhs))
    return (;
        data, pressure_operator, incidence, matrix, jump_block, jump_pressure,
        right_hand_side, pressure_indices, pressure_anchors, mixed_faces,
        stencils, pressure_positive, underresolved, shared_multigrid,
        inherited_cell_flux, inherited_weight,
    )
end

const _CMDGFM_BLOCK_DEFAULTS = let multigrid = MultigridOptions()
    (
        pressure_coarse_size = multigrid.coarse_size,
        pressure_pre_sweeps = multigrid.pre_sweeps,
        pressure_post_sweeps = multigrid.post_sweeps,
        pressure_max_levels = multigrid.max_levels,
        pressure_pruning = :all,
        refresh_pressure_coefficients = true,
        jump_drop_tolerance = 1.0e-3,
        reuse_jump_factorization = true,
    )
end

function _cmdgfm_block_settings(options)
    options isa NamedTuple || throw(
        ArgumentError("CMDGFM block preconditioner options must be named"),
    )
    unknown = setdiff(keys(options), keys(_CMDGFM_BLOCK_DEFAULTS))
    isempty(unknown) || throw(
        ArgumentError("unknown CMDGFM block options: $(Tuple(unknown))"),
    )
    settings = merge(_CMDGFM_BLOCK_DEFAULTS, options)
    MultigridOptions(
        settings.pressure_coarse_size, settings.pressure_pre_sweeps,
        settings.pressure_post_sweeps, settings.pressure_max_levels,
    )
    settings.pressure_pruning in (:all, :deepest) || throw(
        ArgumentError("pressure_pruning must be :all or :deepest"),
    )
    settings.refresh_pressure_coefficients isa Bool || throw(
        ArgumentError("refresh_pressure_coefficients must be Boolean"),
    )
    settings.jump_drop_tolerance >= 0 || throw(
        ArgumentError("jump_drop_tolerance must be nonnegative"),
    )
    return settings
end

function _cmdgfm_pressure_preconditioner(
        solver, assembled, settings, reuse,
    )
    operator = assembled.pressure_operator
    indices = assembled.pressure_indices
    anchors = assembled.pressure_anchors
    reusable = reuse isa CMDGFMReducedPressurePreconditioner &&
        reuse.multigrid.operator.mesh === operator.mesh &&
        reuse.pressure_indices == indices &&
        reuse.pressure_anchors == anchors &&
        _cmdgfm_same_topology(reuse.mesh_keys, operator.mesh) &&
        reuse.geometry_revision == _interface_geometry_revision(solver.interface.geometry)
    shared = assembled.shared_multigrid
    if !isnothing(shared)
        shared.operator === operator || throw(
            ArgumentError(
                "shared CMDGFM multigrid must already describe the current pressure operator"
            )
        )
        if reusable && reuse.multigrid === shared
            copyto!(reuse.fine_coefficients, operator.face_coefficient)
            return reuse, true
        end
        # The balanced-force stage has already numerically refreshed this
        # hierarchy. Wrap its full-cell V-cycle in CMDGFM's pressure gauge;
        # do not construct or refresh a second hierarchy for the same matrix.
        return _cmdgfm_reduced_pressure(
                shared, operator, indices, anchors,
                solver.interface.geometry; reuse
            ), false
    end
    if reusable
        # The phase fractions change at every step even on a fixed mesh. A
        # stale numerical hierarchy was observed to stall BiCGSTAB on Stokes3.
        # Retain its topology and buffers, but update every level whenever the
        # pressure operator's coefficients or cell weights have changed.
        old = reuse.multigrid.operator
        changed = reuse.fine_coefficients != operator.face_coefficient ||
            old.stiffness != operator.stiffness ||
            old.volumes != operator.volumes ||
            old.constraints != operator.constraints
        if settings.refresh_pressure_coefficients && changed
            _refresh_geometric_multigrid!(reuse.multigrid, operator)
            copyto!(reuse.fine_coefficients, operator.face_coefficient)
        end
        return reuse, true
    end
    multigrid = TreeGeometricMultigridPreconditioner(
        operator;
        multigrid = MultigridOptions(
            settings.pressure_coarse_size, settings.pressure_pre_sweeps,
            settings.pressure_post_sweeps, settings.pressure_max_levels,
        ),
        pruning = settings.pressure_pruning,
        geometry = solver.interface.geometry,
        reuse = reuse isa CMDGFMReducedPressurePreconditioner ? reuse.multigrid : nothing,
    )
    return _cmdgfm_reduced_pressure(
            multigrid, operator, indices, anchors,
            solver.interface.geometry; reuse
        ), false
end

function _cmdgfm_reduced_pressure(
        multigrid, operator, indices, anchors, geometry;
        reuse = nothing
    )
    # A new topology needs a new gauge wrapper, not fresh full-cell scratch
    # and snapshots. Retain their capacities independently of whether the
    # hierarchy/operator's concrete type changes at a uniform <-> AMR switch.
    # Refresh snapshots only AFTER deciding numerical/hierarchy validity.
    T = eltype(operator.volumes)
    number_of_cells = length(operator.volumes)
    previous = reuse isa CMDGFMReducedPressurePreconditioner{T} ? reuse : nothing
    pressure_indices = isnothing(previous) ? Int[] : previous.pressure_indices
    pressure_anchors = isnothing(previous) ? Int[] : previous.pressure_anchors
    rhs = isnothing(previous) ? T[] : previous.full_right_hand_side
    solution = isnothing(previous) ? T[] : previous.full_solution
    coefficients = isnothing(previous) ? T[] : previous.fine_coefficients
    keys = isnothing(previous) ? BlockKey[] : previous.mesh_keys
    resize!(pressure_indices, length(indices))
    resize!(pressure_anchors, length(anchors))
    copyto!(pressure_indices, indices)
    copyto!(pressure_anchors, anchors)
    resize!(rhs, number_of_cells)
    resize!(solution, number_of_cells)
    fill!(rhs, zero(T))
    fill!(solution, zero(T))
    resize!(coefficients, length(operator.face_coefficient))
    copyto!(coefficients, operator.face_coefficient)
    resize!(keys, nblocks(operator.mesh))
    for index in eachindex(keys)
        keys[index] = operator.mesh.blocks[index].key
    end
    return CMDGFMReducedPressurePreconditioner(
        multigrid, pressure_indices, pressure_anchors, rhs, solution, coefficients, keys,
        _interface_geometry_revision(geometry),
    )
end

function _cmdgfm_same_jump_matrix(reuse, assembled)
    reuse isa CMDGFMBlockPreconditioner || return false
    settings_match = reuse.mixed_faces == assembled.mixed_faces
    settings_match || return false
    # ILU factors contain numerical values, not just a symbolic pattern.
    # Reusing a factor after a moving interface changes these values can be
    # arbitrarily poor even when the sparsity pattern stays unchanged.
    return reuse.jump_column_pointers == assembled.jump_block.colptr &&
        reuse.jump_row_values == rowvals(assembled.jump_block) &&
        reuse.jump_values == nonzeros(assembled.jump_block)
end


function _cmdgfm_block_preconditioner(
        solver::CMDGFMProjectionSolver{T}, assembled, workspace,
    ) where {T}
    settings = _cmdgfm_block_settings(solver.preconditioner_options)
    reuse = workspace.block_preconditioner
    old_pressure = reuse isa CMDGFMBlockPreconditioner ?
        reuse.pressure_preconditioner : nothing
    pressure, pressure_reused = _cmdgfm_pressure_preconditioner(
        solver, assembled, settings, old_pressure,
    )
    jump_reused = !isempty(assembled.jump_block) &&
        settings.reuse_jump_factorization &&
        _cmdgfm_same_jump_matrix(reuse, assembled)
    jump = if isempty(assembled.jump_block)
        nothing
    elseif jump_reused
        reuse.jump_preconditioner
    else
        incomplete_lu_preconditioner(
            assembled.jump_block;
            drop_tolerance = settings.jump_drop_tolerance,
        )
    end
    jump_work = jump_reused ? reuse.jump_work :
        zeros(T, size(assembled.jump_block, 1))
    jump_column_pointers = jump_reused ? reuse.jump_column_pointers :
        copy(assembled.jump_block.colptr)
    jump_row_values = jump_reused ? reuse.jump_row_values :
        copy(rowvals(assembled.jump_block))
    jump_values = jump_reused ? reuse.jump_values :
        copy(nonzeros(assembled.jump_block))
    mixed_faces = jump_reused ? reuse.mixed_faces :
        copy(assembled.mixed_faces)
    block = CMDGFMBlockPreconditioner{
        T, typeof(pressure), _CMDGFMJumpILU{T}, typeof(assembled.jump_pressure),
    }(
        T, pressure, jump, assembled.jump_pressure,
        jump_work, jump_column_pointers, jump_row_values, jump_values,
        mixed_faces,
        length(assembled.pressure_indices), size(assembled.matrix),
        pressure_reused, jump_reused,
    )
    workspace.block_preconditioner = block
    return block
end

function _cmdgfm_preconditioner(
        solver::CMDGFMProjectionSolver{T}, assembled, workspace; scaling = nothing,
    ) where {T}
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :block
        return _cmdgfm_block_preconditioner(solver, assembled, workspace)
    end
    matrix = isnothing(scaling) ? assembled.matrix :
        _scaled_projection_matrix(assembled.matrix, scaling)
    return _construct_matrix_preconditioner(
        specification, matrix;
        symmetric = false, singular = false,
        options = solver.preconditioner_options,
    )
end

function _cmdgfm_krylov_workspace!(workspace, matrix, right_hand_side)
    krylov = workspace.krylov
    if krylov isa Krylov.BicgstabWorkspace
        _resize_krylov_workspace!(krylov, length(right_hand_side))
    else
        krylov = Krylov.BicgstabWorkspace(matrix, right_hand_side)
        workspace.krylov = krylov
    end
    return krylov
end

"""Dimensionless pressure-volume and local jump-row acceleration defects."""
function _cmdgfm_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    pressure = _projection_pressure_residual_maximum(
        residual, assembled.pressure_indices,
        assembled.pressure_operator.volumes,
    )
    mesh = assembled.pressure_operator.mesh
    # The jump row is an acceleration; dt²/h gives a fractional volume defect.
    jump = zero(T)
    for (index, face) in pairs(assembled.mixed_faces)
        jump = max(jump, abs(residual[np + index]) / _projection_face_spacing(mesh, face))
    end
    return (pressure = T(dt)^2 * pressure, jump = T(dt)^2 * jump)
end

"""Evaluate the omitted pressure equations at the current physical solution."""
function _cmdgfm_anchor_residual_defect(workspace, assembled, dt)
    cache = workspace.pressure_cache
    operator = assembled.pressure_operator
    T = eltype(workspace.solution)
    pressure = cache.elliptic.input
    product = cache.elliptic.output
    fill!(pressure, zero(T))
    for (index, cell) in pairs(assembled.pressure_indices)
        @inbounds pressure[cell] = workspace.solution[index]
    end
    mul!(product, operator.stiffness, pressure)

    # The reduced pressure-jump block omits the anchor rows. Assemble their
    # action from the same oriented face contributions as the retained rows.
    mesh = operator.mesh
    nx, ny = mesh.block_size
    np = length(assembled.pressure_indices)
    for (index, face_index) in pairs(assembled.mixed_faces)
        face = mesh.faces[face_index]
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        minus == plus && continue
        value = assembled.data.coupling[face_index] * T(face.area) *
            workspace.solution[np + index]
        @inbounds begin
            product[minus] -= value
            product[plus] += value
        end
    end

    maximum_defect = zero(T)
    inherited = assembled.inherited_cell_flux
    weight = assembled.inherited_weight
    for cell in eachindex(operator.volumes)
        volume = @inbounds operator.volumes[cell]
        # Nonpositive-volume cells are inactive; retained cells are covered
        # by _cmdgfm_residual_defects. This also checks isolated anchors.
        (volume > zero(T) && iszero(cache.pressure_lookup[cell])) || continue
        rhs = cache.cell_flux[cell] / T(dt)
        if !iszero(weight) && !isnothing(inherited)
            rhs -= weight * inherited[cell]
        end
        maximum_defect = max(maximum_defect, abs(rhs - product[cell]) / volume)
    end
    return T(dt)^2 * maximum_defect
end

function _cmdgfm_scaled_residual_norm(residual, assembled, dt)
    T = eltype(residual)
    h = T(_projection_reference_length(assembled.pressure_operator.mesh))
    np = length(assembled.pressure_indices)
    pressure_scale = (T(dt) / h)^2
    jump_scale = T(dt)^2 / h
    value = zero(T)
    for index in eachindex(residual)
        scale = index <= np ? pressure_scale : jump_scale
        value = hypot(value, scale * residual[index])
    end
    return value
end

"""
    _cmdgfm_row_form!(workspace, matrix)

`_RowForm` of the assembled system `matrix` for the threaded products:
refreshes the workspace's stored transpose (a threaded pattern check and value
gather on the recorded pattern; a serial counting sort when the pattern or the
size changed, which the moving jump set makes the common case).
"""
function _cmdgfm_row_form!(workspace, matrix::SparseMatrixCSC)
    matrix_transpose = workspace.matrix_transpose
    if size(matrix_transpose) != reverse(size(matrix))
        # A size change is a pattern change: the map rebuilds the pattern.
        column_pointers = fill!(resize!(matrix_transpose.colptr, size(matrix, 1) + 1), 1)
        matrix_transpose = SparseMatrixCSC(
            size(matrix, 2), size(matrix, 1), column_pointers,
            resize!(rowvals(matrix_transpose), 0), resize!(nonzeros(matrix_transpose), 0),
        )
        workspace.matrix_transpose = matrix_transpose
    end
    _refresh_matrix_transpose!(matrix_transpose, workspace.transpose_map, matrix)
    return _RowForm(matrix_transpose)
end

"""`residual = rhs - A * solution` with the threaded row-form product of `A`."""
function _cmdgfm_residual!(residual, right_hand_side, row_form, solution)
    mul!(residual, row_form, solution)
    length(right_hand_side) == length(residual) ||
        throw(DimensionMismatch("residual sizes differ"))
    @_parallel for index in 1:length(residual)
        @inbounds residual[index] = right_hand_side[index] - residual[index]
    end
    return residual
end

"""Add the rescaled Krylov `increment` to `solution` and return whether any
entry changed: the serial update, threaded in fixed chunks with one change
flag each (`flags`, grown to the chunk count)."""
function _cmdgfm_apply_increment!(solution, columns, rhs_scale, increment, flags)
    number_of_unknowns = length(solution)
    length(columns) == length(increment) == number_of_unknowns ||
        throw(DimensionMismatch("increment sizes differ"))
    chunks = _reduction_chunks(number_of_unknowns)
    length(flags) < chunks && resize!(flags, chunks)
    @_parallel work = _REDUCTION_CHUNK for chunk in 1:chunks
        changed = false
        for index in _reduction_range(chunk, number_of_unknowns)
            previous = @inbounds solution[index]
            updated = previous + @inbounds(columns[index]) * (rhs_scale * @inbounds(increment[index]))
            @inbounds solution[index] = updated
            changed |= updated != previous
        end
        @inbounds flags[chunk] = changed
    end
    for chunk in 1:chunks
        @inbounds flags[chunk] && return true
    end
    return false
end

function _cmdgfm_initial_solution!(
        work, assembled, previous_increment, dt, preconditioner,
        row_form = _cmdgfm_row_form!(work, assembled.matrix),
    )
    # Guesses are increments, not the input physical pressure. The zero
    # solution is always available as the robust fallback.
    rhs = assembled.right_hand_side
    solution, residual = work.solution, work.correction_rhs
    candidate = work.pressure_cache.initial_guess
    resize!(candidate, length(rhs))
    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    best_defect = maximum(_cmdgfm_residual_defects(rhs, assembled, dt))
    best_l2 = _cmdgfm_scaled_residual_norm(rhs, assembled, dt)
    used = false
    for use_previous_increment in (false, true)
        fill!(candidate, zero(eltype(candidate)))
        if use_previous_increment
            isnothing(previous_increment) && continue
            references = assembled.pressure_operator.references
            nonzero_pressure = false
            for index in 1:np
                cell = assembled.pressure_indices[index]
                anchor = assembled.pressure_anchors[index]
                value = _get(previous_increment, references[cell]) -
                    _get(previous_increment, references[anchor])
                candidate[index] = value
                nonzero_pressure |= !iszero(value)
            end
            nonzero_pressure || continue
        end
        # A stale/nonfinite increment is not a usable guess. In particular,
        # do not feed it into the jump solve before rejecting it.
        all(isfinite, view(candidate, 1:np)) || continue
        if nx > 0
            preconditioner isa CMDGFMBlockPreconditioner || continue
            # For a chosen increment, approximately satisfy the current
            # jump row: J*xi = rhs_jump - jump_pressure*phi. Reuse the current
            # jump ILU and its scratch, not an old face jump on a moved sheet.
            jump_rhs = preconditioner.jump_work
            copyto!(jump_rhs, view(rhs, (np + 1):length(rhs)))
            mul!(
                jump_rhs, assembled.jump_pressure, view(candidate, 1:np),
                -one(eltype(rhs)), one(eltype(rhs))
            )
            ldiv!(
                view(candidate, (np + 1):length(rhs)),
                preconditioner.jump_preconditioner, jump_rhs
            )
        end
        all(isfinite, candidate) || continue
        _cmdgfm_residual!(residual, rhs, row_form, candidate)
        defect = maximum(_cmdgfm_residual_defects(residual, assembled, dt))
        l2 = _cmdgfm_scaled_residual_norm(residual, assembled, dt)
        # Require improvement in both the dimensionless L2 residual and the
        # local volume/jump maximum, so neither can hide the other.
        if isfinite(defect) && isfinite(l2) &&
                defect < eltype(rhs)(0.95) * best_defect && l2 < best_l2
            copyto!(solution, candidate)
            best_defect, best_l2 = defect, l2
            used = true
        end
    end
    _cmdgfm_residual!(residual, rhs, row_form, solution)
    work.initial_guess_used = used
    return used
end

function _cmdgfm_jump_diagonal_margin(jump_block)
    T = eltype(jump_block)
    isempty(jump_block) && return T(Inf)
    diagonal = zeros(T, size(jump_block, 1))
    off_diagonal = zeros(T, size(jump_block, 1))
    rows = rowvals(jump_block)
    for column in axes(jump_block, 2)
        for pointer in nzrange(jump_block, column)
            row = rows[pointer]
            value = abs(nonzeros(jump_block)[pointer])
            if row == column
                diagonal[row] += value
            else
                off_diagonal[row] += value
            end
        end
    end
    return minimum(diagonal .- off_diagonal)
end

"""Reconstruct both phase pressure gradients from a CMDGFM solution."""
function cmdgfm_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::CMDGFMProjectionSolver{T},
        workspace::CMDGFMProjectionWorkspace{T}, pressure_operator, data,
    ) where {T}
    _validate_two_phase_velocity(destination, solver.interface.mesh, T)
    elliptic = workspace.pressure_cache.elliptic
    resize!(elliptic.input, length(pressure_operator.volumes))
    resize!(elliptic.output, length(pressure_operator.volumes))
    resize!(elliptic.face_input, length(pressure_operator.face_coefficient))
    resize!(elliptic.face_output, length(pressure_operator.face_coefficient))
    return _projection_gradient_core!(
        destination, pressure, solver, workspace, pressure_operator, data, elliptic,
    )
end

function cmdgfm_gradient!(
        destination::TwoPhaseVelocity, pressure::CellField,
        solver::CMDGFMProjectionSolver,
        workspace::CMDGFMProjectionWorkspace,
    )
    data = _projection_face_data(solver, workspace; storage = workspace.pressure_cache.face_data)
    workspace.pressure_cache.face_data = data
    cache = _cmdgfm_pressure_cache!(workspace.pressure_cache, solver, data)
    operator = cache.operator
    return cmdgfm_gradient!(
        destination, pressure, solver, workspace, operator, data,
    )
end

"""Evaluate the vector jump interpolant represented by one CMDGFM stencil."""
function cmdgfm_jump_interpolant(
        velocity::TwoPhaseVelocity,
        stencil::CMDGFMJumpStencil{T},
    ) where {T}
    stencil.resolved || throw(
        ArgumentError("CMDGFM jump stencil is underresolved"),
    )
    beta = stencil.beta
    beta > sqrt(eps(T)) || throw(
        ArgumentError("the unscaled jump interpolant is singular for beta=0"),
    )
    liquid, gas = velocity
    face = stencil.face
    normal_jump = @inbounds gas.interior[face] - liquid.interior[face]
    if stencil.gas_inference_face > 0
        inferred = @inbounds(
            gas.interior[stencil.gas_inference_face] -
                liquid.interior[stencil.liquid_inference_face]
        )
        normal_jump = normal_jump / beta + (one(T) - inv(beta)) * inferred
    end
    tangential_jump = zero(T)
    for (candidate, weight) in zip(
            stencil.transverse_faces, stencil.transverse_weights,
        )
        @inbounds tangential_jump += weight *
            (gas.interior[candidate] - liquid.interior[candidate])
    end
    axis = velocity[1].mesh.faces[face].axis
    return axis == 1 ? (normal_jump, tangential_jump) :
        (tangential_jump, normal_jump)
end

function _cmdgfm_normal_velocity_defect(velocity::TwoPhaseVelocity, stencils, dt)
    T = eltype(velocity[1])
    mesh = velocity[1].mesh
    defect = zero(T)
    for stencil in stencils
        value = abs(_cmdgfm_normal_velocity_row(velocity, stencil))
        defect = max(defect, T(dt) * value / _projection_face_spacing(mesh, stencil.face))
    end
    return defect
end

"""
    project!(velocity, pressure, solver::CMDGFMProjectionSolver, dt, workspace;
             tolerance, max_iterations, warm_start=true)

Apply the conservative multidimensional ghost-fluid projection. The predictor
must already contain `-dt` times the current projection's weighted gradient of
the input physical pressure. The nonsymmetric two-block system solves for a
pressure increment and mixed-face gradient jump; velocities are corrected by
that increment before it is added to `pressure`. Pass a zero pressure field for
an absolute velocity projection.

`_reference_dt` must be finite and at least `dt`. When
`_inherited_cell_flux` supplies the pre-transport negative integrated mixture
divergence, the projected cell flux targets `(1 - dt / _reference_dt)` times
that saved flux, while every resolved current-interface normal-velocity jump
targets zero. Convergence measures the mixture defect relative to its target
and the actual normal-velocity defect; the reported mixture and normal defects
always describe the actual projected state. Without an inherited flux, the
mixture flux also targets zero.

BiCGSTAB uses a tree-GMG pressure/active-jump-ILU block preconditioner by
default. A whole-system ILU remains available for comparison. Residual
replacement checks the omitted gauge rows as well as the retained system and
counts all attempts against `max_iterations`. `warm_start` tries
residual-checked previous-increment/current-jump guesses; `false` retains the
zero-start reference solve with the same convergence criteria.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::CMDGFMProjectionSolver{T}, dt,
        workspace::CMDGFMProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
        _pressure_operator = nothing, _pressure_multigrid = nothing,
        _inherited_cell_flux = nothing, _reference_dt = dt, warm_start = true,
    ) where {T}
    interface = solver.interface
    _validate_two_phase_velocity(velocity, interface.mesh, T)
    pressure.mesh === interface.mesh === workspace.pressure_gradient.mesh ||
        throw(ArgumentError("CMDGFM projection arguments must share a mesh"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    tolerance > zero(tolerance) || throw(
        ArgumentError("tolerance must be strictly positive"),
    )
    max_iterations > 0 || throw(
        ArgumentError("max_iterations must be strictly positive"),
    )
    _mark_inactive_phase_velocities!(velocity, interface, workspace.inactive_failures)
    assembled = _assemble_cmdgfm_system!(
        solver, workspace, dt, velocity;
        shared_operator = _pressure_operator, shared_multigrid = _pressure_multigrid,
        _inherited_cell_flux, _reference_dt,
    )
    # Cache/operator fields can change concrete type after AMR (diagonal
    # versus sparse metric). Specialize the solve/recovery on the actual
    # assembly once, instead of dynamically accessing it inside every face
    # and cell loop. This is a function barrier, not another solve equation.
    return _project_cmdgfm_assembled!(
        velocity, pressure, solver, dt, workspace,
        assembled; tolerance, max_iterations, warm_start
    )
end

function _project_cmdgfm_assembled!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::CMDGFMProjectionSolver{T}, dt,
        workspace::CMDGFMProjectionWorkspace{T}, assembled;
        tolerance, max_iterations, warm_start,
    ) where {T}
    interface = solver.interface
    matrix = assembled.matrix
    right_hand_side = assembled.right_hand_side
    previous_increment = _cmdgfm_same_topology(workspace.increment_keys, interface.mesh) ?
        workspace.pressure_increment : nothing
    row_form = _cmdgfm_row_form!(workspace, matrix)
    asymmetry = _cmdgfm_matrix_asymmetry(
        matrix, row_form.matrix_transpose, workspace.reduction_partials,
    )
    initial_residual = norm(right_hand_side)
    solution = workspace.solution
    resize!(solution, length(right_hand_side))
    fill!(solution, zero(T))
    iterations = 0
    workspace.initial_guess_used = false
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        scaling = _cmdgfm_krylov_scaling!(
            workspace.scaling, assembled, maximum(solver.density), dt,
        )
        preconditioner = _cmdgfm_preconditioner(
            solver, assembled, workspace; scaling,
        )
        correction_rhs = workspace.correction_rhs
        resize!(correction_rhs, length(right_hand_side))
        copyto!(correction_rhs, right_hand_side)
        warm_start && _cmdgfm_initial_solution!(
            workspace, assembled, previous_increment, dt, preconditioner, row_form,
        )
        scaled_matrix = _ScaledProjectionOperator(row_form, scaling)
        scaled_preconditioner = if isnothing(preconditioner)
            nothing
        elseif solver.preconditioner === :block
            _ScaledProjectionPreconditioner(preconditioner, scaling)
        else
            preconditioner
        end
        # Residual replacement continues within the caller's iteration
        # budget. The eliminated gauge row must satisfy the same physical
        # divergence tolerance as the retained pressure equations.
        while iterations < max_iterations
            defects = _cmdgfm_residual_defects(
                correction_rhs, assembled, dt,
            )
            defect = max(
                maximum(defects),
                _cmdgfm_anchor_residual_defect(workspace, assembled, dt),
            )
            (!isfinite(defect) || defect <= T(tolerance)) && break
            scaled_rhs, scaling_rows = scaling.rhs, scaling.rows
            @_parallel for index in 1:length(scaled_rhs)
                @inbounds scaled_rhs[index] = scaling_rows[index] * correction_rhs[index]
            end
            rhs_norm = norm(scaling.rhs)
            (!isfinite(rhs_norm) || iszero(rhs_norm)) && break
            requested = min(T(tolerance), T(tolerance) / max(defect, eps(T)))
            # Krylov versions with an absolute machine-epsilon stop can
            # otherwise terminate a tiny correction before its relative
            # tolerance is reached. Restore the scale after the solve.
            rhs_scale = if rhs_norm < one(T) && rhs_norm * requested < eps(T)
                scaling.rhs ./= rhs_norm
                rhs_norm
            else
                one(T)
            end
            krylov = _cmdgfm_krylov_workspace!(
                workspace, scaled_matrix, scaling.rhs,
            )
            settings = (;
                atol = zero(T), rtol = requested, itmax = max_iterations - iterations,
            )
            if isnothing(preconditioner)
                Krylov.bicgstab!(
                    krylov, scaled_matrix, scaling.rhs; settings...,
                )
            else
                Krylov.bicgstab!(
                    krylov, scaled_matrix, scaling.rhs;
                    N = scaled_preconditioner, ldiv = true, settings...,
                )
            end
            increment = Krylov.solution(krylov)
            attempt_iterations = Krylov.iteration_count(krylov)
            iterations += attempt_iterations
            if !all(isfinite, increment) && !isnothing(preconditioner) &&
                    iterations < max_iterations
                # Count the failed attempt too; a retry receives only the
                # remaining budget and cannot corrupt the current solution.
                settings = merge(settings, (; itmax = max_iterations - iterations))
                Krylov.bicgstab!(
                    krylov, scaled_matrix, scaling.rhs; settings...,
                )
                increment = Krylov.solution(krylov)
                attempt_iterations = Krylov.iteration_count(krylov)
                iterations += attempt_iterations
            end
            (!all(isfinite, increment) || iszero(attempt_iterations)) && break
            _cmdgfm_apply_increment!(
                solution, scaling.columns, rhs_scale, increment, workspace.chunk_flags,
            ) || break
            _cmdgfm_residual!(correction_rhs, right_hand_side, row_form, solution)
        end
    end
    # `correction_rhs` is the true residual rhs - A*x after the last Krylov
    # correction. It is already in the reusable workspace.
    if isempty(solution) || iszero(initial_residual)
        resize!(workspace.correction_rhs, length(right_hand_side))
        copyto!(workspace.correction_rhs, right_hand_side)
    end
    final_residual = norm(workspace.correction_rhs)
    residual_defects = _cmdgfm_residual_defects(
        workspace.correction_rhs, assembled, dt,
    )

    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    increment = workspace.pressure_increment
    _resize_interface_cache!(increment)
    _copy_projection_cell_solution!(
        increment, zip(assembled.pressure_indices, view(solution, 1:np)),
    )
    if isnothing(previous_increment)
        resize!(workspace.increment_keys, length(workspace.pressure_cache.keys))
        copyto!(workspace.increment_keys, workspace.pressure_cache.keys)
    end
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.mixed_faces] .=
        view(solution, (np + 1):(np + nx))
    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    cmdgfm_gradient!(
        gradients, increment, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    _projection_correct_phase_velocities!(velocity, gradients, dt, T)
    # Keep the old physical pressure's gauge on each connected component and
    # preserve inactive cells. The increment is zero at anchors and solids.
    references = assembled.pressure_operator.references
    for (index, cell) in pairs(assembled.pressure_indices)
        _add!(pressure, solution[index], references[cell])
    end
    fill_halo!(pressure)

    _mark_inactive_phase_velocities!(velocity, interface, workspace.inactive_failures)
    mixture_constraint = _cmdgfm_cell_flux!(
        workspace.pressure_cache.cell_flux,
        assembled.data, interface, velocity
    )
    mixture_divergence = _projection_strong_maximum(
        mixture_constraint, assembled.pressure_operator.volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    target_mixture_defect = mixture_volume_defect
    if !iszero(assembled.inherited_weight) && !isnothing(assembled.inherited_cell_flux)
        target_scale = T(dt) * assembled.inherited_weight
        for cell in eachindex(mixture_constraint)
            mixture_constraint[cell] -= target_scale * assembled.inherited_cell_flux[cell]
        end
        target_mixture_defect = T(dt) * _projection_strong_maximum(
            mixture_constraint, assembled.pressure_operator.volumes,
        )
    end
    normal_velocity_defect = _cmdgfm_normal_velocity_defect(velocity, assembled.stencils, dt)
    momentum = zeros(T, 2)
    identity_error = zero(T)
    pressure_jump = assembled.pressure_operator.gradient_operator.pressure_jump
    for (index, face) in pairs(interface.mesh.faces)
        @inbounds begin
            liquid_mass = assembled.data.liquid_mass[index]
            gas_mass = assembled.data.gas_mass[index]
            identity_error = max(
                identity_error, _projection_momentum_identity_error(
                    liquid_mass, gas_mass, gradients[1].interior[index],
                    gradients[2].interior[index], workspace.pressure_gradient.interior[index],
                )
            )
            momentum[face.axis] -= T(dt) * pressure_jump[index]
        end
    end
    converged = residual_defects.pressure <= T(tolerance) &&
        residual_defects.jump <= T(tolerance) &&
        target_mixture_defect <= T(tolerance) &&
        normal_velocity_defect <= T(tolerance)
    workspace.matrix = matrix
    workspace.jump_block = assembled.jump_block
    workspace.right_hand_side = right_hand_side
    workspace.solution = solution
    workspace.pressure_indices = assembled.pressure_indices
    workspace.mixed_faces = assembled.mixed_faces
    workspace.stencils = assembled.stencils
    preconditioner_name = isnothing(preconditioner) ? :none :
        _preconditioner_name(preconditioner)
    pressure_hierarchy_reused = preconditioner isa
        CMDGFMBlockPreconditioner &&
        preconditioner.pressure_hierarchy_reused
    jump_factor_reused = preconditioner isa CMDGFMBlockPreconditioner &&
        preconditioner.jump_factor_reused
    return CMDGFMProjectionStats(
        converged, iterations, initial_residual, final_residual, asymmetry,
        mixture_divergence, mixture_volume_defect, normal_velocity_defect,
        T(tolerance), (momentum[1], momentum[2]), identity_error,
        _cmdgfm_jump_diagonal_margin(assembled.jump_block), np, nx,
        assembled.underresolved, preconditioner_name,
        pressure_hierarchy_reused, jump_factor_reused,
    )
end
