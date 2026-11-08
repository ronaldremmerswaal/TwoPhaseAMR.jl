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

"""Reusable fields and the most recently assembled CMDGFM system."""
mutable struct CMDGFMProjectionWorkspace{T, M <: BlockTreeMesh}
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
    pressure_cache::CMDGFMPressureCache{T, M}
    initial_guess_used::Bool
end

function CMDGFMProjectionWorkspace(
        solver::CMDGFMProjectionSolver{T},
    ) where {T}
    mesh = solver.interface.mesh
    return CMDGFMProjectionWorkspace{T, typeof(mesh)}(
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        FaceVelocity(mesh, T), FaceVelocity(mesh, T),
        zeros(T, length(mesh.faces)), FaceField{1}(mesh, T),
        FaceField{2}(mesh, T), spzeros(T, 0, 0), spzeros(T, 0, 0),
        T[], T[], Int[], Int[], CMDGFMJumpStencil{T}[], nothing,
        nothing, T[], CMDGFMPressureCache(T, typeof(mesh)), false,
    )
end

"""Diagnostics for one conservative multidimensional ghost-fluid projection."""
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
    nx = _get(interface.normal_x, reference) / spacing[1]
    ny = _get(interface.normal_y, reference) / spacing[2]
    magnitude = hypot(nx, ny)
    magnitude > sqrt(eps(T)) || return (zero(T), zero(T))
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
        face_normal = nothing, incident_faces = nothing, vertex_patches = nothing,
        storage = nothing,
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
        # The default remains CMDGFM's PLIC-based staggered normal. Mimetic
        # comparisons may supply another face normal without changing the
        # stencil geometry, beta rule, or CMDGFM itself.
        normal = isnothing(face_normal) ?
            _cmdgfm_face_normal(interface, face) : face_normal(face)
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
        shared_multigrid = nothing,
    ) where {T}
    interface = solver.interface
    mesh = interface.mesh
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
        solver, workspace, dt, velocity, cache,
        cache.operator, data, shared_multigrid
    )
end

function _assemble_cmdgfm_cached!(
        solver::CMDGFMProjectionSolver{T}, workspace,
        dt, velocity, cache, pressure_operator, data, shared_multigrid
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
    for (local_index, cell) in pairs(pressure_indices)
        right_hand_side[local_index] = cache.cell_flux[cell] / T(dt)
    end
    copyto!(right_hand_side, np + 1, jump_rhs, 1, length(jump_rhs))
    return (;
        data, pressure_operator, incidence, matrix, jump_block, jump_pressure,
        right_hand_side, pressure_indices, pressure_anchors, mixed_faces,
        stencils,
        pressure_positive, underresolved, shared_multigrid,
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
        solver::CMDGFMProjectionSolver{T}, assembled, workspace,
    ) where {T}
    specification = solver.preconditioner
    specification === nothing && return nothing
    if specification === :block
        return _cmdgfm_block_preconditioner(solver, assembled, workspace)
    end
    return _construct_matrix_preconditioner(
        specification, assembled.matrix;
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

function _cmdgfm_residual_defects(residual, assembled, dt)
    T = eltype(residual)
    np = length(assembled.pressure_indices)
    pressure = _projection_pressure_residual_maximum(
        residual, assembled.pressure_indices,
        assembled.pressure_operator.volumes, Val(true),
    )
    jump = isempty(assembled.mixed_faces) ? zero(T) :
        maximum(abs, view(residual, (np + 1):length(residual)))
    return (pressure = T(dt)^2 * pressure, jump = T(dt) * jump)
end

function _cmdgfm_initial_solution!(work, assembled, pressure, dt, preconditioner)
    # Guess only; no equation, metric, factorization or stopping tolerance is
    # changed. The zero solution is always available as the robust fallback.
    rhs, matrix = assembled.right_hand_side, assembled.matrix
    solution, residual = work.solution, work.correction_rhs
    candidate = work.pressure_cache.initial_guess
    resize!(candidate, length(rhs))
    np = length(assembled.pressure_indices)
    nx = length(assembled.mixed_faces)
    best_defect = maximum(_cmdgfm_residual_defects(rhs, assembled, dt))
    best_l2 = norm(rhs)
    used = false
    for previous_pressure in (false, true)
        fill!(candidate, zero(eltype(candidate)))
        if previous_pressure
            references = assembled.pressure_operator.references
            nonzero_pressure = false
            for index in 1:np
                cell = assembled.pressure_indices[index]
                anchor = assembled.pressure_anchors[index]
                value = _get(pressure, references[cell]) - _get(pressure, references[anchor])
                candidate[index] = value
                nonzero_pressure |= !iszero(value)
            end
            nonzero_pressure || continue
        end
        # A stale/nonfinite pressure field is not a usable guess. In
        # particular, do not feed it into the jump solve before rejecting it.
        all(isfinite, view(candidate, 1:np)) || continue
        if nx > 0
            preconditioner isa CMDGFMBlockPreconditioner || continue
            # For a chosen pressure, approximately satisfy the *existing*
            # jump row: J*xi = rhs_jump - jump_pressure*p. Reuse the current
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
        mul!(residual, matrix, candidate)
        residual .= rhs .- residual
        defect = maximum(_cmdgfm_residual_defects(residual, assembled, dt))
        l2 = norm(residual)
        # Raw L2 alone can hide a large local volume/jump error. Require an
        # improvement in both L2 and the combined physical maximum defect.
        if isfinite(defect) && isfinite(l2) &&
                defect < eltype(rhs)(0.95) * best_defect && l2 < best_l2
            copyto!(solution, candidate)
            best_defect, best_l2 = defect, l2
            used = true
        end
    end
    mul!(residual, matrix, solution)
    residual .= rhs .- residual
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

function _cmdgfm_normal_velocity_defect(
        velocity::TwoPhaseVelocity, stencils,
    )
    T = eltype(velocity[1])
    result = zero(T)
    liquid, gas = velocity
    for stencil in stencils
        face = stencil.face
        value = @inbounds gas.interior[face] - liquid.interior[face]
        stencil.resolved || begin
            result = max(result, abs(value))
            continue
        end
        factor = stencil.beta * stencil.ratio
        for (candidate, weight) in zip(
                stencil.transverse_faces, stencil.transverse_weights,
            )
            @inbounds value += factor * weight *
                (gas.interior[candidate] - liquid.interior[candidate])
        end
        if stencil.gas_inference_face > 0
            @inbounds value += (stencil.beta - one(T)) * (
                gas.interior[stencil.gas_inference_face] -
                    liquid.interior[stencil.liquid_inference_face]
            )
        end
        result = max(result, abs(value))
    end
    return result
end

"""
    project!(velocity, pressure, solver::CMDGFMProjectionSolver, dt, workspace;
             tolerance, max_iterations, warm_start=true)

Apply the conservative multidimensional ghost-fluid projection. The cell
pressure and mixed-face gradient jump form one nonsymmetric two-block system;
BiCGSTAB uses a tree-GMG pressure/active-jump-ILU block preconditioner by
default. A whole-system ILU remains available for comparison.
`warm_start` tries residual-checked pressure/jump guesses; `false` retains
the zero-start reference solve. The same final physical defects are required.
"""
function project!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::CMDGFMProjectionSolver{T}, dt,
        workspace::CMDGFMProjectionWorkspace{T};
        tolerance = sqrt(eps(T)), max_iterations = 500,
        _pressure_operator = nothing, _pressure_multigrid = nothing,
        warm_start = true,
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
    _mark_inactive_phase_velocities!(velocity, interface)
    assembled = _assemble_cmdgfm_system!(
        solver, workspace, dt, velocity;
        shared_operator = _pressure_operator, shared_multigrid = _pressure_multigrid
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
    asymmetry = _cmdgfm_matrix_asymmetry(matrix)
    initial_residual = norm(right_hand_side)
    solution = workspace.solution
    resize!(solution, length(right_hand_side))
    fill!(solution, zero(T))
    iterations = 0
    workspace.initial_guess_used = false
    preconditioner = nothing
    if !isempty(solution) && initial_residual > zero(T)
        preconditioner = _cmdgfm_preconditioner(
            solver, assembled, workspace,
        )
        correction_rhs = workspace.correction_rhs
        resize!(correction_rhs, length(right_hand_side))
        copyto!(correction_rhs, right_hand_side)
        warm_start && _cmdgfm_initial_solution!(
            workspace, assembled, pressure,
            dt, preconditioner
        )
        for _ in 1:8
            remaining = max_iterations - iterations
            remaining > 0 || break
            defects = _cmdgfm_residual_defects(
                correction_rhs, assembled, dt,
            )
            maximum(defects) <= T(tolerance) && break
            krylov = _cmdgfm_krylov_workspace!(
                workspace, matrix, correction_rhs,
            )
            requested = min(
                T(tolerance), T(tolerance) / max(maximum(defects), eps(T)),
            )
            settings = (;
                atol = zero(T), rtol = requested, itmax = remaining,
            )
            if isnothing(preconditioner)
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs; settings...,
                )
            else
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs;
                    N = preconditioner, ldiv = true, settings...,
                )
            end
            increment = Krylov.solution(krylov)
            if !all(isfinite, increment) && !isnothing(preconditioner)
                krylov = _cmdgfm_krylov_workspace!(
                    workspace, matrix, correction_rhs,
                )
                Krylov.bicgstab!(
                    krylov, matrix, correction_rhs; settings...,
                )
                increment = Krylov.solution(krylov)
            end
            solution .+= increment
            iterations += Krylov.iteration_count(krylov)
            mul!(correction_rhs, matrix, solution)
            correction_rhs .= right_hand_side .- correction_rhs
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
    _copy_projection_cell_solution!(
        pressure,
        zip(assembled.pressure_indices, view(solution, 1:np)),
    )
    fill!(workspace.jump, zero(T))
    workspace.jump[assembled.mixed_faces] .=
        view(solution, (np + 1):(np + nx))
    gradients = (workspace.liquid_gradient, workspace.gas_gradient)
    cmdgfm_gradient!(
        gradients, pressure, solver, workspace,
        assembled.pressure_operator, assembled.data,
    )
    _projection_correct_phase_velocities!(velocity, gradients, dt, T)

    _mark_inactive_phase_velocities!(velocity, interface)
    mixture_constraint = _cmdgfm_cell_flux!(
        workspace.pressure_cache.cell_flux,
        assembled.data, interface, velocity
    )
    mixture_divergence = _projection_strong_maximum(
        mixture_constraint, assembled.pressure_operator.volumes,
    )
    mixture_volume_defect = T(dt) * mixture_divergence
    normal_velocity_defect = _cmdgfm_normal_velocity_defect(
        velocity, assembled.stencils,
    )
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
        mixture_volume_defect <= T(tolerance) &&
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
