struct GeometricCell
    i::Int
    j::Int
    scale::Int
end

function _geometric_cells(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    cells = Vector{GeometricCell}(undef, ncells(mesh))
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        scale = 1 << (mesh.max_level - block.key.level)
        lower_i = (block.key.i * nx + i - 1) * scale
        lower_j = (block.key.j * ny + j - 1) * scale
        index = (block_index - 1) * nx * ny + (j - 1) * nx + i
        cells[index] = GeometricCell(lower_i, lower_j, scale)
    end
    return cells
end

@inline function _parent_key(cell::GeometricCell)
    parent_scale = 2cell.scale
    return (fld(cell.i, parent_scale) * parent_scale,
            fld(cell.j, parent_scale) * parent_scale,
            cell.scale)
end

function _complete_sibling_group(cells, indices)
    length(indices) == 4 || return false
    first_cell = cells[first(indices)]
    parent_i, parent_j, child_scale = _parent_key(first_cell)
    positions = Set{Tuple{Int,Int}}()
    for index in indices
        cell = cells[index]
        cell.scale == child_scale || return false
        push!(positions, (cell.i - parent_i, cell.j - parent_j))
    end
    return positions == Set(((0, 0), (child_scale, 0),
                             (0, child_scale), (child_scale, child_scale)))
end

"""Merge every complete, aligned 2-by-2 sibling group in one tree-cell pass."""
function _coarsen_geometric_cells(cells::Vector{GeometricCell}, ::Type{T}) where {T}
    candidates = Dict{Tuple{Int,Int,Int},Vector{Int}}()
    for (index, cell) in pairs(cells)
        push!(get!(candidates, _parent_key(cell), Int[]), index)
    end

    descriptors = Tuple{GeometricCell,Vector{Int}}[]
    merged = falses(length(cells))
    for key in sort!(collect(keys(candidates)); by=key -> (key[3], key[2], key[1]))
        indices = candidates[key]
        _complete_sibling_group(cells, indices) || continue
        parent_i, parent_j, child_scale = key
        push!(descriptors,
              (GeometricCell(parent_i, parent_j, 2child_scale), indices))
        merged[indices] .= true
    end
    for index in eachindex(cells)
        merged[index] && continue
        push!(descriptors, (cells[index], [index]))
    end
    sort!(descriptors; by=entry ->
        (entry[1].scale, entry[1].j, entry[1].i))

    coarse_cells = first.(descriptors)
    fine_to_coarse = Vector{Int}(undef, length(cells))
    for (coarse_index, (_, fine_indices)) in pairs(descriptors)
        for fine_index in fine_indices
            fine_to_coarse[fine_index] = coarse_index
        end
    end
    tentative = sparse(collect(eachindex(cells)), fine_to_coarse,
                       ones(T, length(cells)), length(cells), length(coarse_cells))
    return coarse_cells, tentative
end

function _jacobi_spectral_bound(matrix::SparseMatrixCSC{T}) where {T}
    diagonal = collect(diag(matrix))
    row_sums = zeros(T, size(matrix, 1))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        @inbounds row_sums[rows[pointer]] += abs(values[pointer])
    end
    bound = zero(T)
    for index in eachindex(diagonal)
        @inbounds diagonal[index] > zero(T) || continue
        @inbounds bound = max(bound, row_sums[index] / diagonal[index])
    end
    return bound
end

function _smoothed_prolongation(stiffness::SparseMatrixCSC{T},
                                tentative::SparseMatrixCSC{T}) where {T}
    diagonal = collect(diag(stiffness))
    any(>(zero(T)), diagonal) || return tentative
    inverse_diagonal = map(value -> value > zero(T) ? inv(value) : zero(T), diagonal)
    spectral_bound = _jacobi_spectral_bound(stiffness)
    iszero(spectral_bound) && return tentative
    weight = T(4) / (T(3) * spectral_bound)
    prolongation = tentative -
                   weight * spdiagm(0 => inverse_diagonal) * stiffness * tentative
    dropzeros!(prolongation)
    return prolongation
end

mutable struct MultigridLevel{T}
    matrix::SparseMatrixCSC{T,Int}
    stiffness::SparseMatrixCSC{T,Int}
    volumes::Vector{T}
    prolongation::SparseMatrixCSC{T,Int}
    restriction::SparseMatrixCSC{T,Int}
    inverse_diagonal::Vector{T}
    relaxation_weight::T
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    temporary::Vector{T}
end

function _multigrid_level(matrix::SparseMatrixCSC{T,Int},
                          stiffness::SparseMatrixCSC{T,Int},
                          volumes::Vector{T},
                          prolongation::SparseMatrixCSC{T,Int}) where {T}
    diagonal = collect(diag(matrix))
    all(>(zero(T)), diagonal) ||
        throw(ArgumentError("elliptic matrix has a non-positive diagonal"))
    inverse_diagonal = inv.(diagonal)
    spectral_bound = _jacobi_spectral_bound(matrix)
    relaxation_weight = iszero(spectral_bound) ? one(T) :
                        min(one(T), T(4) / (T(3) * spectral_bound))
    restriction = sparse(transpose(prolongation))
    number_of_cells = size(matrix, 1)
    return MultigridLevel(
        matrix, stiffness, volumes, prolongation, restriction,
        inverse_diagonal, relaxation_weight,
        zeros(T, number_of_cells), zeros(T, number_of_cells),
        zeros(T, number_of_cells), zeros(T, number_of_cells),
    )
end

mutable struct LinearSolverStats{T}
    cycles::Int
    converged::Bool
    initial_l2::T
    final_l2::T
    final_linf::T
    removed_rhs_mean::T
    method::Symbol
    preconditioner::Symbol
end

# Preserve the public name used by earlier releases.
const MultigridStats = LinearSolverStats

LinearSolverStats(::Type{T}; method=:none, preconditioner=:none) where {T} =
    LinearSolverStats(0, false, T(Inf), T(Inf), T(Inf), zero(T),
                      method, preconditioner)

abstract type AbstractGeometricMultigridPreconditioner{T} end

"""
    GeometricMultigridPreconditioner(operator; coarse_size=64,
                                     pre_sweeps=2, post_sweeps=2)

Build a geometric/Galerkin V-cycle preconditioner for a `BlockTreeMesh`. Each
coarsening pass merges complete dyadic 2-by-2 sibling cells, including cells
inside blocks and cells reconstructed across former block boundaries.  Cells
beside a refinement transition remain at their current scale until their
neighbours have caught up.  This is a tree-geometric coarsening, not graph
aggregation.

The tentative piecewise-constant transfer is smoothed once with the diffusion
operator.  It still reproduces constants exactly, but substantially improves
low-frequency correction across block and refinement interfaces.  Coarse
operators are Galerkin products, so symmetry and the constant Poisson nullspace
are inherited on every level. The object implements three-argument `ldiv!`
and is entirely agnostic to the Krylov method. It owns its V-cycle work vectors
and is not safe for concurrent preconditioner applications.
"""
struct GeometricMultigridPreconditioner{T,O<:EllipticOperator{T},F} <:
       AbstractGeometricMultigridPreconditioner{T}
    operator::O
    levels::Vector{MultigridLevel{T}}
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
end

mutable struct StaggeredMultigridComponent{T,F} <:
               AbstractGeometricMultigridPreconditioner{T}
    levels::Vector{MultigridLevel{T}}
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
end

function GeometricMultigridPreconditioner(operator::O; coarse_size=64,
                                          pre_sweeps=2, post_sweeps=2,
                                          max_levels=32) where
        {T,O<:EllipticOperator{T}}
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    matrices = SparseMatrixCSC{T,Int}[operator.matrix]
    stiffnesses = SparseMatrixCSC{T,Int}[operator.stiffness]
    volume_vectors = Vector{T}[operator.volumes]
    prolongations = SparseMatrixCSC{T,Int}[]
    cells = _geometric_cells(operator.mesh)

    while length(cells) > coarse_size && length(matrices) < max_levels
        coarse_cells, tentative = _coarsen_geometric_cells(cells, T)
        length(coarse_cells) < length(cells) || break
        prolongation = _smoothed_prolongation(last(stiffnesses), tentative)
        coarse_matrix = sparse(transpose(prolongation) *
                               last(matrices) * prolongation)
        coarse_stiffness = sparse(transpose(prolongation) *
                                  last(stiffnesses) * prolongation)
        coarse_matrix = sparse((coarse_matrix + transpose(coarse_matrix)) / 2)
        coarse_stiffness = sparse(
            (coarse_stiffness + transpose(coarse_stiffness)) / 2)
        coarse_volumes = vec(transpose(prolongation) * last(volume_vectors))
        all(>(zero(T)), coarse_volumes) ||
            throw(ArgumentError("coarsening produced a non-positive cell volume"))
        push!(prolongations, prolongation)
        push!(matrices, coarse_matrix)
        push!(stiffnesses, coarse_stiffness)
        push!(volume_vectors, coarse_volumes)
        cells = coarse_cells
    end

    levels = MultigridLevel{T}[]
    for level_index in eachindex(matrices)
        prolongation = level_index < length(matrices) ?
            prolongations[level_index] : spzeros(T, size(matrices[level_index], 1), 0)
        push!(levels, _multigrid_level(
            matrices[level_index], stiffnesses[level_index],
            volume_vectors[level_index], prolongation))
    end

    coarse = last(levels)
    singular = iszero(operator.reaction)
    if singular
        number_of_cells = length(coarse.volumes)
        augmented = zeros(T, number_of_cells + 1, number_of_cells + 1)
        augmented[1:number_of_cells, 1:number_of_cells] .= Matrix(coarse.matrix)
        augmented[1:number_of_cells, end] .= coarse.volumes
        augmented[end, 1:number_of_cells] .= coarse.volumes
        factorization = lu(augmented)
        coarse_rhs = zeros(T, number_of_cells + 1)
        coarse_solution = similar(coarse_rhs)
    else
        factorization = lu(Matrix(coarse.matrix))
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    return GeometricMultigridPreconditioner(
        operator, levels, factorization, singular, coarse_rhs,
        coarse_solution, pre_sweeps, post_sweeps)
end

# Compatibility name from before Krylov iteration and multigrid were split.
const GeometricMultigridSolver = GeometricMultigridPreconditioner

struct GeometricFacePoint
    i::Int
    j::Int
end

function _geometric_face_points(mesh::BlockTreeMesh, local_to_global)
    nx, ny = mesh.block_size
    finest_nx = nx << mesh.max_level
    finest_ny = ny << mesh.max_level
    inverse_hx = finest_nx / mesh.extent[1]
    inverse_hy = finest_ny / mesh.extent[2]
    points = Vector{GeometricFacePoint}(undef, length(local_to_global))
    for local_index in eachindex(local_to_global)
        face = mesh.faces[local_to_global[local_index]]
        # Twice-finest-cell coordinates make both normal-face and tangential
        # half-cell locations integral on every tree level.
        i = round(Int, 2(face.center[1] - mesh.origin[1]) * inverse_hx)
        j = round(Int, 2(face.center[2] - mesh.origin[2]) * inverse_hy)
        points[local_index] = GeometricFacePoint(i, j)
    end
    return points
end

"""Aggregate one staggered component in aligned dyadic geometric bins."""
function _coarsen_geometric_faces(points::Vector{GeometricFacePoint},
                                   bin_width::Int, ::Type{T}) where {T}
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for (index, point) in pairs(points)
        key = (fld(point.i, bin_width), fld(point.j, bin_width))
        push!(get!(groups, key, Int[]), index)
    end
    keys_in_order = sort!(collect(keys(groups)); by=key -> (key[2], key[1]))
    coarse_points = Vector{GeometricFacePoint}(undef, length(keys_in_order))
    fine_to_coarse = Vector{Int}(undef, length(points))
    for (coarse_index, key) in pairs(keys_in_order)
        fine_indices = groups[key]
        coarse_points[coarse_index] = GeometricFacePoint(
            minimum(index -> points[index].i, fine_indices),
            minimum(index -> points[index].j, fine_indices))
        for fine_index in fine_indices
            fine_to_coarse[fine_index] = coarse_index
        end
    end
    tentative = sparse(collect(eachindex(points)), fine_to_coarse,
                       ones(T, length(points)), length(points),
                       length(coarse_points))
    return coarse_points, tentative
end

function _staggered_multigrid_component(
        mesh::BlockTreeMesh,
        diffusion_component::StaggeredDiffusionComponent{Axis,T},
        elliptic_component::StaggeredEllipticComponent{Axis,T};
        coarse_size=64, pre_sweeps=2, post_sweeps=2,
        max_levels=32) where {Axis,T}
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))

    matrices = SparseMatrixCSC{T,Int}[elliptic_component.matrix]
    stiffnesses = SparseMatrixCSC{T,Int}[diffusion_component.stiffness]
    volume_vectors = Vector{T}[elliptic_component.volumes]
    prolongations = SparseMatrixCSC{T,Int}[]
    points = _geometric_face_points(mesh, elliptic_component.local_to_global)
    bin_width = 4

    # The first bins span two finest cells in each direction. Coarser tree
    # faces naturally remain singletons until the dyadic bin catches their
    # scale, which mirrors the delayed coarsening used by the primal hierarchy.
    while length(points) > coarse_size && length(matrices) < max_levels
        coarse_points, tentative = _coarsen_geometric_faces(
            points, bin_width, T)
        bin_width *= 2
        length(coarse_points) < length(points) || continue
        prolongation = _smoothed_prolongation(last(stiffnesses), tentative)
        coarse_matrix = sparse(transpose(prolongation) *
                               last(matrices) * prolongation)
        coarse_stiffness = sparse(transpose(prolongation) *
                                  last(stiffnesses) * prolongation)
        # Only the stable diffusion core is symmetric. The widened, consistent
        # AMR correction in `matrix` is mildly nonsymmetric and must not be
        # replaced by its symmetric part on coarse levels.
        coarse_stiffness = sparse(
            (coarse_stiffness + transpose(coarse_stiffness)) / 2)
        coarse_volumes = vec(transpose(prolongation) * last(volume_vectors))
        all(>(zero(T)), coarse_volumes) ||
            throw(ArgumentError(
                "staggered coarsening produced a non-positive dual volume"))
        push!(prolongations, prolongation)
        push!(matrices, coarse_matrix)
        push!(stiffnesses, coarse_stiffness)
        push!(volume_vectors, coarse_volumes)
        points = coarse_points
    end

    levels = MultigridLevel{T}[]
    for level_index in eachindex(matrices)
        prolongation = level_index < length(matrices) ?
            prolongations[level_index] :
            spzeros(T, size(matrices[level_index], 1), 0)
        push!(levels, _multigrid_level(
            matrices[level_index], stiffnesses[level_index],
            volume_vectors[level_index], prolongation))
    end
    coarse = last(levels)
    factorization = lu(Matrix(coarse.matrix))
    coarse_rhs = zeros(T, length(coarse.volumes))
    coarse_solution = similar(coarse_rhs)
    return StaggeredMultigridComponent(
        levels, factorization, false, coarse_rhs, coarse_solution,
        pre_sweeps, post_sweeps)
end

function _remove_volume_mean!(values, volumes)
    mean_value = dot(values, volumes) / sum(volumes)
    for index in eachindex(values)
        @inbounds values[index] -= mean_value
    end
    return mean_value
end

function _project_integrated_rhs!(right_hand_side, volumes)
    mean_value = sum(right_hand_side) / sum(volumes)
    for index in eachindex(right_hand_side)
        @inbounds right_hand_side[index] -= mean_value * volumes[index]
    end
    return mean_value
end

function _smooth!(level::MultigridLevel, sweeps, singular)
    for _ in 1:sweeps
        mul!(level.temporary, level.matrix, level.solution)
        for index in eachindex(level.solution)
            @inbounds level.solution[index] += level.relaxation_weight *
                level.inverse_diagonal[index] *
                (level.right_hand_side[index] - level.temporary[index])
        end
        singular && _remove_volume_mean!(level.solution, level.volumes)
    end
    return level.solution
end

function _coarse_solve!(solver::AbstractGeometricMultigridPreconditioner)
    coarse = last(solver.levels)
    number_of_cells = length(coarse.solution)
    if solver.coarse_augmented
        copyto!(solver.coarse_rhs, 1, coarse.right_hand_side, 1, number_of_cells)
        solver.coarse_rhs[end] = zero(eltype(solver.coarse_rhs))
        ldiv!(solver.coarse_solution, solver.coarse_factorization,
              solver.coarse_rhs)
        copyto!(coarse.solution, 1, solver.coarse_solution, 1, number_of_cells)
    else
        ldiv!(coarse.solution, solver.coarse_factorization,
              coarse.right_hand_side)
    end
    return coarse.solution
end

function _vcycle!(solver::AbstractGeometricMultigridPreconditioner, level_index::Int)
    level = solver.levels[level_index]
    if level_index == length(solver.levels)
        return _coarse_solve!(solver)
    end

    singular = solver.coarse_augmented
    _smooth!(level, solver.pre_sweeps, singular)
    mul!(level.temporary, level.matrix, level.solution)
    for index in eachindex(level.residual)
        @inbounds level.residual[index] =
            level.right_hand_side[index] - level.temporary[index]
    end

    coarse = solver.levels[level_index + 1]
    mul!(coarse.right_hand_side, level.restriction, level.residual)
    singular && _project_integrated_rhs!(coarse.right_hand_side, coarse.volumes)
    fill!(coarse.solution, zero(eltype(coarse.solution)))
    _vcycle!(solver, level_index + 1)
    mul!(level.temporary, level.prolongation, coarse.solution)
    for index in eachindex(level.solution)
        @inbounds level.solution[index] += level.temporary[index]
    end
    _smooth!(level, solver.post_sweeps, singular)
    return level.solution
end

function _residual_norms!(level::MultigridLevel)
    mul!(level.temporary, level.matrix, level.solution)
    squared = zero(eltype(level.solution))
    maximum_norm = zero(eltype(level.solution))
    total_volume = sum(level.volumes)
    for index in eachindex(level.residual)
        @inbounds integrated = level.right_hand_side[index] - level.temporary[index]
        @inbounds level.residual[index] = integrated
        @inbounds strong = integrated / level.volumes[index]
        @inbounds squared += level.volumes[index] * strong^2
        maximum_norm = max(maximum_norm, abs(strong))
    end
    return sqrt(squared / total_volume), maximum_norm
end

function _roundoff_residual_floor!(matrix::SparseMatrixCSC{T}, solution,
                                   right_hand_side, volumes,
                                   temporary) where {T}
    fill!(temporary, zero(T))
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        row = rows[pointer]
        @inbounds temporary[row] +=
            abs(values[pointer]) * abs(solution[column])
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds scale = max(
            scale,
            temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index]))
    end
    return T(128) * eps(T) * scale
end

_roundoff_residual_floor!(level::MultigridLevel) =
    _roundoff_residual_floor!(
        level.matrix, level.solution, level.right_hand_side,
        level.volumes, level.temporary)

function _integrated_residual_norms(residual, volumes)
    T = eltype(residual)
    squared = zero(T)
    maximum_norm = zero(T)
    total_volume = zero(T)
    for index in eachindex(residual)
        @inbounds strong = residual[index] / volumes[index]
        @inbounds squared += volumes[index] * strong^2
        maximum_norm = max(maximum_norm, abs(strong))
        @inbounds total_volume += volumes[index]
    end
    return sqrt(squared / total_volume), maximum_norm
end

function _apply_vcycle_preconditioner!(destination,
                                       solver::AbstractGeometricMultigridPreconditioner,
                                       right_hand_side)
    fine = first(solver.levels)
    copyto!(fine.right_hand_side, right_hand_side)
    fill!(fine.solution, zero(eltype(fine.solution)))
    _vcycle!(solver, 1)
    copyto!(destination, fine.solution)
    return destination
end

function ldiv!(destination::AbstractVector,
               solver::AbstractGeometricMultigridPreconditioner,
               right_hand_side::AbstractVector)
    return _apply_vcycle_preconditioner!(
        destination, solver, right_hand_side)
end

Base.size(solver::AbstractGeometricMultigridPreconditioner) =
    size(first(solver.levels).matrix)
Base.eltype(::AbstractGeometricMultigridPreconditioner{T}) where {T} = T

"""A named wrapper around an external preconditioner implementing `ldiv!`."""
struct ExternalPreconditioner{T,P}
    name::Symbol
    value::P
    dimensions::Tuple{Int,Int}
end

ExternalPreconditioner(name, value, matrix) =
    ExternalPreconditioner{eltype(matrix),typeof(value)}(
        name, value, size(matrix))

function ldiv!(destination::AbstractVector,
               preconditioner::ExternalPreconditioner,
               right_hand_side::AbstractVector)
    ldiv!(destination, preconditioner.value, right_hand_side)
end

Base.size(preconditioner::ExternalPreconditioner) = preconditioner.dimensions
Base.eltype(::ExternalPreconditioner{T}) where {T} = T

_preconditioner_name(::Nothing) = :none
_preconditioner_name(::AbstractGeometricMultigridPreconditioner) = :geometric_multigrid
_preconditioner_name(preconditioner::ExternalPreconditioner) = preconditioner.name
_preconditioner_name(preconditioner) = Symbol(nameof(typeof(preconditioner)))

function _optional_extension(name::Symbol, package::String)
    extension = Base.get_extension(@__MODULE__, name)
    isnothing(extension) && throw(ArgumentError(
        "$package is an optional dependency; run `using $package` before " *
        "constructing this preconditioner"))
    return extension
end

"""
    algebraic_multigrid_preconditioner(matrix; symmetric=true,
                                       singular=false, kwargs...)

Construct an AlgebraicMultigrid.jl preconditioner. Load
`AlgebraicMultigrid` before calling this optional integration.
"""
function algebraic_multigrid_preconditioner(matrix; kwargs...)
    extension = _optional_extension(
        :TwoPhaseAMRAlgebraicMultigridExt, "AlgebraicMultigrid")
    return ExternalPreconditioner(
        :algebraic_multigrid,
        extension.build_algebraic_multigrid_preconditioner(matrix; kwargs...),
        matrix)
end

"""
    incomplete_lu_preconditioner(matrix; drop_tolerance=1e-3)

Construct an IncompleteLU.jl Crout ILU preconditioner. Load `IncompleteLU`
before calling this optional integration.
"""
function incomplete_lu_preconditioner(matrix; kwargs...)
    extension = _optional_extension(:TwoPhaseAMRIncompleteLUExt, "IncompleteLU")
    return ExternalPreconditioner(
        :ilu, extension.build_incomplete_lu_preconditioner(matrix; kwargs...),
        matrix)
end

"""
    incomplete_cholesky_preconditioner(matrix; kwargs...)

Construct a LimitedLDLFactorizations.jl limited-memory incomplete LDLᵀ
preconditioner. For a positive (semi-)definite matrix this is the symmetric
incomplete-Cholesky option. Load `LimitedLDLFactorizations` before calling it.
"""
function incomplete_cholesky_preconditioner(matrix; kwargs...)
    extension = _optional_extension(
        :TwoPhaseAMRLimitedLDLFactorizationsExt,
        "LimitedLDLFactorizations")
    return ExternalPreconditioner(
        :incomplete_cholesky,
        extension.build_incomplete_cholesky_preconditioner(matrix; kwargs...),
        matrix)
end

function _construct_krylov(krylov, matrix, right_hand_side)
    workspace = applicable(krylov, matrix, right_hand_side) ?
                krylov(matrix, right_hand_side) : krylov
    size(matrix) == (workspace.m, workspace.n) ||
        throw(DimensionMismatch(
            "Krylov workspace and elliptic matrix sizes differ"))
    return workspace
end

function _construct_matrix_preconditioner(preconditioner, matrix;
                                          symmetric, singular,
                                          options=(;))
    preconditioner === nothing && return nothing
    if preconditioner === :amg
        return algebraic_multigrid_preconditioner(
            matrix; symmetric, singular, options...)
    elseif preconditioner === :ilu
        return incomplete_lu_preconditioner(matrix; options...)
    elseif preconditioner === :ichol
        symmetric || throw(ArgumentError(
            "incomplete Cholesky requires a symmetric operator"))
        return incomplete_cholesky_preconditioner(matrix; options...)
    elseif preconditioner isa Symbol
        throw(ArgumentError(
            "preconditioner must be :geometric, :amg, :ilu, :ichol, " *
            "nothing, a factory, or a pre-built preconditioner"))
    elseif applicable(preconditioner, matrix)
        isempty(options) || throw(ArgumentError(
            "preconditioner_options cannot accompany a preconditioner factory"))
        return preconditioner(matrix)
    end
    isempty(options) || throw(ArgumentError(
        "preconditioner_options cannot accompany a pre-built preconditioner"))
    return preconditioner
end

function _construct_pressure_preconditioner(preconditioner,
                                            operator::EllipticOperator;
                                            coarse_size, pre_sweeps,
                                            post_sweeps, max_levels,
                                            options=(;))
    if preconditioner === :geometric
        settings = merge((; coarse_size, pre_sweeps, post_sweeps, max_levels),
                         options)
        return GeometricMultigridPreconditioner(operator; settings...)
    end
    return _construct_matrix_preconditioner(
        preconditioner, operator.matrix;
        symmetric=true, singular=iszero(operator.reaction), options)
end

"""
    EllipticSolver(operator, krylov, preconditioner)
    EllipticSolver(operator; krylov=Krylov.CgWorkspace,
                   preconditioner=:geometric, kwargs...)

Couple an elliptic operator to an external Krylov.jl workspace and an
independent preconditioner. The default is Krylov.jl CG with this package's
geometric multigrid V-cycle. A pre-built external Krylov workspace and any
preconditioner implementing three-argument `ldiv!` may be passed directly.
"""
mutable struct EllipticSolver{T,O,K,P}
    operator::O
    krylov::K
    preconditioner::P
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end

function EllipticSolver(operator::O, krylov, preconditioner) where
        {T,O<:EllipticOperator{T}}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"))
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, krylov, preconditioner,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        LinearSolverStats(T; method,
                          preconditioner=_preconditioner_name(preconditioner)))
end

function EllipticSolver(operator::StaggeredEllipticComponent{Axis,T},
                        krylov, preconditioner) where {Axis,T}
    number_of_unknowns = length(operator.volumes)
    (krylov.m, krylov.n) == (number_of_unknowns, number_of_unknowns) ||
        throw(DimensionMismatch(
            "Krylov workspace and elliptic operator sizes differ"))
    method = _krylov_method(krylov)
    return EllipticSolver(
        operator, krylov, preconditioner,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        LinearSolverStats(T; method,
                          preconditioner=_preconditioner_name(preconditioner)))
end

function EllipticSolver(operator::EllipticOperator;
                        krylov=Krylov.CgWorkspace,
                        preconditioner=:geometric,
                        preconditioner_options=(;),
                        coarse_size=64, pre_sweeps=2, post_sweeps=2,
                        max_levels=32)
    right_hand_side = zeros(eltype(operator.volumes), length(operator.volumes))
    workspace = _construct_krylov(krylov, operator.matrix, right_hand_side)
    built_preconditioner = _construct_pressure_preconditioner(
        preconditioner, operator; coarse_size, pre_sweeps, post_sweeps,
        max_levels, options=preconditioner_options)
    return EllipticSolver(operator, workspace, built_preconditioner)
end

_krylov_method(::Krylov.CgWorkspace) = :cg
_krylov_method(::Krylov.BicgstabWorkspace) = :bicgstab
_krylov_method(workspace) = Symbol(nameof(typeof(workspace)))

struct StrongResidualCallback{T,V}
    target::T
    volumes::V
end

function (callback::StrongResidualCallback)(workspace)
    _, maximum_norm = _integrated_residual_norms(
        workspace.r, callback.volumes)
    return maximum_norm <= callback.target
end

function _run_krylov!(workspace::Krylov.CgWorkspace, matrix,
                      right_hand_side, initial_solution, preconditioner,
                      target, volumes, max_iterations)
    callback = StrongResidualCallback(target, volumes)
    if isnothing(preconditioner)
        Krylov.cg!(workspace, matrix, right_hand_side, initial_solution;
                   atol=zero(target), rtol=zero(target),
                   itmax=max_iterations, callback)
    else
        Krylov.cg!(workspace, matrix, right_hand_side, initial_solution;
                   M=preconditioner, ldiv=true,
                   atol=zero(target), rtol=zero(target),
                   itmax=max_iterations, callback)
    end
    return workspace
end

function _run_krylov!(workspace::Krylov.BicgstabWorkspace, matrix,
                      right_hand_side, initial_solution, preconditioner,
                      target, volumes, max_iterations)
    callback = StrongResidualCallback(target, volumes)
    if isnothing(preconditioner)
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            atol=zero(target), rtol=zero(target),
            itmax=max_iterations, callback)
    else
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            N=preconditioner, ldiv=true,
            atol=zero(target), rtol=zero(target),
            itmax=max_iterations, callback)
    end
    # Krylov.jl's BiCGSTAB currently evaluates omega before testing the
    # intermediate residual.  An (almost) exact preconditioner can therefore
    # produce 0/0 for a right-hand side contained in one eigenspace.  Retry the
    # same external method without the preconditioner only for that breakdown;
    # `initial_solution` has not been mutated by `bicgstab!`.
    if !all(isfinite, Krylov.solution(workspace)) &&
       !isnothing(preconditioner)
        Krylov.bicgstab!(
            workspace, matrix, right_hand_side, initial_solution;
            atol=zero(target), rtol=zero(target),
            itmax=max_iterations, callback)
    end
    return workspace
end

function _run_krylov!(workspace, matrix, right_hand_side, initial_solution,
                      preconditioner, target, volumes, max_iterations)
    throw(ArgumentError(
        "unsupported Krylov workspace $(typeof(workspace)); extend " *
        "TwoPhaseAMR._run_krylov! for this external solver"))
end

function _true_residual!(solver::EllipticSolver)
    mul!(solver.product, solver.operator.matrix, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    return solver.residual
end

"""
    solve!(solution, solver, rhs; rtol=1e-10, atol=0,
           max_cycles=100, compatibility=:project)

Solve the strong-form equation `L(solution) = rhs` with the external Krylov
workspace and independent preconditioner stored in `solver`.
For periodic or all-Neumann Poisson problems the right-hand side must have zero
volume mean.
The default `compatibility=:project` removes and reports its discrete mean;
use `:error` to reject an incompatible right-hand side.  The returned pressure
always has zero volume-weighted mean, which fixes the constant nullspace.
"""
function solve!(solution::CellField{T}, solver::EllipticSolver{T},
                right_hand_side::CellField{T}; rtol=1e-10, atol=0,
                max_cycles=100, compatibility=:project) where {T}
    solution.mesh === right_hand_side.mesh === solver.operator.mesh ||
        throw(ArgumentError("fields and multigrid solver must share a mesh"))
    zero(T) <= rtol < one(T) || throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 || throw(ArgumentError("max_cycles must be nonnegative"))
    compatibility in (:project, :error) ||
        throw(ArgumentError("compatibility must be :project or :error"))

    _copy_from_field!(solver.solution, solution, solver.operator.references)
    _copy_from_field!(solver.right_hand_side, right_hand_side,
                      solver.operator.references)
    volumes = solver.operator.volumes
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] *= volumes[index]
    end

    removed_mean = zero(T)
    singular = iszero(solver.operator.reaction)
    if singular
        integrated_sum = sum(solver.right_hand_side)
        scale = sum(abs, solver.right_hand_side)
        tolerance = T(256) * eps(T) * max(one(T), scale)
        if compatibility === :error && abs(integrated_sum) > tolerance
            throw(ArgumentError(
                "periodic Poisson right-hand side has nonzero volume mean"))
        end
        removed_mean = _project_integrated_rhs!(solver.right_hand_side, volumes)
        _remove_volume_mean!(solver.solution, volumes)
    end

    _true_residual!(solver)
    initial_l2, initial_linf =
        _integrated_residual_norms(solver.residual, volumes)
    roundoff_floor = _roundoff_residual_floor!(
        solver.operator.matrix, solver.solution, solver.right_hand_side,
        volumes, solver.product)
    target = max(T(atol), T(rtol) * max(initial_linf, eps(T)),
                 roundoff_floor)
    if initial_linf > target
        _run_krylov!(
            solver.krylov, solver.operator.matrix,
            solver.right_hand_side, solver.solution, solver.preconditioner,
            target, volumes, max_cycles)
        copyto!(solver.solution, Krylov.solution(solver.krylov))
    end
    singular && _remove_volume_mean!(solver.solution, volumes)
    _true_residual!(solver)
    final_l2, final_linf =
        _integrated_residual_norms(solver.residual, volumes)
    _copy_to_field!(solution, solver.solution, solver.operator.references)

    stats = solver.stats
    stats.cycles = initial_linf <= target ? 0 :
                   Krylov.iteration_count(solver.krylov)
    stats.converged = final_linf <= target
    stats.initial_l2 = initial_l2
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.removed_rhs_mean = removed_mean
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""Return `(L2, LInf)` strong residual norms from the most recent solve."""
residual_norms(solver::EllipticSolver) =
    (solver.stats.final_l2, solver.stats.final_linf)

struct ProjectionWorkspace{T,M<:BlockTreeMesh}
    divergence::CellField{T,M}
    weighted_gradient::FaceVelocity{T,M}
    elliptic::EllipticWorkspace{T}
end

function ProjectionWorkspace(solver::EllipticSolver{T}) where {T}
    mesh = solver.operator.mesh
    divergence = CellField(mesh, T)
    weighted_gradient = FaceVelocity(
        mesh, (x, y, time, axis) -> zero(T); T)
    return ProjectionWorkspace(divergence, weighted_gradient,
                               EllipticWorkspace(solver.operator))
end

"""
    project!(velocity, pressure, solver, dt, workspace; solve_options...)

Apply the pressure projection using exactly the weighted gradient in the
solver's Poisson operator. Physical face velocities are first set to zero; the
corresponding pressure correction has homogeneous Neumann data. The pressure is overwritten by the mean-zero
solution of `-diffusion*D*G_kappa*p = -D*u/dt`, then
`u <- u - dt*diffusion*G_kappa*p`.
"""
function project!(velocity::FaceVelocity{T}, pressure::CellField{T},
                  solver::EllipticSolver{T}, dt,
                  workspace::ProjectionWorkspace{T}; kwargs...) where {T}
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh ||
        throw(ArgumentError("projection arguments must share a mesh"))
    iszero(solver.operator.reaction) ||
        throw(ArgumentError("pressure projection requires a zero-reaction operator"))
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))

    enforce_no_permeation!(velocity)
    divergence!(workspace.divergence, velocity)
    nx, ny = velocity.mesh.block_size
    for block in eachindex(velocity.mesh.blocks), j in 1:ny, i in 1:nx
        reference = CellRef(block, i, j)
        _set!(workspace.divergence,
              -_get(workspace.divergence, reference) / dt, reference)
    end
    stats = solve!(pressure, solver, workspace.divergence; kwargs...)
    weighted_gradient!(workspace.weighted_gradient, pressure,
                       solver.operator, workspace.elliptic)
    scale = T(dt) * solver.operator.diffusion
    for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -=
            scale * workspace.weighted_gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -=
            scale * workspace.weighted_gradient.boundary[index]
    end
    enforce_no_permeation!(velocity)
    return stats
end
