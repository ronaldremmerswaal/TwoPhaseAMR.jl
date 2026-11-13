"""Coupled staggered `div(2μD(u))` with cell-centred normal stresses and vertex
shear stresses. The integrated stiffness is `E'WE`, so viscosity dissipates
kinetic energy for positive μ. Density enters the time-step mass matrix.

`dirichlet_faces` contains prescribed zero global face indices. Their original
physical strain columns are stored in `wall_reaction_strain`, with the same
stress rows as `strain` and columns ordered like `dirichlet_faces`; its
algebraic constraint rows are zero. For retained velocities `u`, the integrated
force on those faces is `-wall_reaction_strain' * (stress_weights .* (strain * u))`,
using the current stress weights.
"""
mutable struct VariableStaggeredDiffusionOperator{T, M, B}
    mesh::M
    stiffness::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    boundary::B
    averaging::AbstractViscosityAveraging
    strain::SparseMatrixCSC{T, Int}
    strain_transpose::SparseMatrixCSC{T, Int}
    wall_reaction_strain::SparseMatrixCSC{T, Int}
    dirichlet_faces::Vector{Int}
    stress_interpolation::SparseMatrixCSC{T, Int}
    stress_interpolation_transpose::SparseMatrixCSC{T, Int}
    stress_measures::Vector{T}
    stress_weights::Vector{T}
    cell_work::Vector{T}
    cell_inverse_work::Vector{T}
    stress_work::Vector{T}
    stiffness_pointers::Vector{Int}
    stiffness_stresses::Vector{Int}
    stiffness_factors::Vector{T}
    stiffness_refresh_offsets::Vector{Int}
end

const _MAXIMUM_SHEAR_TEMPLATE_POINTS = 16

struct ShearTemplatePoint{T}
    boundary::Bool
    dx::T
    dy::T
end

struct ShearTemplateKey{T}
    component::UInt8
    boundary_kind::UInt8
    count::UInt8
    points::NTuple{_MAXIMUM_SHEAR_TEMPLATE_POINTS, ShearTemplatePoint{T}}
end

"""Stable geometric identifier for an owned staggered face."""
struct DiffusionFaceKey
    axis::UInt8
    normal::Int
    tangent_low::Int
    tangent_high::Int
end

struct ShearPatchKey
    vertex::NTuple{2, Int}
    component::UInt8
    boundary_kind::UInt8
end

struct CachedShearPatch{T}
    faces::Vector{DiffusionFaceKey}
    coefficients::Vector{T}
end

const DiffusionPatchGeometry{T} = Tuple{
    NTuple{2, Int}, NTuple{2, Int}, Int,
    NTuple{2, T}, NTuple{2, T}, NTuple{2, Bool},
}

"""One persistent local-patch cache for a fixed tree-hierarchy depth."""
mutable struct LocalShearPatchCache{T}
    geometry::Union{Nothing, DiffusionPatchGeometry{T}}
    active_blocks::Set{BlockKey}
    current_blocks::Set{BlockKey}
    changed_blocks::Vector{BlockKey}
    patches::Dict{ShearPatchKey, CachedShearPatch{T}}
    invalidated_patches::Vector{ShearPatchKey}
    face_indices::Dict{DiffusionFaceKey, Int}
    face_keys::Vector{DiffusionFaceKey}
    # Invalidation scratch, reused across topology changes.
    patch_keys::Vector{ShearPatchKey}
    stale_patches::Vector{Bool}
end

LocalShearPatchCache(::Type{T}) where {T} = LocalShearPatchCache{T}(
    nothing, Set{BlockKey}(), Set{BlockKey}(), BlockKey[],
    Dict{ShearPatchKey, CachedShearPatch{T}}(), ShearPatchKey[],
    Dict{DiffusionFaceKey, Int}(), DiffusionFaceKey[], ShearPatchKey[], Bool[],
)

"""Grow-only scratch and local topology storage for stress construction."""
mutable struct VariableDiffusionConstructionCache{T}
    cell_references::Vector{CellRef}
    strain_rows::Vector{Int}
    strain_columns::Vector{Int}
    strain_values::Vector{T}
    wall_reaction_rows::Vector{Int}
    wall_reaction_columns::Vector{Int}
    wall_reaction_values::Vector{T}
    dirichlet_columns::Vector{Int}
    interpolation_rows::Vector{Int}
    interpolation_columns::Vector{Int}
    interpolation_values::Vector{T}
    vertices::Dict{
        NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}},
    }
    vertex_reference_pool::Vector{Vector{Tuple{CellRef, T}}}
    vertex_keys::Vector{NTuple{2, Int}}
    block_interior_faces::Vector{Vector{Int}}
    block_boundary_faces::Vector{Vector{Int}}
    block_indices::Dict{BlockKey, Int}
    derivative_blocks::Vector{Int}
    derivative_block_marks::Vector{Int}
    derivative_interior_marks::Vector{Int}
    derivative_boundary_marks::Vector{Int}
    derivative_search_epoch::Int
    derivative_points::Vector{Tuple{Int, T, T}}
    derivative_candidates::Vector{Int}
    derivative_indices::Vector{Int}
    derivative_coefficients::Vector{T}
    derivative_templates::Dict{ShearTemplateKey{T}, Vector{T}}
    derivative_template_hits::Int
    derivative_template_misses::Int
    patch_layers::Vector{LocalShearPatchCache{T}}
    patch_hits::Int
    patch_misses::Int
    patch_invalidations::Int
    patch_remap_failures::Int
    stiffness_heads::Vector{Int}
    stiffness_column_counts::Vector{Int}
    stiffness_contribution_counts::Vector{Int}
    stiffness_contribution_starts::Vector{Int}
end

function VariableDiffusionConstructionCache(::Type{T}) where {T}
    return VariableDiffusionConstructionCache(
        CellRef[], Int[], Int[], T[], Int[], Int[], T[], Int[],
        Int[], Int[], T[],
        Dict{NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}}}(),
        Vector{Tuple{CellRef, T}}[], NTuple{2, Int}[],
        Vector{Int}[], Vector{Int}[], Dict{BlockKey, Int}(),
        Int[], Int[], Int[], Int[], 0,
        Tuple{Int, T, T}[], Int[], Int[], T[],
        Dict{ShearTemplateKey{T}, Vector{T}}(), 0, 0,
        LocalShearPatchCache{T}[], 0, 0, 0, 0,
        Int[], Int[], Int[], Int[],
    )
end

function _cached_cell_references!(references, mesh)
    empty!(references)
    nx, ny = mesh.block_size
    sizehint!(references, ncells(mesh))
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        push!(references, CellRef(block, i, j))
    end
    return references
end

"""Columns per fixed chunk of `_symmetric_stiffness!`."""
const _STIFFNESS_COLUMN_CHUNK = 256

"""
    _symmetric_stiffness!(pointers, stresses, factors, offsets, cache, strain, strain_transpose, weights)

Return the symmetric stiffness `E'WE` for the strain `E = strain` and the
stress weights `W = Diagonal(weights)`, and fill its coefficient-refresh map.

Column `c` of the stiffness holds every face `r` that shares a stress row with
face `c`, also where the outer products cancel for the current weights:
changing the phase viscosity can make them nonzero. Its value is the mean of
the two algebraically equal halves `(A[r, c] + A[c, r]) / 2`, so it is exactly
symmetric. Each half is the left-to-right sum over the shared stresses `s` in
increasing order of `(E[s, r] W[s]) E[s, c]`, bitwise SparseArrays'
`transpose(E) * spdiagm(0 => W) * E`.

The refresh map lists, for stored entry `p`, the shared stresses `s` in the
same order in `stresses[offsets[p]:(offsets[p + 1] - 1)]` with the factors
`E[s, r] E[s, c]`; `pointers` repeats `p` once per contribution. A refresh
sums `W[s] * factor` in that order.

The columns are built by a merge of the rows of `strain_transpose` at the
stresses of each column: a count pass, a prefix sum and a fill pass, threaded
over fixed chunks of `_STIFFNESS_COLUMN_CHUNK` columns. Every column owns its
outputs, so the result is the same for every thread count.
"""
function _symmetric_stiffness!(
        pointers, stresses, factors, offsets, cache, strain, strain_transpose, weights,
    )
    T = eltype(strain)
    number_of_faces = size(strain, 2)
    strain_pointers = strain.colptr
    stress_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    face_pointers = strain_transpose.colptr
    face_rows = rowvals(strain_transpose)
    transpose_values = nonzeros(strain_transpose)
    width = 0
    for column in 1:number_of_faces
        width = max(width, strain_pointers[column + 1] - strain_pointers[column])
    end
    chunks = cld(number_of_faces, _STIFFNESS_COLUMN_CHUNK)
    merge_heads = resize!(cache.stiffness_heads, chunks * width)
    column_counts = resize!(cache.stiffness_column_counts, number_of_faces)
    contribution_counts = resize!(cache.stiffness_contribution_counts, number_of_faces)
    @_parallel work = _STIFFNESS_COLUMN_CHUNK for chunk in 1:chunks
        head_offset = (chunk - 1) * width
        last_column = min(number_of_faces, chunk * _STIFFNESS_COLUMN_CHUNK)
        for column in ((chunk - 1) * _STIFFNESS_COLUMN_CHUNK + 1):last_column
            entries, contributions = _stiffness_column!(
                nothing, column, head_offset, merge_heads, strain_pointers, stress_rows,
                strain_values, face_pointers, face_rows, transpose_values, weights,
            )
            @inbounds column_counts[column] = entries
            @inbounds contribution_counts[column] = contributions
        end
    end
    column_pointers = _csr_offsets(column_counts)
    contribution_starts = _csr_offsets(
        contribution_counts, cache.stiffness_contribution_starts,
    )
    number_of_entries = column_pointers[end] - 1
    number_of_contributions = contribution_starts[end] - 1
    rows = Vector{Int}(undef, number_of_entries)
    stiffness_values = Vector{T}(undef, number_of_entries)
    resize!(pointers, number_of_contributions)
    resize!(stresses, number_of_contributions)
    resize!(factors, number_of_contributions)
    resize!(offsets, number_of_entries + 1)
    @_parallel work = _STIFFNESS_COLUMN_CHUNK for chunk in 1:chunks
        head_offset = (chunk - 1) * width
        last_column = min(number_of_faces, chunk * _STIFFNESS_COLUMN_CHUNK)
        outputs = (
            column_pointers, contribution_starts, rows, stiffness_values,
            pointers, stresses, factors, offsets,
        )
        for column in ((chunk - 1) * _STIFFNESS_COLUMN_CHUNK + 1):last_column
            _stiffness_column!(
                outputs, column, head_offset, merge_heads, strain_pointers, stress_rows,
                strain_values, face_pointers, face_rows, transpose_values, weights,
            )
        end
    end
    offsets[end] = number_of_contributions + 1
    return SparseMatrixCSC(
        number_of_faces, number_of_faces, column_pointers, rows, stiffness_values,
    )
end

"""Count (`outputs === nothing`) or fill stiffness column `column` of
`_symmetric_stiffness!`: merge the sorted face lists of the stresses of
`column`, in increasing stress order, with one head per stress in
`merge_heads[head_offset .+ (1:width)]`. Return the numbers of stored entries and
refresh contributions of the column."""
@inline function _stiffness_column!(
        outputs, column, head_offset, merge_heads, strain_pointers, stress_rows,
        strain_values, face_pointers, face_rows, transpose_values, weights,
    )
    first_stress = @inbounds strain_pointers[column]
    stress_count = (@inbounds strain_pointers[column + 1]) - first_stress
    contributions = 0
    for local_stress in 1:stress_count
        stress = @inbounds stress_rows[first_stress + local_stress - 1]
        @inbounds merge_heads[head_offset + local_stress] = face_pointers[stress]
        contributions += (@inbounds face_pointers[stress + 1]) - face_pointers[stress]
    end
    if outputs !== nothing
        column_pointers, contribution_starts, rows, stiffness_values,
            pointers, stresses, factors, offsets = outputs
        entry = @inbounds column_pointers[column]
        contribution = @inbounds contribution_starts[column]
    end
    entries = 0
    while true
        # The next face of the column: the smallest head over all stresses.
        row = typemax(Int)
        for local_stress in 1:stress_count
            stress = @inbounds stress_rows[first_stress + local_stress - 1]
            head = @inbounds merge_heads[head_offset + local_stress]
            head < (@inbounds face_pointers[stress + 1]) &&
                (row = min(row, @inbounds face_rows[head]))
        end
        row == typemax(Int) && break
        entries += 1
        if outputs !== nothing
            @inbounds offsets[entry] = contribution
            lower = zero(eltype(stiffness_values))
            upper = zero(eltype(stiffness_values))
        end
        shared = 0
        for local_stress in 1:stress_count
            stress = @inbounds stress_rows[first_stress + local_stress - 1]
            head = @inbounds merge_heads[head_offset + local_stress]
            head < (@inbounds face_pointers[stress + 1]) &&
                (@inbounds face_rows[head]) == row || continue
            @inbounds merge_heads[head_offset + local_stress] = head + 1
            outputs === nothing && continue
            shared += 1
            row_value = @inbounds transpose_values[head]
            column_value = @inbounds strain_values[first_stress + local_stress - 1]
            weight = @inbounds weights[stress]
            # SparseArrays assigns the first product and adds the others.
            lower_term = (row_value * weight) * column_value
            upper_term = (column_value * weight) * row_value
            lower = shared == 1 ? lower_term : lower + lower_term
            upper = shared == 1 ? upper_term : upper + upper_term
            @inbounds pointers[contribution] = entry
            @inbounds stresses[contribution] = stress
            @inbounds factors[contribution] = row_value * column_value
            contribution += 1
        end
        if outputs !== nothing
            @inbounds rows[entry] = row
            @inbounds stiffness_values[entry] = (lower + upper) / 2
            entry += 1
        end
    end
    return entries, contributions
end

"""
    dynamic_viscosity(interface; viscosity, averaging=HarmonicViscosityAveraging())

Return the cell-centred dynamic viscosity used by the one-velocity diffusion
operator, mixed from the phase viscosities with the conditional liquid
fraction `c` of the open fluid part of the cell and the phase-viscosity
averaging rule `averaging`.

[`HarmonicViscosityAveraging`](@ref) (the default) gives
`1 / (c / mu_liquid + (1 - c) / mu_gas)`, the series conductance across the
interface in a cell; [`GeometricViscosityAveraging`](@ref) gives the thesis'
diffuse mixture `mu_liquid^c * mu_gas^(1-c)`. The same rule averages the
viscosity of a stress row from the cell quarters it spans, see
`_fill_stress_weights!`.
"""
function dynamic_viscosity(
        field::InterfaceField{T}; viscosity::NTuple{2},
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    ) where {T}
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    result = CellField(field.mesh, T)
    mesh = field.mesh
    nx, _ = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        viscosity_values = @inbounds result.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            viscosity_values[i + h, j + h] =
                _phase_dynamic_viscosity(fraction[i + h, j + h], phases, averaging)
        end
    end
    return result
end

function _stress_vertices!(cache, mesh, ::Type{T}) where {T}
    # Cell-corner quadrature weights partition the domain volume exactly.
    lattice = _finest_lattice_size(mesh)
    vertices = cache.vertices
    for (_, references) in values(vertices)
        empty!(references)
        push!(cache.vertex_reference_pool, references)
    end
    empty!(vertices)
    for ref in cache.cell_references
        block = mesh.blocks[ref.block]
        low, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
        for sx in (0, 1), sy in (0, 1)
            key_x = low[1] + sx * scale[1]
            key_y = low[2] + sy * scale[2]
            key = (
                mesh.periodic[1] ? mod(key_x, lattice[1]) : key_x,
                mesh.periodic[2] ? mod(key_y, lattice[2]) : key_y,
            )
            volume, refs = get!(vertices, key) do
                references = isempty(cache.vertex_reference_pool) ?
                    Tuple{CellRef, T}[] : pop!(cache.vertex_reference_pool)
                (zero(T), references)
            end
            weight = T(block.volume / 4)
            push!(refs, (ref, weight))
            vertices[key] = (volume + weight, refs)
        end
    end
    return vertices, lattice
end

function _release_stress_vertices!(cache)
    for (_, references) in values(cache.vertices)
        empty!(references)
        push!(cache.vertex_reference_pool, references)
    end
    empty!(cache.vertices)
    empty!(cache.vertex_keys)
    return cache
end

struct DiffusionFaceSearch
    interior_by_block::Vector{Vector{Int}}
    boundary_by_block::Vector{Vector{Int}}
    block_indices::Dict{BlockKey, Int}
end

@inline _shear_boundary_kind(::FreeSlip) = UInt8(1)
@inline _shear_boundary_kind(::NoSlip) = UInt8(2)

function _shear_patch_layer!(
        cache::VariableDiffusionConstructionCache{T}, layer_index::Int,
    ) where {T}
    layer_index >= 1 || throw(ArgumentError("patch-cache layer must be positive"))
    while length(cache.patch_layers) < layer_index
        push!(cache.patch_layers, LocalShearPatchCache(T))
    end
    return cache.patch_layers[layer_index]
end

@inline function _diffusion_face_key(mesh::BlockTreeMesh, face)
    normal, lo, hi = _face_lattice_geometry(mesh, face)
    return DiffusionFaceKey(UInt8(face.axis), normal, lo, hi)
end

@inline function _coordinate_near_interval(
        coordinate, low, high, margin, periodic, period,
    )
    if periodic
        return any(
            low - margin <= coordinate + shift <= high + margin
                for shift in (-period, 0, period)
        )
    end
    return low - margin <= coordinate <= high + margin
end

"""Lattice box and margin of the patches a changed block `key` invalidates."""
function _changed_block_window(key::BlockKey, mesh::BlockTreeMesh)
    lattice = _finest_lattice_size(mesh)
    block_scale = _block_scale(mesh, key.level)
    low = (
        key.i * mesh.block_size[1] * block_scale,
        key.j * mesh.block_size[2] * block_scale,
    )
    high = (
        low[1] + mesh.block_size[1] * block_scale,
        low[2] + mesh.block_size[2] * block_scale,
    )

    # A shear search has radius three local cell widths. Across a balanced
    # 2:1 interface the neighbouring cell can be twice as wide as a changed
    # block cell, hence the conservative six-cell invalidation halo.
    cell_spacing = _spacing_at_level(mesh, key.level)
    physical_margin = 6maximum(cell_spacing)
    finest_spacing = (
        mesh.extent[1] / lattice[1], mesh.extent[2] / lattice[2],
    )
    margin = (
        ceil(Int, physical_margin / finest_spacing[1]),
        ceil(Int, physical_margin / finest_spacing[2]),
    )
    return low, high, margin
end

@inline function _patch_near_changed_block(
        vertex::NTuple{2, Int}, window, mesh::BlockTreeMesh,
    )
    low, high, margin = window
    lattice = _finest_lattice_size(mesh)
    return _coordinate_near_interval(
        vertex[1], low[1], high[1], margin[1],
        mesh.periodic[1], lattice[1],
    ) && _coordinate_near_interval(
        vertex[2], low[2], high[2], margin[2],
        mesh.periodic[2], lattice[2],
    )
end

function _prepare_shear_patch_layer!(
        cache::VariableDiffusionConstructionCache{T}, mesh::BlockTreeMesh,
        layer_index::Int,
    ) where {T}
    layer = _shear_patch_layer!(cache, layer_index)
    geometry = (
        mesh.base_mesh, mesh.block_size, mesh.max_level,
        T.(mesh.origin), T.(mesh.extent),
        mesh.periodic,
    )
    if !isequal(layer.geometry, geometry)
        cache.patch_invalidations += length(layer.patches)
        empty!(layer.patches)
        empty!(layer.active_blocks)
        layer.geometry = geometry
    end

    empty!(layer.current_blocks)
    for block in mesh.blocks
        push!(layer.current_blocks, block.key)
    end
    empty!(layer.changed_blocks)
    for key in layer.current_blocks
        key in layer.active_blocks || push!(layer.changed_blocks, key)
    end
    for key in layer.active_blocks
        key in layer.current_blocks || push!(layer.changed_blocks, key)
    end

    if !isempty(layer.changed_blocks) && !isempty(layer.patches)
        windows = [_changed_block_window(key, mesh) for key in layer.changed_blocks]
        patch_keys = append!(empty!(layer.patch_keys), keys(layer.patches))
        stale = resize!(layer.stale_patches, length(patch_keys))
        _parallel_for(length(patch_keys); work = length(windows)) do index
            vertex = patch_keys[index].vertex
            stale[index] = any(
                window -> _patch_near_changed_block(vertex, window, mesh), windows,
            )
        end
        empty!(layer.invalidated_patches)
        for index in eachindex(patch_keys)
            stale[index] && push!(layer.invalidated_patches, patch_keys[index])
        end
        for patch_key in layer.invalidated_patches
            delete!(layer.patches, patch_key)
        end
        cache.patch_invalidations += length(layer.invalidated_patches)
    end
    empty!(layer.active_blocks)
    union!(layer.active_blocks, layer.current_blocks)

    empty!(layer.face_indices)
    resize!(layer.face_keys, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        key = _diffusion_face_key(mesh, face)
        haskey(layer.face_indices, key) && error(
            "duplicate stable staggered-face key in diffusion patch cache",
        )
        layer.face_keys[index] = key
        layer.face_indices[key] = index
    end
    return layer
end

function _shear_template_key(
        points::Vector{Tuple{Int, T, T}}, component, boundary, scale,
    ) where {T}
    count = length(points)
    count <= _MAXIMUM_SHEAR_TEMPLATE_POINTS || throw(
        ArgumentError("too many points in a shear reconstruction template"),
    )
    padding = ShearTemplatePoint(false, zero(T), zero(T))
    descriptors = ntuple(Val(_MAXIMUM_SHEAR_TEMPLATE_POINTS)) do index
        if index <= count
            point = @inbounds points[index]
            ShearTemplatePoint(
                iszero(point[1]), point[2] / scale, point[3] / scale,
            )
        else
            padding
        end
    end
    return ShearTemplateKey(
        UInt8(component), _shear_boundary_kind(boundary), UInt8(count),
        descriptors,
    )
end

function _resize_face_buckets!(buckets, number_of_blocks)
    old_length = length(buckets)
    resize!(buckets, number_of_blocks)
    for block in 1:number_of_blocks
        if block > old_length
            buckets[block] = Int[]
        else
            empty!(buckets[block])
        end
    end
    return buckets
end

function DiffusionFaceSearch(cache, mesh)
    number_of_blocks = nblocks(mesh)
    interior = _resize_face_buckets!(
        cache.block_interior_faces, number_of_blocks,
    )
    boundary = _resize_face_buckets!(
        cache.block_boundary_faces, number_of_blocks,
    )
    empty!(cache.block_indices)
    for (index, block) in pairs(mesh.blocks)
        cache.block_indices[block.key] = index
    end
    for (index, face) in pairs(mesh.faces)
        push!(interior[face.minus.block], index)
        face.plus.block == face.minus.block ||
            push!(interior[face.plus.block], index)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        push!(boundary[face.cell.block], index)
    end
    resize!(cache.derivative_block_marks, number_of_blocks)
    resize!(cache.derivative_interior_marks, length(mesh.faces))
    resize!(cache.derivative_boundary_marks, length(mesh.boundary_faces))
    fill!(cache.derivative_block_marks, 0)
    fill!(cache.derivative_interior_marks, 0)
    fill!(cache.derivative_boundary_marks, 0)
    cache.derivative_search_epoch = 0
    return DiffusionFaceSearch(
        interior, boundary, cache.block_indices,
    )
end

function _nearby_tree_blocks!(
        blocks, marks, epoch, search, mesh, point, radius,
    )
    empty!(blocks)
    xshifts = mesh.periodic[1] ?
        (-mesh.extent[1], zero(eltype(mesh.extent)), mesh.extent[1]) :
        (zero(eltype(mesh.extent)),)
    yshifts = mesh.periodic[2] ?
        (-mesh.extent[2], zero(eltype(mesh.extent)), mesh.extent[2]) :
        (zero(eltype(mesh.extent)),)
    for yshift in yshifts, xshift in xshifts
        low = (
            point[1] - radius + xshift,
            point[2] - radius + yshift,
        )
        high = (
            point[1] + radius + xshift,
            point[2] + radius + yshift,
        )
        _collect_leaf_blocks_in_window!(
            blocks, marks, epoch, mesh.root, search.block_indices, low, high,
        )
    end
    return blocks
end

@inline function _within_diffusion_radius(dx, dy, radius_squared)
    return muladd(dx, dx, dy * dy) <= radius_squared
end

function _nearby_diffusion_faces!(
        points::Vector{Tuple{Int, T, T}}, candidates,
        search::DiffusionFaceSearch, mesh, point, component, scale,
        cache::VariableDiffusionConstructionCache,
    ) where {T}
    empty!(points)
    empty!(candidates)
    radius = 3scale
    radius_squared = radius * radius
    cache.derivative_search_epoch += 1
    epoch = cache.derivative_search_epoch
    _nearby_tree_blocks!(
        cache.derivative_blocks, cache.derivative_block_marks, epoch,
        search, mesh, point, radius,
    )

    for block in cache.derivative_blocks
        for index in search.interior_by_block[block]
            cache.derivative_interior_marks[index] == epoch && continue
            cache.derivative_interior_marks[index] = epoch
            face = mesh.faces[index]
            face.axis == component || continue
            push!(candidates, index)
        end
    end
    for index in candidates
        face = mesh.faces[index]
        dx = T(
            _wrapped_delta(
                face.center[1] - point[1], mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                face.center[2] - point[2], mesh.extent[2], mesh.periodic[2]
            )
        )
        _within_diffusion_radius(dx, dy, radius_squared) &&
            push!(points, (index, dx, dy))
    end

    empty!(candidates)
    for block in cache.derivative_blocks
        for index in search.boundary_by_block[block]
            cache.derivative_boundary_marks[index] == epoch && continue
            cache.derivative_boundary_marks[index] = epoch
            face = mesh.boundary_faces[index]
            face.axis == component || continue
            push!(candidates, index)
        end
    end
    for index in candidates
        face = mesh.boundary_faces[index]
        dx = T(face.center[1] - point[1])
        dy = T(face.center[2] - point[2])
        _within_diffusion_radius(dx, dy, radius_squared) &&
            push!(points, (0, dx, dy))
    end
    return points
end

@inline function _vertex_on_wall(mesh, vertex, axis, side)
    boundary = side < 0 ? 0 : _finest_lattice_size(mesh)[axis]
    return !mesh.periodic[axis] && vertex[axis] == boundary
end

function _vertex_derivative(
        mesh, vertex, point, component, scale, boundary, ::Type{T}, search,
        points, candidates, ids, stencil, cache
    ) where {T}
    axis = 3 - component
    tolerance = 128eps(T) * maximum(mesh.extent)
    _nearby_diffusion_faces!(
        points, candidates, search, mesh, point, component, scale, cache,
    )
    empty!(ids)
    empty!(stencil)
    # Standard MAC shear derivative on regular lines (including wall closure).
    wall_low = _vertex_on_wall(mesh, vertex, axis, -1)
    wall_high = _vertex_on_wall(mesh, vertex, axis, 1)
    if (wall_low || wall_high) && boundary isa FreeSlip
        return ids, stencil
    end
    low_index = 0
    high_index = 0
    low_coordinate = zero(T)
    high_coordinate = zero(T)
    has_low = wall_low
    has_high = wall_high
    for (index, dx, dy) in points
        normal_coordinate = component == 1 ? dx : dy
        abs(normal_coordinate) <= tolerance || continue
        coordinate = component == 1 ? dy : dx
        if coordinate < -tolerance &&
                (!has_low || coordinate > low_coordinate)
            low_index = index
            low_coordinate = coordinate
            has_low = true
        elseif coordinate > tolerance &&
                (!has_high || coordinate < high_coordinate)
            high_index = index
            high_coordinate = coordinate
            has_high = true
        end
    end
    if wall_low
        low_index = 0
        low_coordinate = zero(T)
    elseif wall_high
        high_index = 0
        high_coordinate = zero(T)
    end
    if has_low && has_high
        distance = high_coordinate - low_coordinate
        low_index > 0 &&
            (push!(ids, low_index); push!(stencil, -inv(distance)))
        high_index > 0 &&
            (push!(ids, high_index); push!(stencil, inv(distance)))
        return ids, stencil
    end
    # At hanging vertices a collocated pair can be absent. A weighted quadratic
    # fit reproduces affine strain while using the same owned face unknowns.
    sort!(points; by = p -> (p[2]^2 + p[3]^2, p[2], p[3], iszero(p[1])))
    length(points) > 16 && resize!(points, 16)
    template_key = _shear_template_key(
        points, component, boundary, scale,
    )
    template = get(cache.derivative_templates, template_key, nothing)
    if !isnothing(template)
        cache.derivative_template_hits += 1
        for (i, p) in pairs(points)
            p[1] > 0 || continue
            push!(ids, p[1])
            push!(stencil, template[i] / scale)
        end
        return ids, stencil
    end
    for degree in (2, 1)
        columns = degree == 2 ? 6 : 3
        length(points) >= columns || continue
        matrix = Matrix{T}(undef, length(points), columns)
        weights = Vector{T}(undef, length(points))
        for (i, p) in pairs(points)
            x, y = p[2] / scale, p[3] / scale
            weights[i] = inv(1 + x * x + y * y)
            row = degree == 2 ? (one(T), x, y, x * x, x * y, y * y) : (one(T), x, y)
            for j in 1:columns
                matrix[i, j] = weights[i] * row[j]
            end
        end
        singular = svdvals(matrix)
        minimum(singular) > 1.0e-10maximum(singular) || continue
        target = zeros(T, columns);target[axis + 1] = inv(scale)
        # Minimal-norm solution of the polynomial reproduction constraints.
        coefficients = transpose(matrix) \ target
        template = Vector{T}(undef, length(points))
        for (i, p) in pairs(points)
            value = weights[i] * coefficients[i]
            template[i] = value * scale
            p[1] > 0 || continue
            push!(ids, p[1]);push!(stencil, value)
        end
        cache.derivative_templates[template_key] = template
        cache.derivative_template_misses += 1
        return ids, stencil
    end
    throw(ArgumentError("insufficient faces for staggered shear reconstruction"))
end

function _restore_shear_patch!(
        ids, coefficients, patch::CachedShearPatch,
        layer::LocalShearPatchCache,
    )
    empty!(ids)
    empty!(coefficients)
    for index in eachindex(patch.faces)
        face_index = get(layer.face_indices, patch.faces[index], 0)
        if iszero(face_index)
            empty!(ids)
            empty!(coefficients)
            return false
        end
        push!(ids, face_index)
        push!(coefficients, patch.coefficients[index])
    end
    return true
end

function _store_shear_patch(
        ids, coefficients, layer::LocalShearPatchCache{T},
    ) where {T}
    faces = Vector{DiffusionFaceKey}(undef, length(ids))
    values = Vector{T}(undef, length(coefficients))
    for index in eachindex(ids)
        @inbounds faces[index] = layer.face_keys[ids[index]]
        @inbounds values[index] = coefficients[index]
    end
    return CachedShearPatch(faces, values)
end

function _cached_vertex_derivative!(
        mesh, vertex_key, point, component, scale, boundary, ::Type{T}, search,
        points, candidates, ids, coefficients,
        cache::VariableDiffusionConstructionCache{T},
        layer::LocalShearPatchCache{T},
    ) where {T}
    patch_key = ShearPatchKey(
        vertex_key, UInt8(component), _shear_boundary_kind(boundary),
    )
    patch = get(layer.patches, patch_key, nothing)
    if !isnothing(patch)
        if _restore_shear_patch!(ids, coefficients, patch, layer)
            cache.patch_hits += 1
            return ids, coefficients
        end
        cache.patch_remap_failures += 1
        delete!(layer.patches, patch_key)
    end

    cache.patch_misses += 1
    _vertex_derivative(
        mesh, vertex_key, point, component, scale, boundary, T, search,
        points, candidates, ids, coefficients, cache,
    )
    layer.patches[patch_key] = _store_shear_patch(
        ids, coefficients, layer,
    )
    return ids, coefficients
end
