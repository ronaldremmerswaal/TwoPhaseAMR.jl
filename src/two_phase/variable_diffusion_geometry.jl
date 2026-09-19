"""Coupled staggered `div(2μD(u))` with cell-centred normal stresses and vertex
shear stresses. The integrated stiffness is `E'WE`, so viscosity dissipates
kinetic energy for positive μ. Density enters the time-step mass matrix.
"""
mutable struct VariableStaggeredDiffusionOperator{T, M, B}
    mesh::M
    stiffness::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    boundary::B
    strain::SparseMatrixCSC{T, Int}
    strain_transpose::SparseMatrixCSC{T, Int}
    stress_interpolation::SparseMatrixCSC{T, Int}
    stress_interpolation_transpose::SparseMatrixCSC{T, Int}
    stress_measures::Vector{T}
    stress_weights::Vector{T}
    cell_work::Vector{T}
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
end

LocalShearPatchCache(::Type{T}) where {T} = LocalShearPatchCache{T}(
    nothing, Set{BlockKey}(), Set{BlockKey}(), BlockKey[],
    Dict{ShearPatchKey, CachedShearPatch{T}}(), ShearPatchKey[],
    Dict{DiffusionFaceKey, Int}(), DiffusionFaceKey[],
)

"""Grow-only scratch and local topology storage for stress construction."""
mutable struct VariableDiffusionConstructionCache{T}
    cell_references::Vector{CellRef}
    strain_rows::Vector{Int}
    strain_columns::Vector{Int}
    strain_values::Vector{T}
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
    stiffness_entries::Vector{Vector{Tuple{Int, T}}}
end

function VariableDiffusionConstructionCache(::Type{T}) where {T}
    return VariableDiffusionConstructionCache(
        CellRef[], Int[], Int[], T[], Int[], Int[], T[],
        Dict{NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}}}(),
        Vector{Tuple{CellRef, T}}[], NTuple{2, Int}[],
        Vector{Int}[], Vector{Int}[], Dict{BlockKey, Int}(),
        Int[], Int[], Int[], Int[], 0,
        Tuple{Int, T, T}[], Int[], Int[], T[],
        Dict{ShearTemplateKey{T}, Vector{T}}(), 0, 0,
        LocalShearPatchCache{T}[], 0, 0, 0, 0,
        Vector{Tuple{Int, T}}[],
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

function _stiffness_refresh_map!(
        pointers, stresses, factors, offsets, entries, strain, stiffness
    )
    T = eltype(stiffness)
    empty!(pointers)
    empty!(stresses)
    empty!(factors)
    old_length = length(entries)
    number_of_stresses = size(strain, 1)
    number_of_stresses > old_length && resize!(entries, number_of_stresses)
    for index in 1:number_of_stresses
        if index > old_length
            entries[index] = Tuple{Int, T}[]
        else
            empty!(entries[index])
        end
    end
    strain_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        row = strain_rows[pointer]
        push!(entries[row], (column, strain_values[pointer]))
    end
    stiffness_rows = rowvals(stiffness)
    for stress in 1:number_of_stresses
        row_entries = entries[stress]
        for (column, right_value) in row_entries
            for (row, left_value) in row_entries
                pointer = 0
                for candidate in nzrange(stiffness, column)
                    if stiffness_rows[candidate] == row
                        pointer = candidate
                        break
                    end
                end
                iszero(pointer) && error(
                    "stress refresh changed diffusion sparsity",
                )
                push!(pointers, pointer)
                push!(stresses, stress)
                push!(factors, left_value * right_value)
            end
        end
    end
    permutation = sortperm(pointers)
    pointers[:] = pointers[permutation]
    stresses[:] = stresses[permutation]
    factors[:] = factors[permutation]
    resize!(offsets, nnz(stiffness) + 1)
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
        "diffusion coefficient refresh contains an invalid sparse pointer",
    )
    return pointers, stresses, factors, offsets
end

function _complete_symmetric_stiffness(strain, weights)
    assembled = transpose(strain) * spdiagm(0 => weights) * strain
    # A stress row can contain coefficients whose outer products cancel for
    # the initial viscosity. Those locations must remain structurally present:
    # changing the phase viscosity can make them nonzero. Build the symbolic
    # pattern with positive unit entries, then sample the two algebraically
    # equivalent halves to store one exactly symmetric value.
    unit_strain = copy(strain)
    fill!(nonzeros(unit_strain), one(eltype(strain)))
    pattern = transpose(unit_strain) * unit_strain
    pattern_rows = rowvals(pattern)
    pattern_values = nonzeros(pattern)
    for column in axes(pattern, 2), pointer in nzrange(pattern, column)
        row = pattern_rows[pointer]
        @inbounds pattern_values[pointer] =
            (assembled[row, column] + assembled[column, row]) / 2
    end
    return pattern
end

@inline function _phase_dynamic_viscosity(c, viscosity::NTuple{2})
    liquid, gas = viscosity
    return exp(muladd(c, log(liquid) - log(gas), log(gas)))
end

"""
    dynamic_viscosity(interface; viscosity)

Return the cell-centred dynamic viscosity used by the one-velocity diffusion
operator. Following the thesis, the coefficient is the phase-fraction-
weighted geometric mean

`mu = mu_liquid^c * mu_gas^(1-c)`, where `c` is the liquid fraction within the
open fluid part of a cut cell.

At shear-stress vertices the operator volume-interpolates `c` before applying
the same formula. Equivalently, it geometrically averages this returned field,
because `log(mu)` is affine in `c`.
"""
function dynamic_viscosity(
        field::InterfaceField{T}; viscosity::NTuple{2}
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
                _phase_dynamic_viscosity(fraction[i + h, j + h], phases)
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
    nx, ny = mesh.block_size
    for ref in cache.cell_references
        block = mesh.blocks[ref.block]
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = block.spacing
        for sx in (-1, 1),sy in (-1, 1)
            key_x = round(
                Int,
                (x + sx * hx / 2 - mesh.origin[1]) /
                    mesh.extent[1] * lattice[1],
            )
            key_y = round(
                Int,
                (y + sy * hy / 2 - mesh.origin[2]) /
                    mesh.extent[2] * lattice[2],
            )
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
    axis = face.axis
    tangent_axis = 3 - axis
    lattice = _finest_lattice_size(mesh)
    normal_spacing = mesh.extent[axis] / lattice[axis]
    tangent_spacing = mesh.extent[tangent_axis] / lattice[tangent_axis]
    normal = round(
        Int, (face.center[axis] - mesh.origin[axis]) / normal_spacing,
    )
    tangent_center =
        (face.center[tangent_axis] - mesh.origin[tangent_axis]) /
        tangent_spacing
    half_length = face.area / (2tangent_spacing)
    return DiffusionFaceKey(
        UInt8(axis), normal,
        round(Int, tangent_center - half_length),
        round(Int, tangent_center + half_length),
    )
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

function _patch_near_changed_block(
        vertex::NTuple{2, Int}, key::BlockKey, mesh::BlockTreeMesh,
    )
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
        empty!(layer.invalidated_patches)
        for patch_key in keys(layer.patches)
            any(
                key -> _patch_near_changed_block(
                    patch_key.vertex, key, mesh,
                ),
                layer.changed_blocks,
            ) && push!(layer.invalidated_patches, patch_key)
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

function _vertex_derivative(
        mesh, point, component, scale, boundary, ::Type{T}, search,
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
    wall_low = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis]) <= tolerance
    wall_high = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis] - mesh.extent[axis]) <= tolerance
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
        mesh, point, component, scale, boundary, T, search,
        points, candidates, ids, coefficients, cache,
    )
    layer.patches[patch_key] = _store_shear_patch(
        ids, coefficients, layer,
    )
    return ids, coefficients
end

