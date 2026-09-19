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

function _variable_staggered_diffusion(
        mesh, viscosity::CellField, boundary, ::Type{T};
        cache = VariableDiffusionConstructionCache(T), reuse = nothing,
        assemble_stiffness = true, cache_layer = 1,
        geometry = nothing,
    ) where {T}
    viscosity.mesh === mesh || throw(ArgumentError("viscosity field must share the operator mesh"))
    boundary isa FreeSlip || (boundary isa NoSlip && boundary.velocity isa StationaryWall) ||
        throw(ArgumentError("variable stress diffusion currently supports stationary NoSlip or FreeSlip walls"))
    all(c -> isfinite(c)&&c > 0, (v for b in eachindex(mesh.blocks) for v in interior(viscosity, b))) ||
        throw(ArgumentError("dynamic viscosity must be finite and positive"))
    cache isa VariableDiffusionConstructionCache{T} || throw(
        ArgumentError("diffusion construction cache has the wrong scalar type"),
    )
    isnothing(reuse) || reuse isa VariableStaggeredDiffusionOperator{T} ||
        throw(ArgumentError("reused diffusion operator has the wrong type"))
    isnothing(geometry) || geometry.mesh === mesh || throw(
        ArgumentError("geometry and variable diffusion must share a mesh"),
    )
    !isnothing(geometry) &&
        !(boundary isa Union{NoSlip, FreeSlip}) && throw(
        ArgumentError(
            "embedded variable diffusion requires NoSlip() or FreeSlip()",
        ),
    )
    refs = _cached_cell_references!(cache.cell_references, mesh)
    nx, ny = mesh.block_size
    rows = cache.strain_rows
    cols = cache.strain_columns
    vals = cache.strain_values
    interpolation_rows = cache.interpolation_rows
    interpolation_columns = cache.interpolation_columns
    interpolation_values = cache.interpolation_values
    for values in (
            rows, cols, vals, interpolation_rows,
            interpolation_columns, interpolation_values,
        )
        empty!(values)
    end
    measures = isnothing(reuse) ? T[] : reuse.stress_measures
    empty!(measures)
    face_search = DiffusionFaceSearch(cache, mesh)
    patch_layer = _prepare_shear_patch_layer!(cache, mesh, cache_layer)
    derivative_points = cache.derivative_points
    derivative_candidates = cache.derivative_candidates
    derivative_indices = cache.derivative_indices
    derivative_coefficients = cache.derivative_coefficients
    for ref in refs,axis in 1:2
        row = length(measures) + 1
        open = isnothing(geometry) ? one(T) :
            _get(geometry.open_fraction, ref)
        push!(measures, 2 * open * mesh.blocks[ref.block].volume)
        push!(interpolation_rows, row)
        push!(interpolation_columns, _cell_index(ref, nx, ny))
        push!(interpolation_values, one(T))
    end
    for (i, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            open = isnothing(geometry) ? one(T) :
                _get(geometry.open_fraction, ref)
            open > zero(T) || continue
            row = 2 * (_cell_index(ref, nx, ny) - 1) + face.axis
            push!(rows, row);push!(cols, i)
            aperture = isnothing(geometry) ? one(T) : geometry.aperture[i]
            push!(
                vals,
                sign * aperture * face.area /
                    (open * mesh.blocks[ref.block].volume),
            )
        end
    end
    vertices, lattice = _stress_vertices!(cache, mesh, T)
    empty!(cache.vertex_keys)
    append!(cache.vertex_keys, keys(vertices))
    sort!(cache.vertex_keys)
    for key in cache.vertex_keys
        volume, cells = vertices[key]
        point = ntuple(a -> mesh.origin[a] + key[a] * mesh.extent[a] / lattice[a], 2)
        open_volume = isnothing(geometry) ? volume : sum(
                weight * _get(geometry.open_fraction, ref)
                for (ref, weight) in cells;
                init = zero(T),
            )
        open_volume > zero(T) || continue
        push!(measures, open_volume);row = length(measures)
        for (ref, raw_weight) in cells
            weight = isnothing(geometry) ? raw_weight :
                raw_weight * _get(geometry.open_fraction, ref)
            weight > zero(T) || continue
            push!(interpolation_rows, row)
            push!(interpolation_columns, _cell_index(ref, nx, ny))
            push!(interpolation_values, weight / open_volume)
        end
        scale = maximum(maximum(mesh.blocks[r.block].spacing) for (r, w) in cells)
        for component in 1:2
            indices, coefficients = _cached_vertex_derivative!(
                mesh, key, point, component, scale, boundary, T,
                face_search, derivative_points, derivative_candidates,
                derivative_indices, derivative_coefficients, cache,
                patch_layer,
            )
            for index in eachindex(indices)
                push!(rows, row)
                push!(cols, indices[index])
                push!(vals, coefficients[index])
            end
        end
    end
    if !isnothing(geometry)
        for (face_index, conductance, cell_index) in
            _embedded_variable_wall_stresses(
                geometry; include_cut_wall = boundary isa NoSlip,
            )
            row = length(measures) + 1
            push!(measures, conductance)
            push!(interpolation_rows, row)
            push!(interpolation_columns, cell_index)
            push!(interpolation_values, one(T))
            push!(rows, row)
            push!(cols, face_index)
            push!(vals, one(T))
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    strain_transpose = copy(transpose(strain))
    interpolation = sparse(
        interpolation_rows, interpolation_columns, interpolation_values,
        length(measures), length(refs)
    )
    interpolation_transpose = copy(transpose(interpolation))
    cell_work = isnothing(reuse) ? T[] : reuse.cell_work
    resize!(cell_work, length(refs))
    for (index, ref) in pairs(refs)
        @inbounds cell_work[index] = log(_get(viscosity, ref))
    end
    stress_work = isnothing(reuse) ? T[] : reuse.stress_work
    resize!(stress_work, length(measures))
    _parallel_sparse_mul_from_transpose!(
        stress_work, interpolation_transpose, cell_work,
    )
    weights = isnothing(reuse) ? T[] : reuse.stress_weights
    resize!(weights, length(measures))
    for index in eachindex(weights)
        @inbounds weights[index] = measures[index] * exp(stress_work[index])
    end
    stiffness_pointers = isnothing(reuse) ? Int[] : reuse.stiffness_pointers
    stiffness_stresses = isnothing(reuse) ? Int[] : reuse.stiffness_stresses
    stiffness_factors = isnothing(reuse) ? T[] : reuse.stiffness_factors
    stiffness_refresh_offsets = isnothing(reuse) ? Int[] :
        reuse.stiffness_refresh_offsets
    for values in (stiffness_pointers, stiffness_stresses, stiffness_factors)
        empty!(values)
    end
    stiffness = if assemble_stiffness
        assembled = _complete_symmetric_stiffness(strain, weights)
        stiffness_pointers, stiffness_stresses, stiffness_factors,
            stiffness_refresh_offsets =
            _stiffness_refresh_map!(
            stiffness_pointers, stiffness_stresses, stiffness_factors,
            stiffness_refresh_offsets,
            cache.stiffness_entries, strain, assembled,
        )
        assembled
    else
        spzeros(T, 0, 0)
    end
    volumes = isnothing(reuse) ? T[] : reuse.volumes
    resize!(volumes, length(mesh.faces))
    if isnothing(geometry)
        for (index, face) in pairs(mesh.faces)
            @inbounds volumes[index] = face.area * (
                mesh.blocks[face.minus.block].spacing[face.axis] +
                    mesh.blocks[face.plus.block].spacing[face.axis]
            ) / 2
        end
    else
        copyto!(volumes, _geometry_face_dual_volumes(geometry))
    end
    _release_stress_vertices!(cache)
    return VariableStaggeredDiffusionOperator(
        mesh, stiffness, volumes, boundary, strain, strain_transpose,
        interpolation, interpolation_transpose,
        measures, weights, cell_work, stress_work,
        stiffness_pointers, stiffness_stresses, stiffness_factors,
        stiffness_refresh_offsets,
    )
end

"""Refresh the stress-location geometric mean without rebuilding geometric
strain and interpolation stencils. This is used after every VOF update."""
function _update_log_dynamic_viscosity!(
        operator::VariableStaggeredDiffusionOperator{T},
        cell_log_viscosity::AbstractVector,
    ) where {T}
    length(cell_log_viscosity) == length(operator.cell_work) || throw(
        DimensionMismatch("one log-viscosity value is required per cell"),
    )
    cell_log_viscosity === operator.cell_work ||
        copyto!(operator.cell_work, cell_log_viscosity)
    _parallel_sparse_mul_from_transpose!(
        operator.stress_work, operator.stress_interpolation_transpose,
        operator.cell_work,
    )
    @_parallel for index in 1:length(operator.stress_weights)
        @inbounds operator.stress_weights[index] =
            operator.stress_measures[index] * exp(operator.stress_work[index])
    end
    isempty(operator.stiffness_pointers) && return operator
    stiffness_values = nonzeros(operator.stiffness)
    offsets = operator.stiffness_refresh_offsets
    @_parallel for pointer in 1:length(stiffness_values)
        value = zero(T)
        for index in offsets[pointer]:(offsets[pointer + 1] - 1)
            stress = @inbounds operator.stiffness_stresses[index]
            factor = @inbounds operator.stiffness_factors[index]
            @inbounds value += operator.stress_weights[stress] * factor
        end
        @inbounds stiffness_values[pointer] = value
    end
    return operator
end

function _update_dynamic_viscosity!(
        operator::VariableStaggeredDiffusionOperator{T},
        field::InterfaceField{T}, viscosity::NTuple{2}
    ) where {T}
    operator.mesh === field.mesh ||
        throw(ArgumentError("interface and diffusion operator must share a mesh"))
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    log_gas = log(phases[2])
    log_jump = log(phases[1]) - log_gas
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            index = offset + (j - 1) * nx + i
            operator.cell_work[index] = muladd(
                fraction[i + h, j + h], log_jump, log_gas,
            )
        end
    end
    return _update_log_dynamic_viscosity!(operator, operator.cell_work)
end

function _staggered_diffusion_operator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
        construction_cache = nothing, reuse = nothing,
        assemble_stiffness = true, cache_layer = 1,
    )
    mu = dynamic_viscosity(field; viscosity)
    return _staggered_diffusion_operator(
        field.mesh; viscosity = mu, boundary, construction_cache, reuse,
        assemble_stiffness, cache_layer, geometry = field.geometry,
    )
end


function StaggeredDiffusionOperator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
    )
    return _staggered_diffusion_operator(field; viscosity, boundary)
end
function diffusion!(
        destination::FaceVelocity, velocity::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator; time = 0
    )
    destination.mesh === velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    destination === velocity && throw(ArgumentError("diffusion output must not alias input"))
    _factorized_variable_stiffness_mul!(
        destination.interior, operator, velocity.interior,
        operator.stress_work,
    )
    result = destination.interior
    volumes = operator.volumes
    @_parallel for index in 1:length(result)
        @inbounds result[index] = -result[index] / volumes[index]
    end
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

"""Apply `E' * W_mu * E` with caller-owned stress scratch."""
function _factorized_variable_stiffness_mul!(
        destination::AbstractVector{T},
        operator::VariableStaggeredDiffusionOperator{T},
        source::AbstractVector{T},
        stress_work::AbstractVector{T},
    ) where {T}
    number_of_unknowns = length(operator.volumes)
    length(destination) == length(source) == number_of_unknowns ||
        throw(DimensionMismatch("variable diffusion vector sizes differ"))
    length(stress_work) == length(operator.stress_weights) ||
        throw(DimensionMismatch("variable diffusion stress-work size differs"))
    destination === source && throw(
        ArgumentError("variable diffusion output must not alias its input"),
    )
    _parallel_sparse_mul_from_transpose!(
        stress_work, operator.strain_transpose, source,
    )
    stress_weights = operator.stress_weights
    @_parallel for index in 1:length(stress_work)
        @inbounds stress_work[index] *= stress_weights[index]
    end
    _parallel_sparse_transpose_mul!(destination, operator.strain, stress_work)
    return destination
end

"""Allocation-free fine-grid action for implicit variable-stress diffusion.

This evaluates `M_rho*x + dt/2*E'*W_mu*E*x` without materializing a new
coefficient-dependent sparse matrix.  The assembled matrix remains available
for the geometric hierarchy; this factorized form provides the exact Krylov
action, an equivalence reference, and a low-level matrix-free building block.
"""
mutable struct FactorizedVariableDiffusionOperator{
        T, O <: VariableStaggeredDiffusionOperator{T}, V <: Vector{T},
    }
    operator::O
    inertia::V
    half_step::T
    stress_work::V
end

function FactorizedVariableDiffusionOperator(
        operator::O, inertia::Vector{T}, dt,
    ) where {T, O <: VariableStaggeredDiffusionOperator{T}}
    return FactorizedVariableDiffusionOperator{
        T, O, Vector{T},
    }(
        operator, inertia, T(dt) / 2,
        zeros(T, length(operator.stress_weights)),
    )
end

Base.size(action::FactorizedVariableDiffusionOperator) =
    (length(action.inertia), length(action.inertia))
Base.size(action::FactorizedVariableDiffusionOperator, dimension::Integer) =
    size(action)[dimension]
Base.eltype(::FactorizedVariableDiffusionOperator{T}) where {T} = T
issymmetric(::FactorizedVariableDiffusionOperator) = true

function mul!(
        destination::AbstractVector{T},
        action::FactorizedVariableDiffusionOperator{T},
        source::AbstractVector{T},
    ) where {T}
    _factorized_variable_stiffness_mul!(
        destination, action.operator, source, action.stress_work,
    )
    half_step = action.half_step
    inertia = action.inertia
    @_parallel for index in 1:length(destination)
        @inbounds destination[index] = muladd(
            half_step, destination[index],
            inertia[index] * source[index],
        )
    end
    return destination
end

function _factorized_variable_level_data(
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    number_of_unknowns = length(inertia)
    diagonal = copy(inertia)
    row_sums = copy(inertia)
    stress_row_sums = zeros(T, length(operator.stress_weights))
    rows = rowvals(operator.strain)
    values = nonzeros(operator.strain)
    for column in axes(operator.strain, 2), pointer in nzrange(operator.strain, column)
        @inbounds stress_row_sums[rows[pointer]] += abs(values[pointer])
    end
    half_step = T(dt) / 2
    for column in axes(operator.strain, 2), pointer in nzrange(operator.strain, column)
        stress = @inbounds rows[pointer]
        value = @inbounds values[pointer]
        weight = @inbounds operator.stress_weights[stress]
        @inbounds diagonal[column] += half_step * weight * value * value
        @inbounds row_sums[column] += half_step * abs(value) * weight *
            stress_row_sums[stress]
    end
    inverse_diagonal = similar(diagonal)
    spectral_bound = zero(T)
    for index in eachindex(diagonal)
        diagonal[index] > zero(T) || throw(
            ArgumentError("factorized diffusion operator has a non-positive diagonal"),
        )
        @inbounds spectral_bound = max(
            spectral_bound, row_sums[index] / diagonal[index],
        )
        @inbounds inverse_diagonal[index] = inv(diagonal[index])
    end
    relaxation_weight = iszero(spectral_bound) ? one(T) :
        min(one(T), T(4) / (T(3) * spectral_bound))
    return inverse_diagonal, relaxation_weight
end

function _factorized_variable_diffusion_level(
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
        prolongation,
    ) where {T}
    action = FactorizedVariableDiffusionOperator(operator, inertia, dt)
    inverse_diagonal, relaxation_weight =
        _factorized_variable_level_data(operator, inertia, dt)
    number_of_unknowns = length(inertia)
    return MultigridLevel(
        action, nothing, inertia, prolongation,
        inverse_diagonal, relaxation_weight,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
    )
end

function _refresh_factorized_variable_diffusion_level!(
        level::MultigridLevel,
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    action = level.matrix
    action isa FactorizedVariableDiffusionOperator || throw(
        ArgumentError("expected a factorized variable-diffusion level"),
    )
    action.operator = operator
    action.inertia = inertia
    action.half_step = T(dt) / 2
    resize!(action.stress_work, length(operator.stress_weights))
    inverse_diagonal, relaxation_weight =
        _factorized_variable_level_data(operator, inertia, dt)
    copyto!(level.inverse_diagonal, inverse_diagonal)
    level.relaxation_weight = relaxation_weight
    level.volumes = inertia
    return level
end

function _roundoff_residual_floor!(
        action::FactorizedVariableDiffusionOperator{T}, solution,
        right_hand_side, volumes, temporary,
    ) where {T}
    operator = action.operator
    strain = operator.strain
    rows = rowvals(strain)
    values = nonzeros(strain)
    stress_bound = action.stress_work
    fill!(stress_bound, zero(T))
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        @inbounds stress_bound[rows[pointer]] +=
            abs(values[pointer]) * abs(solution[column])
    end
    for index in eachindex(temporary)
        @inbounds temporary[index] =
            action.inertia[index] * abs(solution[index])
    end
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        stress = @inbounds rows[pointer]
        @inbounds temporary[column] += action.half_step *
            abs(values[pointer]) * operator.stress_weights[stress] *
            stress_bound[stress]
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds scale = max(
            scale, temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index]),
        )
    end
    return T(128) * eps(T) * scale
end

"""Rediscretized full-vector diffusion multigrid on pruned block trees.

Each coarse level has its own staggered strain/stress discretization. Cell
log-viscosity is volume averaged, staggered density is area averaged on
coincident faces, and the shifted mass matrix is then assembled directly on
the coarse mesh. Thus no level forms `P' * A * P`. The face transfer is a
topology-only staggered interpolation between the nested tree meshes. Once
the root block is reached, its sparse coupled operator is factorized directly.
"""
mutable struct TreeVariableDiffusionPreconditioner{T, F, L, C} <:
    AbstractGeometricMultigridPreconditioner{T}
    operators::Vector{VariableStaggeredDiffusionOperator{T}}
    levels::Vector{L}
    cell_restrictions::Vector{CellCoefficientRestriction{T}}
    density_restrictions::Vector{FaceCoefficientRestriction{T}}
    densities::Vector{Vector{T}}
    construction_cache::C
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
end

_preconditioner_name(::TreeVariableDiffusionPreconditioner) =
    :tree_variable_diffusion_multigrid

function _cell_viscosity_from_log!(field::CellField{T}, values) where {T}
    length(values) == ncells(field.mesh) || throw(
        DimensionMismatch("one log-viscosity value is required per cell"),
    )
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        field_values = @inbounds field.blocks[b]
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            index = offset + (j - 1) * nx + i
            field_values[i + h, j + h] = exp(T(values[index]))
        end
    end
    return field
end

function _tree_variable_diffusion_preconditioner(
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = (;), reuse = nothing,
    ) where {T}
    all(key -> key === :pruning, keys(options)) || throw(
        ArgumentError(
            "tree-geometric diffusion options currently support only pruning",
        ),
    )
    pruning = get(options, :pruning, :all)
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    isnothing(reuse) ||
        reuse isa TreeVariableDiffusionPreconditioner{T} || throw(
        ArgumentError("reused diffusion hierarchy has the wrong type"),
    )
    can_reuse = !isnothing(reuse) && reuse.pruning === pruning &&
        typeof(reuse.operators[1].boundary) === typeof(operator.boundary)

    operators = VariableStaggeredDiffusionOperator{T}[operator]
    matrices = Any[matrix]
    inertias = Vector{T}[inertia]
    fine_density = isnothing(reuse) ? T[] : reuse.densities[1]
    resize!(fine_density, length(inertia))
    for index in eachindex(fine_density)
        @inbounds fine_density[index] = inertia[index] / operator.volumes[index]
    end
    densities = Vector{T}[fine_density]
    prolongations = SparseMatrixCSC{T, Int}[]
    cell_restrictions = CellCoefficientRestriction{T}[]
    density_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    construction_cache = isnothing(reuse) ?
        VariableDiffusionConstructionCache(T) : reuse.construction_cache
    target_size = 2coarse_size
    reused_new_start = 0
    reused_old_start = 0
    matrix_free_fine = isnothing(matrix)
    initial_matching_level = can_reuse && !matrix_free_fine ?
        _matching_tree_level(reuse.operators, current_mesh, 1) : 0
    initial_suffix_length = iszero(initial_matching_level) ? 0 :
        length(reuse.operators) - initial_matching_level + 1
    if !iszero(initial_matching_level) &&
            initial_suffix_length <= max_levels
        reused_new_start = 1
        reused_old_start = initial_matching_level
        fine_density = reuse.densities[initial_matching_level]
        resize!(fine_density, length(inertia))
        for index in eachindex(fine_density)
            @inbounds fine_density[index] =
                inertia[index] / operator.volumes[index]
        end
        densities[1] = fine_density
        for old_index in (initial_matching_level + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
            push!(matrices, reuse.levels[old_index].matrix)
            push!(inertias, reuse.levels[old_index].volumes)
            push!(densities, reuse.densities[old_index])
        end
        for old_index in initial_matching_level:length(
                reuse.cell_restrictions
            )
            push!(cell_restrictions, reuse.cell_restrictions[old_index])
            push!(
                density_restrictions,
                reuse.density_restrictions[old_index],
            )
            push!(prolongations, reuse.levels[old_index].prolongation)
        end
    end

    while iszero(reused_new_start) &&
            length(current_mesh.faces) > target_size && active_level > 0 &&
            length(operators) < max_levels
        coarse_mesh = if pruning === :deepest
            active_level -= 1
            _capped_tree_mesh(operator.mesh, active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        if length(coarse_mesh.faces) >= length(current_mesh.faces)
            pruning === :all && break
            continue
        end

        prolongation = _tree_face_prolongation(
            current_mesh, coarse_mesh, T,
        )
        cell_restriction = _cell_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        density_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        matching_level = can_reuse ?
            _matching_tree_level(reuse.operators, coarse_mesh, 1) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(operators) + suffix_length <= max_levels
        if can_append_suffix
            coarse_operator = reuse.operators[matching_level]
            coarse_density = reuse.densities[matching_level]
            coarse_matrix = reuse.levels[matching_level].matrix
            coarse_inertia = reuse.levels[matching_level].volumes
        else
            coarse_log_viscosity = zeros(T, ncells(coarse_mesh))
            _restrict_cell_coefficient!(
                coarse_log_viscosity, last(operators).cell_work,
                cell_restriction,
            )
            coarse_viscosity = CellField(coarse_mesh, T)
            _cell_viscosity_from_log!(
                coarse_viscosity, coarse_log_viscosity,
            )
            coarse_operator = _staggered_diffusion_operator(
                coarse_mesh; viscosity = coarse_viscosity,
                boundary = operator.boundary,
                construction_cache,
                cache_layer = length(operators),
            )
            coarse_density = zeros(T, length(coarse_mesh.faces))
            _restrict_face_coefficients!(
                coarse_density, last(densities), density_restriction,
            )
            coarse_matrix, coarse_inertia = _variable_diffusion_matrix(
                coarse_operator, coarse_density, dt,
            )
        end
        push!(prolongations, prolongation)
        push!(cell_restrictions, cell_restriction)
        push!(density_restrictions, density_restriction)
        push!(operators, coarse_operator)
        push!(matrices, coarse_matrix)
        push!(inertias, coarse_inertia)
        push!(densities, coarse_density)
        current_mesh = coarse_mesh
        if can_append_suffix
            reused_new_start = length(operators)
            reused_old_start = matching_level
            for old_index in (matching_level + 1):length(reuse.operators)
                push!(operators, reuse.operators[old_index])
                push!(matrices, reuse.levels[old_index].matrix)
                push!(inertias, reuse.levels[old_index].volumes)
                push!(densities, reuse.densities[old_index])
            end
            for old_index in matching_level:length(reuse.cell_restrictions)
                push!(
                    cell_restrictions,
                    reuse.cell_restrictions[old_index],
                )
                push!(
                    density_restrictions,
                    reuse.density_restrictions[old_index],
                )
                push!(
                    prolongations,
                    reuse.levels[old_index].prolongation,
                )
            end
            break
        end
    end

    levels = _multigrid_levels(T, matrix_free_fine)
    for index in eachindex(matrices)
        prolongation = index < length(matrices) ?
            prolongations[index] :
            spzeros(T, length(inertias[index]), 0)
        if matrix_free_fine && index == 1
            length(matrices) > 1 || throw(
                ArgumentError(
                    "a matrix-free finest diffusion level requires a coarser tree level",
                ),
            )
            push!(
                levels,
                _factorized_variable_diffusion_level(
                    operators[index], inertias[index], dt, prolongation,
                ),
            )
        elseif !iszero(reused_new_start) && index >= reused_new_start
            old_index = reused_old_start + index - reused_new_start
            level = reuse.levels[old_index]
            level.matrix = matrices[index]
            level.volumes = inertias[index]
            level.prolongation = prolongation
            push!(levels, level)
        else
            push!(
                levels,
                _multigrid_level(
                    matrices[index], inertias[index], prolongation,
                ),
            )
        end
    end
    coarse = last(levels)
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    else
        factorization = lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    preconditioner = TreeVariableDiffusionPreconditioner(
        operators, levels, cell_restrictions, density_restrictions,
        densities, construction_cache,
        factorization, false, coarse_rhs, coarse_solution,
        pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(operators) - reused_new_start + 1,
    )
    !iszero(reused_new_start) &&
        _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner, operator, matrix, inertia, dt,
    )
    return preconditioner
end

function _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner::TreeVariableDiffusionPreconditioner{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt,
    ) where {T}
    preconditioner.operators[1] = operator
    fine_density = preconditioner.densities[1]
    for index in eachindex(fine_density)
        @inbounds fine_density[index] =
            inertia[index] / operator.volumes[index]
    end
    if preconditioner.levels[1].matrix isa
            FactorizedVariableDiffusionOperator
        _refresh_factorized_variable_diffusion_level!(
            preconditioner.levels[1], operator, inertia, dt,
        )
    else
        preconditioner.levels[1].matrix = matrix
        preconditioner.levels[1].volumes = inertia
    end

    for index in 2:length(preconditioner.operators)
        fine_operator = preconditioner.operators[index - 1]
        coarse_operator = preconditioner.operators[index]
        _restrict_cell_coefficient!(
            coarse_operator.cell_work, fine_operator.cell_work,
            preconditioner.cell_restrictions[index - 1],
        )
        _update_log_dynamic_viscosity!(
            coarse_operator, coarse_operator.cell_work,
        )
        _restrict_face_coefficients!(
            preconditioner.densities[index],
            preconditioner.densities[index - 1],
            preconditioner.density_restrictions[index - 1],
        )
        coarse_inertia = preconditioner.levels[index].volumes
        coarse_density = preconditioner.densities[index]
        for face in eachindex(coarse_inertia)
            @inbounds coarse_inertia[face] =
                coarse_density[face] * coarse_operator.volumes[face]
        end
        _refresh_variable_diffusion_matrix_from_inertia!(
            preconditioner.levels[index].matrix,
            coarse_operator, coarse_inertia, dt,
        )
    end

    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        level.matrix isa FactorizedVariableDiffusionOperator && continue
        _refresh_multigrid_level!(
            level, level.matrix, level.volumes, level.prolongation,
        )
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_factorization = lu(coarse.matrix)
    return preconditioner
end

function _variable_diffusion_preconditioner(
        specification, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options, reuse = nothing,
    )
    if specification in (:geometric, :tree_geometric)
        return _tree_variable_diffusion_preconditioner(
            operator, matrix, inertia, dt;
            coarse_size, pre_sweeps, post_sweeps, max_levels, options, reuse,
        )
    end
    isnothing(reuse) || throw(
        ArgumentError("hierarchy reuse requires a tree-geometric preconditioner"),
    )
    return _construct_matrix_preconditioner(
        specification, matrix;
        symmetric = true, singular = false, options
    )
end

mutable struct VariableStaggeredDiffusionSolver{
        T, K, KO, P, PO, A, KW, PC,
    }
    krylov_factory::K
    krylov_options::KO
    preconditioner_specification::P
    preconditioner_options::PO
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    matrix::Union{Nothing, SparseMatrixCSC{T, Int}}
    action::A
    krylov::KW
    preconditioner::PC
    inertia::Vector{T}
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end

function _variable_diffusion_matrix(operator, density, dt)
    T = eltype(operator.volumes)
    length(density) == length(operator.volumes) ||
        throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho) && rho > zero(T), density) ||
        throw(ArgumentError("density must be finite and positive"))
    inertia = T.(density) .* operator.volumes
    matrix = copy(operator.stiffness)
    matrix.nzval .*= T(dt) / 2
    rows = rowvals(matrix)
    for column in axes(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix.nzval[diagonal] += inertia[column]
    end
    return matrix, inertia
end

function _refresh_variable_diffusion_matrix_from_inertia!(
        matrix::SparseMatrixCSC{T, Int},
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    length(inertia) == length(operator.volumes) || throw(
        DimensionMismatch("one inertia value per staggered volume is required"),
    )
    matrix.colptr == operator.stiffness.colptr &&
        rowvals(matrix) == rowvals(operator.stiffness) || throw(
        ArgumentError("diffusion matrix and stiffness sparsity differ"),
    )
    matrix_values = nonzeros(matrix)
    stiffness_values = nonzeros(operator.stiffness)
    half_step = T(dt) / 2
    @_parallel for index in 1:length(matrix_values)
        @inbounds matrix_values[index] = half_step * stiffness_values[index]
    end
    rows = rowvals(matrix)
    @_parallel for column in 1:size(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix_values[diagonal] += inertia[column]
    end
    return matrix
end

function _refresh_variable_diffusion_matrix!(
        solver::VariableStaggeredDiffusionSolver{T}, operator,
        density, dt
    ) where {T}
    length(density) == length(operator.volumes) || throw(
        DimensionMismatch("one density per staggered volume is required"),
    )
    all(rho -> isfinite(rho) && rho > zero(T), density) || throw(
        ArgumentError("density must be finite and positive"),
    )
    @_parallel for column in 1:length(solver.inertia)
        @inbounds solver.inertia[column] =
            T(density[column]) * operator.volumes[column]
    end
    return _refresh_variable_diffusion_matrix_from_inertia!(
        solver.matrix, operator, solver.inertia, dt,
    )
end

function VariableStaggeredDiffusionSolver(
        operator::VariableStaggeredDiffusionOperator{T}, density, dt;
        krylov = Krylov.CgWorkspace, krylov_options = (;),
        preconditioner = :tree_geometric, preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32, reuse_preconditioner = nothing,
        reuse_solver = nothing,
    ) where {T}
    matrix_free_fine = size(operator.stiffness) !=
        (length(operator.volumes), length(operator.volumes))
    matrix, inertia = if matrix_free_fine
        nothing, T.(density) .* operator.volumes
    else
        _variable_diffusion_matrix(operator, density, dt)
    end
    right_hand_side = zeros(T, length(inertia))
    action = FactorizedVariableDiffusionOperator(operator, inertia, dt)
    krylov_workspace = _construct_krylov(
        krylov, action, right_hand_side
    )
    built_preconditioner = _variable_diffusion_preconditioner(
        preconditioner, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = preconditioner_options,
        reuse = reuse_preconditioner,
    )
    if !isnothing(reuse_solver)
        reconfigured = _reconfigure_variable_diffusion_solver!(
            reuse_solver, operator, matrix, inertia, dt,
            built_preconditioner,
        )
        isnothing(reconfigured) || return reconfigured
    end
    return VariableStaggeredDiffusionSolver{
        T, typeof(krylov), typeof(krylov_options),
        typeof(preconditioner), typeof(preconditioner_options),
        typeof(action),
        typeof(krylov_workspace), typeof(built_preconditioner),
    }(
        krylov, krylov_options, preconditioner,
        preconditioner_options, coarse_size, pre_sweeps,
        post_sweeps, max_levels, matrix, action, krylov_workspace,
        built_preconditioner, inertia, zeros(T, length(inertia)),
        right_hand_side, zeros(T, length(inertia)),
        zeros(T, length(inertia)),
        LinearSolverStats(
            T; method = _krylov_method(krylov_workspace),
            preconditioner = _preconditioner_name(built_preconditioner)
        )
    )
end

"""Retarget package-owned diffusion solver storage after an AMR regrid."""
function _reconfigure_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix, inertia, dt,
        preconditioner,
    ) where {T}
    typeof(operator) === typeof(solver.action.operator) || return nothing
    typeof(preconditioner) === typeof(solver.preconditioner) || return nothing
    solver.matrix = matrix
    solver.inertia = inertia
    action = solver.action
    action.operator = operator
    action.inertia = inertia
    action.half_step = T(dt) / 2
    resize!(action.stress_work, length(operator.stress_weights))
    fill!(action.stress_work, zero(T))
    number_of_unknowns = length(inertia)
    for values in (
            solver.solution, solver.right_hand_side,
            solver.residual, solver.product,
        )
        resize!(values, number_of_unknowns)
        fill!(values, zero(T))
    end
    krylov = _resize_krylov_workspace!(solver.krylov, number_of_unknowns)
    isnothing(krylov) && (
        krylov = _construct_krylov(
            solver.krylov_factory, action, solver.right_hand_side,
        )
    )
    typeof(krylov) === typeof(solver.krylov) || return nothing
    solver.krylov = krylov
    solver.preconditioner = preconditioner
    solver.stats = LinearSolverStats(
        T; method = _krylov_method(krylov),
        preconditioner = _preconditioner_name(preconditioner),
    )
    return solver
end

function _update_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver,
        operator, density, dt
    )
    matrix = if isnothing(solver.matrix)
        length(density) == length(operator.volumes) || throw(
            DimensionMismatch("one density per staggered volume is required"),
        )
        for column in eachindex(solver.inertia)
            @inbounds solver.inertia[column] =
                eltype(solver.inertia)(density[column]) * operator.volumes[column]
        end
        nothing
    else
        _refresh_variable_diffusion_matrix!(solver, operator, density, dt)
    end
    solver.action.operator = operator
    solver.action.half_step = eltype(solver.inertia)(dt) / 2
    if solver.preconditioner_specification in (:geometric, :tree_geometric) &&
            solver.preconditioner isa TreeVariableDiffusionPreconditioner
        _refresh_tree_variable_diffusion_preconditioner!(
            solver.preconditioner, operator, matrix, solver.inertia, dt,
        )
    else
        solver.preconditioner = _variable_diffusion_preconditioner(
            solver.preconditioner_specification, operator, matrix,
            solver.inertia, dt;
            coarse_size = solver.coarse_size,
            pre_sweeps = solver.pre_sweeps,
            post_sweeps = solver.post_sweeps,
            max_levels = solver.max_levels,
            options = solver.preconditioner_options
        )
    end
    return solver
end

function _solve_variable_diffusion!(
        destination, solver::VariableStaggeredDiffusionSolver,
        operator; tolerance, max_cycles
    )
    T = eltype(solver.solution)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))
    copyto!(solver.solution, destination.interior)
    mul!(solver.product, solver.action, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    # Divide the integrated momentum residual by the staggered inertia. This
    # gives a velocity defect for every density and matches the single-phase
    # shifted solve's strong residual.
    volumes = solver.inertia
    initial_l2, initial_linf = _integrated_residual_norms(
        solver.residual, volumes
    )
    roundoff_floor = _roundoff_residual_floor!(
        solver.action, solver.solution, solver.right_hand_side,
        volumes, solver.product
    )
    residual_scale = _strong_maximum(solver.right_hand_side, volumes)
    target = max(dimensionless_tolerance * residual_scale, roundoff_floor)
    iterations = 0
    if initial_linf > target
        iterations = _run_krylov_reliably!(
            solver.krylov, solver.action, solver.right_hand_side,
            solver.solution, solver.preconditioner, target,
            volumes, max_cycles, solver.krylov_options,
            solver.residual, solver.product,
        )
    end
    mul!(solver.product, solver.action, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    final_l2, final_linf = _integrated_residual_norms(
        solver.residual, volumes
    )
    final_target = max(
        target,
        _roundoff_residual_floor!(
            solver.action, solver.solution,
            solver.right_hand_side, volumes, solver.product
        )
    )
    copyto!(destination.interior, solver.solution)
    fill!(destination.boundary, zero(T))
    stats = solver.stats
    stats.cycles = iterations
    stats.converged = final_linf <= final_target
    stats.initial_l2 = initial_l2
    stats.initial_linf = initial_linf
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.target_linf = final_target
    stats.residual_scale = residual_scale
    stats.removed_rhs_mean = zero(T)
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""Implicit midpoint of the coupled stress operator, with an optional positive
staggered density. `old_velocity` supplies the explicit half of the stress;
it defaults to `source` for a standalone diffusion substep. The sparse solve
is rebuilt here when coefficients change."""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity, source::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator, dt;
        old_velocity = source,
        density = ones(eltype(source), length(source.interior))
    )
    destination.mesh === source.mesh === old_velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt)&&dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    length(density) == length(operator.volumes) || throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("density must be finite and positive"))
    inertia = density .* operator.volumes
    matrix = spdiagm(0 => inertia) + (dt / 2) * operator.stiffness
    rhs = inertia .* source.interior -
        (dt / 2) * (operator.stiffness * old_velocity.interior)
    destination.interior .= matrix \ rhs
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, source::FaceVelocity{T},
        operator::VariableStaggeredDiffusionOperator{T}, dt,
        solver::VariableStaggeredDiffusionSolver{T};
        old_velocity = source,
        density = ones(T, length(source.interior)),
        tolerance = sqrt(eps(T)), max_cycles = 500
    ) where {T}
    destination.mesh === source.mesh === old_velocity.mesh ===
        operator.mesh ||
        throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt) && dt > zero(T) ||
        throw(ArgumentError("dt must be finite and positive"))
    _update_variable_diffusion_solver!(
        solver, operator, density, T(dt)
    )
    _factorized_variable_stiffness_mul!(
        solver.product, operator, old_velocity.interior,
        solver.action.stress_work,
    )
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] =
            solver.inertia[index] * source.interior[index] -
            T(dt) * solver.product[index] / 2
    end
    stats = _solve_variable_diffusion!(
        destination, solver, operator; tolerance, max_cycles
    )
    stats.converged ||
        throw(
        ErrorException(
            _solver_failure_message(
                "variable staggered diffusion solve", stats, tolerance;
                details = "dt=$(dt), max_cycles=$(max_cycles)",
            )
        )
    )
    return destination
end
