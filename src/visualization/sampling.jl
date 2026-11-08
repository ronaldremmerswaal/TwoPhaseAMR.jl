"""
    visualization_values(field)

Copy the owned (non-halo) values of a cell field into the same block-major,
`j`, `i` ordering as `visualization_geometry(mesh).cells`.
"""
function visualization_values(field::CellField{T}) where {T}
    nx, ny = field.mesh.block_size
    values = Vector{T}(undef, ncells(field.mesh))
    index = 0
    for block in eachindex(field.blocks)
        owned = interior(field, block)
        for j in 1:ny, i in 1:nx
            index += 1
            @inbounds values[index] = owned[i, j]
        end
    end
    return values
end

"""Copy owned directional face values in the order used by face geometry."""
function visualization_values(field::FaceField{Axis, T}) where {Axis, T}
    values = Vector{T}(undef, length(field.interior) + length(field.boundary))
    copyto!(values, 1, field.interior, 1, length(field.interior))
    copyto!(
        values, length(field.interior) + 1,
        field.boundary, 1, length(field.boundary)
    )
    return values
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::Face{Axis}
    ) where {T, Axis}
    count = _nfaces(velocity.mesh, Axis) +
        _nboundary_faces(velocity.mesh, Axis)
    values = Vector{T}(undef, count)
    local_index = 0
    for (global_index, face) in pairs(velocity.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.interior[global_index]
    end
    for (global_index, face) in pairs(velocity.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.boundary[global_index]
    end
    return values
end

@inline function _visualization_cell_index(mesh::BlockTreeMesh, cell::CellRef)
    nx, ny = mesh.block_size
    return (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
end

@inline _visualization_side_slot(axis::Int, side::Int) =
    2axis - (side < 0 ? 1 : 0)

function _cell_centered_velocity(velocity::FaceVelocity{T};
        interface = nothing, phase = :liquid) where {T}
    work = CenteredVelocityWorkspace(T)
    geometry = isnothing(interface) ? nothing : interface.geometry
    return _cell_centered_velocity!(work, velocity, interface, geometry, phase)
end

function _cell_centered_velocity!(work::CenteredVelocityWorkspace,
        velocity::FaceVelocity{T}, interface, geometry::G, phase) where {T, G}
    mesh = velocity.mesh
    count = ncells(mesh)
    if size(work.weighted, 2) != count
        work.weighted = zeros(T, 4, count)
        work.measures = zeros(T, 4, count)
    else
        fill!(work.weighted, zero(T))
        fill!(work.measures, zero(T))
    end
    weighted, measures = work.weighted, work.measures
    for (face_index, face) in pairs(mesh.faces)
        isnothing(interface) ||
            _phase_face_is_active(interface, geometry, face_index, face, phase) || continue
        value = @inbounds velocity.interior[face_index]
        isnan(value) && continue
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            index = _visualization_cell_index(mesh, cell)
            slot = _visualization_side_slot(face.axis, side)
            @inbounds weighted[slot, index] += value * face.area
            @inbounds measures[slot, index] += face.area
        end
    end
    for (face_index, face) in pairs(mesh.boundary_faces)
        isnothing(interface) || _phase_cell_is_active(interface, geometry, face.cell, phase) || continue
        index = _visualization_cell_index(mesh, face.cell)
        slot = _visualization_side_slot(face.axis, face.side)
        value = @inbounds velocity.boundary[face_index]
        isnan(value) && continue
        @inbounds weighted[slot, index] += value * face.area
        @inbounds measures[slot, index] += face.area
    end

    x, y = work.x, work.y
    resize!(x, count)
    resize!(y, count)
    for index in 1:count
        present = true
        if !isnothing(interface)
            block, local_index = divrem(index - 1, prod(mesh.block_size))
            i = mod(local_index, mesh.block_size[1]) + 1
            j = local_index ÷ mesh.block_size[1] + 1
            present = _phase_cell_is_active(interface, geometry, CellRef(block + 1, i, j), phase)
        end
        for (component, first_slot) in ((x, 1), (y, 3))
            total = zero(T)
            sides = 0
            for slot in first_slot:(first_slot + 1)
                measures[slot, index] > zero(T) || continue
                total += weighted[slot, index] / measures[slot, index]
                sides += 1
            end
            # With one valid side use constant extrapolation. With no side,
            # or in a cell without this phase, the centered value is absent.
            component[index] = present && sides > 0 ? total / sides : T(NaN)
        end
    end
    return x, y
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::CentredMagnitude; interface = nothing, phase = :liquid,
    ) where {T}
    x, y = _cell_centered_velocity(velocity; interface, phase)
    return hypot.(x, y)
end

function visualization_values(
        fields::Tuple{X, Y},
        derived::Union{CentredMagnitude, Vorticity}
    ) where
    {X <: FaceField{1}, Y <: FaceField{2}}
    fields[1].mesh === fields[2].mesh ||
        throw(ArgumentError("face-field components must share one mesh"))
    return visualization_values(FaceVelocity(fields...), derived)
end

function _cell_centers(mesh::BlockTreeMesh{T}) where {T}
    # Keep this allocating convenience function's independent ownership.
    # Numerical curl loops borrow the mesh cache directly instead.
    return copy(_mesh_cell_connectivity(mesh).centers)
end

"""Extend centered values into the one-cell curl stencil, never phase storage.

Fit an affine velocity to valid phase cells within two neighbour rings. This
reproduces a linear field when that stencil spans two dimensions. If it does
not, use its constant mean; if no donor exists, leave the value missing.
Donors are frozen before extension: absent cells never become physical data.
"""
function _extrapolate_curl_stencil!(x, y, mesh, centers, known)
    return _extrapolate_curl_stencil!(x, y, mesh, centers, known,
        _mesh_cell_connectivity(mesh), Int[])
end

function _extrapolate_curl_stencil!(x, y, mesh, centers, known, connectivity, candidates)
    all(known) && return
    T = eltype(x)
    offsets, neighbours = connectivity.offsets, connectivity.neighbours
    for target in eachindex(x)
        known[target] && continue
        valid_neighbour = false
        for pointer in offsets[target]:(offsets[target + 1] - 1)
            if known[neighbours[pointer]]
                valid_neighbour = true
                break
            end
        end
        valid_neighbour || continue
        empty!(candidates)
        for pointer in offsets[target]:(offsets[target + 1] - 1)
            neighbour = neighbours[pointer]
            push!(candidates, neighbour)
            for second in offsets[neighbour]:(offsets[neighbour + 1] - 1)
                push!(candidates, neighbours[second])
            end
        end
        sort!(candidates)
        unique!(candidates)
        sw = sx = sy = sxx = sxy = syy = zero(T)
        su = sv = sxu = syu = sxv = syv = zero(T)
        for donor in candidates
            known[donor] || continue
            dx = T(_wrapped_delta(centers[donor][1] - centers[target][1],
                mesh.extent[1], mesh.periodic[1]))
            dy = T(_wrapped_delta(centers[donor][2] - centers[target][2],
                mesh.extent[2], mesh.periodic[2]))
            distance_squared = dx^2 + dy^2
            distance_squared > zero(T) || continue
            w = inv(distance_squared)
            u, v = x[donor], y[donor]
            sw += w; sx += w * dx; sy += w * dy
            sxx += w * dx^2; sxy += w * dx * dy; syy += w * dy^2
            su += w * u; sv += w * v
            sxu += w * dx * u; syu += w * dy * u
            sxv += w * dx * v; syv += w * dy * v
        end
        iszero(sw) && continue
        mx, my = sx / sw, sy / sw
        u, v = su / sw, sv / sw
        a, b, c = sxx - sx * mx, sxy - sx * my, syy - sy * my
        determinant = a * c - b^2
        if determinant > 128eps(T) * max(abs(a * c), b^2)
            ux, uy = sxu - sx * u, syu - sy * u
            vx, vy = sxv - sx * v, syv - sy * v
            u -= ((c * ux - b * uy) * mx + (a * uy - b * ux) * my) / determinant
            v -= ((c * vx - b * vy) * mx + (a * vy - b * vx) * my) / determinant
        end
        x[target], y[target] = u, v
    end
end

function visualization_values(velocity::FaceVelocity{T}, ::Vorticity;
        interface = nothing, phase = :liquid) where {T}
    return _vorticity_values!(VorticityWorkspace(T), velocity; interface, phase)
end

function _vorticity_values!(work, velocity; interface = nothing, phase = :liquid)
    geometry = isnothing(interface) ? nothing : interface.geometry
    return _vorticity_values!(work, velocity, interface, geometry, phase)
end

function _vorticity_values!(work::VorticityWorkspace,
        velocity::FaceVelocity{T}, interface, geometry::G, phase) where {T, G}
    mesh = velocity.mesh
    xvelocity, yvelocity = _cell_centered_velocity!(work.centered,
        velocity, interface, geometry, phase)
    return _vorticity_from_centered!(work, mesh)
end

# The visualization capture shares centring between components, magnitude and
# curl. Curl extends only this private scratch stencil, never intrinsic data.
function _vorticity_from_centered!(work::VorticityWorkspace{T}, mesh::BlockTreeMesh) where {T}
    xvelocity, yvelocity = work.centered.x, work.centered.y
    connectivity = _mesh_cell_connectivity(mesh)
    centers = connectivity.centers
    count = length(centers)
    known = work.known
    resize!(known, count)
    for index in eachindex(known)
        known[index] = isfinite(xvelocity[index]) && isfinite(yvelocity[index])
    end
    _extrapolate_curl_stencil!(xvelocity, yvelocity, mesh, centers, known,
        connectivity, work.candidates)
    for values in work.fit
        resize!(values, count)
        fill!(values, zero(T))
    end
    a11, a12, a22, bx1, bx2, by1, by2 = work.fit

    for face in mesh.faces
        minus = _visualization_cell_index(mesh, face.minus)
        plus = _visualization_cell_index(mesh, face.plus)
        (isfinite(xvelocity[minus]) && isfinite(xvelocity[plus]) &&
            isfinite(yvelocity[minus]) && isfinite(yvelocity[plus])) || continue
        dx = T(
            _wrapped_delta(
                centers[plus][1] - centers[minus][1],
                mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                centers[plus][2] - centers[minus][2],
                mesh.extent[2], mesh.periodic[2]
            )
        )
        distance_squared = dx^2 + dy^2
        distance_squared > zero(T) || continue
        weight = T(face.area) / distance_squared
        du = xvelocity[plus] - xvelocity[minus]
        dv = yvelocity[plus] - yvelocity[minus]
        for index in (minus, plus)
            @inbounds a11[index] += weight * dx^2
            @inbounds a12[index] += weight * dx * dy
            @inbounds a22[index] += weight * dy^2
            @inbounds bx1[index] += weight * du * dx
            @inbounds bx2[index] += weight * du * dy
            @inbounds by1[index] += weight * dv * dx
            @inbounds by2[index] += weight * dv * dy
        end
    end

    result = work.values
    resize!(result, count)
    for index in 1:count
        if !known[index]
            result[index] = T(NaN)
            continue
        end
        determinant = @inbounds a11[index] * a22[index] - a12[index]^2
        scale = @inbounds max(abs(a11[index] * a22[index]), a12[index]^2)
        if determinant <= 128eps(T) * scale
            result[index] = T(NaN)
            continue
        end
        dvdx = @inbounds (
            a22[index] * by1[index] -
                a12[index] * by2[index]
        ) / determinant
        dudy = @inbounds (
            a11[index] * bx2[index] -
                a12[index] * bx1[index]
        ) / determinant
        @inbounds result[index] = dvdx - dudy
    end
    return result
end

function _phase_centered_components!(registry, cache, velocity, interface, phase,
        key = (velocity, interface, phase))
    return get!(cache, key) do
        work = get!(registry.sampling_workspaces, key) do
            VorticityWorkspace(eltype(velocity))
        end
        x, y = _cell_centered_velocity!(work.centered, velocity, interface,
            interface.geometry, phase)
        # Owned snapshots, independent of the reusable curl/centring scratch.
        (copy(x), copy(y))
    end
end

function _phase_derived_values!(registry, cache, velocity, interface, phase, derived, key)
    x, y = _phase_centered_components!(registry, cache, velocity, interface, phase, key)
    derived isa CentredMagnitude && return hypot.(x, y)
    work = registry.sampling_workspaces[key]
    copyto!(work.centered.x, x)
    copyto!(work.centered.y, y)
    return copy(_vorticity_from_centered!(work, velocity.mesh))
end

function _center_phase_faces(field::FaceField{Axis, T}, interface, phase) where {Axis, T}
    return _center_phase_faces(field, interface, interface.geometry, phase)
end

function _center_phase_faces(field::FaceField{Axis, T}, interface,
        geometry::G, phase) where {Axis, T, G}
    mesh = field.mesh
    count = ncells(mesh)
    weighted = zeros(T, 2, count)
    measures = zeros(T, 2, count)
    local_index = 0
    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        _phase_face_is_active(interface, geometry, global_index, face, phase) || continue
        value = field.interior[local_index]
        isnan(value) && continue
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            index = _visualization_cell_index(mesh, cell)
            slot = side < 0 ? 1 : 2
            weighted[slot, index] += value * face.area
            measures[slot, index] += face.area
        end
    end
    local_index = 0
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        local_index += 1
        _phase_cell_is_active(interface, geometry, face.cell, phase) || continue
        value = field.boundary[local_index]
        isnan(value) && continue
        index = _visualization_cell_index(mesh, face.cell)
        slot = face.side < 0 ? 1 : 2
        weighted[slot, index] += value * face.area
        measures[slot, index] += face.area
    end
    values = Vector{T}(undef, count)
    nx, ny = mesh.block_size
    for b in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(b, i, j)
        index = _visualization_cell_index(mesh, ref)
        total = zero(T)
        sides = 0
        for slot in 1:2
            measures[slot, index] > zero(T) || continue
            total += weighted[slot, index] / measures[slot, index]
            sides += 1
        end
        # As for velocity: one valid side gives constant extrapolation; never
        # substitute the other phase for missing intrinsic values.
        values[index] = sides > 0 && _phase_cell_is_active(interface, geometry, ref, phase) ?
            total / sides : T(NaN)
    end
    return values
end

function _cell_phase_values!(registry, source::PhaseVisualizationSource{<:PhaseScalar},
        replacements, cache)
    return _phase_visualization_values(source, replacements)
end

function _cell_phase_values!(registry, source::PhaseVisualizationSource{<:PhaseFaceScalar},
        replacements, cache)
    scalar = source.quantity
    interface = _visualization_sampling_interface(scalar.interface, replacements)
    return TwoPhaseVisualizationValues(
        _center_phase_faces(_visualization_replacement(replacements, scalar.liquid), interface, :liquid),
        _center_phase_faces(_visualization_replacement(replacements, scalar.gas), interface, :gas),
    )
end

function _cell_phase_values!(registry, source::PhaseVelocityComponentVisualizationSource{Axis},
        replacements, cache) where {Axis}
    interface = _visualization_sampling_interface(source.interface, replacements)
    values = map((:liquid, :gas), source.velocity) do phase, live_velocity
        velocity = _visualization_replacement(replacements, live_velocity)
        components = _phase_centered_components!(registry, cache, velocity, interface, phase,
            (live_velocity, source.interface, phase))
        components[Axis]
    end
    return TwoPhaseVisualizationValues(values...)
end

function _cell_phase_values!(registry, source::PhaseVelocityDerivedVisualizationSource,
        replacements, cache)
    interface = _visualization_sampling_interface(source.interface, replacements)
    values = map((:liquid, :gas), source.velocity) do phase, live_velocity
        velocity = _visualization_replacement(replacements, live_velocity)
        _phase_derived_values!(registry, cache, velocity, interface, phase, source.derived,
            (live_velocity, source.interface, phase))
    end
    return TwoPhaseVisualizationValues(values...)
end

function _majority_phase_values(values::TwoPhaseVisualizationValues, interface, replacements)
    return _majority_phase_values(values, interface,
        _visualization_replacement(replacements, interface.conditional_fraction), interface.geometry)
end

function _majority_phase_values(values, interface, fraction, geometry::G) where {G}
    result = similar(values.liquid)
    nx, ny = interface.mesh.block_size
    for b in eachindex(interface.mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(b, i, j)
        index = _visualization_cell_index(interface.mesh, ref)
        # In embedded cells the fraction is relative to open fluid volume;
        # solid cells have no intrinsic fluid value and stay transparent.
        result[index] = !isnothing(geometry) && iszero(_get(geometry.open_fraction, ref)) ?
            eltype(result)(NaN) : _get(fraction, ref) >= 0.5 ?
            values.liquid[index] : values.gas[index]
    end
    return result
end

function _render_source_values(registry, source, mode::Symbol;
        replacements = IdDict{Any, Any}(), cache = Dict{Tuple{Any, Any, Symbol}, Any}())
    _is_discontinuous_source(source) || return _source_values(source)
    if mode == :cell
        values = _cell_phase_values!(registry, source, replacements, cache)
        interface = _visualization_sampling_interface(_source_interface(source), replacements)
        return _majority_phase_values(values, interface, interface.conditional_fraction, interface.geometry)
    end
    # Centred derived fields share the same scratch in both display modes.
    source isa PhaseVelocityDerivedVisualizationSource &&
        return _cell_phase_values!(registry, source, replacements, cache)
    return source isa PhaseVisualizationSource ?
        _phase_visualization_values(source, replacements) :
        _phase_velocity_visualization_values(source, replacements)
end

function _visualization_vertex_values(field::CellField)
    cell_values = visualization_values(field)
    values = Vector{eltype(cell_values)}(undef, 4length(cell_values))
    for index in eachindex(cell_values)
        value = @inbounds cell_values[index]
        first_vertex = 4index - 3
        @inbounds for vertex in first_vertex:(first_vertex + 3)
            values[vertex] = value
        end
    end
    return values
end

function _visualization_vertex_values(values::AbstractVector)
    result = Vector{eltype(values)}(undef, 4length(values))
    for index in eachindex(values)
        first_vertex = 4index - 3
        @inbounds result[first_vertex] = values[index]
        @inbounds result[first_vertex + 1] = values[index]
        @inbounds result[first_vertex + 2] = values[index]
        @inbounds result[first_vertex + 3] = values[index]
    end
    return result
end

function _visualization_face_vertex_values(
        values::AbstractVector, geometry::AMRFaceGeometry
    )
    result = Vector{eltype(values)}(undef, 4length(geometry.faces))
    for tile in eachindex(geometry.faces)
        value = values[geometry.value_indices[tile]]
        first_vertex = 4tile - 3
        @inbounds result[first_vertex] = value
        @inbounds result[first_vertex + 1] = value
        @inbounds result[first_vertex + 2] = value
        @inbounds result[first_vertex + 3] = value
    end
    return result
end

"""
    cell_for_vertex(geometry, vertex_index)

Return the cell owning a picked mesh vertex, or `nothing` for an invalid
index. Makie reports the largest vertex index of a picked mesh triangle, which
is compatible with this four-private-vertices-per-cell mapping.
"""
function cell_for_vertex(geometry::AMRCellGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.cells[cld(vertex_index, 4)]
end

"""Return the face owning a picked dual-cell vertex, or `nothing`."""
function face_for_vertex(geometry::AMRFaceGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.faces[cld(vertex_index, 4)]
end

function cell_for_vertex(
        geometry::PLICVisualizationGeometry{CellRef}, vertex_index::Integer,
    )
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.owners[vertex_index]
end

function face_for_vertex(
        geometry::PLICVisualizationGeometry{FaceRef}, vertex_index::Integer,
    )
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.owners[vertex_index]
end

function _face_outline(
        geometry::AMRFaceGeometry,
        vertex_index::Integer
    )
    1 <= vertex_index <= length(geometry.points) ||
        return eltype(geometry.points)[]
    first_vertex = 4cld(vertex_index, 4) - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end

function _face_outline(geometry::AMRFaceGeometry, reference::FaceRef)
    tile = findfirst(==(reference), geometry.faces)
    isnothing(tile) && return eltype(geometry.points)[]
    return _face_outline(geometry, 4tile)
end

function _cell_outline(geometry::AMRCellGeometry, cell::CellRef)
    nx, ny = geometry.block_size
    1 <= cell.block || return eltype(geometry.points)[]
    1 <= cell.i <= nx || return eltype(geometry.points)[]
    1 <= cell.j <= ny || return eltype(geometry.points)[]
    cell_index = (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
    cell_index <= length(geometry.cells) || return eltype(geometry.points)[]
    first_vertex = 4cell_index - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end
