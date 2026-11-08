"""
    visualization_values(field)
    visualization_values(velocity::FaceVelocity, Face{Axis}())
    visualization_values(velocity, CentredMagnitude() | Vorticity();
                         interface=nothing, phase=:liquid)

Copy the owned (non-halo) values of a field. Cell values are block-major, then
`j`, then `i`, the order used by frames and [`rasterize`](@ref). Face values
of one direction list the interior faces in schedule order, then the boundary
faces. Derived quantities are cell-centred.
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

"""Copy owned directional face values: interior faces, then boundary faces."""
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
    return _cell_index(cell, nx, ny)
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

# Centred phase magnitude or vorticity of both phases. Phases share centring
# work through `cache` within one capture.
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
