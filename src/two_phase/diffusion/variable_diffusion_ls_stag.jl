# Two-phase LS-STAG shear: both directional derivatives live in the SAME
# stress row. The tangential velocity trace at an open vertex is obtained
# with that same derivative. This lets the cell's Green integral supply the
# remaining wall/transition quadrature, rather than adding a second penalty.
struct _LSStressFace{T}
    column::Int                         # zero for a prescribed domain face
    axis::Int
    width::Int                          # finer adjacent cell on the lattice
    vertices::NTuple{2, Int}             # lower, upper tangential endpoint
    reaches::NTuple{2, Bool}             # open trace reaches this endpoint
    length::T
    midpoint::NTuple{2, T}
end

struct _LSStressLine{T}
    columns::NTuple{3, Int}
    derivative::NTuple{3, T}
    trace::NTuple{3, T}
    box_derivative::T                   # one-sided wall coefficient; zero off the box
    active::Bool
    regular::Bool
end

mutable struct _LSStressCache{T}
    vertex_index::Dict{NTuple{2, Int}, Int}
    keys::Vector{NTuple{2, Int}}
    incident::Vector{NTuple{4, Int}}     # x below/above, y left/right
    faces::Vector{_LSStressFace{T}}
    lines::Vector{NTuple{2, _LSStressLine{T}}}
    vertex_rows::Vector{Int}
    offsets::Vector{Int}
    cell_faces::Vector{Int}             # signed outward orientation
    vertices::Vector{Int}               # one cell's distinct vertices
    lengths::Vector{NTuple{2, T}}        # incident open lengths by axis
    slip_columns::Dict{Int, Tuple{NTuple{3, Int}, NTuple{3, T}}}
end

_LSStressCache(T) = _LSStressCache(
    Dict{NTuple{2, Int}, Int}(), NTuple{2, Int}[], NTuple{4, Int}[],
    _LSStressFace{T}[], NTuple{2, _LSStressLine{T}}[], Int[], Int[], Int[],
    Int[], NTuple{2, T}[], Dict{Int, Tuple{NTuple{3, Int}, NTuple{3, T}}}(),
)

@inline _ls_stress_key(mesh, lattice, key) = ntuple(
    d -> mesh.periodic[d] ? mod(key[d], lattice[d]) : key[d], 2,
)

function _ls_stress_face!(cache, geometry::GeometryField{T}, lattice, face, column, aperture) where {T}
    mesh = geometry.mesh
    axis = face.axis
    tangent = 3 - axis
    normal, lo, hi = _face_lattice_geometry(mesh, face, lattice)
    index = length(cache.faces) + 1
    vertices = ntuple(2) do end_index
        t = end_index == 1 ? lo : hi
        key = _ls_stress_key(mesh, lattice, axis == 1 ? (normal, t) : (t, normal))
        vertex = get!(cache.vertex_index, key) do
            push!(cache.keys, key)
            push!(cache.incident, (0, 0, 0, 0))
            return length(cache.keys)
        end
        slot = 2axis - end_index + 1
        cache.incident[vertex] = Base.setindex(cache.incident[vertex], index, slot)
        return vertex
    end
    width = typemax(Int)
    for (ref, sign) in _ls_stag_sides(face)
        _, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
        width = min(width, scale[axis])
        cell = _cell_index(ref, mesh.block_size...)
        cache.cell_faces[cache.offsets[cell]] = sign * index
        cache.offsets[cell] += 1
    end
    _, _, lower, upper, lo_open, hi_open = _ls_stag_open_part(geometry, face, aperture)
    midpoint = Base.setindex(T.(face.center), (lower + upper) / 2, tangent)
    active = aperture > zero(T)
    push!(
        cache.faces, _LSStressFace(
            column, axis, width, vertices, (active && lo_open, active && hi_open),
            T(aperture * face.area), midpoint,
        ),
    )
    return nothing
end

@inline function _ls_stress_inactive(::Type{T}) where {T}
    z = (zero(T), zero(T), zero(T))
    return _LSStressLine((0, 0, 0), z, z, zero(T), false, false)
end

# Extrapolate the fluid velocity to an endpoint from its two nearest open
# midpoints. A terminating line with no second sample only supplies a
# constant trace. On a box wall this is the interior Green reconstruction;
# the actual wall shear is a separate vertex contribution.
function _ls_stress_fluid_line(cache, face::_LSStressFace{T}, at_upper, box_derivative = zero(T)) where {T}
    axis = face.axis
    away = at_upper ? 1 : 2
    on_box = !iszero(box_derivative)
    other = cache.incident[face.vertices[away]][2axis - 2 + away]
    if !iszero(other) && face.reaches[away]
        far = cache.faces[other]
        if far.reaches[3 - away]
            inverse_length = inv(face.length + far.length)
            q = face.length * inverse_length
            inverse = (at_upper ? T(2) : -T(2)) * inverse_length
            return _LSStressLine(
                (face.column, far.column, 0), (inverse, -inverse, zero(T)),
                (one(T) + q, -q, zero(T)), box_derivative, on_box, on_box,
            )
        end
    end
    return _LSStressLine(
        (face.column, 0, 0), (zero(T), zero(T), zero(T)),
        (one(T), zero(T), zero(T)), box_derivative, on_box, on_box,
    )
end

function _ls_stress_line(cache::_LSStressCache{T}, mesh, lattice, vertex, axis) where {T}
    tangent = 3 - axis
    key = cache.keys[vertex]
    incident = cache.incident[vertex]
    low, high = incident[2axis - 1], incident[2axis]
    if !iszero(low) && !iszero(high)
        a, b = cache.faces[low], cache.faces[high]
        if !(a.reaches[2] && b.reaches[1])
            if a.reaches[2] || b.reaches[1]
                return _ls_stress_fluid_line(cache, a.reaches[2] ? a : b, a.reaches[2])
            end
            return _ls_stress_inactive(T)
        end
        inverse = 2 / (a.length + b.length)
        return _LSStressLine(
            (a.column, b.column, 0), (-inverse, inverse, zero(T)),
            (b.length * inverse / 2, a.length * inverse / 2, zero(T)), zero(T), true,
            a.length == b.length && a.width == b.width,
        )
    end
    iszero(low) && iszero(high) && return _ls_stress_inactive(T)
    below = !iszero(low)
    anchor = cache.faces[below ? low : high]
    anchor.reaches[below ? 2 : 1] || return _ls_stress_inactive(T)
    direction = below ? one(T) : -one(T)
    distance = direction * anchor.length / 2
    if !mesh.periodic[tangent] && (key[tangent] == 0 || key[tangent] == lattice[tangent])
        return _ls_stress_fluid_line(cache, anchor, below, -inv(distance))
    end
    # Hanging vertex: use both coarse fluid samples when available. Never
    # replace a missing sample by a boundary-condition-dependent wall value.
    columns = (anchor.column, 0, 0)
    offsets = ((zero(T), zero(T)), (zero(T), zero(T)))
    for j in 1:2
        k = key[axis] + (j == 1 ? -anchor.width : anchor.width)
        other_key = _ls_stress_key(mesh, lattice, Base.setindex(key, k, axis))
        other_vertex = get(cache.vertex_index, other_key, 0)
        iszero(other_vertex) && return _ls_stress_inactive(T)
        other_index = cache.incident[other_vertex][below ? 2axis : 2axis - 1]
        iszero(other_index) && return _ls_stress_inactive(T)
        other = cache.faces[other_index]
        iszero(other.length) && return _ls_stress_fluid_line(cache, anchor, below)
        delta = ntuple(d -> T(_wrapped_delta(other.midpoint[d] - anchor.midpoint[d], mesh.extent[d], mesh.periodic[d])), 2)
        offsets = Base.setindex(offsets, (delta[axis], delta[tangent]), j)
        columns = Base.setindex(columns, other.column, j + 1)
    end
    derivative = _ls_stag_affine_derivative(offsets...)
    isnothing(derivative) && return _ls_stress_inactive(T)
    trace = ntuple(j -> muladd(distance, derivative[j], T(j == 1)), 3)
    return _LSStressLine(columns, derivative, trace, zero(T), true, false)
end

function _ls_stress_topology!(cache, geometry)
    mesh = geometry.mesh
    lattice = _finest_lattice_size(mesh)
    empty!(cache.vertex_index)
    empty!(cache.keys)
    empty!(cache.incident)
    empty!(cache.faces)
    offsets = resize!(cache.offsets, ncells(mesh) + 1)
    fill!(offsets, 0)
    for faces in (mesh.faces, mesh.boundary_faces), face in faces
        for (ref, _) in _ls_stag_sides(face)
            offsets[_cell_index(ref, mesh.block_size...) + 1] += 1
        end
    end
    offsets[1] = 1
    cumsum!(offsets, offsets)
    resize!(cache.cell_faces, offsets[end] - 1)
    for (i, face) in pairs(mesh.faces)
        _ls_stress_face!(cache, geometry, lattice, face, i, geometry.aperture[i])
    end
    for (i, face) in pairs(mesh.boundary_faces)
        _ls_stress_face!(cache, geometry, lattice, face, 0, geometry.boundary_aperture[i])
    end
    for i in (length(offsets) - 1):-1:2
        offsets[i] = offsets[i - 1]
    end
    offsets[1] = 1
    resize!(cache.lines, length(cache.keys))
    for i in eachindex(cache.lines)
        cache.lines[i] = ntuple(axis -> _ls_stress_line(cache, mesh, lattice, i, axis), 2)
    end
    fill!(resize!(cache.vertex_rows, length(cache.lines)), 0)
    return cache
end

@inline function _ls_stress_line_row!(rows, cols, vals, row, columns, weights, scale)
    for j in eachindex(columns)
        iszero(columns[j]) && continue
        value = scale * weights[j]
        iszero(value) && continue
        push!(rows, row)
        push!(cols, columns[j])
        push!(vals, value)
    end
    return nothing
end

# For a sloping free-slip wall, dropping Cartesian xy alone would remove
# the wrong traction. Give the PLIC segment a free scalar tangential trace
# s (its normal velocity is zero). With the same cell viscosity on all three
# rows the minimum of sum_i m_i (E_i u - b_i s)^2 is obtained by projecting
# E orthogonally to b in the m metric. Box-wall shear is released separately
# in its vertex row; it does not change the cell Green reconstruction.
function _ls_stress_slip!(cache, cell, wall_row, measures, b)
    rows = (2cell - 1, 2cell, wall_row)
    m = map(row -> measures[row], rows)
    scale = maximum(abs, b)
    iszero(scale) && return nothing
    u = b ./ scale
    denom = sum(i -> m[i] * u[i]^2, 1:3)
    for j in 1:3
        cache.slip_columns[rows[j]] = (rows, ntuple(i -> u[i] * (m[j] * u[j] / denom), 3))
    end
    return nothing
end

# At each cell corner the two half-open edges span a vertex rectangle.
# On a regular mesh these rectangles recover the MAC vertex volumes. On a
# cut or nonconforming cell reserve open area for the cell Green completion;
# even independently reconstructed PLIC traces cannot exhaust its measure.
function _ls_stress_cell!(cache, geometry::GeometryField{T}, ref, cell, measures, construction, boundary) where {T}
    open = _get(geometry.open_fraction, ref)
    open > zero(T) || return
    mesh = geometry.mesh
    volume = open * mesh.blocks[ref.block].volume
    vertices = empty!(cache.vertices)
    lengths = empty!(cache.lengths)
    entries = cache.offsets[cell]:(cache.offsets[cell + 1] - 1)
    regular = isone(open) && length(entries) == 4
    wall = !isone(open)
    for entry in entries
        face = cache.faces[abs(cache.cell_faces[entry])]
        wall |= iszero(face.length)
        for j in 1:2
            v = face.vertices[j]
            x, y = cache.lines[v]
            regular &= face.reaches[j] && x.regular && y.regular
            face.reaches[j] && x.active && y.active || continue
            pos = findfirst(==(v), vertices)
            if isnothing(pos)
                push!(vertices, v)
                push!(lengths, (zero(T), zero(T)))
                pos = length(vertices)
            end
            lengths[pos] = Base.setindex(lengths[pos], face.length, face.axis)
        end
    end
    sum_rectangles = sum(x -> x[1] * x[2] / 4, lengths; init = zero(T))
    # Full uniform cells need no completion: their trapezoidal Green
    # integral equals the sum of the four volume-weighted vertex strains.
    scale = regular || iszero(sum_rectangles) ? one(T) : min(one(T), volume / (2sum_rectangles))
    remainder = regular ? zero(T) : volume - scale * sum_rectangles
    tilted_slip = false
    tangent = (zero(T), zero(T))
    if boundary isa FreeSlip && !isone(open)
        h = mesh.blocks[ref.block].spacing
        n1 = _get(geometry.solid.normal_x, ref) / h[1]
        n2 = _get(geometry.solid.normal_y, ref) / h[2]
        norm_n = hypot(n1, n2)
        tilted_slip = min(abs(n1), abs(n2)) > 64eps(T) * norm_n
        tilted_slip && (tangent = (-n2 / norm_n, n1 / norm_n))
    end
    complete = remainder > zero(T) && (!(wall && boundary isa FreeSlip) || tilted_slip)
    rows, cols, vals = construction.strain_rows, construction.strain_columns, construction.strain_values
    ir, ic, iv = construction.interpolation_rows, construction.interpolation_columns, construction.interpolation_values
    cell_row = 0
    slip = nothing
    if complete
        push!(measures, remainder)
        cell_row = length(measures)
        push!(ir, cell_row); push!(ic, cell); push!(iv, remainder)
        # Integral of d_y u + d_x v over c. The tangential traces at both
        # ends of every open Cartesian edge are shared by its two cells;
        # a PLIC endpoint and the stationary wall itself have zero trace.
        normal_flux = (zero(T), zero(T))
        shear_flux = zero(T)
        for entry in entries
            oriented = cache.cell_faces[entry]
            face = cache.faces[abs(oriented)]
            coefficient = sign(oriented) * face.length / (2remainder)
            if tilted_slip
                k = face.axis
                normal_flux = Base.setindex(normal_flux, normal_flux[k] + sign(oriented) * face.length, k)
            end
            for j in 1:2
                face.reaches[j] || continue
                line = cache.lines[face.vertices[j]][3 - face.axis]
                if tilted_slip
                    shear_flux += sign(oriented) * face.length * tangent[3 - face.axis] *
                        sum(line.trace) / 2
                end
                _ls_stress_line_row!(rows, cols, vals, cell_row, line.columns, line.trace, coefficient)
            end
        end
        if tilted_slip
            # The unit wall trace is precisely E applied to constant t: the
            # shared vertex derivatives annihilate constants, and the cell
            # shear retains the same endpoint traces as its Green integral.
            slip = (
                cell_row,
                (
                    tangent[1] * normal_flux[1] / volume,
                    tangent[2] * normal_flux[2] / volume,
                    shear_flux / remainder,
                ),
            )
        end
    end
    for pos in eachindex(vertices)
        a, b = lengths[pos]
        area = scale * (a * b / 4)
        area > zero(T) || continue
        v = vertices[pos]
        lines = cache.lines[v]
        row = cache.vertex_rows[v]
        if iszero(row)
            push!(measures, zero(T))
            row = length(measures)
            cache.vertex_rows[v] = row
            for line in lines
                if iszero(line.box_derivative)
                    _ls_stress_line_row!(rows, cols, vals, row, line.columns, line.derivative, one(T))
                elseif boundary isa NoSlip
                    # The normal velocity is prescribed along a stationary box
                    # wall. Only the wall-normal derivative of its tangential
                    # velocity remains; freeing that trace releases this square.
                    _ls_stress_line_row!(rows, cols, vals, row, (line.columns[1],), (line.box_derivative,), one(T))
                end
            end
        end
        measures[row] += area
        push!(ir, row); push!(ic, cell); push!(iv, area)
        if complete
            for line in lines
                _ls_stress_line_row!(rows, cols, vals, cell_row, line.columns, line.derivative, -area / remainder)
            end
        end
    end
    return slip
end

function _ls_stress_rows!(measures, construction, geometry::GeometryField{T}, boundary) where {T}
    scratch = construction.ls_stag
    if isnothing(scratch)
        scratch = _LSStressCache(T)
        construction.ls_stag = scratch
    end
    cache = _ls_stress_topology!(scratch, geometry)
    empty!(cache.slip_columns)
    ir, iv = construction.interpolation_rows, construction.interpolation_values
    first_entry = length(iv) + 1
    for (cell, ref) in pairs(construction.cell_references)
        slip = _ls_stress_cell!(cache, geometry, ref, cell, measures, construction, boundary)
        if !isnothing(slip)
            wall_row, b = slip
            _ls_stress_slip!(cache, cell, wall_row, measures, b)
        end
    end
    if !isempty(cache.slip_columns)
        rows, cols, vals = construction.strain_rows, construction.strain_columns, construction.strain_values
        for p in eachindex(rows)
            column = get(cache.slip_columns, rows[p], nothing)
            isnothing(column) && continue
            targets, factors = column
            value = vals[p]
            for j in 1:3
                if targets[j] == rows[p]
                    vals[p] = value * (1 - factors[j])
                elseif !iszero(factors[j])
                    push!(rows, targets[j])
                    push!(cols, cols[p])
                    push!(vals, -value * factors[j])
                end
            end
        end
    end
    for i in first_entry:length(iv)
        iv[i] /= measures[ir[i]]
    end
    return nothing
end
