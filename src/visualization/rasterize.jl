"""
    rasterize(frame, field, (xmin, xmax, ymin, ymax), (w, h);
              phase_rendering=:cell)::Matrix{Float32}

Sample field `field` of a [`Frame`](@ref) at the centres of a `w × h` pixel
grid covering the given world rectangle. Entry `[i, j]` is the value at
`x = xmin + (i - 1/2)(xmax - xmin)/w`, `y = ymin + (j - 1/2)(ymax - ymin)/h`
and `NaN` outside the domain or where no value exists.

Cell fields are piecewise constant per cell. Face fields fill their staggered
dual cells: a face value covers the half cells on both sides of its face, and
each of the two fine faces of a coarse–fine interface covers its half of the
coarse cell. For phase fields, `phase_rendering=:cell` shows the value of the
cell's majority phase (at least 1/2 liquid of the open volume) and
`phase_rendering=:plic` splits mixed cells sharply at their PLIC plane, using
the plane of the cell that contains the pixel. Cells without open fluid, and
with `:plic` the solid part of cut cells, have no value. The cost is
proportional to the number of pixels plus blocks.
"""
function rasterize(
        frame::Frame, field::Symbol, rect, size; phase_rendering::Symbol = :cell,
    )
    _check_phase_rendering(phase_rendering)
    entry = _frame_entry(frame, field)
    w, h = Int(size[1]), Int(size[2])
    w >= 1 && h >= 1 || throw(ArgumentError("the raster size must be positive"))
    xmin, xmax, ymin, ymax = Float64.(rect)
    xmax > xmin && ymax > ymin ||
        throw(ArgumentError("the raster rectangle must have positive extent"))
    image = fill(NaN32, w, h)
    topology = frame.topology
    values, phase, solid = _render_source_values(frame, entry)
    _rasterize!(
        image, topology, entry.placement, values, phase, solid,
        phase_rendering === :plic, (xmin, xmax, ymin, ymax),
    )
    return image
end

# Keep both intrinsic phase arrays until the display policy classifies each
# pixel; the snapshot also owns the associated interface and solid geometry.
@inline function _render_source_values(frame::Frame, entry::FieldEntry)
    values = frame.values[entry.slot]
    phase = iszero(entry.interface) ? nothing : frame.phases[entry.interface]
    solid = isnothing(phase) || iszero(phase.solid) ? nothing :
        frame.topology.solids[phase.solid]
    return values, phase, solid
end

@inline function _sorted_index(sorted, first, last, value)
    index = searchsortedfirst(sorted, value, first, last, Base.Order.Forward)
    return index <= last && sorted[index] == value ? index : 0
end

@inline _plane_side(plane, x, y) = plane[1] * x + plane[2] * y - plane[3]

@inline _texel_value(values::AbstractVector, index) = values[index]

@inline function _phase_value(values::TwoPhaseVisualizationValues, index, liquid)
    return liquid ? values.liquid[index] : values.gas[index]
end

function _rasterize!(
        image, topology::TopologySnapshot, placement, values, phase, solid,
        sharp::Bool, rect,
    )
    xmin, xmax, ymin, ymax = rect
    w, h = Base.size(image)
    pixel_x = (xmax - xmin) / w
    pixel_y = (ymax - ymin) / h
    nx, ny = topology.block_size
    columns, rows = placement isa Face{1} ? _tile_shape(nx, ny, 1) :
        placement isa Face{2} ? _tile_shape(nx, ny, 2) : (0, 0)
    tiles = placement isa Face{1} ? topology.face_tiles[1] :
        placement isa Face{2} ? topology.face_tiles[2] : Int32[]
    for block in 1:nblocks(topology)
        ox, oy = Float64.(topology.origins[block])
        dx, dy = Float64.(topology.spacings[block])
        i0 = max(1, ceil(Int, (ox - xmin) / pixel_x + 0.5))
        i1 = min(w, ceil(Int, (ox + nx * dx - xmin) / pixel_x + 0.5) - 1)
        j0 = max(1, ceil(Int, (oy - ymin) / pixel_y + 0.5))
        j1 = min(h, ceil(Int, (oy + ny * dy - ymin) / pixel_y + 0.5) - 1)
        (i0 <= i1 && j0 <= j1) || continue
        first_cell = (block - 1) * nx * ny + 1
        last_cell = block * nx * ny
        mixed_range = isnothing(phase) ? (1:0) :
            searchsortedfirst(phase.mixed, first_cell):searchsortedlast(phase.mixed, last_cell)
        cut_range = isnothing(solid) ? (1:0) :
            searchsortedfirst(solid.cut, first_cell):searchsortedlast(solid.cut, last_cell)
        tile_base = (block - 1) * columns * rows
        for j in j0:j1
            η = (ymin + (j - 0.5) * pixel_y - oy) / dy
            cj = min(floor(Int, η), ny - 1)
            for i in i0:i1
                ξ = (xmin + (i - 0.5) * pixel_x - ox) / dx
                ci = min(floor(Int, ξ), nx - 1)
                cell = first_cell + cj * nx + ci
                index = if placement isa Cell
                    cell
                elseif placement isa Face{1}
                    c = clamp(floor(Int, ξ + 0.5), 0, nx)
                    s = clamp(floor(Int, 2η), 0, 2ny - 1)
                    Int(tiles[tile_base + s * columns + c + 1])
                else
                    s = clamp(floor(Int, 2ξ), 0, 2nx - 1)
                    r = clamp(floor(Int, η + 0.5), 0, ny)
                    Int(tiles[tile_base + r * columns + s + 1])
                end
                iszero(index) && continue
                if isnothing(phase)
                    image[i, j] = Float32(_texel_value(values, index))
                    continue
                end
                local_x = ξ - ci - 0.5
                local_y = η - cj - 0.5
                if !isnothing(solid)
                    solid.full[cell] && continue
                    if sharp && !isempty(cut_range)
                        k = _sorted_index(solid.cut, first(cut_range), last(cut_range), cell)
                        !iszero(k) && _plane_side(solid.planes[k], local_x, local_y) <= 0 &&
                            continue
                    end
                end
                liquid = phase.liquid[cell]
                if sharp && !isempty(mixed_range)
                    k = _sorted_index(phase.mixed, first(mixed_range), last(mixed_range), cell)
                    iszero(k) ||
                        (liquid = _plane_side(phase.planes[k], local_x, local_y) <= 0)
                end
                image[i, j] = Float32(_phase_value(values, index, liquid))
            end
        end
    end
    return image
end
