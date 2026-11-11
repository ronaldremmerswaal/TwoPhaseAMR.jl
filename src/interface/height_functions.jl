struct HeightDonor{T}
    cell::CellRef
    bounds::NTuple{4, T}
    weight::T
end
struct HeightGhost{T}
    block::Int
    i::Int
    j::Int
    donors::Vector{HeightDonor{T}}
end
mutable struct HeightFunctionCache{T}
    keys::Vector{BlockKey}
    values::Vector{Matrix{T}}
    ghosts::Vector{HeightGhost{T}}
end
HeightFunctionCache(::Type{T}) where {T} = HeightFunctionCache(BlockKey[], Matrix{T}[], HeightGhost{T}[])

@inline function _line_liquid_fraction(
        field, ref, direction, transverse_coordinate,
    )
    T = eltype(field)
    fraction = _conditional_fraction(field, ref)
    fraction <= zero(T) && return zero(T)
    fraction >= one(T) && return one(T)

    normal = direction == 1 ?
        (_get(field.normal_x, ref), _get(field.normal_y, ref)) :
        (_get(field.normal_y, ref), _get(field.normal_x, ref))
    intercept = _get(field.intercept, ref)
    rhs = intercept - normal[2] * transverse_coordinate
    tolerance = T(128) * eps(T) * max(
        one(T), abs(intercept), abs(normal[1]), abs(normal[2]),
    )
    if abs(normal[1]) <= tolerance
        return rhs >= -tolerance ? one(T) : zero(T)
    end
    crossing = rhs / normal[1]
    return normal[1] > 0 ?
        clamp(crossing + T(0.5), zero(T), one(T)) :
        clamp(T(0.5) - crossing, zero(T), one(T))
end

@inline function _line_is_liquid(
        field, ref, direction, transverse_coordinate, normal_coordinate,
    )
    T = eltype(field)
    fraction = _conditional_fraction(field, ref)
    fraction <= zero(T) && return false
    fraction >= one(T) && return true
    normal = direction == 1 ?
        (_get(field.normal_x, ref), _get(field.normal_y, ref)) :
        (_get(field.normal_y, ref), _get(field.normal_x, ref))
    intercept = _get(field.intercept, ref)
    value = normal[1] * normal_coordinate +
        normal[2] * transverse_coordinate - intercept
    tolerance = T(128) * eps(T) * max(
        one(T), abs(intercept), abs(normal[1]), abs(normal[2]),
    )
    return value <= tolerance
end

function _interface_height(field, direction, position)
    mesh = field.mesh
    T = eltype(field)
    transverse = 3 - direction
    domain_low = T(mesh.origin[transverse])
    domain_width = T(mesh.extent[transverse])
    domain_high = domain_low + domain_width
    coordinate = T(position)
    isfinite(coordinate) || throw(ArgumentError("position must be finite"))
    if mesh.periodic[transverse]
        coordinate = domain_low + mod(coordinate - domain_low, domain_width)
    else
        domain_low <= coordinate <= domain_high || throw(
            ArgumentError(
                "position=$position lies outside the mesh in the transverse direction",
            ),
        )
        coordinate == domain_high && (coordinate = prevfloat(domain_high))
    end

    normal_low = T(mesh.origin[direction])
    normal_high = normal_low + T(mesh.extent[direction])
    liquid_height = zero(T)
    covered_height = zero(T)
    low_phase = nothing
    high_phase = nothing
    low_cell_coordinate = T(Inf)
    high_cell_coordinate = T(-Inf)
    cells_per_block = mesh.block_size[direction]

    for (block_index, block) in pairs(mesh.blocks)
        transverse_low = T(block.origin[transverse])
        transverse_high = transverse_low + T(block.extent[transverse])
        transverse_low <= coordinate < transverse_high || continue
        transverse_spacing = T(block.spacing[transverse])
        transverse_index = clamp(
            floor(Int, (coordinate - transverse_low) / transverse_spacing) + 1,
            1, mesh.block_size[transverse],
        )
        transverse_center = transverse_low +
            (transverse_index - T(0.5)) * transverse_spacing
        local_transverse =
            (coordinate - transverse_center) / transverse_spacing
        normal_spacing = T(block.spacing[direction])

        for normal_index in 1:cells_per_block
            i, j = direction == 1 ?
                (normal_index, transverse_index) :
                (transverse_index, normal_index)
            ref = CellRef(block_index, i, j)
            if !isnothing(field.geometry)
                open_fraction = _get(field.geometry.open_fraction, ref)
                open_fraction >= one(T) - T(128) * eps(T) || throw(
                    ArgumentError(
                        "interface_height is undefined where the sampled column crosses solid geometry",
                    ),
                )
            end
            liquid_height += normal_spacing * _line_liquid_fraction(
                field, ref, direction, local_transverse,
            )
            covered_height += normal_spacing

            cell_low = T(block.origin[direction]) +
                (normal_index - 1) * normal_spacing
            cell_high = cell_low + normal_spacing
            inward = min(T(64) * eps(T), T(0.25))
            if cell_low < low_cell_coordinate
                low_cell_coordinate = cell_low
                low_phase = _line_is_liquid(
                    field, ref, direction, local_transverse,
                    -T(0.5) + inward,
                )
            end
            if cell_high > high_cell_coordinate
                high_cell_coordinate = cell_high
                high_phase = _line_is_liquid(
                    field, ref, direction, local_transverse,
                    T(0.5) - inward,
                )
            end
        end
    end

    tolerance = T(256) * eps(T) * max(one(T), abs(normal_high - normal_low))
    abs(covered_height - (normal_high - normal_low)) <= tolerance || error(
        "the interface-height column does not cover the mesh extent",
    )
    low_phase == high_phase && throw(
        ArgumentError(
            "interface_height requires one interface separating opposite phases at the column ends",
        ),
    )
    return low_phase ? normal_low + liquid_height : normal_high - liquid_height
end

function _height_cache!(field)
    mesh, cache = field.mesh, field.height_cache
    T = eltype(field)
    length(cache.keys) == nblocks(mesh) &&
        all(cache.keys[b] == mesh.blocks[b].key for b in eachindex(cache.keys)) && return cache
    empty!(cache.keys); empty!(cache.values); empty!(cache.ghosts)
    nx, ny = mesh.block_size
    indices_by_key = Dict(
        block.key => index for (index, block) in pairs(mesh.blocks)
    )
    candidate_blocks = Int[]
    candidate_nodes = _tree_node_type(mesh.root)[]
    lattice = _finest_lattice_size(mesh)
    for (b, block) in pairs(mesh.blocks)
        push!(cache.keys, block.key)
        push!(cache.values, fill(eltype(field)(NaN), nx + 8, ny + 8))
        scale = _block_scale(mesh, block.key.level)
        for j in -3:(ny + 4), i in -3:(nx + 4)
            1 <= i <= nx && 1 <= j <= ny && continue
            xlo = (block.key.i * nx + i - 1) * scale
            ylo = (block.key.j * ny + j - 1) * scale
            if (!mesh.periodic[1] && !(0 <= xlo && xlo + scale <= lattice[1])) ||
                    (!mesh.periodic[2] && !(0 <= ylo && ylo + scale <= lattice[2]))
                continue # physical boundaries do not provide valid height columns
            end
            mesh.periodic[1] && (xlo = mod(xlo, lattice[1]))
            mesh.periodic[2] && (ylo = mod(ylo, lattice[2]))
            xhi, yhi = xlo + scale, ylo + scale
            donors = HeightDonor{T}[]
            # A dyadic target cannot straddle a periodic seam after wrapping.
            empty!(candidate_nodes)
            _collect_leaf_nodes_in_window!(
                candidate_nodes, mesh.root, (xlo, ylo), (xhi, yhi),
                mesh.max_level, mesh.block_size,
            )
            empty!(candidate_blocks)
            for node in candidate_nodes
                push!(candidate_blocks, indices_by_key[node.data.key])
            end
            sort!(candidate_blocks)
            for db in candidate_blocks
                other = mesh.blocks[db]
                other_scale = _block_scale(mesh, other.key.level)
                other_x = other.key.i * nx * other_scale
                other_y = other.key.j * ny * other_scale
                xl = max(xlo, other_x)
                xh = min(xhi, other_x + nx * other_scale)
                yl = max(ylo, other_y)
                yh = min(yhi, other_y + ny * other_scale)
                xh > xl && yh > yl || continue
                ilo = max(1, fld(xl - other_x, other_scale) + 1)
                ihi = min(nx, cld(xh - other_x, other_scale))
                jlo = max(1, fld(yl - other_y, other_scale) + 1)
                jhi = min(ny, cld(yh - other_y, other_scale))
                for dj in jlo:jhi, di in ilo:ihi
                    cell_x = other_x + (di - 1) * other_scale
                    cell_y = other_y + (dj - 1) * other_scale
                    left = max(xl, cell_x)
                    right = min(xh, cell_x + other_scale)
                    low = max(yl, cell_y)
                    high = min(yh, cell_y + other_scale)
                    right > left && high > low || continue
                    bounds = (
                        T((left - cell_x) / other_scale - 0.5),
                        T((right - cell_x) / other_scale - 0.5),
                        T((low - cell_y) / other_scale - 0.5),
                        T((high - cell_y) / other_scale - 0.5),
                    )
                    weight = T((right - left) * (high - low) / (scale * scale))
                    push!(
                        donors, HeightDonor{T}(
                            CellRef(db, di, dj), bounds, weight
                        )
                    )
                end
            end
            push!(cache.ghosts, HeightGhost(b, i + 4, j + 4, donors))
        end
    end
    return cache
end
function _fill_height_values!(field)
    mesh, cache = field.mesh, _height_cache!(field)
    T = eltype(field)
    nx, ny = mesh.block_size
    _parallel_for(nblocks(mesh); work = nx * ny) do b
        if isnothing(field.geometry)
            cache.values[b][5:(nx + 4), 5:(ny + 4)] .= interior(field, b)
        else
            destination = @view cache.values[b][5:(nx + 4), 5:(ny + 4)]
            fraction = interior(field.conditional_fraction, b)
            open = interior(field.geometry.open_fraction, b)
            tolerance = 64eps(T)
            @inbounds for j in axes(destination, 2), i in axes(destination, 1)
                # A height column may terminate at a solid boundary, but it
                # must never integrate the solid as if it were the gas phase.
                # Cut and covered cells are therefore unavailable to the LHF.
                # Triple cells use a separate solid-aware generalized fit.
                destination[i, j] = open[i, j] >= one(T) - tolerance ?
                    fraction[i, j] : T(NaN)
            end
        end
    end
    @_parallel for ghost_index in 1:length(cache.ghosts)
        ghost = cache.ghosts[ghost_index]
        value = zero(eltype(field))
        valid = true
        for donor in ghost.donors
            ref = donor.cell
            if !isnothing(field.geometry) &&
                    _get(field.geometry.open_fraction, ref) <
                    one(eltype(field)) - 64eps(eltype(field))
                valid = false
                break
            end
            c = _conditional_fraction(field, ref)
            portion = c == 0 || c == 1 ? c : _plane_fraction(
                    _get(field.normal_x, ref), _get(field.normal_y, ref),
                    _get(field.intercept, ref), donor.bounds...
                )
            value += donor.weight * portion
        end
        cache.values[ghost.block][ghost.i, ghost.j] = valid ?
            clamp(value, zero(value), one(value)) : eltype(field)(NaN)
    end
    return cache.values
end

# Height is in cell units relative to the column's central cell. Strictly
# monotone mixed values and pure endpoints enforce thesis Sec. 2.4's validity.
function _local_height(values, i, j, axis, orientation; radius = 3)
    total = zero(eltype(values));previous = one(eltype(values))
    tol = 64eps(eltype(values));start = nothing
    for k in -radius:radius
        ii, jj = axis == 1 ? (i + k, j) : (i, j + k)
        available = checkbounds(Bool, values, ii, jj) && isfinite(values[ii, jj])
        if !available
            isnothing(start) && continue
            break
        end
        c = values[ii, jj];v = orientation > 0 ? c : 1 - c
        if isnothing(start)
            abs(v - 1) <= tol || return nothing
            start = k
        else
            v > previous + tol && return nothing
            tol < v < 1 - tol && abs(v - previous) <= tol && return nothing
        end
        total += v;previous = v
    end
    isnothing(start) && return nothing
    abs(previous) <= tol || return nothing
    return total + start - 0.5
end

@inline function _height_quality(principal)
    angle = acos(clamp(principal, zero(principal), one(principal)))
    pi_t = oftype(principal, pi)
    t = clamp(
        (pi_t / 4 + pi_t / 20 - angle) / (pi_t / 10),
        zero(principal), one(principal)
    )
    return t * t * (3 - 2t) # approved correction to the thesis's printed polynomial
end
function _lhf_geometry(values, i, j, spacing)
    T = promote_type(eltype(values), typeof(spacing[1]), typeof(spacing[2]))
    candidates = ntuple(_ -> ntuple(_ -> zero(T), 4), 4)
    count = 0
    for axis in 1:2,orientation in (-1, 1)
        heights = ntuple(3) do k
            offset = k - 2
            ii, jj = axis == 1 ? (i, j + offset) : (i + offset, j)
            _local_height(values, ii, jj, axis, orientation)
        end
        any(isnothing, heights) && continue
        hm, h0, hp = heights
        normal_h, tangent_h = spacing[axis], spacing[3 - axis]
        slope = (hp - hm) * normal_h / (2tangent_h)
        second = (hp - 2h0 + hm) * normal_h / tangent_h^2
        norm = sqrt(1 + slope^2)
        principal, tangent = orientation / norm, -orientation * slope / norm
        nx, ny = axis == 1 ? (principal, tangent) : (tangent, principal)
        quality = _height_quality(abs(principal))
        quality > 0 || continue
        # Outward liquid normal; convex liquid droplets have positive curvature.
        curvature = -orientation * second / norm^3
        count += 1
        candidates = Base.setindex(candidates, T.((nx, ny, curvature, quality)), count)
    end
    count == 0 && return nothing
    nx = ny = kappa = weight = zero(T)
    for index in 1:count
        c = candidates[index]
        product = one(T)
        for j in 1:count
            index == j && continue
            product *= one(T) - candidates[j][4]
        end
        w = c[4] * product
        nx += w * c[1]; ny += w * c[2]; kappa += w * c[3]; weight += w
    end
    weight > 0 || return nothing
    return nx, ny, kappa / weight
end

"""Return `(valid, filament)` for the thesis curvature-AMR tests.

`valid` means that at least one signed principal direction supplies the three
parallel columns needed by the centred LHF reconstruction. `filament` detects
two interfaces in the central seven-cell column: both endpoints contain the
same pure phase while an interior value contains the other phase. Such a
column contains the positive- and negative-facing height branches that a
single PLIC plane cannot represent.
"""
function _lhf_refinement_status(values, i, j; radius = 3)
    valid = false
    filament = false
    tol = 64eps(eltype(values))
    for axis in 1:2
        for orientation in (-1, 1)
            direction_valid = true
            for offset in -1:1
                ii, jj = axis == 1 ?
                    (i, j + offset) : (i + offset, j)
                if isnothing(
                        _local_height(
                            values, ii, jj, axis, orientation; radius
                        )
                    )
                    direction_valid = false
                    break
                end
            end
            valid |= direction_valid
        end
        column = ntuple(2radius + 1) do position
            offset = position - radius - 1
            ii, jj = axis == 1 ? (i + offset, j) : (i, j + offset)
            checkbounds(Bool, values, ii, jj) ? values[ii, jj] :
                eltype(values)(NaN)
        end
        all(isfinite, column) || continue
        low, high = first(column), last(column)
        if low <= tol && high <= tol
            filament |= any(value -> value > tol, column[2:(end - 1)])
        elseif low >= 1 - tol && high >= 1 - tol
            filament |= any(
                value -> value < 1 - tol, column[2:(end - 1)]
            )
        end
    end
    return valid, filament
end

# WY (2010), Sec. 2.1: central-difference orientation, three-cell columns,
# and a one-sided difference selected by c>1/2 and the transverse normal.
function _wy_normal(values, i, j)
    gx = values[i - 1, j] - values[i + 1, j]
    gy = values[i, j - 1] - values[i, j + 1]
    axis = abs(gx) > abs(gy) ? 1 : 2
    principal = axis == 1 ? gx : gy
    tangent = axis == 1 ? gy : gx
    orientation = principal < 0 ? -one(principal) : one(principal)
    direction = (tangent < 0 ? -1 : 1) *
        (values[i, j] > oftype(values[i, j], 0.5) ? 1 : -1)
    h0 = h1 = zero(eltype(values))
    for k in -1:1
        ii, jj = axis == 1 ? (i + k, j) : (i, j + k)
        di, dj = axis == 1 ? (0, direction) : (direction, 0)
        h0 += values[ii, jj]
        h1 += values[ii + di, jj + dj]
    end
    transverse = -(h1 - h0) / direction
    return axis == 1 ? (orientation, transverse) : (transverse, orientation)
end

# Masked WY estimate for a triple cell. Ratios of physical liquid and open
# volume also exclude covered donors in fine-to-coarse halos. One-sided
# differences use the centre if a neighbour is covered and are scaled to the
# same two-cell span as central differences. Column differences contain only
# pairs for which both fractions represent fluid.
function _solid_aware_wy_normal(values, open, i, j)
    fraction(ii, jj) = values[ii, jj] / open[ii, jj]
    center = fraction(i, j)
    function difference(di, dj)
        lower = open[i - di, j - dj] > 0
        upper = open[i + di, j + dj] > 0
        lower && upper && return fraction(i - di, j - dj) -
            fraction(i + di, j + dj)
        lower && return 2 * (fraction(i - di, j - dj) - center)
        upper && return 2 * (center - fraction(i + di, j + dj))
        return zero(center)
    end
    gx, gy = difference(1, 0), difference(0, 1)
    axis = abs(gx) > abs(gy) ? 1 : 2
    principal = axis == 1 ? gx : gy
    tangent = axis == 1 ? gy : gx
    orientation = principal < 0 ? -one(principal) : one(principal)
    direction = (tangent < 0 ? -1 : 1) *
        (center > oftype(center, 0.5) ? 1 : -1)
    for side in (direction, -direction)
        delta = zero(center)
        pairs = 0
        for k in -1:1
            ii, jj = axis == 1 ? (i + k, j) : (i, j + k)
            di, dj = axis == 1 ? (0, side) : (side, 0)
            if open[ii, jj] > 0 && open[ii + di, jj + dj] > 0
                delta += fraction(ii + di, jj + dj) - fraction(ii, jj)
                pairs += 1
            end
        end
        if pairs > 0
            transverse = -delta * (oftype(center, 3) / pairs) / side
            return axis == 1 ? (orientation, transverse) :
                (transverse, orientation)
        end
    end
    transverse = iszero(principal) ? zero(center) : tangent / abs(principal)
    return axis == 1 ? (orientation, transverse) : (transverse, orientation)
end

function _interface_centroid(nx, ny, a)
    T = promote_type(typeof(nx), typeof(ny), typeof(a))
    half = T(0.5)
    tolerance = T(64) * eps(T)
    points = ntuple(_ -> (zero(T), zero(T)), 4)
    count = 0
    for axis in 1:2, edge in (-half, half)
        if axis == 1
            abs(ny) > eps(T) || continue
            y = (a - nx * edge) / ny
            -half - tolerance <= y <= half + tolerance || continue
            point = (edge, clamp(y, -half, half))
        else
            abs(nx) > eps(T) || continue
            x = (a - ny * edge) / nx
            -half - tolerance <= x <= half + tolerance || continue
            point = (clamp(x, -half, half), edge)
        end
        # Match unique!'s isequal semantics, including signed zeros.
        duplicate = false
        for i in 1:count
            if isequal(points[i], point)
                duplicate = true
                break
            end
        end
        duplicate && continue
        count += 1
        points = Base.setindex(points, point, count)
    end
    count == 0 && return (zero(T), zero(T))
    return (
        sum(points[i][1] for i in 1:count) / count,
        sum(points[i][2] for i in 1:count) / count,
    )
end

function _ghf_fit(points, normal, origin, h; contact = nothing)
    length(points) + (!isnothing(contact) ? 1 : 0) >= 3 || return nothing
    nx, ny = normal
    T = promote_type(typeof(nx), typeof(ny), typeof(origin[1]), typeof(h))
    number_of_rows = length(points) + (!isnothing(contact) ? 1 : 0)
    matrix = Matrix{T}(undef, number_of_rows, 3)
    rhs = Vector{T}(undef, number_of_rows)
    for (i, (x, y, w)) in pairs(points)
        dx, dy = (x - origin[1]) / h, (y - origin[2]) / h
        tangent = -ny * dx + nx * dy
        weight = sqrt(w)
        matrix[i, :] .= (weight * tangent^2, weight * tangent, weight)
        rhs[i] = weight * (nx * dx + ny * dy)
    end
    if !isnothing(contact)
        tangent_coordinate, wall_normal, cosine = contact
        tangent = (-ny, nx)
        projection = tangent[1] * wall_normal[1] +
            tangent[2] * wall_normal[2]
        weight = T(1.0e3)
        row = number_of_rows
        matrix[row, :] .= weight .* (
            -2tangent_coordinate * projection, -projection, zero(T),
        )
        rhs[row] = weight * (
            cosine - nx * wall_normal[1] - ny * wall_normal[2]
        )
    end
    singular = svdvals(matrix)
    rank_tolerance = max(T(1.0e-10), T(64) * eps(T))
    minimum(singular) > rank_tolerance * maximum(singular) || return nothing
    a, b, _ = matrix \ rhs
    scale = sqrt(1 + b * b)
    return (nx + b * ny) / scale, (ny - b * nx) / scale, -2a / (h * scale^3)
end

function _embedded_ghf_constraint(
        field, ref, normal, origin, h,
    )
    geometry = field.geometry
    theta = field.embedded_contact_angle
    (isnothing(geometry) || isnothing(theta)) && return nothing
    solid_fraction = _get(geometry.solid.fraction, ref)
    zero(eltype(field)) < solid_fraction < one(eltype(field)) || return nothing
    hx, hy = field.mesh.blocks[ref.block].spacing
    interface_matrix = (
        _get(field.normal_x, ref) / hx,
        _get(field.normal_y, ref) / hy,
    )
    solid_matrix = (
        _get(geometry.solid.normal_x, ref) / hx,
        _get(geometry.solid.normal_y, ref) / hy,
    )
    determinant = interface_matrix[1] * solid_matrix[2] -
        interface_matrix[2] * solid_matrix[1]
    scale = hypot(interface_matrix...) * hypot(solid_matrix...)
    abs(determinant) > 256eps(eltype(field)) * scale || return nothing
    interface_rhs = _get(field.intercept, ref)
    solid_rhs = _get(geometry.solid.intercept, ref)
    x = (
        interface_rhs * solid_matrix[2] -
            interface_matrix[2] * solid_rhs
    ) / determinant
    y = (
        interface_matrix[1] * solid_rhs -
            interface_rhs * solid_matrix[1]
    ) / determinant
    abs(x) <= hx * (0.5 + 256eps(eltype(field))) &&
        abs(y) <= hy * (0.5 + 256eps(eltype(field))) || return nothing
    wall_norm = hypot(solid_matrix...)
    wall_normal = (
        solid_matrix[1] / wall_norm, solid_matrix[2] / wall_norm,
    )
    tangent_coordinate = (
        -normal[2] * (x - origin[1]) +
            normal[1] * (y - origin[2])
    ) / h
    return tangent_coordinate, wall_normal, cos(theta)
end
function _ghf_geometry(
        field, values, b, i, j,
        previous_normal_x = field.normal_x,
        previous_normal_y = field.normal_y,
        previous_intercept = field.intercept,
        ; enforce_contact = true, sample_radius = 1,
    )
    ref = CellRef(b, i, j)
    hx, hy = field.mesh.blocks[b].spacing
    mx, my = _get(field.normal_x, ref), _get(field.normal_y, ref)
    norm = hypot(mx / hx, my / hy)
    normal = (mx / hx / norm, my / hy / norm)
    centroid = _interface_centroid(mx, my, _get(field.intercept, ref))
    origin = (hx * centroid[1], hy * centroid[2])
    h = min(hx, hy)
    contact = enforce_contact ?
        _embedded_ghf_constraint(field, ref, normal, origin, h) : nothing
    T = promote_type(eltype(field), typeof(hx), typeof(hy))
    points = NTuple{3, T}[]
    function addpoint(x, y, w)
        w > 0 || return
        any(p -> hypot(p[1] - x, p[2] - y) <= h / 2, points) && return
        return push!(points, T.((x, y, w)))
    end
    for dj in -sample_radius:sample_radius, di in -sample_radius:sample_radius,
            axis in 1:2, orientation in (-1, 1)
        height = _local_height(values, i + 4 + di, j + 4 + dj, axis, orientation)
        isnothing(height) && continue
        x, y = axis == 1 ? ((di + height) * hx, dj * hy) : (di * hx, (dj + height) * hy)
        addpoint(x, y, max(0, orientation * normal[axis])^2)
    end
    fit = _ghf_fit(points, normal, origin, h; contact)
    !isnothing(fit) && return _orient_ghf_geometry(
        fit, normal, field, ref,
    )
    # Thesis fallback: fit neighbouring interface centroids with unit weights.
    # Try its immediate 3x3 neighbourhood first. If that contains fewer than
    # three separated points, grow the same local-level centroid fit by one
    # ring at a time. This is needed for a locally under-resolved tangent arc
    # (especially beside a wall); returning a WY plane with NaN curvature
    # would make the subsequent surface-tension force undefined.
    for radius in 1:3
        empty!(points)
        for dj in -radius:radius,di in -radius:radius
            ii, jj = i + 4 + di, j + 4 + dj
            c = values[ii, jj]
            isfinite(c) && 0 < c < 1 || continue
            local_i, local_j = i + di, j + dj
            if 1 <= local_i <= field.mesh.block_size[1] &&
                    1 <= local_j <= field.mesh.block_size[2]
                neighbour = CellRef(b, local_i, local_j)
                nx = _get(previous_normal_x, neighbour)
                ny = _get(previous_normal_y, neighbour)
                a = _get(previous_intercept, neighbour)
            else
                checkbounds(Bool, values, ii - 1, jj - 1) &&
                    checkbounds(Bool, values, ii + 1, jj + 1) || continue
                all(
                    isfinite,
                    @view(values[(ii - 1):(ii + 1), (jj - 1):(jj + 1)])
                ) || continue
                nx, ny = _wy_normal(values, ii, jj)
                a = _plane_intercept(c, nx, ny)
            end
            cx, cy = _interface_centroid(nx, ny, a)
            addpoint((di + cx) * hx, (dj + cy) * hy, 1.0)
        end
        fit = _ghf_fit(points, normal, origin, h; contact)
        !isnothing(fit) && return _orient_ghf_geometry(
            fit, normal, field, ref,
        )
    end
    return nothing
end

@inline function _orient_ghf_geometry(geometry, seed_normal, field, ref)
    alignment = geometry[1] * seed_normal[1] +
        geometry[2] * seed_normal[2]
    if !isnothing(field.geometry) &&
            zero(eltype(field)) <
            _get(field.geometry.solid.fraction, ref) < one(eltype(field))
        hx, hy = field.mesh.blocks[ref.block].spacing
        wx = _get(field.geometry.solid.normal_x, ref) / hx
        wy = _get(field.geometry.solid.normal_y, ref) / hy
        wall_scale = hypot(wx, wy)
        if wall_scale > eps(eltype(field))
            tx, ty = -wy / wall_scale, wx / wall_scale
            seed_tangent = seed_normal[1] * tx + seed_normal[2] * ty
            fitted_tangent = geometry[1] * tx + geometry[2] * ty
            tolerance = sqrt(eps(eltype(field)))
            if abs(seed_tangent) > tolerance &&
                    abs(fitted_tangent) > tolerance
                alignment = seed_tangent * fitted_tangent
            end
        end
    end
    alignment >= 0 && return geometry
    return -geometry[1], -geometry[2], -geometry[3]
end

# At a physical wall the centred height stencil has one column outside the
# domain.  Recover the PLIC normal from the boundary column and its first
# inward neighbour instead.  This is a geometric, one-sided two-point
# approximation: it deliberately contains no contact-angle information.
function _one_sided_wall_normal(field, values, b, i, j)
    mesh = field.mesh
    block = mesh.blocks[b]
    ref = CellRef(b, i, j)
    center = cellcenter(mesh, b, i, j)
    T = promote_type(eltype(field), typeof(block.spacing[1]))
    candidates = NTuple{3, T}[]
    for wall in 1:4
        axis = wall <= 2 ? 1 : 2
        mesh.periodic[axis] && continue
        inward = isodd(wall) ? 1 : -1
        boundary = mesh.origin[axis] +
            (inward == 1 ? zero(T) : mesh.extent[axis])
        abs(center[axis] - boundary) <=
            block.spacing[axis] * (T(0.5) + T(64) * eps(T)) || continue
        _plic_touches_wall(field, ref, wall) || continue
        tangent = 3 - axis

        # Interface represented as a height in the wall-tangent direction.
        # Its derivative in the wall-normal direction is one-sided.
        seed = tangent == 1 ?
            _get(field.normal_x, ref) : _get(field.normal_y, ref)
        orientation = seed < 0 ? -1 : 1
        h0 = _local_height(
            values, i + 4, j + 4, tangent, orientation,
        )
        ii, jj = axis == 1 ?
            (i + 4 + inward, j + 4) : (i + 4, j + 4 + inward)
        h1 = _local_height(values, ii, jj, tangent, orientation)
        if !isnothing(h0) && !isnothing(h1)
            H0, H1 = h0 * block.spacing[tangent],
                h1 * block.spacing[tangent]
            slope = inward * (H1 - H0) / block.spacing[axis]
            scale = sqrt(one(T) + slope * slope)
            principal = orientation / scale
            wall_component = -orientation * slope / scale
            nx, ny = tangent == 1 ?
                (principal, wall_component) : (wall_component, principal)
            # A steep graph is less accurate but is still preferable to
            # imposing an unrelated wall angle.  The bounded LHF quality is
            # retained for selecting between valid corner candidates.
            quality = max(_height_quality(abs(principal)), sqrt(eps(T)))
            push!(candidates, T.((nx, ny, quality)))
        end
    end
    isempty(candidates) && return nothing
    # The quality measure chooses the graph whose principal normal component
    # is largest.  Equal-quality candidates are averaged, which removes a
    # directional bias when both tangential neighbours are available.
    best = maximum(c[3] for c in candidates)
    tolerance = T(64) * eps(T) * max(one(T), best)
    nx = ny = weight = zero(T)
    for candidate in candidates
        candidate[3] + tolerance >= best || continue
        nx += candidate[3] * candidate[1]
        ny += candidate[3] * candidate[2]
        weight += candidate[3]
    end
    weight > 0 || return nothing
    scale = hypot(nx, ny)
    scale > eps(T) || return nothing
    return nx / scale, ny / scale
end

function _plic_touches_wall(field, ref, wall)
    T = eltype(field)
    axis = wall <= 2 ? 1 : 2
    inward = isodd(wall) ? 1 : -1
    normal_coordinate = -T(inward) / T(2)
    tangent = 3 - axis
    mx = _get(field.normal_x, ref)
    my = _get(field.normal_y, ref)
    intercept = _get(field.intercept, ref)
    normal_component = axis == 1 ? mx : my
    tangent_component = tangent == 1 ? mx : my
    values = (
        normal_component * normal_coordinate - tangent_component / T(2) - intercept,
        normal_component * normal_coordinate + tangent_component / T(2) - intercept,
    )
    tolerance = T(128) * eps(T) *
        max(one(T), abs(intercept), abs(mx), abs(my))
    return min(values...) <= tolerance && max(values...) >= -tolerance
end


function _wall_height(values, i, j, axis, inward, orientation)
    height = zero(eltype(values));previous = one(eltype(values))
    for k in 0:3
        ii, jj = axis == 1 ? (i + inward * k, j) : (i, j + inward * k)
        checkbounds(Bool, values, ii, jj) || return nothing
        c = values[ii, jj]
        isfinite(c) || return nothing
        v = orientation > 0 ? c : 1 - c
        v <= previous + 64eps(eltype(values)) || return nothing
        height += v;previous = v
    end
    previous <= 64eps(eltype(values)) || return nothing
    return height
end

# Thesis st_curvature.tex: height Neumann data for near-right angles; a
# quadratic with unknown contact location for acute/obtuse angles.  The
# domain-wall contact angle is a curvature boundary condition. It must not
# replace the normal or PLIC plane used to advect the interface.
function _contact_curvature(field, values, b, i, j)
    all(isnothing, field.contact_angles) && return nothing
    mesh = field.mesh;block = mesh.blocks[b];ref = CellRef(b, i, j)
    center = cellcenter(mesh, b, i, j)
    for wall in 1:4
        theta = field.contact_angles[wall]
        isnothing(theta) && continue
        axis = wall <= 2 ? 1 : 2
        mesh.periodic[axis] && continue
        inward = isodd(wall) ? 1 : -1
        boundary = mesh.origin[axis] + (inward == 1 ? 0 : mesh.extent[axis])
        abs(center[axis] - boundary) <= block.spacing[axis] * (0.5 + 64eps(eltype(field))) || continue
        _plic_touches_wall(field, ref, wall) || continue
        tangent = 3 - axis
        if abs(theta - pi / 2) <= pi / 4
            principal = tangent
            seed = principal == 1 ? _get(field.normal_x, ref) : _get(field.normal_y, ref)
            orientation = seed < 0 ? -1 : 1
            h0 = _local_height(values, i + 4, j + 4, principal, orientation)
            ii, jj = axis == 1 ? (i + 4 + inward, j + 4) : (i + 4, j + 4 + inward)
            h1 = _local_height(values, ii, jj, principal, orientation)
            (isnothing(h0)||isnothing(h1)) && continue
            normal_h, tangent_h = block.spacing[principal], block.spacing[axis]
            H0, H1 = h0 * normal_h, h1 * normal_h
            derivative = -cos(theta) / (orientation * sin(theta))
            Hghost = H0 - derivative * tangent_h
            slope = (H1 - Hghost) / (2tangent_h)
            second = (H1 - 2H0 + Hghost) / tangent_h^2
            scale = sqrt(1 + slope * slope)
            return -orientation * second / scale^3
        else
            orientation = cos(theta) > 0 ? 1 : -1
            seed = tangent == 1 ? _get(field.normal_x, ref) : _get(field.normal_y, ref)
            direction = -(seed < 0 ? -1 : 1) * orientation
            ii, jj = tangent == 1 ? (i + 4 + direction, j + 4) : (i + 4, j + 4 + direction)
            h0 = _wall_height(values, ii, jj, axis, inward, orientation)
            ii, jj = tangent == 1 ? (ii + direction, jj) : (ii, jj + direction)
            h1 = _wall_height(values, ii, jj, axis, inward, orientation)
            (isnothing(h0)||isnothing(h1)) && continue
            H0, H1 = h0 * block.spacing[axis], h1 * block.spacing[axis]
            H1 > H0 > 0 || continue
            h = block.spacing[tangent];slope_wall = abs(tan(theta))
            root = sqrt(4H0 * H1 + h * h * slope_wall^2)
            # Rationalization avoids cancellation for a nearly planar interface.
            A = ((H1 - H0)^2 - h * h * slope_wall^2) / (h * h * (H0 + H1 + root))
            B = (H1 - H0) / h - A * h
            slope = B - 2A * h
            scale = sqrt(1 + slope * slope)
            return -orientation * 2A / scale^3
        end
    end
    return nothing
end
