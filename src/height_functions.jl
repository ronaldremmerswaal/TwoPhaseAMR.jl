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
    for (b, block) in pairs(mesh.blocks)
        push!(cache.keys, block.key)
        push!(cache.values, fill(eltype(field)(NaN), nx + 8, ny + 8))
        dx, dy = block.spacing
        for j in -3:(ny + 4),i in -3:(nx + 4)
            1 <= i <= nx && 1 <= j <= ny && continue
            x, y = cellcenter(mesh, b, i, j)
            if (
                    !mesh.periodic[1] &&
                        !(mesh.origin[1] < x < mesh.origin[1] + mesh.extent[1])
                ) ||
                    (
                    !mesh.periodic[2] &&
                        !(mesh.origin[2] < y < mesh.origin[2] + mesh.extent[2])
                )
                continue # physical boundaries do not provide valid height columns
            end
            x = mesh.periodic[1] ? mesh.origin[1] + mod(x - mesh.origin[1], mesh.extent[1]) : x
            y = mesh.periodic[2] ? mesh.origin[2] + mod(y - mesh.origin[2], mesh.extent[2]) : y
            donors = HeightDonor{T}[]
            # A target lies on the dyadic lattice, hence does not cross a
            # periodic boundary after its centre has been wrapped.
            empty!(candidate_blocks)
            _collect_leaf_blocks_in_window!(
                candidate_blocks, mesh.root, indices_by_key,
                (x - dx / 2, y - dy / 2),
                (x + dx / 2, y + dy / 2),
            )
            sort!(candidate_blocks)
            for db in candidate_blocks
                other = mesh.blocks[db]
                xl = max(x - dx / 2, other.origin[1])
                xh = min(x + dx / 2, other.origin[1] + other.extent[1])
                yl = max(y - dy / 2, other.origin[2])
                yh = min(y + dy / 2, other.origin[2] + other.extent[2])
                xh > xl && yh > yl || continue
                hx, hy = other.spacing
                ilo = max(1, floor(Int, (xl - other.origin[1]) / hx) + 1)
                ihi = min(nx, ceil(Int, (xh - other.origin[1]) / hx))
                jlo = max(1, floor(Int, (yl - other.origin[2]) / hy) + 1)
                jhi = min(ny, ceil(Int, (yh - other.origin[2]) / hy))
                for dj in jlo:jhi,di in ilo:ihi
                    cx, cy = cellcenter(mesh, db, di, dj)
                    left = max(xl, cx - hx / 2)
                    right = min(xh, cx + hx / 2)
                    low = max(yl, cy - hy / 2)
                    high = min(yh, cy + hy / 2)
                    right > left && high > low || continue
                    bounds = (
                        T((left - cx) / hx), T((right - cx) / hx),
                        T((low - cy) / hy), T((high - cy) / hy),
                    )
                    weight = T((right - left) * (high - low) / (dx * dy))
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
    nx, ny = mesh.block_size
    _parallel_for(nblocks(mesh); grain = 4) do b
        cache.values[b][5:(nx + 4), 5:(ny + 4)] .= interior(field, b)
    end
    @_parallel for ghost_index in 1:length(cache.ghosts)
        ghost = cache.ghosts[ghost_index]
        value = zero(eltype(field))
        for donor in ghost.donors
            ref = donor.cell
            c = _get(field.fraction, ref)
            portion = c == 0 || c == 1 ? c : _plane_fraction(
                    _get(field.normal_x, ref), _get(field.normal_y, ref),
                    _get(field.intercept, ref), donor.bounds...
                )
            value += donor.weight * portion
        end
        cache.values[ghost.block][ghost.i, ghost.j] = clamp(value, zero(value), one(value))
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
    candidates = NTuple{4, T}[]
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
        push!(candidates, T.((nx, ny, curvature, quality)))
    end
    isempty(candidates) && return nothing
    nx = ny = kappa = weight = zero(T)
    for (index, c) in pairs(candidates)
        w = c[4] * prod(
            one(T) - other[4]
                for (j, other) in pairs(candidates) if index != j;
            init = one(T)
        )
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

function _interface_centroid(nx, ny, a)
    T = promote_type(typeof(nx), typeof(ny), typeof(a))
    half = T(0.5)
    tolerance = T(64) * eps(T)
    points = NTuple{2, T}[]
    for x in (-half, half)
        if abs(ny) > eps(T)
            y = (a - nx * x) / ny
            -half - tolerance <= y <= half + tolerance &&
                push!(points, (x, clamp(y, -half, half)))
        end
    end
    for y in (-half, half)
        if abs(nx) > eps(T)
            x = (a - ny * y) / nx
            -half - tolerance <= x <= half + tolerance &&
                push!(points, (clamp(x, -half, half), y))
        end
    end
    isempty(points) && return (zero(T), zero(T))
    # Remove duplicate corner intersections before averaging segment endpoints.
    unique!(points)
    return (sum(p[1] for p in points) / length(points), sum(p[2] for p in points) / length(points))
end

function _ghf_fit(points, normal, origin, h)
    length(points) >= 3 || return nothing
    nx, ny = normal
    T = promote_type(typeof(nx), typeof(ny), typeof(origin[1]), typeof(h))
    matrix = Matrix{T}(undef, length(points), 3)
    rhs = Vector{T}(undef, length(points))
    for (i, (x, y, w)) in pairs(points)
        dx, dy = (x - origin[1]) / h, (y - origin[2]) / h
        tangent = -ny * dx + nx * dy
        weight = sqrt(w)
        matrix[i, :] .= (weight * tangent^2, weight * tangent, weight)
        rhs[i] = weight * (nx * dx + ny * dy)
    end
    singular = svdvals(matrix)
    rank_tolerance = max(T(1.0e-10), T(64) * eps(T))
    minimum(singular) > rank_tolerance * maximum(singular) || return nothing
    a, b, _ = matrix \ rhs
    scale = sqrt(1 + b * b)
    return (nx + b * ny) / scale, (ny - b * nx) / scale, -2a / (h * scale^3)
end
function _ghf_geometry(
        field, values, b, i, j,
        previous_normal_x = field.normal_x,
        previous_normal_y = field.normal_y,
        previous_intercept = field.intercept,
    )
    ref = CellRef(b, i, j)
    hx, hy = field.mesh.blocks[b].spacing
    mx, my = _get(field.normal_x, ref), _get(field.normal_y, ref)
    norm = hypot(mx / hx, my / hy)
    normal = (mx / hx / norm, my / hy / norm)
    centroid = _interface_centroid(mx, my, _get(field.intercept, ref))
    origin = (hx * centroid[1], hy * centroid[2])
    h = min(hx, hy)
    T = promote_type(eltype(field), typeof(hx), typeof(hy))
    points = NTuple{3, T}[]
    function addpoint(x, y, w)
        w > 0 || return
        any(p -> hypot(p[1] - x, p[2] - y) <= h / 2, points) && return
        return push!(points, T.((x, y, w)))
    end
    for dj in -1:1,di in -1:1,axis in 1:2,orientation in (-1, 1)
        height = _local_height(values, i + 4 + di, j + 4 + dj, axis, orientation)
        isnothing(height) && continue
        x, y = axis == 1 ? ((di + height) * hx, dj * hy) : (di * hx, (dj + height) * hy)
        addpoint(x, y, max(0, orientation * normal[axis])^2)
    end
    fit = _ghf_fit(points, normal, origin, h)
    !isnothing(fit) && return fit
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
        fit = _ghf_fit(points, normal, origin, h)
        !isnothing(fit) && return fit
    end
    return nothing
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
# quadratic with unknown contact location for acute/obtuse angles. Angles
# satisfy n_liquid · n_wall,inward = cos(theta).
function _contact_geometry(field, values, b, i, j)
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
            np = orientation / scale
            nw = -orientation * slope * inward / scale
            nx, ny = principal == 1 ? (np, nw) : (nw, np)
            return nx, ny, -orientation * second / scale^3
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
            np = inward * orientation / scale
            nt = -orientation * slope * direction / scale
            nx, ny = axis == 1 ? (np, nt) : (nt, np)
            return nx, ny, -orientation * 2A / scale^3
        end
    end
    return nothing
end
