# Helpers shared by the code_*.jl and sharp_*.jl scripts of the audit of eq:interface:contact-row
# (they need `TwoPhaseAMR` loaded as `AMR`).  Not a check by itself.
#
# Mirrors the point selection of `_ghf_geometry` (src/interface/height_functions.jl:546-600) so that the
# package's `_ghf_fit` and an independent weighted least-squares solve see the same rows.

using LinearAlgebra

# wall y = y0 + k (x - x0) (solid below), liquid drop of radius R meeting it at theta (measured in the liquid) at x = xc
function triple_field(; slope, y0, x0, radius, theta, xc, hx = 1.0, hy = 1.0, method = :lhf_ghf, embedded = theta)
    nx, ny = 16, 8
    mesh = AMR.BlockTreeMesh(; base_mesh = (nx, ny), block_size = (nx, ny), extent = (nx * hx, ny * hy), periodic = (false, false))
    scale = hypot(1, slope)
    geometry = AMR.GeometryField(mesh, (x, y) -> (y - y0 - slope * (x - x0)) / scale)
    nw = (-slope, 1.0) ./ scale
    tw = (1.0, slope) ./ scale
    p = (xc, y0 + slope * (xc - x0))
    eta = cos(theta) .* nw .+ sin(theta) .* tw
    centre = p .- radius .* eta
    drop(x, y) = hypot(x - centre[1], y - centre[2]) - radius
    field = AMR.InterfaceField(mesh, drop; geometry, embedded_contact_angle = embedded, contact_angle = nothing, reconstruction_method = method)
    return field, geometry, (; centre, contact = p, nw, tw)
end

"""
Quantities of `_ghf_geometry` for a cell, following its logic: the height points of the 3x3 block; if the package's
`_ghf_fit` cannot use them (fewer than three rows, singular), the PLIC-segment midpoints of the mixed cells within
rings of radius 1, 2, 3 (unit weights, previous-pass planes).  Returns the first set for which the package's fit works.
"""
function ghf_setup(field, values, ref)
    b, i, j = ref.block, ref.i, ref.j
    hx, hy = field.mesh.blocks[b].spacing
    nxc, nyc = field.mesh.block_size
    mx, my = AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref)
    norm = hypot(mx / hx, my / hy)
    normal = (mx / hx / norm, my / hy / norm)
    centroid = AMR._interface_centroid(mx, my, AMR._get(field.intercept, ref))
    origin = (hx * centroid[1], hy * centroid[2])
    h = min(hx, hy)
    contact = AMR._embedded_ghf_constraint(field, ref, normal, origin, h)
    function addpoint!(points, x, y, w)
        w > 0 || return
        any(q -> hypot(q[1] - x, q[2] - y) <= h / 2, points) && return
        return push!(points, (x, y, w))
    end
    points = NTuple{3, Float64}[]
    for dj in -1:1, di in -1:1, axis in 1:2, orientation in (-1, 1)
        height = AMR._local_height(values, i + 4 + di, j + 4 + dj, axis, orientation)
        isnothing(height) && continue
        x, y = axis == 1 ? ((di + height) * hx, dj * hy) : (di * hx, (dj + height) * hy)
        addpoint!(points, x, y, max(0, orientation * normal[axis])^2)
    end
    AMR._ghf_fit(points, normal, origin, h; contact) !== nothing && return normal, origin, h, contact, points, 0
    for radius in 1:3
        points = NTuple{3, Float64}[]
        for dj in -radius:radius, di in -radius:radius
            c = values[i + 4 + di, j + 4 + dj]
            isfinite(c) && 0 < c < 1 || continue
            li, lj = i + di, j + dj
            (1 <= li <= nxc && 1 <= lj <= nyc) || error("ring leaves the block: fixture too small")
            neighbour = AMR.CellRef(b, li, lj)
            cx, cy = AMR._interface_centroid(
                AMR._get(field.previous_normal_x, neighbour), AMR._get(field.previous_normal_y, neighbour),
                AMR._get(field.previous_intercept, neighbour),
            )
            addpoint!(points, (di + cx) * hx, (dj + cy) * hy, 1.0)
        end
        AMR._ghf_fit(points, normal, origin, h; contact) !== nothing && return normal, origin, h, contact, points, radius
    end
    return normal, origin, h, contact, points, -1
end

"""Independent weighted least squares of (4.15) with the row (4.20) scaled by `alpha` (nothing: no row)."""
function ls_independent(points, normal, origin, h, contact; alpha, row_rhs = nothing)
    n = normal; t = (-n[2], n[1])
    rows = Vector{NTuple{3, Float64}}(); rhs = Float64[]
    for (x, y, w) in points
        dx, dy = (x - origin[1]) / h, (y - origin[2]) / h
        tau, nu = t[1] * dx + t[2] * dy, n[1] * dx + n[2] * dy
        push!(rows, sqrt(w) .* (tau^2, tau, 1.0)); push!(rhs, sqrt(w) * nu)
    end
    if contact !== nothing && alpha !== nothing
        ts, nw, cs = contact
        proj = t[1] * nw[1] + t[2] * nw[2]
        push!(rows, alpha .* (-2ts * proj, -proj, 0.0)); push!(rhs, alpha * (row_rhs === nothing ? cs - n[1] * nw[1] - n[2] * nw[2] : row_rhs))
    end
    length(rows) >= 3 || return nothing
    M = [r[k] for r in rows, k in 1:3]
    sv = svdvals(M)
    minimum(sv) > 1.0e-10 * maximum(sv) || return nothing
    a2, a1, a0 = M \ rhs
    scale = sqrt(1 + a1^2)
    return (; a2, a1, a0, normal = ((n[1] - a1 * t[1]) / scale, (n[2] - a1 * t[2]) / scale), kappa = -2a2 / (h * scale^3))
end
