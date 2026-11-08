# Helpers shared by the audits of Chapters 4 and 7 of the technical reference
# (plain Julia, standard library only; usable under both environments).
#
# Not a check by itself.  Include with
#     isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
#     using .Geo

module Geo

export clip_halfplane, shoelace, rect_poly, halfplane_area, segment_in, sz_plane_constant,
    plane_area_reference

"""Sutherland--Hodgman clip of a polygon against `{n.x <= a}` (or `>= a`).
Generic in the number type (Float64, BigFloat, Rational)."""
function clip_halfplane(poly, n, a; keep_less = true)
    m = length(poly)
    out = similar(poly, 0)
    m == 0 && return out
    value(p) = n[1] * p[1] + n[2] * p[2] - a
    inside(v) = keep_less ? v <= 0 : v >= 0
    for k in 1:m
        p, q = poly[k], poly[mod1(k + 1, m)]
        vp, vq = value(p), value(q)
        inside(vp) && push!(out, p)
        if inside(vp) != inside(vq)
            t = vp / (vp - vq)
            push!(out, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        end
    end
    return out
end

"""Absolute polygon area (shoelace); zero for fewer than three vertices."""
function shoelace(poly)
    m = length(poly)
    m < 3 && return 0.0
    twice = sum(poly[k][1] * poly[mod1(k + 1, m)][2] - poly[mod1(k + 1, m)][1] * poly[k][2] for k in 1:m)
    return abs(twice) / 2
end

rect_poly(xlo, xhi, ylo, yhi) = [(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]

"""Area of `poly ∩ {n.x <= a}` (`keep_less = true`) or `poly ∩ {n.x >= a}`."""
halfplane_area(poly, n, a; keep_less = true) = shoelace(clip_halfplane(poly, n, a; keep_less))

"""The two end points of the line `{n.x = a}` inside a convex polygon."""
function segment_in(poly, n, a)
    points = similar(poly, 0)
    m = length(poly)
    for k in 1:m
        p, q = poly[k], poly[mod1(k + 1, m)]
        vp = n[1] * p[1] + n[2] * p[2] - a
        vq = n[1] * q[1] + n[2] * q[2] - a
        if (vp <= 0) != (vq <= 0)
            t = vp / (vp - vq)
            push!(points, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        end
    end
    length(points) == 2 || error("the line does not cross the polygon in two points")
    return points[1], points[2]
end

"""Scardovelli--Zaleski (2000) closed-form inverse of the area fraction of the unit square
`[-1/2,1/2]^2` below `{nx x + ny y = a}`; returns the plane constant `a` measured from the centre.
Generic in the number type."""
function sz_plane_constant(c, nx, ny)
    s = abs(nx) + abs(ny)
    m1, m2 = minmax(abs(nx), abs(ny)) ./ s
    v1 = m1 / (2m2)
    alpha = if c <= v1
        sqrt(2m1 * m2 * c)
    elseif c <= 1 - v1
        m2 * c + m1 / 2
    else
        1 - sqrt(2m1 * m2 * (1 - c))
    end
    return s * (alpha - 1 / 2)
end

"""Fraction of the rectangle `[xlo,xhi]x[ylo,yhi]` with `nx x + ny y <= a`, by exact polygon
clipping (reference for the closed-form area function)."""
function plane_area_reference(nx, ny, a, xlo, xhi, ylo, yhi)
    area = (xhi - xlo) * (yhi - ylo)
    return halfplane_area(rect_poly(xlo, xhi, ylo, yhi), (nx, ny), a) / area
end

end # module Geo
