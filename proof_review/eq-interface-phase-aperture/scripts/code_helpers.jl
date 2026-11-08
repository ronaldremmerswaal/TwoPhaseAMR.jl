# Helpers shared by the code_*.jl scripts of the audit of eq:interface:phase-aperture
# (they need `TwoPhaseAMR` loaded as `AMR`).  Not a check by itself.
#
# Independent evaluation of the phase apertures from the stored PLIC planes and solid planes: polygons are
# built by Sutherland--Hodgman clipping in the local coordinates of a cell, the face segment is intersected with
# the convex polygons by Cyrus--Beck clipping.  Nothing of `_face_aperture` / `_phase_face_apertures` is reused.

using LinearAlgebra
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

const SQUARE = rect_poly(-0.5, 0.5, -0.5, 0.5)

"""Length of `segment(p0,p1) ∩ polygon` for a convex counter-clockwise polygon (Cyrus--Beck)."""
function seg_poly_length(p0, p1, poly)
    length(poly) < 3 && return 0.0
    d = (p1[1] - p0[1], p1[2] - p0[2])
    t0, t1 = 0.0, 1.0
    m = length(poly)
    scale = hypot(d...)
    for k in 1:m
        a, b = poly[k], poly[mod1(k + 1, m)]
        e = (b[1] - a[1], b[2] - a[2])
        el = hypot(e...)
        el > 0 || continue
        num = (e[1] * (p0[2] - a[2]) - e[2] * (p0[1] - a[1])) / el      # signed distance of p0 to the edge line (inside > 0)
        den = (e[1] * d[2] - e[2] * d[1]) / el                          # rate of change along the segment
        if abs(den) <= 1.0e-14 * scale
            num < -1.0e-13 && return 0.0
            continue
        end
        t = -num / den
        den > 0 ? (t0 = max(t0, t)) : (t1 = min(t1, t))
    end
    return t1 > t0 ? (t1 - t0) * scale : 0.0
end

"""Open, liquid and gas polygons of a cell (local coordinates), from the stored solid and phase planes."""
function cell_polygons(field, geometry, ref)
    c = AMR._get(field.conditional_fraction, ref)
    alpha = isnothing(geometry) ? 0.0 : AMR._get(geometry.solid.fraction, ref)
    alpha >= 1 && return (; open = NTuple{2, Float64}[], liquid = NTuple{2, Float64}[], gas = NTuple{2, Float64}[], covered = true)
    open = SQUARE
    if alpha > 0
        sn = (AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref))
        open = clip_halfplane(SQUARE, sn, AMR._get(geometry.solid.intercept, ref); keep_less = false)
    end
    if c <= 0
        return (; open, liquid = NTuple{2, Float64}[], gas = open, covered = false)
    elseif c >= 1
        return (; open, liquid = open, gas = NTuple{2, Float64}[], covered = false)
    end
    n = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref))
    a = AMR._get(field.intercept, ref)
    return (; open, liquid = clip_halfplane(open, n, a; keep_less = true), gas = clip_halfplane(open, n, a; keep_less = false), covered = false)
end

"""Tangential extent of the face segment in the local coordinates of `ref` (non-periodic meshes)."""
function face_interval(mesh, ref, face)
    t = 3 - face.axis
    centre = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
    h = mesh.blocks[ref.block].spacing[t]
    lo = (face.center[t] - face.area / 2 - centre[t]) / h
    return lo, lo + face.area / h
end

"""Open, liquid, gas traces (fractions of the face segment) of `face` seen from cell `ref` at local side `side` = ±1/2."""
function traces(field, geometry, ref, face, side)
    polys = cell_polygons(field, geometry, ref)
    lo, hi = face_interval(field.mesh, ref, face)
    p0, p1 = face.axis == 1 ? ((side, lo), (side, hi)) : ((lo, side), (hi, side))
    w = hi - lo
    return (;
        open = seg_poly_length(p0, p1, polys.open) / w, liquid = seg_poly_length(p0, p1, polys.liquid) / w,
        gas = seg_poly_length(p0, p1, polys.gas) / w, covered = polys.covered,
    )
end

"""Liquid and gas apertures of an interior face by the thesis' definition (2.41): half the sum of the one-sided traces."""
function thesis_apertures(field, geometry, face)
    m = traces(field, geometry, face.minus, face, 0.5)
    p = traces(field, geometry, face.plus, face, -0.5)
    return (; liquid = (m.liquid + p.liquid) / 2, gas = (m.gas + p.gas) / 2, open = (m.open + p.open) / 2, minus = m, plus = p)
end

# ---------------------------------------------------------------------------------------------------------------
# Fixtures

"""Liquid film y <= ysurf(x) above the sloped wall y = y0 + k (x - x0) (solid below), contact points where they cross."""
function film_fixture(; amr = false, hx = 1.0, hy = 1.0, k = 0.18, y0 = 1.3, x0 = 8.0, amp = 1.2, wave = 0.55, base = 1.8, phase = 3.0, window = (6.0, 10.0, 0.0, 4.5))
    nx, ny = 16, 8
    refine = if amr
        (key, origin, extent) -> begin
            x1, x2 = origin[1], origin[1] + extent[1]
            y1, y2 = origin[2], origin[2] + extent[2]
            wx1, wx2, wy1, wy2 = window .* (hx, hx, hy, hy)
            x1 < wx2 && x2 > wx1 && y1 < wy2 && y2 > wy1
        end
    else
        (key, origin, extent) -> false
    end
    mesh = AMR.BlockTreeMesh(;
        base_mesh = (nx, ny), block_size = amr ? (4, 4) : (nx, ny), extent = (nx * hx, ny * hy),
        periodic = (false, false), max_level = amr ? 2 : 0, refine,
    )
    # physical wall: y = y0*hy + k*(hy/hx)*(x - x0*hx), i.e. y' = y0 + k (x' - x0) in cell units; solid below
    solid_ls(x, y) = y - y0 * hy - k * (hy / hx) * (x - x0 * hx)
    geometry = AMR.GeometryField(mesh, solid_ls)
    surf(x) = hy * (base + amp * sin(wave * (x / hx - phase)))
    field = AMR.InterfaceField(
        mesh, (x, y) -> y - surf(x); geometry, contact_angle = nothing, embedded_contact_angle = nothing,
    )
    return field, geometry
end
