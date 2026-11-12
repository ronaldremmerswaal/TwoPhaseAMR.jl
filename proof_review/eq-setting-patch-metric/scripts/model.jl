# Exact integer model of a vertex patch on a dyadic quadtree (no dependencies).
#
# Label:   eq:setting:patch-metric  (docs/techref/chapters/03_discrete_setting.tex:256-305)
# Purpose: shared helper of the scripts in this folder; not a check by itself.
#
# For the cells that touch one vertex v the model builds the face segments of the
# vertex patch F(v) by the same rules as TwoPhaseAMR:
#   * `_match_sides!` (src/mesh/topology/mesh.jl:782): the face between two cells is the
#     overlap of their sides; `minus` is the cell whose right/top side lies on the common
#     line, `plus` the cell whose left/bottom side lies on it; area = overlap length;
#   * `_face_displacement` (src/operators/gradient.jl:102): d_f = centre(plus) - centre(minus),
#     with the sign of the normal component fixed by the face orientation and its magnitude
#     taken from the shortest periodic image (half-period tie resolved, #36);
#   * `_vertex_key` (src/operators/gradient.jl:119): a face belongs to the patches of the
#     two endpoints of its overlap segment;
#   * `_local_inverse_metric` (src/operators/gradient.jl:153): rows area*d_f' of Y and
#     (1/2) e_axis' of N, T = Y'N.
#
# Lattice units: all lengths are integers in units of the smallest cell side at the vertex
# (x-units hx, y-units hy when the cells are stretched). `d2` stores TWICE the displacement
# so that everything stays integer.

module PatchModel

export Cell, Face, centre2, touching, faces_at_vertex, moment_integers, physical_moment, half_line,
    corner_cells, hanging_cells, wall_cells, all_trees, leaves, d4_images, sigma_pattern,
    face_graded, corner_balanced, kind_of, class_of

"""Axis-aligned square cell: lower-left corner `(x, y)` and side `s` (lattice units)."""
struct Cell
    x::Int
    y::Int
    s::Int
end

"""Face segment of a patch. `axis` is the normal axis (1 = vertical face, 2 = horizontal)."""
struct Face
    axis::Int
    len::Int
    d2::NTuple{2, Int}
    minus::Cell
    plus::Cell
    lo::Int          # segment endpoints along the tangent (lattice coordinates)
    hi::Int
end

centre2(c::Cell) = (2c.x + c.s, 2c.y + c.s)

closure_contains(c::Cell, vx, vy) = c.x <= vx <= c.x + c.s && c.y <= vy <= c.y + c.s

touching(cells, vx, vy) = [c for c in cells if closure_contains(c, vx, vy)]

"""
All face segments of the complex given by `cells` that END at the vertex `(vx, vy)`.
(`cells` may be any superset of the cells touching the vertex.)
"""
function faces_at_vertex(cells::AbstractVector{Cell}, vx::Int, vy::Int)
    faces = Face[]
    for A in cells, B in cells
        # axis 1: vertical face, A is `minus` (left), B is `plus` (right)
        if A.x + A.s == B.x
            lo = max(A.y, B.y)
            hi = min(A.y + A.s, B.y + B.s)
            if lo < hi && vx == B.x && (vy == lo || vy == hi)
                ca, cb = centre2(A), centre2(B)
                push!(faces, Face(1, hi - lo, (cb[1] - ca[1], cb[2] - ca[2]), A, B, lo, hi))
            end
        end
        # axis 2: horizontal face, A is `minus` (below), B is `plus` (above)
        if A.y + A.s == B.y
            lo = max(A.x, B.x)
            hi = min(A.x + A.s, B.x + B.s)
            if lo < hi && vy == B.y && (vx == lo || vx == hi)
                ca, cb = centre2(A), centre2(B)
                push!(faces, Face(2, hi - lo, (cb[1] - ca[1], cb[2] - ca[2]), A, B, lo, hi))
            end
        end
    end
    return faces
end

"""Direction (:N, :S, :E, :W) of the half-line from v = (vx, vy) that carries the face segment."""
function half_line(f::Face, vx::Int, vy::Int)
    f.axis == 1 && return f.lo == vy ? :N : :S
    return f.lo == vx ? :E : :W
end

"""
`F[i, j] = sum_{f: axis(f)=j} len_f * d2_f[i]`, so that with cell widths `(hx, hy)` of one
lattice unit the moment tensor `T = Y'N` is
`T = (1/4) [hx*hy*F11  hx^2*F12; hy^2*F21  hx*hy*F22]`.
(area = len*hy for axis 1, len*hx for axis 2; displacement = (hx*d2x, hy*d2y)/2.)
"""
function moment_integers(faces::AbstractVector{Face})
    F = zeros(Int, 2, 2)
    for f in faces, i in 1:2
        F[i, f.axis] += f.len * f.d2[i]
    end
    return F
end

"""`T = Y'N` for lattice-unit cell widths `hx`, `hy` (any number type, also symbolic)."""
function physical_moment(F, hx, hy)
    return [
        hx * hy * F[1, 1] / 4 hx^2 * F[1, 2] / 4
        hy^2 * F[2, 1] / 4 hx * hy * F[2, 2] / 4
    ]
end

# ---------------------------------------------------------------------------------------
# Configurations around v = (0, 0)
# ---------------------------------------------------------------------------------------

"""Four cells with a corner at v: sides (NE, NW, SW, SE), lattice units."""
corner_cells(sNE, sNW, sSW, sSE) =
    [Cell(0, 0, sNE), Cell(-sNW, 0, sNW), Cell(-sSW, -sSW, sSW), Cell(0, -sSE, sSE)]

"""Hanging vertex: coarse cell (side 2) on `side`, two unit cells on the opposite side."""
function hanging_cells(side::Symbol)
    side == :W && return [Cell(-2, -1, 2), Cell(0, 0, 1), Cell(0, -1, 1)]
    side == :E && return [Cell(0, -1, 2), Cell(-1, 0, 1), Cell(-1, -1, 1)]
    side == :S && return [Cell(-1, -2, 2), Cell(0, 0, 1), Cell(-1, 0, 1)]
    side == :N && return [Cell(-1, 0, 2), Cell(0, -1, 1), Cell(-1, -1, 1)]
    error("unknown side $side")
end

"""Vertex on a wall (cells only on the interior side): `wall` in (:left, :right, :bottom, :top)."""
function wall_cells(wall::Symbol, s1, s2)
    wall == :left && return [Cell(0, 0, s1), Cell(0, -s2, s2)]       # NE, SE
    wall == :right && return [Cell(-s1, 0, s1), Cell(-s2, -s2, s2)]  # NW, SW
    wall == :bottom && return [Cell(0, 0, s1), Cell(-s2, 0, s2)]     # NE, NW
    wall == :top && return [Cell(0, -s1, s1), Cell(-s2, -s2, s2)]    # SE, SW
    error("unknown wall $wall")
end

# ---------------------------------------------------------------------------------------
# Quadtrees on a window (exhaustive enumeration)
# ---------------------------------------------------------------------------------------

"""
All dyadic trees of depth <= `depth`: `:leaf` or a `Vector{Any}` (SW, SE, NW, NE) of subtrees.
(Vectors, not tuples: nested tuple types would force one compilation per tree shape.)
"""
function all_trees(depth::Int)
    depth == 0 && return Any[:leaf]
    sub = all_trees(depth - 1)
    out = Any[:leaf]
    for a in sub, b in sub, c in sub, d in sub
        push!(out, Any[a, b, c, d])
    end
    return out
end

function leaves!(out::Vector{Cell}, tree::Union{Symbol, Vector{Any}}, x::Int, y::Int, s::Int)
    if tree === :leaf
        push!(out, Cell(x, y, s))
    else
        h = s ÷ 2
        leaves!(out, tree[1], x, y, h)
        leaves!(out, tree[2], x + h, y, h)
        leaves!(out, tree[3], x, y + h, h)
        leaves!(out, tree[4], x + h, y + h, h)
    end
    return out
end

leaves(tree, s) = leaves!(Cell[], tree, 0, 0, s)

# ---------------------------------------------------------------------------------------
# Classification of a patch
# ---------------------------------------------------------------------------------------

"""Grading across face segments: every face of the patch joins cells with side ratio <= 2."""
face_graded(faces) = all(f -> max(f.minus.s, f.plus.s) <= 2 * min(f.minus.s, f.plus.s), faces)

"""2:1 balance including corner contact: all cells touching v have side ratio <= 2."""
function corner_balanced(touch)
    isempty(touch) && return true
    return maximum(c.s for c in touch) <= 2 * minimum(c.s for c in touch)
end

"""
Kind of the cells touching the vertex `(vx, vy)`:
`:corner4` (four cells with a corner at v), `:hanging` (v inside a side of one cell),
`:interior` (v inside a cell: no face ends at v), `:other`.
"""
function kind_of(touch, vx, vy)
    iscorner(c) = (vx == c.x || vx == c.x + c.s) && (vy == c.y || vy == c.y + c.s)
    isinterior(c) = c.x < vx < c.x + c.s && c.y < vy < c.y + c.s
    any(isinterior, touch) && return :interior
    all(iscorner, touch) && length(touch) == 4 && return :corner4
    count(!iscorner, touch) == 1 && length(touch) == 3 && return :hanging
    return :other
end

"""Side pattern (NE, NW, SW, SE) of a `:corner4` configuration, relative to the smallest side."""
function sigma_pattern(touch, vx, vy)
    smin = minimum(c.s for c in touch)
    pat = zeros(Int, 4)
    for c in touch
        q = c.x == vx ? (c.y == vy ? 1 : 4) : (c.y == vy ? 2 : 3)  # NE, NW?, SW, SE
        # lower-left corner at v => NE; cell to the west with lower-left (v-s, v) => NW; ...
        pat[q] = c.s ÷ smin
    end
    return Tuple(pat)
end

"""The 8 images of a pattern (NE, NW, SW, SE) under the dihedral group of the square."""
function d4_images(p::NTuple{4, Int})
    ne, nw, sw, se = p
    # counter-clockwise order NE -> NW -> SW -> SE
    rot(q) = (q[4], q[1], q[2], q[3])
    refl(q) = (q[2], q[1], q[4], q[3])       # x -> -x : NE<->NW, SW<->SE
    images = Set{NTuple{4, Int}}()
    q = p
    for _ in 1:4
        push!(images, q)
        push!(images, refl(q))
        q = rot(q)
    end
    return images
end

"""Canonical D4 class name of a `:corner4` pattern with sides in {1,2} (F = 1, C = 2)."""
function class_of(p::NTuple{4, Int})
    images = d4_images(p)
    canon = minimum(images)
    return canon
end

end # module
