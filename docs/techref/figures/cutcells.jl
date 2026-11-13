# Chapter 7 (cut cells) and Section 4.2 (phase fractions in cut cells): figure
# data and spot checks. Included by generate.jl after interface.jl, whose
# independent polygon helpers it uses. Solid planes, apertures and open
# volumes are read from a GeometryField; the "independent" functions evaluate
# the formulas of Chapter 7 directly from the stored solid planes.

const CYLINDER = (; cells = 8, centre = (4.3, 3.6), radius = 2.6)

function cylinder_fixture()
    n = CYLINDER.cells
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (n, n), extent = (Float64(n), Float64(n)),
        periodic = (false, false),
    )
    levelset(x, y) = hypot(x - CYLINDER.centre[1], y - CYLINDER.centre[2]) - CYLINDER.radius
    return GeometryField(mesh, levelset)
end

solid_plane(geometry, ref) = (
    AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref),
    AMR._get(geometry.solid.intercept, ref),
)
solid_fraction(geometry, ref) = AMR._get(geometry.solid.fraction, ref)

# Local tangential interval of face f in the coordinates of cell `ref`
# (fixtures are not periodic, so no image shift is needed).
function local_interval(mesh, ref, face)
    t = 3 - face.axis
    centre = cellcenter(mesh, ref.block, ref.i, ref.j)
    h = mesh.blocks[ref.block].spacing[t]
    lo = (face.center[t] - face.area / 2 - centre[t]) / h
    return lo, lo + face.area / h
end

# Open part of the local face segment {x_k = side, x_t in [lo, hi]} according
# to the solid half-plane of the cell: {m . x >= a}; returns the open interval.
function open_interval(m, a, k, side, lo, hi)
    t = 3 - k
    threshold = a - m[k] * side
    if m[t] == 0
        return threshold <= 0 ? (lo, hi) : (lo, lo)
    end
    crossing = threshold / m[t]
    m[t] > 0 && return (clamp(crossing, lo, hi), hi)
    return (lo, clamp(crossing, lo, hi))
end

function trace_independent(geometry, ref, face, side)
    alpha = solid_fraction(geometry, ref)
    lo, hi = local_interval(geometry.mesh, ref, face)
    alpha <= 0 && return 1.0, (lo, hi)
    alpha >= 1 && return 0.0, (lo, lo)
    m1, m2, a = solid_plane(geometry, ref)
    interval = open_interval((m1, m2), a, face.axis, side, lo, hi)
    return (interval[2] - interval[1]) / (hi - lo), interval
end

function aperture_independent(geometry, face)
    any(ref -> solid_fraction(geometry, ref) >= 1, (face.minus, face.plus)) && return 0.0
    minus, _ = trace_independent(geometry, face.minus, face, 0.5)
    plus, _ = trace_independent(geometry, face.plus, face, -0.5)
    return clamp((minus + plus) / 2, 0.0, 1.0)
end

# Open area of the strip of `ref` spanning the cell normal to f and the
# tangential extent of f, in local units (independent clip).
function strip_open_area(geometry, ref, face)
    lo, hi = local_interval(geometry.mesh, ref, face)
    strip = face.axis == 1 ?
        NTuple{2, Float64}[(-0.5, lo), (0.5, lo), (0.5, hi), (-0.5, hi)] :
        NTuple{2, Float64}[(lo, -0.5), (hi, -0.5), (hi, 0.5), (lo, 0.5)]
    alpha = solid_fraction(geometry, ref)
    alpha <= 0 && return shoelace(strip), strip
    alpha >= 1 && return 0.0, NTuple{2, Float64}[]
    m1, m2, a = solid_plane(geometry, ref)
    open = clip_halfplane(strip, (m1, m2), a; keep_less = false)
    return shoelace(open), open
end

function aperture_check(geometry)
    mesh = geometry.mesh
    worst_aperture = worst_boundary = worst_dual = worst_open = 0.0
    cut_faces = 0
    for (index, face) in pairs(mesh.faces)
        a = aperture_independent(geometry, face)
        worst_aperture = max(worst_aperture, abs(a - geometry.aperture[index]))
        0 < a < 1 && (cut_faces += 1)
        volume = sum(
            0.5 * strip_open_area(geometry, ref, face)[1] * mesh.blocks[ref.block].volume
                for ref in (face.minus, face.plus)
        )
        worst_dual = max(worst_dual, abs(volume - geometry.face_dual_volumes[index]))
    end
    for (index, face) in pairs(mesh.boundary_faces)
        a, _ = trace_independent(geometry, face.cell, face, face.side / 2)
        worst_boundary = max(worst_boundary, abs(a - geometry.boundary_aperture[index]))
    end
    cut_cells = 0
    for j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
        ref = CellRef(1, i, j)
        alpha = solid_fraction(geometry, ref)
        0 < alpha < 1 || continue
        cut_cells += 1
        m1, m2, a = solid_plane(geometry, ref)
        open = shoelace(clip_halfplane(SQUARE, (m1, m2), a; keep_less = false))
        worst_open = max(worst_open, abs(open - AMR._get(geometry.open_fraction, ref)))
    end
    @assert worst_aperture <= 1.0e-15 && worst_boundary <= 1.0e-15
    @assert worst_dual <= 1.0e-14 && worst_open <= 1.0e-14
    return (;
        worst_aperture, worst_boundary, worst_dual, worst_open, cut_faces, cut_cells,
        faces = length(mesh.faces),
    )
end

# --- Figure: cut cells, traces, shared apertures and open staggered volume ----

const CUT_WINDOW = (5, 8, 4, 7)          # i range, j range
const CUT_DUAL_FACE = ((6, 6), (7, 6))   # minus and plus cell of the shown face

function cutcell_data(geometry)
    io = IOBuffer()
    mesh = geometry.mesh
    println(
        io, "\\gdef\\fxCylX{$(fmt(CYLINDER.centre[1]))}\\gdef\\fxCylY{$(fmt(CYLINDER.centre[2]))}",
        "\\gdef\\fxCylR{$(fmt(CYLINDER.radius))}"
    )
    ilo, ihi, jlo, jhi = CUT_WINDOW
    inside(ref) = ilo <= ref.i <= ihi && jlo <= ref.j <= jhi
    for j in jlo:jhi, i in ilo:ihi
        ref = CellRef(1, i, j)
        alpha = solid_fraction(geometry, ref)
        open = AMR._get(geometry.open_fraction, ref)
        kind = alpha <= 0 ? 0 : alpha >= 1 ? 2 : 1
        println(io, "\\fxccell{$(fmt(i - 1.0))}{$(fmt(j - 1.0))}{$(@sprintf("%.2f", open))}{$kind}")
        kind == 1 || continue
        m1, m2, a = solid_plane(geometry, ref)
        solid = AMR._clip_polygon(SQUARE, m1, m2, a; keep_less = true)
        coords = join(("($(fmt(i - 0.5 + p[1])),$(fmt(j - 0.5 + p[2])))" for p in solid), " -- ")
        println(io, "\\fxcsolid{$coords -- cycle}")
        p, q = segment_in(SQUARE, (m1, m2), a)
        println(
            io, "\\fxcline{$(fmt(i - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}",
            "{$(fmt(i - 0.5 + q[1]))}{$(fmt(j - 0.5 + q[2]))}"
        )
    end
    # Faces with both cells in the window and at least one cut cell: the open
    # intervals seen from either side and the shared aperture.
    for (index, face) in pairs(mesh.faces)
        (inside(face.minus) && inside(face.plus)) || continue
        cut = any(ref -> 0 < solid_fraction(geometry, ref) < 1, (face.minus, face.plus))
        cut || continue
        t = 3 - face.axis
        for (ref, side, label) in ((face.minus, 0.5, -1), (face.plus, -0.5, 1))
            _, (lo, hi) = trace_independent(geometry, ref, face, side)
            centre = cellcenter(mesh, ref.block, ref.i, ref.j)
            # interval in physical coordinates along the face
            a, b = centre[t] + lo, centre[t] + hi
            println(
                io, "\\fxctrace{$(face.axis)}{$(fmt(face.center[face.axis]))}",
                "{$(fmt(a))}{$(fmt(b))}{$label}"
            )
        end
        println(
            io, "\\fxcaperture{$(face.axis)}{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}",
            "{$(@sprintf("%.2f", geometry.aperture[index]))}"
        )
    end
    # Panel (b): open staggered volume of one face.
    (im, jm), (ip, jp) = CUT_DUAL_FACE
    minus, plus = CellRef(1, im, jm), CellRef(1, ip, jp)
    index = findfirst(f -> f.minus == minus && f.plus == plus, mesh.faces)
    face = mesh.faces[index]
    for ref in (minus, plus)
        _, strip = strip_open_area(geometry, ref, face)
        centre = cellcenter(mesh, ref.block, ref.i, ref.j)
        coords = join(("($(fmt(centre[1] + p[1])),$(fmt(centre[2] + p[2])))" for p in strip), " -- ")
        println(io, "\\fxcstrip{$coords -- cycle}")
    end
    println(
        io, "\\gdef\\fxCdual{$(@sprintf("%.3f", geometry.face_dual_volumes[index]))}",
        "\\gdef\\fxCdualfull{$(fmt(mesh.face_dual_volumes[index]))}",
        "\\gdef\\fxCaperture{$(@sprintf("%.2f", geometry.aperture[index]))}",
        "\\gdef\\fxCfx{$(fmt(face.center[1]))}\\gdef\\fxCfy{$(fmt(face.center[2]))}"
    )
    return String(take!(io)), (;
            dual = geometry.face_dual_volumes[index],
            aperture = geometry.aperture[index], aperture_dual = geometry.aperture[index] * mesh.face_dual_volumes[index],
        )
end

# --- Phase in cut cells: open plane constant, phase apertures, contact --------

# Solid below the line y = offset + slope (x - 8); a liquid drop of radius
# `radius` meets it at the angle `theta` (measured in the liquid).
const WALL = (; slope = 0.2, offset = 1.7, radius = 4.2, theta = 60.0, contact_x = 11.3)

function triple_fixture(;
        embedded_contact_angle = deg2rad(WALL.theta), method = :lhf, cells = 16,
        slope = WALL.slope, contact_x = WALL.contact_x,
    )
    mesh = BlockTreeMesh(;
        base_mesh = (cells, cells ÷ 2), block_size = (16, 8), extent = (16.0, 8.0),
        periodic = (false, false),
    )
    scale = hypot(1, slope)
    wall(x, y) = (y - WALL.offset - slope * (x - 8)) / scale
    geometry = GeometryField(mesh, wall)
    # Centre of the drop: the circle through the contact point whose outward
    # normal there makes the angle WALL.theta with the wall normal.
    nw = (-slope, 1.0) ./ scale                          # into the fluid
    tw = (1.0, slope) ./ scale                           # along the wall, to the right
    theta = deg2rad(WALL.theta)
    p = (contact_x, WALL.offset + slope * (contact_x - 8))
    eta = cos(theta) .* nw .+ sin(theta) .* tw           # outward liquid normal at p
    centre = p .- WALL.radius .* eta
    drop(x, y) = hypot(x - centre[1], y - centre[2]) - WALL.radius
    field = InterfaceField(
        mesh, drop;
        geometry, embedded_contact_angle, contact_angle = nothing,
        reconstruction_method = method,
    )
    return geometry, field, (; centre, contact = p, nw, tw, eta)
end

# Accuracy at embedded triple cells (cells cut by the solid and by the
# interface): largest deviation of the PLIC normal from the exact normal at
# the PLIC centroid, and of the curvature from 1/R.
function triple_accuracy(method, cells)
    geometry, field, info = triple_fixture(; method, cells)
    mesh = field.mesh
    normal_error = curvature_error = 0.0
    triple = negative = reversed = undefined = 0
    for (b, block) in pairs(mesh.blocks), j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
        ref = CellRef(b, i, j)
        c = AMR._get(field.conditional_fraction, ref)
        (0 < c < 1 && 0 < solid_fraction(geometry, ref) < 1) || continue
        mx, my, a = plane_of_ref(field, ref)
        h = block.spacing
        n = (mx / h[1], my / h[2]) ./ hypot(mx / h[1], my / h[2])
        o = AMR._interface_centroid(mx, my, a)
        x = cellcenter(mesh, b, i, j) .+ o .* h
        exact = (x .- info.centre) ./ hypot((x .- info.centre)...)
        kappa = AMR._get(field.curvature, ref)
        normal_error = max(normal_error, hypot((n .- exact)...))
        reversed += n[1] * exact[1] + n[2] * exact[2] <= 0
        triple += 1
        if isfinite(kappa)
            curvature_error = max(curvature_error, abs(kappa - 1 / WALL.radius))
            negative += kappa < 0
        else
            undefined += 1
        end
    end
    return (; normal_error, curvature_error, triple, negative, reversed, undefined)
end

# Liquid fraction of the local face segment in one cell: open by the solid
# plane first, then below the liquid plane (independent).
function liquid_trace_independent(field, geometry, ref, face, side)
    c = AMR._get(field.conditional_fraction, ref)
    c <= 0 && return 0.0
    open, (lo, hi) = trace_independent(geometry, ref, face, side)
    full = local_interval(field.mesh, ref, face)
    width = full[2] - full[1]
    open <= 0 && return 0.0
    c >= 1 && return (hi - lo) / width
    m1, m2, a = plane_of_ref(field, ref)
    # liquid where m . x <= a on the segment x_k = side
    k = face.axis; t = 3 - k
    threshold = a - (k == 1 ? m1 : m2) * side
    mt = k == 1 ? m2 : m1
    if mt == 0
        return threshold >= 0 ? (hi - lo) / width : 0.0
    end
    crossing = threshold / mt
    l, h = mt > 0 ? (lo, clamp(crossing, lo, hi)) : (clamp(crossing, lo, hi), hi)
    return max(h - l, 0.0) / width
end

plane_of_ref(field, ref) = (
    AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref), AMR._get(field.intercept, ref),
)

# Height values of a cut fixture: cut and covered cells are unavailable.
function open_fraction_matrix(field, geometry)
    n1, n2 = field.mesh.block_size
    return [
        AMR._get(geometry.open_fraction, CellRef(1, i, j)) >= 1 - 64eps() ?
            AMR._get(field.conditional_fraction, CellRef(1, i, j)) : NaN
            for i in 1:n1, j in 1:n2
    ]
end

previous_plane(field, ref) = (
    AMR._get(field.previous_normal_x, ref), AMR._get(field.previous_normal_y, ref),
    AMR._get(field.previous_intercept, ref),
)

# GHF fit of a triple cell (Sections 4.5 and 4.6), independent: height points
# of the 7x7 block in the frame of the plane of the last pass, the contact row
# if `theta` is given and the two planes meet in the cell, then the centroid
# rings of the previous-pass planes; oriented by the components along the
# wall tangent. Unit cells of a single block; (; normal, kappa, contact).
function triple_fit_independent(field, geometry, V, i, j; theta = nothing)
    ref = CellRef(1, i, j)
    mx, my, a = previous_plane(field, ref)
    n = (mx, my) ./ hypot(mx, my)
    t = (-n[2], n[1])
    o = segment_midpoint(mx, my, a)
    s1, s2, as = solid_plane(geometry, ref)
    nw = (s1, s2) ./ hypot(s1, s2)                              # wall normal into the fluid
    contact = nothing
    if !isnothing(theta)
        det = mx * s2 - my * s1
        if abs(det) > 256eps() * hypot(mx, my) * hypot(s1, s2)
            xs = ((a * s2 - my * as) / det, (mx * as - a * s1) / det)   # triple point x*
            if all(abs.(xs) .<= 0.5 + 256eps())
                tau = t[1] * (xs[1] - o[1]) + t[2] * (xs[2] - o[2])
                tn = t[1] * nw[1] + t[2] * nw[2]
                contact = 1.0e3 .* (-2tau * tn, -tn, 0.0, cos(theta) - (n[1] * nw[1] + n[2] * nw[2]))
            end
        end
    end
    function fit(points)
        length(points) + !isnothing(contact) >= 3 || return nothing
        function row(p)
            d = (p[1] - o[1], p[2] - o[2])
            ta = t[1] * d[1] + t[2] * d[2]
            return sqrt(p[3]) .* (ta^2, ta, 1.0, n[1] * d[1] + n[2] * d[2])
        end
        rows = NTuple{4, Float64}[row(p) for p in points]
        isnothing(contact) || push!(rows, contact)
        M = [r[k] for r in rows, k in 1:3]
        b = [r[4] for r in rows]
        sv = svdvals(M)
        minimum(sv) > 1.0e-10 * maximum(sv) || return nothing
        A2, A1, _ = M \ b
        scale = sqrt(1 + A1^2)
        normal = ((n[1] - A1 * t[1]) / scale, (n[2] - A1 * t[2]) / scale)
        kappa = -2A2 / scale^3
        # Orientation in a cut cell: compare the wall-tangential components.
        tw = (-nw[2], nw[1])
        seed_t = n[1] * tw[1] + n[2] * tw[2]
        fit_t = normal[1] * tw[1] + normal[2] * tw[2]
        alignment = normal[1] * n[1] + normal[2] * n[2]
        if abs(seed_t) > sqrt(eps()) && abs(fit_t) > sqrt(eps())
            alignment = seed_t * fit_t
        end
        alignment >= 0 && return (; normal, kappa, contact = !isnothing(contact))
        return (; normal = .-normal, kappa = -kappa, contact = !isnothing(contact))
    end
    points, _ = ghf_points(V, i, j, n; radius = 3)
    result = fit(points)
    isnothing(result) || return result
    for radius in 1:3
        centroids = NTuple{5, Float64}[]
        for dj in -radius:radius, di in -radius:radius
            checkbounds(Bool, V, i + di, j + dj) || continue
            c = V[i + di, j + dj]
            (isfinite(c) && 0 < c < 1) || continue
            q = segment_midpoint(previous_plane(field, CellRef(1, i + di, j + dj))...)
            p = (di + q[1], dj + q[2])
            any(r -> hypot(r[1] - p[1], r[2] - p[2]) <= 0.5, centroids) && continue
            push!(centroids, (p[1], p[2], 1.0, 0.0, 0.0))
        end
        result = fit(centroids)
        isnothing(result) || return result
    end
    return nothing
end

# Masked WY normal (Section 4.4), independent: conditional fractions L/O,
# covered cells (O = 0) excluded; local coordinates (zero gradient beyond the
# block, as the halo).
function masked_wy_independent(L, O, i, j)
    n1, n2 = size(L)
    free(ii, jj) = O[clamp(ii, 1, n1), clamp(jj, 1, n2)] > 0
    C(ii, jj) = L[clamp(ii, 1, n1), clamp(jj, 1, n2)] / O[clamp(ii, 1, n1), clamp(jj, 1, n2)]
    centre = C(i, j)
    function difference(di, dj)
        lower, upper = free(i - di, j - dj), free(i + di, j + dj)
        lower && upper && return C(i - di, j - dj) - C(i + di, j + dj)
        lower && return 2 * (C(i - di, j - dj) - centre)
        upper && return 2 * (centre - C(i + di, j + dj))
        return 0.0
    end
    gx, gy = difference(1, 0), difference(0, 1)
    axis = abs(gx) > abs(gy) ? 1 : 2
    p, t = axis == 1 ? (gx, gy) : (gy, gx)
    s = p < 0 ? -1.0 : 1.0
    d = (t < 0 ? -1 : 1) * (centre > 0.5 ? 1 : -1)
    for side in (d, -d)
        delta = 0.0; pairs = 0
        for k in -1:1
            a = axis == 1 ? (i + k, j) : (i, j + k)
            b = axis == 1 ? (a[1], a[2] + side) : (a[1] + side, a[2])
            free(a...) && free(b...) || continue
            delta += C(b...) - C(a...); pairs += 1
        end
        if pairs > 0
            transverse = -delta * 3 / pairs / side
            return axis == 1 ? (s, transverse) : (transverse, s)
        end
    end
    transverse = p == 0 ? 0.0 : t / abs(p)
    return axis == 1 ? (s, transverse) : (transverse, s)
end

# Plane constant of the line m.x = a in the open polygon of a cut cell for the
# conditional fraction c (bisection on the clipped area; independent).
function open_intercept_independent(geometry, ref, c, m)
    s1, s2, as = solid_plane(geometry, ref)
    open = clip_halfplane(SQUARE, (s1, s2), as; keep_less = false)
    target = c * shoelace(open)
    values = [m[1] * p[1] + m[2] * p[2] for p in open]
    lo, hi = minimum(values), maximum(values)
    for _ in 1:100
        mid = (lo + hi) / 2
        shoelace(clip_halfplane(open, m, mid)) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# Plane rule of a triple cell (Section 4.4), independent: the unconstrained
# fit unless it opposes the masked WY seed (code 2); else the nearest aligned
# LHF normal of a fully open mixed cell in the 5x5 block (code 1); else the
# seed (code 3). Returns the local normal as stored and the code.
function triple_plane_independent(field, geometry, V, L, O, i, j)
    seed = masked_wy_independent(L, O, i, j)
    plane = triple_fit_independent(field, geometry, V, i, j)
    if !isnothing(plane) && plane.normal[1] * seed[1] + plane.normal[2] * seed[2] > 0
        return plane.normal ./ (abs(plane.normal[1]) + abs(plane.normal[2])), 2
    end
    ref = CellRef(1, i, j)
    c = AMR._get(field.conditional_fraction, ref)
    target = segment_midpoint(seed..., open_intercept_independent(geometry, ref, c, seed))
    reach = sqrt(0.5) * hypot(seed...)
    best = nothing; best_distance = Inf
    for dj in -2:2, di in -2:2
        di == 0 && dj == 0 && continue
        checkbounds(Bool, V, i + di, j + dj) || continue
        v = V[i + di, j + dj]
        (isfinite(v) && 0 < v < 1) || continue
        lhf = lhf_independent(V, i + di, j + dj)
        isnothing(lhf) && continue
        nl = (lhf[1], lhf[2]) ./ hypot(lhf[1], lhf[2])
        nl[1] * seed[1] + nl[2] * seed[2] >= reach || continue
        q = segment_midpoint(nl..., plane_constant_exact(v, nl...))
        distance = (di + q[1] - target[1])^2 + (dj + q[2] - target[2])^2
        distance < best_distance || continue
        best, best_distance = nl, distance
    end
    isnothing(best) || return best ./ (abs(best[1]) + abs(best[2])), 1
    return seed, 3
end

# Triple cells of the sloped-wall fixtures (16 x 8 unit cells; four slopes,
# four contact points, every method, prescribed angles 60, 90, 120 degrees):
# independent plane rule and curvature, angle independence of the planes and
# orientation against the exact normal.
function triple_check()
    worst_plane = worst_curvature = 0.0
    cells = constrained = reversed = mismatched = angle_dependent = configurations = 0
    codes = Dict(1 => 0, 2 => 0, 3 => 0)
    for slope in (0.0, 0.2, -0.3, 0.5), contact_x in (9.7, 10.9, 11.3, 12.1),
            method in (:lhf, :lhf_ghf, :wy)
        reference = nothing
        for angle in (60.0, 90.0, 120.0)
            theta = deg2rad(angle)
            geometry, field, info = triple_fixture(;
                method, slope, contact_x, embedded_contact_angle = theta,
            )
            configurations += 1
            n1, n2 = field.mesh.block_size
            V = open_fraction_matrix(field, geometry)
            L = [AMR._get(field.fraction, CellRef(1, i, j)) for i in 1:n1, j in 1:n2]
            O = [AMR._get(geometry.open_fraction, CellRef(1, i, j)) for i in 1:n1, j in 1:n2]
            planes = Dict{Tuple{Int, Int}, NTuple{3, Float64}}()
            for j in 1:n2, i in 1:n1
                ref = CellRef(1, i, j)
                c = AMR._get(field.conditional_fraction, ref)
                (0 < c < 1 && 0 < solid_fraction(geometry, ref) < 1) || continue
                mx, my, a = plane_of_ref(field, ref)
                planes[(i, j)] = (mx, my, a)
                expected, code = triple_plane_independent(field, geometry, V, L, O, i, j)
                stored = Int(AMR._get(field.reconstruction, ref))
                mismatched += stored != code
                worst_plane = max(worst_plane, abs(mx - expected[1]), abs(my - expected[2]))
                fit = triple_fit_independent(field, geometry, V, i, j; theta)
                if isnothing(fit)
                    fit = triple_fit_independent(field, geometry, V, i, j)
                else
                    constrained += angle == 60.0 && fit.contact
                end
                kappa = isnothing(fit) ? NaN : clamp(fit.kappa, -0.5, 0.5)
                actual = AMR._get(field.curvature, ref)
                worst_curvature = isequal(kappa, actual) ? worst_curvature :
                    max(worst_curvature, abs(kappa - actual))
                n = (mx, my) ./ hypot(mx, my)
                q = segment_midpoint(mx, my, a)
                x = (i - 0.5 + q[1], j - 0.5 + q[2])
                exact = (x .- info.centre) ./ hypot((x .- info.centre)...)
                reversed += n[1] * exact[1] + n[2] * exact[2] <= 0
                if angle == 60.0
                    cells += 1
                    codes[stored] = get(codes, stored, 0) + 1
                end
            end
            isnothing(reference) && (reference = planes)
            angle_dependent += !isequal(planes, reference)
        end
    end
    println(
        "triple check: ", (;
            worst_plane, worst_curvature, cells, constrained, reversed, mismatched,
            angle_dependent, configurations, codes,
        )
    )
    @assert mismatched == 0 && worst_plane <= 1.0e-9 && worst_curvature <= 1.0e-9
    @assert reversed == 0 && angle_dependent == 0 && constrained >= 2
    return (;
        worst_plane, worst_curvature, cells, constrained, configurations,
        ghf = codes[2], lhf = codes[1], wy = codes[3],
    )
end

function phase_check()
    geometry, field, _ = triple_fixture()
    mesh = field.mesh
    worst_intercept = 0.0; triple_cells = 0
    for j in 1:mesh.block_size[2], i in 1:mesh.block_size[1]
        ref = CellRef(1, i, j)
        c = AMR._get(field.conditional_fraction, ref)
        alpha = solid_fraction(geometry, ref)
        (0 < c < 1 && 0 < alpha < 1) || continue
        triple_cells += 1
        m1, m2, a_s = solid_plane(geometry, ref)
        open = clip_halfplane(SQUARE, (m1, m2), a_s; keep_less = false)
        n1, n2, a = plane_of_ref(field, ref)
        liquid = clip_halfplane(open, (n1, n2), a; keep_less = true)
        worst_intercept = max(worst_intercept, abs(shoelace(liquid) - c * shoelace(open)))
    end
    worst_aperture = 0.0; worst_partition = 0.0; mixed_faces = 0; rule_faces = 0
    for (index, face) in pairs(mesh.faces)
        open = geometry.aperture[index]
        liquid, gas = AMR._phase_face_apertures(field, index)
        worst_partition = max(worst_partition, abs(liquid + gas - open))
        relevant = [ref for ref in (face.minus, face.plus) if AMR._get(geometry.open_fraction, ref) > 0]
        liquid_possible = any(ref -> AMR._get(field.conditional_fraction, ref) > 0, relevant)
        gas_possible = any(ref -> AMR._get(field.conditional_fraction, ref) < 1, relevant)
        mine = if !liquid_possible
            0.0
        elseif !gas_possible
            open
        else
            mixed_faces += 1
            minus_trace = liquid_trace_independent(field, geometry, face.minus, face, 0.5)
            plus_trace = liquid_trace_independent(field, geometry, face.plus, face, -0.5)
            minus_fraction = AMR._get(field.conditional_fraction, face.minus)
            plus_fraction = AMR._get(field.conditional_fraction, face.plus)
            minus_pure = iszero(minus_fraction) || isone(minus_fraction)
            plus_pure = iszero(plus_fraction) || isone(plus_fraction)
            # The pure-neighbour rule: exactly one pure cell and a strictly
            # partial trace of the other cell take the pure cell's value.
            trace = minus_pure == plus_pure ? nothing :
                minus_pure ? plus_trace : minus_trace
            if !isnothing(trace) && 0 < trace < 1
                rule_faces += 1
                clamp(minus_pure ? minus_fraction : plus_fraction, 0.0, open)
            else
                clamp((minus_trace + plus_trace) / 2, 0.0, open)
            end
        end
        worst_aperture = max(worst_aperture, abs(mine - liquid))
    end
    @assert worst_intercept <= 1.0e-14 && worst_aperture <= 1.0e-15 && worst_partition <= 1.0e-15
    return (;
        worst_intercept, triple_cells, worst_aperture, worst_partition, mixed_faces,
        rule_faces,
    )
end

# --- Figure: a phase interface in cut cells and the embedded contact angle ----

const TRIPLE_WINDOW = (10, 13, 2, 4)   # i range, j range (unit cells)

function triple_data()
    io = IOBuffer()
    geometry, field, info = triple_fixture()
    ilo, ihi, jlo, jhi = TRIPLE_WINDOW
    contact_cell = nothing
    for j in jlo:jhi, i in ilo:ihi
        ref = CellRef(1, i, j)
        alpha_s = solid_fraction(geometry, ref)
        open = AMR._get(geometry.open_fraction, ref)
        c = AMR._get(field.conditional_fraction, ref)
        liquid = AMR._get(field.fraction, ref)
        kind = alpha_s >= 1 ? 2 : alpha_s > 0 ? 1 : 0
        # \fxtcell{x0}{y0}{open fraction}{liquid fraction}{conditional}{kind}
        println(
            io, "\\fxtcell{$(fmt(i - 1.0))}{$(fmt(j - 1.0))}{$(@sprintf("%.2f", open))}",
            "{$(@sprintf("%.2f", liquid))}{$(@sprintf("%.2f", c))}{$kind}"
        )
        kind == 2 && continue
        local_open = SQUARE
        if kind == 1
            s1, s2, as = solid_plane(geometry, ref)
            solid = clip_halfplane(SQUARE, (s1, s2), as)
            local_open = clip_halfplane(SQUARE, (s1, s2), as; keep_less = false)
            println(io, "\\fxtsolid{", join(("($(fmt(i - 0.5 + p[1])),$(fmt(j - 0.5 + p[2])))" for p in solid), " -- "), " -- cycle}")
            p, q = segment_in(SQUARE, (s1, s2), as)
            println(io, "\\fxtwall{$(fmt(i - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}{$(fmt(i - 0.5 + q[1]))}{$(fmt(j - 0.5 + q[2]))}")
        end
        (0 < c) || continue
        if c >= 1
            println(io, "\\fxtliquid{", join(("($(fmt(i - 0.5 + p[1])),$(fmt(j - 0.5 + p[2])))" for p in local_open), " -- "), " -- cycle}")
            continue
        end
        m1, m2, a = plane_of_ref(field, ref)
        liquid_poly = clip_halfplane(local_open, (m1, m2), a)
        @assert isapprox(shoelace(liquid_poly), c * shoelace(local_open); atol = 1.0e-13)
        println(io, "\\fxtliquid{", join(("($(fmt(i - 0.5 + p[1])),$(fmt(j - 0.5 + p[2])))" for p in liquid_poly), " -- "), " -- cycle}")
        p, q = segment_in(local_open, (m1, m2), a)
        println(io, "\\fxtplic{$(fmt(i - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}{$(fmt(i - 0.5 + q[1]))}{$(fmt(j - 0.5 + q[2]))}")
        if kind == 1
            s1, s2, as = solid_plane(geometry, ref)
            det = m1 * s2 - m2 * s1
            xs = ((a * s2 - m2 * as) / det, (m1 * as - a * s1) / det)
            if all(abs.(xs) .<= 0.5)
                @assert isnothing(contact_cell)
                n = (m1, m2) ./ hypot(m1, m2)
                nw = (s1, s2) ./ hypot(s1, s2)
                contact_cell = (; i, j, xs = (i - 0.5 + xs[1], j - 0.5 + xs[2]), n, nw, c, open, liquid)
            end
        end
    end
    cc = contact_cell
    println(
        io, "\\gdef\\fxTx{$(fmt(cc.xs[1]))}\\gdef\\fxTy{$(fmt(cc.xs[2]))}",
        "\\gdef\\fxTnx{$(fmt(cc.n[1]))}\\gdef\\fxTny{$(fmt(cc.n[2]))}",
        "\\gdef\\fxTwx{$(fmt(cc.nw[1]))}\\gdef\\fxTwy{$(fmt(cc.nw[2]))}",
        "\\gdef\\fxTi{$(cc.i)}\\gdef\\fxTj{$(cc.j)}\\gdef\\fxTtheta{$(fmt(WALL.theta))}",
        "\\gdef\\fxTC{$(@sprintf("%.2f", cc.c))}\\gdef\\fxTopen{$(@sprintf("%.2f", cc.open))}",
        "\\gdef\\fxTliquid{$(@sprintf("%.2f", cc.liquid))}"
    )
    # Exact interface and wall for reference.
    println(
        io, "\\gdef\\fxTcx{$(fmt(info.centre[1]))}\\gdef\\fxTcy{$(fmt(info.centre[2]))}\\gdef\\fxTR{$(fmt(WALL.radius))}",
        "\\gdef\\fxTslope{$(fmt(WALL.slope))}\\gdef\\fxToffset{$(fmt(WALL.offset))}"
    )
    # Angle between the PLIC normal and the wall normal at the triple point.
    angle = acos(clamp(cc.n[1] * cc.nw[1] + cc.n[2] * cc.nw[2], -1.0, 1.0))
    return String(take!(io)), (; cell = (cc.i, cc.j), plic_angle = rad2deg(angle))
end
