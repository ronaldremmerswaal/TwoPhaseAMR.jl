# Chapter 4 (interface reconstruction): figure data and spot checks.
#
# Included by generate.jl. Every drawn fraction, polygon, plane, height, fit
# point and curvature is read from an InterfaceField. The functions marked
# "independent" evaluate the formulas as printed in Chapter 4 without calling
# the package's reconstruction internals; the differences to the code are the
# numbers quoted in the text, and the assertions fail if they disagree.

const SQUARE = NTuple{2, Float64}[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]

# --- Independent polygon geometry ------------------------------------------

# Sutherland--Hodgman clip of a polygon against {n.x <= a} (or >= a).
function clip_halfplane(poly, n, a; keep_less = true)
    out = NTuple{2, Float64}[]
    m = length(poly)
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

function shoelace(poly)
    m = length(poly)
    m < 3 && return 0.0
    twice = sum(poly[k][1] * poly[mod1(k + 1, m)][2] - poly[mod1(k + 1, m)][1] * poly[k][2] for k in 1:m)
    return abs(twice) / 2
end

function centroid(poly)
    m = length(poly)
    twice = 0.0; cx = 0.0; cy = 0.0
    for k in 1:m
        p, q = poly[k], poly[mod1(k + 1, m)]
        cross = p[1] * q[2] - q[1] * p[2]
        twice += cross; cx += (p[1] + q[1]) * cross; cy += (p[2] + q[2]) * cross
    end
    return (cx / (3twice), cy / (3twice))
end

# End points of the line {n.x = a} inside a convex polygon (the PLIC segment).
function segment_in(poly, n, a)
    points = NTuple{2, Float64}[]
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
    @assert length(points) == 2
    return points[1], points[2]
end

# Scardovelli & Zaleski (2000): closed-form inverse of the area fraction of
# the unit square below the line, for the plane constant measured from the
# cell centre (independent of the package's bisection).
function plane_constant_exact(c, mx, my)
    s = abs(mx) + abs(my)
    m1, m2 = minmax(abs(mx), abs(my)) ./ s
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

smoothstep_quality(principal) = begin
    angle = acos(clamp(principal, 0.0, 1.0))
    t = clamp((pi / 4 + pi / 20 - angle) / (pi / 10), 0.0, 1.0)
    t * t * (3 - 2t)
end

# --- Fixtures ----------------------------------------------------------------

const CIRCLE = (; cells = 14, centre = (7.15, 6.35), radius = 4.3)

circle_levelset(x, y) = hypot(x - CIRCLE.centre[1], y - CIRCLE.centre[2]) - CIRCLE.radius

function circle_fixture(method = :lhf)
    n = CIRCLE.cells
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (n, n), extent = (Float64(n), Float64(n)),
        periodic = (false, false),
    )
    return InterfaceField(mesh, circle_levelset; reconstruction_method = method)
end

ref1(i, j) = CellRef(1, i, j)
cfrac(field, i, j) = AMR._get(field.conditional_fraction, ref1(i, j))
plane_of(field, i, j) = (
    AMR._get(field.normal_x, ref1(i, j)), AMR._get(field.normal_y, ref1(i, j)),
    AMR._get(field.intercept, ref1(i, j)),
)
method_of(field, i, j) = Int(AMR._get(field.reconstruction, ref1(i, j)))
curvature_of(field, i, j) = AMR._get(field.curvature, ref1(i, j))

# Conditional fractions of a single-block fixture (unit cells); cells outside
# the domain are absent, as the physical-boundary slots of the height values.
fraction_matrix(field) = [
    cfrac(field, i, j) for i in 1:field.mesh.block_size[1], j in 1:field.mesh.block_size[2]
]

# --- Independent height functions (Chapter 4, Section "Height functions") ----

# Column height H in cell units relative to the centre of cell (i, j), for the
# principal direction s*e_axis; `nothing` if the column is not valid.
function column_height(V, i, j, axis, s; radius = 3)
    tol = 64eps()
    total = 0.0; first = nothing; previous = 1.0
    for k in -radius:radius
        ii, jj = axis == 1 ? (i + k, j) : (i, j + k)
        available = checkbounds(Bool, V, ii, jj) && isfinite(V[ii, jj])
        if !available
            isnothing(first) && continue
            break
        end
        v = s > 0 ? V[ii, jj] : 1 - V[ii, jj]
        if isnothing(first)
            abs(v - 1) <= tol || return nothing
            first = k
        else
            v > previous + tol && return nothing
            tol < v < 1 - tol && abs(v - previous) <= tol && return nothing
        end
        total += v; previous = v
    end
    (isnothing(first) || abs(previous) > tol) && return nothing
    return total + first - 0.5
end

# Weighted LHF normal (physical, unit-cell mesh) and curvature, or nothing.
function lhf_independent(V, i, j)
    candidates = NTuple{4, Float64}[]
    for axis in 1:2, s in (-1, 1)
        H = [column_height(V, axis == 1 ? i : i + o, axis == 1 ? j + o : j, axis, s) for o in -1:1]
        any(isnothing, H) && continue
        slope = (H[3] - H[1]) / 2
        second = H[3] - 2H[2] + H[1]
        nu = sqrt(1 + slope^2)
        principal, tangential = s / nu, -s * slope / nu
        q = smoothstep_quality(abs(principal))
        q > 0 || continue
        n = axis == 1 ? (principal, tangential) : (tangential, principal)
        push!(candidates, (n[1], n[2], -s * second / nu^3, q))
    end
    isempty(candidates) && return nothing
    nx = ny = kappa = weight = 0.0
    for (k, c) in pairs(candidates)
        w = c[4] * prod((1 - d[4] for (l, d) in pairs(candidates) if l != k); init = 1.0)
        nx += w * c[1]; ny += w * c[2]; kappa += w * c[3]; weight += w
    end
    weight > 0 || return nothing
    return (nx, ny, kappa / weight, length(candidates))
end

function wy_independent(V, i, j)
    gx = V[i - 1, j] - V[i + 1, j]
    gy = V[i, j - 1] - V[i, j + 1]
    axis = abs(gx) > abs(gy) ? 1 : 2
    p, t = axis == 1 ? (gx, gy) : (gy, gx)
    s = p < 0 ? -1 : 1
    d = (t < 0 ? -1 : 1) * (V[i, j] > 0.5 ? 1 : -1)
    column(i0, j0) = axis == 1 ? sum(V[i0 + k, j0] for k in -1:1) : sum(V[i0, j0 + k] for k in -1:1)
    shifted = axis == 1 ? column(i, j + d) : column(i + d, j)
    transverse = -(shifted - column(i, j)) / d
    return axis == 1 ? (Float64(s), transverse) : (transverse, Float64(s))
end

# GHF point set of the 3x3 neighbourhood, in the order of the text.
function ghf_points(V, i, j, n)
    points = NTuple{5, Float64}[]  # x, y, weight, axis, s (relative to the cell centre)
    rejected = NTuple{5, Float64}[]
    for dj in -1:1, di in -1:1, axis in 1:2, s in (-1, 1)
        H = column_height(V, i + di, j + dj, axis, s)
        isnothing(H) && continue
        p = axis == 1 ? (di + H, Float64(dj)) : (Float64(di), dj + H)
        w = max(0.0, s * n[axis])^2
        entry = (p[1], p[2], w, Float64(axis), Float64(s))
        if w <= 0 || any(q -> hypot(q[1] - p[1], q[2] - p[2]) <= 0.5, points)
            push!(rejected, entry)
        else
            push!(points, entry)
        end
    end
    return points, rejected
end

# Weighted least-squares parabola x_eta = A2 x_tau^2 + A1 x_tau + A0 about the
# interface centroid o with seed normal n (unit-cell mesh, h = 1).
function ghf_fit_independent(points, n, o)
    length(points) >= 3 || return nothing
    t = (-n[2], n[1])
    rows = [
        begin
            d = (p[1] - o[1], p[2] - o[2])
            tau = t[1] * d[1] + t[2] * d[2]
            sqrt(p[3]) .* (tau^2, tau, 1.0, n[1] * d[1] + n[2] * d[2])
        end for p in points
    ]
    M = [r[k] for r in rows, k in 1:3]
    b = [r[4] for r in rows]
    sv = svdvals(M)
    minimum(sv) > 1.0e-10 * maximum(sv) || return nothing
    A2, A1, A0 = M \ b
    scale = sqrt(1 + A1^2)
    return (;
        normal = ((n[1] - A1 * t[1]) / scale, (n[2] - A1 * t[2]) / scale),
        kappa = -2A2 / scale^3, A2, A1, A0,
    )
end

# GHF with the centroid fallback: height points first; otherwise the PLIC
# segment midpoints of the mixed cells within rings of radius 1, 2, 3 (unit
# weights). Single-block unit-cell fixture; `field` supplies the stored planes.
function ghf_independent(field, V, i, j)
    mx, my, a = plane_of(field, i, j)
    seed = (mx, my) ./ hypot(mx, my)
    o = segment_midpoint(mx, my, a)
    points, _ = ghf_points(V, i, j, seed)
    fit = ghf_fit_independent(points, seed, o)
    isnothing(fit) || return (; fit..., fallback = 0)
    for radius in 1:3
        centroids = NTuple{5, Float64}[]
        for dj in -radius:radius, di in -radius:radius
            checkbounds(Bool, V, i + di, j + dj) || continue
            c = V[i + di, j + dj]
            (isfinite(c) && 0 < c < 1) || continue
            q = segment_midpoint(plane_of(field, i + di, j + dj)...)
            p = (di + q[1], dj + q[2])
            any(r -> hypot(r[1] - p[1], r[2] - p[2]) <= 0.5, centroids) && continue
            push!(centroids, (p[1], p[2], 1.0, 0.0, 0.0))
        end
        fit = ghf_fit_independent(centroids, seed, o)
        isnothing(fit) || return (; fit..., fallback = radius)
    end
    return nothing
end

function segment_midpoint(mx, my, a)
    p, q = segment_in(SQUARE, (mx, my), a)
    return ((p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
end

# --- Figure: PLIC reconstruction in a cell -----------------------------------

const PLIC_TARGET = (6, 11)
const PLIC_WINDOW = (4, 12, 8, 12)   # i range, j range of panel (a)

function plic_data(field)
    io = IOBuffer()
    mesh = field.mesh
    println(
        io, "\\gdef\\fxCircX{$(fmt(CIRCLE.centre[1]))}\\gdef\\fxCircY{$(fmt(CIRCLE.centre[2]))}",
        "\\gdef\\fxCircR{$(fmt(CIRCLE.radius))}"
    )
    ilo, ihi, jlo, jhi = PLIC_WINDOW
    for j in jlo:jhi, i in ilo:ihi
        c = cfrac(field, i, j)
        x0, y0 = i - 1.0, j - 1.0
        code = method_of(field, i, j)
        # \fxpcell{x0}{y0}{C (2 digits)}{pure 0/1 or mixed 2}{reconstruction code}
        kind = c == 0 ? 0 : c == 1 ? 1 : 2
        println(io, "\\fxpcell{$(fmt(x0))}{$(fmt(y0))}{$(@sprintf("%.2f", c))}{$kind}{$code}")
        kind == 2 || continue
        mx, my, a = plane_of(field, i, j)
        liquid = AMR._clip_polygon(SQUARE, mx, my, a; keep_less = true)
        @assert isapprox(AMR._polygon_area(liquid), c; atol = 1.0e-13)
        coords = join(("($(fmt(x0 + 0.5 + p[1])),$(fmt(y0 + 0.5 + p[2])))" for p in liquid), " -- ")
        println(io, "\\fxpliquid{$coords -- cycle}")
        p, q = segment_in(SQUARE, (mx, my), a)
        println(
            io, "\\fxpseg{$(fmt(x0 + 0.5 + p[1]))}{$(fmt(y0 + 0.5 + p[2]))}",
            "{$(fmt(x0 + 0.5 + q[1]))}{$(fmt(y0 + 0.5 + q[2]))}"
        )
    end
    # Panel (b): the target cell in local coordinates.
    i, j = PLIC_TARGET
    c = cfrac(field, i, j)
    mx, my, a = plane_of(field, i, j)
    @assert method_of(field, i, j) == 1 && abs(mx) + abs(my) ≈ 1
    liquid = AMR._clip_polygon(SQUARE, mx, my, a; keep_less = true)
    coords = join(("($(fmt(p[1])),$(fmt(p[2])))" for p in liquid), " -- ")
    p, q = segment_in(SQUARE, (mx, my), a)
    o = AMR._interface_centroid(mx, my, a)
    @assert hypot(o[1] - (p[1] + q[1]) / 2, o[2] - (p[2] + q[2]) / 2) < 1.0e-14
    norm2 = hypot(mx, my)
    foot = (a * mx / norm2^2, a * my / norm2^2)       # closest point of the line to the centre
    lc = centroid(liquid)
    println(
        io, "\\gdef\\fxZC{$(@sprintf("%.2f", c))}\\gdef\\fxZmx{$(fmt(mx))}\\gdef\\fxZmy{$(fmt(my))}",
        "\\gdef\\fxZa{$(fmt(a))}"
    )
    println(io, "\\gdef\\fxZnx{$(fmt(mx / norm2))}\\gdef\\fxZny{$(fmt(my / norm2))}")
    println(io, "\\gdef\\fxZliquid{$coords -- cycle}")
    println(io, "\\gdef\\fxZpx{$(fmt(p[1]))}\\gdef\\fxZpy{$(fmt(p[2]))}\\gdef\\fxZqx{$(fmt(q[1]))}\\gdef\\fxZqy{$(fmt(q[2]))}")
    println(io, "\\gdef\\fxZox{$(fmt(o[1]))}\\gdef\\fxZoy{$(fmt(o[2]))}")
    println(io, "\\gdef\\fxZfx{$(fmt(foot[1]))}\\gdef\\fxZfy{$(fmt(foot[2]))}")
    println(io, "\\gdef\\fxZlx{$(fmt(lc[1]))}\\gdef\\fxZly{$(fmt(lc[2]))}")
    println(io, "\\gdef\\fxZi{$i}\\gdef\\fxZj{$j}")
    return String(take!(io)), (; c, mx, my, a, distance = a / norm2)
end

# --- Figure: height-function stencil and GHF points ---------------------------

const LHF_TARGET = (9, 11)
const GHF_TARGET = (11, 10)

function heights_data(field)
    io = IOBuffer()
    V = fraction_matrix(field)
    n = size(V, 1)
    println(
        io, "\\gdef\\fxCircX{$(fmt(CIRCLE.centre[1]))}\\gdef\\fxCircY{$(fmt(CIRCLE.centre[2]))}",
        "\\gdef\\fxCircR{$(fmt(CIRCLE.radius))}"
    )
    # Panel (a): LHF columns of the target, principal direction +y.
    i, j = LHF_TARGET
    for jj in (j - 4):(j + 3), ii in (i - 2):(i + 2)
        (1 <= ii <= n && 1 <= jj <= n) || continue
        role = abs(ii - i) <= 1 && abs(jj - j) <= 3 ? (ii == i && jj == j ? 2 : 1) : 0
        println(io, "\\fxhcell{a}{$(fmt(ii - 1.0))}{$(fmt(jj - 1.0))}{$(@sprintf("%.2f", V[ii, jj]))}{$role}")
    end
    H = [column_height(V, i + o, j, 2, 1) for o in -1:1]
    @assert all(!isnothing, H)
    for (o, h) in zip(-1:1, H)
        println(io, "\\fxhpoint{$(fmt(i + o - 0.5))}{$(fmt(j - 0.5 + h))}")
    end
    lhf = lhf_independent(V, i, j)
    @assert lhf[4] == 1   # a single valid signed direction (+y) at this cell
    nn = hypot(lhf[1], lhf[2])
    println(
        io, "\\gdef\\fxHnx{$(fmt(lhf[1] / nn))}\\gdef\\fxHny{$(fmt(lhf[2] / nn))}",
        "\\gdef\\fxHi{$i}\\gdef\\fxHj{$j}"
    )
    # Panel (b): GHF points of the WY/GHF target.
    i, j = GHF_TARGET
    @assert isnothing(lhf_independent(V, i, j))
    for jj in (j - 2):(j + 2), ii in (i - 2):(i + 2)
        (1 <= ii <= n && 1 <= jj <= n) || continue
        role = abs(ii - i) <= 1 && abs(jj - j) <= 1 ? (ii == i && jj == j ? 2 : 1) : 0
        println(io, "\\fxhcell{b}{$(fmt(ii - 1.0))}{$(fmt(jj - 1.0))}{$(@sprintf("%.2f", V[ii, jj]))}{$role}")
    end
    mx, my, a = plane_of(field, i, j)
    seed = (mx, my) ./ hypot(mx, my)
    o = AMR._interface_centroid(mx, my, a)
    points, rejected = ghf_points(V, i, j, seed)
    for p in points
        println(io, "\\fxgpoint{$(fmt(i - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}{$(Int(p[4]))}{$(Int(p[5]))}{$(@sprintf("%.2f", p[3]))}{1}")
    end
    for p in rejected
        println(io, "\\fxgpoint{$(fmt(i - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}{$(Int(p[4]))}{$(Int(p[5]))}{$(@sprintf("%.2f", p[3]))}{0}")
    end
    fit = ghf_fit_independent(points, seed, o)
    t = (-seed[2], seed[1])
    curve = String[]
    for tau in range(-1.6, 1.6; length = 33)
        eta = fit.A2 * tau^2 + fit.A1 * tau + fit.A0
        x = i - 0.5 + o[1] + tau * t[1] + eta * seed[1]
        y = j - 0.5 + o[2] + tau * t[2] + eta * seed[2]
        push!(curve, "($(fmt(x)),$(fmt(y)))")
    end
    println(io, "\\gdef\\fxGcurve{$(join(curve, " -- "))}")
    println(
        io, "\\gdef\\fxGox{$(fmt(i - 0.5 + o[1]))}\\gdef\\fxGoy{$(fmt(j - 0.5 + o[2]))}",
        "\\gdef\\fxGnx{$(fmt(seed[1]))}\\gdef\\fxGny{$(fmt(seed[2]))}\\gdef\\fxGi{$i}\\gdef\\fxGj{$j}"
    )
    p, q = segment_in(SQUARE, (mx, my), a)
    println(io, "\\gdef\\fxGseg{($(fmt(i - 0.5 + p[1])),$(fmt(j - 0.5 + p[2]))) -- ($(fmt(i - 0.5 + q[1])),$(fmt(j - 0.5 + q[2])))}")
    return String(take!(io)), (;
            ghf_points = length(points), ghf_rejected = length(rejected),
            ghf_kappa = fit.kappa,
        )
end

# --- Spot checks: plane constant, LHF, WY and GHF against the code -----------

function plane_constant_check()
    # The plane constant is ill-conditioned near pure cells (da/dC ~ 1/sqrt(C)),
    # so it is compared with the closed form for 0.01 <= C <= 0.99 only; the
    # area residual A(a) - C is checked for every sample.
    worst_inverse = 0.0; worst_area = 0.0; samples = 0
    for scale in (1.0, 1.7), k in 0:63
        angle = 2pi * (k + 0.37) / 64
        mx, my = scale .* (cos(angle), sin(angle)) ./ (abs(cos(angle)) + abs(sin(angle)))
        for c in (1.0e-8, 1.0e-4, 0.013, 0.1, 0.27, 0.5, 0.71, 0.9, 0.987, 1 - 1.0e-4, 1 - 1.0e-8)
            a = AMR._plane_intercept(c, mx, my)
            if 0.01 <= c <= 0.99
                worst_inverse = max(worst_inverse, abs(a - plane_constant_exact(c, mx, my)))
            end
            worst_area = max(worst_area, abs(shoelace(clip_halfplane(SQUARE, (mx, my), a)) - c))
            samples += 1
        end
    end
    @assert worst_inverse <= 1.0e-14 && worst_area <= 1.0e-14
    return (; worst_inverse, worst_area, samples)
end

function reconstruction_check()
    lhf_field = circle_fixture(:lhf)
    wy_field = circle_fixture(:wy)
    ghf_field = circle_fixture(:lhf_ghf)
    V = fraction_matrix(lhf_field)
    n = size(V, 1)
    normal_lhf = curvature_lhf = normal_wy = curvature_ghf = 0.0
    counts = Dict{Int, Int}()
    exact_lhf = exact_ghf = 0.0
    fallbacks = 0
    for j in 2:(n - 1), i in 2:(n - 1)
        c = V[i, j]
        0 < c < 1 || continue
        code = method_of(lhf_field, i, j)
        counts[code] = get(counts, code, 0) + 1
        mx, my, _ = plane_of(lhf_field, i, j)
        if code == 1
            lhf = lhf_independent(V, i, j)
            m = (lhf[1], lhf[2]) ./ (abs(lhf[1]) + abs(lhf[2]))
            normal_lhf = max(normal_lhf, hypot(m[1] - mx, m[2] - my))
            curvature_lhf = max(curvature_lhf, abs(lhf[3] - curvature_of(lhf_field, i, j)))
            exact_lhf = max(exact_lhf, abs(curvature_of(lhf_field, i, j) - 1 / CIRCLE.radius))
        else
            # :lhf keeps the WY plane where no LHF exists; its curvature is the
            # GHF fit seeded with that plane.
            @assert code == 3
            ghf = ghf_independent(lhf_field, V, i, j)
            fallbacks += ghf.fallback > 0
            curvature_ghf = max(curvature_ghf, abs(ghf.kappa - curvature_of(lhf_field, i, j)))
            exact_ghf = max(exact_ghf, abs(curvature_of(lhf_field, i, j) - 1 / CIRCLE.radius))
        end
        wx, wy = plane_of(wy_field, i, j)
        wi = wy_independent(V, i, j)
        normal_wy = max(normal_wy, hypot(wi[1] - wx, wi[2] - wy))
    end
    ghf_cells = count(
        method_of(ghf_field, i, j) == 2 for j in 1:n, i in 1:n if 0 < V[i, j] < 1
    )
    @assert normal_lhf <= 1.0e-14 && curvature_lhf <= 1.0e-14
    @assert normal_wy == 0 && curvature_ghf <= 1.0e-12
    return (;
        normal_lhf, curvature_lhf, normal_wy, curvature_ghf, exact_lhf, exact_ghf,
        lhf_cells = get(counts, 1, 0), wy_cells = get(counts, 3, 0), ghf_cells, fallbacks,
        volume_error = abs(integrate(lhf_field.fraction) - pi * CIRCLE.radius^2),
    )
end

# --- Figure and checks: contact angle at a wall --------------------------------

const SESSILE = (; extent = (16.0, 8.0), radius = 6.0, centre_x = 8.2)

function sessile_fixture(theta)
    mesh = BlockTreeMesh(;
        base_mesh = (16, 8), block_size = (16, 8), extent = SESSILE.extent,
        periodic = (false, false),
    )
    cy = -SESSILE.radius * cos(theta)
    levelset(x, y) = hypot(x - SESSILE.centre_x, y - cy) - SESSILE.radius
    field = InterfaceField(mesh, levelset; contact_angle = (nothing, nothing, theta, nothing))
    return field, cy
end

# Height of the column above the wall: sum over the first four cells of the
# phase s, non-increasing, last cell empty (independent).
function wall_height(V, i, s)
    total = 0.0; previous = 1.0
    for k in 1:4
        checkbounds(Bool, V, i, k) || return nothing
        v = s > 0 ? V[i, k] : 1 - V[i, k]
        v <= previous + 64eps() || return nothing
        total += v; previous = v
    end
    previous <= 64eps() || return nothing
    return total
end

function contact_independent(field, V, i, theta)
    mx, _, _ = plane_of(field, i, 1)
    if abs(theta - pi / 2) <= pi / 4
        s = mx < 0 ? -1 : 1
        H0 = column_height(V, i, 1, 1, s); H1 = column_height(V, i, 2, 1, s)
        (isnothing(H0) || isnothing(H1)) && return nothing
        derivative = -cos(theta) / (s * sin(theta))
        Hg = H0 - derivative
        slope = (H1 - Hg) / 2
        return (; kappa = -s * (H1 - 2H0 + Hg) / (1 + slope^2)^1.5, branch = :right, H0, H1, Hg, s)
    end
    s = cos(theta) > 0 ? 1 : -1
    d = -(mx < 0 ? -1 : 1) * s
    W0 = wall_height(V, i + d, s)
    W1 = wall_height(V, i + 2d, s)
    (isnothing(W0) || isnothing(W1) || !(W1 > W0 > 0)) && return nothing
    tn = abs(tan(theta))
    A2 = ((W1 - W0)^2 - tn^2) / (W0 + W1 + sqrt(4W0 * W1 + tn^2))
    A1 = (W1 - W0) - A2
    slope = A1 - 2A2
    return (; kappa = -s * 2A2 / (1 + slope^2)^1.5, branch = :acute, W0, W1, A2, A1, d, s)
end

function contact_data_and_check()
    io = IOBuffer()
    results = Dict{Symbol, Any}()
    for (panel, degrees) in ((:a, 70), (:b, 30))
        theta = degrees * pi / 180
        field, cy = sessile_fixture(theta)
        V = fraction_matrix(field)
        R = SESSILE.radius
        xr = SESSILE.centre_x + R * sin(theta)   # right contact point
        println(
            io, "\\gdef\\fxS$(panel)theta{$degrees}\\gdef\\fxS$(panel)cx{$(fmt(SESSILE.centre_x))}",
            "\\gdef\\fxS$(panel)cy{$(fmt(cy))}\\gdef\\fxS$(panel)R{$(fmt(R))}\\gdef\\fxS$(panel)xr{$(fmt(xr))}"
        )
        worst = 0.0; exact = 0.0; cells = 0
        target = nothing
        for i in 1:16
            c = V[i, 1]
            0 < c < 1 || continue
            AMR._plic_touches_wall(field, ref1(i, 1), 3) || continue
            mine = contact_independent(field, V, i, theta)
            isnothing(mine) && continue
            k = curvature_of(field, i, 1)
            worst = max(worst, abs(mine.kappa - k))
            exact = max(exact, abs(k - 1 / R))
            cells += 1
            # The figure shows the right contact cell.
            i - 0.5 > SESSILE.centre_x && (target = (i, mine))
        end
        @assert cells >= 2 && worst <= 1.0e-13
        i, mine = target
        for j in 1:4, ii in (i - 3):(i + 3)
            1 <= ii <= size(V, 1) || continue
            cc = V[ii, j]
            println(io, "\\fxscell{$panel}{$(fmt(ii - 1.0))}{$(fmt(j - 1.0))}{$(@sprintf("%.2f", cc))}")
            0 < cc < 1 || continue
            mx, my, a = plane_of(field, ii, j)
            p, q = segment_in(SQUARE, (mx, my), a)
            println(
                io, "\\fxsseg{$panel}{$(fmt(ii - 0.5 + p[1]))}{$(fmt(j - 0.5 + p[2]))}",
                "{$(fmt(ii - 0.5 + q[1]))}{$(fmt(j - 0.5 + q[2]))}"
            )
        end
        println(io, "\\gdef\\fxS$(panel)i{$i}")
        if mine.branch == :right
            # Rows j = 1 (wall cell), 2 (inward) and the ghost row 0, principal +x.
            for (row, h) in ((0, mine.Hg), (1, mine.H0), (2, mine.H1))
                println(io, "\\fxsheight{$panel}{$(fmt(i - 0.5 + h))}{$(fmt(row - 0.5))}{$row}")
            end
        else
            for (k, w) in ((1, mine.W0), (2, mine.W1))
                println(io, "\\fxsheight{$panel}{$(fmt(i - 0.5 + k * mine.d))}{$(fmt(w))}{$k}")
            end
            # Parabola through the two heights with slope tan(theta) at the
            # (unknown) contact point; x is measured from column i + d along d.
            curve = String[]
            x1 = i - 0.5 + mine.d
            # Contact point: the zero of the parabola with increasing height.
            roots = [(-mine.A1 + sg * sqrt(mine.A1^2 - 4mine.A2 * mine.W0)) / (2mine.A2) for sg in (-1, 1)]
            xstar = only(filter(x -> 2mine.A2 * x + mine.A1 > 0 && -2 < x < 0, roots))
            for x in range(xstar, 2.4; length = 25)
                push!(curve, "($(fmt(x1 + mine.d * x)),$(fmt(mine.A2 * x^2 + mine.A1 * x + mine.W0)))")
            end
            println(
                io, "\\gdef\\fxSbcurve{$(join(curve, " -- "))}\\gdef\\fxSbxstar{$(fmt(x1 + mine.d * xstar))}",
                "\\gdef\\fxSbslope{$(fmt(2mine.A2 * xstar + mine.A1))}"
            )
        end
        results[panel] = (; worst, exact, cells, kappa = curvature_of(field, i, 1))
    end
    return String(take!(io)), results
end
