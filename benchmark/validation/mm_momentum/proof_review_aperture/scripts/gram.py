"""Faithful NumPy/pure-Python port of the Gram block of one cut cell (unit cell coordinates, h = 1, rho = 1).

Source (read only): wt-mm-momentum/src/two_phase/projection/mimetic_metric.jl
  _mimetic_corner, _mimetic_normal, _mimetic_side_parts, _mimetic_symmetric_eigen,
  _mimetic_pressure_point, _mimetic_admissible_point, _mimetic_kernel_point,
  _mimetic_gram_block!, _mimetic_fill_gram_block!   (constants: margin 0.01, floor 0.02)
Sides k = 0..3 are bottom, right, top, left (the Julia k = 1..4); `a[k]` is the liquid aperture of the side,
`starts[k]` True iff the liquid part sits at the counter-clockwise start of the side (only used for 0 < a < 1).
Run nothing here directly; imported by the step scripts.
"""
import math
import numpy as np

MARGIN = 0.01
FLOOR = 0.02

CORNER = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
NORMAL = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]


def side_parts(k, a, start):
    ax, ay = CORNER[k]
    bx, by = CORNER[(k + 1) % 4]
    s = a if start else 1.0 - a
    qx, qy = ax + s * (bx - ax), ay + s * (by - ay)
    low = ((ax + qx) / 2, (ay + qy) / 2)
    high = ((qx + bx) / 2, (qy + by) / 2)
    return (qx, qy), (low if start else high), (high if start else low)


def sym_eigen(a, b, d):
    mean = (a + d) / 2
    radius = math.hypot((a - d) / 2, b)
    angle = math.atan2(2 * b, a - d) / 2
    return mean - radius, mean + radius, math.cos(angle), math.sin(angle)


def kernel_point(apertures, starts, g11, g12, g22, xs, ys):
    l1, l2, c, s = sym_eigen(g11, g12, g22)
    tol = 1e-10
    for lam, vx, vy in ((l1, -s, c), (l2, c, s)):
        for phase in (1, 2):
            if not ((lam if phase == 1 else 1 - lam) <= tol):
                continue
            total = weighted = 0.0
            for k in range(4):
                b = apertures[k] if phase == 1 else 1 - apertures[k]
                if not b > 0:
                    continue
                _, liq, gas = side_parts(k, apertures[k], starts[k])
                mx, my = liq if phase == 1 else gas
                total += b
                weighted += b * (vx * mx + vy * my)
            if not total > 0:
                continue
            shift = weighted / total - (vx * xs + vy * ys)
            xs += shift * vx
            ys += shift * vy
    return xs, ys


def admissible_point(m11, m12, m21, m22, ex, ey, xs, ys):
    tol = 1e-12
    c = math.sqrt(ex ** 2 + ey ** 2)
    hx, hy = ex / c, ey / c
    px, py = -hy, hx
    g11, g22 = m11 - ex * xs, m22 - ey * ys
    g12 = ((m12 - ex * ys) + (m21 - ey * xs)) / 2
    q11 = hx * (g11 * hx + g12 * hy) + hy * (g12 * hx + g22 * hy)
    q12 = hx * (g11 * px + g12 * py) + hy * (g12 * px + g22 * py)
    q22 = px * (g11 * px + g12 * py) + py * (g12 * px + g22 * py)
    if not (-tol <= q22 <= 1 + tol):
        return xs, ys, False
    q22 = min(max(q22, 0.0), 1.0)
    lo = q12 ** 2 / q22 if q22 > 1e-14 else (0.0 if abs(q12) < tol else math.inf)
    hi = 1 - q12 ** 2 / (1 - q22) if 1 - q22 > 1e-14 else (1.0 if abs(q12) < tol else -math.inf)
    if lo > hi + tol:
        return xs, ys, False
    smin, smax = (q11 - hi) / c, (q11 - lo) / c
    margin = min(max(smax - smin, 0.0) / 100, MARGIN)
    hi_s = max(smin + margin, smax - margin)
    s = min(max(0.0, smin + margin), hi_s)
    return xs + s * hx, ys + s * hy, True


def pressure_point(apertures, starts):
    ex = ey = sx = sy = 0.0
    m11 = m12 = m21 = m22 = 0.0
    crossed = 0
    for k in range(4):
        a = apertures[k]
        nx, ny = NORMAL[k]
        (qx, qy), (lx, ly), _ = side_parts(k, a, starts[k])
        ex += a * nx
        ey += a * ny
        m11 += a * nx * lx
        m12 += a * nx * ly
        m21 += a * ny * lx
        m22 += a * ny * ly
        if 0 < a < 1:
            sx += qx
            sy += qy
            crossed += 1
    if crossed == 0:
        minority = 0.0 if sum(apertures) >= 2 else 1.0
        for k in range(4):
            if apertures[k] != minority:
                continue
            ax, ay = CORNER[k]
            bx, by = CORNER[(k + 1) % 4]
            sx += (ax + bx) / 2
            sy += (ay + by) / 2
            crossed += 1
    xr, yr = (sx / crossed, sy / crossed) if crossed > 0 else (0.0, 0.0)
    e2 = ex ** 2 + ey ** 2
    if e2 > 1e-18:
        defect = (m12 - m21) - (ex * yr - ey * xr)
        xs, ys, adm = admissible_point(m11, m12, m21, m22, ex, ey, xr - defect * ey / e2, yr + defect * ex / e2)
    else:
        xs, ys = kernel_point(apertures, starts, m11, (m12 + m21) / 2, m22, xr, yr)
        adm = False
    if not adm and math.hypot(xs, ys) > 1.5:
        xs, ys = xr, yr
    return xs, ys, ex, ey, adm


def fill_block(m, sides, lengths, moments, l1, l2, c, s, floored, phase, ex, ey):
    def weight(lam):
        if floored:
            return 1.0 / max(lam, FLOOR)
        return 1.0 / lam if lam > 1e-10 else 0.0

    w1, w2 = weight(l1), weight(l2)
    i11 = w1 * s ** 2 + w2 * c ** 2
    i12 = (w2 - w1) * c * s
    i22 = w1 * c ** 2 + w2 * s ** 2
    e2 = ex ** 2 + ey ** 2
    closing = e2 > 1e-18
    cl = math.sqrt(e2) if closing else 0.0
    orient = -1.0 if phase == 1 else 1.0
    cnx, cny = (orient * ex / cl, orient * ey / cl) if closing else (0.0, 0.0)

    def edge(q):
        if q < m:
            return lengths[q], NORMAL[sides[q]]
        return cl, (cnx, cny)

    p11 = p12 = p22 = 0.0
    for q in range(m + (1 if closing else 0)):
        L, (nx, ny) = edge(q)
        p11 += (L * nx) ** 2
        p12 += L ** 2 * nx * ny
        p22 += (L * ny) ** 2
    mu1, mu2, pc, ps = sym_eigen(p11, p12, p22)
    tol = 1e-12 * max(mu2, 2.2250738585072014e-308)
    v1 = 1 / mu1 if mu1 > tol else 0.0
    v2 = 1 / mu2 if mu2 > tol else 0.0
    q11 = v1 * ps ** 2 + v2 * pc ** 2
    q12 = (v2 - v1) * pc * ps
    q22 = v1 * pc ** 2 + v2 * ps ** 2
    gamma = m

    def stab(q, r):
        Lq, (nqx, nqy) = edge(q)
        Lr, (nrx, nry) = edge(r)
        fqx, fqy, frx, fry = Lq * nqx, Lq * nqy, Lr * nrx, Lr * nry
        proj = (1.0 if q == r else 0.0) - (fqx * (q11 * frx + q12 * fry) + fqy * (q12 * frx + q22 * fry))
        return Lq * proj * Lr / 2

    block = np.zeros((m, m))
    for q in range(m):
        for r in range(q, m):
            rqx, rqy = moments[q]
            rrx, rry = moments[r]
            val = rqx * (i11 * rrx + i12 * rry) + rqy * (i12 * rrx + i22 * rry) + stab(q, r)
            if closing:
                sq, sr = lengths[q] / cl, lengths[r] / cl
                val += -sr * stab(q, gamma) - sq * stab(gamma, r) + sq * sr * stab(gamma, gamma)
            block[q, r] = block[r, q] = val
    return block


def gram_block(apertures, starts, phase, xs, ys, ex, ey):
    """Return (m, consistent, block, sides, lengths, moments, G) for `phase` (1 liquid, 2 gas)."""
    sides, lengths, moments = [], [], []
    g11 = g12 = g21 = g22 = 0.0
    for k in range(4):
        b = apertures[k] if phase == 1 else 1.0 - apertures[k]
        if not b > 0:
            continue
        _, liq, gas = side_parts(k, apertures[k], starts[k])
        mx, my = liq if phase == 1 else gas
        nx, ny = NORMAL[k]
        sides.append(k)
        lengths.append(b)
        rx, ry = b * (mx - xs), b * (my - ys)
        moments.append((rx, ry))
        g11 += nx * rx
        g12 += nx * ry
        g21 += ny * rx
        g22 += ny * ry
    m = len(sides)
    G = np.array([[g11, g12], [g21, g22]])
    if m == 0:
        return 0, True, np.zeros((0, 0)), sides, lengths, moments, G
    l1, l2, c, s = sym_eigen(g11, (g12 + g21) / 2, g22)
    consistent = abs(g12 - g21) <= 1e-10 and l1 > -1e-10
    block = None
    if consistent:
        block = fill_block(m, sides, lengths, moments, l1, l2, c, s, False, phase, ex, ey)
        for q in range(m):
            rx = ry = 0.0
            for r in range(m):
                nx, ny = NORMAL[sides[r]]
                rx += block[q, r] * nx
                ry += block[q, r] * ny
            consistent = consistent and max(abs(rx - moments[q][0]), abs(ry - moments[q][1])) <= 1e-8
    if not consistent:
        block = fill_block(m, sides, lengths, moments, l1, l2, c, s, True, phase, ex, ey)
    return m, consistent, block, sides, lengths, moments, G


def cell_blocks(apertures, starts):
    """Pressure point and both blocks of a cut cell (as the assembly does it)."""
    xs, ys, ex, ey, adm = pressure_point(apertures, starts)
    out = {}
    for phase in (1, 2):
        out[phase] = gram_block(apertures, starts, phase, xs, ys, ex, ey)
    return (xs, ys, ex, ey, adm), out


def line_apertures(theta, offset):
    """Exact liquid apertures and starts of the straight PLIC line n.x = offset in the unit cell, liquid n.x < offset,
    n = (cos theta, sin theta).  Returns (a[4], starts[4])."""
    nx, ny = math.cos(theta), math.sin(theta)
    a = [0.0] * 4
    st = [False] * 4
    for k in range(4):
        (ax, ay), (bx, by) = CORNER[k], CORNER[(k + 1) % 4]
        fa, fb = nx * ax + ny * ay - offset, nx * bx + ny * by - offset
        if fa <= 0 and fb <= 0:
            a[k] = 1.0
        elif fa > 0 and fb > 0:
            a[k] = 0.0
        else:
            t = fa / (fa - fb)  # parameter of the crossing from A to B
            if fa < 0:
                a[k], st[k] = t, True   # liquid at the start
            else:
                a[k], st[k] = 1.0 - t, False
    return a, st
