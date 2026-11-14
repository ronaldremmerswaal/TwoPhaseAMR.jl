# Exact face geometry of the mimetic-projection audit laboratory
# (mimetic-projection-audit/scripts/Lab.jl, written from the audit's
# definitions): the unit square, periodic in x, walls at y = 0 and y = 1,
# liquid below a graph y = eta(x) or inside a circle; exact apertures and
# staggered fractions by the composite midpoint rule of the exact liquid length
# in y (4000 points). The package's mimetic metric and solve are fed with these
# data, so that the laboratory's numbers can be reproduced independently of
# the PLIC reconstruction.
module MimeticLab

using TwoPhaseAMR
const TP = TwoPhaseAMR
const TOL = 1.0e-9

struct Graph{F, G}
    eta::F
    deta::G
end

struct Circle
    xc::Float64
    yc::Float64
    R::Float64
end

wavy(y0, epsilon; k = 2pi) = Graph(x -> y0 + epsilon * cos(k * x), x -> -epsilon * k * sin(k * x))
triangle(y0, s) = Graph(
    x -> y0 + s * (0.25 - abs(mod(x, 1.0) - 0.5)), x -> -s * sign(mod(x, 1.0) - 0.5),
)

vertical(g::Graph, x) = (-1.0, g.eta(x))
function vertical(g::Circle, x)
    s = sqrt(max(g.R^2 - (x - g.xc)^2, 0.0))
    return (g.yc - s, g.yc + s)
end

function normal(g::Graph, x, y)
    d = g.deta(x)
    s = sqrt(1 + d * d)
    return (-d / s, 1 / s)
end
function normal(g::Circle, x, y)
    dx, dy = x - g.xc, y - g.yc
    r = hypot(dx, dy)
    return (dx / r, dy / r)
end

function may_cut(g::Graph, x0, x1, y0, y1)
    emin, emax, dmax = Inf, -Inf, 0.0
    for k in 0:16
        x = x0 + k * (x1 - x0) / 16
        e = g.eta(x)
        emin = min(emin, e); emax = max(emax, e); dmax = max(dmax, abs(g.deta(x)))
    end
    pad = 0.5 * (x1 - x0) * dmax + 1.0e-12
    y0 > emax + pad && return 0.0
    y1 < emin - pad && return 1.0
    return nothing
end
function may_cut(g::Circle, x0, x1, y0, y1)
    d = hypot(0.5 * (x0 + x1) - g.xc, 0.5 * (y0 + y1) - g.yc)
    half = 0.5 * hypot(x1 - x0, y1 - y0)
    d > g.R + half + 1.0e-12 && return 0.0
    d < g.R - half - 1.0e-12 && return 1.0
    return nothing
end

function frac_rect(g, x0, x1, y0, y1; nq = 4000)
    pre = may_cut(g, x0, x1, y0, y1)
    isnothing(pre) || return pre
    acc = 0.0
    for k in 1:nq
        lo, hi = vertical(g, x0 + (k - 0.5) * (x1 - x0) / nq)
        acc += max(min(hi, y1) - max(lo, y0), 0.0)
    end
    return acc / nq / (y1 - y0)
end

function aperture_xface(g, x, y0, y1)
    lo, hi = vertical(g, x)
    return clamp((min(hi, y1) - max(lo, y0)) / (y1 - y0), 0.0, 1.0)
end

function aperture_yface(g::Graph, x0, x1, y)
    pre = may_cut(g, x0, x1, y - 1.0e-9, y + 1.0e-9)
    isnothing(pre) || return pre
    len = 0.0
    m = 64
    for k in 1:m
        a = x0 + (k - 1) * (x1 - x0) / m
        b = x0 + k * (x1 - x0) / m
        ga, gb = g.eta(a) - y, g.eta(b) - y
        if ga > 0 && gb > 0
            len += b - a
        elseif (ga > 0) != (gb > 0)
            lo, hi, glo = a, b, ga
            for _ in 1:60
                mid = 0.5 * (lo + hi)
                gm = g.eta(mid) - y
                (gm > 0) == (glo > 0) ? ((lo, glo) = (mid, gm)) : (hi = mid)
            end
            root = 0.5 * (lo + hi)
            len += ga > 0 ? root - a : b - root
        end
    end
    return len / (x1 - x0)
end
function aperture_yface(g::Circle, x0, x1, y)
    s2 = g.R^2 - (y - g.yc)^2
    s2 <= 0 && return 0.0
    s = sqrt(s2)
    return max(0.0, min(x1, g.xc + s) - max(x0, g.xc - s)) / (x1 - x0)
end

liquid_at(g, x, y) = (v = vertical(g, x); v[1] < y < v[2])

"""Exact face data of `geo` on the package mesh (n x n cells of the unit square)."""
function face_data(mesh, geo; density)
    h = mesh.blocks[1].spacing[1]
    nf = length(mesh.faces)
    c = zeros(nf); al = zeros(nf); flags = zeros(Int8, nf)
    for (f, face) in pairs(mesh.faces)
        x, y = face.center
        if face.axis == 1
            c[f] = frac_rect(geo, x - h / 2, x + h / 2, y - h / 2, y + h / 2)
            al[f] = aperture_xface(geo, x, y - h / 2, y + h / 2)
        else
            c[f] = frac_rect(geo, x - h / 2, x + h / 2, y - h / 2, y + h / 2)
            al[f] = aperture_yface(geo, x - h / 2, x + h / 2, y)
        end
        c[f] = clamp(c[f], 0.0, 1.0)
        c[f] < TOL && (c[f] = 0.0)
        c[f] > 1 - TOL && (c[f] = 1.0)
        # Laboratory apertures: exact on mixed faces, 0/1 elsewhere, snapped at TOL.
        a = 0 < c[f] < 1 ? al[f] : (c[f] > 0.5 ? 1.0 : 0.0)
        a = a < TOL ? 0.0 : (a > 1 - TOL ? 1.0 : a)
        al[f] = a
        if 0 < a < 1
            t = 0.5 * min(a, 1 - a) * h
            px, py = face.axis == 1 ? (x, y - h / 2 + t) : (x - h / 2 + t, y)
            flags[f] = liquid_at(geo, px, py) ? TP._MIMETIC_LIQUID_LOW : TP._MIMETIC_LIQUID_HIGH
        end
    end
    nb = length(mesh.boundary_faces)
    bal = zeros(nb)
    for (b, face) in pairs(mesh.boundary_faces)
        x, y = face.center
        bal[b] = face.axis == 2 ? aperture_yface(geo, x - h / 2, x + h / 2, y) :
            aperture_xface(geo, x, y - h / 2, y + h / 2)
    end
    data = (;
        liquid_fraction = c, liquid_aperture = al, gas_aperture = 1 .- al,
        liquid_mass = density[1] .* c, gas_mass = density[2] .* (1 .- c),
    )
    return data, flags, bal, falses(nb)
end

"""Laboratory projection: the package's mimetic metric and solve on exact geometry.

`ul(f)`, `ug(f)` give the predictor phase velocities of face `f`. Returns the
projected phase values, the jump on the mixed faces and the solve result.
"""
function project(
        mesh, interface, geo, ul, ug; density,
        dt = 1.0, tolerance = 1.0e-12,
    )
    solver = TP.MimeticProjectionSolver(interface; density)
    work = TP.MimeticProjectionWorkspace(solver)
    data, flags, bal, blow = face_data(mesh, geo; density)
    assembly = TP._mimetic_assemble_from_data!(
        work, mesh, data, flags, bal, blow, solver.density,
    )
    work.predictor = [
        assembly.dof_phase[r] == 1 ? ul(assembly.dof_face[r]) : ug(assembly.dof_face[r])
            for r in eachindex(assembly.dof_face)
    ]
    fill!(work.boundary_flux, 0.0)
    fill!(work.multiplier, 0.0)
    preconditioner = TP._mimetic_preconditioner!(solver, work)
    result = TP._mimetic_solve!(
        work, dt, preconditioner, nothing, 0.0; tolerance, max_iterations = 2000,
    )
    nf = length(mesh.faces)
    uln = fill(NaN, nf); ugn = fill(NaN, nf)
    for r in eachindex(assembly.dof_face)
        (assembly.dof_phase[r] == 1 ? uln : ugn)[assembly.dof_face[r]] = work.velocity[r]
    end
    mixed = findall(f -> 0 < data.liquid_fraction[f] < 1, 1:nf)
    return (; uln, ugn, mixed, jump = ugn[mixed] .- uln[mixed], data, result, work, assembly)
end

"""K-weighted rms of a jump field on the mixed faces (laboratory `norms`)."""
function weighted_rms(mesh, data, mixed, e)
    h = mesh.blocks[1].spacing[1]
    K = [h^2 * data.liquid_mass[f] * data.gas_mass[f] / (data.liquid_mass[f] + data.gas_mass[f]) for f in mixed]
    return sqrt(sum(K .* e .^ 2) / sum(K))
end

"""Laboratory T5: created jump of a single-valued gradient predictor (equal densities)."""
function t5_error(n; kwargs...)
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (true, false))
    geo = Circle(0.5 + 0.21 / n, 0.5 + 0.37 / n, 0.25)
    interface = InterfaceField(mesh, (x, y) -> hypot(x - geo.xc, y - geo.yc) - geo.R; lipschitz = 1)
    h = 1 / n
    phi(reference) = (
        c = cellcenter(mesh, reference.block, reference.i, reference.j);
        cos(2pi * c[1]) * cos(pi * c[2])
    )
    u = [-(phi(face.plus) - phi(face.minus)) / h for face in mesh.faces]
    out = project(mesh, interface, geo, f -> u[f], f -> u[f]; density = (1.0, 1.0), kwargs...)
    return maximum(abs, out.jump) / maximum(abs, u), out
end

"""Laboratory T7: manufactured Poisson problem on a circle of radius 0.3."""
function t7_error(n, density; kwargs...)
    R0 = 0.3
    pout(x, y) = sin(2pi * x) + cos(2pi * y)
    pin(x, y) = pout(x, y) + 40 * ((x - 0.5)^2 + (y - 0.5)^2 - R0^2) * (x - y)
    dpout(x, y, axis) = axis == 1 ? 2pi * cos(2pi * x) : -2pi * sin(2pi * y)
    function dpin(x, y, axis)
        r2 = (x - 0.5)^2 + (y - 0.5)^2 - R0^2
        extra = axis == 1 ? 80 * (x - 0.5) * (x - y) + 40 * r2 : 80 * (y - 0.5) * (x - y) - 40 * r2
        return dpout(x, y, axis) + extra
    end
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (true, false))
    geo = Circle(0.5, 0.5, R0)
    interface = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - R0; lipschitz = 1)
    ul(f) = (face = mesh.faces[f]; dpin(face.center..., face.axis) / density[1])
    ug(f) = (face = mesh.faces[f]; dpout(face.center..., face.axis) / density[2])
    out = project(mesh, interface, geo, ul, ug; density, kwargs...)
    predictor = [ug(f) - ul(f) for f in out.mixed]
    j0 = weighted_rms(mesh, out.data, out.mixed, predictor)
    exact = map(TP._cell_references(mesh)) do reference
        x, y = cellcenter(mesh, reference.block, reference.i, reference.j)
        hypot(x - 0.5, y - 0.5) < R0 ? pin(x, y) : pout(x, y)
    end
    dp = out.work.multiplier ./ 1.0 .- exact
    dp .-= sum(dp) / length(dp)
    return weighted_rms(mesh, out.data, out.mixed, out.jump) / j0, sqrt(sum(abs2, dp) / length(dp)), out
end

"""Laboratory T3: error of the created sheet strength under a uniform acceleration."""
function t3_error(n, density; dy = 0.37, epsilon = 0.005, k = 2pi, kwargs...)
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (true, false))
    y0 = 0.5 + dy / n
    geo = wavy(y0, epsilon; k)
    interface = InterfaceField(mesh, (x, y) -> y - geo.eta(x); lipschitz = 1 + epsilon * k)
    out = project(
        mesh, interface, geo, f -> mesh.faces[f].axis == 2 ? -1.0 : 0.0,
        f -> mesh.faces[f].axis == 2 ? -1.0 : 0.0; density, kwargs...
    )
    rl, rg = density
    Tl, Tg = tanh(k * y0), tanh(k * (1 - y0))
    A = (rl - rg) * epsilon / (1 + Tl * rg / (Tg * rl))
    gamma(x) = -k * sin(k * x) * (A / rl) * (1 + Tl / Tg)
    exact = map(out.mixed) do f
        x, y = mesh.faces[f].center
        nx, ny = normal(geo, x, y)
        tau = mesh.faces[f].axis == 1 ? ny : -nx
        gamma(x) * tau
    end
    return weighted_rms(mesh, out.data, out.mixed, (out.jump .- exact) ./ (epsilon * k)), out
end
end
