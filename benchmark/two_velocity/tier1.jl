# Tier 1: single projections with known exact answers (seconds per method).
# The laboratory T1–T7 of the mimetic-projection audit (§"Laboratory and
# tests"), on the package geometry (PLIC apertures and staggered fractions):
# the unit square, periodic in x, walls at y = 0 and y = 1, liquid below the
# graph or inside the circle; one projection with dt = 1 per case and method.
# T7 also carries the periodic manufactured Poisson problem of the former
# gallery example 12. Norms follow the audit: K-weighted rms over the mixed
# faces (K = m_l m_g / M the relative mass), T2/T3 errors divided by
# `U ε k` and `g Δt ε k`.

const KW = 2pi

wavy(y0, epsilon) = (x -> y0 + epsilon * cos(KW * x), x -> -epsilon * KW * sin(KW * x))
triangle(y0, s) = (x -> y0 + s * (0.25 - abs(mod(x, 1.0) - 0.5)), x -> -s * sign(mod(x, 1.0) - 0.5))

function lab_mesh(n; periodic = (true, false))
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (0.0, 0.0), extent = (1.0, 1.0), periodic,
    )
    @assert ncells(mesh) == n * n
    return mesh
end

"""Mesh, interface, solver and workspace of one laboratory case."""
function lab_case(m::Method, n, levelset; density = (1.0, 1.0), lipschitz = 1.0, periodic = (true, false))
    mesh = lab_mesh(n; periodic)
    interface = InterfaceField(mesh, levelset; lipschitz)
    solver, workspace = projection_solver(m, interface, density)
    return (; mesh, interface, solver, workspace, density)
end

velocity_pair(mesh, liquid, gas) = (
    FaceVelocity(mesh, (x, y, t, axis) -> liquid(x, y, axis)),
    FaceVelocity(mesh, (x, y, t, axis) -> gas(x, y, axis)),
)

function lab_project!(case, velocity; dt = 1.0, tolerance = PROJECTION_TOLERANCE)
    pressure = CellField(case.mesh)
    stats = project!(
        velocity, pressure, case.solver, dt, case.workspace;
        tolerance, max_iterations = PROJECTION_ITERATIONS,
    )
    return checked(stats), pressure
end

"""Face data of the projection (fractions, masses, apertures, mixed faces;
the internal `TwoPhaseAMR._projection_face_data` shared by both projections)."""
face_data(case) = TP._projection_face_data(case.solver, case.workspace)

jumps(velocity) = velocity[2].interior .- velocity[1].interior

created_jump(velocity) = maximum((abs(x) for x in jumps(velocity) if isfinite(x)); init = 0.0)

finite_maximum(values) = maximum((abs(x) for x in values if isfinite(x)); init = 0.0)

"""K-weighted rms of `e` over the mixed faces passing `keep`, and the largest
`|e|` over those with `min(c, 1 - c) >= minfrac`."""
function kweighted(data, e; keep = i -> true, minfrac = 0.0)
    num = den = emax = 0.0
    for i in data.mixed
        keep(i) && isfinite(e[i]) || continue
        K = data.relative_mass[i]
        num += K * e[i]^2
        den += K
        c = data.liquid_fraction[i]
        min(c, 1 - c) >= minfrac && (emax = max(emax, abs(e[i])))
    end
    return den > 0 ? sqrt(num / den) : 0.0, emax
end

"""Mass-weighted mean of the present phase values of every face."""
function mean_velocity(data, velocity)
    l, g = velocity[1].interior, velocity[2].interior
    return map(eachindex(l)) do i
        a, b = isfinite(l[i]), isfinite(g[i])
        a && b ? (data.liquid_mass[i] * l[i] + data.gas_mass[i] * g[i]) / data.total_mass[i] :
            a ? l[i] : b ? g[i] : 0.0
    end
end

density_label(density) = "DR=" * string(round(Int, density[1] / density[2]))

# ---------------------------------------------------------------- T1
METRIC_DESCRIPTIONS["T1/max_jump"] = "largest created phase-velocity jump, exact 0"
METRIC_DESCRIPTIONS["T1/max_change"] = "largest change of a phase value, exact 0"

function tier1_t1!(suite, m)
    delta = 0.05
    U(y) = 0.5 * tanh((y - 0.5) / delta)
    cases = [
        "flat 0.37h" => n -> wavy(0.5 + 0.37 / n, 0.0),
        "cos a=0.01" => n -> wavy(0.5, 0.01),
        "triangle s=0.06" => n -> triangle(0.5, 0.06),
        "triangle s=0.3" => n -> triangle(0.5, 0.3),
    ]
    for (name, geometry) in cases, density in ((1.0, 1.0), (1000.0, 1.0)), n in suite.resolutions
        case_name = name * " " * density_label(density)
        guarded(suite, 1, "T1", case_name, n, m) do
            eta, _ = geometry(n)
            case = lab_case(m, n, (x, y) -> y - eta(x); density, lipschitz = 1.5)
            shear(x, y, axis) = axis == 1 ? U(y) : 0.0
            velocity = velocity_pair(case.mesh, shear, shear)
            initial = copy.(velocity)
            lab_project!(case, velocity)
            change = maximum(
                finite_maximum(velocity[p].interior .- initial[p].interior) for p in 1:2
            )
            record!(suite, 1, "T1", case_name, n, m, "max_jump", created_jump(velocity); atol = 1.0e-10)
            record!(suite, 1, "T1", case_name, n, m, "max_change", change; atol = 1.0e-10)
        end
    end
    return
end

# ---------------------------------------------------------------- T2 / T3
METRIC_DESCRIPTIONS["T2/e2"] = "K-weighted rms of (j⁺ − exact)/(U ε k)"
METRIC_DESCRIPTIONS["T2/e2_wetted"] = "e2 over the faces with both values wetted"
METRIC_DESCRIPTIONS["T2/e2_slaved"] = "e2 over the faces with a value without aperture"
METRIC_DESCRIPTIONS["T2/extension"] = "mass-weighted rms of each value against its own phase's exact field"
METRIC_DESCRIPTIONS["T3/e2"] = "K-weighted rms of (j⁺ − exact)/(g Δt ε k)"
METRIC_DESCRIPTIONS["T3/e2_wetted"] = "e2 over the faces with both values wetted"
METRIC_DESCRIPTIONS["T3/e2_slaved"] = "e2 over the faces with a value without aperture"
METRIC_DESCRIPTIONS["T3/extension"] = "mass-weighted rms of each value against its own phase's exact field"
METRIC_DESCRIPTIONS["T3/max_u"] = "largest projected value at equal densities, exact rest"

"""Exact first-order per-phase fields of T2 (sheet, U = 1) and T3 (gravity,
g dt = 1), extended analytically beyond the interface (harmonic pressures,
dt = 1)."""
function exact_field(test, density, y0, epsilon, x, y, axis, phase)
    k = KW
    ρl, ρg = density
    Tl, Tg = tanh(k * y0), tanh(k * (1 - y0))
    cl(y) = cosh(k * y) / cosh(k * y0)
    sl(y) = sinh(k * y) / cosh(k * y0)
    cg(y) = cosh(k * (1 - y)) / cosh(k * (1 - y0))
    sg(y) = sinh(k * (1 - y)) / cosh(k * (1 - y0))
    if test === :sheet
        Pa = -epsilon / (Tg / ρg + Tl / ρl)          # p^π = Pa sin kx (cosh profile)
        phase == 1 && return axis == 1 ? -0.5 - Pa * k * cos(k * x) * cl(y) / ρl : -Pa * k * sin(k * x) * sl(y) / ρl
        return axis == 1 ? 0.5 - Pa * k * cos(k * x) * cg(y) / ρg : Pa * k * sin(k * x) * sg(y) / ρg
    end
    A = (ρl - ρg) * epsilon / (1 + Tl * ρg / (Tg * ρl))
    B = A - (ρl - ρg) * epsilon                    # p1^π = A|B cos kx (cosh profile)
    phase == 1 && return axis == 1 ? A * k * sin(k * x) * cl(y) / ρl : -A * k * cos(k * x) * sl(y) / ρl
    return axis == 1 ? B * k * sin(k * x) * cg(y) / ρg : B * k * cos(k * x) * sg(y) / ρg
end

function tier1_t2t3!(suite, m)
    epsilon = 0.005
    for test in (:sheet, :gravity), density in ((1.0, 1.0), (1000.0, 1.0)), n in suite.resolutions
        group = test === :sheet ? "T2" : "T3"
        case_name = (test === :sheet ? "sheet" : "gravity") * " wavy ε=0.005 " * density_label(density)
        guarded(suite, 1, group, case_name, n, m) do
            y0 = 0.5 + 0.37 / n
            eta, deta = wavy(y0, epsilon)
            Tl, Tg = tanh(KW * y0), tanh(KW * (1 - y0))
            ρl, ρg = density
            dγ = if test === :sheet
                R = (1 / ρg - 1 / ρl) / (Tg / ρg + Tl / ρl)
                x -> epsilon * KW * R * cos(KW * x)
            else
                A = (ρl - ρg) * epsilon / (1 + Tl * ρg / (Tg * ρl))
                x -> -KW * sin(KW * x) * (A / ρl) * (1 + Tl / Tg)
            end
            case = lab_case(m, n, (x, y) -> y - eta(x); density)
            velocity = if test === :sheet
                velocity_pair(case.mesh, (x, y, a) -> a == 1 ? -0.5 : 0.0, (x, y, a) -> a == 1 ? 0.5 : 0.0)
            else
                w = velocity_pair(case.mesh, (x, y, a) -> a == 2 ? -1.0 : 0.0, (x, y, a) -> a == 2 ? -1.0 : 0.0)
                fill!(w[1].boundary, 0.0)
                fill!(w[2].boundary, 0.0)
                w
            end
            hydrostatic = test === :gravity && density[1] != density[2]
            lab_project!(case, velocity; tolerance = hydrostatic ? HYDROSTATIC_TOLERANCE : PROJECTION_TOLERANCE)
            data = face_data(case)
            j = jumps(velocity)
            e = fill(NaN, length(j))
            num = den = 0.0
            for i in data.mixed
                face = case.mesh.faces[i]
                x = face.center[1]
                s = sqrt(1 + deta(x)^2)
                τ = (1 / s, deta(x) / s)
                γ = (test === :sheet ? τ[1] : 0.0) + dγ(x)
                e[i] = (j[i] - γ * τ[face.axis]) / (epsilon * KW)
                for (phase, mass) in ((1, data.liquid_mass[i]), (2, data.gas_mass[i]))
                    u = velocity[phase].interior[i]
                    isfinite(u) || continue
                    d = (u - exact_field(test, density, y0, epsilon, face.center..., face.axis, phase)) / (epsilon * KW)
                    num += mass * d^2
                    den += mass
                end
            end
            e2, _ = kweighted(data, e; minfrac = 0.05)
            wet(i) = data.liquid_aperture[i] > 1.0e-12 && data.gas_aperture[i] > 1.0e-12
            e2w, _ = kweighted(data, e; keep = wet)
            e2s, _ = kweighted(data, e; keep = i -> !wet(i))
            rest = test === :gravity && density[1] == density[2]
            atol = rest ? 1.0e-8 : 0.0
            record!(suite, 1, group, case_name, n, m, "e2", e2; atol)
            record!(suite, 1, group, case_name, n, m, "e2_wetted", e2w; atol)
            record!(suite, 1, group, case_name, n, m, "e2_slaved", e2s; atol)
            record!(suite, 1, group, case_name, n, m, "extension", sqrt(num / den); atol)
            rest && record!(
                suite, 1, group, case_name, n, m, "max_u",
                maximum(finite_maximum(velocity[p].interior) for p in 1:2); atol = 1.0e-8,
            )
        end
    end
    return
end

# ---------------------------------------------------------------- T4
METRIC_DESCRIPTIONS["T4/e2"] = "K-weighted rms of j⁺ − (J* − n(n·J*))·n_f, kinks ±0.1 excluded"

function tier1_t4!(suite, m)
    away(x) = min(abs(mod(x, 1.0) - 0.5), mod(x, 1.0), 1 - mod(x, 1.0)) > 0.1
    cases = [
        ("triangle s=0.3 DR=1", n -> triangle(0.5 + 0.37 / n, 0.3), away, 1.1),
        ("triangle s=1 DR=1", n -> triangle(0.5 + 0.37 / n, 1.0), away, 1.5),
        ("circle R=0.25 DR=1", n -> nothing, x -> true, 1.0),
    ]
    for (case_name, geometry, window, lipschitz) in cases, n in suite.resolutions
        guarded(suite, 1, "T4", case_name, n, m) do
            xc, yc, R = 0.5 + 0.21 / n, 0.5 + 0.37 / n, 0.25
            g = geometry(n)
            levelset = isnothing(g) ? (x, y) -> hypot(x - xc, y - yc) - R : (x, y) -> y - g[1](x)
            normal(x, y) = if isnothing(g)
                r = hypot(x - xc, y - yc)
                ((x - xc) / r, (y - yc) / r)
            else
                d = g[2](x)
                s = sqrt(1 + d^2)
                (-d / s, 1 / s)
            end
            case = lab_case(m, n, levelset; lipschitz)
            velocity = velocity_pair(case.mesh, (x, y, a) -> a == 1 ? -0.5 : 0.0, (x, y, a) -> a == 1 ? 0.5 : 0.0)
            lab_project!(case, velocity)
            data = face_data(case)
            j = jumps(velocity)
            e = fill(NaN, length(j))
            for i in data.mixed
                face = case.mesh.faces[i]
                nx, ny = normal(face.center...)
                exact = face.axis == 1 ? 1 - nx * nx : -ny * nx
                e[i] = j[i] - exact
            end
            e2, _ = kweighted(data, e; keep = i -> window(case.mesh.faces[i].center[1]), minfrac = 0.05)
            record!(suite, 1, "T4", case_name, n, m, "e2", e2)
        end
    end
    return
end

# ---------------------------------------------------------------- T5
METRIC_DESCRIPTIONS["T5/max_jump"] = "max|j⁺|/max|Gφ|, exact rest"
METRIC_DESCRIPTIONS["T5/max_u"] = "max|u⁺|/max|Gφ|, exact rest"

function tier1_t5!(suite, m)
    cases = [
        "wavy ε=0.02 DR=1" => n -> (x, y) -> y - wavy(0.5 + 0.37 / n, 0.02)[1](x),
        "triangle s=0.3 DR=1" => n -> (x, y) -> y - triangle(0.5 + 0.37 / n, 0.3)[1](x),
        "circle R=0.25 DR=1" => n -> (x, y) -> hypot(x - 0.5 - 0.21 / n, y - 0.5 - 0.37 / n) - 0.25,
    ]
    for (case_name, levelset) in cases, n in suite.resolutions
        guarded(suite, 1, "T5", case_name, n, m) do
            case = lab_case(m, n, levelset(n); lipschitz = 1.1)
            phi = CellField(case.mesh, (x, y) -> cospi(2x) * cospi(y))
            velocity = (FaceVelocity(case.mesh), FaceVelocity(case.mesh))
            h = 1 / n
            for (i, face) in pairs(case.mesh.faces)
                g = (TP._get(phi, face.plus) - TP._get(phi, face.minus)) / h
                velocity[1].interior[i] = velocity[2].interior[i] = -g
            end
            fill!(velocity[1].boundary, 0.0)
            fill!(velocity[2].boundary, 0.0)
            scale = maximum(abs, velocity[1].interior)
            lab_project!(case, velocity)
            # The rest state is exact: values at the projection tolerance are
            # equal (CMDGFM removes a single-valued gradient to round-off).
            record!(suite, 1, "T5", case_name, n, m, "max_jump", created_jump(velocity) / scale; atol = 1.0e-8)
            record!(
                suite, 1, "T5", case_name, n, m, "max_u",
                maximum(finite_maximum(velocity[p].interior) for p in 1:2) / scale; atol = 1.0e-8,
            )
        end
    end
    return
end

# ---------------------------------------------------------------- T6 / T7
METRIC_DESCRIPTIONS["T6/jump_ratio"] = "rms(j⁺)/rms(j*), exact rest"
METRIC_DESCRIPTIONS["T6/mean_ratio"] = "max|ū⁺|/max|ū*| of the mass-weighted mean, exact rest"
METRIC_DESCRIPTIONS["T7/jump_ratio"] = "rms(j⁺)/rms(j*), exact rest"
METRIC_DESCRIPTIONS["T7/mean_ratio"] = "max|ū⁺|/max|ū*| of the mass-weighted mean, exact rest"
METRIC_DESCRIPTIONS["T7/p_rms_centre"] = "rms pressure error at the cell centres, mean removed"
METRIC_DESCRIPTIONS["T7/p_rms_point"] = "rms pressure error at the method's pressure points, mean removed"

"""Pressure points of the projection: the mimetic metric's (the chord
points of cut cells), the cell centres otherwise."""
function pressure_point(case, cell, x, y)
    case.workspace isa MimeticProjectionWorkspace || return (x, y)
    assembly = case.workspace.assembly
    return (assembly.pressure_x[cell], assembly.pressure_y[cell])
end

"""One projection of the two-phase gradient `(∇p_l/ρ_l, ∇p_g/ρ_g)`: the
rest state is exact."""
function gradient_case!(suite, m, group, case_name, n, density, levelset, lipschitz, dpl, dpg, pex)
    return guarded(suite, 1, group, case_name, n, m) do
        case = lab_case(m, n, levelset; density, lipschitz)
        velocity = velocity_pair(
            case.mesh, (x, y, a) -> dpl(x, y, a) / density[1], (x, y, a) -> dpg(x, y, a) / density[2],
        )
        fill!(velocity[1].boundary, 0.0)
        fill!(velocity[2].boundary, 0.0)
        TP._mark_inactive_phase_velocities!(velocity, case.interface)
        data = face_data(case)
        e0, _ = kweighted(data, jumps(velocity))
        ub0 = maximum(abs, mean_velocity(data, velocity))
        _, pressure = lab_project!(case, velocity)
        e1, _ = kweighted(data, jumps(velocity))
        ub1 = maximum(abs, mean_velocity(data, velocity))
        record!(suite, 1, group, case_name, n, m, "jump_ratio", e1 / e0)
        record!(suite, 1, group, case_name, n, m, "mean_ratio", ub1 / ub0)
        isnothing(pex) && return
        # Pressure error (constant removed) at the cell centres and at the
        # method's pressure points.
        centre, point = Float64[], Float64[]
        cell = 0
        for (b, _) in pairs(case.mesh.blocks), jj in 1:case.mesh.block_size[2], ii in 1:case.mesh.block_size[1]
            cell += 1
            x, y = cellcenter(case.mesh, b, ii, jj)
            value = interior(pressure, b)[ii, jj]
            push!(centre, value - pex(x, y))
            push!(point, value - pex(pressure_point(case, cell, x, y)...))
        end
        centre .-= sum(centre) / length(centre)
        point .-= sum(point) / length(point)
        record!(suite, 1, group, case_name, n, m, "p_rms_centre", sqrt(sum(abs2, centre) / length(centre)))
        record!(suite, 1, group, case_name, n, m, "p_rms_point", sqrt(sum(abs2, point) / length(point)))
    end
end

function tier1_t6!(suite, m)
    k = KW
    for density in ((1.0, 1.0), (1000.0, 1.0)), n in suite.resolutions
        y0 = 0.5 + 0.37 / n
        dpl(x, y, ax) = ax == 1 ? -k * sin(k * x) * cosh(k * y) / cosh(k * y0) : k * cos(k * x) * sinh(k * y) / cosh(k * y0)
        dpg(x, y, ax) = ax == 1 ? -k * sin(k * x) * cosh(k * (1 - y)) / cosh(k * (1 - y0)) :
            -k * cos(k * x) * sinh(k * (1 - y)) / cosh(k * (1 - y0))
        gradient_case!(
            suite, m, "T6", "flat 0.37h " * density_label(density), n, density, (x, y) -> y - y0, 1.0, dpl, dpg, nothing,
        )
    end
    return
end

# The manufactured pressure of T7 and of the former gallery example 12: a
# smooth periodic background plus, inside the circle r < 0.3 about (½, ½), a
# term that vanishes on the circle, so the pressure is continuous with a jump
# of its gradient.
const MANUFACTURED_RADIUS = 0.3
const MANUFACTURED_SCALE = 40.0

manufactured_outer(x, y) = sinpi(2x) + cospi(2y)
manufactured_level(x, y) = (x - 0.5)^2 + (y - 0.5)^2 - MANUFACTURED_RADIUS^2
manufactured_pressure(x, y) = manufactured_outer(x, y) +
    (manufactured_level(x, y) < 0 ? MANUFACTURED_SCALE * manufactured_level(x, y) * (x - y) : 0.0)

"""Exact pressure gradient of phase 1 (liquid, inside) or 2 (gas)."""
function manufactured_gradient(x, y, axis, phase)
    outer = axis == 1 ? 2pi * cospi(2x) : -2pi * sinpi(2y)
    phase == 2 && return outer
    level = manufactured_level(x, y)
    extra = axis == 1 ? 2 * (x - 0.5) * (x - y) + level : 2 * (y - 0.5) * (x - y) - level
    return outer + MANUFACTURED_SCALE * extra
end

"""Divergence-free liquid rotation of the former gallery example 12 (tangential
on the circle); the gas is at rest."""
function manufactured_velocity(x, y, axis, phase)
    phase == 2 && return 0.0
    factor = cospi(2 * ((x - 0.5)^2 + (y - 0.5)^2))
    return axis == 1 ? 2 * (y - 0.5) * factor : -2 * (x - 0.5) * factor
end

"""Surface rms of the normal and tangential components of the exact scaled
jump `[∇p/ρ]` on the circle; the former gallery example 12 was designed with
equal amplitudes."""
function manufactured_jump_rms(density; samples = 4096)
    normal = tangential = 0.0
    for sample in 0:(samples - 1)
        angle = 2pi * sample / samples
        nx, ny = cos(angle), sin(angle)
        x, y = 0.5 + MANUFACTURED_RADIUS * nx, 0.5 + MANUFACTURED_RADIUS * ny
        jump = ntuple(
            axis -> manufactured_gradient(x, y, axis, 2) / density[2] - manufactured_gradient(x, y, axis, 1) / density[1], 2,
        )
        normal += (jump[1] * nx + jump[2] * ny)^2
        tangential += (jump[2] * nx - jump[1] * ny)^2
    end
    return sqrt(normal / samples), sqrt(tangential / samples)
end

METRIC_DESCRIPTIONS["T7/p_l2"] = "volume-weighted rms pressure error at the cell centres, gauge aligned"
METRIC_DESCRIPTIONS["T7/p_linf"] = "largest pressure error at the cell centres, gauge aligned"
METRIC_DESCRIPTIONS["T7/grad_l2"] = "phase-fraction-weighted rms error of the realised phase pressure gradients ρ(u* − u⁺)/Δt"
METRIC_DESCRIPTIONS["T7/grad_linf"] = "largest realised phase-gradient error (phase fraction > √eps)"
METRIC_DESCRIPTIONS["T7/xi_l2"] = "rms error of the scaled gradient jump ξ = [∇p/ρ] on the mixed faces"
METRIC_DESCRIPTIONS["T7/xi_linf"] = "largest error of the scaled gradient jump on the mixed faces"

"""The periodic manufactured Poisson problem of the former gallery example
12: predictor = divergence-free liquid rotation + ∇p/ρ, one projection with
Δt = 1, so the projection should return the rotation and the manufactured
pressure. Errors of the gauge-aligned pressure, of the realised phase
gradients `ρ_π (u*_π − u⁺_π)/Δt` and of their scaled jump on the mixed faces."""
function manufactured_case!(suite, m, n, density)
    case_name = "ex12 periodic circle R=0.3 " * density_label(density)
    return guarded(suite, 1, "T7", case_name, n, m) do
        case = lab_case(
            m, n, (x, y) -> hypot(x - 0.5, y - 0.5) - MANUFACTURED_RADIUS;
            density, periodic = (true, true),
        )
        mesh = case.mesh
        predictor = ntuple(
            phase -> FaceVelocity(
                mesh, (x, y, t, axis) -> manufactured_velocity(x, y, axis, phase) +
                    manufactured_gradient(x, y, axis, phase) / density[phase],
            ), 2,
        )
        velocity = copy.(predictor)
        _, pressure = lab_project!(case, velocity)
        # Gauge-aligned pressure (volume-weighted mean offset removed).
        error = Float64[]
        for (b, _) in pairs(mesh.blocks), jj in 1:mesh.block_size[2], ii in 1:mesh.block_size[1]
            x, y = cellcenter(mesh, b, ii, jj)
            push!(error, interior(pressure, b)[ii, jj] - manufactured_pressure(x, y))
        end
        error .-= sum(error) / length(error)
        record!(suite, 1, "T7", case_name, n, m, "p_l2", sqrt(sum(abs2, error) / length(error)))
        record!(suite, 1, "T7", case_name, n, m, "p_linf", maximum(abs, error))
        # Realised phase gradients and their scaled jump.
        data = face_data(case)
        volumes = mesh.face_dual_volumes
        gnum = gden = gmax = 0.0
        realised = ntuple(2) do phase
            map(eachindex(mesh.faces)) do i
                density[phase] * (predictor[phase].interior[i] - velocity[phase].interior[i])
            end
        end
        for (i, face) in pairs(mesh.faces), phase in 1:2
            fraction = phase == 1 ? data.liquid_fraction[i] : 1 - data.liquid_fraction[i]
            fraction > 0 && isfinite(realised[phase][i]) || continue
            e = realised[phase][i] - manufactured_gradient(face.center..., face.axis, phase)
            gnum += fraction * volumes[i] * e^2
            gden += fraction * volumes[i]
            fraction > sqrt(eps()) && (gmax = max(gmax, abs(e)))
        end
        record!(suite, 1, "T7", case_name, n, m, "grad_l2", sqrt(gnum / gden))
        record!(suite, 1, "T7", case_name, n, m, "grad_linf", gmax)
        xnum = xden = xmax = 0.0
        for i in data.mixed
            face = mesh.faces[i]
            xi = realised[2][i] / density[2] - realised[1][i] / density[1]
            isfinite(xi) || continue
            exact = manufactured_gradient(face.center..., face.axis, 2) / density[2] -
                manufactured_gradient(face.center..., face.axis, 1) / density[1]
            xnum += volumes[i] * (xi - exact)^2
            xden += volumes[i]
            xmax = max(xmax, abs(xi - exact))
        end
        record!(suite, 1, "T7", case_name, n, m, "xi_l2", sqrt(xnum / xden))
        record!(suite, 1, "T7", case_name, n, m, "xi_linf", xmax)
    end
end

function tier1_t7!(suite, m)
    R0 = MANUFACTURED_RADIUS
    pex(x, y) = manufactured_pressure(x, y)
    dpin(x, y, axis) = manufactured_gradient(x, y, axis, 1)
    dpout(x, y, axis) = manufactured_gradient(x, y, axis, 2)
    for density in ((1.0, 1.0), (1000.0, 1.0)), n in suite.resolutions
        gradient_case!(
            suite, m, "T7", "circle R=0.3 walls " * density_label(density), n, density,
            (x, y) -> hypot(x - 0.5, y - 0.5) - R0, 1.0, dpin, dpout, pex,
        )
    end
    # The former gallery Example 12 was designed so that the normal and tangential
    # parts of the scaled jump have nearly equal surface rms (its former test:
    # 0.8–1.25).
    normal, tangential = manufactured_jump_rms((1000.0, 1.0))
    @assert 0.8 <= tangential / normal <= 1.25
    for n in suite.resolutions
        manufactured_case!(suite, m, n, (1000.0, 1.0))
    end
    return
end

# Marks of the summary table: values exact, or limited, by construction (the
# docs page explains them; measured where stated there).
MARK_TEXT["E"] = "exact by construction (round-off or projection tolerance)"
MARK_TEXT["C1"] = "CMDGFM-C one-donor rows: the two-point difference ±2hU' of the shear, first order (#171)"
MARK_TEXT["C2"] = "CMDGFM-C: mixed faces without a regular row keep their predictor jump; no convergence"
MARK_TEXT["L"] = "CMDGFM-L limitation pinned as @test_broken (#171)"
MARK_TEXT["G"] = "n = 128: the wave crests cross a grid line and leave liquid slivers (T2, T3); not solver limited"
MARK_TEXT["M"] = "MM: the sampled gradient is not a gradient of the mimetic metric in cut cells, O(h)"
MARK_TEXT["?"] = "does not converge; cause not established"
for m in (:cmdgfm, :cmdgfm_constant, :mimetic)
    push!(TABLE_MARKS, ("T3", r"DR=1$", r"", m, "E"), ("T3", r"DR=1000$", r"", m, "G"))
end
push!(
    TABLE_MARKS,
    ("T1", r"", r"", :mimetic, "E"),
    ("T1", r"^flat", r"", :cmdgfm, "E"),
    ("T1", r"^flat", r"", :cmdgfm_constant, "E"),
    ("T1", r"^(cos|triangle)", r"", :cmdgfm_constant, "C1"),
    ("T1", r"^triangle s=0\.3 ", r"", :cmdgfm, "L"),
    ("T2", r"", r"^e2(_slaved)?$", :cmdgfm_constant, "G"),
    ("T2", r"DR=1000$", r"^e2_slaved$", :cmdgfm, "G"),
    ("T4", r"^triangle s=1 ", r"", :cmdgfm, "L"),
    ("T4", r"", r"", :cmdgfm_constant, "C2"),
    ("T5", r"", r"", :cmdgfm, "E"),
    ("T5", r"", r"", :cmdgfm_constant, "E"),
    ("T5", r"", r"", :mimetic, "M"),
    ("T6", r"DR=1000$", r"^jump_ratio$", :cmdgfm_constant, "C2"),
    ("T6", r"", r"ratio", :mimetic, "M"),
    ("T7", r"walls", r"ratio", :cmdgfm_constant, "C2"),
    ("T7", r"walls", r"ratio", :cmdgfm, "?"),
    ("T7", r"walls", r"ratio", :mimetic, "M"),
    ("T7", r"^ex12", r"^(xi|grad_linf)", :cmdgfm, "?"),
    ("T7", r"^ex12", r"^(xi|grad_linf|p_linf)", :cmdgfm_constant, "?"),
)

const TIER1_TESTS = (tier1_t1!, tier1_t2t3!, tier1_t4!, tier1_t5!, tier1_t6!, tier1_t7!)

function run_tier1!(suite)
    for m in suite.methods
        two_velocity(m) || continue
        println("Tier 1, ", m.label)
        flush(stdout)
        for test in TIER1_TESTS
            test(suite, m)
        end
    end
    record_orders!(suite, 1)
    return suite
end
