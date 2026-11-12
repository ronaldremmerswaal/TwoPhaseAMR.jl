# Chapter 8 (surface tension, gravity and body forces): figure data and spot
# checks. Included by generate.jl after momentum.jl, whose helpers it reuses
# (all_cells, cell_box_of, cell_level, cell_spacing, cell_volume, cond,
# liquid_plane, open_part, to_physical, polygon_text, segment_in, SQUARE,
# strip_open_area, wrap). The functions marked "independent" evaluate the
# formulas of Chapter 8 without the package's force kernels: the face
# curvature, the staggered density, the integrated jumps and the force map
# (with the code's inverse face metric of Chapter 3). The differences to the
# code are the numbers quoted in the text; the assertions fail if they
# disagree.

using SparseArrays: spdiagm

# --- Fixtures ------------------------------------------------------------------

# Walled 8 x 8 box of 16 x 16 level-0 cells; :amr refines the right half
# (level transition x = 4), :cut adds a sloped solid at the bottom.
const FORCE_BOX = 8.0
const FORCE_DROP = (; centre = (4.3, 4.1), radius = 2.23)
const FORCE_DENSITY = (1000.0, 1.0)
const FORCE_SIGMA = 0.7
const FORCE_SOLID_LOW = (; offset = 0.9, slope = 0.1)    # solid y < 0.9 + 0.1x
const FORCE_SOLID_HIGH = (; offset = 7.1, slope = -0.1)  # solid y > 7.1 - 0.1x

function force_mesh(kind::Symbol)
    refine = kind == :uniform ? ((key, origin, extent) -> false) :
        ((key, origin, extent) -> key.level == 0 && key.i >= 2)
    return BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (FORCE_BOX, FORCE_BOX),
        periodic = (false, false), max_level = 1, refine,
    )
end

force_solid_low(x, y) = y - FORCE_SOLID_LOW.offset - FORCE_SOLID_LOW.slope * x
force_solid_high(x, y) = FORCE_SOLID_HIGH.offset + FORCE_SOLID_HIGH.slope * x - y
force_drop(x, y) = hypot(x - FORCE_DROP.centre[1], y - FORCE_DROP.centre[2]) - FORCE_DROP.radius

# A drop on the uniform, composite or cut mesh (the solid does not touch it).
function force_drop_fixture(kind::Symbol)
    mesh = force_mesh(kind == :uniform ? :uniform : :amr)
    if kind == :cut
        geometry = GeometryField(mesh, force_solid_low; lipschitz = hypot(1, FORCE_SOLID_LOW.slope))
        return InterfaceField(mesh, force_drop; geometry, contact_angle = pi / 2, lipschitz = 1)
    end
    return InterfaceField(mesh, force_drop; lipschitz = 1)
end

# The thin film of the test "Unresolved surface-tension curvature": its
# mixed cells have no curvature, so faces need the extension or stay
# unresolved.
function force_film_fixture()
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    film = InterfaceField(mesh, (x, y) -> 1.0)
    values = interior(film.fraction, 1)
    fill!(values, 0)
    values[1, 2:6] .= (0.0, 0.167, 0.961, 0.994, 1.0)
    reconstruct!(film; force = true)
    return film
end

force_operator(field) = isnothing(field.geometry) ?
    PoissonOperator(field; density = FORCE_DENSITY) :
    PoissonOperator(field, field.geometry; density = FORCE_DENSITY)

# --- Independent face curvature (Section "curvature") -------------------------------

# Weighted average over the two cells, else the extension search over all
# cells in block and cell order; `nothing` if unresolved.
function face_curvature_independent(field, face)
    mesh = field.mesh
    total = weight = 0.0
    for ref in (face.minus, face.plus)
        c = cond(field, ref)
        k = AMR._get(field.curvature, ref)
        w = c * (1 - c)
        if w > 0 && isfinite(k)
            total += w * k
            weight += w
        end
    end
    weight > 0 && return total / weight, :neighbours
    radius = 3 * max(maximum(cell_spacing(mesh, face.minus)), maximum(cell_spacing(mesh, face.plus)))
    for ref in all_cells(mesh)
        c = cond(field, ref)
        k = AMR._get(field.curvature, ref)
        w = c * (1 - c)
        (w > 0 && isfinite(k)) || continue
        h = maximum(cell_spacing(mesh, ref))
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        dx = wrap(x[1] - face.center[1], mesh.extent[1], mesh.periodic[1])
        dy = wrap(x[2] - face.center[2], mesh.extent[2], mesh.periodic[2])
        d = hypot(dx, dy)
        d <= radius + h || continue
        g = w / max(d^2, (h / 4)^2)
        total += g * k
        weight += g
    end
    weight > 0 && return total / weight, :extension
    return nothing, :unresolved
end

# Every face is compared; `counts` are over the faces that carry a CSF jump
# (conditional fractions differ), i.e. that need a curvature, `all_counts`
# over all faces (the comparison also exercises the extension there).
function face_curvature_check(field)
    worst = 0.0
    counts = Dict(:neighbours => 0, :extension => 0, :unresolved => 0)
    all_counts = Dict(:neighbours => 0, :extension => 0, :unresolved => 0)
    for face in field.mesh.faces
        expected, kind = face_curvature_independent(field, face)
        actual = AMR._face_curvature(field, face)
        if isnothing(expected)
            @assert isnothing(actual)
        else
            worst = max(worst, abs(actual - expected) / max(abs(expected), 1.0e-300))
        end
        all_counts[kind] += 1
        cond(field, face.minus) == cond(field, face.plus) && continue
        counts[kind] += 1
    end
    return (; worst, counts, all_counts)
end

# --- Independent staggered density and surface-tension jumps --------------------------

# Liquid fraction of the open staggered volume (volume-weighted interpolation
# of the conditional fractions, clamped), from independent strip areas.
function staggered_fraction_independent(field, face)
    geometry = field.geometry
    weighted = volume = 0.0
    for ref in (face.minus, face.plus)
        # Local area of the strip of `ref` over the tangential extent of f:
        # |f| / h_t without solid, its open part with a solid.
        strip = isnothing(geometry) ? face.area / cell_spacing(field.mesh, ref)[3 - face.axis] :
            first(strip_open_area(geometry, ref, face))
        contribution = 0.5 * strip * cell_volume(field.mesh, ref)
        weighted += contribution * cond(field, ref)
        volume += contribution
    end
    return volume > 0 ? clamp(weighted / volume, 0.0, 1.0) : 0.0
end

staggered_density_independent(field, face; density = FORCE_DENSITY) =
let c = staggered_fraction_independent(field, face)
    density[1] * c + density[2] * (1 - c)
end

function surface_tension_jumps_independent(field, formulation; sigma = FORCE_SIGMA, curvature = nothing)
    mesh = field.mesh
    jumps = zeros(length(mesh.faces))
    unresolved = 0
    for (index, face) in pairs(mesh.faces)
        !isnothing(field.geometry) && iszero(field.geometry.aperture[index]) && continue
        cm, cp = cond(field, face.minus), cond(field, face.plus)
        psi = formulation isa CSF ? cp - cm : Float64((cp > 0.5) - (cm > 0.5))
        (sigma == 0 || psi == 0) && continue
        kappa = isnothing(curvature) ? first(face_curvature_independent(field, face)) : curvature
        if isnothing(kappa)
            unresolved += 1
            continue
        end
        jumps[index] = sigma * kappa * psi * face.area
    end
    return jumps, unresolved
end

# Force map R of the pressure (Chapter 8, Section "map"), with the code's
# inverse face metric (Chapter 3; scaled by W with a solid) and the
# independent coefficient 1/rho or a^2/rho.
function force_map_independent(field, operator, jumps)
    mesh = field.mesh
    metric = operator.gradient_operator.inverse_metric
    K = metric isa AMR.DiagonalInverseFaceMetric ? spdiagm(0 => metric.diagonal) : metric.matrix
    geometry = field.geometry
    a = [isnothing(geometry) ? 1.0 : geometry.aperture[i] for i in eachindex(mesh.faces)]
    gamma = [a[i]^2 / staggered_density_independent(field, face) for (i, face) in pairs(mesh.faces)]
    root = sqrt.(gamma)
    out = root .* (K * (root .* jumps))
    isnothing(geometry) && return out
    return [a[i] > 0 ? out[i] / a[i] : 0.0 for i in eachindex(out)]
end

function surface_tension_check(kind::Symbol)
    field = force_drop_fixture(kind)
    operator = force_operator(field)
    work = EllipticWorkspace(operator)
    worst_jump = worst_map = worst_balance = 0.0
    projected = Dict{Symbol, Float64}()
    unresolved = Ref(0)
    for formulation in (CSF(), CGFM())
        force = FaceVelocity(field.mesh)
        surface_tension!(
            force, field, operator, work; sigma = FORCE_SIGMA, formulation,
            unresolved_curvature_faces = unresolved
        )
        jumps, missing = surface_tension_jumps_independent(field, formulation)
        @assert missing == unresolved[]
        scale = maximum(abs, jumps)
        worst_jump = max(worst_jump, maximum(abs, operator.gradient_operator.pressure_jump .- jumps) / scale)
        replica = force_map_independent(field, operator, jumps)
        worst_map = max(worst_map, maximum(abs, force.interior .- replica) / maximum(abs, replica))
        # Balanced force: with a constant face curvature the force is the
        # weighted gradient of sigma*kappa*psi and the projection removes it.
        kappa = 1 / FORCE_DROP.radius
        surface_tension!(force, field, operator, work; sigma = FORCE_SIGMA, formulation, curvature = kappa)
        potential = CellField(field.mesh)
        for ref in all_cells(field.mesh)
            c = cond(field, ref)
            AMR._set!(potential, FORCE_SIGMA * kappa * (formulation isa CSF ? c : Float64(c > 0.5)), ref)
        end
        fill_halo!(potential)
        gradient = FaceVelocity(field.mesh)
        if isnothing(field.geometry)
            weighted_gradient!(gradient, potential, operator, work)
        else
            AMR._cut_cell_weighted_gradient!(gradient, potential, operator, work, field.geometry)
        end
        scale = maximum(abs, force.interior)
        worst_balance = max(worst_balance, maximum(abs, force.interior .- gradient.interior) / scale)
        projected[Symbol(typeof(formulation).name.name)] = project_force(field, operator, force) / scale
    end
    return (; worst_jump, worst_map, worst_balance, projected, faces = length(field.mesh.faces))
end

# Velocity left by one projection of dt * acceleration, divided by dt.
function project_force(field, operator, acceleration; dt = 0.01)
    velocity = FaceVelocity(field.mesh)
    velocity.interior .= dt .* acceleration.interior
    solver = EllipticSolver(operator; preconditioner = :tree_geometric)
    stats = if isnothing(field.geometry)
        project!(
            velocity, CellField(field.mesh), solver, dt, ProjectionWorkspace(solver);
            tolerance = 1.0e-12, max_cycles = 1000
        )
    else
        project!(
            velocity, CellField(field.mesh), solver, dt, ProjectionWorkspace(solver), field.geometry;
            tolerance = 1.0e-12, max_cycles = 1000
        )
    end
    @assert stats.converged
    return maximum(abs, velocity.interior) / dt
end

function unresolved_check()
    film = force_film_fixture()
    operator = PoissonOperator(film)
    count = Ref(0)
    force = FaceVelocity(film.mesh)
    surface_tension!(
        force, film, operator, EllipticWorkspace(operator);
        sigma = 1.0, formulation = CGFM(), unresolved_curvature_faces = count
    )
    jumps, missing = surface_tension_jumps_independent(film, CGFM(); sigma = 1.0)
    curvature = face_curvature_check(film)
    @assert count[] == missing > 0
    @assert jumps == operator.gradient_operator.pressure_jump
    return (;
        unresolved = count[], extension = curvature.counts[:extension],
        worst = curvature.worst,
    )
end

# --- Gravity (Section "gravity") ----------------------------------------------

# Planar interface normal to g through (4.3, 4.6); `solid` = :none, :liquid
# (sloped solid at the bottom, in the liquid) or :gas (at the top).
function gravity_fixture(kind::Symbol; gravity = (0.0, -1.0), solid = :none)
    mesh = force_mesh(kind)
    levelset = (x, y) -> -(gravity[1] * (x - 4.3) + gravity[2] * (y - 4.6))
    solid == :none && return InterfaceField(mesh, levelset; lipschitz = 1)
    shape = solid == :liquid ? force_solid_low : force_solid_high
    geometry = GeometryField(mesh, shape; lipschitz = hypot(1, 0.1))
    return InterfaceField(mesh, levelset; geometry, contact_angle = pi / 2, lipschitz = 1)
end

# Independent potential g . x^I_c: midpoint of the PLIC segment in the full
# reference cell for 0 < C < 1, the cell centre otherwise.
function gravity_potential_independent(field, gravity)
    mesh = field.mesh
    values = Dict{CellRef, Float64}()
    for ref in all_cells(mesh)
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = cond(field, ref)
        if 0 < c < 1
            n, a = liquid_plane(field, ref)
            p, q = segment_in(SQUARE, n, a)
            h = cell_spacing(mesh, ref)
            x = (x[1] + h[1] * (p[1] + q[1]) / 2, x[2] + h[2] * (p[2] + q[2]) / 2)
        end
        values[ref] = gravity[1] * x[1] + gravity[2] * x[2]
    end
    return values
end

# Aperture density (rho_l a_l + rho_g a_g) / a_o of the open face, with the
# phase apertures of Chapter 4; closed faces (a_o = 0) carry no jump.
function gravity_jumps_independent(field, gravity; density = FORCE_DENSITY)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    potential = gravity_potential_independent(field, gravity)
    jumps = zeros(length(field.mesh.faces))
    for (index, face) in pairs(field.mesh.faces)
        a = isnothing(field.geometry) ? 1.0 : field.geometry.aperture[index]
        a > 0 || continue
        rho = (density[1] * liquid[index] + density[2] * gas[index]) / a
        jumps[index] = rho * face.area * (potential[face.plus] - potential[face.minus])
    end
    return jumps
end

function gravity_check(kind::Symbol; gravity = (0.0, -1.0), solid = :none)
    field = gravity_fixture(kind; gravity, solid)
    operator = force_operator(field)
    work = EllipticWorkspace(operator)
    force = FaceVelocity(field.mesh)
    gravity!(force, field, operator, work; density = FORCE_DENSITY, gravity)
    jumps = gravity_jumps_independent(field, gravity)
    jump_difference = maximum(abs, operator.gradient_operator.pressure_jump .- jumps) / maximum(abs, jumps)
    g = hypot(gravity...)
    residual = project_force(field, operator, force) / g
    replica = force_map_independent(field, operator, jumps)
    map_difference = maximum(abs, force.interior .- replica) / maximum(abs, replica)
    liquid, gas = AMR._current_phase_face_apertures!(field)
    cut_liquid = isnothing(field.geometry) ? 0 :
        count(i -> 0 < field.geometry.aperture[i] < 1 && gas[i] == 0 && liquid[i] > 0, eachindex(liquid))
    return (; jump_difference, map_difference, residual, cut_liquid)
end

# --- Two velocities (Section "two velocities") ----------------------------------

two_maxabs(v::FaceVelocity) = maximum(x -> isnan(x) ? 0.0 : abs(x), v.interior)
two_maxabs(v) = max(two_maxabs(v[1]), two_maxabs(v[2]))

# One full inviscid step from rest of a heavy liquid below a planar interface
# (gravity (0, -1)); the velocity left, divided by dt |g|.
function hydrostatic_step(kind::Symbol, model; solid = :none, dt = 0.01)
    field = gravity_fixture(kind; solid)
    physics = TwoPhase(field; density = FORCE_DENSITY, dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -1.0))
    flow = FlowSimulation(field.mesh, physics; velocity_model = model, dt)
    simulate!(flow, dt; tolerance = 1.0e-10)
    return two_maxabs(flow.velocity) / dt
end

# The pre-#76 increment: the same gravity jumps through the one-velocity map,
# whose coefficient is (aperture)^2/rho instead of the projection's
# aperture/rho. Projection only, as a counterfactual: on the composite mesh
# with a solid the cut face at the level transition is coupled by the inverse
# metric to faces that carry the jump, so CMDGFM cannot remove it (#76).
function hydrostatic_former_step(kind::Symbol; solid = :gas, dt = 0.01)
    field = gravity_fixture(kind; solid)
    geometry = field.geometry
    isnothing(geometry) && return 0.0
    operator = PoissonOperator(field, geometry; density = FORCE_DENSITY)
    solver = CMDGFMProjectionSolver(field; density = FORCE_DENSITY)
    force = FaceVelocity(field.mesh)
    gravity!(
        force, field, operator, EllipticWorkspace(operator);
        density = FORCE_DENSITY, gravity = (0.0, -1.0),
    )
    phases = (FaceVelocity(field.mesh), FaceVelocity(field.mesh))
    for phase in phases
        phase.interior .= dt .* force.interior
    end
    AMR._mark_inactive_phase_velocities!(phases, field)
    stats = project!(
        phases, CellField(field.mesh), solver, dt,
        CMDGFMProjectionWorkspace(solver);
        tolerance = 1.0e-12, max_iterations = 4000,
    )
    @assert stats.converged
    return two_maxabs(phases) / dt
end

# A drop with prescribed constant curvature: the CSF tension force, added to
# both phase velocities and projected by CMDGFM. As in the two-velocity step
# (_phase_force_substep!), the force uses the pressure map of the projection:
# the coefficient a/rho with a solid, the one-velocity map without one.
function two_velocity_tension(
        kind::Symbol, projection::Symbol; dt = 0.01, formulation = CSF(),
    )
    field = force_drop_fixture(kind)
    solver = projection == :one ? nothing :
        projection == :cmdgfm ? CMDGFMProjectionSolver(field; density = FORCE_DENSITY) :
        error("unknown projection $projection")
    operator = isnothing(solver) ? force_operator(field) : AMR._projection_pressure_operator(solver)
    force = FaceVelocity(field.mesh)
    surface_tension!(
        force, field, operator, EllipticWorkspace(operator);
        sigma = FORCE_SIGMA, formulation, curvature = 1 / FORCE_DROP.radius
    )
    scale = maximum(abs, force.interior)
    isnothing(solver) && return project_force(field, operator, force) / scale
    phases = (FaceVelocity(field.mesh), FaceVelocity(field.mesh))
    for phase in phases
        phase.interior .= dt .* force.interior
    end
    AMR._mark_inactive_phase_velocities!(phases, field)
    workspace = CMDGFMProjectionWorkspace(solver)
    stats = project!(phases, CellField(field.mesh), solver, dt, workspace; tolerance = 1.0e-12, max_iterations = 4000)
    @assert stats.converged
    return two_maxabs(phases) / (dt * scale)
end

# --- Figure data: jumps and accelerations near a level transition ---------------------

const FORCE_WINDOW = (2.0, 5.5, 4.0, 6.5)    # x0, x1, y0, y1

#   \fxfcell{x0}{y0}{x1}{y1}{level}  \fxfliquid{path}  \fxfplic{x1}{y1}{x2}{y2}
#   \fxfjump{panel}{axis}{x}{y}{length}{jump}{acceleration}{class}
#     class 0: jump and acceleration, 1: acceleration without jump (metric coupling)
#   \fxfhanging{x}{y}
function force_data()
    field = force_drop_fixture(:amr)
    mesh = field.mesh
    operator = force_operator(field)
    work = EllipticWorkspace(operator)
    io = IOBuffer()
    inside(x, y) = FORCE_WINDOW[1] <= x <= FORCE_WINDOW[2] && FORCE_WINDOW[3] <= y <= FORCE_WINDOW[4]
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        (
            box[3] > FORCE_WINDOW[1] && box[1] < FORCE_WINDOW[2] &&
                box[4] > FORCE_WINDOW[3] && box[2] < FORCE_WINDOW[4]
        ) || continue
        println(io, "\\fxfcell{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}{$(cell_level(mesh, ref))}")
        c = cond(field, ref)
        c <= 0 && continue
        if c >= 1
            println(io, "\\fxfliquid{$(polygon_text(to_physical(mesh, ref, SQUARE)))}")
            continue
        end
        n, a = liquid_plane(field, ref)
        println(io, "\\fxfliquid{$(polygon_text(to_physical(mesh, ref, clip_halfplane(SQUARE, n, a; keep_less = true))))}")
        p, q = to_physical(mesh, ref, segment_in(SQUARE, n, a))
        println(io, "\\fxfplic{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
    end
    info = Dict{Symbol, Int}()
    coupled_faces = 0
    for (panel, formulation) in enumerate((CSF(), CGFM()))
        force = FaceVelocity(mesh)
        surface_tension!(force, field, operator, work; sigma = FORCE_SIGMA, formulation)
        jumps = copy(operator.gradient_operator.pressure_jump)
        scale = maximum(abs, force.interior)
        for (index, face) in pairs(mesh.faces)
            jump = jumps[index]
            acceleration = force.interior[index] / scale
            (iszero(jump) && abs(acceleration) < 1.0e-12) && continue
            class = iszero(jump) ? 1 : 0
            key = Symbol("panel$(panel)_class$(class)")
            info[key] = get(info, key, 0) + 1
            inside(face.center...) || continue
            println(io, "\\fxfjump{$panel}{$(face.axis)}{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}{$(fmt(face.area))}{$(fmt(jump))}{$(fmt(acceleration))}{$class}")
        end
    end
    for face in mesh.faces
        face.axis == 1 || continue
        abs(face.center[1] - 4.0) < 1.0e-12 || continue
        y = face.center[2] + face.area / 2
        # hanging vertex: the end point shared by two fine segments
        if abs(mod(y, 0.5) - 0.25) < 1.0e-12 && inside(4.0, y)
            println(io, "\\fxfhanging{4}{$(fmt(y))}")
        end
    end
    return String(take!(io)), info
end

# --- Outputs -------------------------------------------------------------------

function force_outputs(write_output)
    data, figure_info = force_data()
    write_output("forces.tex", data)

    curvature = Dict(kind => face_curvature_check(force_drop_fixture(kind)) for kind in (:uniform, :amr, :cut))
    tension = Dict(kind => surface_tension_check(kind) for kind in (:uniform, :amr, :cut))
    unresolved = unresolved_check()
    gravity = Dict(
        (kind, solid) => gravity_check(kind; solid) for kind in (:uniform, :amr) for solid in (:none, :gas, :liquid)
    )
    oblique = Dict(kind => gravity_check(kind; gravity = (0.6, -0.8)) for kind in (:uniform, :amr))
    models = (
        :one => OneVelocityModel(),
        :cmdgfm => TwoVelocityModel(pressure_projection = CMDGFMProjection()),
    )
    steps = Dict(
        (kind, solid, name) => hydrostatic_step(kind, model; solid)
            for kind in (:uniform, :amr) for solid in (:none, :gas, :liquid) for (name, model) in models
    )
    tension_two = Dict(
        (kind, projection) => two_velocity_tension(kind, projection)
            for kind in (:uniform, :amr, :cut) for projection in (:one, :cmdgfm)
    )

    println("force curvature check:       ", curvature)
    println("force tension check:         ", tension)
    println("force unresolved check:      ", unresolved)
    println("force gravity check:         ", gravity, " ", oblique)
    println("force two-velocity steps:    ", steps)
    println("force two-velocity tension:  ", tension_two)
    println("force figure:                ", figure_info)

    curvature_worst = maximum(c.worst for c in values(curvature))
    jump_worst = maximum(t.worst_jump for t in values(tension))
    map_worst = maximum(t.worst_map for t in values(tension))
    balance_worst = maximum(t.worst_balance for t in values(tension))
    projected_worst = maximum(maximum(values(t.projected)) for t in values(tension))
    @assert curvature_worst <= 1.0e-15 && jump_worst <= 1.0e-15
    @assert map_worst <= 1.0e-13 && balance_worst <= 1.0e-12 && projected_worst <= 1.0e-5
    gravity_jump = maximum(g.jump_difference for g in values(gravity))
    gravity_map = maximum(g.map_difference for g in (values(gravity)..., values(oblique)...))
    gravity_balanced = maximum(g.residual for g in (values(gravity)..., values(oblique)...))
    @assert gravity_jump <= 1.0e-14 && gravity_map <= 1.0e-13 && gravity_balanced <= 1.0e-8
    liquid_amr = gravity[(:amr, :liquid)]
    @assert liquid_amr.cut_liquid > 0
    one_steps = maximum(v for ((kind, solid, name), v) in steps if name == :one)
    @assert one_steps <= 1.0e-6
    # The two-velocity force stage uses the projection's pressure map (#76):
    # CMDGFM keeps the rest state with a solid in either phase, and the
    # constant-curvature drop on the cut mesh, while the former one-velocity
    # increment of the same jumps is not balanced on the composite mesh.
    cmdgfm_solid = Dict(
        kind => maximum(steps[(kind, solid, :cmdgfm)] for solid in (:gas, :liquid))
            for kind in (:uniform, :amr)
    )
    former = maximum(
        hydrostatic_former_step(kind; solid)
            for kind in (:uniform, :amr) for solid in (:gas, :liquid)
    )
    @assert maximum(values(cmdgfm_solid)) <= 1.0e-9
    @assert tension_two[(:cut, :cmdgfm)] <= 1.0e-9
    @assert former > 1.0e-3
    # The conservative jump of CGFM stays off that cut face, so it was always
    # balanced on the cut mesh (contrast of the CSF row above).
    cgfm_cut = two_velocity_tension(:cut, :cmdgfm; formulation = CGFM())
    @assert cgfm_cut <= 1.0e-9

    return Any[
        ("chkForceCurvDiff", sci(curvature_worst)),
        ("chkForceNeighbourFaces", curvature[:amr].counts[:neighbours]),
        ("chkForceExtensionFaces", curvature[:amr].counts[:extension]),
        ("chkForceUnresolvedFaces", curvature[:amr].counts[:unresolved]),
        ("chkForceAllFaces", sum(c.all_counts[k] for c in values(curvature) for k in keys(c.all_counts))),
        ("chkForceAllExtension", sum(c.all_counts[:extension] for c in values(curvature))),
        ("chkForceAllUndefined", sum(c.all_counts[:unresolved] for c in values(curvature))),
        ("chkForceFilmUnresolved", unresolved.unresolved),
        ("chkForceFilmExtension", unresolved.extension),
        ("chkForceJumpDiff", sci(jump_worst)),
        ("chkForceMapDiff", sci(map_worst)),
        ("chkForceBalanceDiff", sci(balance_worst)),
        ("chkForceProjected", sci(projected_worst)),
        ("chkForceCsfFaces", figure_info[:panel1_class0]),
        ("chkForceCgfmFaces", figure_info[:panel2_class0]),
        ("chkForceCsfCoupled", get(figure_info, :panel1_class1, 0)),
        ("chkForceCgfmCoupled", get(figure_info, :panel2_class1, 0)),
        ("chkGravityJumpDiff", sci(gravity_jump)),
        ("chkGravityMapDiff", sci(gravity_map)),
        ("chkGravityBalanced", sci(gravity_balanced)),
        ("chkGravityLiquidCut", liquid_amr.cut_liquid),
        ("chkStepOne", sci(one_steps)),
        ("chkStepCmdgfm", sci(max(steps[(:uniform, :none, :cmdgfm)], steps[(:amr, :none, :cmdgfm)]))),
        ("chkStepCmdgfmSolidUniform", sci(cmdgfm_solid[:uniform])),
        ("chkStepCmdgfmSolidAmr", sci(cmdgfm_solid[:amr])),
        ("chkStepCmdgfmFormer", sci(former)),
        ("chkTensionTwoOne", sci(maximum(tension_two[(k, :one)] for k in (:uniform, :amr, :cut)))),
        ("chkTensionTwoCmdgfm", sci(max(tension_two[(:uniform, :cmdgfm)], tension_two[(:amr, :cmdgfm)]))),
        ("chkTensionTwoCmdgfmCut", sci(tension_two[(:cut, :cmdgfm)])),
        ("chkTensionTwoCgfmCut", sci(cgfm_cut)),
    ]
end
