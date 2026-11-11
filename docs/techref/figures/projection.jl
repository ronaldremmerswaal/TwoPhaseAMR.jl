# Chapter 10 (pressure and velocity constraints): figure data and spot
# checks. Included by generate.jl after forces.jl, whose fixtures it reuses
# (force_mesh, force_solid_low, force_operator, FORCE_DENSITY), and vof.jl
# (all_cells, cell_box_of, cell_spacing, cond, liquid_plane, solid_plane_of,
# open_part, to_physical, polygon_text, in_window). The functions marked
# "independent" evaluate the formulas of Chapter 10 without the projection
# kernels: the defect identity and the kinetic energy of the one-velocity
# projection, the CMDGFM staggered normal, stencils, beta and row condition,
# the mixture divergence of the two-velocity projector. They read only mesh topology,
# fractions, apertures and the code's face metric; the assertions fail if
# the code no longer matches the chapter.

using LinearAlgebra: cholesky, Symmetric, dot
using SparseArrays: SparseMatrixCSC, spdiagm, nzrange, rowvals, nonzeros

# --- Fixtures --------------------------------------------------------------------

# A drop resting on the sloped solid of Chapter 8 (y < 0.9 + 0.1x) in the walled
# 8 x 8 box: contact points near x = 3.13 and x = 5.67, on both sides of the
# level transition x = 4 of the composite mesh. Without solid it is a free drop.
const PROJ_DROP = (; centre = (4.3, 2.3), radius = 1.6)
const PROJ_DT = 0.01
const PROJ_TOL_ONE = 1.0e-12
const PROJ_TOL_TWO = 1.0e-10

proj_drop(x, y) = hypot(x - PROJ_DROP.centre[1], y - PROJ_DROP.centre[2]) - PROJ_DROP.radius

function projection_fixture(kind::Symbol; solid::Bool)
    mesh = force_mesh(kind)
    solid || return InterfaceField(mesh, proj_drop; lipschitz = 1)
    geometry = GeometryField(mesh, force_solid_low; lipschitz = hypot(1, FORCE_SOLID_LOW.slope))
    return InterfaceField(mesh, proj_drop; geometry, contact_angle = pi / 2, lipschitz = 1)
end

# A divergent predictor: smooth, the same in both phases, plus a phase jump.
proj_common(x, y, axis) = axis == 1 ?
    0.4 + sinpi(x / 4) * cospi(y / 8) : -0.2 + cospi(x / 8) * sinpi(y / 4)
proj_jump(x, y, axis) = axis == 1 ? 0.37 + 0.1sinpi(y / 4) : -0.21 + 0.1cospi(x / 4)

function proj_one_predictor(field)
    velocity = FaceVelocity(field.mesh, (x, y, t, axis) -> proj_common(x, y, axis))
    enforce_no_permeation!(velocity)
    isnothing(field.geometry) || enforce_geometry_no_slip!(velocity, field.geometry)
    return velocity
end

function proj_two_predictor(field)
    mesh = field.mesh
    liquid = FaceVelocity(mesh, (x, y, t, axis) -> proj_common(x, y, axis))
    gas = FaceVelocity(mesh, (x, y, t, axis) -> proj_common(x, y, axis) + proj_jump(x, y, axis))
    velocity = (liquid, gas)
    enforce_no_permeation!(velocity)
    isnothing(field.geometry) || enforce_geometry_no_slip!(velocity, field.geometry)
    AMR._mark_inactive_phase_velocities!(velocity, field)
    return velocity
end

proj_copy(v::FaceVelocity) = (w = FaceVelocity(v.mesh); w.interior .= v.interior; w.boundary .= v.boundary; w)
proj_copy(v::Tuple) = map(proj_copy, v)

proj_aperture(field, index) = isnothing(field.geometry) ? 1.0 : field.geometry.aperture[index]
proj_open_volume(field, ref) = cell_volume(field.mesh, ref) *
    (isnothing(field.geometry) ? 1.0 : AMR._get(field.geometry.open_fraction, ref))

# Code face metric K^o (scaled by W with a solid) as a sparse matrix.
proj_metric(operator) = let metric = operator.gradient_operator.inverse_metric
    metric isa AMR.DiagonalInverseFaceMetric ? spdiagm(0 => metric.diagonal) : metric.matrix
end

# Independent squared kinetic norm |v|_M^2 of the pressure map with face
# coefficient gamma: z' (K^o_PP)^{-1} z with z = (a gamma^{-1/2} v)_P on the
# faces P with gamma > 0.
function proj_kinetic_norm(field, operator, gamma, values; factor = nothing)
    P = findall(>(0), gamma)
    z = [proj_aperture(field, f) * values[f] / sqrt(gamma[f]) for f in P]
    F = isnothing(factor) ? cholesky(Symmetric(proj_metric(operator)[P, P])) : factor
    return dot(z, F \ z), F
end

# --- One velocity (Section "one") -------------------------------------------------------

# Defect identity |c^o| (D u^{n+1})_c = -dt (r_c + b |c^o|) and the
# Pythagoras identity of the M-orthogonal projection, independently.
function one_velocity_check(kind::Symbol; solid::Bool)
    field = projection_fixture(kind; solid)
    mesh = field.mesh
    geometry = field.geometry
    operator = force_operator(field)
    solver = EllipticSolver(operator; preconditioner = :tree_geometric)
    predictor = proj_one_predictor(field)
    velocity = proj_copy(predictor)
    pressure = CellField(mesh)
    work = ProjectionWorkspace(solver)
    stats = isnothing(geometry) ?
        project!(velocity, pressure, solver, PROJ_DT, work; tolerance = PROJ_TOL_ONE, max_cycles = 1000) :
        project!(velocity, pressure, solver, PROJ_DT, work, geometry; tolerance = PROJ_TOL_ONE, max_cycles = 1000)
    @assert stats.converged
    divergence = CellField(mesh)
    isnothing(geometry) ? divergence!(divergence, predictor) : divergence!(divergence, predictor, geometry)
    initial = maximum(abs(AMR._get(divergence, ref)) for ref in all_cells(mesh))
    isnothing(geometry) ? divergence!(divergence, velocity) : divergence!(divergence, velocity, geometry)
    references = operator.references
    volumes = operator.volumes
    identity = 0.0
    final = 0.0
    for (index, ref) in pairs(references)
        volumes[index] > 0 || continue
        d = AMR._get(divergence, ref)
        final = max(final, abs(d))
        expected = -PROJ_DT * (solver.residual[index] / volumes[index] + stats.removed_rhs_mean)
        identity = max(identity, abs(d - expected))
    end
    gamma = operator.face_coefficient
    energy_star, factor = proj_kinetic_norm(field, operator, gamma, predictor.interior)
    energy_end, _ = proj_kinetic_norm(field, operator, gamma, velocity.interior; factor)
    energy_change, _ = proj_kinetic_norm(field, operator, gamma, velocity.interior .- predictor.interior; factor)
    mean = sum(volumes[i] * AMR._get(pressure, references[i]) for i in eachindex(volumes))
    scale = sum(volumes[i] * abs(AMR._get(pressure, references[i])) for i in eachindex(volumes))
    return (;
        identity = identity / initial,
        defect = PROJ_DT * final, target = stats.final_linf / stats.residual_scale,
        energy = abs(energy_star - energy_end - energy_change) / energy_star,
        removed = energy_change / energy_star,
        mean = abs(mean) / scale,
    )
end

# A channel with a parabolic inflow and an Outflow side: compare the volume
# defect dt |D u| next to the outflow with the interior defect.
function outflow_check()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i == 3,
    )
    profile(y) = 4y * (1 - y)
    boundary = VelocityBoundaryConditions(;
        left = Inflow((x, y, t, axis) -> axis == 1 ? profile(y) : 0.0),
        right = Outflow(), bottom = NoSlip(), top = NoSlip(),
    )
    velocity = FaceVelocity(
        mesh, (x, y, t, axis) -> axis == 1 ?
            profile(y) + 0.3sinpi(2x) * sinpi(y) : 0.2sinpi(x) * sinpi(2y),
    )
    solver = EllipticSolver(
        PoissonOperator(mesh; velocity_boundary = boundary); preconditioner = :tree_geometric,
    )
    pressure = CellField(mesh)
    enforce_velocity_boundary!(velocity, boundary; time = 0.0)
    initial = proj_copy(velocity)
    divergence = CellField(mesh)
    divergence!(divergence, initial)
    predictor = maximum(abs(AMR._get(divergence, ref)) for ref in all_cells(mesh))
    work = ProjectionWorkspace(solver)
    stats = project!(
        velocity, pressure, solver, PROJ_DT, work;
        velocity_boundary = boundary, tolerance = PROJ_TOL_ONE, max_cycles = 1000,
    )
    @assert stats.converged
    divergence!(divergence, velocity)
    outflow_cells = Set(face.cell for face in mesh.boundary_faces if face.axis == 1 && face.side > 0)
    near = maximum(abs(AMR._get(divergence, ref)) for ref in outflow_cells)
    other = maximum(abs(AMR._get(divergence, ref)) for ref in all_cells(mesh) if !(ref in outflow_cells))
    target = stats.final_linf / stats.residual_scale
    repeated = [PROJ_DT * max(near, other)]
    for _ in 1:4
        repeat_stats = project!(
            velocity, CellField(mesh), solver, PROJ_DT, work;
            velocity_boundary = boundary, tolerance = PROJ_TOL_ONE, max_cycles = 1000,
        )
        @assert repeat_stats.converged
        divergence!(divergence, velocity)
        push!(repeated, PROJ_DT * maximum(abs(AMR._get(divergence, ref)) for ref in all_cells(mesh)))
    end
    return (;
        outflow = PROJ_DT * near, interior = PROJ_DT * other, predictor = PROJ_DT * predictor,
        cells = length(outflow_cells), target, repeated,
    )
end

# --- Two velocities: face data and mixture divergence (Section "two") ----------------------

function proj_face_data(field)
    liquid_aperture, gas_aperture = AMR._current_phase_face_apertures!(field)
    mesh = field.mesh
    n = length(mesh.faces)
    C = [staggered_fraction_independent(field, face) for face in mesh.faces]
    ml = FORCE_DENSITY[1] .* C
    mg = FORCE_DENSITY[2] .* (1 .- C)
    m = ml .+ mg
    coupling = (liquid_aperture .* (-mg) .+ gas_aperture .* ml) ./ m
    floor = field.fraction_floor
    mixed = [f for f in 1:n if floor < C[f] < 1 - floor && liquid_aperture[f] + gas_aperture[f] > 0]
    return (; C, ml, mg, m, coupling, liquid_aperture, gas_aperture, mixed)
end

# Independent |c^o| D^mix: sum over the faces of a cell of o |f| (a_l u_l + a_g u_g).
function proj_mixture_divergence(field, velocity)
    mesh = field.mesh
    liquid_aperture, gas_aperture = AMR._current_phase_face_apertures!(field)
    flux = Dict{CellRef, Float64}()
    product(a, u) = iszero(a) ? 0.0 : a * u
    for (index, face) in pairs(mesh.faces)
        value = face.area * (
            product(liquid_aperture[index], velocity[1].interior[index]) +
                product(gas_aperture[index], velocity[2].interior[index])
        )
        flux[face.minus] = get(flux, face.minus, 0.0) + value
        flux[face.plus] = get(flux, face.plus, 0.0) - value
    end
    for (index, face) in pairs(mesh.boundary_faces)
        liquid, gas = AMR._phase_boundary_face_apertures(field, index)
        value = face.side * face.area * (
            product(liquid, velocity[1].boundary[index]) + product(gas, velocity[2].boundary[index])
        )
        flux[face.cell] = get(flux, face.cell, 0.0) + value
    end
    result = 0.0
    for (ref, value) in flux
        volume = proj_open_volume(field, ref)
        volume > 0 || continue
        result = max(result, abs(value) / volume)
    end
    return result
end

# --- CMDGFM stencils, independently (Section "cmdgfm") --------------------------------------

function proj_cell_normal(field, ref)
    h = cell_spacing(field.mesh, ref)
    n, _ = liquid_plane(field, ref)
    v = (n[1] / h[1], n[2] / h[2])
    m = hypot(v...)
    return m > sqrt(eps()) ? (v[1] / m, v[2] / m) : (0.0, 0.0)
end

# Staggered normal: c(1-c)-weighted mean of the two cell normals, else the
# fraction difference along the face normal.
function proj_staggered_normal(field, face)
    cm, cp = cond(field, face.minus), cond(field, face.plus)
    wm, wp = cm * (1 - cm), cp * (1 - cp)
    nm, np = proj_cell_normal(field, face.minus), proj_cell_normal(field, face.plus)
    v = (wm * nm[1] + wp * np[1], wm * nm[2] + wp * np[2])
    if iszero(wm + wp) || hypot(v...) <= sqrt(eps()) * (wm + wp)
        d = cm - cp
        v = face.axis == 1 ? (d, 0.0) : (0.0, d)
    end
    m = hypot(v...)
    return m > sqrt(eps()) ? (v[1] / m, v[2] / m) : nothing
end

# Faces of axis `axis` in the vertex patches of the two end points of f (not f).
function proj_patch_faces(mesh, patches, index, axis)
    face = mesh.faces[index]
    result = Int[]
    for endpoint in 1:2
        for candidate in get(patches, AMR._vertex_key(mesh, face, endpoint), [])
            g = candidate.index
            g == index && continue
            mesh.faces[g].axis == axis || continue
            g in result || push!(result, g)
        end
    end
    return sort!(result)
end

# The classification and stencil of every mixed face: (:closed, :normal_neighbour,
# :no_normal, :transverse, :parallel, :dominance) or :resolved with its stencil.
function cmdgfm_stencils_independent(field, data)
    mesh = field.mesh
    patches = AMR._vertex_patches(mesh, Float64)
    floor = field.fraction_floor
    active(g) = floor < data.C[g] < 1 - floor
    open(g) = data.liquid_aperture[g] + data.gas_aperture[g]
    aperture_mixed(g) = data.liquid_aperture[g] > floor && data.gas_aperture[g] > floor
    incident = Dict{CellRef, Vector{Int}}()
    for (g, face) in pairs(mesh.faces)
        push!(get!(incident, face.minus, Int[]), g)
        push!(get!(incident, face.plus, Int[]), g)
    end
    result = Dict{Int, Any}()
    for f in data.mixed
        face = mesh.faces[f]
        if open(f) <= floor
            result[f] = (; class = :closed)
            continue
        end
        if !aperture_mixed(f)
            neighbours = [g for cell in (face.minus, face.plus) for g in incident[cell] if g != f && mesh.faces[g].axis == face.axis]
            if any(aperture_mixed, neighbours)
                result[f] = (; class = :normal_neighbour)
                continue
            end
        end
        normal = proj_staggered_normal(field, face)
        k, t = face.axis, 3 - face.axis
        if isnothing(normal) || abs(normal[k]) <= sqrt(eps())
            result[f] = (; class = :no_normal)
            continue
        end
        ratio = normal[t] / normal[k]
        if abs(normal[k]) >= 1 - 64eps()
            result[f] = (; class = :resolved, ratio, beta = 1.0, transverse = Int[], weights = Float64[], gas = 0, liquid = 0, full = true)
            continue
        end
        transverse = filter(g -> active(g) && open(g) > floor, proj_patch_faces(mesh, patches, f, t))
        weights = [data.C[g] * (1 - data.C[g]) for g in transverse]
        boundary = any(e -> AMR._physical_vertex(mesh, AMR._vertex_key(mesh, face, e)), 1:2)
        level = any(g -> mesh.faces[g].area != face.area, transverse)
        required = boundary || level ? 1 : 2
        if length(transverse) < required || sum(weights; init = 0.0) <= 0
            result[f] = (; class = :transverse)
            continue
        end
        weights ./= sum(weights)
        beta, gas, liquid = 1.0, 0, 0
        if abs(ratio) > 1
            parallel = proj_patch_faces(mesh, patches, f, k)
            isnothing(field.geometry) || filter!(g -> field.geometry.face_dual_volumes[g] > 0, parallel)
            if length(parallel) < 2
                result[f] = (; class = :parallel)
                continue
            end
            sort!(parallel; by = g -> (data.ml[g] / data.m[g], g))
            gas, liquid = first(parallel), last(parallel)
            if data.mg[gas] < data.ml[gas] || data.ml[liquid] < data.mg[liquid]
                result[f] = (; class = :dominance)
                continue
            end
            s = data.ml[gas] / data.m[gas] + data.mg[liquid] / data.m[liquid]
            beta = clamp((1 - s) / (abs(ratio) - s), 0.0, 1.0)
        end
        result[f] = (; class = :resolved, ratio, beta, transverse, weights, gas, liquid, full = false)
    end
    return result
end

# The discrete normal jump N_f(u) of a resolved row.
function proj_normal_jump(stencil, f, velocity)
    jump(g) = velocity[2].interior[g] - velocity[1].interior[g]
    value = jump(f)
    for (g, w) in zip(stencil.transverse, stencil.weights)
        value += stencil.beta * stencil.ratio * w * jump(g)
    end
    if stencil.gas > 0
        value += (stencil.beta - 1) * (velocity[2].interior[stencil.gas] - velocity[1].interior[stencil.liquid])
    end
    return value
end

function cmdgfm_check(kind::Symbol; solid::Bool)
    field = projection_fixture(kind; solid)
    mesh = field.mesh
    solver = CMDGFMProjectionSolver(field; density = FORCE_DENSITY)
    work = CMDGFMProjectionWorkspace(solver)
    code_data = AMR._projection_face_data(solver, work)
    stencils = Dict(s.face => s for s in AMR._cmdgfm_jump_stencils(solver, code_data))
    data = proj_face_data(field)
    @assert sort(data.mixed) == sort(collect(code_data.mixed))
    independent = cmdgfm_stencils_independent(field, data)
    classes = Dict{Symbol, Int}()
    beta_difference = 0.0
    ratio_difference = 0.0
    weight_difference = 0.0
    for (f, s) in independent
        classes[s.class] = get(classes, s.class, 0) + 1
        code = stencils[f]
        @assert code.resolved == (s.class == :resolved)
        s.class == :resolved || continue
        @assert code.full_remove == s.full && code.transverse_faces == s.transverse
        @assert code.gas_inference_face == s.gas && code.liquid_inference_face == s.liquid
        beta_difference = max(beta_difference, abs(code.beta - s.beta))
        ratio_difference = max(ratio_difference, abs(code.ratio - s.ratio) / max(1, abs(s.ratio)))
        isempty(s.weights) || (weight_difference = max(weight_difference, maximum(abs, code.transverse_weights .- s.weights)))
    end
    resolved = [f for (f, s) in independent if s.class == :resolved]
    inferred = count(f -> independent[f].gas > 0, resolved)
    # Diagonal dominance of the assembled jump block (eq. dominance).
    velocity = proj_two_predictor(field)
    assembled = AMR._assemble_cmdgfm_system!(solver, work, PROJ_DT, velocity)
    B = assembled.jump_block
    margins = zeros(size(B, 1))
    for column in axes(B, 2), pointer in nzrange(B, column)
        row = rowvals(B)[pointer]
        value = abs(nonzeros(B)[pointer])
        margins[row] += row == column ? value : -value
    end
    tight = count(<=(64eps()), margins)
    # Projection: row conditions and mixture divergence of the end-of-step velocity.
    pressure = CellField(mesh)
    stats = project!(velocity, pressure, solver, PROJ_DT, work; tolerance = PROJ_TOL_TWO, max_iterations = 4000)
    @assert stats.converged
    row_defect = maximum(
        PROJ_DT * abs(proj_normal_jump(independent[f], f, velocity)) /
            min(
            minimum(cell_spacing(mesh, mesh.faces[f].minus)),
            minimum(cell_spacing(mesh, mesh.faces[f].plus)),
        )
            for f in resolved
    )
    volume_defect = PROJ_DT * proj_mixture_divergence(field, velocity)
    # Jump and momentum identities of the phase gradients of the solution.
    gradients = (FaceVelocity(mesh), FaceVelocity(mesh))
    cmdgfm_gradient!(gradients, pressure, solver, work)
    jump_identity = 0.0
    momentum_identity = 0.0
    for f in eachindex(mesh.faces)
        jump_identity = max(jump_identity, abs(gradients[2].interior[f] - gradients[1].interior[f] - work.jump[f]))
        momentum_identity = max(
            momentum_identity,
            abs(data.ml[f] * gradients[1].interior[f] + data.mg[f] * gradients[2].interior[f] - data.m[f] * work.pressure_gradient.interior[f]) / data.m[f],
        )
    end
    return (;
        classes, beta_difference, ratio_difference, weight_difference, inferred,
        resolved = length(resolved), mixed = length(data.mixed),
        margin = minimum(margins; init = Inf), tight, rows = length(margins),
        row_defect, code_row_defect = stats.normal_velocity_defect,
        volume_defect, code_volume_defect = stats.mixture_volume_defect,
        jump_identity, momentum_identity, iterations = stats.iterations,
        unknowns = (stats.pressure_unknowns, stats.jump_unknowns),
    )
end

# The velocity of the regrid projection does not depend on dt (only its
# defect gauge does): project the same field with dt and 3 dt.
function regrid_scaling_check()
    field = projection_fixture(:amr; solid = true)
    results = map((PROJ_DT, 3PROJ_DT)) do dt
        solver = CMDGFMProjectionSolver(field; density = FORCE_DENSITY)
        velocity = proj_two_predictor(field)
        stats = project!(velocity, CellField(field.mesh), solver, dt, CMDGFMProjectionWorkspace(solver); tolerance = 1.0e-13, max_iterations = 4000)
        @assert stats.converged
        velocity
    end
    difference = 0.0
    scale = 0.0
    for phase in 1:2, f in eachindex(results[1][phase].interior)
        a, b = results[1][phase].interior[f], results[2][phase].interior[f]
        isnan(a) && continue
        difference = max(difference, abs(a - b))
        scale = max(scale, abs(a))
    end
    return difference / scale
end

# --- Figure data: CMDGFM rows ---------------------------------------------------------------

const PROJ_WINDOWS = ((3.25, 5.0, 3.0, 4.25),)   # x0, x1, y0, y1
const PROJ_STENCIL_FACE = (4.5, 3.875, 1)    # beta < 1 row: centre, axis

proj_face_at(mesh, x, y, axis) = findfirst(
    face -> face.axis == axis && hypot(face.center[1] - x, face.center[2] - y) < 1.0e-9, mesh.faces,
)
proj_inside(window, x, y; margin = 1.0e-9) =
    window[1] - margin <= x <= window[2] + margin && window[3] - margin <= y <= window[4] + margin
proj_face_line(face) = "{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}{$(face.axis)}{$(fmt(face.area))}"

# Cells, solid, liquid and PLIC of a window:
#   \fxpcell{panel}{x0}{y0}{x1}{y1}{level}{kind}  kind 0 uncut, 1 cut, 2 covered
#   \fxpsolid{panel}{path}  \fxpwall{panel}{x1}{y1}{x2}{y2}
#   \fxpliquid{panel}{path}  \fxpplic{panel}{x1}{y1}{x2}{y2}
function proj_state_text(io, field, panel, window)
    mesh = field.mesh
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        in_window(box, window) || continue
        alpha = solid_fraction_of(field, ref)
        kind = alpha <= 0 ? 0 : alpha >= 1 ? 2 : 1
        println(io, "\\fxpcell{$panel}{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}{$(cell_level(mesh, ref))}{$kind}")
        kind == 2 && continue
        open = open_part(field, ref, SQUARE)
        if kind == 1
            n, a = solid_plane_of(field, ref)
            println(io, "\\fxpsolid{$panel}{$(polygon_text(to_physical(mesh, ref, clip_halfplane(SQUARE, n, a; keep_less = true))))}")
            p, q = to_physical(mesh, ref, segment_in(SQUARE, n, a))
            println(io, "\\fxpwall{$panel}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
        end
        c = cond(field, ref)
        c <= 0 && continue
        if c >= 1
            println(io, "\\fxpliquid{$panel}{$(polygon_text(to_physical(mesh, ref, open)))}")
            continue
        end
        n, a = liquid_plane(field, ref)
        println(io, "\\fxpliquid{$panel}{$(polygon_text(to_physical(mesh, ref, clip_halfplane(open, n, a; keep_less = true))))}")
        p, q = to_physical(mesh, ref, segment_in(open, n, a))
        println(io, "\\fxpplic{$panel}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
    end
    return
end

#   \fxpclass{x}{y}{axis}{length}{class}    class 0: beta = 1, 1: beta < 1, 2: unresolved
#   \fxpstencil{x}{y}{axis}{length}{role}   role 0: f, 1: transverse, 2: h^g, 3: h^l
#   \fxpnormal{x}{y}{nx}{ny}                staggered normal of f
function projection_data()
    field = projection_fixture(:amr; solid = true)
    mesh = field.mesh
    io = IOBuffer()
    for (panel, window) in pairs(PROJ_WINDOWS)
        proj_state_text(io, field, panel, window)
    end
    solver = CMDGFMProjectionSolver(field; density = FORCE_DENSITY)
    work = CMDGFMProjectionWorkspace(solver)
    data = AMR._projection_face_data(solver, work)
    stencils = Dict(s.face => s for s in AMR._cmdgfm_jump_stencils(solver, data))
    for (f, s) in stencils
        face = mesh.faces[f]
        proj_inside(PROJ_WINDOWS[1], face.center...) || continue
        class = !s.resolved ? 2 : s.beta < 1 ? 1 : 0
        println(io, "\\fxpclass", proj_face_line(face), "{$class}")
    end
    f = proj_face_at(mesh, PROJ_STENCIL_FACE...)
    s = stencils[f]
    @assert s.resolved && s.beta < 1 && s.gas_inference_face > 0
    println(io, "\\fxpstencil", proj_face_line(mesh.faces[f]), "{0}")
    for h in s.transverse_faces
        println(io, "\\fxpstencil", proj_face_line(mesh.faces[h]), "{1}")
    end
    println(io, "\\fxpstencil", proj_face_line(mesh.faces[s.gas_inference_face]), "{2}")
    println(io, "\\fxpstencil", proj_face_line(mesh.faces[s.liquid_inference_face]), "{3}")
    normal = proj_staggered_normal(field, mesh.faces[f])
    println(io, "\\fxpnormal{$(fmt(mesh.faces[f].center[1]))}{$(fmt(mesh.faces[f].center[2]))}{$(fmt(normal[1]))}{$(fmt(normal[2]))}")
    stencil_info = (; beta = s.beta, ratio = s.ratio, weights = s.transverse_weights)
    return String(take!(io)), stencil_info
end

# --- Outputs -------------------------------------------------------------------

function projection_outputs(write_output)
    figure, figure_info = projection_data()
    write_output("projection.tex", figure)
    fixtures = ((:uniform, false), (:uniform, true), (:amr, false), (:amr, true))
    one = [one_velocity_check(kind; solid) for (kind, solid) in fixtures]
    outflow = outflow_check()
    cmdgfm = [cmdgfm_check(kind; solid) for (kind, solid) in fixtures]
    scaling = regrid_scaling_check()

    println("projection one velocity:     ", one)
    println("projection outflow:          ", outflow)
    println("projection cmdgfm:           ", cmdgfm)
    println("projection regrid scaling:   ", scaling)
    println("projection figure:           ", figure_info)

    @assert maximum(r.identity for r in one) <= 1.0e-14
    @assert maximum(r.energy for r in one) <= 1.0e-10
    @assert maximum(r.mean for r in one) <= 1.0e-14
    @assert maximum(r.defect for r in one) <= PROJ_TOL_ONE
    @assert outflow.interior <= PROJ_TOL_ONE && outflow.outflow <= PROJ_TOL_ONE
    @assert outflow.outflow <= 10outflow.interior
    @assert all(value -> value <= PROJ_TOL_ONE, outflow.repeated)
    for r in cmdgfm
        @assert r.beta_difference <= 1.0e-14 && r.ratio_difference <= 1.0e-14 && r.weight_difference <= 1.0e-14
        @assert r.margin >= -1.0e-14 && r.tight > 0
        @assert r.row_defect <= PROJ_TOL_TWO && r.volume_defect <= PROJ_TOL_TWO
        @assert abs(r.row_defect - r.code_row_defect) <= 1.0e-15 && abs(r.volume_defect - r.code_volume_defect) <= 1.0e-15
    end
    @assert scaling <= 1.0e-10

    percent(x) = @sprintf("%.0f", 100x)
    row(values) = join(values, " & ")
    unresolved(r, class) = get(r.classes, class, 0)
    return Any[
        ("chkProjOneIdentity", sci(maximum(r.identity for r in one))),
        ("chkProjOneEnergy", sci(maximum(r.energy for r in one))),
        ("chkProjOneRemovedMin", percent(minimum(r.removed for r in one))),
        ("chkProjOneRemovedMax", percent(maximum(r.removed for r in one))),
        ("chkProjOneMean", sci(maximum(r.mean for r in one))),
        ("chkProjOneDefect", sci(maximum(r.defect for r in one))),
        ("chkProjOutflow", sci(outflow.outflow)),
        ("chkProjOutflowInterior", sci(outflow.interior)),
        ("chkProjOutflowPredictor", sci(outflow.predictor)),
        ("chkProjOutflowCells", outflow.cells),
        ("chkProjOutflowRepeated", join((sci(value) for value in outflow.repeated[2:end]), ", ")),
        ("chkCmdStencilDiff", sci(maximum(max(r.beta_difference, r.ratio_difference, r.weight_difference) for r in cmdgfm))),
        ("chkCmdMargin", sci(minimum(r.margin for r in cmdgfm))),
        ("chkCmdRowDefect", sci(maximum(r.row_defect for r in cmdgfm))),
        ("chkCmdVolumeDefect", sci(maximum(r.volume_defect for r in cmdgfm))),
        ("chkCmdJumpIdentity", sci(maximum(r.jump_identity for r in cmdgfm))),
        ("chkCmdMomentumIdentity", sci(maximum(r.momentum_identity for r in cmdgfm))),
        ("chkRegridScaling", sci(scaling)),
        ("chkFigBeta", @sprintf("%.3f", figure_info.beta)),
        ("chkFigRatio", @sprintf("%.2f", figure_info.ratio)),
        # Table rows: the four fixtures in the order of `fixtures`.
        ("chkTabMixed", row(r.mixed for r in cmdgfm)),
        ("chkTabResolved", row(r.resolved for r in cmdgfm)),
        ("chkTabInferred", row(r.inferred for r in cmdgfm)),
        ("chkTabTight", row(r.tight for r in cmdgfm)),
        ("chkTabNeighbour", row(unresolved(r, :normal_neighbour) for r in cmdgfm)),
        ("chkTabOtherUnresolved", row(r.mixed - r.resolved - unresolved(r, :normal_neighbour) for r in cmdgfm)),
        ("chkTabCmdIterations", row(r.iterations for r in cmdgfm)),
    ]
end
