# Chapter 6 (staggered momentum transport): figure data and spot checks.
# Included by generate.jl after vof.jl, whose fixtures, streamfunction
# velocities and cell helpers (all_cells, cell_volume, open_fraction_of,
# aperture_of, cond, wrap) it reuses. The functions marked "independent"
# rebuild the dual schedule from the incidence rules of Chapter 6 and evaluate
# the face states and the staggered updates without the package's schedule,
# state or update kernels; the differences to the code are the numbers quoted
# in the text, and the assertions fail if they disagree.

# --- Fixtures ------------------------------------------------------------------

# Open-solenoidal velocity on a cut mesh from a vertex streamfunction psi that
# vanishes on solid vertices (level set `solid` <= 0): the open flux
# |f| a_f u_f = psi(end) - psi(start) telescopes around every open cell.
function solid_streamfunction_velocity(mesh, geometry, solid, psi)
    value(x, y) = solid(x, y) <= 0 ? 0.0 : psi(x, y)
    velocity = FaceVelocity(mesh)
    for (index, face) in pairs(mesh.faces)
        x, y = face.center
        h = face.area / 2
        flux = face.axis == 1 ? value(x, y + h) - value(x, y - h) :
            -(value(x + h, y) - value(x - h, y))
        a = isnothing(geometry) ? 1.0 : geometry.aperture[index]
        @assert a > 0 || abs(flux) <= 1.0e-14
        velocity.interior[index] = a > 0 ? flux / (face.area * a) : 0.0
    end
    return velocity
end

# A solid disc across the level transition x = 8 of the periodic composite
# mesh of Chapter 5 (vof_amr_mesh): coarse cut cells whose sides carry two
# fine segments occur. The disc is tangent to no grid line.
const MOM_DISC = (; centre = (8.07, 6.33), radius = 1.55)
mom_disc(x, y) = hypot(x - MOM_DISC.centre[1], y - MOM_DISC.centre[2]) - MOM_DISC.radius
mom_open_psi(x, y) = vof_psi_periodic(x, y)

function momentum_fixture(kind::Symbol)
    if kind == :uniform
        mesh = vof_uniform_mesh()
        geometry = nothing
    elseif kind == :amr
        mesh = vof_amr_mesh()
        geometry = nothing
    elseif kind == :cut
        mesh = vof_amr_mesh()
        geometry = GeometryField(mesh, mom_disc; lipschitz = 1)
    else
        error("unknown fixture $kind")
    end
    velocity = isnothing(geometry) ? streamfunction_velocity(mesh, vof_psi_periodic) :
        solid_streamfunction_velocity(mesh, geometry, mom_disc, mom_open_psi)
    isnothing(geometry) || enforce_geometry_no_slip!(velocity, geometry)
    return mesh, geometry, velocity
end

mom_field(x, y) = 1 + 0.4sinpi(2x / VOF_BOX) * cospi(4y / VOF_BOX) + 0.2cospi(6x / VOF_BOX)

# --- Independent dual schedule (Section "dual") ------------------------------------

# Degrees of freedom of component k: the interior k-faces in the order of
# mesh.faces, then (phase schedule of bounded domains) the boundary k-faces.
function component_dofs(mesh, k; boundary = false)
    interior = [index for (index, face) in pairs(mesh.faces) if face.axis == k]
    local_of = Dict(global_index => l for (l, global_index) in enumerate(interior))
    boundary_faces = boundary ?
        [index for (index, face) in pairs(mesh.boundary_faces) if face.axis == k] : Int[]
    boundary_of = Dict(b => length(interior) + l for (l, b) in enumerate(boundary_faces))
    centres = [mesh.faces[index].center for index in interior]
    append!(centres, (mesh.boundary_faces[b].center for b in boundary_faces))
    return (; k, interior, local_of, boundary_faces, boundary_of, centres, n = length(centres))
end

# Side incidences: per cell and side s = -1 (low), +1 (high) the dofs on that
# side with the fraction varpi_{c,g} = |g| / h_{c,t} of the side they cover.
function side_incidences(mesh, dofs)
    k = dofs.k
    t = 3 - k
    sides = Dict{CellRef, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    entry(cell) = get!(sides, cell) do
        Dict(-1 => Tuple{Int, Float64}[], 1 => Tuple{Int, Float64}[])
    end
    for (l, index) in enumerate(dofs.interior)
        face = mesh.faces[index]
        for (cell, s) in ((face.minus, 1), (face.plus, -1))
            push!(entry(cell)[s], (l, face.area / mesh.blocks[cell.block].spacing[t]))
        end
    end
    for (b, l) in dofs.boundary_of
        face = mesh.boundary_faces[b]
        push!(entry(face.cell)[face.side], (l, face.area / mesh.blocks[face.cell.block].spacing[t]))
    end
    return sides
end

# Rules (N) and (T) of Chapter 6: weight w_{g,f} of the primal flux of f in the
# dual face g joining dofs (i, j), i < j, oriented from i to j. With a
# `geometry`, the side fractions of the segments of a hanging side of a cut
# cell are replaced by the open-strip fractions (one-velocity momentum).
function independent_dual_weights(mesh, k; boundary = false, geometry = nothing)
    dofs = component_dofs(mesh, k; boundary)
    sides = side_incidences(mesh, dofs)
    isnothing(geometry) || open_strip_sides!(sides, mesh, geometry, dofs)
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    function add!(from, to, axis, flux, w)
        @assert from != to
        key = from < to ? (from, to, axis) : (to, from, axis)
        d = get!(weights, key, Dict{Int, Float64}())
        d[flux] = get(d, flux, 0.0) + (from < to ? w : -w)
        return nothing
    end
    empty_side = Dict(-1 => Tuple{Int, Float64}[], 1 => Tuple{Int, Float64}[])
    for (index, face) in pairs(mesh.faces)
        if face.axis == k
            l = dofs.local_of[index]
            for (cell, sigma) in ((face.minus, 1.0), (face.plus, -1.0)), s in (-1, 1)
                for (g, varpi) in get(sides, cell, empty_side)[s]
                    g == l || add!(g, l, k, index, sigma * varpi / 2)
                end
            end
        else
            for s in (-1, 1)
                for (g, a) in get(sides, face.minus, empty_side)[s],
                        (gp, ap) in get(sides, face.plus, empty_side)[s]
                    add!(g, gp, face.axis, index, a * ap / 2)
                end
            end
        end
    end
    return weights, dofs, sides
end

# Open-strip fractions |Lambda_{c,f} cap c^o| / |c^o| (independent clipping)
# in place of varpi_{c,f} < 1 in every cut cell 0 < alpha^o_c < 1: the
# weights of the one-velocity momentum schedule (issue 63).
function open_strip_sides!(sides, mesh, geometry, dofs)
    n = length(dofs.interior)
    for (cell, entry) in sides
        open = open_fraction_of_geometry(geometry, cell)
        0 < open < 1 || continue
        for s in (-1, 1)
            entry[s] = map(entry[s]) do (g, varpi)
                varpi < 1 || return (g, varpi)
                face = g <= n ? mesh.faces[dofs.interior[g]] :
                    mesh.boundary_faces[dofs.boundary_faces[g - n]]
                return (g, strip_open_area(geometry, cell, face)[1] / open)
            end
        end
    end
    return sides
end

# The code's schedule in the same form (orientation lower index -> higher).
function code_dual_weights(schedule::AMR.AMRDualAdvectionSchedule)
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    for face in schedule.faces
        @assert face.minus < face.plus
        d = Dict{Int, Float64}()
        for p in face.first_weight:(face.first_weight + face.weight_count - 1)
            d[schedule.flux_indices[p]] = schedule.flux_weights[p]
        end
        weights[(face.minus, face.plus, face.axis)] = d
    end
    return weights
end

function code_dual_weights(schedule::AMR.UniformDualAdvectionSchedule, mesh)
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    for face in schedule.faces
        sign = face.minus < face.plus ? 1.0 : -1.0
        key = (min(face.minus, face.plus), max(face.minus, face.plus), face.axis)
        d = get!(weights, key, Dict{Int, Float64}())
        for flux in (face.flux_a, face.flux_b)
            d[flux] = get(d, flux, 0.0) + sign * face.area / (2mesh.faces[flux].area)
        end
    end
    return weights
end

function weight_difference(a, b; scale = Dict{Int, Float64}())
    keys(a) == keys(b) || return Inf
    worst = 0.0
    for (key, d) in a
        e = b[key]
        for flux in union(keys(d), keys(e))
            worst = max(worst, abs(get(d, flux, 0.0) * get(scale, flux, 1.0) - get(e, flux, 0.0)))
        end
    end
    return worst
end

# Open part |Lambda_{c,f} cap c^o| / |c| of the strip of `ref` behind a k-face
# (full width of the cell normal to f, tangential extent of f), by independent
# clipping with the solid half-plane (cutcells.jl); varpi_{c,f} without solid.
function strip_open_fraction(mesh, geometry, ref, face)
    varpi = face.area / mesh.blocks[ref.block].spacing[3 - face.axis]
    isnothing(geometry) && return varpi
    return strip_open_area(geometry, ref, face)[1]
end

# Interpolation weights (1/2)|Lambda_{c,f} cap c^o| of the cells of every dof.
function interpolation_weights(mesh, geometry, dofs)
    weights = Vector{Vector{Tuple{CellRef, Float64}}}(undef, dofs.n)
    for (l, index) in enumerate(dofs.interior)
        face = mesh.faces[index]
        weights[l] = [
            (cell, strip_open_fraction(mesh, geometry, cell, face) * mesh.blocks[cell.block].volume / 2)
                for cell in (face.minus, face.plus)
        ]
    end
    for (b, l) in dofs.boundary_of
        face = mesh.boundary_faces[b]
        weights[l] = [
            (
                face.cell,
                strip_open_fraction(mesh, geometry, face.cell, face) *
                    mesh.blocks[face.cell.block].volume / 2,
            ),
        ]
    end
    return weights
end

# Transport volumes V_f of the dofs: the sum of the interpolation weights,
# i.e. |omega_f| without solid and |omega_f^o| with one (a zero replaced by one).
function independent_volumes(weights)
    V = [sum(w for (_, w) in entry; init = 0.0) for entry in weights]
    return [v > 0 ? v : 1.0 for v in V]
end

# Interpolation (H p)_f of a cell function p onto the dofs; the boundary dofs
# of the phase schedule take the value of their cell.
function independent_interpolation(weights, value)
    return map(weights) do entry
        den = sum(w for (_, w) in entry; init = 0.0)
        length(entry) == 1 ? value(entry[1][1]) :
            den > 0 ? sum(w * value(cell) for (cell, w) in entry) / den : 0.0
    end
end

# --- Independent face states (Section "states") -----------------------------------

independent_limiter(r, ::Fromm) = (1 + r) / 2
independent_limiter(r, ::Minmod) = max(0.0, min(1.0, r))
independent_limiter(r, ::MonotonizedCentral) = max(0.0, min(2r, (1 + r) / 2, 2.0))
independent_limiter(r, ::VanLeer) = r <= 0 ? 0.0 : isinf(r) ? 2.0 : 2r / (1 + r)
independent_limiter(r, ::Superbee) = max(0.0, min(2r, 1.0), min(r, 2.0))

const THREE_POINT = Union{Fromm, Minmod, MonotonizedCentral, VanLeer, Superbee}

# Upstream neighbour of the upwind dof u of a dual face (u, d) of axis kappa:
# among the dofs joined to u by another dual face of axis kappa and lying beyond
# u (away from d) along kappa, the one nearest to the reflection of d through u.
# Returns (dof, h^d / h^uu); (0, 0) if there is none or if u, d share x_kappa.
function independent_upstream(mesh, dofs, adjacency, u, d, kappa)
    disp(a, b) = (
        wrap(dofs.centres[b][1] - dofs.centres[a][1], mesh.extent[1], mesh.periodic[1]),
        wrap(dofs.centres[b][2] - dofs.centres[a][2], mesh.extent[2], mesh.periodic[2]),
    )
    t = 3 - kappa
    dd = disp(u, d)
    tol = 64eps() * max(1.0, mesh.extent[kappa])
    abs(dd[kappa]) > tol || return (0, 0.0)
    best, best_distance, best_axial = 0, Inf, 0.0
    for c in get(adjacency, (u, kappa), Int[])
        c == d && continue
        cd = disp(u, c)
        -sign(dd[kappa]) * cd[kappa] > tol || continue
        distance = (cd[kappa] + dd[kappa])^2 + (cd[t] + dd[t])^2
        if distance < best_distance || (distance == best_distance && c < best)
            best, best_distance, best_axial = c, distance, abs(cd[kappa])
        end
    end
    best == 0 && return (0, 0.0)
    return (best, abs(dd[kappa]) / best_axial)
end

function dual_adjacency(keys_list)
    adjacency = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (i, j, axis) in keys_list
        push!(get!(adjacency, (i, axis), Int[]), j)
        push!(get!(adjacency, (j, axis), Int[]), i)
    end
    return adjacency
end

# Smoothness ratio with the tolerance rule of Chapter 5, then the limited state.
function independent_three_point(qu, qd, quu, ratio, high, scheme)
    up, down = qu - quu, qd - qu
    tol = 16eps() * max(abs(quu), abs(qu), abs(qd), 1.0)
    theta = abs(down) <= tol ? (abs(up) <= tol ? 1.0 : 0.0) : ratio * up / down
    return qu + independent_limiter(theta, scheme) * (high - qu)
end

# Face state of one dual face g = (m, p) with rate (or time-integrated volume)
# `rate`; `nu` is the Courant number, `correction` the divergence term of the
# single-phase Lax--Wendroff state (zero for phase transport).
function independent_state(scheme, q, m, p, rate, nu, correction, upstream)
    positive = rate >= 0
    qu, qd = positive ? (q[m], q[p]) : (q[p], q[m])
    scheme isa Upwind && return qu
    scheme isa Centred && return (q[m] + q[p]) / 2
    lw = ((1 - nu) * qd + (1 + nu) * qu) / 2 - correction
    scheme isa LaxWendroff && return lw
    j, ratio = positive ? upstream[1] : upstream[2]
    j == 0 && return qu
    return independent_three_point(qu, qd, q[j], ratio, lw, scheme)
end

# --- Independent single-phase convection fluxes ------------------------------------

# Dual rates Upsilon_g = sum_f w_{g,f} |f| a_f u_f of the faces in `weights`.
function independent_rates(mesh, geometry, weights, velocity)
    return Dict(
        key => sum(
            w * mesh.faces[f].area * aperture_of_geometry(geometry, f) * velocity.interior[f]
                for (f, w) in d; init = 0.0
        ) for (key, d) in weights
    )
end
aperture_of_geometry(geometry, index) = isnothing(geometry) ? 1.0 : geometry.aperture[index]

# Regular dual faces: two weights of magnitude 1/2 and equal uncut volumes.
function independent_regular(weights, uncut_volumes)
    return Dict(
        key => length(d) == 2 && all(w -> abs(abs(w) - 0.5) <= 64eps(), values(d)) &&
            abs(uncut_volumes[key[1]] - uncut_volumes[key[2]]) <= 64eps() * max(1.0, uncut_volumes[key[1]])
            for (key, d) in weights
    )
end

function face_class(mesh, geometry, dofs, key, d, regular)
    if !isnothing(geometry)
        cut(i) = i <= length(dofs.interior) && any(
            cell -> open_fraction_of_geometry(geometry, cell) < 1,
            (mesh.faces[dofs.interior[i]].minus, mesh.faces[dofs.interior[i]].plus),
        )
        (any(f -> geometry.aperture[f] < 1, keys(d)) || cut(key[1]) || cut(key[2])) && return :cut
    end
    return regular[key] ? :uniform : :coarse_fine
end
open_fraction_of_geometry(geometry, ref) = AMR._get(geometry.open_fraction, ref)

function staggered_workspace(field, geometry)
    return isnothing(geometry) ? StaggeredAdvectionWorkspace(field) :
        StaggeredAdvectionWorkspace(field, geometry)
end

# One directional sweep of the convection operator: the code's dual fluxes
# (workspace.dual_flux) against the independent Upsilon_g (F q)_g.
function convection_flux_check(kind; schemes = (Upwind(), Centred(), LaxWendroff(), MonotonizedCentral(), VanLeer()))
    mesh, geometry, velocity = momentum_fixture(kind)
    worst = Dict(:uniform => 0.0, :coarse_fine => 0.0, :cut => 0.0)
    count = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    schedule_difference = identity_residual = 0.0
    for k in (1, 2)
        field = FaceField{k}(mesh, mom_field)
        ws = staggered_workspace(field, geometry)
        weights, dofs, _ = independent_dual_weights(mesh, k)
        code = ws.schedule isa AMR.UniformDualAdvectionSchedule ?
            code_dual_weights(ws.schedule, mesh) : code_dual_weights(ws.schedule)
        scale = isnothing(geometry) ? Dict{Int, Float64}() :
            Dict(f => geometry.aperture[f] for f in eachindex(mesh.faces))
        schedule_difference = max(schedule_difference, weight_difference(weights, code; scale))
        uncut = [mesh.face_dual_volumes[index] for index in dofs.interior]
        V = independent_volumes(interpolation_weights(mesh, geometry, dofs))
        regular = independent_regular(weights, uncut)
        rates = independent_rates(mesh, geometry, weights, velocity)
        # Factorisation identity: dual divergence = sum_c varpi/2 (primal divergence).
        dual_div = zeros(dofs.n)
        for ((i, j, _), r) in rates
            dual_div[i] += r; dual_div[j] -= r
        end
        sides = side_incidences(mesh, dofs)
        primal = Dict{CellRef, Float64}()
        for (index, face) in pairs(mesh.faces)
            F = face.area * aperture_of_geometry(geometry, index) * velocity.interior[index]
            primal[face.minus] = get(primal, face.minus, 0.0) + F
            primal[face.plus] = get(primal, face.plus, 0.0) - F
        end
        expected = zeros(dofs.n)
        for (cell, entry) in sides, s in (-1, 1), (g, varpi) in entry[s]
            expected[g] += varpi / 2 * get(primal, cell, 0.0)
        end
        identity_residual = max(identity_residual, maximum(abs, dual_div .- expected))
        adjacency = dual_adjacency(collect(keys(weights)))
        # Map schedule faces to keys (orientation sign for the uniform schedule).
        face_keys = [
            (min(f.minus, f.plus), max(f.minus, f.plus), f.axis, f.minus < f.plus ? 1.0 : -1.0)
                for f in ws.schedule.faces
        ]
        dt = stable_timestep(velocity, ws; cfl = 0.5)
        for kappa in (1, 2)
            D = zeros(dofs.n)
            for ((i, j, axis), r) in rates
                axis == kappa || continue
                D[i] += r / V[i]; D[j] -= r / V[j]
            end
            q = field.interior
            for (e, (i, j, axis, _)) in enumerate(face_keys)
                axis == kappa || continue
                key = (i, j, axis)
                count[face_class(mesh, geometry, dofs, key, weights[key], regular)] += 1
            end
            for scheme in schemes
                interface_scheme = default_amr_interface_scheme(scheme)
                destination = FaceField{k}(mesh)
                AMR._dispatch_convection!(destination, field, velocity, dt, ws, scheme, interface_scheme, kappa, nothing)
                scale_flux = maximum(abs, ws.dual_flux)
                split = AMR._directionally_split_scheme(scheme, interface_scheme, ws.schedule)
                for (e, (i, j, axis, sign)) in enumerate(face_keys)
                    axis == kappa || continue
                    key = (i, j, axis)
                    r = rates[key]
                    selected = regular[key] ? scheme : interface_scheme
                    nu = abs(dt * r) / min(V[i], V[j])
                    correction = split ? dt * (q[i] * D[i] + q[j] * D[j]) / 4 : 0.0
                    upstream = selected isa THREE_POINT ? (
                            independent_upstream(mesh, dofs, adjacency, i, j, axis),
                            independent_upstream(mesh, dofs, adjacency, j, i, axis),
                        ) : ((0, 0.0), (0, 0.0))
                    state = independent_state(selected, q, i, j, r, nu, correction, upstream)
                    class = face_class(mesh, geometry, dofs, key, weights[key], regular)
                    worst[class] = max(worst[class], abs(sign * ws.dual_flux[e] - r * state) / scale_flux)
                end
            end
        end
    end
    return (; worst, count, schedule_difference, identity_residual)
end

# --- Conservation and kinetic energy of the single-phase operator ---------------

staggered_energy(V, q) = sum(V .* q .^ 2) / 2

# Momentum sum_f V_f q_f before and after one advect! per scheme; the all-centred
# operator at dt = 0 assembled column by column: weighted skew-symmetry, twice
# its diagonal against the dual divergence, and one forward-Euler step with the
# exact energy identity E1 - E0 = dt^2 |C q|_V^2 / 2; energy change of one
# default (Lax--Wendroff) step.
function convection_conservation_check(kind; schemes = (Upwind(), LaxWendroff(), MonotonizedCentral(), Superbee()))
    mesh, geometry, velocity = momentum_fixture(kind)
    momentum = skew = diagonal = euler = 0.0
    lw_energy = Inf
    for k in (1, 2)
        dofs = component_dofs(mesh, k)
        V = independent_volumes(interpolation_weights(mesh, geometry, dofs))
        for scheme in schemes
            field = FaceField{k}(mesh, mom_field)
            ws = staggered_workspace(field, geometry)
            dt = stable_timestep(velocity, ws; cfl = 0.5)
            before = sum(V .* field.interior)
            E0 = staggered_energy(V, field.interior)
            advect!(field, velocity, dt, ws; scheme)
            momentum = max(momentum, abs(sum(V .* field.interior) - before) / sum(V .* abs.(field.interior)))
            scheme isa LaxWendroff && (lw_energy = min(lw_energy, (staggered_energy(V, field.interior) - E0) / E0))
        end
        basis = FaceField{k}(mesh)
        column = FaceField{k}(mesh)
        ws = staggered_workspace(basis, geometry)
        n = dofs.n
        C = zeros(n, n)
        for source in 1:n
            fill!(basis.interior, 0.0)
            basis.interior[source] = 1.0
            convection!(column, basis, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred())
            C[:, source] .= column.interior
        end
        VC = V .* C
        skew = max(skew, maximum(abs, VC .+ transpose(VC)) / maximum(abs, VC))
        weights, _, _ = independent_dual_weights(mesh, k)
        dual_div = zeros(n)
        for ((i, j, _), r) in independent_rates(mesh, geometry, weights, velocity)
            dual_div[i] += r / V[i]; dual_div[j] -= r / V[j]
        end
        rate_scale = maximum(abs, C)
        diagonal = max(diagonal, maximum(abs, 2 .* [C[i, i] for i in 1:n] .- dual_div) / rate_scale)
        field = FaceField{k}(mesh, mom_field)
        dt = stable_timestep(velocity, ws; cfl = 0.5)
        q0 = copy(field.interior)
        Cq = C * q0
        advect!(field, velocity, dt, ws; scheme = Centred(), amr_interface_scheme = Centred())
        E0, E1 = staggered_energy(V, q0), staggered_energy(V, field.interior)
        growth = dt^2 * sum(V .* Cq .^ 2) / 2
        euler = max(euler, abs(E1 - E0 - growth) / growth)
    end
    return (; momentum, skew, diagonal, euler, lw_energy)
end

# Boundary dual faces of a no-slip wall on three bounded meshes: a composite
# mesh, a composite mesh whose refined root touches a wall (kind B3) and a
# uniform mesh with a sloped solid wall (cut cells next to the walls).
# (1) Closed identity: V_f C(u)1 (centred, dt = 0, boundary values one,
#     FreeSlip) against the independent right-hand side
#     sum_c varpi_{c,f}/2 sum_{f' in F(c)} o_{c,f'} |f'| a_{f'} u_{f'} with the
#     boundary faces in F(c), for a velocity that is neither solenoidal nor zero
#     on the boundary; relative to the largest sum of absolute terms.
# (2) |C(u)1| for an open-solenoidal u with no-slip walls, relative to the
#     largest rate of the CFL bound, and the number of unknowns above 1e-12.
# (3) The all-centred operator C with zero boundary values, assembled column by
#     column for a velocity with zero boundary values that is not solenoidal:
#     |VC + (VC)^T - diag(V C(u)1)| relative to the largest entry of VC.
function boundary_identity_check(kind)
    if kind == :walls
        mesh = BlockTreeMesh(;
            base_mesh = (12, 12), block_size = (4, 4), extent = (VOF_BOX, VOF_BOX),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key == BlockKey(0, 1, 1),
        )
        geometry = nothing
    elseif kind == :amr_walls
        mesh = BlockTreeMesh(;
            base_mesh = (12, 12), block_size = (4, 4), extent = (VOF_BOX, VOF_BOX),
            periodic = (false, false), max_level = 1,
            refine = (key, origin, extent) -> key == BlockKey(0, 0, 1),
        )
        geometry = nothing
    elseif kind == :cut_walls
        mesh = vof_uniform_mesh_walls()
        geometry = GeometryField(mesh, vof_wall; lipschitz = hypot(1, VOF_WALL.slope))
    end
    solenoidal = solid_streamfunction_velocity(
        mesh, geometry, isnothing(geometry) ? ((x, y) -> 1.0) : vof_wall, vof_psi_walls,
    )
    arbitrary = FaceVelocity(
        mesh, (x, y, t, axis) -> axis == 1 ?
            0.2 + 0.1x / VOF_BOX - 0.05y / VOF_BOX : -0.1 + 0.08x / VOF_BOX + 0.12y / VOF_BOX,
    )
    interior_only = copy(arbitrary)
    fill!(interior_only.boundary, 0.0)
    aperture(index) = isnothing(geometry) ? 1.0 : geometry.aperture[index]
    boundary_aperture(index) = isnothing(geometry) ? 1.0 : geometry.boundary_aperture[index]
    # Net outflow sum_{f' in F(c)} o_{c,f'} |f'| a_{f'} u_{f'} and its absolute counterpart.
    outflow = Dict{CellRef, Float64}(); outflow_abs = Dict{CellRef, Float64}()
    for ref in all_cells(mesh)
        outflow[ref] = outflow_abs[ref] = 0.0
    end
    for (index, face) in pairs(mesh.faces)
        F = face.area * aperture(index) * arbitrary.interior[index]
        outflow[face.minus] += F; outflow[face.plus] -= F
        outflow_abs[face.minus] += abs(F); outflow_abs[face.plus] += abs(F)
    end
    for (index, face) in pairs(mesh.boundary_faces)
        F = face.area * boundary_aperture(index) * arbitrary.boundary[index]
        outflow[face.cell] += face.side * F; outflow_abs[face.cell] += abs(F)
    end
    closed = worst = skew = 0.0
    where = (0.0, 0.0)
    count = 0
    kinds = Dict(:B1 => 0, :B2 => 0, :B3 => 0)
    for k in (1, 2)
        dofs = component_dofs(mesh, k)
        V = independent_volumes(interpolation_weights(mesh, geometry, dofs))
        one = FaceField{k}(mesh; init = 1)
        destination = FaceField{k}(mesh)
        ws = staggered_workspace(one, geometry)
        for c in ws.boundary_faces
            kinds[!c.normal_component ? :B2 : c.wall_axis == k ? :B1 : :B3] += 1
        end
        # (1)
        convection!(destination, one, arbitrary, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = FreeSlip())
        expected = zeros(dofs.n); scale = zeros(dofs.n)
        for (l, index) in enumerate(dofs.interior)
            face = mesh.faces[index]
            for cell in (face.minus, face.plus)
                varpi = face.area / mesh.blocks[cell.block].spacing[3 - k]
                expected[l] += varpi / 2 * outflow[cell]
                scale[l] += varpi / 2 * outflow_abs[cell]
            end
        end
        closed = max(closed, maximum(abs, V .* destination.interior .- expected) / maximum(scale))
        # (2)
        convection!(destination, one, solenoidal, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
        rates = FaceField{k}(mesh)
        AMR._accumulate_staggered_rates!(rates.interior, solenoidal, ws)
        rate_scale = maximum(rates.interior)
        for (i, value) in pairs(destination.interior)
            relative = abs(value) / rate_scale
            relative > 1.0e-12 && (count += 1)
            relative > worst && (worst = relative; where = dofs.centres[i])
        end
        # (3)
        convection!(destination, one, interior_only, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
        theta = copy(destination.interior)
        basis = FaceField{k}(mesh)
        C = zeros(dofs.n, dofs.n)
        for source in 1:dofs.n
            fill!(basis.interior, 0.0)
            basis.interior[source] = 1.0
            convection!(destination, basis, interior_only, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred(), boundary = NoSlip())
            C[:, source] .= destination.interior
        end
        VC = V .* C
        symmetric = VC .+ transpose(VC)
        for i in 1:dofs.n
            symmetric[i, i] -= V[i] * theta[i]
        end
        skew = max(skew, maximum(abs, symmetric) / maximum(abs, VC))
    end
    return (; closed, worst, where, count, skew, kinds)
end

vof_uniform_mesh_walls() = BlockTreeMesh(;
    base_mesh = (12, 12), block_size = (12, 12), extent = (VOF_BOX, VOF_BOX),
    periodic = (false, false),
)

# The code's phase schedules against the replica, on the walled composite
# fixture (boundary dofs) and the cut fixture (phase-scalar and one-velocity
# momentum schedules): weights, the regularity flag (eq:momentum:regular, a
# dual face incident to a boundary dof is non-regular) and the upstream dofs of
# the three-point states. Counts, per fixture and for the phase-scalar
# schedule of both components, the regular dual faces (those next to a cut
# cell separately), the non-regular ones between interior dofs and those
# incident to a boundary dof.
function phase_classification_check()
    counts = Dict{Symbol, Dict{Symbol, Int}}()
    mismatched = stencils = 0
    weights_difference = 0.0
    for kind in (:walls, :cut)
        field, _ = two_phase_fixture(kind)
        mesh, geometry = field.mesh, field.geometry
        momentum = MomentumAdvectionWorkspace(field)
        tally = Dict(:regular => 0, :cut_regular => 0, :nonregular => 0, :wall => 0)
        for k in (1, 2)
            component = k == 1 ? momentum.x : momentum.y
            scalar_schedule = isnothing(geometry) ?
                AMR._phase_transport_connectivity(mesh, Val(k), Float64).schedule :
                AMR._geometry_phase_schedule(geometry, Val(k))
            for (is_momentum, code) in ((false, scalar_schedule), (true, component.advection.schedule))
                s = phase_schedule(mesh, geometry, k; momentum = is_momentum)
                weights_difference = max(weights_difference, weight_difference(s.weights, code_dual_weights(code)))
                n = length(s.dofs.interior)
                cut(i) = !isnothing(geometry) && any(
                    ((cell, _),) -> 0 < open_fraction_of_geometry(geometry, cell) < 1, s.interpolation[i],
                )
                for face in code.faces
                    key = (face.minus, face.plus, face.axis)
                    mismatched += face.refinement_interface == s.regular[key]
                    for (u, d, stencil) in ((face.minus, face.plus, face.minus_stencil), (face.plus, face.minus, face.plus_stencil))
                        mine = independent_upstream(mesh, s.dofs, s.adjacency, u, d, face.axis)
                        stencils += 1
                        mismatched += !(
                            mine[1] == stencil.upstream &&
                                (iszero(mine[1]) || isapprox(mine[2], stencil.ratio_scale; rtol = 1.0e-14))
                        )
                    end
                    is_momentum && continue
                    if face.refinement_interface
                        tally[(face.minus > n || face.plus > n) ? :wall : :nonregular] += 1
                    else
                        tally[(cut(face.minus) || cut(face.plus)) ? :cut_regular : :regular] += 1
                    end
                end
            end
        end
        counts[kind] = tally
    end
    return (; counts, mismatched, stencils, weights_difference)
end

# --- Two-phase staggered transport: independent replica -------------------------

# Two-phase fixtures: the drop of Chapter 5 on the uniform and composite
# periodic meshes; a gas bubble across the transition x = 4 of the composite
# mesh with the solid disc of `momentum_fixture(:cut)` in the liquid (cut
# coarse cells next to fine cells occur at x = 8); the drop on the composite
# mesh with walls on all sides (boundary dofs of the phase schedule).
const MOM_BUBBLE = (; centre = (2.55, 5.95), radius = 1.62)
mom_bubble(x, y) = MOM_BUBBLE.radius - hypot(x - MOM_BUBBLE.centre[1], y - MOM_BUBBLE.centre[2])

function two_phase_fixture(kind::Symbol)
    if kind in (:uniform, :amr)
        return vof_fixture(kind)
    elseif kind == :cut
        mesh = vof_amr_mesh()
        geometry = GeometryField(mesh, mom_disc; lipschitz = 1)
        field = InterfaceField(mesh, mom_bubble; geometry, lipschitz = 1)
        velocity = solid_streamfunction_velocity(mesh, geometry, mom_disc, mom_open_psi)
        enforce_geometry_no_slip!(velocity, geometry)
        return field, velocity
    elseif kind == :walls
        mesh = vof_amr_mesh(; periodic = (false, false))
        field = InterfaceField(mesh, vof_drop; lipschitz = 1)
        return field, streamfunction_velocity(mesh, vof_psi_walls)
    end
    error("unknown fixture $kind")
end

# Phase schedule of the code: the general dual schedule on fully periodic
# meshes without solid; otherwise the boundary faces are dofs as well (open
# volumes with a solid). On every schedule a dual face is regular by the uncut
# weights and volumes unless it joins a boundary dof, and the three-point
# states find their upstream dof among all dofs, boundary dofs included
# (issue 62). `momentum = true`: the one-velocity momentum schedule, whose
# weights use the open-strip fractions of cut cells (issue 63).
function phase_schedule(mesh, geometry, k; momentum = false)
    bounded = !all(mesh.periodic) || !isnothing(geometry)
    weights, dofs, sides = independent_dual_weights(mesh, k; boundary = bounded)
    uncut_weights = weights
    if momentum && !isnothing(geometry)
        weights, _, sides = independent_dual_weights(mesh, k; boundary = bounded, geometry)
    end
    interpolation = interpolation_weights(mesh, geometry, dofs)
    V = independent_volumes(interpolation)
    uncut = independent_volumes(interpolation_weights(mesh, nothing, dofs))
    regular = independent_regular(uncut_weights, uncut)
    n = length(dofs.interior)
    for key in keys(regular)
        (key[1] > n || key[2] > n) && (regular[key] = false)
    end
    adjacency = dual_adjacency(collect(keys(weights)))
    return (; bounded, weights, dofs, sides, interpolation, V, regular, adjacency)
end

# Phase-volume rates of a sweep through the dual faces of axis kappa:
# Upsilon_g = sum_f w_{g,f} (dt |f| upsilon_f) with the code's primal volumes.
function phase_dual_volumes(schedule, volumes, kappa)
    return Dict(
        key => sum(w * volumes[f] for (f, w) in d; init = 0.0)
            for (key, d) in schedule.weights if key[3] == kappa
    )
end

# Fluctuation increments sum_g o_{f,g} (Upsilon (q_f - (F q)_g)) / V_f of one
# phase: CTU (term-wise upwinding of the partial volumes) where a predicted
# staggered fraction of the face is below the threshold, else the bulk state.
function phase_increment(mesh, schedule, q, volumes, kappa, predicted, threshold, scheme, interface_scheme)
    increment = zeros(length(q))
    V = schedule.V
    for (key, d) in schedule.weights
        i, j, axis = key
        axis == kappa || continue
        if min(predicted[i], predicted[j]) < threshold
            di = dj = 0.0
            for (f, w) in d
                partial = w * volumes[f]
                up = partial >= 0 ? q[i] : q[j]
                di += partial * (q[i] - up); dj += partial * (q[j] - up)
            end
        else
            flux = sum(w * volumes[f] for (f, w) in d; init = 0.0)
            selected = schedule.regular[key] ? scheme : interface_scheme
            nu = abs(flux) / min(V[i], V[j])
            upstream = selected isa THREE_POINT ? (
                    independent_upstream(mesh, schedule.dofs, schedule.adjacency, i, j, axis),
                    independent_upstream(mesh, schedule.dofs, schedule.adjacency, j, i, axis),
                ) : ((0, 0.0), (0, 0.0))
            state = independent_state(selected, q, i, j, flux, nu, 0.0, upstream)
            di, dj = flux * (q[i] - state), flux * (q[j] - state)
        end
        increment[i] += di / V[i]
        increment[j] -= dj / V[j]
    end
    return increment
end

# Cell quantities of one sweep from the code's time-integrated primal volumes:
# open-carrier dilation D_c and liquid increment of the fraction update.
function sweep_cell_terms(field, work, kappa, chi)
    mesh = field.mesh
    D = Dict{CellRef, Float64}(); dC = Dict{CellRef, Float64}()
    for ref in all_cells(mesh)
        D[ref] = dC[ref] = 0.0
    end
    for (index, face) in pairs(mesh.faces)
        face.axis == kappa || continue
        for (cell, o) in ((face.minus, 1.0), (face.plus, -1.0))
            open = open_volume_of(field, cell)
            open > 0 || continue
            D[cell] += o * work.open_flux[index] / open
            dC[cell] += o * (chi[cell] * work.open_flux[index] - work.liquid_flux[index]) / open
        end
    end
    return D, dC
end

component_values(velocity, dofs) = [
    [velocity.interior[i] for i in dofs.interior];
    [velocity.boundary[b] for b in dofs.boundary_faces]
]

snap_fraction(c, floor) = (s = clamp(c, 0.0, 1.0); s <= floor ? 0.0 : s >= 1 - floor ? 1.0 : s)

# One-velocity momentum: replica of the staggered update (density-weighted
# fluctuation form) for both components, driven by the code's primal volumes,
# against the code's velocity after advect!; momentum sum_f V_f rho_f u_f
# before and after; largest defect of the staggered mass identity per sweep,
# separately at the dofs where condition (b) fails (a cut cell with two
# segments on the side of the dof), and there also with the side-fraction
# weights of the phase-scalar schedule instead of the open-strip ones.
function one_velocity_momentum_check(kind; scheme = LaxWendroff(), order = (1, 2), density = (1000.0, 1.0))
    field, velocity = two_phase_fixture(kind)
    reconstruct!(field)
    mesh, geometry = field.mesh, field.geometry
    work = WYAdvectionWorkspace(field)
    dt = stable_timestep(velocity, work)
    interface_scheme = default_amr_interface_scheme(scheme)
    cells = all_cells(mesh)
    C0 = Dict(ref => cond(field, ref) for ref in cells)
    chi = Dict(ref => C0[ref] > 0.5 ? 1.0 : 0.0 for ref in cells)
    floor = field.fraction_floor
    rho_l, rho_g = density
    schedules = [phase_schedule(mesh, geometry, k; momentum = true) for k in (1, 2)]
    scalar_schedules = isnothing(geometry) ? schedules : [phase_schedule(mesh, geometry, k) for k in (1, 2)]
    values = [component_values(velocity, schedules[k].dofs) for k in (1, 2)]
    sources = deepcopy(values)
    mixture(C) = rho_l * C + rho_g * (1 - C)
    momentum_of(C, q, s) = sum(s.V .* mixture.(independent_interpolation(s.interpolation, r -> C[r])) .* q)
    before = sum(momentum_of(C0, values[k], schedules[k]) for k in (1, 2))
    destination = FaceVelocity(mesh)
    advect!(
        field, velocity, dt, work; momentum = destination, density,
        momentum_workspace = MomentumAdvectionWorkspace(field), sweep_order = order,
        scheme, amr_interface_scheme = interface_scheme,
    )
    C = copy(C0)
    mass_defect = mass_defect_partial = scalar_defect_partial = 0.0
    snaps = partial_dofs = 0
    for kappa in order
        D, dC = sweep_cell_terms(field, work, kappa, chi)
        predicted = Dict(ref => clamp(C[ref] + dC[ref], 0.0, 1.0) for ref in cells)
        dilation = Dict(ref => (rho_l * chi[ref] + rho_g * (1 - chi[ref])) * D[ref] for ref in cells)
        for k in (1, 2)
            s = schedules[k]
            old = independent_interpolation(s.interpolation, r -> C[r])
            new = independent_interpolation(s.interpolation, r -> predicted[r])
            R = independent_interpolation(s.interpolation, r -> dilation[r])
            delta = zeros(s.dofs.n)
            mass = zeros(s.dofs.n)
            scalar_mass = zeros(s.dofs.n)
            for (rho, flux, liquid) in ((rho_l, work.liquid_flux, true), (rho_g, work.gas_flux, false))
                phase_new = liquid ? new : 1 .- new
                delta .+= rho .* phase_increment(
                    mesh, s, values[k], flux, kappa, phase_new, work.ctu_threshold, scheme, interface_scheme,
                )
                for ((i, j, axis), r) in phase_dual_volumes(s, flux, kappa)
                    mass[i] += rho * r / s.V[i]; mass[j] -= rho * r / s.V[j]
                end
                for ((i, j, axis), r) in phase_dual_volumes(scalar_schedules[k], flux, kappa)
                    scalar_mass[i] += rho * r / s.V[i]; scalar_mass[j] -= rho * r / s.V[j]
                end
            end
            # Staggered mass identity: rho(new) - rho(old) + div(rho Upsilon) - R = 0,
            # separately where every adjacent strip is consistent (uncut cell or a
            # strip over a whole side, condition (b)) and elsewhere.
            defect = abs.(mixture.(new) .- mixture.(old) .+ mass .- R) ./ rho_l
            scalar_defect = abs.(mixture.(new) .- mixture.(old) .+ scalar_mass .- R) ./ rho_l
            for i in eachindex(defect)
                consistent = all(
                    ((cell, w),) -> isnothing(geometry) || open_fraction_of_geometry(geometry, cell) >= 1 ||
                        abs(
                        strip_open_fraction(
                            mesh, nothing, cell, i <= length(s.dofs.interior) ?
                                mesh.faces[s.dofs.interior[i]] : mesh.boundary_faces[s.dofs.boundary_faces[i - length(s.dofs.interior)]]
                        ) - 1
                    ) <= 64eps(),
                    s.interpolation[i],
                )
                if consistent
                    mass_defect = max(mass_defect, defect[i])
                else
                    mass_defect_partial = max(mass_defect_partial, defect[i])
                    scalar_defect_partial = max(scalar_defect_partial, scalar_defect[i])
                    partial_dofs += 1
                end
            end
            q = values[k]
            for i in eachindex(q)
                q[i] += (delta[i] + R[i] * (sources[k][i] - q[i])) / mixture(new[i])
            end
        end
        for ref in cells
            raw = C[ref] + dC[ref]
            C[ref] = snap_fraction(raw, floor)
            abs(C[ref] - raw) > 256eps() && (snaps += 1)
        end
    end
    replica = 0.0
    scale = maximum(abs, velocity.interior)
    for k in (1, 2)
        code = component_values(destination, schedules[k].dofs)
        replica = max(replica, maximum(abs, code .- values[k]) / scale)
    end
    Cn = Dict(ref => cond(field, ref) for ref in cells)
    after = sum(momentum_of(Cn, component_values(destination, schedules[k].dofs), schedules[k]) for k in (1, 2))
    total = sum(
        sum(
            schedules[k].V .* mixture.(independent_interpolation(schedules[k].interpolation, r -> C0[r])) .*
                abs.(component_values(velocity, schedules[k].dofs))
        ) for k in (1, 2)
    )
    return (;
        dt, replica, momentum = abs(after - before) / total, mass_defect,
        mass_defect_partial, scalar_defect_partial, partial_dofs, snaps,
    )
end

# Absent intrinsic values (no open staggered volume of the phase) become NaN and
# are then extended by two averaging passes over the dual faces; zero beyond.
function prepared_phase_values(field, s, velocity, liquid)
    mesh, geometry = field.mesh, field.geometry
    q = component_values(velocity, s.dofs)
    fraction = clamp.(independent_interpolation(s.interpolation, r -> cond(field, r)), 0.0, 1.0)
    physical = [sum(w for (_, w) in entry; init = 0.0) for entry in s.interpolation]
    for i in eachindex(q)
        open = isnothing(geometry) || physical[i] > 0
        present = open && (liquid ? fraction[i] > 0 : fraction[i] < 1)
        present || (q[i] = NaN)
    end
    neighbours = [Int[] for _ in eachindex(q)]
    for (i, j, _) in keys(s.weights)
        push!(neighbours[i], j); push!(neighbours[j], i)
    end
    for _ in 1:2
        next = copy(q)
        for i in eachindex(q)
            isfinite(q[i]) && continue
            finite = [q[j] for j in neighbours[i] if isfinite(q[j])]
            next[i] = isempty(finite) ? NaN : sum(finite) / length(finite)
        end
        q = next
    end
    return [isnan(v) ? 0.0 : v for v in q]
end

# Two velocities: replica of the intrinsic fluctuation update of both phases
# (PhaseFaceScalar) for both components, against the code; phase amounts
# sum_f V_f C_f^phase u_f^phase before and after.
function two_velocity_momentum_check(kind; scheme = LaxWendroff(), order = (1, 2), density = (1000.0, 1.0))
    field, liquid_velocity = two_phase_fixture(kind)
    gas_velocity = kind in (:uniform, :amr) ? vof_gas_velocity(field, kind) :
        kind == :walls ? streamfunction_velocity(field.mesh, vof_psi2_walls) :
        solid_streamfunction_velocity(field.mesh, field.geometry, mom_disc, (x, y) -> vof_psi2_periodic(x, y))
    velocity = (liquid_velocity, gas_velocity)
    all(field.mesh.periodic) || enforce_no_permeation!(velocity)
    isnothing(field.geometry) || foreach(v -> enforce_geometry_no_slip!(v, field.geometry), velocity)
    reconstruct!(field)
    mesh, geometry = field.mesh, field.geometry
    work = WYAdvectionWorkspace(field, velocity)
    dt = stable_timestep(velocity, work)
    interface_scheme = default_amr_interface_scheme(scheme)
    cells = all_cells(mesh)
    C0 = Dict(ref => cond(field, ref) for ref in cells)
    chi = Dict(ref => C0[ref] > 0.5 ? 1.0 : 0.0 for ref in cells)
    floor = field.fraction_floor
    schedules = [phase_schedule(mesh, geometry, k) for k in (1, 2)]
    values = [[prepared_phase_values(field, schedules[k], velocity[p], p == 1) for k in (1, 2)] for p in 1:2]
    sources = deepcopy(values)
    open(s, i) = isnothing(geometry) || sum(w for (_, w) in s.interpolation[i]; init = 0.0) > 0
    amount(C, q, s, liquid) = begin
        f = independent_interpolation(s.interpolation, r -> C[r])
        w = liquid ? f : 1 .- f
        sum(w[i] == 0 || !open(s, i) ? 0.0 : s.V[i] * w[i] * q[i] for i in eachindex(q))
    end
    before = [sum(amount(C0, values[p][k], schedules[k], p == 1) for k in (1, 2)) for p in 1:2]
    scale = [sum(sum(schedules[k].V .* abs.(values[p][k])) for k in (1, 2)) for p in 1:2]
    destination = (FaceVelocity(mesh), FaceVelocity(mesh))
    advect!(
        field, velocity, dt, work; momentum = destination, density,
        momentum_workspace = MomentumAdvectionWorkspace(field, velocity), sweep_order = order,
        scheme, amr_interface_scheme = interface_scheme,
    )
    C = copy(C0)
    dilation = [0.0, 0.0]
    snaps = 0
    for kappa in order
        D, dC = sweep_cell_terms(field, work, kappa, chi)
        for k in (1, 2)
            s = schedules[k]
            liquid_new = independent_interpolation(s.interpolation, r -> clamp(C[r] + dC[r], 0.0, 1.0))
            gas_new = 1 .- liquid_new
            for (p, flux, target) in ((1, work.liquid_flux, liquid_new), (2, work.gas_flux, gas_new))
                rates = independent_interpolation(
                    s.interpolation, r -> (p == 1 ? chi[r] : 1 - chi[r]) * D[r],
                )
                q = values[p][k]
                increment = phase_increment(mesh, s, q, flux, kappa, target, work.ctu_threshold, scheme, interface_scheme)
                for i in eachindex(q)
                    target[i] > floor && (q[i] += (increment[i] + (sources[p][k][i] - q[i]) * rates[i]) / target[i])
                end
                dilation[p] += sum(s.V .* rates .* sources[p][k])
            end
        end
        for ref in cells
            raw = C[ref] + dC[ref]
            C[ref] = snap_fraction(raw, floor)
            abs(C[ref] - raw) > 256eps() && (snaps += 1)
        end
    end
    replica = 0.0
    nan_pattern = true
    for p in 1:2, k in (1, 2)
        code = component_values(destination[p], schedules[k].dofs)
        for i in eachindex(code)
            if isnan(code[i])
                fraction = independent_interpolation(schedules[k].interpolation, r -> C[r])[i]
                nan_pattern &= !open(schedules[k], i) || (p == 1 ? fraction <= 0 : fraction >= 1)
            else
                replica = max(replica, abs(code[i] - values[p][k][i]) / maximum(abs, velocity[p].interior))
            end
        end
    end
    Cn = Dict(ref => cond(field, ref) for ref in cells)
    after = [
        sum(amount(Cn, component_values(destination[p], schedules[k].dofs), schedules[k], p == 1) for k in (1, 2))
            for p in 1:2
    ]
    # Balance: change of the phase amount = frozen dilation.
    balance = [
        abs(after[1] - before[1] - dilation[1]) / scale[1],
        abs(density[2] * (after[2] - before[2] - dilation[2])) / (density[2] * scale[2]),
    ]
    return (;
        dt, replica, nan_pattern, snaps,
        liquid = abs(after[1] - before[1]) / scale[1], gas = abs(after[2] - before[2]) / scale[2],
        balance, dilation, gas_change = density[2] * (after[2] - before[2]),
    )
end

# --- Figure data ---------------------------------------------------------------------

# Dual faces of one dof from the code's schedule: (other dof, axis, weights
# keyed by primal face index, interface flag), oriented from the dof with the
# smaller centre (x first, then y) to the other, as drawn in the figure.
function code_connections(ws, dof, centres)
    s = ws.schedule
    result = Any[]
    for face in s.faces
        (face.minus == dof || face.plus == dof) || continue
        other = face.minus == dof ? face.plus : face.minus
        sign = centres[face.minus] < centres[face.plus] ? 1.0 : -1.0
        weights = if s isa AMR.AMRDualAdvectionSchedule
            Dict(
                s.flux_indices[p] => sign * s.flux_weights[p]
                    for p in face.first_weight:(face.first_weight + face.weight_count - 1)
            )
        else
            Dict(face.flux_a => sign / 2, face.flux_b => sign / 2)
        end
        flag = s isa AMR.AMRDualAdvectionSchedule ? face.refinement_interface : false
        push!(result, (; other, axis = face.axis, weights, flag))
    end
    return result
end

face_index_at(mesh, centre) = findfirst(f -> all(isapprox.(f.center, centre; atol = 1.0e-12)), mesh.faces)
component_index(mesh, k, index) = count(i -> mesh.faces[i].axis == k, 1:index)

function weight_label(weights, names)
    terms = sort!(collect(weights); by = p -> names[p[1]])
    text = ""
    for (n, (index, w)) in enumerate(terms)
        sign = w < 0 ? "-" : (n == 1 ? "" : "+")
        text *= "$sign$(frac(abs(w)))\\,$(names[index])"
    end
    return text
end

# Figure "dual": (a) a face of the uniform periodic mesh, (b) a fine segment of
# a hanging coarse side of the composite periodic mesh, (c) a face between two
# cut cells of the disc fixture. Component 1 throughout.
#   \fxmcell{panel}{x0}{y0}{x1}{y1}{level}{kind}   kind 0 uncut, 1 cut, 2 covered
#   \fxmsolid{panel}{path}  \fxmwall{panel}{x1}{y1}{x2}{y2}
#   \fxmdual{panel}{x0}{y0}{x1}{y1}{highlight}     uncut staggered control volume
#   \fxmopen{panel}{path}                           open part of a strip of f
#   \fxmface{panel}{axis}{x}{y}{length}{name}{aperture}
#   \fxmlink{panel}{x1}{y1}{x2}{y2}{axis}{flag}{label}{bend}
#   \fxmdualface{panel}{x1}{y1}{x2}{y2}{label}{side}  dual face segment, side l/r/b/t
#   \fxmhanging{panel}{x}{y}                        hanging vertex
const DUAL_WINDOWS = (;
    uniform = (4.6, 7.4, 4.3, 6.7), coarse_fine = (2.45, 5.05, 3.85, 7.15),
    cut = (6.3, 7.7, 4.72, 5.78),
)

function dual_cells_text(io, mesh, geometry, window, panel)
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        in_window(box, window) || continue
        alpha = isnothing(geometry) ? 0.0 : solid_fraction(geometry, ref)
        kind = alpha <= 0 ? 0 : alpha >= 1 ? 2 : 1
        println(
            io, "\\fxmcell{$panel}{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}",
            "{$(cell_level(mesh, ref))}{$kind}",
        )
        if kind == 1
            m1, m2, a = solid_plane(geometry, ref)
            solid = clip_halfplane(SQUARE, (m1, m2), a; keep_less = true)
            println(io, "\\fxmsolid{$panel}{$(polygon_text(to_physical(mesh, ref, solid)))}")
            s, e = to_physical(mesh, ref, segment_in(SQUARE, (m1, m2), a))
            println(io, "\\fxmwall{$panel}{$(fmt(s[1]))}{$(fmt(s[2]))}{$(fmt(e[1]))}{$(fmt(e[2]))}")
        elseif kind == 2
            println(io, "\\fxmsolid{$panel}{$(polygon_text(to_physical(mesh, ref, SQUARE)))}")
        end
    end
    return
end

function dual_volumes_text(io, mesh, window, panel, highlight)
    for (index, primal) in pairs(mesh.faces)
        primal.axis == 1 || continue
        x, y = primal.center
        x0 = x - mesh.blocks[primal.minus.block].spacing[1] / 2
        x1 = x + mesh.blocks[primal.plus.block].spacing[1] / 2
        y0, y1 = y - primal.area / 2, y + primal.area / 2
        (x1 > window[1] && x0 < window[2] && y1 > window[3] && y0 < window[4]) || continue
        println(
            io, "\\fxmdual{$panel}{$(fmt(x0))}{$(fmt(y0))}{$(fmt(x1))}{$(fmt(y1))}",
            "{$(index == highlight ? 1 : 0)}",
        )
    end
    return
end

function dual_face_text(io, mesh, panel, index, name, aperture = 1.0)
    primal = mesh.faces[index]
    println(
        io, "\\fxmface{$panel}{$(primal.axis)}{$(fmt(primal.center[1]))}{$(fmt(primal.center[2]))}",
        "{$(fmt(primal.area))}{$name}{$(@sprintf("%.2f", aperture))}",
    )
    return
end

function momentum_dual_data()
    io = IOBuffer()
    info = Dict{Symbol, Any}()
    cells_text(args...) = dual_cells_text(io, args...)
    duals_text(args...) = dual_volumes_text(io, args...)
    face_text(args...) = dual_face_text(io, args...)
    # (a) Uniform: the four dual faces of f, each the average of two primal faces.
    mesh = vof_uniform_mesh()
    f = face_index_at(mesh, (6.0, 5.5))
    cells_text(mesh, nothing, DUAL_WINDOWS.uniform, 1)
    duals_text(mesh, DUAL_WINDOWS.uniform, 1, f)
    ws = StaggeredAdvectionWorkspace(FaceField{1}(mesh))
    @assert ws.schedule isa AMR.UniformDualAdvectionSchedule
    names = Dict(
        face_index_at(mesh, (5.0, 5.5)) => "g", f => "f", face_index_at(mesh, (7.0, 5.5)) => "g'",
        face_index_at(mesh, (5.5, 6.0)) => "k", face_index_at(mesh, (6.5, 6.0)) => "k'",
        face_index_at(mesh, (5.5, 5.0)) => "l", face_index_at(mesh, (6.5, 5.0)) => "l'",
    )
    for (index, name) in names
        face_text(mesh, 1, index, name)
    end
    dof_centres = [face.center for face in mesh.faces if face.axis == 1]
    connections = code_connections(ws, component_index(mesh, 1, f), dof_centres)
    @assert length(connections) == 4 &&
        all(c -> all(w -> abs(abs(w) - 0.5) < 1.0e-15, values(c.weights)), connections)
    p = mesh.faces[f].center
    for c in connections
        q = dof_centres[c.other]
        segment = c.axis == 1 ?
            ((p[1] + q[1]) / 2, p[2] - 0.5, (p[1] + q[1]) / 2, p[2] + 0.5) :
            (p[1] - 0.5, (p[2] + q[2]) / 2, p[1] + 0.5, (p[2] + q[2]) / 2)
        side = c.axis == 1 ? (q[1] < p[1] ? "l" : "r") : (q[2] < p[2] ? "b" : "t")
        println(
            io, "\\fxmdualface{1}{", join(fmt.(segment), "}{"), "}{",
            weight_label(c.weights, names), "}{$side}",
        )
    end

    # (b) Coarse--fine: every dual face of the fine segment f_1.
    mesh = vof_amr_mesh()
    f1 = face_index_at(mesh, (4.0, 5.25))
    cells_text(mesh, nothing, DUAL_WINDOWS.coarse_fine, 2)
    duals_text(mesh, DUAL_WINDOWS.coarse_fine, 2, f1)
    ws = StaggeredAdvectionWorkspace(FaceField{1}(mesh))
    names = Dict(
        face_index_at(mesh, (3.0, 5.5)) => "g", f1 => "f_1", face_index_at(mesh, (4.0, 5.75)) => "f_2",
        face_index_at(mesh, (4.5, 5.25)) => "h_1", face_index_at(mesh, (4.25, 5.5)) => "k",
        face_index_at(mesh, (4.25, 5.0)) => "k_1", face_index_at(mesh, (3.5, 5.0)) => "\\bar{k}^{-}",
        face_index_at(mesh, (3.5, 6.0)) => "\\bar{k}^{+}",
    )
    for (index, name) in names
        face_text(mesh, 2, index, name)
    end
    dof_centres = [face.center for face in mesh.faces if face.axis == 1]
    local_f1 = component_index(mesh, 1, f1)
    connections = code_connections(ws, local_f1, dof_centres)
    p = mesh.faces[f1].center
    for c in connections
        q = dof_centres[c.other]
        # Connections along the hanging side are drawn as arcs (x-flux factors
        # bend right, y-flux factors left) so that they do not overlap.
        bend = p[1] == q[1] ? (c.axis == 1 ? 1 : -1) : 0
        println(
            io, "\\fxmlink{2}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}{$(c.axis)}",
            "{$(c.flag ? 1 : 0)}{$(weight_label(c.weights, names))}{$bend}",
        )
    end
    info[:coarse_volume] = ws.schedule.volumes[local_f1]
    info[:coarse_links] = length(connections)
    @assert all(c -> c.flag, connections)
    # Hanging vertices: the interior end point of every fine segment of a coarse side.
    for primal in mesh.faces
        primal.axis == 1 || continue
        coarse = cell_level(mesh, primal.minus) < cell_level(mesh, primal.plus) ? primal.minus :
            cell_level(mesh, primal.plus) < cell_level(mesh, primal.minus) ? primal.plus : nothing
        isnothing(coarse) && continue
        y = cellcenter(mesh, coarse.block, coarse.i, coarse.j)[2]
        x = primal.center[1]
        (
            DUAL_WINDOWS.coarse_fine[1] < x < DUAL_WINDOWS.coarse_fine[2] &&
                DUAL_WINDOWS.coarse_fine[3] < y < DUAL_WINDOWS.coarse_fine[4]
        ) || continue
        abs(primal.center[2] - y) < primal.area || continue
        println(io, "\\fxmhanging{2}{$(fmt(x))}{$(fmt(y))}")
    end

    # (c) Cut: a face between two cut cells of the disc fixture (fine region,
    # cells of width 1/2); its dual faces carry aperture-weighted primal fluxes.
    mesh, geometry, _ = momentum_fixture(:cut)
    fc = face_index_at(mesh, (7.0, 5.25))
    cells_text(mesh, geometry, DUAL_WINDOWS.cut, 3)
    face = mesh.faces[fc]
    for cell in (face.minus, face.plus)
        open = strip_open_area(geometry, cell, face)[2]
        isempty(open) || println(io, "\\fxmopen{3}{$(polygon_text(to_physical(mesh, cell, open)))}")
    end
    println(
        io, "\\fxmdual{3}{$(fmt(face.center[1] - 0.25))}{$(fmt(face.center[2] - 0.25))}",
        "{$(fmt(face.center[1] + 0.25))}{$(fmt(face.center[2] + 0.25))}{1}",
    )
    names = Dict(
        face_index_at(mesh, (6.5, 5.25)) => "g", fc => "f", face_index_at(mesh, (7.5, 5.25)) => "g'",
        face_index_at(mesh, (6.75, 5.5)) => "k", face_index_at(mesh, (7.25, 5.5)) => "k'",
        face_index_at(mesh, (6.75, 5.0)) => "l", face_index_at(mesh, (7.25, 5.0)) => "l'",
    )
    for (index, name) in names
        face_text(mesh, 3, index, name, geometry.aperture[index])
    end
    ws = StaggeredAdvectionWorkspace(FaceField{1}(mesh), geometry)
    connections = code_connections(
        ws, component_index(mesh, 1, fc), [face.center for face in mesh.faces if face.axis == 1],
    )
    @assert length(connections) == 4
    p = face.center
    centres = [face.center for face in mesh.faces if face.axis == 1]
    for c in connections
        for (index, w) in c.weights
            @assert abs(w - geometry.aperture[index] / 2) <= 1.0e-15
        end
        q = centres[c.other]
        segment = c.axis == 1 ?
            ((p[1] + q[1]) / 2, p[2] - 0.25, (p[1] + q[1]) / 2, p[2] + 0.25) :
            (p[1] - 0.25, (p[2] + q[2]) / 2, p[1] + 0.25, (p[2] + q[2]) / 2)
        terms = sort!([names[index] for index in keys(c.weights)])
        label = "\\tfrac12(" * join(("\\aperture{$n}\\,$n" for n in terms), "+") * ")"
        side = c.axis == 1 ? (q[1] < p[1] ? "l" : "r") : (q[2] < p[2] ? "b" : "t")
        println(io, "\\fxmdualface{3}{", join(fmt.(segment), "}{"), "}{$label}{$side}")
    end
    info[:cut_aperture] = geometry.aperture[fc]
    info[:cut_open_dual] = geometry.face_dual_volumes[fc]
    info[:cut_uncut_dual] = mesh.face_dual_volumes[fc]
    info[:cut_open_fractions] = (
        AMR._get(geometry.open_fraction, face.minus), AMR._get(geometry.open_fraction, face.plus),
    )
    return String(take!(io)), info
end

# Figure "limiter": Sweby diagram of the code's limiter functions, and the
# three-point stencils of the fine segment f_1 of panel (b) of "dual".
function momentum_limiter_data()
    io = IOBuffer()
    for (name, scheme) in (
            ("minmod", Minmod()), ("mc", MonotonizedCentral()), ("vanleer", VanLeer()),
            ("superbee", Superbee()), ("fromm", Fromm()),
        )
        points = [(r, AMR._flux_limiter(r, scheme)) for r in 0.0:0.02:3.0]
        @assert all(((r, psi),) -> psi == independent_limiter(r, scheme), points)
        println(io, "\\fxlcurve{$name}{", join(("($(fmt(r)),$(fmt(psi)))" for (r, psi) in points), " -- "), "}")
    end
    # Stencils: dual faces of f_1 normal to e_1 with their upstream neighbours.
    mesh = vof_amr_mesh()
    ws = StaggeredAdvectionWorkspace(FaceField{1}(mesh))
    centres = [face.center for face in mesh.faces if face.axis == 1]
    f1 = component_index(mesh, 1, face_index_at(mesh, (4.0, 5.25)))
    stencils = Any[]
    for face in ws.schedule.faces
        face.axis == 1 || continue
        (face.minus == f1 || face.plus == f1) || continue
        for (upwind, downwind, stencil) in (
                (face.minus, face.plus, face.minus_stencil), (face.plus, face.minus, face.plus_stencil),
            )
            upwind == f1 || continue
            iszero(stencil.upstream) && continue
            u, d, uu = centres[upwind], centres[downwind], centres[stencil.upstream]
            weights, dofs, _ = independent_dual_weights(mesh, 1)
            mine = independent_upstream(mesh, dofs, dual_adjacency(collect(keys(weights))), upwind, downwind, 1)
            @assert mine[1] == stencil.upstream && mine[2] == stencil.ratio_scale
            println(
                io, "\\fxlstencil{$(fmt(u[1]))}{$(fmt(u[2]))}{$(fmt(d[1]))}{$(fmt(d[2]))}",
                "{$(fmt(uu[1]))}{$(fmt(uu[2]))}{$(fmt(2u[1] - d[1]))}{$(fmt(2u[2] - d[2]))}",
                "{$(frac(stencil.ratio_scale))}",
            )
            push!(stencils, (; u, d, uu, ratio = stencil.ratio_scale))
        end
    end
    return String(take!(io)), stencils
end

# Figure "rules": the face state of every dual face of component 1, read from
# the code's schedules on a walled 4 x 4 mesh of unit cells whose lower-left
# root is refined once; panel 1 single-phase convection, panel 2 two-phase
# transport of a liquid layer below y = 1.1 + 0.35 x.
#   \fxrcell{x0}{y0}{x1}{y1}{level}   \fxrliquid{path}   \fxrplic{x1}{y1}{x2}{y2}
#   \fxrdof{panel}{x}{y}{kind}        kind 0 interior unknown, 1 boundary unknown
#   \fxrlink{panel}{x1}{y1}{x2}{y2}{class}{bend}
#       class 0 regular (scheme), 1 non-regular between interior unknowns and
#       2 incident to a boundary unknown (amr_interface_scheme), 3 CTU for a
#       phase with volume at one of the two unknowns, 4, 5, 6 boundary dual
#       faces B1, B2, B3; bend +1 (axis-1 fluxes) or -1 (axis-2 fluxes) where
#       two dual faces join the same pair of unknowns.
const RULES_PLANE = (; offset = 1.1, slope = 0.35)
rules_levelset(x, y) = y - RULES_PLANE.offset - RULES_PLANE.slope * x

function momentum_rules_data()
    io = IOBuffer()
    info = Dict{Symbol, Int}()
    mesh = BlockTreeMesh(;
        base_mesh = (4, 4), block_size = (2, 2), extent = (4.0, 4.0),
        periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key == BlockKey(0, 0, 0),
    )
    field = InterfaceField(mesh, rules_levelset; lipschitz = hypot(1, RULES_PLANE.slope))
    reconstruct!(field)
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        println(io, "\\fxrcell{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}{$(cell_level(mesh, ref))}")
        c = cond(field, ref)
        c <= 0 && continue
        open = open_part(field, ref, SQUARE)
        if c >= 1
            println(io, "\\fxrliquid{$(polygon_text(to_physical(mesh, ref, open)))}")
            continue
        end
        n, a = liquid_plane(field, ref)
        println(io, "\\fxrliquid{$(polygon_text(to_physical(mesh, ref, clip_halfplane(open, n, a; keep_less = true))))}")
        p, q = to_physical(mesh, ref, segment_in(open, n, a))
        println(io, "\\fxrplic{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}")
    end
    k = 1
    interior = [face.center for face in mesh.faces if face.axis == k]
    boundary = [face.center for face in mesh.boundary_faces if face.axis == k]
    function link(panel, p, q, class, bend)
        key = Symbol("panel$(panel)_class$(class)")
        info[key] = get(info, key, 0) + 1
        println(io, "\\fxrlink{$panel}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(fmt(q[1]))}{$(fmt(q[2]))}{$class}{$bend}")
        return nothing
    end
    function links(panel, schedule, centres, class_of)
        multiplicity = Dict{Tuple{Int, Int}, Int}()
        for face in schedule.faces
            multiplicity[(face.minus, face.plus)] = get(multiplicity, (face.minus, face.plus), 0) + 1
        end
        for face in schedule.faces
            bend = multiplicity[(face.minus, face.plus)] > 1 ? (face.axis == k ? 1 : -1) : 0
            link(panel, centres[face.minus], centres[face.plus], class_of(face), bend)
        end
        return nothing
    end
    # (1) Single-phase convection: the schedule of the walled mesh and its
    # boundary dual faces, drawn to the centre of their boundary face.
    ws = StaggeredAdvectionWorkspace(FaceField{k}(mesh))
    links(1, ws.schedule, interior, face -> face.refinement_interface ? 1 : 0)
    for c in ws.boundary_faces
        kind = !c.normal_component ? 5 : c.wall_axis == k ? 4 : 6
        link(1, interior[c.degree_of_freedom], mesh.boundary_faces[c.boundary_index].center, kind, 0)
    end
    for p in interior
        println(io, "\\fxrdof{1}{$(fmt(p[1]))}{$(fmt(p[2]))}{0}")
    end
    # (2) Two-phase transport: the phase schedule with boundary unknowns. A
    # phase uses CTU on a dual face if its staggered fraction is below the
    # threshold at one of the two unknowns; shown where that phase has volume
    # at one of them (fractions at the start of the sweep, i.e. the predicted
    # fractions of a vanishing step).
    schedule = PhaseFaceScalar{k}(field).workspace.schedule
    n = length(interior)
    centres = [interior; boundary]
    fraction = AMR._phase_face_average!(FaceField{k}(mesh), field.conditional_fraction, nothing)
    C = [fraction.interior; fraction.boundary]
    threshold = WYAdvectionWorkspace(field).ctu_threshold
    switch(a, b) = min(a, b) < threshold && max(a, b) > 0
    ctu(face) = switch(C[face.minus], C[face.plus]) || switch(1 - C[face.minus], 1 - C[face.plus])
    links(
        2, schedule, centres,
        face -> ctu(face) ? 3 : !face.refinement_interface ? 0 : (face.minus > n || face.plus > n) ? 2 : 1,
    )
    for (i, p) in enumerate(centres)
        println(io, "\\fxrdof{2}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(i > n ? 1 : 0)}")
    end
    @assert all(get(info, Symbol("panel1_class$c"), 0) > 0 for c in (0, 1, 4, 5, 6))
    @assert all(get(info, Symbol("panel2_class$c"), 0) > 0 for c in (0, 1, 2, 3))
    return String(take!(io)), info
end

# Constant transported velocities stay constant: one velocity with a constant
# momentum source and the fixture's carrier; two velocities with equal constant
# phase velocities. Largest deviation over all stored values.
function constant_velocity_check()
    worst = 0.0
    U = (x, y, t, axis) -> axis == 1 ? 0.3 : -0.2
    for kind in (:uniform, :amr, :cut, :walls)
        field, velocity = two_phase_fixture(kind)
        mesh = field.mesh
        constant = FaceVelocity(mesh, U)
        work = WYAdvectionWorkspace(field)
        destination = FaceVelocity(mesh)
        advect!(
            field, velocity, stable_timestep(velocity, work), work; momentum = destination,
            density = (1000.0, 1.0), momentum_workspace = MomentumAdvectionWorkspace(field),
            momentum_source = constant,
        )
        worst = max(
            worst, maximum(abs, destination.interior .- constant.interior),
            maximum(abs, destination.boundary .- constant.boundary),
        )
    end
    for kind in (:uniform, :amr)
        field, _ = two_phase_fixture(kind)
        mesh = field.mesh
        velocity = (FaceVelocity(mesh, U), FaceVelocity(mesh, U))
        work = WYAdvectionWorkspace(field, velocity)
        destination = (FaceVelocity(mesh), FaceVelocity(mesh))
        advect!(
            field, velocity, stable_timestep(velocity, work), work; momentum = destination,
            density = (1000.0, 1.0), momentum_workspace = MomentumAdvectionWorkspace(field, velocity),
        )
        for p in 1:2
            active = findall(isfinite, destination[p].interior)
            worst = max(worst, maximum(abs, destination[p].interior[active] .- velocity[p].interior[active]))
        end
    end
    return worst
end

# --- Outputs of Chapter 6: figure data and the numbers quoted in the text ------------

function momentum_outputs(write_output)
    dual, dual_info = momentum_dual_data()
    write_output("momentum_dual.tex", dual)
    limiter, stencils = momentum_limiter_data()
    write_output("momentum_limiter.tex", limiter)
    rules, rules_info = momentum_rules_data()
    write_output("momentum_rules.tex", rules)

    worst = Dict(:uniform => 0.0, :coarse_fine => 0.0, :cut => 0.0)
    count = Dict(:uniform => 0, :coarse_fine => 0, :cut => 0)
    schedule = identity = 0.0
    conservation = Dict{Symbol, Any}()
    for kind in (:uniform, :amr, :cut)
        flux = convection_flux_check(kind)
        for class in keys(worst)
            worst[class] = max(worst[class], flux.worst[class]); count[class] += flux.count[class]
        end
        schedule = max(schedule, flux.schedule_difference); identity = max(identity, flux.identity_residual)
        conservation[kind] = convection_conservation_check(kind)
    end
    momentum = maximum(c.momentum for c in values(conservation))
    skew = maximum(c.skew for c in values(conservation))
    diagonal = maximum(c.diagonal for c in values(conservation))
    euler = maximum(c.euler for c in values(conservation))
    walls = Dict(kind => boundary_identity_check(kind) for kind in (:walls, :amr_walls, :cut_walls))
    walls_closed = maximum(w.closed for w in values(walls))
    walls_identity = maximum(w.worst for w in values(walls))
    walls_skew = maximum(w.skew for w in values(walls))
    classes = phase_classification_check()

    one = Dict(kind => one_velocity_momentum_check(kind) for kind in (:uniform, :amr, :walls, :cut))
    one_mc = one_velocity_momentum_check(:amr; scheme = MonotonizedCentral(), order = (2, 1))
    exact = (values(one)..., one_mc)
    one_replica = maximum(o.replica for o in values(one))
    one_momentum = maximum(o.momentum for o in exact)
    one_mass = maximum(max(o.mass_defect, o.mass_defect_partial) for o in exact)

    two = [two_velocity_momentum_check(kind) for kind in (:uniform, :amr, :walls)]
    two_mc = two_velocity_momentum_check(:amr; scheme = MonotonizedCentral(), order = (2, 1))
    two_replica = maximum(t.replica for t in (two..., two_mc))
    two_balance = maximum(maximum(t.balance) for t in (two..., two_mc))
    constant = constant_velocity_check()

    println("momentum flux check:         ", (; worst, count, schedule, identity))
    println("momentum conservation:       ", conservation)
    println("momentum boundary identity:  ", walls)
    println("momentum phase classes:      ", classes)
    println("one-velocity momentum:       ", one, " ", one_mc)
    println("two-velocity momentum:       ", two, " ", two_mc)
    println("momentum figures:            ", dual_info, " ", stencils, " ", rules_info)
    @assert schedule == 0 && identity <= 1.0e-15
    @assert maximum(values(worst)) <= 1.0e-15
    @assert momentum <= 1.0e-15 && skew <= 1.0e-14 && diagonal <= 1.0e-15 && euler <= 1.0e-6
    @assert walls_closed <= 1.0e-14 && walls_identity <= 1.0e-14 && walls_skew <= 1.0e-14
    @assert all(w.count == 0 for w in values(walls)) && walls[:amr_walls].kinds[:B3] > 0
    @assert classes.mismatched == 0 && classes.weights_difference <= 1.0e-14
    @assert classes.counts[:walls][:wall] > 0 && classes.counts[:walls][:nonregular] > 0 &&
        classes.counts[:cut][:cut_regular] > 0
    @assert one_replica <= 1.0e-14 && one_momentum <= 1.0e-15 && one_mass <= 1.0e-14
    @assert all(o.snaps == 0 for o in values(one)) && one[:cut].partial_dofs > 0 &&
        one[:cut].scalar_defect_partial > 1.0e-3
    @assert two_replica <= 1.0e-14 && two_balance <= 1.0e-15
    @assert all(t.nan_pattern for t in two) && all(t.snaps == 0 for t in two)
    @assert constant == 0
    fix2(x) = @sprintf("%.2f", x)
    return Any[
        ("chkMomFluxUniformFaces", count[:uniform]), ("chkMomFluxUniformDiff", sci(worst[:uniform])),
        ("chkMomFluxCoarseFaces", count[:coarse_fine]), ("chkMomFluxCoarseDiff", sci(worst[:coarse_fine])),
        ("chkMomFluxCutFaces", count[:cut]), ("chkMomFluxCutDiff", sci(worst[:cut])),
        ("chkMomIdentity", sci(identity)), ("chkMomMomentum", sci(momentum)),
        ("chkMomSkew", sci(skew)), ("chkMomDiagonal", sci(diagonal)), ("chkMomEuler", sci(euler)),
        ("chkMomLwEnergyUniform", sci(-conservation[:uniform].lw_energy)),
        ("chkMomLwEnergyAmr", sci(-conservation[:amr].lw_energy)),
        ("chkMomWallsClosed", sci(walls_closed)), ("chkMomWallsIdentity", sci(walls_identity)),
        ("chkMomWallsSkew", sci(walls_skew)), ("chkMomWallsBthree", walls[:amr_walls].kinds[:B3]),
        ("chkMomClassRegular", classes.counts[:walls][:regular]),
        ("chkMomClassNonregular", classes.counts[:walls][:nonregular]),
        ("chkMomClassWall", classes.counts[:walls][:wall]),
        ("chkMomClassCutRegular", classes.counts[:cut][:cut_regular]),
        ("chkMomClassStencils", classes.stencils),
        ("chkMomOneReplica", sci(one_replica)), ("chkMomOneMomentum", sci(one_momentum)),
        ("chkMomOneMass", sci(one_mass)), ("chkMomOneCutDofs", one[:cut].partial_dofs),
        ("chkMomOneCutScalarMass", fix3(one[:cut].scalar_defect_partial)),
        ("chkMomTwoReplica", sci(two_replica)), ("chkMomTwoBalance", sci(two_balance)),
        ("chkMomCoarseVolume", frac(dual_info[:coarse_volume])), ("chkMomCoarseLinks", dual_info[:coarse_links]),
        ("chkMomCutAperture", fix2(dual_info[:cut_aperture])),
        ("chkMomCutOpenDual", fix3(dual_info[:cut_open_dual])),
        ("chkMomCutUncutDual", frac(dual_info[:cut_uncut_dual])),
        ("chkMomCutOpenA", fix3(dual_info[:cut_open_fractions][1])),
        ("chkMomCutOpenB", fix3(dual_info[:cut_open_fractions][2])),
    ]
end
