# Chapter 12 (refinement criteria and conservative transfer): figure data and
# spot checks. Included by generate.jl, whose helpers (write_output, sci, fmt)
# it uses; it shares no fixture with the other chapter scripts. The functions
# marked "independent" evaluate the formulas of Chapter 12 without the
# indicator or transfer kernels: the band, interpolation-error and vorticity
# labels, the cell divergence and its regrid identities, the limited linear
# prolongation of a cell field, the PLIC clipping of the fraction
# prolongation, cell and phase integrals and the staggered integrals. The
# flow measurements repeat the loop body of `simulate!` around `adapt!`
# (checked against `simulate!` itself); the assertions fail if the code no
# longer matches the chapter.

using Printf: @sprintf

# --- Fixtures --------------------------------------------------------------------

# Walled 2 x 1 box with a wavy solid floor (solid where y < 0.15 + 0.1 sin(pi x))
# and a drop that rests on it; base mesh 32 x 16 cells in 8 x 8 blocks.
amr_solid(x, y) = y - 0.15 - 0.1sinpi(x)
amr_walled_drop(x, y) = hypot(x - 0.7, y - 0.42) - 0.22
# Periodic unit square with a drop; base mesh 32 x 32 cells in 8 x 8 blocks.
amr_periodic_drop(x, y) = hypot(x - 0.45, y - 0.5) - 0.2

function amr_walled_mesh(; delay = 0, interval = 1)
    return BlockTreeMesh(;
        base_mesh = (32, 16), block_size = (8, 8), extent = (2.0, 1.0),
        periodic = (false, false), max_level = 2,
        adaptation = MeshAdaptationOptions(update_interval = interval, coarsen_delay = delay),
    )
end

function amr_periodic_mesh(; delay = 0, interval = 1)
    return BlockTreeMesh(;
        base_mesh = (32, 32), block_size = (8, 8), extent = (1.0, 1.0),
        periodic = (true, true), max_level = 2,
        adaptation = MeshAdaptationOptions(update_interval = interval, coarsen_delay = delay),
    )
end

amr_parent(key) = BlockKey(key.level - 1, key.i ÷ 2, key.j ÷ 2)
amr_children(key) = [BlockKey(key.level + 1, 2key.i + ci, 2key.j + cj) for cj in 0:1 for ci in 0:1]
amr_cell_volume(mesh, b) = prod(mesh.blocks[b].spacing)
amr_wrap(delta, extent, periodic) = periodic ? delta - round(delta / extent) * extent : delta

# --- Independent measures -----------------------------------------------------------

# Face fluxes from a stream function psi at the face end points: every cell
# sum of o |f| u_f telescopes, so the Cartesian divergence vanishes.
function amr_streamfunction!(velocity, psi)
    mesh = velocity.mesh
    for (faces, values) in ((mesh.faces, velocity.interior), (mesh.boundary_faces, velocity.boundary))
        for (index, face) in pairs(faces)
            x, y = face.center
            a = face.area
            values[index] = face.axis == 1 ?
                (psi(x, y + a / 2) - psi(x, y - a / 2)) / a :
                -(psi(x + a / 2, y) - psi(x - a / 2, y)) / a
        end
    end
    return velocity
end

# Independent: (D u)_c = sum_f o_{c,f} |f| u_f / |c|, per block key.
function amr_divergence_by_key(mesh, velocity)
    nx, ny = mesh.block_size
    flux = Dict(block.key => zeros(nx, ny) for block in mesh.blocks)
    key(cell) = mesh.blocks[cell.block].key
    for (index, face) in pairs(mesh.faces)
        f = velocity.interior[index] * face.area
        flux[key(face.minus)][face.minus.i, face.minus.j] += f
        flux[key(face.plus)][face.plus.i, face.plus.j] -= f
    end
    for (index, face) in pairs(mesh.boundary_faces)
        flux[key(face.cell)][face.cell.i, face.cell.j] += face.side * velocity.boundary[index] * face.area
    end
    for (b, block) in pairs(mesh.blocks)
        flux[block.key] ./= amr_cell_volume(mesh, b)
    end
    return flux
end

# Independent: worst deviation from the three regrid identities of a cell
# quantity: unchanged blocks, children of a refined cell (parent value) and
# merged cells (mean of the four children).
function amr_identity_errors(mesh, old::Dict, new::Dict)
    nx, ny = mesh.block_size
    unchanged = refined = coarsened = 0.0
    for (key, values) in new
        if haskey(old, key)
            unchanged = max(unchanged, maximum(abs.(values .- old[key])))
        elseif key.level > 0 && haskey(old, amr_parent(key))
            parent = old[amr_parent(key)]
            ci, cj = mod(key.i, 2), mod(key.j, 2)
            for j in 1:ny, i in 1:nx
                fi, fj = ci * nx + i, cj * ny + j
                refined = max(refined, abs(values[i, j] - parent[(fi + 1) ÷ 2, (fj + 1) ÷ 2]))
            end
        else
            kids = amr_children(key)
            all(k -> haskey(old, k), kids) || error("block $key has no old parent or children")
            for j in 1:ny, i in 1:nx
                total = 0.0
                for fj in (2j - 1):2j, fi in (2i - 1):2i
                    child = kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]
                    total += old[child][mod(fi - 1, nx) + 1, mod(fj - 1, ny) + 1]
                end
                coarsened = max(coarsened, abs(values[i, j] - total / 4))
            end
        end
    end
    return (; unchanged, refined, coarsened)
end

# Independent: staggered integrals M_k = sum |omega_f| u_f over the interior
# faces of axis k, their absolute counterparts and the kinetic energy.
function amr_staggered_integrals(mesh, velocity)
    M = [0.0, 0.0]
    A = [0.0, 0.0]
    E = 0.0
    for (index, face) in pairs(mesh.faces)
        w = mesh.face_dual_volumes[index]
        u = velocity.interior[index]
        M[face.axis] += w * u
        A[face.axis] += w * abs(u)
        E += w * u^2 / 2
    end
    return (; M, A, E)
end

function amr_affine_error(mesh, field, axis, g)
    worst = 0.0
    for (faces, values) in ((mesh.faces, field.interior), (mesh.boundary_faces, field.boundary))
        k = 0
        for face in faces
            face.axis == axis || continue
            k += 1
            worst = max(worst, abs(values[k] - g(face.center...)))
        end
    end
    return worst
end

# Independent polygon clipping (Sutherland--Hodgman) by n.x <= a or n.x >= a.
function amr_clip(poly, n, a; keep_less = true)
    out = NTuple{2, Float64}[]
    isempty(poly) && return out
    side(p) = keep_less ? a - (n[1] * p[1] + n[2] * p[2]) : (n[1] * p[1] + n[2] * p[2]) - a
    for k in eachindex(poly)
        p, q = poly[k], poly[mod1(k + 1, length(poly))]
        sp, sq = side(p), side(q)
        sp >= 0 && push!(out, p)
        if (sp >= 0) != (sq >= 0)
            t = sp / (sp - sq)
            push!(out, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        end
    end
    return out
end
amr_area(poly) = length(poly) < 3 ? 0.0 : abs(
    sum(
        poly[k][1] * poly[mod1(k + 1, length(poly))][2] -
            poly[mod1(k + 1, length(poly))][1] * poly[k][2] for k in eachindex(poly)
    )
) / 2

# PLIC planes (liquid and solid) and the halo-filled pressure of every block.
function amr_snapshot(interface, geometry, pressure)
    reconstruct!(interface)
    fill_halo!(pressure, ZeroGradient())
    mesh = interface.mesh
    snapshot = Dict{BlockKey, Any}()
    for (b, block) in pairs(mesh.blocks)
        snapshot[block.key] = (
            nx = copy(interface.normal_x.blocks[b]), ny = copy(interface.normal_y.blocks[b]),
            a = copy(interface.intercept.blocks[b]),
            sx = isnothing(geometry) ? nothing : copy(geometry.solid.normal_x.blocks[b]),
            sy = isnothing(geometry) ? nothing : copy(geometry.solid.normal_y.blocks[b]),
            sa = isnothing(geometry) ? nothing : copy(geometry.solid.intercept.blocks[b]),
            p = copy(pressure.blocks[b]),
        )
    end
    return snapshot
end

# Independent replica of the fraction prolongation of every refined cell:
# 4|Q_r cap Q^o cap Q^l| clamped by 4|Q_r cap Q^o| in the parent's local
# coordinates. Children whose conditional fraction lies within the floor of
# 0 or 1 are snapped by the reconstruction and counted separately.
function amr_vof_replica(mesh, interface, geometry, snapshot)
    nx, ny = mesh.block_size
    h = mesh.halo
    worst = snapped_volume = 0.0
    snapped = children = 0
    for (b, block) in pairs(mesh.blocks)
        key = block.key
        haskey(snapshot, key) && continue
        key.level > 0 && haskey(snapshot, amr_parent(key)) || continue
        P = snapshot[amr_parent(key)]
        ci, cj = mod(key.i, 2), mod(key.j, 2)
        for j in 1:ny, i in 1:nx
            fi, fj = ci * nx + i, cj * ny + j
            pi, pj = (fi + 1) ÷ 2 + h, (fj + 1) ÷ 2 + h
            xlo, ylo = isodd(fi) ? -0.5 : 0.0, isodd(fj) ? -0.5 : 0.0
            quadrant = [(xlo, ylo), (xlo + 0.5, ylo), (xlo + 0.5, ylo + 0.5), (xlo, ylo + 0.5)]
            open = isnothing(geometry) ? quadrant :
                amr_clip(quadrant, (P.sx[pi, pj], P.sy[pi, pj]), P.sa[pi, pj]; keep_less = false)
            liquid = amr_clip(open, (P.nx[pi, pj], P.ny[pi, pj]), P.a[pi, pj])
            alpha = clamp(4amr_area(liquid), 0.0, 4amr_area(open))
            stored = interface.fraction.blocks[b][i + h, j + h]
            available = isnothing(geometry) ? 1.0 : geometry.open_fraction.blocks[b][i + h, j + h]
            c = available > 0 ? alpha / available : 0.0
            children += 1
            if 0 < c <= interface.fraction_floor || 1 - interface.fraction_floor <= c < 1
                snapped += 1
                snapped_volume += abs(stored - alpha) * amr_cell_volume(mesh, b)
            else
                worst = max(worst, abs(stored - alpha))
            end
        end
    end
    return (; worst, snapped, snapped_volume, children)
end

amr_mc(a, b) = a * b <= 0 ? 0.0 : copysign(min(abs(a + b) / 2, 2abs(a), 2abs(b)), a + b)

# Independent replica of the cell transfer of `pressure`: limited linear
# prolongation, mean restriction, copies; and the local bound of every child
# by the parent and its four axial neighbours.
function amr_cell_replica(mesh, pressure, snapshot)
    nx, ny = mesh.block_size
    h = mesh.halo
    worst = 0.0
    violations = 0
    for (b, block) in pairs(mesh.blocks)
        key = block.key
        values = pressure.blocks[b]
        if haskey(snapshot, key)
            worst = max(worst, maximum(abs.(values[(h + 1):(h + nx), (h + 1):(h + ny)] .-
                snapshot[key].p[(h + 1):(h + nx), (h + 1):(h + ny)])))
        elseif key.level > 0 && haskey(snapshot, amr_parent(key))
            p = snapshot[amr_parent(key)].p
            ci, cj = mod(key.i, 2), mod(key.j, 2)
            for j in 1:ny, i in 1:nx
                fi, fj = ci * nx + i, cj * ny + j
                pi, pj = (fi + 1) ÷ 2 + h, (fj + 1) ÷ 2 + h
                v = p[pi, pj]
                sx = amr_mc(v - p[pi - 1, pj], p[pi + 1, pj] - v)
                sy = amr_mc(v - p[pi, pj - 1], p[pi, pj + 1] - v)
                child = v + (isodd(fi) ? -0.25 : 0.25) * sx + (isodd(fj) ? -0.25 : 0.25) * sy
                worst = max(worst, abs(values[i + h, j + h] - child))
                neighbourhood = (v, p[pi - 1, pj], p[pi + 1, pj], p[pi, pj - 1], p[pi, pj + 1])
                lo, hi = extrema(neighbourhood)
                tol = 4eps() * maximum(abs, neighbourhood)
                lo - tol <= values[i + h, j + h] <= hi + tol || (violations += 1)
            end
        else
            kids = amr_children(key)
            for j in 1:ny, i in 1:nx
                total = 0.0
                for fj in (2j - 1):2j, fi in (2i - 1):2i
                    child = kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]
                    total += snapshot[child].p[mod(fi - 1, nx) + 1 + h, mod(fj - 1, ny) + 1 + h]
                end
                worst = max(worst, abs(values[i + h, j + h] - total / 4))
            end
        end
    end
    return (; worst, violations)
end

# --- Static transfer audit (Sections "cell", "phase", "face") -----------------------

# One regrid sequence (labels from `sequence`) with every registered quantity
# measured before and after each regrid; returns per-regrid records.
function amr_regrid_sequence(mesh, interface, geometry, workspace, sequence; scalar, pressure, general, solenoidal)
    records = []
    for (step, labels_of) in enumerate(sequence)
        V = integrate(interface)
        S = isnothing(geometry) ? 0.0 : integrate(geometry)
        amounts = (integrate(scalar, :liquid), integrate(scalar, :gas))
        P = integrate(pressure)
        Pabs = sum(sum(abs, interior(pressure, b)) * amr_cell_volume(mesh, b) for b in 1:nblocks(mesh))
        D = amr_divergence_by_key(mesh, general)
        Dmax = maximum(maximum(abs, x) for x in values(D))
        integrals = (general = amr_staggered_integrals(mesh, general), solenoidal = amr_staggered_integrals(mesh, solenoidal))
        snapshot = amr_snapshot(interface, geometry, pressure)
        stats = adapt!(workspace, labels_of(); step, force = true)
        @assert stats.changed
        vof = amr_vof_replica(mesh, interface, geometry, snapshot)
        cell = amr_cell_replica(mesh, pressure, snapshot)
        identity = amr_identity_errors(mesh, D, amr_divergence_by_key(mesh, general))
        hmin = minimum(minimum(block.spacing) for block in mesh.blocks)
        umax = max(maximum(abs, solenoidal.interior), maximum(abs, solenoidal.boundary; init = 0.0))
        Dsol = maximum(maximum(abs, x) for x in values(amr_divergence_by_key(mesh, solenoidal)))
        after = (general = amr_staggered_integrals(mesh, general), solenoidal = amr_staggered_integrals(mesh, solenoidal))
        momentum = Dict{Symbol, Float64}()
        for name in (:general, :solenoidal), axis in 1:2
            change = abs(after[name].M[axis] - integrals[name].M[axis]) / integrals[name].A[axis]
            kind = mesh.periodic[axis] ? :periodic : :wall
            momentum[kind] = max(get(momentum, kind, 0.0), change)
        end
        energy = maximum(abs(after[name].E - integrals[name].E) / integrals[name].E for name in (:general, :solenoidal))
        push!(
            records, (;
                stats,
                volume = abs(integrate(interface) - V) / V,
                solid = isnothing(geometry) ? 0.0 : abs(integrate(geometry) - S) / S,
                scalar = maximum(abs.((integrate(scalar, :liquid), integrate(scalar, :gas)) .- amounts) ./ amounts),
                pressure = abs(integrate(pressure) - P) / Pabs,
                vof, cell,
                identity = max(identity.unchanged, identity.refined, identity.coarsened) / Dmax,
                solenoidal = Dsol * hmin / umax,
                momentum, energy,
            ),
        )
    end
    return records
end

function amr_transfer_check()
    records = []
    affine = 0.0
    # Walled box with solid: interface band at level 2, solid band at level 1.
    mesh = amr_walled_mesh()
    geometry = GeometryField(mesh, amr_solid; lipschitz = 1.5)
    interface = InterfaceField(mesh, amr_walled_drop; geometry, lipschitz = 1)
    scalar = PhaseScalar(interface; liquid = (x, y) -> 1 + x, gas = (x, y) -> 2 - y)
    pressure = CellField(mesh, (x, y) -> 1 + x * y)
    general = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? sinpi(x) * cospi(y) : 0.5x * y)
    solenoidal = amr_streamfunction!(FaceVelocity(mesh, Float64), (x, y) -> 0.3y + 0.1sinpi(2x) * sinpi(y) + 0.05x)
    ax = FaceField{1}(mesh, (x, y) -> 1 + 2x - 3y)
    ay = FaceField{2}(mesh, (x, y) -> -2 + 4x + y)
    workspace = AMRWorkspace(mesh, interface, geometry, scalar, pressure, general, solenoidal, ax, ay)
    indicators = (InterfaceRefinementIndicator(interface; level = 2), GeometryRefinementIndicator(geometry; level = 1))
    refine = () -> copy(refinement_labels!(workspace, indicators))
    coarsen = () -> fill(:coarsen, nblocks(mesh))
    append!(records, amr_regrid_sequence(mesh, interface, geometry, workspace, (refine, refine, coarsen, coarsen);
        scalar, pressure, general, solenoidal))
    affine = max(affine, amr_affine_error(mesh, ax, 1, (x, y) -> 1 + 2x - 3y), amr_affine_error(mesh, ay, 2, (x, y) -> -2 + 4x + y))
    # The affine fields are also exact on the intermediate meshes.
    # Periodic square without solid: band at level 2.
    mesh = amr_periodic_mesh()
    interface = InterfaceField(mesh, amr_periodic_drop; lipschitz = 1)
    scalar = PhaseScalar(interface; liquid = (x, y) -> 1 + sinpi(2x), gas = (x, y) -> 2 - cospi(2y))
    pressure = CellField(mesh, (x, y) -> 1.5 + sinpi(2x) * cospi(2y))
    general = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 1 + sinpi(2x) * cospi(2y) : 0.5 + cospi(2x) * sinpi(4y))
    solenoidal = amr_streamfunction!(FaceVelocity(mesh, Float64), (x, y) -> 0.3y - 0.2x + 0.1sinpi(2x) * sinpi(2y))
    workspace = AMRWorkspace(mesh, interface, scalar, pressure, general, solenoidal)
    indicators = (InterfaceRefinementIndicator(interface; level = 2),)
    refine = () -> copy(refinement_labels!(workspace, indicators))
    coarsen = () -> fill(:coarsen, nblocks(mesh))
    append!(records, amr_regrid_sequence(mesh, interface, nothing, workspace, (refine, refine, coarsen, coarsen);
        scalar, pressure, general, solenoidal))
    worst(f) = maximum(f(r) for r in records)
    return (;
        regrids = length(records),
        refined = sum(r.stats.refined_blocks for r in records),
        coarsened = sum(r.stats.coarsened_families for r in records),
        volume = worst(r -> r.volume), solid = worst(r -> r.solid), scalar = worst(r -> r.scalar),
        pressure = worst(r -> r.pressure),
        children = sum(r.vof.children for r in records), snapped = sum(r.vof.snapped for r in records),
        snapped_volume = worst(r -> r.vof.snapped_volume), replica = worst(r -> r.vof.worst),
        cell = worst(r -> r.cell.worst), bound = sum(r.cell.violations for r in records),
        identity = worst(r -> r.identity), solenoidal = worst(r -> r.solenoidal),
        periodic = worst(r -> get(r.momentum, :periodic, 0.0)),
        wall = worst(r -> get(r.momentum, :wall, 0.0)),
        wall_min = minimum(r.momentum[:wall] for r in records if haskey(r.momentum, :wall)),
        energy = worst(r -> r.energy), affine,
    )
end

# --- Indicators (Section "indicators") -----------------------------------------------

# Independent replica of the band rule on the stored (refreshed) conditional
# fractions of the block and its ring of halo slots.
function amr_band_replica(field, labels, level, ring)
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    floor = field.fraction_floor
    mismatches = 0
    for (b, block) in pairs(mesh.blocks)
        values = field.conditional_fraction.blocks[b][(h + 1 - ring):(h + nx + ring), (h + 1 - ring):(h + ny + ring)]
        near = minimum(values) < 1 - floor && maximum(values) > floor
        expected = near ? (block.key.level < level ? :refine : block.key.level > level ? :coarsen : :keep) : :coarsen
        mismatches += expected != labels[b]
    end
    return mismatches
end

function amr_error_label(level, error, refine, coarsen, minimum_level, maximum_level; resolved = true)
    level < minimum_level && return :refine
    level > maximum_level && return :coarsen
    error > refine && level < maximum_level && return :refine
    resolved && error < coarsen && (level == 0 || level > minimum_level) && return :coarsen
    return :keep
end

# Independent replica of the relative interpolation defect of every block.
function amr_interpolation_errors(field; floor = sqrt(eps()))
    mesh = field.mesh
    fill_halo!(field, ZeroGradient())
    nx, ny = mesh.block_size
    h = mesh.halo
    errors = zeros(nblocks(mesh))
    for b in 1:nblocks(mesh)
        v = field.blocks[b]
        for j in (h + 1):(h + ny), i in (h + 1):(h + nx)
            neighbours = (v[i - 1, j], v[i + 1, j], v[i, j - 1], v[i, j + 1])
            scale = max(floor, abs(v[i, j]), maximum(abs, neighbours))
            errors[b] = max(errors[b], abs(v[i, j] - sum(neighbours) / 4) / scale)
        end
    end
    return errors
end

# Independent replica of the vorticity estimate: side-mean cell velocities and
# the face-weighted least-squares gradient; returns the block maxima of
# max(h) |omega|.
function amr_vorticity_errors(velocity)
    mesh = velocity.mesh
    nx, ny = mesh.block_size
    count = ncells(mesh)
    index(c) = (c.block - 1) * nx * ny + (c.j - 1) * nx + c.i
    sums = zeros(4, count)
    lengths = zeros(4, count)
    slot(axis, side) = 2axis - (side < 0 ? 1 : 0)
    for (k, face) in pairs(mesh.faces)
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            sums[slot(face.axis, side), index(cell)] += face.area * velocity.interior[k]
            lengths[slot(face.axis, side), index(cell)] += face.area
        end
    end
    for (k, face) in pairs(mesh.boundary_faces)
        sums[slot(face.axis, face.side), index(face.cell)] += face.area * velocity.boundary[k]
        lengths[slot(face.axis, face.side), index(face.cell)] += face.area
    end
    centred = zeros(2, count)
    for c in 1:count, axis in 1:2
        present = [s for s in (2axis - 1, 2axis) if lengths[s, c] > 0]
        centred[axis, c] = sum(sums[s, c] / lengths[s, c] for s in present) / length(present)
    end
    centres = Vector{NTuple{2, Float64}}(undef, count)
    for b in 1:nblocks(mesh), j in 1:ny, i in 1:nx
        centres[index(CellRef(b, i, j))] = cellcenter(mesh, b, i, j)
    end
    A = zeros(3, count)
    B = zeros(4, count)
    for face in mesh.faces
        m, p = index(face.minus), index(face.plus)
        dx = amr_wrap(centres[p][1] - centres[m][1], mesh.extent[1], mesh.periodic[1])
        dy = amr_wrap(centres[p][2] - centres[m][2], mesh.extent[2], mesh.periodic[2])
        w = face.area / (dx^2 + dy^2)
        du, dv = centred[1, p] - centred[1, m], centred[2, p] - centred[2, m]
        for c in (m, p)
            A[:, c] .+= w .* (dx^2, dx * dy, dy^2)
            B[:, c] .+= w .* (du * dx, du * dy, dv * dx, dv * dy)
        end
    end
    errors = zeros(nblocks(mesh))
    for b in 1:nblocks(mesh), j in 1:ny, i in 1:nx
        c = index(CellRef(b, i, j))
        a11, a12, a22 = A[:, c]
        det = a11 * a22 - a12^2
        @assert det > 128eps() * max(abs(a11 * a22), a12^2)
        dvdx = (a22 * B[3, c] - a12 * B[4, c]) / det
        dudy = (a11 * B[2, c] - a12 * B[1, c]) / det
        errors[b] = max(errors[b], maximum(mesh.blocks[b].spacing) * abs(dvdx - dudy))
    end
    return errors
end

# Blocks refined by a band without a liquid--gas interface in block and ring
# (wetted solid boundary), and blocks refined by the curvature criterion
# without a mixed cell, next to a solid (Chapter 12, issues 118 and 119).
function amr_solid_artifacts()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (8, 8), extent = (1.0, 1.0),
        periodic = (false, false), max_level = 2,
    )
    geometry = GeometryField(mesh, (x, y) -> y - 0.3 - 0.1x; lipschitz = 2)
    interface = InterfaceField(mesh, (x, y) -> y - 0.65; geometry, lipschitz = 1)
    nx, ny = mesh.block_size
    h = mesh.halo
    mixed(b, ring) = any(
        c -> 0 < c < 1,
        interface.conditional_fraction.blocks[b][(h + 1 - ring):(h + nx + ring), (h + 1 - ring):(h + ny + ring)],
    )
    labels = Vector{Symbol}(undef, nblocks(mesh))
    refinement_labels!(labels, InterfaceRefinementIndicator(interface; level = 2))
    wetted = count(b -> labels[b] === :refine && !mixed(b, 1), 1:nblocks(mesh))
    refinement_labels!(labels, CurvatureRefinementIndicator(interface))
    curvature = count(b -> labels[b] === :refine && !mixed(b, 0), 1:nblocks(mesh))
    return (; blocks = nblocks(mesh), wetted, curvature)
end

# Regrids of a static drop under the band criterion alone (issue 117): the
# number of topology changes after the initial two refinements, and the
# smallest period of the leaf sets from step 5 on (1: a fixed point).
function amr_band_cycle(; ring = 1, delay = 2, steps = 14)
    mesh = amr_walled_mesh(; delay)
    interface = InterfaceField(mesh, amr_walled_drop; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = 2, buffer_cells = ring)
    workspace = AMRWorkspace(mesh, interface)
    history = Set{BlockKey}[]
    for step in 1:steps
        adapt!(workspace, indicator; step)
        push!(history, Set(block.key for block in mesh.blocks))
    end
    changes = count(s -> history[s] != history[s - 1], 3:steps)
    period = something(findfirst(p -> all(history[s] == history[s - p] for s in (5 + p):steps), 1:4), 0)
    return (; changes, period, steps)
end

function amr_indicator_check()
    mismatches = 0
    compared = 0
    # Band criteria on the walled fixture before and after one regrid.
    mesh = amr_walled_mesh()
    geometry = GeometryField(mesh, amr_solid; lipschitz = 1.5)
    interface = InterfaceField(mesh, amr_walled_drop; geometry, lipschitz = 1)
    workspace = AMRWorkspace(mesh, interface, geometry)
    labels = Vector{Symbol}()
    for pass in 1:2
        resize!(labels, nblocks(mesh))
        for (field, indicator) in (
                (interface, InterfaceRefinementIndicator(interface; level = 2)),
                (geometry.solid, GeometryRefinementIndicator(geometry; level = 1)),
            )
            refinement_labels!(labels, indicator)
            mismatches += amr_band_replica(field, labels, indicator.level, indicator.buffer_cells)
            compared += nblocks(mesh)
        end
        pass == 1 && adapt!(
            workspace, (InterfaceRefinementIndicator(interface; level = 2), GeometryRefinementIndicator(geometry; level = 1));
            step = 1, force = true,
        )
    end
    # Error-based criteria on a composite periodic mesh.
    mesh = BlockTreeMesh(;
        base_mesh = (32, 32), block_size = (8, 8), extent = (1.0, 1.0), max_level = 2,
        refine = (key, origin, extent) -> key.level == 0 ? key.i + key.j < 4 : key.i < 3,
    )
    field = CellField(mesh, (x, y) -> 1 + 0.5sinpi(4x) * cospi(4y))
    velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? sinpi(2y) * (1 + 0.3cospi(2x)) : -0.5sinpi(2x))
    labels = Vector{Symbol}(undef, nblocks(mesh))
    interpolation = InterpolationErrorIndicator(field; E_refine = 0.02, E_coarsen = 0.006, maximum_level = 2)
    refinement_labels!(labels, interpolation)
    errors = amr_interpolation_errors(field)
    for (b, block) in pairs(mesh.blocks)
        mismatches += labels[b] != amr_error_label(block.key.level, errors[b], 0.02, 0.006, 0, 2)
    end
    vorticity = VorticityRefinementIndicator(velocity; refine_tolerance = 0.2, coarsen_tolerance = 0.05)
    refinement_labels!(labels, vorticity)
    errors = amr_vorticity_errors(velocity)
    for (b, block) in pairs(mesh.blocks)
        mismatches += labels[b] != amr_error_label(block.key.level, errors[b], 0.2, 0.05, 0, 2)
    end
    compared += 2nblocks(mesh)
    # Interpolation defect of a smooth field on two uniform meshes.
    defect(n) = maximum(amr_interpolation_errors(CellField(
        BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8)),
        (x, y) -> 1 + 0.5sinpi(2x) * cospi(2y),
    )))
    ratio = defect(32) / defect(64)
    artifacts = amr_solid_artifacts()
    cycle = amr_band_cycle()
    still = amr_band_cycle(; ring = 0)
    return (; mismatches, compared, ratio, artifacts, cycle, still)
end

# --- Flow measurements (Section "divergence") -------------------------------------

const AMR_DENSITY = (1.0, 0.1)

function amr_flow(model; solid::Bool)
    if solid
        mesh = amr_walled_mesh(; delay = 2)
        geometry = GeometryField(mesh, amr_solid; lipschitz = 1.5)
        drop = amr_walled_drop
        initialization = (; lipschitz = 1)
        interface = InterfaceField(mesh, drop; geometry, initialization...)
        indicators = (
            InterfaceRefinementIndicator(interface; level = 2),
            GeometryRefinementIndicator(geometry; level = 1),
        )
        velocity = (x, y, t, axis) -> axis == 1 ? pi * sinpi(x / 2)^2 * sinpi(2y) :
            -pi / 2 * sinpi(x) * sinpi(y)^2
        options = (; geometry, boundary = NoSlip())
    else
        mesh = amr_periodic_mesh(; delay = 2)
        geometry = nothing
        drop = (x, y) -> hypot(x - 0.4, y - 0.45) - 0.1
        # The Lipschitz-bounded quadrature does not resolve this drop on the
        # base mesh (issue 49); the default adaptive quadrature does.
        initialization = (;)
        interface = InterfaceField(mesh, drop)
        indicators = (
            InterfaceRefinementIndicator(interface; level = 1),
            CurvatureRefinementIndicator(interface),
        )
        velocity = (x, y, t, axis) -> axis == 1 ? 1 + 0.25sinpi(2y) : 0.5 + 0.25sinpi(2x)
        options = (;)
    end
    workspace = isnothing(geometry) ? AMRWorkspace(mesh, interface) : AMRWorkspace(mesh, interface, geometry)
    for pass in 1:3
        adapt!(workspace, indicators; step = pass, force = true).changed || break
        initialize!(interface, drop; initialization...)
    end
    physics = TwoPhase(interface; density = AMR_DENSITY, dynamic_viscosity = (1.0e-3, 1.0e-4), surface_tension = 0.01)
    flow = FlowSimulation(
        mesh, physics; velocity_model = model, velocity,
        adaptivity = Adaptivity(criteria = indicators), options...,
    )
    project!(flow; tolerance = 1.0e-12)
    return flow
end

# Independent: max over open cells of |D^o u| (one velocity) or |D^mix| (two
# velocities) with the current apertures; whether the maximum is in a cut cell.
function amr_constraint_defect(flow)
    velocity = flow.velocity
    interface = flow.interface
    geometry = interface.geometry
    mesh = interface.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    flux = zeros(ncells(mesh))
    index(c) = (c.block - 1) * nx * ny + (c.j - 1) * nx + c.i
    product(a, u) = iszero(a) ? 0.0 : a * u
    if velocity isa FaceVelocity
        for (k, face) in pairs(mesh.faces)
            f = face.area * product(isnothing(geometry) ? 1.0 : geometry.aperture[k], velocity.interior[k])
            flux[index(face.minus)] += f
            flux[index(face.plus)] -= f
        end
        for (k, face) in pairs(mesh.boundary_faces)
            a = isnothing(geometry) ? 1.0 : geometry.boundary_aperture[k]
            flux[index(face.cell)] += face.side * face.area * product(a, velocity.boundary[k])
        end
    else
        liquid, gas = TwoPhaseAMR._current_phase_face_apertures!(interface)
        for (k, face) in pairs(mesh.faces)
            f = face.area * (product(liquid[k], velocity[1].interior[k]) + product(gas[k], velocity[2].interior[k]))
            flux[index(face.minus)] += f
            flux[index(face.plus)] -= f
        end
        for (k, face) in pairs(mesh.boundary_faces)
            l, g = TwoPhaseAMR._phase_boundary_face_apertures(interface, k)
            flux[index(face.cell)] += face.side * face.area *
                (product(l, velocity[1].boundary[k]) + product(g, velocity[2].boundary[k]))
        end
    end
    worst = 0.0
    cut = false
    for (b, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        open = isnothing(geometry) ? 1.0 : geometry.open_fraction.blocks[b][i + h, j + h]
        open > 0 || continue
        d = abs(flux[index(CellRef(b, i, j))]) / (open * block.volume)
        d > worst && ((worst, cut) = (d, open < 1))
    end
    return worst, cut
end

# Independent sum of rho |u| over the staggered (open) volumes: the momentum scale.
function amr_absolute_momentum(flow)
    velocity, interface = flow.velocity, flow.interface
    mesh, geometry = interface.mesh, interface.geometry
    volumes = isnothing(geometry) ? mesh.face_dual_volumes : geometry.face_dual_volumes
    total = 0.0
    for (k, face) in pairs(mesh.faces)
        c = TwoPhaseAMR._staggered_liquid_fraction(interface, geometry, face)
        if velocity isa FaceVelocity
            total += (AMR_DENSITY[1] * c + AMR_DENSITY[2] * (1 - c)) * abs(velocity.interior[k]) * volumes[k]
        else
            iszero(c) || (total += AMR_DENSITY[1] * c * abs(velocity[1].interior[k]) * volumes[k])
            c == 1 || (total += AMR_DENSITY[2] * (1 - c) * abs(velocity[2].interior[k]) * volumes[k])
        end
    end
    return total
end

function amr_flow_state(flow, time)
    state = SimulationState(flow.velocity, flow.pressure, flow.solver, time, 0, flow.workspace)
    defect, cut = amr_constraint_defect(flow)
    return (;
        volume = integrate(flow.interface),
        momentum = collect(diagnostic_value(LinearMomentum(), state)),
        energy = diagnostic_value(KineticEnergy(), state),
        defect, cut, scale = amr_absolute_momentum(flow),
    )
end

# The loop body of `simulate!` for a fixed step, measuring the state after the
# physical step, after the transfer and rebuild, and after the two-velocity
# projection of every regrid.
function amr_flow_loop!(flow, steps, dt; tolerance = 1.0e-10, record = true)
    velocity, pressure, solver, workspace = flow.velocity, flow.pressure, flow.solver, flow.workspace
    amr = TwoPhaseAMR._navier_stokes_amr_workspace(velocity, pressure, solver)
    cycles = TwoPhaseAMR._navier_stokes_cycle_limit(solver, nothing)
    time = flow.time
    records = []
    for n in 1:steps
        stats = step!(
            velocity, pressure, solver, dt, workspace; time, tolerance, max_cycles = cycles,
            _transport_geometry_prepared = false, _reference_dt = dt,
        )
        @assert stats.converged
        time += dt
        solver.mesh.adaptation.completed_steps += 1
        before = record ? amr_flow_state(flow, time) : nothing
        regrid = adapt!(amr, solver.refinement_indicators; step = solver.mesh.adaptation.completed_steps, time)
        regrid.changed || continue
        TwoPhaseAMR._rebuild_topology_dependent_state!(solver, workspace; dt)
        transferred = record ? amr_flow_state(flow, time) : nothing
        velocity isa TwoPhaseVelocity && TwoPhaseAMR._project_regridded_two_velocity!(
            velocity, solver, workspace, dt, time; tolerance, max_iterations = cycles,
        )
        projected = record ? amr_flow_state(flow, time) : nothing
        push!(records, (; n, regrid, before, transferred, projected))
    end
    flow.time = time
    return records
end

amr_velocity_values(v) = v isa FaceVelocity ? [v.interior; v.boundary] :
    [v[1].interior; v[1].boundary; v[2].interior; v[2].boundary]

# Replica loop against simulate!: same topology changes and identical
# velocities (including the absent-phase NaN pattern).
function amr_loop_validation(model; solid, steps)
    reference = amr_flow(model; solid)
    dt = 0.5 * stable_timestep(reference; cfl = 0.5)
    result = simulate!(reference, steps * dt; dt, tolerance = 1.0e-10)
    replica = amr_flow(model; solid)
    records = amr_flow_loop!(replica, result.steps, dt; record = false)
    a, b = amr_velocity_values(reference.velocity), amr_velocity_values(replica.velocity)
    @assert result.steps == steps && result.mesh_updates == length(records)
    @assert length(a) == length(b) && isequal(isnan.(a), isnan.(b))
    return maximum(abs(x - y) for (x, y) in zip(a, b) if !isnan(x); init = 0.0)
end

function amr_flow_case(model; solid, steps)
    flow = amr_flow(model; solid)
    dt = 0.5 * stable_timestep(flow; cfl = 0.5)
    tolerance = 1.0e-10
    records = amr_flow_loop!(flow, steps, dt; tolerance)
    worst(f) = maximum(f(r) for r in records; init = 0.0)
    relative(a, b) = abs(a - b) / abs(b)
    return (;
        steps, dt, regrids = length(records), tolerance,
        refined = sum(r.regrid.refined_blocks for r in records),
        coarsened = sum(r.regrid.coarsened_families for r in records),
        volume = worst(r -> relative(r.transferred.volume, r.before.volume)),
        momentum = worst(r -> maximum(abs, r.transferred.momentum .- r.before.momentum) / r.before.scale),
        energy = worst(r -> relative(r.transferred.energy, r.before.energy)),
        before = worst(r -> dt * r.before.defect),
        transferred = worst(r -> dt * r.transferred.defect),
        growth = worst(r -> dt * (r.transferred.defect - r.before.defect)),
        transferred_cut = any(r -> r.transferred.cut && dt * r.transferred.defect > 1.0e3 * tolerance, records),
        projected = worst(r -> dt * r.projected.defect),
        projection_energy = worst(r -> relative(r.projected.energy, r.transferred.energy)),
        projection_momentum = worst(r -> maximum(abs, r.projected.momentum .- r.transferred.momentum) / r.transferred.scale),
    )
end

function amr_flow_check()
    validation = max(
        amr_loop_validation(OneVelocityModel(); solid = false, steps = 12),
        amr_loop_validation(TwoVelocityModel(); solid = false, steps = 12),
    )
    one = amr_flow_case(OneVelocityModel(); solid = false, steps = 60)
    two = amr_flow_case(TwoVelocityModel(); solid = false, steps = 60)
    cut = amr_flow_case(OneVelocityModel(); solid = true, steps = 40)
    return (; validation, one, two, cut)
end

# --- Figure data: labels, balance and a refined cell -------------------------------

#   \fxablock{panel}{x0}{y0}{x1}{y1}{level}{mark}  mark: R, K, C (labels), B (balance)
#   \fxaplic{panel}{x1}{y1}{x2}{y2}   \fxasolid{panel}{x1}{y1}{x2}{y2}
#   \fxacell{x0}{y0}{x1}{y1}          one cell refined in the second regrid
function amr_figure_data()
    mesh = amr_walled_mesh()
    geometry = GeometryField(mesh, amr_solid; lipschitz = 1.5)
    interface = InterfaceField(mesh, amr_walled_drop; geometry, lipschitz = 1)
    workspace = AMRWorkspace(mesh, interface, geometry)
    indicators = (InterfaceRefinementIndicator(interface; level = 2), GeometryRefinementIndicator(geometry; level = 1))
    adapt!(workspace, indicators; step = 1, force = true)
    labels = copy(refinement_labels!(workspace, indicators))
    io = IOBuffer()
    f6(x) = fmt(round(x; digits = 6))
    function segments(panel)
        for (macro_name, field) in (("fxaplic", interface), ("fxasolid", geometry.solid))
            for (p, q) in plic_segments(field)
                println(io, "\\$(macro_name){$panel}{$(f6(p[1]))}{$(f6(p[2]))}{$(f6(q[1]))}{$(f6(q[2]))}")
            end
        end
    end
    marks = Dict(:refine => "R", :keep => "K", :coarsen => "C")
    counts = Dict("R" => 0, "K" => 0, "C" => 0, "B" => 0)
    old = Dict(block.key => labels[b] for (b, block) in pairs(mesh.blocks))
    for (b, block) in pairs(mesh.blocks)
        x0, y0 = block.origin
        mark = marks[labels[b]]
        counts[mark] += 1
        println(io, "\\fxablock{1}{$(f6(x0))}{$(f6(y0))}{$(f6(x0 + block.extent[1]))}{$(f6(y0 + block.extent[2]))}{$(block.key.level)}{$mark}")
    end
    segments(1)
    adapt!(workspace, labels; step = 2, force = true)
    cell = nothing
    for (b, block) in pairs(mesh.blocks)
        x0, y0 = block.origin
        parent = block.key.level > 0 ? amr_parent(block.key) : nothing
        balanced = !haskey(old, block.key) && !isnothing(parent) && get(old, parent, :refine) !== :refine
        balanced && (counts["B"] += 1)
        println(io, "\\fxablock{2}{$(f6(x0))}{$(f6(y0))}{$(f6(x0 + block.extent[1]))}{$(f6(y0 + block.extent[2]))}{$(block.key.level)}{$(balanced ? "B" : "")}")
        if isnothing(cell) && block.key.level == 2 && !haskey(old, block.key)
            s = 2 .* block.spacing
            cell = (block.origin[1] + s[1], block.origin[2] + s[2], block.origin[1] + 2s[1], block.origin[2] + 2s[2])
        end
    end
    segments(2)
    println(io, "\\fxacell{$(f6(cell[1]))}{$(f6(cell[2]))}{$(f6(cell[3]))}{$(f6(cell[4]))}")
    return String(take!(io)), (; counts, blocks = nblocks(mesh))
end

# --- Outputs -------------------------------------------------------------------

function amr_outputs(write_output)
    figure, figure_info = amr_figure_data()
    write_output("amr_regrid.tex", figure)
    transfer = amr_transfer_check()
    indicators = amr_indicator_check()
    flows = amr_flow_check()
    println("amr transfer:                ", transfer)
    println("amr indicators:              ", indicators)
    println("amr flows:                   ", flows)
    println("amr figure:                  ", figure_info)

    @assert transfer.volume <= 1.0e-14 && transfer.solid <= 1.0e-14 && transfer.scalar <= 1.0e-14
    @assert transfer.pressure <= 1.0e-14 && transfer.replica <= 1.0e-15 && transfer.cell <= 1.0e-15
    @assert transfer.bound == 0 && transfer.identity <= 1.0e-14 && transfer.solenoidal <= 1.0e-12
    @assert transfer.periodic <= 1.0e-13 && transfer.wall_min > 1.0e-5 && transfer.affine <= 1.0e-13
    @assert transfer.snapped > 0 && transfer.snapped_volume <= 1.0e-15
    @assert indicators.mismatches == 0 && 3.5 < indicators.ratio < 4.5
    @assert indicators.artifacts.wetted > 0 && indicators.artifacts.curvature > 0
    @assert indicators.cycle.period == 3 && indicators.still.changes == 0
    @assert flows.validation <= 1.0e-12
    @assert flows.one.transferred <= 1.0e3 * flows.one.tolerance && flows.one.growth <= 1.0e-15
    @assert flows.two.transferred > 1.0e3 * flows.two.tolerance && flows.two.projected <= flows.two.tolerance
    @assert flows.cut.transferred > 1.0e3 * flows.cut.tolerance && flows.cut.transferred_cut
    @assert max(flows.one.volume, flows.two.volume, flows.cut.volume) <= 1.0e-14

    case(prefix, r) = Any[
        ("chkAmrFlow$(prefix)Regrids", r.regrids),
        ("chkAmrFlow$(prefix)Refined", r.refined),
        ("chkAmrFlow$(prefix)Coarsened", r.coarsened),
        ("chkAmrFlow$(prefix)Volume", sci(r.volume)),
        ("chkAmrFlow$(prefix)Momentum", sci(r.momentum)),
        ("chkAmrFlow$(prefix)Energy", sci(r.energy)),
        ("chkAmrFlow$(prefix)Before", sci(r.before)),
        ("chkAmrFlow$(prefix)Transferred", sci(r.transferred)),
        ("chkAmrFlow$(prefix)Projected", sci(r.projected)),
    ]
    return Any[
        ("chkAmrRegrids", transfer.regrids),
        ("chkAmrRefinedBlocks", transfer.refined),
        ("chkAmrCoarsenedFamilies", transfer.coarsened),
        ("chkAmrVolume", sci(transfer.volume)),
        ("chkAmrSolid", sci(transfer.solid)),
        ("chkAmrScalar", sci(transfer.scalar)),
        ("chkAmrPressure", sci(transfer.pressure)),
        ("chkAmrChildren", transfer.children),
        ("chkAmrSnapped", transfer.snapped),
        ("chkAmrSnappedVolume", sci(transfer.snapped_volume)),
        ("chkAmrReplica", sci(transfer.replica)),
        ("chkAmrCellReplica", sci(transfer.cell)),
        ("chkAmrIdentity", sci(transfer.identity)),
        ("chkAmrSolenoidal", sci(transfer.solenoidal)),
        ("chkAmrPeriodicMomentum", sci(transfer.periodic)),
        ("chkAmrWallMomentumMin", sci(transfer.wall_min)),
        ("chkAmrWallMomentum", sci(transfer.wall)),
        ("chkAmrEnergy", sci(transfer.energy)),
        ("chkAmrAffine", sci(transfer.affine)),
        ("chkAmrLabelsCompared", indicators.compared),
        ("chkAmrInterpRatio", @sprintf("%.2f", indicators.ratio)),
        ("chkAmrWettedBlocks", indicators.artifacts.wetted),
        ("chkAmrCurvatureBlocks", indicators.artifacts.curvature),
        ("chkAmrArtifactBlocks", indicators.artifacts.blocks),
        ("chkAmrCycleChanges", indicators.cycle.changes),
        ("chkAmrCyclePeriod", indicators.cycle.period),
        ("chkAmrCycleSteps", indicators.cycle.steps),
        ("chkAmrFlowValidation", sci(flows.validation)),
        ("chkAmrFlowSteps", flows.one.steps),
        ("chkAmrFlowCutSteps", flows.cut.steps),
        ("chkAmrFlowTolerance", sci(flows.one.tolerance)),
        case("One", flows.one)...,
        case("Two", flows.two)...,
        case("Cut", flows.cut)...,
        ("chkAmrFlowTwoProjectionEnergy", sci(flows.two.projection_energy)),
        ("chkAmrFlowTwoProjectionMomentum", sci(flows.two.projection_momentum)),
        ("chkAmrFigRefine", figure_info.counts["R"]),
        ("chkAmrFigKeep", figure_info.counts["K"]),
        ("chkAmrFigCoarsen", figure_info.counts["C"]),
        ("chkAmrFigBalance", figure_info.counts["B"]),
    ]
end
