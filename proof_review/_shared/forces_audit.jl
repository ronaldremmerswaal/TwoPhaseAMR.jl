# Shared helpers of the audits of Chapter 8 (forces) dated 2026-11-09: mesh/interface/solid fixtures,
# independent evaluations of the staggered density, phase-aperture-free linear algebra (jump matrix D,
# inverse metric K), and the "exists a cell function a with R J = S a" test that decides balance.
# Not an audit. Included by the scripts of proof_review/eq-forces-*/scripts (needs TwoPhaseAMR).
#
# The fixtures are the ones of docs/techref/figures/forces.jl (same walled 8 x 8 box of 16 x 16 level-0
# cells, composite = right half refined, same drop, planar interface through (4.3, 4.6), same sloped solids),
# re-implemented here so that the audit does not depend on the figure script.

module ForcesAudit

using LinearAlgebra, SparseArrays, Random
using TwoPhaseAMR
const AMR = TwoPhaseAMR

export AMR

const BOX = 8.0
const DROP = (; centre = (4.3, 4.1), radius = 2.23)
const DENSITY = (1000.0, 1.0)
const SIGMA = 0.7
const SOLID_LOW = (; offset = 0.9, slope = 0.1)     # solid y < 0.9 + 0.1 x
const SOLID_HIGH = (; offset = 7.1, slope = -0.1)   # solid y > 7.1 - 0.1 x

function mesh_of(kind::Symbol; base = (16, 16), block = (4, 4))
    refine = kind == :uniform ? ((key, origin, extent) -> false) :
        ((key, origin, extent) -> key.level == 0 && key.i >= 2)
    return BlockTreeMesh(;
        base_mesh = base, block_size = block, extent = (BOX, BOX),
        periodic = (false, false), max_level = 1, refine,
    )
end

solid_low(x, y) = y - SOLID_LOW.offset - SOLID_LOW.slope * x
solid_high(x, y) = SOLID_HIGH.offset + SOLID_HIGH.slope * x - y
drop_levelset(x, y) = hypot(x - DROP.centre[1], y - DROP.centre[2]) - DROP.radius

"""Drop fixture: kind in :uniform, :amr, :cut (the sloped solid below does not touch the drop)."""
function drop_field(kind::Symbol)
    mesh = mesh_of(kind == :uniform ? :uniform : :amr)
    if kind == :cut
        geometry = GeometryField(mesh, solid_low; lipschitz = hypot(1, SOLID_LOW.slope))
        return InterfaceField(mesh, drop_levelset; geometry, contact_angle = pi / 2, lipschitz = 1)
    end
    return InterfaceField(mesh, drop_levelset; lipschitz = 1)
end

"""Planar interface {g . (x - x0) = 0}, liquid on the side g . (x - x0) > 0 ... (liquid below for g = (0, -1));
solid = :none, :liquid (sloped solid at the bottom, in the liquid) or :gas (at the top)."""
function plane_field(kind::Symbol; gravity = (0.0, -1.0), solid = :none, point = (4.3, 4.6), mesh = nothing, contact = pi / 2)
    mesh === nothing && (mesh = mesh_of(kind))
    levelset = (x, y) -> -(gravity[1] * (x - point[1]) + gravity[2] * (y - point[2]))
    solid == :none && return InterfaceField(mesh, levelset; lipschitz = 1)
    shape = solid == :liquid ? solid_low : solid_high
    geometry = GeometryField(mesh, shape; lipschitz = hypot(1, 0.1))
    return InterfaceField(mesh, levelset; geometry, contact_angle = contact, lipschitz = 1)
end

disc_levelset(x, y) = hypot(x - 6.9, y - 6.6) - 0.95     # solid disc {< 0}, clear of the drop

"""Drop on a uniform or composite mesh with a solid disc (clear of the drop): kind in :uniform, :amr."""
function drop_disc_field(kind::Symbol)
    mesh = mesh_of(kind)
    geometry = GeometryField(mesh, disc_levelset; lipschitz = 1)
    return InterfaceField(mesh, drop_levelset; geometry, contact_angle = pi / 2, lipschitz = 1)
end

"""The five fixtures of the balance checks: uniform/composite without solid, composite with the sloped solid,
uniform/composite with a disc."""
function balance_fixtures()
    return [
        :uniform => drop_field(:uniform), :amr => drop_field(:amr), :amr_sloped => drop_field(:cut),
        :uniform_disc => drop_disc_field(:uniform), :amr_disc => drop_disc_field(:amr),
    ]
end

# --- linear algebra of the discrete setting ------------------------------------------

cell_count(mesh) = length(AMR._cell_references(mesh))

"""Jump matrix D (faces x cells): (D a)_f = |f| (a_plus - a_minus), cells in `_cell_index` order."""
function jump_matrix(mesh)
    nx, ny = mesh.block_size
    n = length(mesh.blocks) * nx * ny
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (i, face) in pairs(mesh.faces)
        m = AMR._cell_index(face.minus, nx, ny)
        p = AMR._cell_index(face.plus, nx, ny)
        push!(rows, i, i); push!(cols, m, p); push!(vals, -face.area, face.area)
    end
    return sparse(rows, cols, vals, length(mesh.faces), n)
end

"""Inverse face metric K of the operator as a sparse matrix."""
function metric_matrix(operator)
    metric = operator.gradient_operator.inverse_metric
    return metric isa AMR.DiagonalInverseFaceMetric ? spdiagm(0 => copy(metric.diagonal)) : copy(metric.matrix)
end

apertures(field) = isnothing(field.geometry) ? ones(length(field.mesh.faces)) : copy(field.geometry.aperture)

"""Independent staggered liquid fraction (volume weights without solid, open strip areas are read from the package)."""
function staggered_density(field, density = DENSITY)
    return AMR.mixture_density(field, density, field.geometry === nothing ? error("use package for solid") : field.geometry)
end

"""Cell function (vector over `_cell_index` order) from f(ref)."""
function cell_vector(f, mesh)
    nx, ny = mesh.block_size
    v = zeros(length(mesh.blocks) * nx * ny)
    for ref in AMR._cell_references(mesh)
        v[AMR._cell_index(ref, nx, ny)] = f(ref)
    end
    return v
end

function cell_field_of(values, mesh)
    nx, ny = mesh.block_size
    field = CellField(mesh)
    for ref in AMR._cell_references(mesh)
        AMR._set!(field, values[AMR._cell_index(ref, nx, ny)], ref)
    end
    fill_halo!(field)
    return field
end

"""R J computed independently from the operator data: R J = diag(sqrt(g)/a) K diag(sqrt(g)) J (a = 1 without solid);
`gamma` is the face coefficient (a^2/rho one-velocity, a/rho two-velocity). Faces with a = 0 give 0."""
function force_map(K, gamma, aperture, J, solid::Bool)
    root = sqrt.(gamma)
    out = root .* (K * (root .* J))
    solid || return out
    return [aperture[i] > 0 ? out[i] / aperture[i] : 0.0 for i in eachindex(out)]
end

"""Relative least-squares residual of  J_S = D_S a  over faces S (gamma > 0): decides whether J is the jump of a cell function."""
function jump_consistency(D, J, support)
    A = Matrix(D[support, :])
    b = J[support]
    x = A \ b
    nb = norm(b)
    return nb == 0 ? 0.0 : norm(A * x - b) / nb
end

"""Relative residual of  R J = S a  for the best cell function a (least squares in the Euclidean face norm),
S = R D (the code's weighted gradient). Faces with a = 0 give 0 rows."""
function map_consistency(R_apply, S, J)
    rhs = R_apply(J)
    x = S \ rhs
    return norm(S * x - rhs) / max(norm(rhs), 1.0e-300)
end

"""Velocity left by one projection of dt*acceleration from rest (one velocity), divided by dt."""
function project_force(field, operator, acceleration; dt = 0.01, velocity0 = nothing)
    velocity = FaceVelocity(field.mesh)
    velocity.interior .= dt .* acceleration.interior
    velocity0 === nothing || (velocity.interior .+= velocity0.interior)
    solver = EllipticSolver(operator; preconditioner = :tree_geometric)
    stats = if isnothing(field.geometry)
        project!(velocity, CellField(field.mesh), solver, dt, ProjectionWorkspace(solver);
            tolerance = 1.0e-12, max_cycles = 1000)
    else
        project!(velocity, CellField(field.mesh), solver, dt, ProjectionWorkspace(solver), field.geometry;
            tolerance = 1.0e-12, max_cycles = 1000)
    end
    stats.converged || error("projection did not converge")
    velocity0 === nothing ? nothing : (velocity.interior .-= velocity0.interior)
    return maximum(abs, velocity.interior) / dt
end

"""Project `velocity` (modified in place) with the one-velocity projection of `operator`; returns (pressure, stats)."""
function project_velocity!(field, operator, velocity; dt = 0.01)
    solver = EllipticSolver(operator; preconditioner = :tree_geometric)
    pressure = CellField(field.mesh)
    stats = if isnothing(field.geometry)
        project!(velocity, pressure, solver, dt, ProjectionWorkspace(solver);
            tolerance = 1.0e-12, max_cycles = 1000)
    else
        project!(velocity, pressure, solver, dt, ProjectionWorkspace(solver), field.geometry;
            tolerance = 1.0e-12, max_cycles = 1000)
    end
    stats.converged || error("projection did not converge")
    return pressure, stats
end

"""One-velocity pressure operator of the field (coefficient 1/rho, or a^2/rho with a solid)."""
one_velocity_operator(field; density = DENSITY) = isnothing(field.geometry) ?
    PoissonOperator(field; density) : PoissonOperator(field, field.geometry; density)

"""Two-velocity (CMDGFM/mimetic) pressure operator (coefficient 1/rho, or a/rho with a solid)."""
two_velocity_operator(field; density = DENSITY) =
    AMR._projection_pressure_operator(CMDGFMProjectionSolver(field; density))

"""Weighted gradient of the cell function `values` through the code (S p of the text)."""
function weighted_gradient_of(field, operator, values)
    p = cell_field_of(values, field.mesh)
    out = FaceVelocity(field.mesh)
    work = EllipticWorkspace(operator)
    if isnothing(field.geometry)
        weighted_gradient!(out, p, operator, work)
    else
        AMR._cut_cell_weighted_gradient!(out, p, operator, work, field.geometry)
    end
    return out
end

"""Exact hydrostatic cell function of Lemma 7.1 for the planar interface g.(x - x0) = 0 (liquid on the g.(x-x0) > 0 side):
rho^pi g.(x_c - x0) in a pure cell of phase pi, 0 in a mixed cell (conditional fraction in (0,1))."""
function hydrostatic_cell_function(field, gravity, point; density = DENSITY)
    mesh = field.mesh
    return cell_vector(mesh) do ref
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = AMR._conditional_fraction(field, ref)
        s = gravity[1] * (x[1] - point[1]) + gravity[2] * (x[2] - point[2])
        c >= 1 ? density[1] * s : c <= 0 ? density[2] * s : 0.0
    end
end

"""Sutherland-Hodgman clip of the polygon with {n.x <= a} (keep_less) or {n.x >= a}."""
function clip(poly, n, a; keep_less = true)
    m = length(poly)
    out = NTuple{2, Float64}[]
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

function polygon_area(poly)
    m = length(poly)
    m < 3 && return 0.0
    return abs(sum(poly[k][1] * poly[mod1(k + 1, m)][2] - poly[mod1(k + 1, m)][1] * poly[k][2] for k in 1:m)) / 2
end

"""Independent open area (local units) of the strip of cell `ref` over the tangential extent of `face`:
{|x| <= 1/2} x [lo, hi] (axis 1) intersected with the open side {n.x >= a} of the cell's solid plane."""
function open_strip_area(field, ref, face)
    mesh = field.mesh
    axis = face.axis
    tangent = 3 - axis
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    spacing = mesh.blocks[ref.block].spacing[tangent]
    lo = AMR._wrapped_delta(face.center[tangent] - face.area / 2 - center[tangent], mesh.extent[tangent], mesh.periodic[tangent]) / spacing
    hi = lo + face.area / spacing
    poly = axis == 1 ? [(-0.5, lo), (0.5, lo), (0.5, hi), (-0.5, hi)] : [(lo, -0.5), (hi, -0.5), (hi, 0.5), (lo, 0.5)]
    geometry = field.geometry
    geometry === nothing && return polygon_area(poly)
    solid = geometry.solid
    f = AMR._get(solid.fraction, ref)
    f <= 0 && return polygon_area(poly)
    f >= 1 && return 0.0
    return polygon_area(clip(poly, (AMR._get(solid.normal_x, ref), AMR._get(solid.normal_y, ref)), AMR._get(solid.intercept, ref); keep_less = false))
end

"""Independent staggered mixture density of the text (strip weights of the two cells, conditional fractions, clamp)."""
function staggered_density_independent(field, face; density = DENSITY)
    weighted = volume = 0.0
    for ref in (face.minus, face.plus)
        w = 0.5 * open_strip_area(field, ref, face) * field.mesh.blocks[ref.block].volume
        weighted += w * AMR._conditional_fraction(field, ref)
        volume += w
    end
    # both strips empty (up to clipping round-off): fraction 0, as in the text (such a face is closed, a = 0, and carries no force)
    full = sum(0.5 * field.mesh.blocks[ref.block].volume for ref in (face.minus, face.plus))
    c = volume > 1.0e-12 * full ? clamp(weighted / volume, 0.0, 1.0) : 0.0
    return density[1] * c + density[2] * (1 - c)
end

include(joinpath(@__DIR__, "techref_geometry.jl"))
using .Geo: segment_in
const SQUARE = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]

"""Independent cell potential phi_c = g . x^I_c of eq:forces:potential: midpoint of the PLIC segment in the whole reference cell
for 0 < C < 1 (anisotropic scaling by the cell widths), the cell centre otherwise."""
function gravity_potential_independent(field, gravity)
    mesh = field.mesh
    return cell_vector(mesh) do ref
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = AMR._conditional_fraction(field, ref)
        if 0 < c < 1
            n = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref))
            p, q = segment_in(SQUARE, n, AMR._get(field.intercept, ref))
            h = mesh.blocks[ref.block].spacing
            x = (x[1] + h[1] * (p[1] + q[1]) / 2, x[2] + h[2] * (p[2] + q[2]) / 2)
        end
        gravity[1] * x[1] + gravity[2] * x[2]
    end
end

"""Cell potential with a RANDOM admissible point: any point of the cell in a pure cell, any point of the PLIC segment in a mixed cell."""
function gravity_potential_random_points(field, gravity, rng)
    mesh = field.mesh
    return cell_vector(mesh) do ref
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        h = mesh.blocks[ref.block].spacing
        c = AMR._conditional_fraction(field, ref)
        if 0 < c < 1
            n = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref))
            p, q = segment_in(SQUARE, n, AMR._get(field.intercept, ref))
            t = rand(rng)
            local_point = (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2]))
        else
            local_point = (rand(rng) - 0.5, rand(rng) - 0.5)
        end
        gravity[1] * (x[1] + h[1] * local_point[1]) + gravity[2] * (x[2] + h[2] * local_point[2])
    end
end

"""Jumps rho_a |f| (phi_+ - phi_-) of eq:forces:gravity-jump for a GIVEN cell potential vector `phi`."""
function gravity_jumps_for_potential(field, phi; density = DENSITY)
    mesh = field.mesh
    liquid, gas = AMR._current_phase_face_apertures!(field)
    nx, ny = mesh.block_size
    a = apertures(field)
    J = zeros(length(mesh.faces))
    for (i, face) in pairs(mesh.faces)
        a[i] > 0 || continue
        J[i] = (density[1] * liquid[i] + density[2] * gas[i]) / a[i] * face.area *
            (phi[AMR._cell_index(face.plus, nx, ny)] - phi[AMR._cell_index(face.minus, nx, ny)])
    end
    return J
end

"""Integrated gravity jumps of eq:forces:gravity-jump (`mode = :open`), or of the code before issue #74 (`mode = :old`,
density rho_l a_l + rho_g (1 - a_l) in place of the open-face phase mixture)."""
function gravity_jumps_independent(field, gravity; density = DENSITY, mode = :open)
    mesh = field.mesh
    liquid, gas = AMR._current_phase_face_apertures!(field)
    phi = gravity_potential_independent(field, gravity)
    nx, ny = mesh.block_size
    a = apertures(field)
    J = zeros(length(mesh.faces))
    for (i, face) in pairs(mesh.faces)
        a[i] > 0 || continue
        rho = mode == :open ? (density[1] * liquid[i] + density[2] * gas[i]) / a[i] :
            density[1] * liquid[i] + density[2] * (1 - liquid[i])
        J[i] = rho * face.area * (phi[AMR._cell_index(face.plus, nx, ny)] - phi[AMR._cell_index(face.minus, nx, ny)])
    end
    return J
end

maxabs(v::FaceVelocity) = maximum(x -> isnan(x) ? 0.0 : abs(x), v.interior)
maxabs(v::Tuple) = max(maxabs(v[1]), maxabs(v[2]))

end # module
