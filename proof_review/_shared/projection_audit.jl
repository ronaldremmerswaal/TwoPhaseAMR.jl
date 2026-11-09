# Shared helpers of the audits of Chapter 10 (pressure and velocity constraints) dated 2026-11-09:
# fixtures of the chapter (drop of radius 1.6 at (4.3, 2.3) in the walled 8 x 8 box, uniform / composite mesh,
# with and without the sloped solid y < 0.9 + 0.1 x), predictors, independent face data and independent dense
# evaluations of the continuous-to-discrete formulas of the chapter (mixture divergence, kinetic norm, phase
# energy, vertex patches from face coordinates).  Not an audit.  Included by the scripts of
# proof_review/*projection*/scripts (needs TwoPhaseAMR; run with --project=docs/techref/figures from the repo root).
#
# The fixtures re-implement docs/techref/figures/projection.jl (same mesh/drop/solid) on top of
# proof_review/_shared/forces_audit.jl, so that the audit does not use the figure script.

module ProjectionAudit

using LinearAlgebra, SparseArrays, Random
include(joinpath(@__DIR__, "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const FA = ForcesAudit

export AMR, FA

const DENSITY = (1000.0, 1.0)
const DT = 0.01
const DROP = (; centre = (4.3, 2.3), radius = 1.6)

drop_levelset(x, y) = hypot(x - DROP.centre[1], y - DROP.centre[2]) - DROP.radius

"""Chapter fixture: kind in :uniform, :amr (composite, right half refined once); solid = sloped solid y < 0.9 + 0.1 x."""
function fixture(kind::Symbol; solid::Bool, fraction_floor = nothing)
    mesh = FA.mesh_of(kind == :uniform ? :uniform : :amr)
    extra = isnothing(fraction_floor) ? (;) : (; fraction_floor)
    solid || return InterfaceField(mesh, drop_levelset; lipschitz = 1, extra...)
    geometry = GeometryField(mesh, FA.solid_low; lipschitz = hypot(1, FA.SOLID_LOW.slope))
    return InterfaceField(mesh, drop_levelset; geometry, contact_angle = pi / 2, lipschitz = 1, extra...)
end

"""The four fixtures of the chapter: (name, field)."""
fixtures() = [
    "uniform" => fixture(:uniform; solid = false), "composite" => fixture(:amr; solid = false),
    "uniform+solid" => fixture(:uniform; solid = true), "composite+solid" => fixture(:amr; solid = true),
]

"""Fixture whose solid is a horizontal band that splits the open domain into two connected components."""
function band_fixture(kind::Symbol)
    mesh = FA.mesh_of(kind == :uniform ? :uniform : :amr)
    band(x, y) = abs(y - 4.05) - 0.62      # fluid where > 0
    geometry = GeometryField(mesh, band; lipschitz = 1)
    levelset(x, y) = hypot(x - 4.3, y - 1.9) - 1.0
    return InterfaceField(mesh, levelset; geometry, contact_angle = pi / 2, lipschitz = 1)
end

# --- predictors ------------------------------------------------------------------------

"""Smooth divergent one-velocity predictor with random Fourier content, no-permeation and no-slip applied."""
function one_predictor(field, rng; amplitude = 1.0)
    a = amplitude .* randn(rng, 8)
    f(x, y, t, axis) = axis == 1 ?
        a[1] + a[2] * sinpi(x / 4) * cospi(y / 8) + a[3] * cospi(y / 3 + a[4]) :
        a[5] + a[6] * cospi(x / 8) * sinpi(y / 4) + a[7] * sinpi(x / 5 + a[8])
    v = FaceVelocity(field.mesh, f)
    enforce_no_permeation!(v)
    isnothing(field.geometry) || enforce_geometry_no_slip!(v, field.geometry)
    return v
end

"""Two-velocity predictor (liquid, gas): common part plus a phase jump; absent phases NaN."""
function two_predictor(field, rng; jump = 0.4)
    mesh = field.mesh
    a = randn(rng, 8); b = jump .* randn(rng, 4)
    common(x, y, axis) = axis == 1 ? a[1] + a[2] * sinpi(x / 4) * cospi(y / 8) + a[3] * cospi(y / 3 + a[4]) :
        a[5] + a[6] * cospi(x / 8) * sinpi(y / 4) + a[7] * sinpi(x / 5 + a[8])
    dj(x, y, axis) = axis == 1 ? b[1] + b[2] * sinpi(y / 4) : b[3] + b[4] * cospi(x / 4)
    liquid = FaceVelocity(mesh, (x, y, t, axis) -> common(x, y, axis))
    gas = FaceVelocity(mesh, (x, y, t, axis) -> common(x, y, axis) + dj(x, y, axis))
    velocity = (liquid, gas)
    enforce_no_permeation!(velocity)
    isnothing(field.geometry) || enforce_geometry_no_slip!(velocity, field.geometry)
    AMR._mark_inactive_phase_velocities!(velocity, field)
    return velocity
end

copy_velocity(v::FaceVelocity) = (w = FaceVelocity(v.mesh); w.interior .= v.interior; w.boundary .= v.boundary; w)
copy_velocity(v::Tuple) = map(copy_velocity, v)

# --- geometry: cells, faces, incidence ---------------------------------------------------

"""Open cell volumes |c^o| in `_cell_index` order (|c| without solid)."""
function open_volumes(field)
    mesh = field.mesh
    return FA.cell_vector(
        ref -> begin
            v = mesh.blocks[ref.block].volume
            isnothing(field.geometry) ? v : v * AMR._get(field.geometry.open_fraction, ref)
        end, mesh
    )
end

"""Open dual volume V^o_f of every interior face (V_f without solid)."""
open_dual_volumes(field) = isnothing(field.geometry) ? copy(field.mesh.face_dual_volumes) : copy(field.geometry.face_dual_volumes)

aperture(field) = FA.apertures(field)

"""Boundary face data: (cell index, side s_f, area, open aperture) for every physical boundary face."""
function boundary_faces(field)
    mesh = field.mesh
    nx, ny = mesh.block_size
    out = NamedTuple[]
    for (index, face) in pairs(mesh.boundary_faces)
        ap = isnothing(field.geometry) ? 1.0 : field.geometry.boundary_aperture[index]
        push!(out, (; index, cell = AMR._cell_index(face.cell, nx, ny), side = face.side, area = face.area, axis = face.axis, aperture = ap))
    end
    return out
end

"""Independent open divergence |c^o| (D v)_c = sum_f o_{c,f} |f| a_f v_f (faces and boundary faces), cell vector."""
function open_flux(field, v::FaceVelocity)
    mesh = field.mesh
    nx, ny = mesh.block_size
    a = aperture(field)
    out = zeros(length(mesh.blocks) * nx * ny)
    for (i, face) in pairs(mesh.faces)
        w = a[i] == 0 ? 0.0 : face.area * a[i] * v.interior[i]
        out[AMR._cell_index(face.minus, nx, ny)] += w       # o = +1 : normal points out of the minus cell
        out[AMR._cell_index(face.plus, nx, ny)] -= w
    end
    for b in boundary_faces(field)
        out[b.cell] += b.side * b.area * b.aperture * v.boundary[b.index]
    end
    return out
end

"""Independent mixture flux |c^o| (D^mix(u^l,u^g))_c, cell vector; zero weights never multiply an absent (NaN) phase."""
function mixture_flux(field, velocity)
    mesh = field.mesh
    nx, ny = mesh.block_size
    liquid, gas = AMR._current_phase_face_apertures!(field)
    prod(w, u) = iszero(w) ? 0.0 : w * u
    out = zeros(length(mesh.blocks) * nx * ny)
    for (i, face) in pairs(mesh.faces)
        w = face.area * (prod(liquid[i], velocity[1].interior[i]) + prod(gas[i], velocity[2].interior[i]))
        out[AMR._cell_index(face.minus, nx, ny)] += w
        out[AMR._cell_index(face.plus, nx, ny)] -= w
    end
    for (index, face) in pairs(mesh.boundary_faces)
        l, g = AMR._phase_boundary_face_apertures(field, index)
        out[AMR._cell_index(face.cell, nx, ny)] += face.side * face.area * (prod(l, velocity[1].boundary[index]) + prod(g, velocity[2].boundary[index]))
    end
    return out
end

# --- independent two-velocity face data (eq:projection:face-data) -------------------------

"""Independent face data of eq:projection:face-data: ĉ (own staggered fraction), masses, ζ, a^zeta, reduced mass, mixed set."""
function face_data(field; density = DENSITY)
    mesh = field.mesh
    n = length(mesh.faces)
    if isnothing(field.geometry)
        C = [
            begin
                hm = mesh.blocks[f.minus.block].spacing[f.axis]; hp = mesh.blocks[f.plus.block].spacing[f.axis]
                clamp((hm * AMR._conditional_fraction(field, f.minus) + hp * AMR._conditional_fraction(field, f.plus)) / (hm + hp), 0.0, 1.0)
            end for f in mesh.faces
        ]
    else
        C = [(FA.staggered_density_independent(field, f; density = (1.0, 0.0)))  for f in mesh.faces]   # = ĉ (density (1,0))
    end
    ml = density[1] .* C
    mg = density[2] .* (1 .- C)
    m = ml .+ mg
    al, ag = AMR._current_phase_face_apertures!(field)
    zl = -mg; zg = copy(ml)
    apgfm = (al .* zl .+ ag .* zg) ./ m
    mr = ml .* mg ./ m
    fl = field.fraction_floor
    mixed = [f for f in 1:n if fl < C[f] < 1 - fl && al[f] + ag[f] > 0]
    return (; C, ml, mg, m, zl, zg, apgfm, mr, al, ag, mixed, floor = fl)
end

# --- dense linear algebra of the pressure map ----------------------------------------------

"""Dense J (faces x cells, in `_cell_index` order)."""
jump_dense(field) = Matrix(FA.jump_matrix(field.mesh))

"""Dense open face metric K^o (W K W with a solid; the code stores it already scaled)."""
metric_dense(operator) = Matrix(FA.metric_matrix(operator))

"""Dense force map of the operator with face coefficient `gamma`: R = diag(sqrt(γ)/a) K^o diag(sqrt(γ)) (a = 1 without solid; rows a = 0 are 0)."""
function force_map_dense(field, operator, gamma)
    K = metric_dense(operator)
    a = aperture(field)
    root = sqrt.(gamma)
    scale = [a[i] > 0 ? root[i] / a[i] : 0.0 for i in eachindex(a)]
    return scale .* K .* root'
end


"""One-velocity projection with a choice of preconditioner; returns (pressure, stats, solver)."""
function run_projection!(
        field, op, velocity; dt = DT, tolerance = 1.0e-12, preconditioner = :tree_geometric,
        pressure = CellField(field.mesh), compatibility = :project, max_cycles = 2000, velocity_boundary = nothing, boundary_time = 0.0
    )
    solver = EllipticSolver(op; preconditioner)
    work = ProjectionWorkspace(solver)
    kw = (; tolerance, max_cycles, compatibility)
    isnothing(velocity_boundary) || (kw = merge(kw, (; velocity_boundary, boundary_time)))
    stats = isnothing(field.geometry) ?
        project!(velocity, pressure, solver, dt, work; kw...) :
        project!(velocity, pressure, solver, dt, work, field.geometry; kw...)
    return pressure, stats, solver
end

# --- independent vertex patches from face coordinates --------------------------------------

"""End points of an interior face (physical coordinates), from its centre, axis and area (tangential extent)."""
function face_endpoints(face)
    c = face.center; h = face.area
    return face.axis == 1 ? ((c[1], c[2] - h / 2), (c[1], c[2] + h / 2)) : ((c[1] - h / 2, c[2]), (c[1] + h / 2, c[2]))
end

"""Faces of `axis` (other than f) that have an end point of f as an end point, by coordinates (periodic images ignored: walled box)."""
function patch_faces(mesh, f, axis; tol = 1.0e-9)
    ends = face_endpoints(mesh.faces[f])
    out = Int[]
    for (g, other) in pairs(mesh.faces)
        g == f && continue
        other.axis == axis || continue
        oe = face_endpoints(other)
        any(e -> any(o -> hypot(e[1] - o[1], e[2] - o[2]) < tol, oe), ends) && push!(out, g)
    end
    return out
end

end # module
