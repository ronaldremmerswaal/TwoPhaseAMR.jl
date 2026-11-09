# Helpers shared by the audits of Chapter 9 (viscous stress and implicit solve) of the technical reference.
#
# Not a check by itself.  Needs the TwoPhaseAMR package (docs/techref/figures environment or the repository environment with its
# dependencies) and is included with
#     isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
#     using .ViscosityAudit
#
# It builds on `_shared/momentum_audit.jl` (random composite meshes, solids and interfaces) and adds
#   * the three drop fixtures of docs/techref/figures/{forces,viscosity}.jl (walled composite, cut, periodic composite),
#   * a brute-force, mesh-only reimplementation of the stress map of eq:viscosity:rows / :shear / :weights
#     (`independent_strain`): the normal rows from the faces of each cell, the corner vertices from the cell corners, the shear
#     derivative from the pair/least-squares rule of the chapter, the wall rows of a solid; nothing of the package's
#     variable_diffusion_geometry.jl is called,
#   * small utilities (cell lists, conditional fractions, wrapped distances).

isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "momentum_audit.jl"))

module ViscosityAudit

using TwoPhaseAMR
using LinearAlgebra
using SparseArrays
using Random
using ..MomentumAudit
const AMR = TwoPhaseAMR
const MA = MomentumAudit

export cells, cell_number, cell_volume, cell_spacing, cond_fraction, open_fraction, lattice_of, vertex_point
export fixture_walled, fixture_cut, fixture_periodic, FIX_BOX, FIX_DENSITY, FIX_VISC, FIX_DT
export shear_derivative, independent_strain, wrapped, mean_fraction_mu, independent_wall_rows, cell_faces_of_axis

# ---------------------------------------------------------------------------------------------------------
# Fixtures (the drop of Chapter 8 on the walled composite mesh, the cut mesh and the periodic composite mesh).
# ---------------------------------------------------------------------------------------------------------
const FIX_BOX = 8.0
const FIX_DROP = (; centre = (4.3, 4.1), radius = 2.23)
const FIX_DENSITY = (1000.0, 1.0)
const FIX_VISC = (0.1, 0.002)
const FIX_DT = 0.05
fix_drop(x, y) = hypot(x - FIX_DROP.centre[1], y - FIX_DROP.centre[2]) - FIX_DROP.radius
fix_solid_low(x, y) = y - 0.9 - 0.1x

function fixture_walled()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (FIX_BOX, FIX_BOX), periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    return InterfaceField(mesh, fix_drop; lipschitz = 1)
end

function fixture_cut()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (FIX_BOX, FIX_BOX), periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    geometry = GeometryField(mesh, fix_solid_low; lipschitz = hypot(1, 0.1))
    return InterfaceField(mesh, fix_drop; geometry, contact_angle = pi / 2, lipschitz = 1)
end

function fixture_periodic()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (FIX_BOX, FIX_BOX), periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    return InterfaceField(mesh, fix_drop; lipschitz = 1)
end

# ---------------------------------------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------------------------------------
function cells(mesh)
    nx, ny = mesh.block_size
    return [CellRef(b, i, j) for b in eachindex(mesh.blocks) for j in 1:ny for i in 1:nx]
end
cell_number(mesh, ref) = AMR._cell_index(ref, mesh.block_size[1], mesh.block_size[2])
cell_volume(mesh, ref) = mesh.blocks[ref.block].volume
cell_spacing(mesh, ref) = mesh.blocks[ref.block].spacing
cond_fraction(field, ref) = AMR._conditional_fraction(field, ref)
open_fraction(field, ref) = isnothing(field.geometry) ? 1.0 : AMR._get(field.geometry.open_fraction, ref)
lattice_of(mesh) = mesh.base_mesh .* 2^mesh.max_level
vertex_point(mesh, key) = ntuple(k -> mesh.origin[k] + key[k] * mesh.extent[k] / lattice_of(mesh)[k], 2)

wrapped(d, extent, periodic) = periodic ? d - extent * round(d / extent) : d

mean_fraction_mu(c, mus) = exp(c * log(mus[1]) + (1 - c) * log(mus[2]))

# ---------------------------------------------------------------------------------------------------------
# Brute-force reimplementation of the strain map E of eq:viscosity:rows.
# ---------------------------------------------------------------------------------------------------------

# Derivative d_t u_k at the point v (eq:viscosity:shear), as a Dict{face index => coefficient}.  Faces are searched by brute force
# among all interior faces of axis k and all boundary faces of axis k (value zero, dropped from the row).
function shear_derivative(mesh, v, k, cmax, boundary; force_lsq = false, info = nothing)
    t = 3 - k
    tol = 128eps() * maximum(mesh.extent)
    radius = 3cmax
    pts = Tuple{Int, Float64, Float64}[]      # (face index or 0 for a boundary face, dx, dy)
    for (i, f) in pairs(mesh.faces)
        f.axis == k || continue
        dx = wrapped(f.center[1] - v[1], mesh.extent[1], mesh.periodic[1])
        dy = wrapped(f.center[2] - v[2], mesh.extent[2], mesh.periodic[2])
        dx^2 + dy^2 <= radius^2 && push!(pts, (i, dx, dy))
    end
    for f in mesh.boundary_faces
        f.axis == k || continue
        dx = f.center[1] - v[1]
        dy = f.center[2] - v[2]
        dx^2 + dy^2 <= radius^2 && push!(pts, (0, dx, dy))
    end
    wall_low = !mesh.periodic[t] && abs(v[t] - mesh.origin[t]) <= tol
    wall_high = !mesh.periodic[t] && abs(v[t] - mesh.origin[t] - mesh.extent[t]) <= tol
    out = Dict{Int, Float64}()
    if (wall_low || wall_high) && boundary isa FreeSlip
        info === nothing || (info[:kind] = :free_slip_zero)
        return out
    end
    # pair: nearest collinear unknown on each side (along t), wall point replaces the missing side
    if !force_lsq
        lo = nothing; hi = nothing
        for (i, dx, dy) in pts
            normal_offset = k == 1 ? dx : dy
            abs(normal_offset) <= tol || continue
            s = k == 1 ? dy : dx           # offset along t
            if s < -tol && (lo === nothing || s > lo[2])
                lo = (i, s)
            elseif s > tol && (hi === nothing || s < hi[2])
                hi = (i, s)
            end
        end
        has_lo = wall_low || lo !== nothing
        has_hi = wall_high || hi !== nothing
        if has_lo && has_hi
            lo_i, lo_s = wall_low ? (0, 0.0) : lo
            hi_i, hi_s = wall_high ? (0, 0.0) : hi
            d = hi_s - lo_s
            if info !== nothing
                info[:kind] = (wall_low || wall_high) ? :wall_pair : (abs(lo_s + hi_s) <= 1.0e-9 * d ? :pair_sym : :pair_asym)
                info[:points] = [(i, dx, dy) for (i, dx, dy) in pts]
                info[:pair] = (lo_i, lo_s, hi_i, hi_s)
            end
            lo_i > 0 && (out[lo_i] = get(out, lo_i, 0.0) - 1 / d)
            hi_i > 0 && (out[hi_i] = get(out, hi_i, 0.0) + 1 / d)
            return out
        end
    end
    # weighted least squares on the (at most 16) nearest unknowns, exact for quadratics, else affine
    sort!(pts; by = p -> (p[2]^2 + p[3]^2, p[2], p[3], iszero(p[1])))
    length(pts) > 16 && resize!(pts, 16)
    for degree in (2, 1)
        ncol = degree == 2 ? 6 : 3
        length(pts) >= ncol || continue
        A = zeros(length(pts), ncol)
        w = zeros(length(pts))
        for (r, (i, dx, dy)) in enumerate(pts)
            x, y = dx / cmax, dy / cmax
            w[r] = 1 / (1 + x^2 + y^2)
            row = degree == 2 ? [1.0, x, y, x^2, x * y, y^2] : [1.0, x, y]
            A[r, :] = w[r] .* row
        end
        sv = svdvals(A)
        minimum(sv) > 1.0e-10 * maximum(sv) || continue
        target = zeros(ncol); target[t + 1] = 1 / cmax
        c = pinv(A') * target                  # minimum-norm solution of A' c = target
        if info !== nothing
            info[:kind] = degree == 2 ? :lsq2 : :lsq1
            info[:points] = copy(pts)
            info[:weights] = copy(w)
        end
        for (r, (i, dx, dy)) in enumerate(pts)
            i > 0 && (out[i] = get(out, i, 0.0) + w[r] * c[r])
        end
        return out
    end
    error("insufficient faces")
end

"""
Strain map `E` (sparse, rows: 2 per cell [cell order, axis], one per corner vertex with positive open volume [sorted vertex key],
then wall rows of an embedded solid [(face, conductance, cell) list given by the caller]) and the measures of the rows, from the
mesh, the conditional fractions and the open fractions only.  `boundary` is `NoSlip()` or `FreeSlip()`.  Returns
`(E, measure, logmu_at_row, vertex_keys)`; the wall rows are NOT included (the caller appends them).
"""
function independent_strain(field, mus; boundary = NoSlip())
    mesh = field.mesh
    geometry = field.geometry
    cs = cells(mesh)
    n = length(cs)
    lat = lattice_of(mesh)
    rows = Int[]; cols = Int[]; vals = Float64[]
    measure = Float64[]
    logmu = Float64[]
    lmu(ref) = begin
        c = cond_fraction(field, ref)
        c * log(mus[1]) + (1 - c) * log(mus[2])
    end
    for ref in cs, axis in 1:2
        a = open_fraction(field, ref)
        push!(measure, 2a * cell_volume(mesh, ref))
        push!(logmu, lmu(ref))
    end
    for (i, f) in pairs(mesh.faces)
        for (ref, sgn) in ((f.minus, 1), (f.plus, -1))
            a = open_fraction(field, ref)
            a > 0 || continue
            aper = isnothing(geometry) ? 1.0 : geometry.aperture[i]
            push!(rows, 2 * (cell_number(mesh, ref) - 1) + f.axis); push!(cols, i)
            push!(vals, sgn * aper * f.area / (a * cell_volume(mesh, ref)))
        end
    end
    # vertices
    verts = Dict{NTuple{2, Int}, Vector{Tuple{CellRef, Float64}}}()
    for ref in cs
        x = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
        h = cell_spacing(mesh, ref)
        for sx in (-1, 1), sy in (-1, 1)
            key = ntuple(2) do k
                val = round(Int, (x[k] + (k == 1 ? sx : sy) * h[k] / 2 - mesh.origin[k]) / mesh.extent[k] * lat[k])
                mesh.periodic[k] ? mod(val, lat[k]) : val
            end
            push!(get!(verts, key, Tuple{CellRef, Float64}[]), (ref, cell_volume(mesh, ref) / 4))
        end
    end
    vkeys = sort!(collect(keys(verts)))
    used = NTuple{2, Int}[]
    for key in vkeys
        cl = verts[key]
        vol = sum(w * open_fraction(field, ref) for (ref, w) in cl)
        vol > 0 || continue
        push!(used, key)
        push!(measure, vol)
        push!(logmu, sum(w * open_fraction(field, ref) * lmu(ref) for (ref, w) in cl if open_fraction(field, ref) > 0) / vol)
        row = length(measure)
        p = vertex_point(mesh, key)
        cmax = maximum(maximum(cell_spacing(mesh, ref)) for (ref, w) in cl)
        for k in 1:2
            for (i, c) in shear_derivative(mesh, p, k, cmax, boundary)
                push!(rows, row); push!(cols, i); push!(vals, c)
            end
        end
    end
    E = sparse(rows, cols, vals, length(measure), length(mesh.faces))
    return E, measure, logmu, used
end


# ---------------------------------------------------------------------------------------------------------
# Wall rows of an embedded solid (eq:viscosity:cut, :weights): brute force from the faces of each cut cell.
# ---------------------------------------------------------------------------------------------------------

"""Faces (global index, side fraction varpi_{c,f}) of axis `k` on the sides of the cell `ref`: |f| / (width of c across k)."""
function cell_faces_of_axis(mesh, ref, k)
    t = 3 - k
    out = Tuple{Int, Float64}[]
    for (i, f) in pairs(mesh.faces)
        f.axis == k || continue
        (f.minus == ref || f.plus == ref) || continue
        push!(out, (i, f.area / cell_spacing(mesh, ref)[t]))
    end
    return out
end

"""
List of `(face, conductance, cell number, kind)` of the embedded wall rows of the field's solid, kind `:cut` (eq:viscosity:cut,
first line; `include_cut = false` for FreeSlip) or `:covered` (second line).  The PLIC segment of a cut cell is taken from the
package (`_local_plic_segment`); everything else is recomputed here.
"""
function independent_wall_rows(field; include_cut = true)
    mesh = field.mesh
    geometry = field.geometry
    rows = Tuple{Int, Float64, Int, Symbol}[]
    if include_cut
        for axis in 1:2
            for ref in cells(mesh)
                seg = AMR._local_plic_segment(geometry.solid, ref)
                seg === nothing && continue
                center = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
                spacing = cell_spacing(mesh, ref)
                p0 = AMR._local_to_physical(Float64, center, spacing, seg[1])
                p1 = AMR._local_to_physical(Float64, center, spacing, seg[2])
                mid = ((p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
                len = hypot(p1[1] - p0[1], p1[2] - p0[2])
                for (i, frac) in cell_faces_of_axis(mesh, ref, axis)
                    f = mesh.faces[i]
                    d = hypot(wrapped(f.center[1] - mid[1], mesh.extent[1], mesh.periodic[1]), wrapped(f.center[2] - mid[2], mesh.extent[2], mesh.periodic[2]))
                    d = max(d, 0.25 * minimum(spacing))
                    push!(rows, (i, 0.5 * frac * len / d, cell_number(mesh, ref), :cut))
                end
            end
        end
    end
    for (i, f) in pairs(mesh.faces)
        geometry.face_dual_volumes[i] > 0 && continue
        push!(rows, (i, 1.0, cell_number(mesh, f.minus), :covered))
    end
    return rows
end

end # module ViscosityAudit
