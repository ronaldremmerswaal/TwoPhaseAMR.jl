# Code check 01: the open staggered volumes (7.5) against the definition (3.7), in physical coordinates.
#
# Label:  eq:cutcells:dual
# Claim:  07_cutcells.tex:134-160: V^o_f = (1/2) sum_c |c| (r_hi - r_lo) (1 - Phi(n_s, a_s; [-1/2,1/2] x [r_lo, r_hi])) equals
#         (3.7) V^o_f = (1/2) sum_c |S(c,f) ∩ c^o|; boundary faces get the half strip of their cell; "without a solid
#         V^o_f = V_f (3.3), but in general V^o_f != a_f V_f: for the face of Figure 7.2, V^o_f = 0.490, a_f V_f = 0.429".
# Code:   src/embedded/geometry.jl:184-262 (_open_subcell_area, _geometry_dual_volumes, _geometry_boundary_dual_volumes).
# Method: (i) code `geometry.face_dual_volumes` / `boundary_dual_volumes`;
#         (ii) (3.7) evaluated in PHYSICAL coordinates from the stored solid planes: the strip is the physical rectangle
#              (full cell width normal to f, extent of f tangentially), the open cell the physical half-plane
#              (n_s1/h_1) x_1 + (n_s2/h_2) x_2 >= a_s + n.x_c, areas by Sutherland--Hodgman + shoelace (no Phi, no local frame);
#         (iii) (7.5) as printed with the package's Phi (`_plane_fraction` with bounds [-1/2,1/2]x[r_lo,r_hi]).
#         Fixtures: cylinder (8x8), a 2:1-graded AMR mesh with hanging faces and an elliptic solid (anisotropic cells),
#         a solid-free geometry (V^o_f = V_f).  Covered faces, partially covered strips, boundary faces all included.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-cutcells-dual/scripts/code_01_dual_vs_package.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

"""Open strip area of cell `ref` for the face segment (centre `fc`, length `fa`, normal axis `axis`), physical coordinates."""
function open_strip_physical(geometry, ref, axis, fc, fa)
    mesh = geometry.mesh
    h = mesh.blocks[ref.block].spacing
    centre = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
    t = 3 - axis
    tlo, thi = fc[t] - fa / 2, fc[t] + fa / 2
    nlo, nhi = centre[axis] - h[axis] / 2, centre[axis] + h[axis] / 2
    rect = axis == 1 ? rect_poly(nlo, nhi, tlo, thi) : rect_poly(tlo, thi, nlo, nhi)
    full = (nhi - nlo) * (thi - tlo)
    alpha = AMR._get(geometry.solid.fraction, ref)
    alpha >= 1 && return 0.0, full
    alpha <= 0 && return full, full
    s = (AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref))
    a = AMR._get(geometry.solid.intercept, ref)
    n = (s[1] / h[1], s[2] / h[2])                       # physical normal of the solid line, relative to the cell centre
    offset = a + n[1] * centre[1] + n[2] * centre[2]
    return halfplane_area(rect, n, offset; keep_less = false), full
end

"""(7.5) as printed: (1/2) sum |c| (rhi - rlo) (1 - Phi) with the package's closed-form Phi on [-1/2,1/2] x [rlo, rhi]."""
function open_dual_75(geometry, ref_list, axis, fc, fa)
    mesh = geometry.mesh
    total = 0.0
    for ref in ref_list
        h = mesh.blocks[ref.block].spacing
        centre = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
        t = 3 - axis
        lo = (fc[t] - fa / 2 - centre[t]) / h[t]; hi = lo + fa / h[t]
        alpha = AMR._get(geometry.solid.fraction, ref)
        alpha >= 1 && continue                                                    # covered: zero contribution
        phi = 0.0
        if alpha > 0
            sx, sy, sa = AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref), AMR._get(geometry.solid.intercept, ref)
            phi = axis == 1 ? AMR._plane_fraction(sx, sy, sa, -0.5, 0.5, lo, hi) : AMR._plane_fraction(sx, sy, sa, lo, hi, -0.5, 0.5)
        end
        total += mesh.blocks[ref.block].volume * (hi - lo) * (1 - phi)
    end
    return total / 2
end

function check_geometry(geometry; label)
    mesh = geometry.mesh
    worst_code = worst_75 = worst_boundary = 0.0
    counts = Dict(:faces => 0, :hanging => 0, :partial => 0, :covered => 0, :differs_from_aperture => 0, :boundary => 0)
    max_rel_aperture = 0.0
    for (index, face) in pairs(mesh.faces)
        areas = [open_strip_physical(geometry, ref, face.axis, face.center, face.area) for ref in (face.minus, face.plus)]
        v37 = sum(a[1] for a in areas) / 2
        v75 = open_dual_75(geometry, (face.minus, face.plus), face.axis, face.center, face.area)
        code = geometry.face_dual_volumes[index]
        worst_code = max(worst_code, abs(code - v37)); worst_75 = max(worst_75, abs(v75 - v37))
        @test abs(code - v37) <= 1.0e-13 * max(1, mesh.face_dual_volumes[index])
        @test abs(v75 - v37) <= 1.0e-13 * max(1, mesh.face_dual_volumes[index])
        counts[:faces] += 1
        mesh.blocks[face.minus.block].key.level != mesh.blocks[face.plus.block].key.level && (counts[:hanging] += 1)
        full = sum(a[2] for a in areas) / 2
        abs(full - mesh.face_dual_volumes[index]) <= 1.0e-13 * full || error("strip bookkeeping disagrees with (3.3)")
        v37 < full - 1.0e-12 && (counts[:partial] += 1)
        v37 == 0 && (counts[:covered] += 1)
        # V^o_f versus a_f V_f
        a = geometry.aperture[index]
        if abs(v37 - a * mesh.face_dual_volumes[index]) > 1.0e-12
            counts[:differs_from_aperture] += 1
            max_rel_aperture = max(max_rel_aperture, abs(v37 - a * mesh.face_dual_volumes[index]) / mesh.face_dual_volumes[index])
        end
    end
    local_index = [0, 0]
    for face in mesh.boundary_faces
        local_index[face.axis] += 1
        v37 = open_strip_physical(geometry, face.cell, face.axis, face.center, face.area)[1] / 2
        code = geometry.boundary_dual_volumes[face.axis][local_index[face.axis]]
        worst_boundary = max(worst_boundary, abs(code - v37))
        @test abs(code - v37) <= 1.0e-13
        counts[:boundary] += 1
    end
    @printf("  %-24s %4d interior faces (%3d hanging, %3d partially open strip, %3d closed), %3d boundary faces; max |code - (3.7)| = %.1e, max |(7.5) - (3.7)| = %.1e, boundary %.1e;\n", label,
        counts[:faces], counts[:hanging], counts[:partial], counts[:covered], counts[:boundary], worst_code, worst_75, worst_boundary)
    @printf("  %-24s V^o_f != a_f V_f on %d faces (max |V^o - a V|/V_f = %.3f)\n", "", counts[:differs_from_aperture], max_rel_aperture)
    return counts
end

@testset "(7.5) = (3.7) on cylinder, AMR with hanging faces and anisotropic cells" begin
    # cylinder fixture of the technical reference
    mesh = AMR.BlockTreeMesh(; base_mesh = (8, 8), block_size = (8, 8), extent = (8.0, 8.0), periodic = (false, false))
    cyl = AMR.GeometryField(mesh, (x, y) -> hypot(x - 4.3, y - 3.6) - 2.6)
    c1 = check_geometry(cyl; label = "cylinder 8x8")
    @test c1[:partial] > 20
    # the numbers quoted with Figure 7.2: face between cells (6,6) and (7,6)
    minus, plus = AMR.CellRef(1, 6, 6), AMR.CellRef(1, 7, 6)
    index = findfirst(f -> f.minus == minus && f.plus == plus, mesh.faces)
    face = mesh.faces[index]
    v37 = sum(open_strip_physical(cyl, ref, face.axis, face.center, face.area)[1] for ref in (minus, plus)) / 2
    @printf("  Figure face: V^o = %.4f (source: 0.490), a_f = %.4f (source 0.43), a_f V_f = %.4f (source 0.429), V_f = %.1f\n",
        v37, cyl.aperture[index], cyl.aperture[index] * mesh.face_dual_volumes[index], mesh.face_dual_volumes[index])
    @test round(v37; digits = 3) == 0.490 && round(cyl.aperture[index]; digits = 2) == 0.43
    @test round(cyl.aperture[index] * mesh.face_dual_volumes[index]; digits = 3) == 0.429
    @test abs(v37 - cyl.face_dual_volumes[index]) <= 1.0e-14

    # AMR (2:1 graded) with hanging faces, anisotropic cells, elliptic solid
    for (label, hx, hy) in (("AMR unit cells", 1.0, 1.0), ("AMR anisotropic 1.7x0.6", 1.7, 0.6))
        window = (5.5, 10.0, 1.0, 6.0)
        refine = (key, origin, extent) -> begin
            x1, x2 = origin[1], origin[1] + extent[1]; y1, y2 = origin[2], origin[2] + extent[2]
            wx1, wx2, wy1, wy2 = window .* (hx, hx, hy, hy)
            x1 < wx2 && x2 > wx1 && y1 < wy2 && y2 > wy1
        end
        amr = AMR.BlockTreeMesh(; base_mesh = (16, 8), block_size = (4, 4), extent = (16hx, 8hy), periodic = (false, false), max_level = 2, refine)
        geometry = AMR.GeometryField(amr, (x, y) -> hypot(x / hx - 7.3, y / hy - 3.4) - 2.1)
        counts = check_geometry(geometry; label)
        @test counts[:hanging] > 10 && counts[:partial] > 20
    end

    # without a solid: V^o_f = V_f of (3.3), including hanging faces
    window = (5.0, 11.0, 1.0, 6.0)
    refine = (key, origin, extent) -> begin
        x1, x2 = origin[1], origin[1] + extent[1]; y1, y2 = origin[2], origin[2] + extent[2]
        x1 < window[2] && x2 > window[1] && y1 < window[4] && y2 > window[3]
    end
    amr = AMR.BlockTreeMesh(; base_mesh = (16, 8), block_size = (4, 4), extent = (16.0, 8.0), periodic = (false, false), max_level = 2, refine)
    none = AMR.GeometryField(amr, (x, y) -> 1.0)                       # level set > 0 everywhere: no solid
    hanging = count(f -> amr.blocks[f.minus.block].key.level != amr.blocks[f.plus.block].key.level, amr.faces)
    @test hanging > 10
    @test maximum(abs.(none.face_dual_volumes .- amr.face_dual_volumes)) <= 1.0e-14
    @test all(none.aperture .== 1)
    println("  no solid: max |V^o_f - V_f| = $(maximum(abs.(none.face_dual_volumes .- amr.face_dual_volumes))) on $(length(amr.faces)) faces ($hanging hanging)")
end
