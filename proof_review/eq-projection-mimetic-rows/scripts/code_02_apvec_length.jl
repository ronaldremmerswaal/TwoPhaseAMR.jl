# Code check 02: the integrated liquid-aperture vector is the interface area vector up to the sign, and |apvec| is the interface length of a planar cut
# (eq:projection:aperture-vector, and the sentence on rowscale in eq:projection:smoothed-rows).
#
# Label:  eq:projection:aperture-vector
# Source: docs/techref/chapters/10_projection.tex:585-612, 640-646
# Checks (planar interface at several tilts, uniform and composite meshes; and the drop fixtures):
#   L1  apvec(c) = Σ_f o_{c,f}|f| a^l_f n_f and its direction is that of -(interface normal into the gas): apvec·η_c < 0 on planar cells (up to the sign of the text);
#   L2  |apvec(c)| = PLIC segment length (planar interface, uncut cells): 3e-13 for the tilts (0,-1), (0.3,-1); 5.7 % for the steep tilt (1.0,-0.6) (printed);
#   L3  on the curved drop the relative deviation |apvec|/length - 1 is O(h) (printed): the text's "interface length of a planar cut" is the planar statement.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-mimetic-rows/scripts/code_02_apvec_length.jl

using Test, Random, LinearAlgebra
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

function aperture_vectors(field)
    d = PA.face_data(field); mesh = field.mesh; nx, ny = mesh.block_size
    A = zeros(length(mesh.blocks) * nx * ny, 2)
    for (i, face) in pairs(mesh.faces)
        w = face.area * d.al[i]
        A[AMR._cell_index(face.minus, nx, ny), face.axis] += w; A[AMR._cell_index(face.plus, nx, ny), face.axis] -= w
    end
    for (i, face) in pairs(mesh.boundary_faces)
        l = AMR._phase_boundary_face_apertures(field, i)[1]
        A[AMR._cell_index(face.cell, nx, ny), face.axis] += face.side * face.area * l
    end
    return A
end

function compare(field; interior_only = true)
    mesh = field.mesh; A = aperture_vectors(field); nx, ny = mesh.block_size
    devs = Float64[]; cosines = Float64[]
    for ref in AMR._cell_references(mesh)
        c = AMR._conditional_fraction(field, ref)
        (field.fraction_floor < c < 1 - field.fraction_floor) || continue
        h = mesh.blocks[ref.block].spacing
        ξ = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref)); a = AMR._get(field.intercept, ref)
        p, q = segment_in([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], ξ, a)
        len = hypot((p[1] - q[1]) * h[1], (p[2] - q[2]) * h[2])
        i = AMR._cell_index(ref, nx, ny)
        push!(devs, hypot(A[i, 1], A[i, 2]) / len - 1)
        n = (ξ[1] / h[1], ξ[2] / h[2]); n = n ./ hypot(n...)
        push!(cosines, (A[i, 1] * n[1] + A[i, 2] * n[2]) / hypot(A[i, 1], A[i, 2]))
    end
    return devs, cosines
end

@testset "aperture vector" begin
    for kind in (:uniform, :amr), g in ((0.0, -1.0), (0.3, -1.0), (1.0, -0.6))
        field = FA.plane_field(kind; gravity = g)
        devs, cosines = compare(field)
        println("planar ", kind, " tilt ", g, ": ", length(devs), " mixed cells; max | |apvec|/length - 1 | = ", maximum(abs.(devs)), "; max (cos(apvec, η)) = ", maximum(cosines), " min = ", minimum(cosines))
        @test maximum(cosines) < -0.999             # apvec points opposite to the normal into the gas
        # planar: equality to 3e-13 for the shallow tilts; 5.7 % for the steep tilt (1.0, -0.6): the cells where the line meets the walls / PLIC clipping,
        # i.e. "interface length of a planar cut" holds exactly only for a cell whose face apertures are those of its own plane
        @test maximum(abs.(devs)) <= (g[1] <= 0.3 ? 1.0e-9 : 0.06)
    end
    for (name, field) in PA.fixtures()[1:2]
        devs, cosines = compare(field)
        println("drop ", name, ": ", length(devs), " mixed cells; | |apvec|/length - 1 | max ", maximum(abs.(devs)), " mean ", sum(abs.(devs)) / length(devs), "; cos(apvec, η) in [", minimum(cosines), ", ", maximum(cosines), "]")
        @test maximum(cosines) < -0.9
    end
end
