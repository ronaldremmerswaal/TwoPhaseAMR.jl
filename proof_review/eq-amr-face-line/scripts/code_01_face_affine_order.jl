# Code check 01: face transfer exactness for affine data and its order for smooth data (eq:amr:face-line, eq:amr:face-bracket).
#
# Labels: eq:amr:face-line, eq:amr:face-bracket
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:547-583; src/mesh/adaptivity/field_transfer.jl:246-385
# Checks:
#   F1  affine data are reproduced exactly at EVERY face (interior and boundary, all three branches: f in g, f containing several g, bracketing lines) on random
#       2:1-graded walled meshes, for FaceField{1}, FaceField{2} and for FaceVelocity (whose divergence correction vanishes for affine data), re-initialised
#       exactly before every regrid (non-periodic mesh: an affine field is not periodic);
#   F3  order for smooth periodic data under uniform refinement (RMS error at the new faces vs the face-centre value): second order for FaceField and for
#       the divergence-corrected FaceVelocity;  the limited slopes only act near extrema.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-face-line/scripts/code_01_face_affine_order.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

affine1(x, y) = 1.0 + 2.0x - 3.0y
affine2(x, y) = -2.0 + 4.0x + 1.5y

function set_affine!(fx, fy, u)
    mesh = u.mesh
    for (faces, vals) in ((mesh.faces, u.interior), (mesh.boundary_faces, u.boundary))
        for (k, face) in pairs(faces)
            x, y = face.center
            vals[k] = face.axis == 1 ? affine1(x, y) : affine2(x, y)
        end
    end
    local_indices = [0, 0]
    for (k, face) in pairs(mesh.faces)
        local_indices[face.axis] += 1
        (face.axis == 1 ? fx : fy).interior[local_indices[face.axis]] = u.interior[k]
    end
    local_indices .= 0
    for (k, face) in pairs(mesh.boundary_faces)
        local_indices[face.axis] += 1
        (face.axis == 1 ? fx : fy).boundary[local_indices[face.axis]] = u.boundary[k]
    end
end

function affine_error(fx, fy, u)
    mesh = u.mesh
    worst_u = worst_f = 0.0
    for (faces, vals) in ((mesh.faces, u.interior), (mesh.boundary_faces, u.boundary))
        for (k, face) in pairs(faces)
            x, y = face.center
            worst_u = max(worst_u, abs(vals[k] - (face.axis == 1 ? affine1(x, y) : affine2(x, y))))
        end
    end
    for (field, axis, g) in ((fx, 1, affine1), (fy, 2, affine2))
        for (faces, vals) in ((mesh.faces, field.interior), (mesh.boundary_faces, field.boundary))
            k = 0
            for face in faces
                face.axis == axis || continue
                k += 1
                x, y = face.center
                worst_f = max(worst_f, abs(vals[k] - g(x, y)))
            end
        end
    end
    return worst_u, worst_f
end

@testset "F1 affine exactness on random walled meshes" begin
    for seed in 1:4
        rng = MersenneTwister(seed)
        mesh = AA.walled_mesh()
        u = FaceVelocity(mesh, Float64)
        fx = FaceField{1}(mesh, (x, y) -> 0.0); fy = FaceField{2}(mesh, (x, y) -> 0.0)
        workspace = AMRWorkspace(mesh, u, fx, fy)
        worst_u = worst_f = 0.0
        nfaces = 0
        for step in 1:8
            set_affine!(fx, fy, u)
            labels = AA.random_labels(rng, mesh; pr = step <= 3 ? 0.55 : 0.3, pc = step <= 3 ? 0.1 : 0.6)
            adapt!(workspace, labels; step, force = true).changed || continue
            wu, wf = affine_error(fx, fy, u)
            worst_u = max(worst_u, wu); worst_f = max(worst_f, wf)
            nfaces += length(mesh.faces) + length(mesh.boundary_faces)
        end
        println("F1 seed ", seed, ": faces checked ", nfaces, "; max |u - affine|: FaceVelocity ", worst_u, ", FaceFields ", worst_f)
        @test worst_u <= 1e-12 && worst_f <= 1e-12
    end
end

@testset "F3 order for smooth periodic data (uniform refinement)" begin
    g1(x, y) = sin(2pi * x + 0.3) * cos(4pi * y - 0.2) + 0.5
    g2(x, y) = cos(2pi * x - 0.1) * sin(2pi * y + 0.4) - 0.2
    function new_face_errors(n; velocity)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, true), max_level = 1,
            adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = 0))
        u = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? g1(x, y) : g2(x, y))
        fx = FaceField{1}(mesh, g1); fy = FaceField{2}(mesh, g2)
        workspace = velocity ? AMRWorkspace(mesh, u) : AMRWorkspace(mesh, fx, fy)
        adapt!(workspace, fill(:refine, nblocks(mesh)); step = 1, force = true)
        s2 = 0.0; m = 0
        if velocity
            for (k, face) in pairs(mesh.faces)
                x, y = face.center
                e = u.interior[k] - (face.axis == 1 ? g1(x, y) : g2(x, y))
                s2 += e^2; m += 1
            end
        else
            for (field, axis, g) in ((fx, 1, g1), (fy, 2, g2))
                k = 0
                for face in mesh.faces
                    face.axis == axis || continue
                    k += 1
                    x, y = face.center
                    s2 += (field.interior[k] - g(x, y))^2; m += 1
                end
            end
        end
        return sqrt(s2 / m)
    end
    for velocity in (false, true)
        errs = [new_face_errors(n; velocity) for n in (16, 32, 64, 128)]
        overall = log2(errs[1] / errs[end]) / 3
        println("F3 ", velocity ? "FaceVelocity (with divergence correction)" : "FaceFields", ": RMS error at n = 16, 32, 64, 128: ", errs, "; overall order ", overall)
        @test overall > 1.8
    end
end
