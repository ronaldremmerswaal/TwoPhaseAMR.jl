# Code check 03: hypotheses of the lemma, probed on the package.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305; hypotheses: 2:1 grading, periodic wrap, positive definiteness.
# Code:   src/operators/gradient.jl:74-77 (_wrapped_delta), 139-184 (_local_inverse_metric),
#         src/mesh/topology/mesh.jl:605-646 (_balance!).
# Checks: (a) the package's _local_inverse_metric applied to an UNGRADED 4:1 patch (square cells)
#             symmetrises a non-symmetric tensor and loses affine exactness: the grading hypothesis
#             is really used. (BlockTreeMesh cannot produce such patches: it balances.)
#         (b) on binary-friendly extents (1.0, 2.0, 2pi, ...) BlockTreeMesh refines until 2:1
#             balanced, however aggressive the refinement function. (For other floating-point
#             extents the balancing can miss contacts: code_04 / code_05.)
#         (c) _wrapped_delta returns the covering-space displacement iff |d| < L/2; at |d| = L/2
#             (a periodic direction with two level-0 cells) the sign of a seam displacement is lost.
#         (d) the behaviour of GradientOperator on periodic directions with two level-0 cells.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/code_03_assumptions.jl

import TwoPhaseAMR   # not `using`: the package also exports Cell and Face (names used by model.jl)
using LinearAlgebra: eigvals, Symmetric
using Test
const AMR = TwoPhaseAMR
include(joinpath(@__DIR__, "code_helpers.jl"))

@testset "code_03: hypotheses" begin
    @testset "(a) ungraded 4:1 patch through _local_inverse_metric" begin
        # coarse cell W = [-4,0] x [0,4], fine cells of side 1 east of it, v = (0,1); h = 1
        ideal = faces_at_vertex([Cell(-4, 0, 4), Cell(0, 1, 1), Cell(0, 0, 1)], 0, 1)
        faces = [AMR.MetricFace{Float64}(i, f.axis, Float64(f.len), (f.d2[1] / 2, f.d2[2] / 2)) for (i, f) in enumerate(ideal)]
        Y, N = moment_matrices(faces)
        T = transpose(Y) * N
        @test T ≈ [5 / 2 0; -1 1 / 2]
        @test T[1, 2] != T[2, 1]
        R = AMR._local_inverse_metric(copy(faces))      # symmetrises T, does not throw (sym(T) is PD)
        @test minimum(eigvals(Symmetric((T + transpose(T)) / 2))) > 0
        residual = maximum(abs, R * Y - N)
        println("4:1 patch: T = ", T, ",  max|R Y - N| = ", residual)
        @test residual > 0.05                           # affine exactness lost
        # the same configuration at 2:1 (midpoint hanging vertex) is exact
        ideal2 = faces_at_vertex(hanging_cells(:W), 0, 0)
        faces2 = [AMR.MetricFace{Float64}(i, f.axis, Float64(f.len), (f.d2[1] / 2, f.d2[2] / 2)) for (i, f) in enumerate(ideal2)]
        Y2, N2 = moment_matrices(faces2)
        @test maximum(abs, AMR._local_inverse_metric(copy(faces2)) * Y2 - N2) <= 1.0e-14
    end

    @testset "(b) BlockTreeMesh is graded on binary-friendly extents" begin
        for (periodic, refine) in (
                ((false, false), (key, o, e) -> key.i == 0 && key.j == 0),                 # deep corner refinement
                ((true, true), (key, o, e) -> key.i == 0 && key.j == 0),                   # ... across the seams
                ((true, false), (key, o, e) -> key.level < 4 && (key.i + key.j) % 3 == 0),
            )
            mesh = AMR.BlockTreeMesh(;
                base_mesh = (8, 8), block_size = (2, 2), extent = (8.0, 8.0),
                periodic = periodic, max_level = 4, refine = refine,
            )
            levels = [block.key.level for block in mesh.blocks]
            @test maximum(levels) >= 2 && AMR.isbalanced(mesh)
            ratio = maximum(
                max(mesh.blocks[f.minus.block].spacing[f.axis] / mesh.blocks[f.plus.block].spacing[f.axis],
                    mesh.blocks[f.plus.block].spacing[f.axis] / mesh.blocks[f.minus.block].spacing[f.axis])
                    for f in mesh.faces
            )
            @test ratio <= 2 + 1.0e-12
        end
    end

    @testset "(c) _wrapped_delta" begin
        L = 1.0
        for d in range(-0.49, 0.49; length = 99), m in (-2, -1, 0, 1, 2)
            @test AMR._wrapped_delta(d + m * L, L, true) ≈ d atol = 1.0e-12
        end
        # at |d| = L/2 the covering-space displacement is ambiguous: the sign of the argument is kept
        @test AMR._wrapped_delta(0.5, L, true) == 0.5
        @test AMR._wrapped_delta(-0.5, L, true) == -0.5
    end

    @testset "(d) a periodic direction with exactly two level-0 cells" begin
        # seam faces get displacement -h instead of +h (round-half-even in _wrapped_delta),
        # T11 <= 0 and the constructor throws; with >= 4 cells everything is fine
        mesh = AMR.BlockTreeMesh(; base_mesh = (2, 4), block_size = (2, 2), extent = (2.0, 4.0), periodic = (true, false))
        seam = [f for f in mesh.faces if f.axis == 1 && abs(f.center[1] - mesh.origin[1]) < 1.0e-12]
        inner = [f for f in mesh.faces if f.axis == 1 && abs(f.center[1] - 1.0) < 1.0e-12]
        @test !isempty(seam) && !isempty(inner)
        @test all(f -> AMR._face_displacement(mesh, f)[1] == -1.0, seam)     # true covering-space displacement: +1
        @test all(f -> AMR._face_displacement(mesh, f)[1] == +1.0, inner)
        for (base, periodic) in (((2, 2), (true, true)), ((2, 4), (true, false)), ((4, 2), (false, true)), ((2, 6), (true, false)))
            m = AMR.BlockTreeMesh(; base_mesh = base, block_size = (2, 2), extent = (Float64(base[1]), Float64(base[2])), periodic = periodic)
            @test_throws ArgumentError AMR.GradientOperator(m)
        end
        for (base, periodic) in (((4, 4), (true, true)), ((4, 4), (true, false)), ((6, 6), (true, true)), ((2, 6), (false, false)))
            m = AMR.BlockTreeMesh(; base_mesh = base, block_size = (2, 2), extent = (Float64(base[1]), Float64(base[2])), periodic = periodic)
            @test AMR.GradientOperator(m) isa AMR.GradientOperator
        end
        # the same two-cell direction is harmless where the stripe is refined (cells of level 1 across)
        refined = AMR.BlockTreeMesh(; base_mesh = (2, 2), block_size = (2, 2), extent = (2.0, 2.0),
            periodic = (true, true), max_level = 1, refine = (key, o, e) -> key.level == 0)
        @test AMR.GradientOperator(refined) isa AMR.GradientOperator
    end

    @testset "(e) faces with one wall endpoint: diagonal closure, exact only if conforming" begin
        # the wall endpoint contributes (1/2)(g.d)/d_k, the complete endpoint (1/2) g_k, so
        # the gradient of an affine pressure is g_k + (1/2) g_t d_t / d_k: exact iff d_t = 0
        mesh = random_mesh(3; block_size = (2, 2), roots = (3, 3), max_level = 2, periodic = (false, false))
        operator = AMR.GradientOperator(mesh)
        gvec = (0.3, -0.7)
        pressure = AMR.CellField(mesh, (x, y) -> gvec[1] * x + gvec[2] * y)
        gradient = AMR.FaceVelocity(mesh)
        AMR.gradient!(gradient, pressure, operator)
        n_conforming = n_hanging = 0
        for (index, face) in pairs(mesh.faces)
            ends = [AMR._vertex_key(mesh, face, e) for e in 1:2]
            count(k -> AMR._physical_vertex(mesh, k), ends) == 1 || continue
            d = AMR._face_displacement(mesh, face)
            k = face.axis
            t = 3 - k
            @test gradient.interior[index] ≈ gvec[k] + gvec[t] * d[t] / (2 * d[k]) atol = 1.0e-12
            if abs(d[t]) < 1.0e-12
                n_conforming += 1
            else
                n_hanging += 1
            end
        end
        println("wall-adjacent faces: ", n_conforming, " conforming (exact), ", n_hanging, " coarse-fine (error g_t d_t/(2 d_k), e.g. 0.7/6)")
        @test n_conforming > 0 && n_hanging > 0
    end
end
