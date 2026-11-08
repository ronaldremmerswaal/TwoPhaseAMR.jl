# Sharpness 02: is the 2:1 grading needed? (square cells throughout, hx = hy)
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305; hypothesis "2:1-graded quadtree".
# Results (model.jl, exact integers; F = 4T/h^2):
#   * FACE grading is what the proof uses: if every face segment at v joins cells of side ratio <= 2 then
#     F12 = F21, even when the diagonal cells differ by 4:1 (corner balance of _balance! is not needed);
#   * a single 4:1 face contact breaks the symmetry although all cells are square:
#       - hanging vertex at a quarter point of a side of length 4:  F = [10 0; -4 2], T21 = -h^2 != T12 = 0;
#       - corner pattern (NE,NW,SW,SE) = (4,2,1,1):               F12 - F21 = -2.
#   So the hypothesis cannot be dropped (no mesh of the package is of this kind: BlockTreeMesh
#   refines until balanced, `isbalanced` is asserted in code_01).
# Run:    julia --heap-size-hint=3G -t 1 proof_review/eq-setting-patch-metric/scripts/sharp_02_grading.jl

using Test
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@testset "sharp_02: grading" begin
    # 4:1 hanging vertex: coarse cell W = [-4,0] x [0,4], fine cells of side 1 east of it, v = (0,1)
    cells = [Cell(-4, 0, 4), Cell(0, 1, 1), Cell(0, 0, 1)]
    faces = faces_at_vertex(cells, 0, 1)
    F = moment_integers(faces)
    @test length(faces) == 3
    @test !face_graded(faces)
    @test F == [10 0; -4 2]
    println("4:1 quarter-point hanging vertex: F = ", F, "  (T12 = ", F[1, 2] / 4, ", T21 = ", F[2, 1] / 4, " in units h^2)")

    # the midpoint 2:1 hanging vertex for comparison (symmetric)
    F21 = moment_integers(faces_at_vertex(hanging_cells(:W), 0, 0))
    @test F21 == [6 0; 0 2]

    # corner pattern with a 4:1 face contact: NE = 4, NW = 2, SW = 1, SE = 1 (NE/SE = 4)
    cells = corner_cells(4, 2, 1, 1)
    faces = faces_at_vertex(cells, 0, 0)
    F = moment_integers(faces)
    @test !face_graded(faces)
    @test F[1, 2] - F[2, 1] == -2
    println("corner (4,2,1,1): F = ", F)

    # face-graded corner patterns with a 4:1 DIAGONAL (no corner balance) stay symmetric
    for pattern in ((4, 2, 1, 2), (1, 2, 4, 2), (2, 4, 2, 1), (4, 4, 2, 2), (2, 2, 4, 4), (2, 4, 4, 2))
        faces = faces_at_vertex(corner_cells(pattern...), 0, 0)
        F = moment_integers(faces)
        @test face_graded(faces)
        @test F[1, 2] == F[2, 1]
    end
    # all face-graded corner patterns with sides in {1,2,4}: symmetric
    n = 0
    for a in (1, 2, 4), b in (1, 2, 4), c in (1, 2, 4), d in (1, 2, 4)
        faces = faces_at_vertex(corner_cells(a, b, c, d), 0, 0)
        face_graded(faces) || continue
        n += 1
        F = moment_integers(faces)
        @test F[1, 2] == F[2, 1]
    end
    println("face-graded corner patterns with sides in {1,2,4}: ", n, " (all symmetric)")
end
