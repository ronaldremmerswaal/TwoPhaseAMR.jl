# Step 04: vertices on walls never carry a moment tensor; locality of the patch.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:244-248 (wall vertices use the diagonal half-face metric) and
#         the scope of 304-305 ("every patch of type (patch-metric)").
# Code:   src/operators/gradient.jl:214-217 (_physical_vertex), 280-283 (diagonal closure if
#         wall vertex or <= 2 faces), 139-184 (_local_inverse_metric only for the rest).
# Checks: (a) a vertex on a wall (cells only on one side, sides in {h, 2h} with 2:1 grading, also
#             the two cells of different size) has exactly ONE interior face segment; a wall
#             corner has none. So wall patches have <= 2 faces and take the diagonal closure;
#         (b) the faces ending at a vertex depend only on the cells that touch it: a vertex one
#             lattice unit away from a wall (all touching cells inside the domain) carries an
#             ordinary patch (here: a hanging vertex whose coarse cell touches the wall).
# Run:    julia --heap-size-hint=3G -t 1 proof_review/eq-setting-patch-metric/scripts/step04_wall_vertices.jl

using Test
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@testset "step04: wall vertices" begin
    for wall in (:left, :right, :bottom, :top), s1 in (1, 2), s2 in (1, 2)
        cells = wall_cells(wall, s1, s2)
        faces = faces_at_vertex(cells, 0, 0)
        @test length(faces) == 1            # only the face between the two wall-adjacent cells
        @test length(faces) <= 2            # => diagonal closure, `_local_inverse_metric` is not called
    end
    # wall corner: a single cell, no interior face at all
    for cell in (Cell(0, 0, 1), Cell(0, 0, 2), Cell(-1, 0, 1), Cell(0, -2, 2), Cell(-2, -2, 2))
        @test isempty(faces_at_vertex([cell], 0, 0))
    end
    # a vertex one lattice unit above a bottom wall (wall at y = 0, v = (0, 1)): the coarse cell
    # on the west touches the wall and v is the midpoint of its east side (aligned: y in 2Z).
    # All cells around v lie in the domain, so the patch is an ordinary hanging patch.
    cells = [Cell(-2, 0, 2), Cell(0, 1, 1), Cell(0, 0, 1)]
    faces = faces_at_vertex(cells, 0, 1)
    F = moment_integers(faces)
    @test length(faces) == 3 && F[1, 2] == F[2, 1] == 0
end
