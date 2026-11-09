# Defect D2 (pre-existing, found by the audit): a solid whose boundary lies exactly on cell faces has no cut cell, hence no embedded wall
# row: the no-slip condition (eq:viscosity:cut, first line; eq:viscosity:rows, wall rows) is not imposed on it, tangentially the wall is
# stress free.  A shift of the solid surface by 1e-3 cell widths makes the same unknown feel a strong no-slip row.
#
# Reproducer: 16x16 level-0 cells in an 8x8 box (right half refined), solid y < 1 + delta, u_1 = 1 everywhere, NoSlip.  The rate of the
# unknown u_1 at (2.0, 1.25) (first open row above the solid; covered unknowns set to 0) is 0 for delta = 0 and -5.68 for delta = 1e-3
# (single-phase operator).  The variable-viscosity operator imposes the condition only through the covered unknown, i.e. over a full cell
# instead of the clipped distance: the printed rates differ by a factor (see output).
#
# Label:  eq:viscosity:cut (also :rows)
# Source: src/embedded/operators.jl (_embedded_wall_connections: only cells with `_local_plic_segment`; covered unknowns tied to 0),
#         src/embedded/operators.jl (_embedded_variable_wall_stresses)
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-cut/scripts/defect_02_grid_aligned_solid.jl
using Test
using TwoPhaseAMR
const AMR = TwoPhaseAMR

mesh() = BlockTreeMesh(;
    base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1,
    refine = (key, o, e) -> key.level == 0 && key.i >= 2
)

function rate_above(delta; variable = false)
    m = mesh()
    geometry = GeometryField(m, (x, y) -> y - 1.0 - delta; lipschitz = 1)
    u = FaceVelocity(m, (x, y, t, axis) -> axis == 1 ? 1.0 : 0.0)
    for i in eachindex(m.faces)
        geometry.face_dual_volumes[i] == 0 && (u.interior[i] = 0.0)       # covered unknowns hold the solid velocity 0 (enforce_geometry_no_slip!)
    end
    rate = FaceVelocity(m)
    if variable
        field = InterfaceField(m, (x, y) -> 1.0; geometry, lipschitz = 1)
        diffusion!(rate, u, StaggeredDiffusionOperator(field; viscosity = (0.1, 0.002)))
    else
        diffusion!(rate, u, StaggeredDiffusionOperator(geometry; viscosity = 1.0))
    end
    i = findfirst(f -> f.axis == 1 && abs(f.center[1] - 2.0) < 1.0e-9 && abs(f.center[2] - 1.25) < 1.0e-9, m.faces)
    return rate.interior[i]
end

@testset "D2: grid-aligned solid face carries no no-slip row" begin
    r0, r1 = rate_above(0.0), rate_above(1.0e-3)
    println("single-phase: rate at (2.0, 1.25): delta = 0 -> ", r0, ";  delta = 1e-3 -> ", r1)
    @test r0 == 0.0
    @test r1 < -1.0
    v0, v1 = rate_above(0.0; variable = true), rate_above(1.0e-3; variable = true)
    println("variable:     rate at (2.0, 1.25): delta = 0 -> ", v0, ";  delta = 1e-3 -> ", v1)
    @test abs(v0) > 0        # variable operator: the covered unknown acts as a (weaker/farther) wall
end
