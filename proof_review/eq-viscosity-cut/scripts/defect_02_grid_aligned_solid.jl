# Regression for #79: a solid on cell faces must carry the same no-slip row as
# the limit of a slightly shifted cut surface. Before the fix the first open
# u₁ rate was 0 at δ=0 and approximately -5.68 at δ=1e-3.
#
# Fixture: 16×16 level-0 cells in an 8×8 box, with the right half refined;
# solid y < 1 + δ, u₁ = 1 on open unknowns, and stationary NoSlip walls.
# The exact plane initializer and small fraction floor keep the δ=1e-9 case
# genuinely cut, rather than rounding the sliver away.
#
# Label: eq:viscosity:cut (also :rows)
# Run: julia --project=docs/techref/figures proof_review/eq-viscosity-cut/scripts/defect_02_grid_aligned_solid.jl
using Test
using TwoPhaseAMR
const AMR = TwoPhaseAMR

mesh() = BlockTreeMesh(;
    base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1,
    refine = (key, o, e) -> key.level == 0 && key.i >= 2
)

function rate_above(delta; variable = false)
    m = mesh()
    solid = InterfaceField(m; fraction_floor = 1.0e-14)
    initialize!(solid, (0.0, 1.0), 1.0 + delta)
    geometry = GeometryField(solid)
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
    has_cut = any(
        ref -> 0 < AMR._get(solid.fraction, ref) < 1,
        AMR._cell_references(m),
    )
    return rate.interior[i], has_cut
end

@testset "D2: grid-aligned embedded wall has the cut-wall limit" begin
    for variable in (false, true)
        (aligned, aligned_cut) = rate_above(0.0; variable)
        (shifted, shifted_cut) = rate_above(1.0e-9; variable)
        println(
            variable ? "variable" : "single-phase",
            ": δ=0 -> ", aligned, "; δ=1e-9 -> ", shifted,
        )
        @test !aligned_cut && shifted_cut
        @test aligned < 0 && shifted < 0
        @test isapprox(aligned, shifted; rtol = 1.0e-6, atol = 0)
    end
end
