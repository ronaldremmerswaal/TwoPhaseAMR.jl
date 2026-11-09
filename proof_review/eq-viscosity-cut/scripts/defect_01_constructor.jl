# Defect D1 (pre-existing, found by the audit): StaggeredDiffusionOperator(geometry; ...) throws
#   ArgumentError: degenerate AMR dual diffusion connection
# for a solid that cuts through a level transition (the connection between two vertically adjacent x-unknowns on the interface carries
# an x-flux weight but has zero separation along x, so `_dual_diffusion_connections` drops it and then throws because the dual face has
# non-zero absolute measure).  Minimal reproducer: 8x8 level-0 cells, only the lower-left 4x4 block refined, solid y < 2.3 + 0.1 x.
# The same geometry builds the VARIABLE operator (StaggeredDiffusionOperator(field; viscosity = (mu_l, mu_g))) without problems, and the
# uniform-mesh disc of the tests builds; the check below asserts the CURRENT behaviour (throws), so it fails as soon as the defect is fixed
# (then update the audit of eq:viscosity:cut and this script).
#
# Label:  eq:viscosity:cut
# Source: src/operators/staggered_diffusion.jl (_dual_diffusion_connections, final `throw(ArgumentError("degenerate ..."))`),
#         src/embedded/operators.jl (_cut_cell_diffusion_component)
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-cut/scripts/defect_01_constructor.jl
using Test, Random
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

@testset "D1: cut-cell single-phase constructor on a level transition" begin
    mesh = BlockTreeMesh(; base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1,
        refine = (key, o, e) -> key.level == 0 && key.i == 0 && key.j == 0)
    geometry = GeometryField(mesh, (x, y) -> y - 2.3 - 0.1x; lipschitz = 2)
    @test_throws ArgumentError StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    # the horizontal half plane and the disc on the same mesh build
    @test StaggeredDiffusionOperator(GeometryField(mesh, (x, y) -> y - 2.3; lipschitz = 2); viscosity = 1.0) isa StaggeredDiffusionOperator
    # the variable-viscosity operator builds on the offending geometry
    field = InterfaceField(mesh, (x, y) -> 1.0; geometry, lipschitz = 1)
    @test StaggeredDiffusionOperator(field; viscosity = (0.1, 0.002)) isa AMR.VariableStaggeredDiffusionOperator
    # frequency on random composite meshes with a random disc (seeds 1..30, 12x12 box, levels 1-2)
    thrown = 0; total = 0
    for seed in 1:30
        rng = MersenneTwister(seed)
        periodic = Tuple(rand(rng, Bool, 2)); levels = rand(rng, 1:2)
        m = MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
        g = try GeometryField(m, MA.random_solid(rng, m.extent; kind = :disc); lipschitz = 2) catch; continue end
        total += 1
        try
            StaggeredDiffusionOperator(g; viscosity = 1.0)
        catch err
            err isa ArgumentError && (thrown += 1)
        end
    end
    println("random disc geometries on composite meshes: ", thrown, " of ", total, " rejected")
    @test thrown > 0
end
