# Regression for #78 (formerly defect D1 of the audit). Before the fix StaggeredDiffusionOperator(geometry; ...) threw
#   ArgumentError: degenerate AMR dual diffusion connection
# for a solid that cuts through a level transition (the connection between two vertically adjacent x-unknowns on the interface carried
# an x-flux weight but had zero separation along x, so `_dual_diffusion_connections` dropped it and then threw because the dual face had
# non-zero absolute measure): the minimal reproducer below (8x8 level-0 cells, only the lower-left 4x4 block refined, solid
# y < 2.3 + 0.1 x) and 16 of 29 random composite meshes with a disc were rejected. Since #78 (PR #88) the cut-cell constructor rebuilds
# the transverse connections from open-trace overlaps, and every geometry builds with an exactly symmetric stiffness.
#
# Label:  eq:viscosity:cut (also eq:viscosity:connections)
# Source: src/embedded/operators.jl (_cut_cell_diffusion_connections, _cut_cell_diffusion_component),
#         src/operators/staggered_diffusion.jl (_dual_diffusion_connections)
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-cut/scripts/defect_01_constructor.jl
using Test, Random, LinearAlgebra
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

symmetric_stiffness(operator) = all(
    component -> norm(component.stiffness - transpose(component.stiffness), Inf) == 0, (operator.x, operator.y),
)

@testset "D1: cut-cell single-phase constructor on a level transition (fixed by #78)" begin
    mesh = BlockTreeMesh(;
        base_mesh = (8, 8), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1,
        refine = (key, o, e) -> key.level == 0 && key.i == 0 && key.j == 0
    )
    geometry = GeometryField(mesh, (x, y) -> y - 2.3 - 0.1x; lipschitz = 2)
    operator = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    @test operator isa StaggeredDiffusionOperator
    @test symmetric_stiffness(operator)
    # the horizontal half plane on the same mesh builds as before
    @test StaggeredDiffusionOperator(GeometryField(mesh, (x, y) -> y - 2.3; lipschitz = 2); viscosity = 1.0) isa StaggeredDiffusionOperator
    # the variable-viscosity operator builds on the same geometry
    field = InterfaceField(mesh, (x, y) -> 1.0; geometry, lipschitz = 1)
    @test StaggeredDiffusionOperator(field; viscosity = (0.1, 0.002)) isa AMR.VariableStaggeredDiffusionOperator
    # random composite meshes with a random disc (seeds 1..30, 12x12 box, levels 1-2): none is rejected any more
    thrown = 0; total = 0; symmetric = 0
    for seed in 1:30
        rng = MersenneTwister(seed)
        periodic = Tuple(rand(rng, Bool, 2)); levels = rand(rng, 1:2)
        m = MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
        g = try
            GeometryField(m, MA.random_solid(rng, m.extent; kind = :disc); lipschitz = 2)
        catch
            continue
        end
        total += 1
        try
            symmetric_stiffness(StaggeredDiffusionOperator(g; viscosity = 1.0)) && (symmetric += 1)
        catch err
            err isa ArgumentError || rethrow()
            thrown += 1
        end
    end
    println("random disc geometries on composite meshes: ", thrown, " of ", total, " rejected, ", symmetric, " with a symmetric stiffness")
    @test total > 0
    @test thrown == 0
    @test symmetric == total
end
