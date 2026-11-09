# Minimal reproducer of a pre-existing defect (related to issue #95): with a solid that splits the open domain into two components, the one-velocity
# projection with the default tree-geometric preconditioner throws ("The linear operator `A` or the preconditioner `M` is not symmetric positive definite"),
# while the unpreconditioned (`nothing`) and ILU-preconditioned conjugate gradients converge. Single phase, no interface.
#
# Label:  eq:projection:one-gauge (sentence on disconnected domains)
# Source: docs/techref/chapters/10_projection.tex:84-96; src/solvers/multigrid (coarse level augmented by ONE constant); src/navier_stokes/projection.jl
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/repro_01_split_domain.jl
# This script PASSES while the defect is present (it asserts the error) and fails when the defect is fixed (then the text sentence should be reread).

using TwoPhaseAMR, Test
mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> false)
band(x, y) = abs(y - 4.05) - 0.62                       # fluid where > 0: a horizontal solid band across the whole box
geometry = GeometryField(mesh, band; lipschitz = 1)
op = PoissonOperator(geometry)
function project_with(pre)
    solver = EllipticSolver(op; preconditioner = pre)
    v = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? sinpi(x / 4) * cospi(y / 8) : cospi(x / 8) * sinpi(y / 4) + 0.3)
    enforce_no_permeation!(v); enforce_geometry_no_slip!(v, geometry)
    return project!(v, CellField(mesh), solver, 0.01, ProjectionWorkspace(solver), geometry; tolerance = 1.0e-10, max_cycles = 2000)
end
@testset "split domain" begin
    @test_throws ErrorException project_with(:tree_geometric)      # default preconditioner: CG reports a non-SPD operator/preconditioner
    @test project_with(nothing).converged                         # no preconditioner
    @test project_with(:ilu).converged
    println("repro_01_split_domain: the default (tree-geometric) projection throws on a domain split by a solid; CG without or with ILU converges")
end
