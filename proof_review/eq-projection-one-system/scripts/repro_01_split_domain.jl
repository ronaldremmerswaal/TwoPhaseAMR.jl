# Regression for issue #95: a solid splits the open domain into two components.
# The default tree-geometric, unpreconditioned and ILU-preconditioned
# one-velocity projections must all converge.
#
# Label:  eq:projection:one-gauge (sentence on disconnected domains)
# Source: docs/techref/chapters/10_projection.tex; src/solvers/multigrid
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/repro_01_split_domain.jl

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
    @test project_with(:tree_geometric).converged
    @test project_with(:geometric).converged
    @test project_with(nothing).converged
    @test project_with(:ilu).converged
    println("repro_01_split_domain: all four preconditioner choices converged")
end
