# Code check 05: compatibility and pressure gauges are per connected open component.
#
# Label:  eq:projection:one-gauge
# Source: src/solvers/multigrid/elliptic_solver.jl and src/embedded/operators.jl
# Checks (a solid band splits the box into two components):
#   Q1  globally balanced, component-incompatible data are rejected by :error;
#       :project removes and reports each mean and produces bounded pressure.
#   Q2  the global mean remains reported, alongside the different component means.
#   Q3  nonzero data compatible on each component converge to a zero-mean gauge.
#   Q4  globally balanced outflow need not balance each component: the pressure
#       solve reports the removed means, while the physical flux defect remains.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_05_component_flux.jl

using Test, Random, LinearAlgebra
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit: AMR, FA
using TwoPhaseAMR
const PA = ProjectionAudit

band = PA.band_fixture(:uniform)
mesh = band.mesh
op = FA.one_velocity_operator(band)
vol = PA.open_volumes(band)
refs = AMR._cell_references(mesh)
ycen = [cellcenter(mesh, r.block, r.i, r.j)[2] for r in refs]
lower = [c for c in eachindex(vol) if vol[c] > 0 && ycen[c] < 4]
upper = [c for c in eachindex(vol) if vol[c] > 0 && ycen[c] > 4]
components = (lower, upper)
anchors = sort([minimum(cells) for cells in components])
component_means(values) = [dot(vol[cells], values[cells]) / sum(vol[cells]) for cells in components]
ordered_means(means) = [only(mean for (cells, mean) in zip(components, means) if minimum(cells) == anchor) for anchor in anchors]
cell_values(field) = FA.cell_vector(ref -> AMR._get(field, ref), mesh)
solver = EllipticSolver(op; preconditioner = :ilu)
@testset "per-component compatibility" begin
    # Q1: unequal component sums, zero total.
    strong = zeros(length(vol))
    alpha = 1.0
    strong[lower] .= alpha
    strong[upper] .= -alpha * sum(vol[lower]) / sum(vol[upper])
    rhs = FA.cell_field_of(strong, mesh)
    @test abs(dot(vol, strong)) <= 1.0e-12
    @test_throws ArgumentError solve!(CellField(mesh), solver, rhs; compatibility = :error)
    sol1 = CellField(mesh)
    st = solve!(sol1, solver, rhs; rtol = 0.0, atol = 1.0e-8, max_cycles = 300)
    pmax = maximum(abs, cell_values(sol1))
    println(
        "Q1: component sums ±", alpha * sum(vol[lower]), ": removed global mean ", st.removed_rhs_mean,
        ", component means ", st.removed_rhs_component_means, ", converged ", st.converged,
        ", final residual (strong, inf) ", st.final_linf, ", stop level ", st.target_linf, ", max|p| ", pmax,
    )
    # Constant component sources project to zero up to volume-sum roundoff.
    @test abs(st.removed_rhs_mean) <= 1.0e-12
    @test st.removed_rhs_component_anchors == anchors
    @test st.removed_rhs_component_means ≈ ordered_means(component_means(strong)) atol = 1.0e-12 rtol = 0
    @test st.converged && st.final_linf <= 1.0e-8
    @test st.target_linf <= 1.0e-8
    @test pmax <= 1.0e-8

    # Q2: adding a global offset changes both component means by that offset.
    strong2 = strong .+ 0.25
    st2 = solve!(CellField(mesh), solver, FA.cell_field_of(strong2, mesh); rtol = 0.0, atol = 1.0e-8, max_cycles = 300)
    println("Q2: removed global mean ", st2.removed_rhs_mean, "; component means ", st2.removed_rhs_component_means)
    @test abs(st2.removed_rhs_mean - 0.25) <= 1.0e-12
    @test st2.removed_rhs_component_anchors == anchors
    @test st2.removed_rhs_component_means ≈ ordered_means(component_means(strong2)) atol = 1.0e-12 rtol = 0
    @test maximum(abs, solver.right_hand_side) <= 1.0e-12
    @test st2.converged && st2.final_linf <= 1.0e-8

    # Q3: seeded, nonzero data with zero volume integral on each component.
    strong3 = zeros(length(vol))
    strong3[lower] .= randn(MersenneTwister(1), length(lower))
    strong3[upper] .= randn(MersenneTwister(2), length(upper))
    for cells in components
        strong3[cells] .-= dot(vol[cells], strong3[cells]) / sum(vol[cells])
    end
    sol3 = CellField(mesh)
    st3 = solve!(sol3, solver, FA.cell_field_of(strong3, mesh); compatibility = :error, rtol = 0.0, atol = 1.0e-8, max_cycles = 500)
    applied = CellField(mesh)
    apply!(applied, op, sol3, EllipticWorkspace(op))
    actual_residual = maximum(abs, (cell_values(applied) - strong3)[vol .> 0])
    p3 = cell_values(sol3)
    println("Q3: zero flux per component: converged ", st3.converged, ", final residual ", st3.final_linf)
    # The independent strong residual allows sparse/matrix-free rounding at the
    # requested 1e-8 solve tolerance; gauge rounding scales with |p|.
    @test st3.converged && st3.final_linf <= 1.0e-8
    @test actual_residual <= 2.0e-8
    @test maximum(abs, st3.removed_rhs_component_means) <= 1.0e-12
    @test maximum(abs, component_means(p3)) <= 256eps() * max(1.0, maximum(abs, p3))
    @test all(iszero, p3[iszero.(vol)])

    # Q4: outflow balances total flux, not each compartment's flux. Capture
    # the actual pre-solve means; reimposing Outflow after projection may change
    # the boundary flux, so check the final physical flux independently.
    bc = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 && y < 4 ? 0.3 : 0.0), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    v = FaceVelocity(mesh, (x, y, t, axis) -> 0.0)
    enforce_velocity_boundary!(v, bc; geometry = band.geometry)
    enforce_geometry_no_slip!(v, band.geometry)
    initial_flux = PA.open_flux(band, v)
    initial_means = [-sum(initial_flux[cells]) / (PA.DT * sum(vol[cells])) for cells in components]
    pressure = CellField(mesh)
    projection_solver = EllipticSolver(op; preconditioner = :ilu)
    st4 = project!(v, pressure, projection_solver, PA.DT, ProjectionWorkspace(projection_solver), band.geometry; velocity_boundary = bc, tolerance = 1.0e-8, max_cycles = 300)
    flux = PA.open_flux(band, v)
    dtdiv = PA.DT * maximum(abs.(flux[vol .> 0] ./ vol[vol .> 0]))
    physical_means = [sum(flux[cells]) / sum(vol[cells]) for cells in components]
    p4 = cell_values(pressure)
    println("Q4: project! with unbalanced component fluxes: converged ", st4.converged, ", removed component means ", st4.removed_rhs_component_means, ", dt max|D v| after = ", dtdiv, " (tolerance 1e-8), max|p| ", maximum(abs, p4))
    @test st4.converged
    @test st4.final_linf <= st4.target_linf
    @test abs(st4.removed_rhs_mean) <= 1.0e-10
    @test st4.removed_rhs_component_anchors == anchors
    @test st4.removed_rhs_component_means ≈ ordered_means(initial_means) atol = 1.0e-10 rtol = 0
    @test minimum(abs, initial_means) > 1.0e-3
    @test abs(sum(flux)) <= 1.0e-12
    @test minimum(abs, physical_means) > 1.0e-6
    @test dtdiv > 1.0e-8
    @test dtdiv >= PA.DT * maximum(abs, physical_means) * (1 - 128eps())
    # Density contrast and box length make pressure O(1e4), not O(1e19).
    @test maximum(abs, p4) < 1.0e7
    @test maximum(abs, component_means(p4)) <= 256eps() * max(1.0, maximum(abs, p4))
    @test all(iszero, p4[iszero.(vol)])
end
