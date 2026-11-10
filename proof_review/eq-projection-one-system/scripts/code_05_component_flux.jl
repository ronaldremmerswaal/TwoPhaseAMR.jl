# Code check 05: the compatibility condition of the one-velocity projection is per connected component (sharpness of eq:projection:one-gauge).
#
# Label:  eq:projection:one-gauge  (sentences: "the prescribed normal flux must have zero total over every connected component", "a nonzero b̄ is distributed over all components")
# Source: docs/techref/chapters/10_projection.tex:84-118 and 26-58; src/solvers/multigrid/cycles.jl (_project_integrated_rhs!)
# Checks (band solid splitting the box in two components; ILU-preconditioned CG, the only converging choice there):
#   Q1  a right-hand side whose integral vanishes globally but not per component is NOT made compatible by the global mean removal: removed mean = 0, the strong
#       residual stays larger than the right-hand side (it cannot be reduced), and yet `converged = true`: the round-off floor grows with the diverged iterate
#       (DEFECT D2, asserted while present);
#   Q2  the removed mean is subtracted from every cell proportionally to its volume (all components), as printed;
#   Q3  with zero flux per component the solve converges.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_05_component_flux.jl

using Test, Random, LinearAlgebra
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

band = PA.band_fixture(:uniform)
mesh = band.mesh; nx, ny = mesh.block_size
op = FA.one_velocity_operator(band)
vol = PA.open_volumes(band)
refs = AMR._cell_references(mesh)
ycen = [cellcenter(mesh, r.block, r.i, r.j)[2] for r in refs]
lower = [c for c in eachindex(vol) if vol[c] > 0 && ycen[c] < 4]
upper = [c for c in eachindex(vol) if vol[c] > 0 && ycen[c] > 4]
solver = EllipticSolver(op; preconditioner = :ilu)
@testset "per-component compatibility" begin
    # Q1: unequal component sums, zero total
    strong = zeros(length(vol)); α = 1.0
    strong[lower] .= α; strong[upper] .= -α * sum(vol[lower]) / sum(vol[upper])
    @test abs(dot(vol, strong)) <= 1.0e-12
    sol1 = CellField(mesh)
    st = solve!(sol1, solver, FA.cell_field_of(strong, mesh); rtol = 0.0, atol = 1.0e-8, max_cycles = 300)
    pmax = maximum(abs, FA.cell_vector(ref -> AMR._get(sol1, ref), mesh))
    println(
        "Q1: component sums ±", α * sum(vol[lower]), ": removed mean ", st.removed_rhs_mean, ", converged ", st.converged, ", final residual (strong, inf) ", st.final_linf,
        ", stop level (floor) ", st.target_linf, ", max|p| ", pmax
    )
    @test abs(st.removed_rhs_mean) <= 1.0e-12
    # DEFECT D2 (documented, asserted while present): the inconsistent system is reported as CONVERGED, because the round-off floor scales with the (diverged) iterate:
    # the strong residual is larger than the right-hand side itself, yet final_linf <= target_linf
    @test st.final_linf > 10 * α
    @test st.converged                    # flips to a failure when the defect is fixed (then the sentence on per-component flux in the text should say that the solve fails)
    # Q2: nonzero global mean is subtracted everywhere proportionally to the volume
    strong2 = copy(strong) .+ 0.25
    rhs2 = FA.cell_field_of(strong2, mesh)
    st2 = solve!(CellField(mesh), solver, rhs2; rtol = 0.0, atol = 1.0e-8, max_cycles = 300)
    println("Q2: removed mean ", st2.removed_rhs_mean, " (expected 0.25 = global volume mean of the strong rhs)")
    @test abs(st2.removed_rhs_mean - 0.25) <= 1.0e-12
    # Q3: zero sum per component
    strong3 = zeros(length(vol))
    strong3[lower] .= randn(MersenneTwister(1), length(lower)); strong3[lower] .-= dot(vol[lower], strong3[lower]) / sum(vol[lower])
    strong3[upper] .= randn(MersenneTwister(2), length(upper)); strong3[upper] .-= dot(vol[upper], strong3[upper]) / sum(vol[upper])
    st3 = solve!(CellField(mesh), solver, FA.cell_field_of(strong3, mesh); rtol = 0.0, atol = 1.0e-8, max_cycles = 500)
    println("Q3: zero flux per component: converged ", st3.converged, ", final residual ", st3.final_linf)
    @test st3.converged
    # Q4: consequence in `project!`: Inflow into the lower component only + Outflow on the whole right side (the uniform outflow correction balances the TOTAL flux,
    # not the flux of each component): the system is inconsistent per component, the solve reports converged and the end-of-step velocity is far from solenoidal
    bc = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 && y < 4 ? 0.3 : 0.0), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    v = FaceVelocity(mesh, (x, y, t, axis) -> 0.0)
    pr, st4, _ = PA.run_projection!(band, op, v; preconditioner = :ilu, velocity_boundary = bc, tolerance = 1.0e-8, max_cycles = 300)
    flux = PA.open_flux(band, v)
    dtdiv = PA.DT * maximum(abs.(flux[vol .> 0] ./ vol[vol .> 0]))
    println("Q4: project! with unbalanced component fluxes: converged ", st4.converged, ", dt max|D v| after = ", dtdiv, " (tolerance 1e-8), max|p| ", maximum(abs, FA.cell_vector(ref -> AMR._get(pr, ref), mesh)))
    @test st4.converged
    @test dtdiv > 1.0e-3
end
