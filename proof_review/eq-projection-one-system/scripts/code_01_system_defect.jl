# Code check 01: the one-velocity system (eq:projection:one-system) and the exact defect identity (eq:projection:one-defect).
#
# Labels: eq:projection:one-system, eq:projection:one-defect
# Source: docs/techref/chapters/10_projection.tex:60-114; src/navier_stokes/projection.jl (project!), src/embedded/operators.jl
# Checks (fixtures: uniform/composite x without/with the sloped solid, plus a band solid that splits the domain; random smooth
# predictors; tolerance 1e-12):
#   A1  S (operator.stiffness on the open cells) = J' diag(sqrt(γ)) K^o diag(sqrt(γ)) J, with γ = 1/ρ̄ (no solid) or a²/ρ̄ (solid),
#       ρ̄ from an independent staggered fraction                                                                   (one-system, matrix)
#   A2  -|c^o| D^o G_γ p = S p for random p, G_γ p = the code's weighted gradient (strong-form statement -D G p = ..., adjoint)
#   A3  the update v^{n+1}_f = v*_f - dt (R J p)_f on interior faces, v_f unchanged on boundary faces; R J p = the code's gradient
#   A4  |c^o| (D^o v^{n+1})_c = -dt (r_c + b̄ |c^o|), r = b - S p with b the compatible right-hand side, for every open cell
#   A5  covered cells: identity row, p_c = 0, no flux
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_01_system_defect.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR

function check_fixture(name, field, rng; preconditioner = :tree_geometric)
    mesh = field.mesh; geometry = field.geometry
    nx, ny = mesh.block_size
    op = FA.one_velocity_operator(field)
    n = length(mesh.blocks) * nx * ny
    d = ProjectionAudit.face_data(field)
    γ = isnothing(geometry) ? 1 ./ d.m : ProjectionAudit.aperture(field) .^ 2 ./ d.m
    @test maximum(abs.(op.face_coefficient .- γ) ./ max.(γ, 1.0e-300)) <= 1.0e-13                       # coefficient of eq:forces:coefficient
    J = ProjectionAudit.jump_dense(field)
    K = ProjectionAudit.metric_dense(op)
    S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
    vol = ProjectionAudit.open_volumes(field)
    open_cells = findall(>(0), vol)
    Sc = Matrix(op.stiffness)
    e_A1 = maximum(abs.(Sc[open_cells, open_cells] .- S[open_cells, open_cells])) / maximum(abs.(S[open_cells, open_cells]))
    # A2: -|c^o| D^o G p = S p
    p = randn(rng, n)
    pf = FA.cell_field_of(p, mesh)
    G = FA.weighted_gradient_of(field, op, p)
    flux = ProjectionAudit.open_flux(field, G)
    e_A2 = maximum(abs.(-flux[open_cells] .- (S * p)[open_cells])) / maximum(abs.((S * p)[open_cells]))
    # the gradient equals R J p of eq:forces:map with the independent R
    R = ProjectionAudit.force_map_dense(field, op, γ)
    e_R = maximum(abs.(G.interior .- R * (J * p))) / maximum(abs.(G.interior))
    # projection
    v = ProjectionAudit.one_predictor(field, rng; amplitude = 1.0)
    vstar = ProjectionAudit.copy_velocity(v)
    pr, stats, _ = ProjectionAudit.run_projection!(field, op, v; dt = ProjectionAudit.DT, preconditioner)
    @test stats.converged
    dt = ProjectionAudit.DT
    pvals = FA.cell_vector(ref -> AMR._get(pr, ref), mesh)
    # A3: the update, evaluated from the returned pressure with the independent R
    Gp = R * (J * pvals)
    e_A3 = maximum(abs.(v.interior .- (vstar.interior .- dt .* Gp))) / maximum(abs.(vstar.interior))
    e_A3b = maximum(abs.(v.boundary .- vstar.boundary))
    # A4: defect identity with the independent S, b, b̄
    b = -ProjectionAudit.open_flux(field, vstar) ./ dt           # b_c = -(1/dt)|c^o| D v*
    bbar = sum(b) / sum(vol)
    r = (b .- bbar .* vol) .- S * pvals
    lhs = ProjectionAudit.open_flux(field, v)
    rhs = -dt .* (r .+ bbar .* vol)
    scale = maximum(abs.(ProjectionAudit.open_flux(field, vstar)[open_cells]))
    e_A4 = maximum(abs.(lhs[open_cells] .- rhs[open_cells])) / scale
    e_bar = abs(bbar - stats.removed_rhs_mean) / max(abs(bbar), 1.0e-300)
    defect = dt * maximum(abs.(lhs[open_cells] ./ vol[open_cells]))
    # A5: covered cells
    covered = findall(==(0), vol)
    e_A5 = isempty(covered) ? 0.0 : maximum(abs, pvals[covered]) / maximum(abs, pvals)
    e_A5f = isempty(covered) ? 0.0 : maximum(abs, lhs[covered]) / scale
    println(
        rpad(name, 18), " A1=", round(e_A1, sigdigits = 3), " A2=", round(e_A2, sigdigits = 3), " R=", round(e_R, sigdigits = 3),
        " A3=", round(e_A3, sigdigits = 3), "/", round(e_A3b, sigdigits = 3), " A4=", round(e_A4, sigdigits = 3), " b̄=", round(bbar, sigdigits = 3),
        " (code ", round(stats.removed_rhs_mean, sigdigits = 3), ") defect=", round(defect, sigdigits = 3), " A5=", round(e_A5, sigdigits = 3), " A5f=", round(e_A5f, sigdigits = 3), " covered=", length(covered)
    )
    return (; e_A1, e_A2, e_R, e_A3, e_A3b, e_A4, defect, e_A5, e_A5f, bbar, vol_scale = maximum(vol))
end

rng = MersenneTwister(20261110)
results = Dict()
@testset "one-velocity system and defect identity" begin
    cases = vcat(
        [(n, f, :tree_geometric) for (n, f) in ProjectionAudit.fixtures()],
        [("band uniform (ilu)", ProjectionAudit.band_fixture(:uniform), :ilu), ("band composite (ilu)", ProjectionAudit.band_fixture(:amr), :ilu)]
    )
    for (name, field, pre) in cases
        r = check_fixture(name, field, rng; preconditioner = pre)
        results[name] = r
        @test r.e_A1 <= 1.0e-12
        @test r.e_A2 <= 1.0e-12
        @test r.e_R <= 1.0e-12
        @test r.e_A3 <= 1.0e-12
        @test r.e_A3b == 0.0
        @test r.e_A4 <= 1.0e-8        # relative to the predictor's divergence: bounded by solver rounding, see A4 values
        @test r.e_A5 <= 1.0e-9         # covered cells: p_c = 0 up to the Krylov tolerance (relative to max |p|)
        @test r.e_A5f <= 1.0e-12
    end
end
