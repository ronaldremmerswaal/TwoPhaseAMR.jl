# Audit ch:solvers -- claim 6: the restricted, row-normalised smoothed transfer (eq:solvers:active-transfer) satisfies
# P 1^act_{J+1} = 1^act_J (the null vector of the active domain: one on active cells, zero on covered cells), whereas the
# unrestricted smoothed transfer does not and sends coarse corrections into covered cells.
# Source: docs/techref/chapters/11_solvers.tex:331-352; code src/solvers/multigrid/hierarchy.jl:348-420
#   (_restrict_transfer_to_active_domain!), src/solvers/multigrid/geometric_preconditioner.jl:225-350.
# Per level: max |P 1_c - 1_f|, exact zeros in covered rows/columns, the unrestricted transfer's violation (recomputed from
# the tentative transfer and the level stiffness), A 1_act (null vector of fine and coarse operators), and the smallest row
# sum before normalisation (the code errors out on a zero sum).
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step06_transfer_null_vector.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

function transfer_report(preconditioner, name)
    ops = preconditioner.operators
    worst = 0.0; worst_raw = 0.0; min_rowsum = Inf; min_entry = Inf; covered_leak = 0
    for l in 1:(length(preconditioner.levels) - 1)
        P = preconditioner.levels[l].prolongation
        fine = preconditioner.levels[l].volumes .> 0
        coarse = preconditioner.levels[l + 1].volumes .> 0
        residual = P * Float64.(coarse) - Float64.(fine)
        worst = max(worst, norm(residual, Inf))
        # exact zeros on covered rows and columns
        rows = rowvals(P); vals = nonzeros(P)
        for j in axes(P, 2), p in nzrange(P, j)
            (!fine[rows[p]] || !coarse[j]) && @test vals[p] == 0
        end
        active_entries = [vals[p] for j in axes(P, 2) if coarse[j] for p in nzrange(P, j) if fine[rows[p]]]
        min_entry = min(min_entry, minimum(active_entries))
        # unrestricted transfer from the same tentative transfer and stiffness
        tentative = AMR._tree_cell_prolongation(ops[l].mesh, ops[l + 1].mesh, Float64)
        raw = AMR._smoothed_transfer_refresh(ops[l].stiffness, tentative).prolongation
        raw_residual = raw * Float64.(coarse) - Float64.(fine)
        worst_raw = max(worst_raw, norm(raw_residual, Inf))
        covered_leak += count(i -> !fine[i] && abs((raw * Float64.(coarse))[i]) > 1.0e-14, eachindex(fine))
        # row sums of the unrestricted transfer over active coarse columns
        sums = zeros(length(fine))
        rr = rowvals(raw); rv = nonzeros(raw)
        for j in axes(raw, 2), p in nzrange(raw, j)
            (coarse[j] && fine[rr[p]]) && (sums[rr[p]] += rv[p])
        end
        min_rowsum = min(min_rowsum, minimum(sums[fine]))
    end
    @printf(
        "  %-34s levels=%d  max|P 1c-1f|=%.1e   unrestricted: max|P 1c-1f|=%.2e, covered cells receiving a correction=%d   min row sum=%.3f min active entry=%.3f\n",
        name, length(preconditioner.levels), worst, worst_raw, covered_leak, min_rowsum, min_entry
    )
    @test worst <= 1.0e-13
    return (; worst, worst_raw, covered_leak)
end

@testset "restricted transfer preserves the active-domain null vector" begin
    r1 = transfer_report(
        let g = GeometryField(solv_uniform(64), solv_band_solid; lipschitz = 1)
            TreeGeometricMultigridPreconditioner(PoissonOperator(g); geometry = g)
        end, "split cut-cell 64^2"
    )
    r2 = transfer_report(
        let g = GeometryField(solv_amr(16), solv_band_solid; lipschitz = 1)
            TreeGeometricMultigridPreconditioner(PoissonOperator(g); geometry = g)
        end, "composite 16 + solid band"
    )
    disc = GeometryField(solv_uniform(32), (x, y) -> hypot(x - 0.5, y - 0.5) - 0.25; lipschitz = 1)   # solid disc: one fluid component
    r3 = transfer_report(TreeGeometricMultigridPreconditioner(PoissonOperator(disc); geometry = disc), "solid disc, one component, 32^2")
    # Dirichlet/decoupled inactive blob without geometry ("a coarse cell is active if one of its fine cells is active")
    mesh = solv_uniform(32)
    nx, ny = mesh.block_size
    n = AMR.ncells(mesh)
    mask = falses(n)
    for (block, leaf) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x = leaf.origin[1] + (i - 0.5) * leaf.spacing[1]; y = leaf.origin[2] + (j - 0.5) * leaf.spacing[2]
        mask[(block - 1) * nx * ny + (j - 1) * nx + i] = hypot(x - 0.4, y - 0.55) < 0.3 || (x > 0.6 && y < 0.3)
    end
    for mode in (:dirichlet, :decoupled)
        op = EllipticOperator(mesh; volumes = ifelse.(mask, AMR._cell_volumes(mesh, Float64), 0.0), active_cells = mask, inactive_condition = mode)
        transfer_report(TreeGeometricMultigridPreconditioner(op), "blob, no geometry, $(mode)")
    end
    # the null vectors themselves: A 1^act = 0 on the fine and the coarse levels when the single component is unanchored
    println("== null vector of the levels (closed domain, one fluid component) ==")
    pre = TreeGeometricMultigridPreconditioner(PoissonOperator(disc); geometry = disc)
    for (l, level) in pairs(pre.levels)
        act = Float64.(level.volumes .> 0)
        v = norm(level.matrix * act, Inf) / norm(level.matrix, Inf)
        @printf("    level %d: |A 1_act|/|A| = %.1e (%d components)\n", l, v, length(level.component_volumes))
        l == 1 && @test v <= 1.0e-13
    end
    @test r1.worst_raw > 1.0e-3 || r1.covered_leak > 0        # the unrestricted transfer does violate the property
    @test r2.worst_raw > 1.0e-3 || r2.covered_leak > 0
end
