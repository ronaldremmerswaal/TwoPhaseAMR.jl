# Code check 01: curvature indicator thresholds and the hysteresis condition (dev:amr:curvature, eq:amr:curvature).
#
# Labels: dev:amr:curvature, eq:amr:curvature
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:190-260; src/mesh/adaptivity/phase_criteria.jl:111-129, 273-327
# Checks:
#   K1  thresholds: the package labels equal the replica "refine iff n max(h)|kappa| > 1 (level < max), coarsen iff n max(h)|kappa| < rho" on static drops of several radii;
#       in the units of the thesis (max(h)|kappa|): eps_refine = 1/n, eps_coarsen = rho/n, so 2 eps_coarsen < eps_refine <=> 2 rho < 1 (n > 0);
#   K2  the constructor accepts exactly 0 < 2 rho < 1;
#   K3  the scaling behind the condition: the error n h |kappa| is first order in h, so coarsening multiplies it by ~2 (measured by an actual merge through adapt!,
#       curvature re-estimated on the merged mesh): E(merged) / max E(children) for several radii;
#   K4  consequence: for a static circle, children below rho merge; the merged block is NOT refined again iff E(merged) <= 1: guaranteed by 2 rho < 1 only if the curvature
#       estimate scales exactly like h; the measured spread of the ratio around 2 is the margin that is needed.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/dev-amr-curvature/scripts/code_01_curvature_hysteresis.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

function block_curvature_errors(interface; n = 8.0)
    mesh = interface.mesh
    reconstruct!(interface)
    errors = zeros(nblocks(mesh))
    unresolved = falses(nblocks(mesh))
    for (b, block) in pairs(mesh.blocks)
        c = interior(interface, b)
        κ = interior(interface.curvature, b)
        for j in axes(c, 2), i in axes(c, 1)
            0 < c[i, j] < 1 || continue
            if !isfinite(κ[i, j])
                unresolved[b] = true
                continue
            end
            errors[b] = max(errors[b], maximum(block.spacing) * abs(κ[i, j]) * n)
        end
    end
    return errors, unresolved
end

function drop_mesh(R, level; cx = 0.5, cy = 0.5, delay = 0)
    mesh = AA.mesh_with(; base = (16, 16), extent = (1.0, 1.0), periodic = (true, true), max_level = 2, delay, block = (8, 8))
    for _ in 1:level
        adapt!(AMRWorkspace(mesh), fill(:refine, nblocks(mesh)); step = 1, force = true)
    end
    interface = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - R; lipschitz = 1)
    return mesh, interface
end

@testset "K1 labels = replica of the thresholds" begin
    for (R, level, rho, n) in ((0.2, 2, 1 / 3, 8.0), (0.15, 1, 0.45, 8.0), (0.25, 2, 0.1, 12.0), (0.3, 1, 0.3, 4.0))
        mesh, interface = drop_mesh(R, level)
        indicator = CurvatureRefinementIndicator(interface; cells_per_radius = n, coarsen_ratio = rho, minimum_level = 0, maximum_level = 2)
        labels = Vector{Symbol}(undef, nblocks(mesh))
        refinement_labels!(labels, indicator)
        errors, unresolved = block_curvature_errors(interface; n)
        kinds = Set{Symbol}()
        for (b, block) in pairs(mesh.blocks)
            e = unresolved[b] ? Inf : errors[b]
            expect = e > 1 && block.key.level < 2 ? :refine : e < rho ? :coarsen : :keep
            @test labels[b] == expect
            push!(kinds, labels[b])
        end
        println("K1 R = ", R, " level ", level, " rho = ", rho, " n = ", n, ": label kinds ", sort(collect(kinds)), "; error range of blocks with mixed cells ", extrema(filter(>(0), errors)))
    end
end

@testset "K2 constructor accepts exactly 0 < 2 rho < 1" begin
    mesh, interface = drop_mesh(0.2, 1)
    for (rho, ok) in ((0.01, true), (1 / 3, true), (0.4999, true), (0.5, false), (0.6, false), (1.0, false), (0.0, false), (-0.1, false))
        if ok
            @test CurvatureRefinementIndicator(interface; coarsen_ratio = rho).coarsen_ratio == rho
        else
            @test_throws ArgumentError CurvatureRefinementIndicator(interface; coarsen_ratio = rho)
        end
    end
    # equivalence in the thesis' variables: eps_refine = 1/n, eps_coarsen = rho/n
    for n in (2.0, 8.0, 40.0), rho in (0.1, 0.3, 0.4999, 0.5, 0.7)
        @test (2rho / n < 1 / n) == (2rho < 1)
    end
end

@testset "K3/K4 scaling of the error under a merge (static circle)" begin
    ratios = Float64[]         # per merged family with mixed cells: E(parent) / max E(children)
    totals = Float64[]
    for R in (0.12, 0.15, 0.18, 0.2, 0.23, 0.27, 0.31)
        mesh, interface = drop_mesh(R, 2; delay = 0)
        e1, u1 = block_curvature_errors(interface)
        all(!, u1) || continue
        by_key1 = Dict(block.key => e1[b] for (b, block) in pairs(mesh.blocks))
        workspace = AMRWorkspace(mesh, interface)
        adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 2, force = true)     # merge every family; curvature re-estimated on the merged mesh
        e0, u0 = block_curvature_errors(interface)
        all(!, u0) || continue
        fam = Float64[]
        for (b, block) in pairs(mesh.blocks)
            e0[b] > 0 || continue
            kids = maximum(by_key1[k] for k in AA.child_keys(block.key))
            push!(fam, e0[b] / kids)
        end
        append!(ratios, fam)
        push!(totals, maximum(e0) / maximum(e1))
        println("K3 R = ", R, ": E(level 2) max = ", round(maximum(e1); digits = 4), ", E(merged) max = ", round(maximum(e0); digits = 4), ", per-family ratios in [", round(minimum(fam); digits = 3), ", ", round(maximum(fam); digits = 3), "]")
    end
    println("K3 per-family ratios E(merged)/max E(children): min ", minimum(ratios), ", max ", maximum(ratios), " (first-order scaling predicts 2);  max-over-mesh ratios ", round.(totals; digits = 3))
    @test length(ratios) >= 20 && length(totals) >= 4
    @test all(1.5 < r < 2.6 for r in ratios)
    # the default rho = 1/3 leaves a safety factor 1/(2 rho) = 1.5 over the ideal scaling; the largest observed ratio / 2 is the needed margin
    println("K4 largest observed ratio/2 = ", maximum(ratios) / 2, ": rho needs 2 rho * ", maximum(ratios) / 2, " < 1, i.e. rho < ", 1 / maximum(ratios))
end
