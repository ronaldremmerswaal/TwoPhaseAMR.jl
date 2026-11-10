# Code check 01: the interpolation-error indicator under coarsening (eq:amr:interpolation): factor 4 and the hysteresis condition 4 E_coarsen <= E_refine.
#
# Labels: eq:amr:interpolation, eq:amr:error-labels
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:222-250 (indicator), 713-720 (measured factor 3.83); src/mesh/adaptivity/indicators.jl:refinement_labels!
# Checks:
#   I1  the package labels equal an independent replica of eq:amr:interpolation + eq:amr:error-labels on random smooth and noisy fields;
#   I2  algebra: for a harmonic sin(k.x) the ratio of the second differences at spacing 2h and h is 4 cos^2(k h / 2) <= 4 (1D identity
#       u_{i+2}+u_{i-2}-2u_i = D_{i+1}+D_{i-1}+2 D_i with D the second difference); the max-norm ratio of a smooth field tends to 4 from below;
#   I3  measured ratio by actual coarsening: E of the merged block (cell-average restriction through adapt!) over E of its four children, for smooth
#       fields on the fixture of the chapter (1 + 0.5 sin 2 pi x cos 2 pi y; chapter value 3.83 for E(h)/E(h/2) on uniform meshes 32 and 64);
#   I4  hysteresis: a static smooth field, E_coarsen just above max E(children): the family merges; if E_refine < ratio*E_coarsen the merged block is
#       refined again (leaf set cycles with period 2), if E_refine >= 4 E_coarsen it stays merged; the threshold is the measured ratio (< 4), so 4 E_coarsen <= E_refine
#       is sufficient and nearly necessary for smooth data.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-interpolation/scripts/code_01_ratio_hysteresis.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

# independent replica of the block error (the chapter's formula) with the halo-filled values of the package
function block_errors(field; floor = sqrt(eps()), boundary = ZeroGradient())
    mesh = field.mesh
    fill_halo!(field, boundary)
    nx, ny = mesh.block_size
    h = mesh.halo
    errors = zeros(nblocks(mesh))
    for b in 1:nblocks(mesh)
        v = field.blocks[b]
        for j in (h + 1):(h + ny), i in (h + 1):(h + nx)
            nb = (v[i - 1, j], v[i + 1, j], v[i, j - 1], v[i, j + 1])
            errors[b] = max(errors[b], abs(v[i, j] - sum(nb) / 4) / max(floor, abs(v[i, j]), maximum(abs, nb)))
        end
    end
    return errors
end

function replica_label(level, E, Er, Ec, lmin, lmax)
    level < lmin && return :refine
    level > lmax && return :coarsen
    E > Er && level < lmax && return :refine
    E < Ec && (level == 0 || level > lmin) && return :coarsen
    return :keep
end

@testset "I1 labels = replica" begin
    rng = MersenneTwister(3)
    mesh = BlockTreeMesh(;
        base_mesh = (32, 32), block_size = (8, 8), extent = (1.0, 1.0), max_level = 2, periodic = (true, true),
        refine = (key, origin, extent) -> key.level == 0 ? key.i + key.j < 4 : key.i < 3
    )
    for amp in (0.0, 0.3), kx in (1, 2, 3)
        field = CellField(mesh, (x, y) -> 1 + 0.5sinpi(2kx * x) * cospi(2y))
        for b in 1:nblocks(mesh)
            field.blocks[b] .+= amp .* randn(rng, size(field.blocks[b])) .* 0.05
        end
        ind = InterpolationErrorIndicator(field; E_refine = 0.02, E_coarsen = 0.006, maximum_level = 2)
        labels = Vector{Symbol}(undef, nblocks(mesh))
        refinement_labels!(labels, ind)
        E = block_errors(field)
        for (b, block) in pairs(mesh.blocks)
            @test labels[b] == replica_label(block.key.level, E[b], 0.02, 0.006, 0, 2)
        end
    end
end

@testset "I2 second-difference ratio" begin
    # exact 1D identity
    rng = MersenneTwister(1)
    u = randn(rng, 12)
    D(i) = u[i + 1] + u[i - 1] - 2u[i]
    for i in 3:10
        @test u[i + 2] + u[i - 2] - 2u[i] ≈ D(i + 1) + D(i - 1) + 2D(i)
    end
    # harmonic: ratio of second differences = 4 cos^2(theta/2)
    for theta in (0.05, 0.2, 0.6, 1.2)
        d(h) = 2(1 - cos(h))
        @test d(2theta) / d(theta) ≈ 4cos(theta / 2)^2
        @test 4cos(theta / 2)^2 <= 4
    end
end

# E of a smooth field on a uniform mesh with n x n cells (cell-centre samples; the chapter's `defect`)
function uniform_E(n, f)
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, true))
    return maximum(block_errors(CellField(mesh, f)))
end

@testset "I3 measured factor" begin
    f = (x, y) -> 1 + 0.5sinpi(2x) * cospi(2y)
    r = uniform_E(32, f) / uniform_E(64, f)
    println("I3 chapter fixture: E(32)/E(64) = ", r, "  (chapter: 3.83)")
    @test round(r; digits = 2) == 3.83
    # finer pairs: the ratio tends to 4 from below
    ratios = [uniform_E(n, f) / uniform_E(2n, f) for n in (16, 32, 64, 128)]
    println("I3 ratios n -> 2n for n = 16, 32, 64, 128: ", ratios, "  (4cos^2(pi h) limit 4)")
    @test all(<(4), ratios) && issorted(ratios)
    # actual coarsening through adapt!: restriction by the cell mean, E of the merged block over the max of its four children
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, true), max_level = 1,
        adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = 0)
    )
    adapt!(AMRWorkspace(mesh), fill(:refine, nblocks(mesh)); step = 1, force = true)
    # cell AVERAGES of f on the fine mesh (exact, separable), the restriction then is exact averaging
    avg(x, y, hx, hy) = 1 + 0.5 * (cos(2pi * (x - hx / 2)) - cos(2pi * (x + hx / 2))) / (2pi * hx) * (sin(2pi * (y + hy / 2)) - sin(2pi * (y - hy / 2))) / (2pi * hy)
    field = CellField(mesh, (x, y) -> 0.0)
    for (b, block) in pairs(mesh.blocks), j in 1:8, i in 1:8
        x, y = cellcenter(mesh, b, i, j)
        field.blocks[b][i + mesh.halo, j + mesh.halo] = avg(x, y, block.spacing...)
    end
    E1 = block_errors(field)
    fam = Dict{BlockKey, Vector{Float64}}()
    for (b, block) in pairs(mesh.blocks)
        push!(get!(fam, AA.parent_key(block.key), Float64[]), E1[b])
    end
    workspace = AMRWorkspace(mesh, field)
    adapt!(workspace, fill(:coarsen, nblocks(mesh)); step = 2, force = true)
    E0 = block_errors(field)
    ratios = Float64[]
    for (b, block) in pairs(mesh.blocks)
        push!(ratios, E0[b] / maximum(fam[block.key]))
    end
    println("I3 coarsening by adapt!: E(merged)/max E(children): min ", minimum(ratios), " max ", maximum(ratios))
    @test all(<(4.0), ratios) && minimum(ratios) > 3.3      # n = 32 -> 16: the same as the point-sample ratio 3.40 at n = 16
end

@testset "I4 hysteresis cycle on a static smooth field" begin
    f = (x, y) -> 1 + 0.5sinpi(2x) * cospi(2y)
    function run(Er_factor, Ec_factor; steps = 8)
        mesh = BlockTreeMesh(;
            base_mesh = (16, 16), block_size = (8, 8), extent = (1.0, 1.0), periodic = (true, true), max_level = 1,
            adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = 0)
        )
        adapt!(AMRWorkspace(mesh), fill(:refine, nblocks(mesh)); step = 1, force = true)
        field = CellField(mesh, f)
        E1 = maximum(block_errors(field))
        Ec = Ec_factor * E1
        Er = Er_factor * Ec
        indicator = InterpolationErrorIndicator(field; E_refine = Er, E_coarsen = Ec, maximum_level = 1)
        workspace = AMRWorkspace(mesh, field)
        leaves = Int[]
        changes = 0
        prev = Set(b.key for b in mesh.blocks)
        for step in 2:(steps + 1)
            adapt!(workspace, indicator; step, force = true)
            now = Set(b.key for b in mesh.blocks)
            changes += now != prev
            prev = now
            push!(leaves, nblocks(mesh))
        end
        return (; E1, Ec, Er, leaves, changes)
    end
    # E_coarsen just above the children's maximum: every family is below E_coarsen and merges
    # E1 = max E of the children, E0 = 3.35 E1 (16 -> 32 cells; the ratio grows towards 4 with resolution, 3.83 at 32 -> 64)
    A = run(1.5, 1.05)      # E_refine = 1.5 E_coarsen  < E0 : the merged block is refined again at the next attempt
    D = run(3.0, 1.05)      # E_refine = 3.0 E_coarsen  < E0 = 3.19 E_coarsen : refined again
    E = run(3.4, 1.05)      # E_refine = 3.4 E_coarsen  > E0 : stays merged although 4 E_coarsen > E_refine (the factor 4 is conservative)
    B = run(3.9, 1.05)
    C = run(4.2, 1.05)      # E_refine >= 4 E_coarsen
    for (name, r) in (("A  E_r = 1.5 E_c", A), ("D  E_r = 3.0 E_c", D), ("E  E_r = 3.4 E_c", E), ("B  E_r = 3.9 E_c", B), ("C  E_r = 4.2 E_c", C))
        println("I4 ", name, ": leaf counts per attempt ", r.leaves, "  leaf-set changes ", r.changes)
    end
    # merge at attempt 1 (4 blocks) and refinement again at attempt 2 (16 blocks): one wasted merge/split pair
    @test A.leaves[1:2] == [4, 16] && D.leaves[1:2] == [4, 16]
    @test E.changes == 1 && B.changes == 1 && C.changes == 1 && C.leaves[end] == 4   # merged once, then a fixed point (4 blocks = the roots)
    # the sufficient condition of the chapter: 4 E_coarsen <= E_refine never re-refines (C); the exact threshold is E_refine >= E0 = ratio * max E(children)
end
