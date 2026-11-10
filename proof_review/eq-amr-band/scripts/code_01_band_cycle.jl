# Code check 01: issue #117 mechanism of the interface band (eq:amr:band): the ring of halo slots has the width of the block's own cells.
#
# Labels: eq:amr:band, dev:amr:band (issue #117)
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:154-170 (ring, "merged, refined again ... no fixed point"), 760-775 (measurement); src/mesh/adaptivity/phase_criteria.jl:38-61
# Checks:
#   B1  the package labels equal an independent replica of eq:amr:band on the stored conditional fractions (several drops, rings 0,1,2);
#   B2  static drop on the walled 2 x 1 mesh, target level 2, coarsen_delay 2: leaf-set history over 14 attempts: ring 1 cycles with period delay+1 = 3, ring 0 is a fixed point;
#   B3  the mechanism: for every parent produced by a merge and refined at the next attempt, the gap g (Chebyshev distance between the parent rectangle and the
#       nearest mixed cell, independent geometry) satisfies  h_p/2 <= g < h_p  (children's ring width h_p/2 does not reach, the parent's ring h_p does),
#       and the parent contains no mixed cell; with ring 0 no such parent exists;
#   B4  consequence stated in the chapter: with ring r the cycle exists for any r >= 1 (r cells of the parent are twice the width of r cells of the children).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-band/scripts/code_01_band_cycle.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

function mixed_rects(mesh, interface)
    h = mesh.halo
    nx, ny = mesh.block_size
    floor_ = interface.fraction_floor
    rects = NTuple{4, Float64}[]
    for (b, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        c = interface.conditional_fraction.blocks[b][i + h, j + h]
        floor_ < c < 1 - floor_ || continue
        dx, dy = block.spacing
        x0, y0 = block.origin[1] + (i - 1) * dx, block.origin[2] + (j - 1) * dy
        push!(rects, (x0, x0 + dx, y0, y0 + dy))
    end
    return rects
end

function gap(block, rects)
    x0, y0 = block.origin
    x1, y1 = x0 + block.extent[1], y0 + block.extent[2]
    g = Inf
    for (a0, a1, b0, b1) in rects
        gx = max(0.0, a0 - x1, x0 - a1)
        gy = max(0.0, b0 - y1, y0 - b1)
        g = min(g, max(gx, gy))
    end
    return g
end

function drop_fixture(; ring, delay, cx = 0.7, cy = 0.42, R = 0.22, max_level = 2)
    mesh = AA.mesh_with(; base = (32, 16), extent = (2.0, 1.0), periodic = (false, false), max_level, delay, halo = max(1, ring))
    interface = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - R; lipschitz = 1)
    indicator = InterfaceRefinementIndicator(interface; level = max_level, buffer_cells = ring)
    return mesh, interface, indicator, AMRWorkspace(mesh, interface)
end

@testset "B1 labels = replica" begin
    for (ring, cx, cy, R) in ((0, 0.7, 0.42, 0.22), (1, 0.9, 0.5, 0.17), (2, 1.3, 0.5, 0.2517))
        mesh, interface, indicator, workspace = drop_fixture(; ring, delay = 0, cx, cy, R)
        for step in 1:3                       # a few regrids, labels checked on every mesh
            labels = Vector{Symbol}(undef, nblocks(mesh))
            refinement_labels!(labels, indicator)
            h = mesh.halo
            nx, ny = mesh.block_size
            floor_ = interface.fraction_floor
            for (b, block) in pairs(mesh.blocks)
                v = interface.conditional_fraction.blocks[b][(h + 1 - ring):(h + nx + ring), (h + 1 - ring):(h + ny + ring)]
                near = minimum(v) < 1 - floor_ && maximum(v) > floor_
                expect = near ? (block.key.level < 2 ? :refine : block.key.level > 2 ? :coarsen : :keep) : :coarsen
                @test labels[b] == expect
            end
            adapt!(workspace, indicator; step)
        end
    end
end

function cycle_history(ring; delay = 2, steps = 14)
    mesh, interface, indicator, workspace = drop_fixture(; ring, delay)
    history = Set{BlockKey}[]
    for step in 1:steps
        adapt!(workspace, indicator; step)
        push!(history, Set(b.key for b in mesh.blocks))
    end
    return history
end

@testset "B2 period of the static band" begin
    changes(history) = count(s -> history[s] != history[s - 1], 3:length(history))
    period(history) = something(findfirst(p -> all(history[s] == history[s - p] for s in (5 + p):length(history)), 1:4), 0)
    out = Dict()
    for ring in (0, 1, 2)
        history = cycle_history(ring)
        out[ring] = (changes(history), period(history))
        println("B2 ring ", ring, ": leaf-set changes in attempts 3..14 = ", out[ring][1], ", period = ", out[ring][2])
    end
    @test out[0] == (0, 1)
    @test out[1][2] == 3 && out[1][1] >= 6
    # ring 2 on this drop: no cycle (the window of gaps exists but is not hit); see B5
end

@testset "B3 mechanism: gap between the merged parent and the interface" begin
    # replay attempt by attempt: right after a merge produces a parent P (level target-1), compute the labels; if P is labelled refine, classify the gap
    ring, delay, steps = 1, 2, 14
    mesh, interface, indicator, workspace = drop_fixture(; ring, delay)
    prev = Set(b.key for b in mesh.blocks)
    gaps = Float64[]; hp = Float64[]; contains_mixed = Bool[]
    splits_after_merge = 0
    merged_keys = Set{BlockKey}()
    for step in 1:steps
        adapt!(workspace, indicator; step)
        now = Set(b.key for b in mesh.blocks)
        # a parent created by a merge in this attempt?
        new_parents = [k for k in now if !(k in prev) && all(c -> c in prev, AA.child_keys(k))]
        # parents produced by merges in the previous attempt and refined now (their children are new leaves)
        for k in merged_keys
            if !(k in now) && all(c -> c in now, AA.child_keys(k))
                splits_after_merge += 1
            end
        end
        merged_keys = Set(new_parents)
        # classify gap of the new parents (refined at the next attempt if labelled :refine)
        if !isempty(new_parents)
            labels = Vector{Symbol}(undef, nblocks(mesh))
            refinement_labels!(labels, indicator)
            rects = mixed_rects(mesh, interface)
            for (b, block) in pairs(mesh.blocks)
                block.key in new_parents || continue
                labels[b] == :refine || continue
                g = gap(block, rects)
                h = mesh.halo; nx, ny = mesh.block_size
                inside = any(c -> 0 < c < 1, interface.conditional_fraction.blocks[b][(h + 1):(h + nx), (h + 1):(h + ny)])
                push!(gaps, g); push!(hp, block.spacing[1]); push!(contains_mixed, inside)
            end
        end
        prev = now
    end
    println("B3 parents created by a merge and labelled refine: ", length(gaps), "; gap/h_p = ", round.(gaps ./ hp; digits = 3), "; contain a mixed cell: ", contains_mixed)
    println("B3 parents that were split again after the merge: ", splits_after_merge)
    @test length(gaps) > 0 && splits_after_merge > 0
    ratio = gaps ./ hp
    @test all(0.5 - 1.0e-12 .<= ratio .< 1 + 1.0e-12)      # children's ring (h_p/2) does not reach the mixed cell, the parent's ring (h_p) does
    @test !any(contains_mixed)                          # the interface is outside the parent block: only the ring sees it
end

@testset "B3b ring 0: no merged parent is refined again" begin
    mesh, interface, indicator, workspace = drop_fixture(; ring = 0, delay = 2)
    prev = Set(b.key for b in mesh.blocks)
    resplit = 0
    merged_keys = Set{BlockKey}()
    for step in 1:14
        adapt!(workspace, indicator; step)
        now = Set(b.key for b in mesh.blocks)
        resplit += count(k -> !(k in now) && all(c -> c in now, AA.child_keys(k)), merged_keys)
        merged_keys = Set(k for k in now if !(k in prev) && all(c -> c in prev, AA.child_keys(k)))
        prev = now
    end
    println("B3b ring 0: parents split again after a merge: ", resplit)
    @test resplit == 0
end

# B5: how general is "the band has no fixed point"?  Survey of static interfaces (random drops and planar interfaces) for rings 1 and 2:
# fraction of fixtures whose leaf set is NOT periodic-1 after the initial refinement (cycle exists) -- the claim "for a static interface the band has no fixed point"
# holds only for some interfaces; ring 0 always has a fixed point.
function survey(ring, kind, n; seed = 1)
    rng = MersenneTwister(seed)
    cycling = 0
    periods = Int[]
    for _ in 1:n
        mesh = AA.mesh_with(; base = (32, 16), extent = (2.0, 1.0), periodic = (false, false), max_level = 2, delay = 2, halo = max(1, ring))
        if kind === :drop
            cx, cy, R = 0.4 + 1.2rand(rng), 0.3 + 0.4rand(rng), 0.1 + 0.15rand(rng)
            interface = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - R; lipschitz = 1)
        else
            c0, tilt = 0.3 + 0.4rand(rng), 0.3 * (rand(rng) - 0.5)
            interface = InterfaceField(mesh, (x, y) -> y - c0 - tilt * (x - 1); lipschitz = 1)
        end
        indicator = InterfaceRefinementIndicator(interface; level = 2, buffer_cells = ring)
        workspace = AMRWorkspace(mesh, interface)
        history = Set{BlockKey}[]
        for step in 1:14
            adapt!(workspace, indicator; step)
            push!(history, Set(b.key for b in mesh.blocks))
        end
        p = something(findfirst(p -> all(history[s] == history[s - p] for s in (5 + p):14), 1:4), 0)
        push!(periods, p)
        p != 1 && (cycling += 1)
    end
    return cycling, periods
end

@testset "B5 survey: fixed point or cycle for static interfaces" begin
    for ring in (0, 1, 2), kind in (:drop, :plane)
        c, p = survey(ring, kind, 12; seed = 7)
        println("B5 ring ", ring, " ", kind, ": ", c, " of 12 static interfaces have no fixed point (periods: ", sort(unique(p)), ")")
        ring == 0 && @test c == 0
    end
end
