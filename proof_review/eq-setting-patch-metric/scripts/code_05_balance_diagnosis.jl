# Code check 05: diagnosis of the grading defect of code_04 by a runtime patch (experiment).
#
# Label:  eq:setting:patch-metric
# Code:   src/mesh/topology/mesh.jl: _tree_box_intersects (466-473), _root_window_indices (500-516),
#         both used by _balance!'s neighbour search (605-646).
# Claim of the experiment: the missing 2:1 refinement of code_04 is caused by the two exact floating
#         point tests of the neighbour search (a block that touches a neighbour only up to one ulp
#         is treated as disjoint). With a relative tolerance (1e-9 of the node / root width) in
#         both functions NO mesh of the scan below keeps a face of size ratio > 2.
# This script REDEFINES the two internal functions in the running session and restores the
# original source text at the end; it must not run concurrently with other checks (run_code.jl
# executes it last). It does not edit any file. (The scans use `invokelatest`: a redefinition
# inside one top-level expression is otherwise invisible to calls made from that expression.)
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/code_05_balance_diagnosis.jl

import TwoPhaseAMR   # not `using`: the package also exports Cell and Face
using Random: MersenneTwister
using Test
const AMR = TwoPhaseAMR

function source_of(name)
    path = joinpath(@__DIR__, "..", "..", "..", "src", "mesh", "topology", "mesh.jl")
    lines = readlines(path)
    start = findfirst(l -> occursin("function $name(", l), lines)
    start === nothing && return nothing
    stop = findnext(l -> l == "end", lines, start)
    return join(lines[start:stop], "\n")
end

function max_face_ratio(mesh)
    ratio = 1.0
    for f in mesh.faces
        a = mesh.blocks[f.minus.block].spacing[f.axis]
        b = mesh.blocks[f.plus.block].spacing[f.axis]
        ratio = max(ratio, a / b, b / a)
    end
    return ratio
end

function scan(extents; seeds = 1:40, periodics = ((false, false), (true, true)))
    result = Dict{Float64, Tuple{Int, Int}}()
    for extent in extents
        bad = total = 0
        for periodic in periodics, seed in seeds
            rng = MersenneTwister(seed)
            refine = (key, o, e) -> rand(rng) < (0.5, 0.45, 0.4, 0.35)[min(key.level + 1, 4)]
            mesh = AMR.BlockTreeMesh(;
                base_mesh = (8, 8), block_size = (2, 2), extent = (extent, extent),
                periodic = periodic, max_level = 3, refine = refine
            )
            total += 1
            max_face_ratio(mesh) > 2 + 1.0e-9 && (bad += 1)
        end
        result[extent] = (bad, total)
    end
    return result
end

const EXTENTS = (0.1, 0.3, 0.7, 3.3, 0.37 * 8, 5.353681183789219, 1.0, 2pi)
# extents for which no violation has been observed in the original source (an observation, not a theorem)
const NICE_EXTENTS = (1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 5.0, 10.0, 12.0, 100.0, 1.0e-3, Float64(pi), 2 * Float64(pi))

@testset "code_05: diagnosis of the balance defect" begin
    original_box = source_of("_tree_box_intersects")
    original_root = source_of("_root_window_indices")
    if original_box === nothing || original_root === nothing
        @info "the neighbour-search helpers changed or were renamed; experiment not applicable"
    else
        before = Base.invokelatest(scan, EXTENTS)
        println("original source:")
        for e in EXTENTS
            println("  extent = ", rpad(e, 20), before[e][1], " of ", before[e][2], " meshes with a face ratio > 2")
        end
        try
            Base.eval(
                AMR, :(
                    @inline function _tree_box_intersects(low, high, node)
                        origin = node.boundary.origin
                        upper = origin + node.boundary.widths
                        tol = 1.0e-9 * maximum(node.boundary.widths)
                        return !(
                            upper[1] < low[1] - tol || high[1] < origin[1] - tol ||
                                upper[2] < low[2] - tol || high[2] < origin[2] - tol
                        )
                    end
                )
            )
            Base.eval(
                AMR, :(
                    @inline function _root_window_indices(roots::RootGrid, low, high, axis)
                        domain_low = roots.origin[axis]
                        domain_high = domain_low + roots.extent[axis]
                        tol = 1.0e-9 * roots.root_extent[axis]
                        high[axis] < domain_low - tol && return 1:0
                        low[axis] > domain_high + tol && return 1:0
                        first_root = clamp(
                            ceil(Int, (low[axis] - domain_low - tol) / roots.root_extent[axis]) - 1,
                            0, roots.layout[axis] - 1,
                        )
                        last_root = clamp(
                            floor(Int, (high[axis] - domain_low + tol) / roots.root_extent[axis]),
                            0, roots.layout[axis] - 1,
                        )
                        return first_root:last_root
                    end
                )
            )
            patched = Base.invokelatest(scan, EXTENTS)
            println("with a relative tolerance 1e-9 in both functions:")
            for e in EXTENTS
                println("  extent = ", rpad(e, 20), patched[e][1], " of ", patched[e][2], " meshes with a face ratio > 2")
            end
            @test all(patched[e][1] == 0 for e in EXTENTS)           # the tolerance removes every violation
            @test before[1.0][1] == 0 && before[2pi][1] == 0         # binary-friendly extents were never affected
        finally
            Base.eval(AMR, Meta.parse(original_box))
            Base.eval(AMR, Meta.parse(original_root))
        end
        after = Base.invokelatest(scan, EXTENTS)
        @test after == before                                         # original definitions restored
        nice = Base.invokelatest(scan, NICE_EXTENTS; seeds = 1:60, periodics = ((false, false), (true, true), (true, false)))
        println("extents that were never affected (original source, 180 meshes each): ", join(string.(NICE_EXTENTS), ", "))
        @test all(nice[e][1] == 0 for e in NICE_EXTENTS)
    end
end
