# Fresh process: julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/latency/invalidations.jl
using LinearAlgebra
BLAS.set_num_threads(1)
using SnoopCompileCore
const invalidations = @snoop_invalidations using TwoPhaseAMR
# Analysis is loaded only after recording so it cannot affect the captured load.
using SnoopCompile

function report_invalidations(invalidations)
    victims = SnoopCompile.uinvalidated(invalidations)
    trees = SnoopCompile.invalidation_trees(invalidations)
    println("Julia=", VERSION, " unique invalidated MethodInstances=", length(victims),
        " (excludes method-table cache entries and Core.Compiler); invalidating trees=", length(trees))
    # Number of reachable nodes, including roots; different trees can overlap.
    size(tree) = length(tree.backedges) + length(tree.mt_backedges) + SnoopCompile.countchildren(tree)
    for tree in Iterators.take(sort!(trees; by = size, rev = true), 20)
        println("reachable=", size(tree), " ", tree.reason, " ", tree.method)
    end
end
report_invalidations(invalidations)
