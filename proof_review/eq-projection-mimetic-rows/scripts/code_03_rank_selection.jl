# Code check 03: the rank selection of the fitted rows ("Rank and fallback" paragraph of Section "Mimetic projection").
#
# Label:  eq:projection:kkt (assumption "the rows of N are independent, which the rank selection ensures")
# Source: docs/techref/chapters/10_projection.tex:752-767; src/two_phase/projection/mimetic_projection.jl (_local_independent_rows)
# Checks (synthetic row sets with duplicated / proportional / combined rows, random positive metrics):
#   Q1  the retained rows are linearly independent (rank in the metric D^{-1/2}) and span all candidate rows (numerical rank of the candidate set = number retained);
#   Q2  which rows are dropped: all rows are normalised to unit length in the metric D^{-1/2}, so the pivoted QR breaks ties by column order and keeps the EARLIER of
#       duplicate/dependent rows (printed experiment: the later duplicates are dropped, the retained set is sorted) -- consistent with the text.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-mimetic-rows/scripts/code_03_rank_selection.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

rng = MersenneTwister(5)
@testset "rank selection" begin
    nF = 30
    D = 0.5 .+ rand(rng, nF)                               # relative_diagonal
    base = [sprand(rng, nF, 0.15) for _ in 1:6]
    rows = Vector{Vector{Float64}}()
    for b in base[1:4]
        push!(rows, Vector(b) .+ 0.1 .* randn(rng, nF) .* (abs.(Vector(b)) .> 0))
    end
    # rows 5,6 dependent on earlier ones: duplicate (scaled) of row 1; combination of rows 2,3
    push!(rows, 3.0 .* rows[1]); push!(rows, rows[2] .- 0.5 .* rows[3])
    # a row that appears EARLY but is a combination of later ones
    pushfirst!(rows, rows[4] .+ rows[5] .* 0 .+ 0.0)       # duplicate of (the original) row 4 placed first
    Mat = sparse(reduce(vcat, (r' for r in rows)))
    retained = AMR._local_independent_rows(Mat, D, sqrt(eps()), Float64)
    dropped = setdiff(1:size(Mat, 1), retained)
    W = Mat * Diagonal(1 ./ sqrt.(D)); W = W ./ sqrt.(sum(abs2, W; dims = 2))
    rk_all = rank(Matrix(W)); rk_ret = rank(Matrix(W[retained, :]))
    spans = all(i -> norm(Matrix(W[retained, :])' * (Matrix(W[retained, :])' \ W[i, :]) - W[i, :]) <= 1.0e-10 * norm(W[i, :]), dropped)
    println("rows ", size(Mat, 1), ": numerical rank ", rk_all, ", retained ", retained, ", dropped ", dropped, "; retained rank ", rk_ret, "; dropped rows in the span of the retained: ", spans, "; retained sorted: ", issorted(retained))
    @test rk_ret == length(retained) == rk_all
    @test spans
    @test issorted(retained)
end
