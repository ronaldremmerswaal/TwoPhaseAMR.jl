# Code check 01: catalogue spot check, row "Capillary wave, La 3000, DR 1, one velocity (8, 16, 32)" of tab:verif:benchmarks.
#
# Label:  tab:verif:benchmarks (row 3 of the random sample, see audit.md)
# Source: docs/techref/chapters/14_verification.tex (table row: errors 0.0646, 0.0126, 0.0043; end 1.96; asserted decreasing, end >= 1.8),
#         test/integration/two_phase_waves.jl ("Prosperetti standing waves on uniform and AMR meshes", studies[1] = (1.0, :capillary, 1.8))
# Checks: the helpers of the test file (everything before its first @testset, read from the file and evaluated here) reproduce the
#         tabulated errors to the printed digits, the endpoint rate log2(e1/e3)/2 is the "end" rate of the text, and the asserted
#         bounds hold.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/sec-verif-catalogue-spotcheck/scripts/code_01_capillary_row.jl

using Test
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TESTFILE = joinpath(ROOT, "test", "integration", "two_phase_waves.jl")
text = read(TESTFILE, String)
head = text[1:(first(findfirst("@testset \"Two-phase wave integration\"", text)) - 1)]
# the test file includes the Prosperetti reference through a path relative to itself
head = replace(head, "joinpath(@__DIR__, \"..\", \"..\", \"examples\"" => "joinpath(\"$(ROOT)\", \"examples\"")
include_string(Main, head, TESTFILE)

@testset "capillary wave DR=1, one velocity" begin
    resolutions = (8, 16, 32)
    errors = [Base.invokelatest(wave_case, :uniform, n, 1.0, :capillary; formulation = CGFM()).error for n in resolutions]
    rates = log2.(errors[1:2] ./ errors[2:3])
    endpoint = log2(errors[1] / errors[3]) / 2
    println("errors ", errors, " rates ", rates, " endpoint ", endpoint)
    @test round.(errors; sigdigits = 3) ≈ [0.0646, 0.0126, 0.0043] atol = 6.0e-5
    @test all(diff(errors) .< 0)
    @test endpoint >= 1.8
    @test round(endpoint; digits = 2) == 1.96
end
