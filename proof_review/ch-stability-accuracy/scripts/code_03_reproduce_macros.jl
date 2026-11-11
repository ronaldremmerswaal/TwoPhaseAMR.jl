# Code check 03: the numbers of Section 13.3 (bounds of four flows, step sequences, empirical margins) reproduce from docs/techref/figures/stability.jl.
#
# Label:  sec:stab:measured, eq:stab:selection (ch:stability-accuracy)
# Source: docs/techref/figures/stability.jl (stability_outputs), docs/techref/figures/generated/checks.tex:37-75 (chkStab* macros),
#         docs/techref/chapters/13_stability_accuracy.tex:268-378 (text and table tab:stab:cases)
# Checks:
#   M1  stability_outputs() (the case bounds, the selection check with its asserts, the stability margins with their asserts) is rerun and every
#       chkStab* macro of the committed checks.tex equals the freshly computed value (2.5x / 2.75x capillary, 1.35x / 1.5x Kelvin--Helmholtz, ...);
#   M2  derived statements of the text: 0.45 x 1.35 = 0.61, 0.45 x 1.5 = 0.68 (WY cfl of the stable / failing multiples), margin 0.5/0.45 = 1.11,
#       the capillary-wave margin exceeds the Denner--van Wachem static factor sqrt2.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-stability-accuracy/scripts/code_03_reproduce_macros.jl

isdefined(Main, :STAB_CFL) || include(joinpath(@__DIR__, "_prelude.jl"))

function committed_macros(path)
    macros = Dict{String, String}()
    for line in eachline(path)
        m = match(r"^\\newcommand\\(chkStab\w+)\{(.*)\}$", line)
        isnothing(m) || (macros[m.captures[1]] = m.captures[2])
    end
    return macros
end

@testset "ch13 code: reproduce the measured numbers" begin
    committed = committed_macros(joinpath(OUTPUT, "checks.tex"))
    fresh = stability_outputs(nothing)
    @testset "M1 macros" begin
        @test length(fresh) == length(committed) == 39
        for (name, value) in fresh
            @test get(committed, String(name), nothing) == string(value)
            string(get(committed, String(name), "")) == string(value) ||
                println("MISMATCH ", name, ": committed ", get(committed, String(name), nothing), " fresh ", value)
        end
    end
    @testset "M2 derived statements" begin
        v = Dict(String(n) => string(x) for (n, x) in fresh)
        @test v["chkStabKhMarginCfl"] == @sprintf("%.2f", 0.45 * parse(Float64, v["chkStabKhMargin"]))
        @test v["chkStabKhUnstableCfl"] == @sprintf("%.2f", 0.45 * parse(Float64, v["chkStabKhUnstable"]))
        @test round(0.5 / 0.45; digits = 2) == 1.11
        @test round(1.0 / 0.45; digits = 2) == 2.22
        @test parse(Float64, v["chkStabWaveMargin"]) > sqrt(2)
        println(
            "M2 capillary margin ", v["chkStabWaveMargin"], " (unstable at ", v["chkStabWaveUnstable"], "), Denner static factor ", sqrt(2),
            "; KH margin ", v["chkStabKhMargin"], " (cfl ", v["chkStabKhMarginCfl"], "), unstable ", v["chkStabKhUnstable"], " (cfl ", v["chkStabKhUnstableCfl"], ")"
        )
    end
end
