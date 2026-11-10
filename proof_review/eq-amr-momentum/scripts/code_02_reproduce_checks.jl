# Code check 02: reproduce the numbers that Chapter 12 quotes (generated/checks.tex, chkAmr*) from the figure script itself.
#
# Labels: tab:amr:conservation, sec:amr:measurements
# Source: docs/techref/figures/amr.jl (amr_transfer_check, amr_indicator_check, amr_flow_check), docs/techref/figures/generated/checks.tex:273-336
# Checks: the committed macros equal a fresh run of the figure script (two significant digits; deterministic, no RNG);
#         the normalisation of the flow quantities is read from the script (momentum / sum rho dual |u|, energy / E).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-momentum/scripts/code_02_reproduce_checks.jl

using Test
using TwoPhaseAMR
using Printf: @sprintf
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "docs", "techref", "figures", "amr.jl"))

macros = Dict{String, String}()
for line in eachline(joinpath(ROOT, "docs", "techref", "figures", "generated", "checks.tex"))
    m = match(r"^\\newcommand\\(chkAmr\w+)\{(.*)\}$", line)
    m === nothing || (macros[m.captures[1]] = m.captures[2])
end
@assert length(macros) > 60

# "3.8\times10^{-16}" -> 3.8e-16 ; "0" -> 0 ; "3.83" -> 3.83 ; integers
function parse_macro(s)
    m = match(r"^([0-9.]+)\\times10\^\{(-?[0-9]+)\}$", s)
    m === nothing && return parse(Float64, s)
    return parse(Float64, m.captures[1]) * 10.0^parse(Int, m.captures[2])
end
agree(value, name; digits = 2) = begin
    ref = parse_macro(macros[name])
    iszero(ref) ? abs(value) <= 1.0e-30 : abs(value - ref) <= 0.51 * 10.0^(floor(log10(abs(ref))) - (digits - 1)) + 1.0e-12 * abs(ref)
end

@testset "chkAmr macros = fresh run of amr.jl" begin
    t = amr_transfer_check()
    ind = amr_indicator_check()
    println("transfer: ", t)
    checks = [
        ("chkAmrRegrids", t.regrids), ("chkAmrRefinedBlocks", t.refined), ("chkAmrCoarsenedFamilies", t.coarsened),
        ("chkAmrChildren", t.children), ("chkAmrSnapped", t.snapped), ("chkAmrLabelsCompared", ind.compared),
        ("chkAmrInterpRatio", ind.ratio), ("chkAmrWettedBlocks", ind.artifacts.wetted), ("chkAmrCurvatureBlocks", ind.artifacts.curvature),
        ("chkAmrArtifactBlocks", ind.artifacts.blocks), ("chkAmrCycleChanges", ind.cycle.changes), ("chkAmrCyclePeriod", ind.cycle.period), ("chkAmrCycleSteps", ind.cycle.steps),
        ("chkAmrWallMomentumMin", t.wall_min), ("chkAmrWallMomentum", t.wall), ("chkAmrEnergy", t.energy),
    ]
    for (name, value) in checks
        ok = agree(value, name)
        println(rpad(name, 28), "code ", value, "   macro ", macros[name], ok ? "" : "   <-- DIFFERENT")
        @test ok
    end
    # the quantities that are round-off level: only the order of magnitude is meaningful
    for (name, value) in (("chkAmrVolume", t.volume), ("chkAmrSolid", t.solid), ("chkAmrScalar", t.scalar), ("chkAmrPressure", t.pressure), ("chkAmrIdentity", t.identity), ("chkAmrPeriodicMomentum", t.periodic))
        println(rpad(name, 28), "code ", value, "   macro ", macros[name])
        @test value <= 1.0e-13
    end
end

@testset "flow measurements (60 + 60 + 40 steps)" begin
    @time flows = amr_flow_check()
    for (case, prefix) in ((flows.one, "One"), (flows.two, "Two"), (flows.cut, "Cut"))
        println(
            prefix, ": regrids ", case.regrids, " refined ", case.refined, " coarsened ", case.coarsened, " | volume ", case.volume, " momentum ", case.momentum, " energy ", case.energy,
            " | defect before ", case.before, " transferred ", case.transferred, " projected ", case.projected, " | projection energy ", case.projection_energy
        )
        @test agree(case.momentum, "chkAmrFlow$(prefix)Momentum")
        @test agree(case.energy, "chkAmrFlow$(prefix)Energy")
        @test agree(case.regrids, "chkAmrFlow$(prefix)Regrids")
        @test agree(case.transferred, "chkAmrFlow$(prefix)Transferred")
    end
    @test agree(flows.two.projection_energy, "chkAmrFlowTwoProjectionEnergy")
    @test flows.validation <= 1.0e-12
end
