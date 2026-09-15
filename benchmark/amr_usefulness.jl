using BenchmarkTools
using TwoPhaseAMR

# Keep the benchmark and its CI assertions on one manufactured problem.
include(joinpath(@__DIR__, "..", "test", "amr_manufactured_problem.jl"))

function summarize_amr(level)
    result = _localized_vortex_case(level)
    return (error=result[1], divergence=result[2], cells=result[3],
            blocks=result[4], finest_level=result[5],
            pressure_cycles=result[6].pressure.cycles)
end

function summarize_uniform(resolution)
    result = _uniform_localized_vortex_case(resolution)
    return (error=result[1], divergence=NaN, cells=result[2], blocks=1,
            finest_level=0,
            pressure_cycles=result[3].pressure.cycles)
end

configurations = [
    (label="AMR Lmax=1", run=() -> summarize_amr(1)),
    (label="AMR Lmax=2", run=() -> summarize_amr(2)),
    (label="AMR Lmax=3", run=() -> summarize_amr(3)),
    (label="uniform 48x48", run=() -> summarize_uniform(48)),
    (label="uniform 64x64", run=() -> summarize_uniform(64)),
]

println("Localized manufactured-vortex AMR study")
println("case | cells | blocks | velocity L2 error | max divergence | " *
        "median end-to-end time | memory | allocations")
println("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
results = NamedTuple[]
for configuration in configurations
    sample = configuration.run() # warm compilation and collect the result
    trial = @benchmark $(configuration.run)() samples=3 seconds=60 evals=1
    estimate = BenchmarkTools.median(trial)
    push!(results, sample)
    println(configuration.label, " | ", sample.cells, " | ", sample.blocks,
            " | ", sample.error, " | ", sample.divergence, " | ",
            BenchmarkTools.prettytime(estimate.time), " | ",
            BenchmarkTools.prettymemory(estimate.memory), " | ",
            estimate.allocs)
end

amr_errors = getproperty.(results[1:3], :error)
println("Observed max-level rates: ",
        log2.(amr_errors[1:2] ./ amr_errors[2:3]))
println("Lmax=3 AMR error / equal-cell uniform error: ",
        results[3].error / results[4].error)
println("Lmax=3 AMR error / equal-resolution uniform error: ",
        results[3].error / results[5].error)
println("Lmax=3 cells / uniform 64x64 cells: ",
        results[3].cells / 64^2)
