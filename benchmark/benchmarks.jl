using BenchmarkTools

include("common.jl")

const SUITE = BenchmarkGroup()
include("setup_and_stepping.jl")
include("adaptivity_and_interface.jl")
include("interface_transport.jl")
include("linear_solvers.jl")

function run_benchmarks()
    print_benchmark_header()
    results = run(
        SUITE; samples = BENCHMARK_SAMPLES,
        seconds = BENCHMARK_SECONDS, verbose = true,
    )
    estimates = median(results)
    show(stdout, MIME("text/plain"), estimates)
    println()
    output = get(ENV, "TWOPHASE_BENCHMARK_OUTPUT", "")
    if !isempty(output)
        BenchmarkTools.save(output, results)
        println("Saved raw benchmark results to ", output)
    end
    return results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_benchmarks()
