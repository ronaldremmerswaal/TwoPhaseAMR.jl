# Run once per fresh process, after setting up the benchmark environment.
# Example: julia --heap-size-hint=3G -t 1 --project=benchmark benchmark/latency_breakdown.jl stokes3_wave

const script_start_ns = time_ns()
using LinearAlgebra
BLAS.set_num_threads(1)

const allowed_examples = ("taylor_green", "capillary_wave", "stokes3_wave")
if isempty(ARGS) || !(ARGS[1] in allowed_examples)
    error("usage: latency_breakdown.jl {$(join(allowed_examples, "|"))} [--max-level=N] [--dt=N] [--two-velocity]")
end
const example_name = ARGS[1]
const max_level_arg = filter(x -> startswith(x, "--max-level="), ARGS[2:end])
const dt_arg = filter(x -> startswith(x, "--dt="), ARGS[2:end])
const max_level = isempty(max_level_arg) ? 1 : parse(Int, split(only(max_level_arg), "="; limit = 2)[2])
const dt = isempty(dt_arg) ? 0.001 : parse(Float64, split(only(dt_arg), "="; limit = 2)[2])
const two_velocity = "--two-velocity" in ARGS[2:end]
for arg in ARGS[2:end]
    (startswith(arg, "--max-level=") || startswith(arg, "--dt=") || arg == "--two-velocity") || error("unknown argument: $arg")
end
max_level >= 0 || error("max level must be nonnegative")
isfinite(dt) && dt > 0 || error("dt must be positive and finite")

println("Import timings (LinearAlgebra preloaded to set BLAS threads):")
const package_load = @timed Base.eval(Main, :(Base.@time_imports using TwoPhaseAMR))
module Example end
const example_include = @timed Base.include(Example, joinpath(@__DIR__, "..", "examples", only(filter(d -> endswith(d, "_" * example_name), readdir(joinpath(@__DIR__, "..", "examples")))), example_name * ".jl"))
include(joinpath(@__DIR__, "latency", "cases.jl"))

const construction = @timed setup_case(Val(Symbol(example_name)), Val(two_velocity); max_level, dt)
const flow = construction.value
const projection = example_name == "stokes3_wave" ? (@timed initial_projection!(flow)) : (@timed nothing)
const first_step = @timed one_step!(flow, dt)
const second_step = @timed one_step!(flow, dt)
const script_seconds = (time_ns() - script_start_ns) / 1e9

using Printf
println("\nTwoPhaseAMR latency breakdown: ", example_name,
    two_velocity ? " (two-velocity CMDGFM)" : " (default velocity model)")
println("Julia ", VERSION, ", threads=", Threads.nthreads(), ", BLAS threads=", BLAS.get_num_threads(),
    ", max_level=", max_level, ", dt=", dt)
println("CPU: ", Sys.cpu_info()[1].model)
println("Package source: ", pathof(TwoPhaseAMR))
@printf("%-25s %10s %10s %10s %12s\n", "Phase", "wall (s)", "compile", "recompile", "alloc (MiB)")
for (label, measurement) in (
        ("using TwoPhaseAMR", package_load),
        ("include example", example_include),
        ("setup + initial AMR", construction),
        ("initial projection", projection),
        ("first simulate! step", first_step),
        ("second simulate! step", second_step),
    )
    @printf("%-25s %10.3f %10.3f %10.3f %12.2f\n", label, measurement.time,
        measurement.compile_time, measurement.recompile_time, measurement.bytes / 2.0^20)
end
@printf("Script wall (excluding Julia startup and output): %.3f s; final steps=%d, time=%.6g\n",
    script_seconds, flow.steps, flow.time)
