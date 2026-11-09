@assert startswith(Base.active_project(), normpath(joinpath(@__DIR__, "..", "..")))

using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
using TwoPhaseAMRStudio

root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
# Example-only dependencies (such as the capillary reference solution) live
# in the examples environment, not in Studio's runtime dependency list.
push!(LOAD_PATH, joinpath(root, "examples"))
example = only(ARGS)
if example == "capillary-wave"
    include(joinpath(root, "examples", "06_capillary_wave", "capillary_wave.jl"))
    run = run_example(; max_level = 1, final_time = 0.002, velocity_model = TwoVelocityModel())
elseif example == "decaying-turbulence"
    include(joinpath(root, "examples", "03_decaying_turbulence", "decaying_turbulence.jl"))
    run = run_example(; max_level = 2, final_time = 0.01)
else
    error("Unknown example: $example")
end
app = studio(run.recorder; port = 0, open = :none, name = example)
try
    println("STUDIO_URL ", url(app))
    flush(stdout)
    readline(stdin)
finally
    close(app)
end
