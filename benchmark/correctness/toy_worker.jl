using Serialization

const TOY_ROOT = normpath(joinpath(@__DIR__, "..", "..", "research", "UniformLWAdvection"))
push!(LOAD_PATH, TOY_ROOT)
using UniformLWAdvection
using Polyester

const FLUX_POLICIES = (:recompute, :stored)
const BACKENDS = (:serial, :threads, :spawn, :polyester)

function exact_initial!(state)
    nx, ny = size(state.q)
    for j in 1:ny, i in 1:nx
        state.q[i, j] = 1 + 0.2sinc(1 / nx) * sinpi(2(i - 0.5) / nx)
    end
    state.t = 0.0
    return state
end

function fixture(name::String, flux::Symbol, backend::Symbol)
    nx, ny, steps = name == "constant_square" ? (32, 32, 3) :
                    name == "constant_odd" ? (35, 27, 4) :
                    name == "variable_shear" ? (32, 32, 3) :
                    error("unknown fixture $name")
    state = name == "variable_shear" ? State(nx, ny; u = 0.0, v = 0.0) :
            State(nx, ny; u = 0.7, v = -0.3)
    if name == "variable_shear"
        exact_initial!(state)
    else
        initialize!(state)
    end
    carrier = name == "variable_shear" ? :variable : :constant
    workspace = Workspace(
        state; flux, backend, carrier, tile = (7, 5), threshold = 0,
        batches_per_worker = 3,
    )
    initial_mass = mass(state)
    dt = name == "variable_shear" ? 0.003 :
         min(0.004, stable_timestep(state; cfl = 0.35))
    for _ in 1:steps
        if name == "variable_shear"
            set_velocity!(
                state,
                (x, y, t) -> 0.6cospi(2t) * sinpi(2y),
                (x, y, t) -> 0.0,
                state.t + dt / 2,
            )
        end
        step!(state, workspace, dt)
    end
    return Dict{String, Any}(
        "q" => vec(copy(state.q)),
        "shape" => [nx, ny],
        "steps" => steps,
        "dt" => dt,
        "time" => state.t,
        "mass_before" => initial_mass,
        "mass_after" => mass(state),
    )
end

function main()
    length(ARGS) == 1 || error("usage: toy_worker.jl OUTPUT_FILE")
    results = Dict{String, Any}()
    for name in ("constant_square", "constant_odd", "variable_shear"),
        flux in FLUX_POLICIES, backend in BACKENDS
        results["$name/$flux/$backend"] = fixture(name, flux, backend)
    end
    output = Dict{String, Any}(
        "schema" => 1,
        "toy_source" => read(joinpath(TOY_ROOT, "src", "UniformLWAdvection.jl")),
        "toy_extension" => read(joinpath(TOY_ROOT, "ext", "UniformLWAdvectionPolyesterExt.jl")),
        "julia_version" => string(VERSION),
        "default_threads" => Threads.nthreads(:default),
        "results" => results,
    )
    open(ARGS[1], "w") do io
        serialize(io, output)
    end
end

main()
