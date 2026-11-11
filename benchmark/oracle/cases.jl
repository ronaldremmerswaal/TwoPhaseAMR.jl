using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
using SHA
using TOML

# Separate modules prevent the examples' run_example methods from overwriting one another.
module TaylorGreen
    include(joinpath(@__DIR__, "../../examples/01_taylor_green/taylor_green.jl"))
end
module Cavity
    include(joinpath(@__DIR__, "../../examples/02_lid_driven_cavity/lid_driven_cavity.jl"))
end
module Capillary
    include(joinpath(@__DIR__, "../../examples/06_capillary_wave/capillary_wave.jl"))
end
module Sessile
    include(joinpath(@__DIR__, "../../examples/08_sessile_droplet/sessile_droplet.jl"))
end
module Sloped
    include(joinpath(@__DIR__, "../../examples/09_sloped_sessile_droplet/sloped_sessile_droplet.jl"))
end
module Vortex
    include(joinpath(@__DIR__, "../../examples/05_vortex_reverse/vortex_reverse.jl"))
end
module Stokes
    include(joinpath(@__DIR__, "../../examples/10_stokes3_wave/stokes3_wave.jl"))
end
module Cylinder
    include(joinpath(@__DIR__, "../../examples/04_cylinder_flow/cylinder_flow.jl"))
end

const CASES = [
    "taylor_green" => (() -> TaylorGreen.run_example(max_level = 2, final_time = 0.02, dt = 0.002)),
    "cavity" => (() -> Cavity.run_example(max_level = 1, final_time = 0.02, dt = 0.002)),
    "capillary_one" => (() -> Capillary.run_example(max_level = 1, final_time = 0.02, dt = 0.001, velocity_model = OneVelocityModel())),
    "capillary_cmdgfm" => (() -> Capillary.run_example(max_level = 1, final_time = 0.02, dt = 0.001, velocity_model = TwoVelocityModel(pressure_projection = CMDGFMProjection()))),
    "sessile" => (() -> Sessile.run_example(max_level = 1, final_time = 0.01, dt = 0.001)),
    "sloped" => (() -> Sloped.run_example(max_level = 1, final_time = 0.01, dt = 0.001)),
    "vortex_reverse" => (() -> Vortex.run_example(max_level = 2, final_time = 0.05)),
    "stokes3" => (() -> Stokes.run_example(max_level = 1, final_time = 0.01, dt = 0.001)),
    "cylinder" => (() -> Cylinder.run_example(max_level = 1, final_time = 0.01, dt = 0.001)),
]

function stats(values)
    data = collect(Float64, values)
    any(isinf, data) && error("infinite value in oracle field")
    finite = filter(isfinite, data) # inactive two-phase face values are NaN
    return Dict(
        "sha256" => bytes2hex(sha256(reinterpret(UInt8, data))),
        "l2" => norm(finite), "max" => isempty(finite) ? 0.0 : maximum(abs, finite),
        "count" => length(data), "finite_count" => length(finite)
    )
end

# FaceVelocity stores both directions in one owned-face vector; the mesh face
# schedule is the internal detail needed to separate directional components.
function face_data(velocity, axis)
    return [velocity.interior[i] for (i, face) in pairs(velocity.mesh.faces) if face.axis == axis]
end

function cell_data(field)
    return [value for block in eachindex(field.blocks) for value in interior(field, block)]
end

toml_value(value::Tuple) = [toml_value(item) for item in value]
toml_value(value) = value

function capture(run; perturb = false)
    flow = hasproperty(run, :flow) ? run.flow : nothing
    velocity = isnothing(flow) ? run.velocity : flow.velocity
    interface = isnothing(flow) ? run.interface : flow.interface
    fields = Dict{String, Any}()
    phases = velocity isa TwoPhaseAMR.TwoPhaseVelocity ? collect(velocity) : [velocity]
    for (phase, v) in enumerate(phases), axis in 1:2
        name = length(phases) == 1 ? "velocity_$axis" : "velocity_phase$(phase)_$axis"
        data = face_data(v, axis)
        if perturb && phase == 1 && axis == 1 && !isempty(data)
            data[1] = nextfloat(data[1])
        end
        fields[name] = stats(data)
    end
    isnothing(flow) || (fields["pressure"] = stats(cell_data(flow.pressure)))
    isnothing(interface) || (fields["volume_fraction"] = stats(cell_data(interface.fraction)))
    diagnostics = Dict{String, Any}()
    if !isnothing(flow)
        for name in diagnostic_names(flow)
            history = diagnostic_history(flow, name)
            isempty(history) || (diagnostics[string(name)] = toml_value(last(history.values)))
        end
    else
        diagnostics["volume_error"] = run.result.volume_error
    end
    return Dict(
        "steps" => isnothing(flow) ? run.result.steps : flow.steps,
        "time" => isnothing(flow) ? run.result.time : flow.time,
        "fields" => fields, "diagnostics" => diagnostics
    )
end

function record_cases()
    cases = Dict{String, Any}()
    for (name, run) in CASES
        println("Running ", name); flush(stdout)
        cases[name] = capture(run())
    end
    return Dict("julia_version" => string(VERSION), "threads" => Threads.nthreads(), "cases" => cases)
end
