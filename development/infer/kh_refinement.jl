module InferKHRefinement

using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))

"""Complex wave-history difference, with the fine history sampled at coarse times."""
function history_difference(coarse, fine)
    coarse.final_time ≈ fine.final_time || throw(
        ArgumentError("KH runs must have the same final time"),
    )
    fine_times = fine.times
    fine_values = fine.amplitudes
    index = 1
    squared = reference = 0.0
    previous_time = coarse.times[1]
    previous_difference = 0.0 + 0.0im
    previous_reference = coarse.analytical[1]
    for coarse_index in eachindex(coarse.times)
        time = coarse.times[coarse_index]
        while index < length(fine_times) - 1 && fine_times[index + 1] < time
            index += 1
        end
        left, right = fine_times[index], fine_times[index + 1]
        weight = clamp((time - left) / (right - left), 0.0, 1.0)
        fine_value = (1 - weight) * fine_values[index] +
            weight * fine_values[index + 1]
        difference = coarse.amplitudes[coarse_index] - fine_value
        analytical = coarse.analytical[coarse_index]
        if coarse_index > 1
            width = time - previous_time
            squared += width * (
                abs2(previous_difference) + abs2(difference)
            ) / 2
            reference += width * (
                abs2(previous_reference) + abs2(analytical)
            ) / 2
        end
        previous_time = time
        previous_difference = difference
        previous_reference = analytical
    end
    return sqrt(squared / reference)
end

function projection(method::Symbol)
    method === :cmdgfm && return CMDGFMProjection()
    method === :mimetic && return MimeticProjection()
    method === :joint_face && return MimeticProjection(variant = :joint_face)
    throw(ArgumentError("unknown projection method: $method"))
end

"""Run matched KH mesh and time refinements; print every result immediately."""
function study(; densities = (1.0, 1000.0), webers = (0.5, 5.0),
        meshes = (8, 16, 32), factors = (1.0, 0.5, 0.25),
        methods = (:cmdgfm, :mimetic, :joint_face),
        final_time = nothing,
    )
    results = Dict{Tuple{Float64,Float64,Int,Float64,Symbol},Any}()
    for density in densities, weber in webers, method in methods
        for mesh in meshes
            key = (float(density), float(weber), mesh, 1.0, method)
            try
                run = interfacial_instability(
                    ; cells_per_wavelength = mesh,
                    density_ratio = density, weber,
                    bond = 0.0, liquid_above = false,
                    pressure_projection = projection(method),
                    final_time, tolerance = 1.0e-7,
                    max_cycles = 3000,
                )
                results[key] = run
                error = normalized_amplitude_error(
                    run.times, run.amplitudes, run.analytical,
                )
                @printf(
                    "KH rho=%g We=%g method=%s n=%d dt=1 error=%.6e steps=%d\n",
                    density, weber, string(method), mesh, error,
                    run.flow.steps,
                )
                flush(stdout)
            catch exception
                @printf(
                    "KH rho=%g We=%g method=%s n=%d dt=1 FAILED: %s\n",
                    density, weber, string(method), mesh,
                    sprint(showerror, exception),
                )
                flush(stdout)
            end
        end
        time_mesh = length(meshes) >= 2 ? meshes[2] : first(meshes)
        for factor in Iterators.drop(factors, 1)
            key = (float(density), float(weber), time_mesh,
                float(factor), method)
            try
                run = interfacial_instability(
                    ; cells_per_wavelength = time_mesh,
                    density_ratio = density, weber,
                    bond = 0.0, liquid_above = false,
                    pressure_projection = projection(method),
                    timestep_factor = factor,
                    final_time, tolerance = 1.0e-7,
                    max_cycles = 3000,
                )
                results[key] = run
                error = normalized_amplitude_error(
                    run.times, run.amplitudes, run.analytical,
                )
                previous_factor = factors[findfirst(==(factor), factors) - 1]
                previous_key = (float(density), float(weber), time_mesh,
                    float(previous_factor), method)
                self_error = haskey(results, previous_key) ?
                    history_difference(results[previous_key], run) : NaN
                @printf(
                    "KH rho=%g We=%g method=%s n=%d dt=%g error=%.6e self=%.6e steps=%d\n",
                    density, weber, string(method), time_mesh,
                    factor, error, self_error, run.flow.steps,
                )
                flush(stdout)
            catch exception
                @printf(
                    "KH rho=%g We=%g method=%s n=%d dt=%g FAILED: %s\n",
                    density, weber, string(method), time_mesh,
                    factor, sprint(showerror, exception),
                )
                flush(stdout)
            end
        end
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module InferKHRefinement
