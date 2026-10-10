"""Analytic cell-average translation check for the dormant Fortran toy."""
module FortranConvergence

include(joinpath(@__DIR__, "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport: run_capture, writejson
using JSON
using Statistics

sinc(z) = sinpi(z) / (pi * z)

function average(i, j, n, t; u=0.7, v=-0.3)
    x = (i + 0.5) / n - u * t
    y = (j + 0.5) / n - v * t
    return 1.0 + 0.2sinc(1 / n) * sin(2pi * x) +
           0.1sinc(2 / n) * cos(4pi * y) +
           0.15sinc(1 / n)^2 * sin(2pi * (x + y))
end

function one(binary, n, workers)
    final_time = 0.25
    steps = ceil(Int, final_time / (0.35 / n))
    dt = final_time / steps
    initial = [average(i - 1, j - 1, n, 0.0) for j in 1:n for i in 1:n]
    exact = [average(i - 1, j - 1, n, final_time) for j in 1:n for i in 1:n]
    return mktempdir() do directory
        source, target = joinpath(directory, "initial.bin"), joinpath(directory, "final.bin")
        open(source, "w") do io
            write(io, initial)
        end
        command = String[binary, "--nx", string(n), "--ny", string(n),
                         "--steps", string(steps), "--threads", string(workers),
                         "--dt", string(dt), "--tile-x", "512", "--tile-y", "16",
                         "--input", source, "--output", target]
        result = run_capture(command)
        result.exitcode == 0 || error("Fortran failed: $(result.stderr)")
        record = JSON.parse(strip(result.stdout))
        actual = open(target) do io
            read!(io, Vector{Float64}(undef, n * n))
        end
        errors = actual .- exact
        return Dict{String,Any}(
            "n" => n, "workers" => workers, "steps" => steps, "dt" => dt,
            "final_time" => final_time,
            "rms_error" => sqrt(sum(abs2, errors) / (n * n)),
            "max_error" => maximum(abs, errors),
            "mass_error" => abs(sum(actual) / (n * n) - 1.0),
            "active_worker_counts" => record["active_worker_counts"],
            "active_worker_ids_by_pass" => record["active_worker_ids_by_pass"],
            "field" => actual,
        )
    end
end

function main(args=ARGS)
    length(args) == 2 || error("usage: julia validate_convergence.jl BINARY RECEIPT")
    binary, receipt_path = abspath(args[1]), abspath(args[2])
    entries = Dict{String,Any}[]
    runs_by_workers = Dict{Int,Vector{Dict{String,Any}}}()
    for workers in (1, 4)
        runs = [one(binary, n, workers) for n in (32, 64, 128)]
        for run in runs
            n = run["n"]
            expected_workers = min(workers, cld(n, 512) * cld(n, 16))
            @assert run["active_worker_counts"] == fill(expected_workers, 3)
            @assert run["mass_error"] <= 512eps(1.0)
        end
        for (coarse, fine) in zip(runs, runs[2:end])
            order = log2(coarse["rms_error"] / fine["rms_error"])
            fine["observed_order_from_previous"] = order
            @assert 1.8 <= order <= 2.2
        end
        runs_by_workers[workers] = runs
    end
    for (serial, parallel) in zip(runs_by_workers[1], runs_by_workers[4])
        delta = maximum(abs.(serial["field"] .- parallel["field"]))
        @assert delta == 0.0
        serial["parallel_field_delta"] = delta
        parallel["serial_field_delta"] = delta
    end
    for workers in (1, 4), run in runs_by_workers[workers]
        delete!(run, "field")
        push!(entries, run)
    end
    writejson(receipt_path, Dict(
        "kind" => "fortran_analytic_cell_average_translation_validation",
        "binary" => binary, "physical_time" => 0.25,
        "mesh_sizes" => [32, 64, 128], "worker_counts" => [1, 4],
        "timestep_rule" => "dt=T/ceil(T/(0.35/N))", "results" => entries,
    ))
    for run in entries
        println("N=$(run["n"]) p=$(run["workers"]) steps=$(run["steps"]) " *
                "RMS=$(run["rms_error"]) max=$(run["max_error"]) " *
                "mass=$(run["mass_error"]) " *
                "order=$(get(run, "observed_order_from_previous", NaN))")
    end
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && FortranConvergence.main()
