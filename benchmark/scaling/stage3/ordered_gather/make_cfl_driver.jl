#!/usr/bin/env julia
"""Generate a separate fixed-carrier, per-step-CFL Stage 3 timing driver.

Run the generated driver against the frozen control and candidate source with
identical input, step count, worker placement, and `kernel=upwind_cfl`. The
control output is the fresh oracle for a full-field comparison. This generator
does not change the existing fixed-dt driver or its historical measurements.
"""
module OrderedGatherCFLDriver

using SHA
using TOML

const DRIVER = normpath(joinpath(@__DIR__, "..", "main_driver.jl"))
const EXPECTED_DRIVER_SHA256 =
    "caebf0791323a7a6db49fe83055bb724fb641e5085ddab5fa66b8025be7d0b84"
const WORKLOAD = "fixed_carrier_per_step_cfl_upwind_v1"

hashfile(path) = bytes2hex(SHA.sha256(read(path)))

function replace_once(source, original, replacement)
    first = findfirst(original, source)
    first === nothing && error("driver anchor missing: $original")
    findnext(original, source, last(first) + 1) === nothing ||
        error("driver anchor repeated: $original")
    return replace(source, original => replacement; count=1)
end

function cfl_driver(source)
    result = replace_once(source,
        "kernel == \"upwind\" || error(\"this driver only measures general production upwind\")",
        "kernel == \"upwind_cfl\" || error(\"this driver only measures fixed-carrier per-step CFL upwind\")")
    result = replace_once(result,
        "    dt = 0.35 / n\n" *
        "    # The toy uses this constant-carrier CFL formula outside its timer.\n" *
        "    record[\"dt\"] = dt\n",
        "    dt_factor = 0.4\n" *
        "    record[\"workload\"] = \"$WORKLOAD\"\n" *
        "    record[\"carrier_policy\"] = \"prescribed constant owned-face velocity\"\n" *
        "    record[\"dt_policy\"] = \"0.4 * stable_timestep(velocity, workspace; cfl=0.9) each step\"\n" *
        "    record[\"cfl_scans_per_timed_step\"] = 1\n")
    result = replace_once(result,
        "    dt <= bound*(1+1e-12) || error(\"CFL violation\")",
        "    isfinite(bound) && bound > 0 || error(\"invalid prescribed-carrier CFL bound\")\n" *
        "    record[\"dt\"] = dt_factor * bound\n" *
        "    record[\"dt\"] <= bound || error(\"CFL violation\")")
    result = replace_once(result,
        "    step!() = advect!(field,velocity,dt,workspace)",
        "    step!() = begin\n" *
        "        bound = stable_timestep(velocity,workspace;cfl=0.9)\n" *
        "        dt = dt_factor * bound\n" *
        "        advect!(field,velocity,dt,workspace)\n" *
        "    end")
    return replace_once(result,
        "# Fresh-process Stage 3 main-package timing. The Julia coordinator supplies\n",
        "# Generated fixed-carrier per-step-CFL Stage 3 timing. The coordinator supplies\n")
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: make_cfl_driver.jl --output NEW_DIRECTORY")
        return nothing
    end
    length(args) == 2 && args[1] == "--output" ||
        error("usage: make_cfl_driver.jl --output NEW_DIRECTORY")
    output = abspath(args[2])
    !ispath(output) || error("output already exists: $output")
    hashfile(DRIVER) == EXPECTED_DRIVER_SHA256 ||
        error("Stage 3 driver differs from the frozen source")
    generated = cfl_driver(read(DRIVER, String))
    mkpath(output)
    target = joinpath(output, "cfl_driver.jl")
    write(target, generated)
    receipt = Dict{String,Any}(
        "schema" => 1,
        "workload" => WORKLOAD,
        "carrier_policy" => "prescribed constant owned-face velocity",
        "dt_policy" => "0.4 * stable_timestep(velocity, workspace; cfl=0.9) each step",
        "cfl_scans_per_timed_step" => 1,
        "source_driver" => DRIVER,
        "source_driver_sha256" => EXPECTED_DRIVER_SHA256,
        "generated_driver" => target,
        "generated_driver_sha256" => hashfile(target),
        "generator_sha256" => hashfile(@__FILE__),
    )
    open(joinpath(output, "cfl-driver-receipt.toml"), "w") do io
        TOML.print(io, receipt)
    end
    println("created ", target)
    return receipt
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
