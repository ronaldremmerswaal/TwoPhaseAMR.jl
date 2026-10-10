# Bounded, sequential Stage 2 toy policy screen at 1024² (exploratory).
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
include(joinpath(@__DIR__, "Stage2Common.jl"))

function main(args=ARGS)
    opts = stage2_options(args, ("project", "depot", "julia", "output");
        defaults=Dict{String, Any}("project" => STAGE2_ROOT))
    isnothing(opts) && return
    julia = stage2_julia_path(get(opts, "julia", nothing))
    output = get(opts, "output", joinpath(STAGE2_ROOT, "benchmark", "scaling",
                                          "results", stage2_stamp() * "-toy-screen"))
    ispath(output) && error("output already exists: $output")
    mkpath(output)
    first_policy = ("threads", "recompute", 128)
    policies = [(backend, flux, tile_x)
                for backend in ("threads", "spawn", "polyester")
                for flux in ("recompute", "stored") for tile_x in (128, 1024)]
    filter!(policy->policy != first_policy, policies)
    shuffle!(MersenneTwister(29092026), policies)
    policies = [first_policy; policies; [("serial", flux, 128)
                                        for flux in ("recompute", "stored")]]
    writejson(joinpath(output, "matrix.json"), Dict(
        "purpose" => "bounded exploratory screen; no acceptance claim",
        "size" => [1024, 1024], "threads_parallel" => [1, 4, 8],
        "threads_serial" => [1], "repeats" => 2,
        "fastest_worker_seconds_target" => 0.25, "placement" => "spread",
        "pin_workers" => true, "tile_y" => 32,
        "policies_in_run_order" => [collect(policy) for policy in policies],
        "project" => opts["project"], "depot" => get(opts, "depot", nothing),
        "julia" => julia,
    ))
    workload = nothing
    for (number, (backend, flux, tile_x)) in enumerate(policies)
        name = "$backend-$flux-$(tile_x)x32"
        destination = joinpath(output, name)
        command = [julia, "--startup-file=no", "--project=$(joinpath(@__DIR__))",
                   joinpath(@__DIR__, "run.jl"), "--project", opts["project"],
                   "--julia", julia]
        haskey(opts, "depot") && append!(command, ["--depot", opts["depot"]])
        append!(command, ["--cases", "toy", "--sizes", "1024", "--threads"])
        append!(command, backend == "serial" ? ["1"] : ["1", "4", "8"])
        append!(command, ["--repeats", "2", "--seconds", "0.25",
                          "--placement", "spread", "--pin-workers",
                          "--backend", backend, "--flux", flux,
                          "--tile-x", string(tile_x), "--tile", "32",
                          "--output", destination])
        !isnothing(workload) && append!(command, ["--workload", workload])
        println("SCREEN $number/$(length(policies)) $name")
        result = run_capture(command)
        result.exitcode == 0 || error("policy failed: $name\n$(result.stdout)\n$(result.stderr)")
        isnothing(workload) && (workload = joinpath(destination, "workload.json"))
    end
    println("Completed $(length(policies)) sequential exploratory policies: $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
