#!/usr/bin/env julia
"""Small fresh-process correctness smoke for the checked CFL-inclusive driver."""
module OrderedGatherCFLSmoke

include(joinpath(@__DIR__, "..", "narrow_ids", "study.jl"))
include(joinpath(@__DIR__, "make_probe.jl"))
include(joinpath(@__DIR__, "make_checked_cfl_driver.jl"))
using .BlockInteriorStudy
using .OrderedGatherProbe
using .CheckedCFLOrderedGatherDriver
using TOML

const Study = BlockInteriorStudy
const Probe = OrderedGatherProbe
const Support = Study.ScalingSupport
const N = 64
const STEPS = 2

function main(args=ARGS)
    length(args) == 2 && args[1] == "--output" ||
        error("usage: cfl_smoke.jl --output NEW_DIRECTORY")
    output = abspath(args[2])
    !ispath(output) || error("output must be a new directory")
    source = joinpath(Study.ROOT, "benchmark", "scaling", "results",
                      "20261010-narrow-ids-final", "current-source")
    Support.sha256file(joinpath(source, Probe.RELATIVE)) ==
        Probe.EXPECTED_CLEAN_SHA256 || error("frozen control source differs")
    mkpath(output)
    control = joinpath(output, "control")
    candidate = joinpath(output, "pair_select")
    mkpath(control)
    cp(joinpath(source, "src"), joinpath(control, "src"))
    probe = Probe.make_probe(source, candidate)
    driver = joinpath(output, "checked_cfl_driver.jl")
    driver_hash = CheckedCFLOrderedGatherDriver.make_checked_cfl_driver(driver)
    initial = joinpath(output, "initial.bin")
    q = [1 + 0.2sinpi(2(i - 0.5)/N)*cospi(2(j - 0.5)/N)
         for i in 1:N, j in 1:N]
    open(initial, "w") do io
        write(io, q)
    end
    fields = Dict{String,String}()
    reports = Dict{String,Any}()
    for (variant, root) in (("control",control), ("pair_select",candidate))
        field = joinpath(output, "$variant.bin")
        report = joinpath(output, "$variant.toml")
        Study.launch(output, driver, 1, [0],
            ["source=$root", "mode=measure_cfl", "kernel=upwind_cfl",
             "n=$N", "block=$N", "steps=$STEPS", "cpus=0",
             "input=$initial", "output=$field", "report=$report"])
        row = TOML.parsefile(report)
        row["source"] == root && row["kernel"] == "upwind_cfl" &&
            row["mode"] == "measure_cfl" && row["steps"] == STEPS &&
            row["cfl_scans_per_timed_step"] == 1 &&
            row["workload"] ==
                CheckedCFLOrderedGatherDriver.OrderedGatherCFLDriver.WORKLOAD &&
            row["dt"] == 0.4row["cfl_bound"] &&
            row["field_sha256"] == Support.sha256file(field) ||
            error("$variant CFL smoke report differs")
        fields[variant] = field
        reports[variant] = row
    end
    comparison = Study.full_field(fields["pair_select"], fields["control"], N)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 ||
        error("CFL smoke fields differ")
    Support.writejson(joinpath(output, "smoke.json"), Dict(
        "status" => "passed", "scope" => "small-mesh driver and field correctness only",
        "n" => N, "steps" => STEPS, "threads" => 1,
        "driver_sha256" => driver_hash,
        "generator_sha256" => Support.sha256file(joinpath(@__DIR__, "make_checked_cfl_driver.jl")),
        "source_sha256" => Probe.EXPECTED_CLEAN_SHA256,
        "candidate_sha256" => probe["candidate_source_sha256"],
        "initial_sha256" => Support.sha256file(initial),
        "field_sha256" => Support.sha256file(fields["control"]),
        "comparison" => comparison,
        "reports" => Dict(name => Dict("dt" => row["dt"],
            "cfl_bound" => row["cfl_bound"], "field_sha256" => row["field_sha256"])
            for (name,row) in reports)))
    println("CFL correctness smoke passed; see ", joinpath(output, "smoke.json"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
