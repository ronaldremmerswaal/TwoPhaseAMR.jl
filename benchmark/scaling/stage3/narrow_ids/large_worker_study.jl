#!/usr/bin/env julia
module NarrowIDLargeWorkerStudy

include(joinpath(@__DIR__, "study.jl"))
using .BlockInteriorStudy
using TOML
const Support = BlockInteriorStudy.ScalingSupport

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: large_worker_study.jl COHORT_DIRECTORY OUTPUT_DIRECTORY")
        return
    end
    length(args) == 2 || error("usage: large_worker_study.jl COHORT_DIRECTORY OUTPUT_DIRECTORY")
    cohort,output = abspath.(args)
    config = Support.readjson(joinpath(cohort,"configuration.json"))
    (2048,2048) in [(n,b) for (n,b) in config["cases"]] ||
        error("accepted cohort lacks large one-block case")
    cases = [(2048,2048)]
    source = Dict("baseline" => BlockInteriorStudy.BASELINE,
                  "current" => joinpath(cohort,"current-source"))
    probes = Dict(variant => joinpath(output,"probes",variant) for variant in keys(source))
    for variant in ("baseline","current")
        clean = source[variant]
        probe = probes[variant]
        receipt = Support.readjson(joinpath(probe,"probe.json"))
        relative = joinpath("src","operators","advection","cell_advection.jl")
        receipt["clean_source_sha256"] == Support.sha256file(joinpath(clean,relative)) &&
            receipt["probe_source_sha256"] == Support.sha256file(joinpath(probe,relative)) ||
            error("probe source mismatch: $variant")
    end
    mkpath(output)
    placements = Support.readjson(joinpath(cohort,"placement.json"))
    for (n,b) in cases
        BlockInteriorStudy.fixture(output,n,b,1,placements["1"])
    end
    rows = Any[]
    for variant in ("baseline","current")
        probe = probes[variant]
        expected_passes = Support.readjson(joinpath(probe,"probe.json"))["passes"]
        for p in (4,8)
            cpus = placements[string(p)]
            batch = Dict{String,Any}[]
            for (n,b) in cases
                stem = "$variant-n$n-b$b-p$p"
                push!(batch,Dict("mode" => "measure", "kernel" => "upwind",
                                 "n" => n, "block" => b, "steps" => 1,
                                 "input" => joinpath(output,"initial-n$n.bin"),
                                 "output" => joinpath(output,"$stem.bin"),
                                 "report" => joinpath(output,"$stem.toml")))
            end
            batch_path = joinpath(output,"probe-$variant-p$p.toml")
            BlockInteriorStudy.write_batch(batch_path,batch)
            BlockInteriorStudy.launch(output,joinpath(probe,"probe_driver.jl"),p,cpus,
                ["source=$probe", "batch=$batch_path", "cpus=$(join(cpus,','))"])
            for (n,b) in cases
                stem = "$variant-n$n-b$b-p$p"
                report = TOML.parsefile(joinpath(output,"$stem.toml"))
                affinity = report["affinity"]
                [entry["cpu"] for entry in affinity] == cpus &&
                    all(entry["observed_cpu"] == entry["cpu"] for entry in affinity) ||
                    error("worker affinity differs: $stem")
                field = joinpath(output,"$stem.bin")
                oracle = joinpath(output,"oracle-n$n-b$b-s1.bin")
                comparison = BlockInteriorStudy.full_field(field,oracle,n)
                comparison["tolerance_violations"] == 0 && comparison["bitwise_equal"] ||
                    error("one-step field differs: $stem")
                participation = Dict{String,Vector{Int}}()
                for name in expected_passes
                    key = "active_" * name * "_worker_ids"
                    actual = get(report,key,nothing)
                    actual isa Vector && all(worker in 1:p for worker in actual) ||
                        error("invalid pass counter: $stem, $name")
                    participation[name] = Int.(actual)
                end
                row = Dict("variant" => variant, "n" => n, "block" => b,
                           "workers" => p, "participation" => participation,
                           "all_requested_participated" => all(
                               actual == collect(1:p) for actual in values(participation)),
                           "field_sha256" => Support.sha256file(field),
                           "oracle_sha256" => Support.sha256file(oracle),
                           "probe_source_sha256" => Support.sha256file(joinpath(probe,"src",
                               "operators","advection","cell_advection.jl")))
                push!(rows,row)
                Support.writejson(joinpath(output,"worker-evidence.json"),rows)
                rm(field)
                println("$stem: $participation")
                flush(stdout)
            end
        end
    end
    length(rows) == 4 || error("incomplete large-block participation study")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
