"""Run four copied-source flux/boundary/gather timing diagnostics."""
module Stage3ComponentProfile

include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "block_interior", "study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file
using TOML

const Study = MainStudy
const BlockStudy = BlockInteriorStudy

function layout(matrix)
    configuration = readjson(joinpath(matrix, "configuration.json"))
    if get(configuration, "phase", "") == "matrix" &&
            haskey(configuration, "steps_by_n")
        [2048, 2048] in configuration["cases"] ||
            error("final block cohort lacks the 2048 one-block case")
        steps = configuration["steps_by_n"]["2048"]
        source = Dict("baseline" => BlockStudy.BASELINE,
                      "current" => joinpath(matrix, "current-source"))
        manifest = readjson(joinpath(matrix, "manifest.json"))
        relative = joinpath("src", "operators", "advection", "cell_advection.jl")
        manifest_key = joinpath(BlockStudy.ROOT, relative)
        locked = get(manifest, manifest_key, get(manifest, relative, nothing))
        locked == sha256file(joinpath(source["current"], relative)) ||
            error("final cohort source snapshot differs from its locked manifest")
        initial = joinpath(matrix, "initial-n2048.bin")
        oracle = joinpath(matrix, "oracle-n2048-b2048-s$steps.bin")
        filesize(initial) == filesize(oracle) == 8*2048^2 ||
            error("final cohort field fixture has the wrong size")
        return (; final=true, source, variants=("baseline", "current"),
                steps, initial, oracle)
    end
    return (; final=false,
            source=Dict("frozen" => Study.FROZEN_SOURCE,
                        "current" => Study.ROOT),
            variants=("frozen", "current"), steps=16,
            initial=joinpath(matrix, "initial-2048.bin"),
            oracle=joinpath(matrix, "upwind-oracle-2048-b2048-s16.bin"))
end

function main(args=ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: julia component_profile_study.jl MATRIX OUTPUT")
        return nothing
    end
    length(args) == 2 || error("usage: julia component_profile_study.jl MATRIX OUTPUT")
    matrix, output = abspath.(args)
    study = layout(matrix)
    isdir(output) || error("component output must already contain two probe copies")
    !ispath(joinpath(output, "component-evidence.json")) ||
        error("component evidence already exists")
    ENV["STAGE3_FAILURE_DIR"] = output
    placements = readjson(joinpath(matrix, "placement.json"))
    rows = Dict{String,Any}[]
    log = Any[]
    for variant in study.variants
        source = joinpath(output, variant)
        receipt = readjson(joinpath(source, "component-probe.json"))
        clean = study.source[variant]
        relative = joinpath("src", "operators", "advection", "cell_advection.jl")
        receipt["source"] == clean &&
            receipt["source_sha256"] == sha256file(joinpath(clean, relative)) &&
            receipt["component_source_sha256"] ==
                sha256file(joinpath(source, relative)) &&
            receipt["component_driver_sha256"] ==
                sha256file(joinpath(source, "component_driver.jl")) ||
            error("clean $variant source changed since probe creation")
        for p in (4,8)
            cpus = placements[string(p)]
            stem = "$variant-2048-one-p$p"
            field = joinpath(output, "$stem.bin")
            report_path = joinpath(output, "$stem.toml")
            arguments = String["source=$source", "mode=measure", "kernel=upwind",
                "n=2048", "block=2048", "steps=$(study.steps)",
                "cpus=$(join(cpus, ','))", "input=$(study.initial)",
                "output=$field", "report=$report_path"]
            if study.final
                BlockStudy.launch(output, joinpath(source, "component_driver.jl"),
                                  p, cpus, arguments)
            else
                Study.run_julia(joinpath(source, "component_driver.jl"),
                                Study.FROZEN_PROJECT, p, arguments, cpus, log)
            end
            report = TOML.parsefile(report_path)
            report["source"] == source && report["n"] == 2048 &&
                report["block"] == 2048 && report["steps"] == study.steps &&
                report["threads"] == p || error("component report identity differs: $stem")
            if study.final
                comparison = BlockStudy.full_field(field, study.oracle, 2048)
                comparison["bitwise_equal"] &&
                    comparison["tolerance_violations"] == 0 &&
                    comparison["max_abs_vs_oracle"] == 0 ||
                    error("component-timed field differs: $stem")
            else
                comparison = Study.full_field(field, study.oracle, 2048)
                comparison["max_abs_vs_oracle"] == 0 &&
                    comparison["tolerance_violations"] == 0 &&
                    comparison["field_sha256"] == Study.digest(study.oracle) ||
                    error("component-timed field differs: $stem")
            end
            total = report["time_seconds"]
            components = Dict(name=>report["component_$(name)_seconds"]
                              for name in ("flux", "boundary", "gather"))
            accounted = sum(values(components))
            0 < accounted <= total*1.01 || error("implausible component accounting: $stem")
            row = Dict{String,Any}(
                "variant"=>variant, "n"=>2048, "block"=>2048,
                "requested_workers"=>p, "steps"=>study.steps, "total_seconds"=>total,
                "component_seconds"=>components, "accounted_fraction"=>accounted/total,
                "component_fraction_of_total"=>Dict(key=>value/total
                                                       for (key,value) in components),
                "field_sha256"=>sha256file(field),
                "oracle_sha256"=>sha256file(study.oracle),
                "instrumented_source_sha256"=>sha256file(joinpath(source, relative)))
            push!(rows, row)
            writejson(joinpath(output, "component-evidence.json"), rows)
            println("$stem: flux $(round(100components["flux"]/total; digits=1))%, " *
                    "gather $(round(100components["gather"]/total; digits=1))%, " *
                    "boundary $(round(100components["boundary"]/total; digits=1))%")
        end
    end
    length(rows) == 4 || error("component profile study incomplete")
    println("validated four copied-source component observations")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3ComponentProfile.main()
