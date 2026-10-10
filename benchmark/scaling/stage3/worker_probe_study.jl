"""Verify untimed flux/gather participation on instrumented source copies."""
module Stage3WorkerProbeStudy

include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson, writejson, sha256file
using TOML

const Study = MainStudy

function main(args=ARGS)
    length(args) == 2 || error("usage: julia worker_probe_study.jl MATRIX OUTPUT")
    matrix, output = abspath.(args)
    probes = Dict(variant=>joinpath(output, variant) for variant in ("frozen", "current"))
    for (variant, root) in probes
        receipt = readjson(joinpath(root, "worker-probe.json"))
        clean = variant == "frozen" ? Study.FROZEN_SOURCE : Study.ROOT
        receipt["source_sha256"] == Study.digest(joinpath(clean, "src", "operators",
            "advection", "cell_advection.jl")) ||
            error("clean $variant source changed since probe creation")
    end
    ENV["STAGE3_FAILURE_DIR"] = output
    placements = readjson(joinpath(matrix, "placement.json"))
    log = Any[]
    rows = Dict{String,Any}[]
    for variant in ("frozen", "current"), p in (1,4,8), block in (1024,8)
        cpus = placements[string(p)]
        stem = "$variant-1024-b$block-p$p"
        field = joinpath(output, "$stem.bin")
        report_path = joinpath(output, "$stem.toml")
        Study.run_julia(Study.DRIVER, Study.FROZEN_PROJECT, p,
            String["source=$(probes[variant])", "mode=measure", "kernel=upwind",
                "n=1024", "block=$block", "steps=128", "cpus=$(join(cpus, ','))",
                "input=$(joinpath(matrix, "initial-1024.bin"))", "output=$field",
                "report=$report_path"], cpus, log)
        report = TOML.parsefile(report_path)
        oracle = joinpath(matrix, "upwind-oracle-1024-b$block-s128.bin")
        comparison = Study.full_field(field, oracle, 1024)
        comparison["tolerance_violations"] == 0 &&
            comparison["max_abs_vs_oracle"] == 0 &&
            comparison["field_sha256"] == Study.digest(oracle) ||
            error("instrumented full field differs: $stem")
        [entry["cpu"] for entry in report["affinity"]] == cpus ||
            error("instrumented pinning differs: $stem")
        flux = report["active_flux_worker_ids"]
        gather = report["active_gather_worker_ids"]
        row = Dict{String,Any}(
            "variant"=>variant, "n"=>1024, "block"=>block,
            "requested_workers"=>p, "actual_flux_workers"=>flux,
            "actual_gather_workers"=>gather,
            "all_requested_participated"=>flux == collect(1:p) && gather == collect(1:p),
            "field_sha256"=>comparison["field_sha256"],
            "oracle_sha256"=>Study.digest(oracle),
            "probe_source_sha256"=>Study.digest(joinpath(probes[variant], "src",
                "operators", "advection", "cell_advection.jl")))
        push!(rows, row)
        writejson(joinpath(output, "worker-evidence.json"), rows)
        println("$stem: flux $flux, gather $gather")
    end
    length(rows) == 12 || error("worker probe study incomplete")
    println("validated 12 instrumented worker/field observations")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && Stage3WorkerProbeStudy.main()
