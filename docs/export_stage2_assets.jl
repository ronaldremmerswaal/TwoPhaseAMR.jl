# Export selected audited Stage 2 figures and aggregate rows into the docs.
include(joinpath(@__DIR__, "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DESTINATION = joinpath(@__DIR__, "src", "assets")
const SIZES = Set((512, 2048, 4096))
const WORKERS = Set((1, 4, 8))
const FIGURES = (
    "stage2-final-512.svg",
    "stage2-final-2048.svg",
    "stage2-final-4096.svg",
    "stage2-final-throughput-working-set.svg",
    "stage2-final-memory-allocation.svg",
)

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia docs/export_stage2_assets.jl RESULT_DIRECTORY [--destination DIR]")
        return
    end
    (length(args) == 1 || (length(args) == 3 && args[2] == "--destination")) ||
        error("usage: julia docs/export_stage2_assets.jl RESULT_DIRECTORY [--destination DIR]")
    destination = length(args) == 3 ? abspath(args[3]) : DESTINATION
    result_dir = abspath(args[1])
    raw_path = joinpath(result_dir, "raw.jsonl")
    report_path = joinpath(result_dir, "report-corrected.json")
    report = readjson(report_path)
    raw_hash = sha256file(raw_path)
    report["raw_jsonl_sha256"] == raw_hash ||
        error("report does not match the frozen raw samples")
    report["input_samples"] == 280 ||
        error("expected the completed 280-sample Stage 2 matrix")
    rows = [row for row in report["rows"] if
            row["size"] in SIZES && row["requested_workers"] in WORKERS]
    isempty(rows) && error("no mandatory-size rows in audited report")
    mkpath(destination)
    figure_hashes = Dict{String, String}()
    for name in FIGURES
        target = joinpath(destination, name)
        cp(joinpath(result_dir, name), target; force=true)
        figure_hashes[name] = sha256file(target)
    end
    relative = relpath(result_dir, ROOT)
    payload = Dict(
        "schema_version" => 1,
        "provenance" => Dict(
            "machine_local_artifact" => relative,
            "raw_jsonl_sha256" => raw_hash,
            "report_corrected_sha256" => sha256file(report_path),
            "report_script_sha256" => report["report_script_sha256"],
            "configuration_sha256" => report["config_sha256"],
            "figure_sha256" => figure_hashes,
            "generation_command" =>
                "julia --project=benchmark/scaling benchmark/scaling/report_final.jl $relative && " *
                "julia --project=benchmark/scaling docs/export_stage2_assets.jl $relative",
            "context" => "Ryzen 9 5950X, Julia 1.10.12, 2026-09-29 pinned spread",
        ),
        "selection" =>
            "Audited aggregate rows at 512/2048/4096 squared and p1/p4/p8; no raw samples",
        "rows" => rows,
    )
    writejson(joinpath(destination, "stage2-primary-summary.json"), payload)
    println("Exported $(length(rows)) aggregate rows and $(length(FIGURES)) SVGs to $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
