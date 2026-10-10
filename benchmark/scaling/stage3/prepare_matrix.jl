# Prepare fixed Stage 3 production matrix metadata in a new directory.
include(joinpath(@__DIR__, "main_study.jl"))
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport

const MATRIX_SIZES = (512, 1024, 2048)
const MATRIX_WORKERS = (1, 4, 8)
const MATRIX_REPEATS = 5
const MATRIX_STEPS = Dict("512" => 1024, "1024" => 128, "2048" => 16)

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia prepare_matrix.jl --output DIRECTORY")
        return
    end
    length(args) == 2 && args[1] == "--output" ||
        error("usage: julia prepare_matrix.jl --output DIRECTORY")
    output = args[2]
    ispath(output) && error("output directory already exists: $output")
    freeze = readjson(joinpath(MainStudy.FREEZE, "freeze.json"))
    required = (MainStudy.JULIA,
                joinpath(MainStudy.FROZEN_PROJECT, "Project.toml"),
                joinpath(MainStudy.FROZEN_PROJECT, "Manifest.toml"),
                joinpath(MainStudy.FROZEN_SOURCE, "src", "TwoPhaseAMR.jl"))
    for path in required
        isfile(path) || error("required frozen-cohort file is missing: $path")
    end
    MainStudy.JULIA == freeze["julia_executable"] ||
        error("runner Julia executable differs from frozen receipt")
    MainStudy.source_manifest()
    config = Dict(
        "study" => "stage3_general_upwind_frozen_vs_final_remap",
        "baseline_git_revision" => freeze["git_revision"],
        "julia" => MainStudy.JULIA,
        "project" => relpath(MainStudy.FROZEN_PROJECT, MainStudy.ROOT),
        "depot" => MainStudy.DEPOT, "sizes" => collect(MATRIX_SIZES),
        "layouts" => ["one block", "8x8-cell blocks"],
        "workers" => collect(MATRIX_WORKERS), "repeats" => MATRIX_REPEATS,
        "fresh_processes_per_configuration" => MATRIX_REPEATS,
        "batching" => "one fresh process per repeat,worker count,source variant; six independently reset/warmed/timed size/layout cases per process",
        "minimum_timed_seconds" => 0.2, "steps" => MATRIX_STEPS,
        "velocity" => [0.7, -0.3], "dt" => "0.35/n",
        "numerics" => "general forward Euler upwind",
        "current_advection_sha256" => MainStudy.digest(joinpath(MainStudy.ROOT,
            "src", "operators", "advection", "cell_advection.jl")),
        "current_embedded_sha256" => MainStudy.digest(joinpath(MainStudy.ROOT,
            "src", "indicator", "embedded_operators.jl")),
    )
    mkpath(output)
    writejson(joinpath(output, "steps.json"), MATRIX_STEPS)
    writejson(joinpath(output, "configuration.json"), config)
    println("Prepared matrix metadata in $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
