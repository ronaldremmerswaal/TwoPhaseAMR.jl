#!/usr/bin/env julia
"""Bounded paired diagnostic: original versus 128×8 copied regular-face traversal."""
module BlockInteriorFluxTileStudy

include(joinpath(@__DIR__, "study.jl"))
include(joinpath(@__DIR__, "make_flux_tile_probe.jl"))
using .BlockInteriorStudy
using .BlockInteriorFluxTileProbe
using Random
using TOML

const Study = BlockInteriorStudy
const Probe = BlockInteriorFluxTileProbe
const Support = Study.ScalingSupport
const VALIDATOR = joinpath(@__DIR__, "flux_tile_validation_driver.jl")
const TEST_FILE = joinpath(Study.ROOT, "test", "operators", "advection",
                           "cell_advection_compact.jl")
const DEFAULT_STEPS = Dict(2048 => 128, 1024 => 1024)
const WORKERS = (1, 4, 8)

function parse_args(args)
    if any(arg -> arg in ("-h", "--help"), args)
        println("usage: flux_tile_study.jl --cohort FINAL_COHORT --output NEW_DIRECTORY ",
                "[--repeats 3] [--case 2048:2048] [--case 1024:8]")
        return nothing
    end
    iseven(length(args)) || error("every option requires one value")
    values = Dict{String,String}()
    cases = Tuple{Int,Int}[]
    for i in 1:2:length(args)
        flag, value = args[i], args[i + 1]
        if flag == "--case"
            parts = split(value, ':'; limit=2)
            length(parts) == 2 || error("--case must be N:BLOCK")
            push!(cases, (parse(Int, parts[1]), parse(Int, parts[2])))
        elseif flag in ("--cohort", "--output", "--repeats")
            !haskey(values, flag) || error("duplicate option $flag")
            values[flag] = value
        else
            error("unknown option $flag")
        end
    end
    haskey(values, "--cohort") && haskey(values, "--output") ||
        error("--cohort and --output are required")
    cohort, output = abspath(values["--cohort"]), abspath(values["--output"])
    isdir(cohort) || error("final cohort missing: $cohort")
    !ispath(output) || error("output must be a new directory: $output")
    isempty(cases) && push!(cases, (2048, 2048))
    length(cases) <= 2 && length(unique(cases)) == length(cases) ||
        error("one or two distinct cases required")
    all(haskey(DEFAULT_STEPS, n) && b > 0 && n % b == 0 for (n,b) in cases) ||
        error("cases must use supported 1024/2048 meshes and divisible blocks")
    repeats = parse(Int, get(values, "--repeats", "3"))
    1 <= repeats <= 5 || error("bounded diagnostic requires 1–5 repeats")
    return (; cohort, output, cases, repeats)
end

function completed_cohort(cohort, cases)
    configuration = Support.readjson(joinpath(cohort, "configuration.json"))
    configuration["phase"] == "matrix" || error("input is not the accepted final matrix")
    available = Set((Int(pair[1]), Int(pair[2])) for pair in configuration["cases"])
    all(case in available for case in cases) || error("requested case absent from final cohort")
    summary = Support.readjson(joinpath(cohort, "summary.json"))
    rows = Support.readjsonl(joinpath(cohort, "raw.jsonl"))
    expected = 2 * length(available) * length(WORKERS) * configuration["repeats"]
    summary["samples"] == expected && length(rows) == expected ||
        error("final cohort has not completed its accepted matrix")
    all(row["bitwise_equal"] && row["tolerance_violations"] == 0 for row in rows) ||
        error("final cohort contains a field mismatch")
    source = joinpath(cohort, "current-source")
    clean = joinpath(source, Probe.RELATIVE)
    Support.sha256file(clean) == Probe.EXPECTED_CLEAN_SHA256 ||
        error("final cohort current-source is not the locked clean implementation")
    manifest = Support.readjson(joinpath(cohort, "manifest.json"))
    manifest_key = joinpath(Study.ROOT, Probe.RELATIVE)
    locked = get(manifest, manifest_key, get(manifest, Probe.RELATIVE, nothing))
    locked == Probe.EXPECTED_CLEAN_SHA256 ||
        error("final cohort manifest differs from copied clean source")
    placements = Support.readjson(joinpath(cohort, "placement.json"))
    all(length(placements[string(p)]) == p for p in WORKERS) ||
        error("final cohort placement lacks the required worker counts")
    return (; configuration, source, placements)
end

function locked_hashes(options, clean, tile, driver, fixtures)
    paths = String[@__FILE__, joinpath(@__DIR__, "study.jl"),
        joinpath(@__DIR__, "make_flux_tile_probe.jl"), VALIDATOR, TEST_FILE,
        joinpath(@__DIR__, "..", "..", "Support.jl"), driver,
        Study.JULIA, joinpath(Study.PROJECT, "Project.toml"),
        joinpath(Study.PROJECT, "Manifest.toml"),
        joinpath(Study.ROOT, "benchmark", "scaling", "Project.toml"),
        joinpath(Study.ROOT, "benchmark", "scaling", "Manifest.toml"),
        joinpath(options.cohort, "manifest.json"),
        joinpath(options.cohort, "configuration.json"),
        joinpath(options.cohort, "placement.json"),
        joinpath(options.cohort, "raw.jsonl"),
        joinpath(options.cohort, "summary.json"),
        joinpath(clean, Probe.RELATIVE), joinpath(tile, Probe.RELATIVE),
        joinpath(tile, "flux-tile-receipt.json"),
        joinpath(tile, "flux-tile.patch")]
    for fixture in values(fixtures)
        append!(paths, (fixture.initial, fixture.oracle, fixture.oracle_report))
    end
    all(isfile, paths) || error("a required diagnostic source or fixture is missing")
    return Dict(path => Support.sha256file(path) for path in sort!(unique(paths)))
end

function reference_fixture(options, n, b, clean, driver, pin1)
    initial = joinpath(options.output, "initial-n$n.bin")
    original = joinpath(options.cohort, "initial-n$n.bin")
    isfile(original) && filesize(original) == 8*n*n ||
        error("final cohort initial field missing or incomplete")
    if isfile(initial)
        Support.sha256file(initial) == Support.sha256file(original) ||
            error("shared initial field differs between selected blocks")
    else
        cp(original, initial)
    end
    steps = DEFAULT_STEPS[n]
    oracle = joinpath(options.output, "oracle-n$n-b$b-s$steps.bin")
    oracle_report = joinpath(options.output, "oracle-n$n-b$b-s$steps.toml")
    Study.launch(options.output, driver, 1, pin1,
        ["source=$clean", "mode=measure", "kernel=upwind", "n=$n",
         "block=$b", "steps=$steps", "cpus=$(join(pin1, ','))",
         "input=$initial", "output=$oracle", "report=$oracle_report"])
    report = TOML.parsefile(oracle_report)
    report["source"] == clean && report["n"] == n && report["block"] == b &&
        report["steps"] == steps && report["field_sha256"] == Support.sha256file(oracle) ||
        error("fresh clean-source oracle identity differs")
    return (; initial, oracle, oracle_report, steps,
            initial_sha256=Support.sha256file(initial),
            oracle_sha256=Support.sha256file(oracle))
end

function validate_copied_suite(output, label, source, threads, cpus, expected_hash)
    report = joinpath(output, "suite-$label-p$threads.toml")
    Study.launch(output, VALIDATOR, threads, cpus,
        [source, TEST_FILE, expected_hash, report])
    result = TOML.parsefile(report)
    result["status"] == "passed" &&
        result["loaded_advection_sha256"] == expected_hash &&
        result["loaded_advection_file"] == joinpath(source, Probe.RELATIVE) &&
        result["regular_flux_method_file"] == realpath(joinpath(source, Probe.RELATIVE)) &&
        result["original_test_sha256"] == Support.sha256file(TEST_FILE) &&
        result["threads"] == threads || error("copied-module suite receipt differs")
    return result
end

function validate_sample(output, variant, source, n, b, p, repeat, fixture, cpus)
    stem = "$variant-n$n-b$b-p$p-r$repeat"
    field = joinpath(output, "$stem.bin")
    report = TOML.parsefile(joinpath(output, "$stem.toml"))
    for (key, expected) in ("mode" => "measure", "kernel" => "upwind",
                            "n" => n, "block" => b, "threads" => p,
                            "steps" => fixture.steps, "source" => source)
        report[key] == expected || error("$stem: $key differs")
    end
    report["dt"] == 0.35/n &&
        isfinite(report["time_seconds"]) && report["time_seconds"] >= 0.2 &&
        isapprox(report["ns_per_unknown_step"],
                 1e9*report["time_seconds"]/(n*n*fixture.steps); rtol=1e-12) ||
        error("$stem: short or invalid timed trajectory")
    [entry["cpu"] for entry in report["affinity"]] == cpus &&
        all(entry["observed_cpu"] == entry["cpu"] for entry in report["affinity"]) ||
        error("$stem: affinity differs")
    field_hash = Support.sha256file(field)
    report["field_sha256"] == field_hash &&
        field_hash == fixture.oracle_sha256 || error("$stem: final-field SHA differs")
    comparison = Study.full_field(field, fixture.oracle, n)
    comparison["bitwise_equal"] && comparison["tolerance_violations"] == 0 &&
        comparison["max_abs_vs_oracle"] == 0 ||
        error("$stem: full field differs from the fresh clean-source oracle")
    merge!(report, comparison)
    merge!(report, Dict("variant" => variant, "repeat" => repeat,
        "initial_sha256" => fixture.initial_sha256,
        "oracle_sha256" => fixture.oracle_sha256,
        "source_sha256" => Support.sha256file(joinpath(source, Probe.RELATIVE))))
    rm(field)
    return report
end

function summarize(rows, options)
    result = Dict{String,Any}[]
    for (n,b) in options.cases, p in WORKERS
        control = [row["ns_per_unknown_step"] for row in rows if
                   row["n"] == n && row["block"] == b &&
                   row["threads"] == p && row["variant"] == "control"]
        tile = [row["ns_per_unknown_step"] for row in rows if
                row["n"] == n && row["block"] == b &&
                row["threads"] == p && row["variant"] == "tile128x8"]
        length(control) == length(tile) == options.repeats ||
            error("incomplete paired diagnostic group")
        cmed, tmed = Support.median_value(control), Support.median_value(tile)
        push!(result, Dict("n" => n, "block" => b, "workers" => p,
            "repeats" => options.repeats, "steps" => DEFAULT_STEPS[n],
            "control_median_ns_per_unknown_step" => cmed,
            "tile_median_ns_per_unknown_step" => tmed,
            "control_mad_ns_per_unknown_step" => Support.mad_value(control),
            "tile_mad_ns_per_unknown_step" => Support.mad_value(tile),
            "control_over_tile_median_ratio" => cmed/tmed,
            "interpretation" => "exploratory paired copied-source diagnostic"))
    end
    return Dict("schema" => 1, "samples" => length(rows), "rows" => result,
                "comparison" => "same fixed trajectory, fresh process per variant and pair")
end

function main(args=ARGS)
    options = parse_args(args)
    options === nothing && return
    final = completed_cohort(options.cohort, options.cases)
    mkpath(options.output)
    control, tile = joinpath(options.output, "control"), joinpath(options.output, "tile128x8")
    mkpath(control)
    cp(joinpath(final.source, "src"), joinpath(control, "src"))
    receipt = Probe.make_probe(final.source, tile)
    Support.source_hash(control) == Support.source_hash(final.source) ||
        error("control copy differs from final-cohort source")
    driver = joinpath(options.output, "driver.jl")
    cp(Study.DRIVER, driver)
    pin1 = final.placements["1"]
    fixtures = Dict{Tuple{Int,Int},Any}()
    for (n,b) in options.cases
        fixtures[n,b] = reference_fixture(options, n, b, control, driver, pin1)
    end
    suite = Dict(
        "control" => validate_copied_suite(options.output, "control", control, 1,
                                            pin1, Probe.EXPECTED_CLEAN_SHA256),
        "tile128x8" => validate_copied_suite(options.output, "tile128x8", tile, 4,
                                             final.placements["4"],
                                             receipt["candidate_source_sha256"]))
    source_hashes = Dict("control" => Support.source_hash(control),
                         "tile128x8" => Support.source_hash(tile))
    frozen = locked_hashes(options, control, tile, driver, fixtures)
    schedule = Any[]
    pairs = [(repeat,n,b,p) for repeat in 0:options.repeats-1
             for (n,b) in options.cases for p in WORKERS]
    shuffle!(MersenneTwister(20261010), pairs)
    for (repeat,n,b,p) in pairs
        variants = iseven(repeat + p) ? ("control", "tile128x8") :
                                        ("tile128x8", "control")
        push!(schedule, Dict("repeat" => repeat, "n" => n, "block" => b,
                             "workers" => p, "order" => collect(variants)))
    end
    Support.writejson(joinpath(options.output, "configuration.json"), Dict(
        "schema" => 1, "final_cohort" => options.cohort,
        "cases" => [[n,b] for (n,b) in options.cases],
        "steps" => Dict(string(n) => DEFAULT_STEPS[n] for (n,_) in options.cases),
        "workers" => collect(WORKERS), "repeats" => options.repeats,
        "minimum_timed_seconds" => 0.2, "placement" => final.placements,
        "source_tree_sha256" => source_hashes, "locked_files_sha256" => frozen,
        "copied_suite" => suite, "probe_receipt" => receipt,
        "driver_sha256" => Support.sha256file(driver)))
    Support.writejson(joinpath(options.output, "job_order.json"), schedule)
    rows = Any[]
    for job in schedule
        repeat, n, b, p = job["repeat"], job["n"], job["block"], job["workers"]
        fixture = fixtures[n,b]
        cpus = final.placements[string(p)]
        for variant in job["order"]
            locked_hashes(options, control, tile, driver, fixtures) == frozen &&
                all(Support.source_hash(root) == source_hashes[name]
                    for (name,root) in (("control",control), ("tile128x8",tile))) ||
                error("diagnostic source, input, or environment changed")
            source = variant == "control" ? control : tile
            stem = "$variant-n$n-b$b-p$p-r$repeat"
            field, report = joinpath(options.output, "$stem.bin"),
                            joinpath(options.output, "$stem.toml")
            Study.launch(options.output, driver, p, cpus,
                ["source=$source", "mode=measure", "kernel=upwind",
                 "n=$n", "block=$b", "steps=$(fixture.steps)",
                 "cpus=$(join(cpus, ','))", "input=$(fixture.initial)",
                 "output=$field", "report=$report"])
            locked_hashes(options, control, tile, driver, fixtures) == frozen ||
                error("diagnostic source or input changed during launch")
            row = validate_sample(options.output, variant, source, n, b, p,
                                  repeat, fixture, cpus)
            Support.appendjsonl(joinpath(options.output, "raw.jsonl"), row)
            push!(rows, row)
            println("$variant n$n b$b p$p r$repeat: ",
                    round(row["ns_per_unknown_step"]; digits=3), " ns/cell-step")
            flush(stdout)
        end
    end
    Support.writejson(joinpath(options.output, "summary.json"), summarize(rows, options))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end

end # module
