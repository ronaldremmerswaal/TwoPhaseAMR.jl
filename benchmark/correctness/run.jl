using Serialization
include("compare.jl")

const WORKER = joinpath(@__DIR__, "worker.jl")
const COUNTS = (1, 2, 3, 4, 7, 8, 16, 32)

function run_worker(threads, path, project)
    # Julia 1.10 rejects an explicit zero-thread interactive pool. It starts
    # with no interactive pool by default; Julia 1.12 accepts p,0 explicitly.
    thread_option = VERSION < v"1.12" ? string(threads) : "$threads,0"
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project --threads=$thread_option --gcthreads=1 $WORKER $path`
    run(command)
    return open(deserialize, path)
end

function validate_invariants(results)
    for name in ("scalar_one", "scalar_many", "scalar_mixed")
        item = results[name]
        q = item["quantities"]
        compare_number(q["mass_before"], q["mass_after"], "$name.mass")
        item["cells"] >= 1024 || error("$name does not reach parallel grain")
    end
    wy = results["wy"]
    wy["cells"] >= 1024 || error("WY does not reach parallel grain")
    for index in 1:3
        q = wy["quantities"]
        compare_number(
            q["phase_before"][index], q["phase_after"][index],
            "wy.phase_integral[$index]",
        )
    end
    results["staggered"]["cells"] >= 1024 ||
        error("staggered does not reach parallel grain")
    adaptive = results["adaptive"]["quantities"]
    compare_number(
        adaptive["mass_before"], adaptive["mass_refined"],
        "adaptive.refinement_mass",
    )
    compare_number(adaptive["mass_after"], 1.0, "adaptive.constant_mass")
end

function main()
    output_option = findfirst(argument -> startswith(argument, "--output-dir="), ARGS)
    project_option = findfirst(argument -> startswith(argument, "--project="), ARGS)
    directory = if isnothing(output_option)
        nothing
    else
        abspath(split(ARGS[output_option], "="; limit = 2)[2])
    end
    project = isnothing(project_option) ? dirname(dirname(@__DIR__)) :
        abspath(split(ARGS[project_option], "="; limit = 2)[2])
    excluded = filter(x -> !isnothing(x), [output_option, project_option])
    arguments = ARGS[setdiff(eachindex(ARGS), excluded)]
    counts = isempty(arguments) ? [1, 4] :
        arguments == ["--all"] ? collect(COUNTS) : parse.(Int, arguments)
    all(in(COUNTS), counts) || error("allowed counts: $(collect(COUNTS))")
    1 in counts || pushfirst!(counts, 1)
    length(unique(counts)) == length(counts) || error("duplicate thread count")
    function execute(output_directory)
        baseline = run_worker(1, joinpath(output_directory, "1.bin"), project)
        baseline["default_threads"] == 1 || error("one-worker process used wrong pool")
        baseline["blas_threads"] == 1 || error("BLAS was not serial")
        validate_invariants(baseline["results"])
        println("1 thread: recorded $(length(baseline["results"])) fixtures")
        for threads in counts
            threads == 1 && continue
            candidate = run_worker(threads, joinpath(output_directory, "$threads.bin"), project)
            candidate["schema"] == baseline["schema"] || error("schema mismatch")
            candidate["repository_revision"] == baseline["repository_revision"] ||
                error("repository revision mismatch")
            candidate["source_sha256"] == baseline["source_sha256"] ||
                error("package source changed during comparison")
            candidate["julia_version"] == baseline["julia_version"] ||
                error("Julia version mismatch")
            candidate["default_threads"] == threads || error("worker pool mismatch")
            candidate["blas_threads"] == 1 || error("BLAS was not serial")
            results = candidate["results"]
            validate_invariants(results)
            compare_value(
                baseline["results"]["scalar_one"],
                results["concurrent_one"], "concurrent_one",
            )
            compare_value(
                baseline["results"]["scalar_many"],
                results["concurrent_many"], "concurrent_many",
            )
            delete!(results, "concurrent_one")
            delete!(results, "concurrent_many")
            largest = compare_value(baseline["results"], results, "results")
            println("$threads threads: passed; largest absolute delta $largest")
        end
    end
    if isnothing(directory)
        mktempdir(execute)
    else
        mkpath(directory)
        execute(directory)
        println("serialized records: $directory")
    end
end

main()
