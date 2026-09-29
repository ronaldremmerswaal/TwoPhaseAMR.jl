using Serialization
include("compare.jl")

const WORKER = joinpath(@__DIR__, "toy_worker.jl")
const COUNTS = (1, 2, 3, 4, 7, 8, 16, 32)

function run_worker(threads, path, project)
    thread_option = VERSION < v"1.12" ? string(threads) : "$threads,0"
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project --threads=$thread_option --gcthreads=1 $WORKER $path`
    run(command)
    return open(deserialize, path)
end

function validate_record(output)
    output["schema"] == 1 || error("toy record schema mismatch")
    results = output["results"]
    length(results) == 24 || error("toy record missing fixture/backend/flux case")
    for (name, record) in results
        compare_number(record["mass_before"], record["mass_after"], "$name.mass")
        length(record["q"]) == prod(record["shape"]) ||
            error("toy field shape mismatch at $name")
        record["steps"] > 0 || error("empty toy trajectory at $name")
    end
    for name in ("constant_square", "constant_odd", "variable_shear")
        reference = results["$name/recompute/serial"]
        for flux in (:recompute, :stored), backend in (:serial, :threads, :spawn, :polyester)
            compare_value(reference, results["$name/$flux/$backend"],
                          "$name/$flux/$backend")
        end
    end
end

function main()
    output_option = findfirst(argument -> startswith(argument, "--output-dir="), ARGS)
    project_option = findfirst(argument -> startswith(argument, "--project="), ARGS)
    directory = isnothing(output_option) ? nothing :
        abspath(split(ARGS[output_option], "="; limit = 2)[2])
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
        baseline = run_worker(1, joinpath(output_directory, "toy-1.bin"), project)
        baseline["default_threads"] == 1 || error("one-worker process used wrong pool")
        validate_record(baseline)
        println("1 thread: recorded $(length(baseline["results"])) toy cases")
        for threads in counts
            threads == 1 && continue
            candidate = run_worker(threads, joinpath(output_directory, "toy-$threads.bin"), project)
            candidate["default_threads"] == threads || error("worker pool mismatch")
            candidate["julia_version"] == baseline["julia_version"] ||
                error("Julia version mismatch")
            candidate["toy_source"] == baseline["toy_source"] ||
                error("toy source changed during comparison")
            candidate["toy_extension"] == baseline["toy_extension"] ||
                error("toy Polyester extension changed during comparison")
            validate_record(candidate)
            largest = compare_value(
                baseline["results"], candidate["results"], "toy_results",
            )
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
