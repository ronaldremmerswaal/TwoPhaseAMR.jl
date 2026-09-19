using DelimitedFiles

function requested_threads()
    raw = get(ENV, "TWOPHASE_THREAD_COUNTS", "1,2,4")
    values = unique(sort(parse.(Int, split(raw, ','))))
    all(>(0), values) || error("thread counts must be positive")
    return values
end

output_directory = isempty(ARGS) ?
    joinpath(@__DIR__, "threaded-kernel-results") : abspath(ARGS[1])
mkpath(output_directory)
project = dirname(@__DIR__)
worker = joinpath(@__DIR__, "threaded_kernels.jl")
results = String[]
reuse_results = lowercase(get(ENV, "TWOPHASE_REUSE_RESULTS", "false")) in
    ("1", "true", "yes")
for threads in requested_threads()
    output = joinpath(output_directory, "threads-$threads.csv")
    if reuse_results && isfile(output)
        println("Reusing ", output)
    else
        command = `$(Base.julia_cmd()) --project=$project --threads=$threads $worker $output`
        println("Running core-kernel suite with ", threads, " thread(s)")
        run(setenv(command, "OPENBLAS_NUM_THREADS" => "1"))
    end
    push!(results, output)
end

function read_rows(path)
    data, header = readdlm(path, ',', header = true)
    names = Symbol.(vec(header))
    return [NamedTuple{Tuple(names)}(Tuple(data[row, :])) for row in axes(data, 1)]
end

all_rows = reduce(vcat, read_rows.(results))
baseline = Dict(
    (String(row.case), String(row.kernel)) => row for row in all_rows
        if Int(row.threads) == 1
)
summary = joinpath(output_directory, "scaling.csv")
open(summary, "w") do io
    println(
        io,
        "case,kernel,threads,time_ns,speedup,efficiency,bytes,allocations,checksum",
    )
    for row in all_rows
        key = (String(row.case), String(row.kernel))
        reference = baseline[key]
        speedup = Float64(reference.time_ns) / Float64(row.time_ns)
        efficiency = speedup / Int(row.threads)
        reference_checksum = Float64(reference.checksum)
        checksum = Float64(row.checksum)
        isapprox(
            checksum, reference_checksum; rtol = 5.0e-11,
            atol = 5.0e-12 * max(1.0, abs(reference_checksum)),
        ) || error(
            "threaded checksum mismatch for $(key): " *
                "$checksum != $reference_checksum",
        )
        println(
            io, row.case, ',', row.kernel, ',', Int(row.threads), ',',
            row.time_ns, ',', speedup, ',', efficiency, ',', row.bytes, ',',
            row.allocations, ',', row.checksum,
        )
    end
end
println("Wrote ", summary)
