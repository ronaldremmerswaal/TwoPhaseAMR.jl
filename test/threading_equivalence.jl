# Serial-versus-threaded bitwise equivalence of complete flows. Every case of
# `threading_equivalence_cases.jl` runs in a fresh Julia process per thread
# count with the threading grain at one, so the threaded branch of every loop
# is taken however small the mesh is; states, step counts and warm-step
# allocations must be identical. The thread counts run concurrently, which
# cannot change a result: no kernel decision depends on wall time.
#
# Not part of the default suite: the `threading-equivalence` CI job runs this
# group on every pull request. `THREADING_EQUIVALENCE_THREADS` (default
# `1,2,4`) lists the thread counts, the first being the serial reference.
const EQUIVALENCE_THREADS = "1,2,4"

"""Case whose run must adapt the mesh, so that the comparison covers the
transfer and rebuild paths. One regrid is part of the case: a run that stops
adapting would silently stop covering them."""
const ADAPTED_CASE = "stokes3_two"

const EQUIVALENCE_CASES = (
    "taylor_green", "droplet", "capillary", "capillary_mimetic", "stokes3_two", "cylinder",
)

"""
Bytes one warm step may allocate after two warm-up steps, measured on the
audited commit (main, Ryzen 9 5950X) plus a small margin. `serial` is the
one-thread path and `threaded` the path with two or more threads.
`WARM_STEP_BUDGET` holds the Julia 1.12.7 values; `WARM_STEP_BUDGET_111` the
Julia 1.11.9 values, which are higher because Polyester boxes the closure
argument of a threaded launch there (16-32 B per launch, nothing from 1.12 on,
as in `test/threading.jl`) and because the same warm step allocates more
serially on 1.11.
"""
const WARM_STEP_BUDGET = (;
    # Measured on main (Julia 1.12.7): 6_416 B serial, 32_736 B threaded.
    taylor_green = (serial = 7_168, threaded = 35_840),
    # Measured on main (Julia 1.12.7): 1_223_448 B serial, 1_386_728 B threaded.
    capillary = (serial = 1_280_000, threaded = 1_450_000),
)
const WARM_STEP_BUDGET_111 = (;
    # Measured on main (Julia 1.11.9): 6_624 B serial, 41_328 B threaded.
    taylor_green = (serial = 7_168, threaded = 45_000),
    # Measured on main (Julia 1.11.9): 2_410_440 B serial, 2_586_456 B threaded.
    capillary = (serial = 2_500_000, threaded = 2_700_000),
)

function warm_step_budget(case, threads)
    budgets = VERSION < v"1.12" ? WARM_STEP_BUDGET_111 : WARM_STEP_BUDGET
    tables = getproperty(budgets, Symbol(case))
    return threads == 1 ? tables.serial : tables.threaded
end

function equivalence_command(threads)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    project = dirname(Base.active_project())
    script = joinpath(@__DIR__, "threading_equivalence_cases.jl")
    command = `$julia --project=$project --threads=$threads --startup-file=no --history-file=no $script`
    return addenv(
        command, "OPENBLAS_NUM_THREADS" => "1",
        "JULIA_NUM_THREADS" => string(threads), "JULIA_LOAD_PATH" => "@:@stdlib",
    )
end

"""Split a `key=value` line of the case process into a dictionary."""
function record_fields(line)
    fields = Dict{String, String}()
    for token in split(line)
        parts = split(token, '='; limit = 2)
        length(parts) == 2 || error("unexpected record in the case output: $(repr(line))")
        fields[first(parts)] = last(parts)
    end
    return fields
end

"""Parse the case and allocation records of one thread count."""
function parse_records(output)
    cases = Dict{String, NamedTuple}()
    allocations = Dict{String, Int}()
    for line in split(output, '\n')
        isempty(line) && continue
        fields = record_fields(line)
        if haskey(fields, "alloc")
            allocations[fields["alloc"]] = parse(Int, fields["bytes"])
        else
            cases[fields["case"]] = (
                threads = parse(Int, fields["threads"]),
                steps = parse(Int, fields["steps"]),
                mesh_updates = parse(Int, fields["mesh_updates"]),
                blocks = parse(Int, fields["blocks"]),
                seconds = parse(Float64, fields["seconds"]),
                series = parse(Int, fields["series"]),
                diagnostics = fields["diagnostics"],
                sha256 = fields["sha256"],
            )
        end
    end
    return cases, allocations
end

"""
    equivalence_runs(thread_counts)

Run the case script once per thread count, all at the same time, and return one
`(cases, allocations)` pair per thread count in order.
"""
function equivalence_runs(thread_counts)
    outputs = [IOBuffer() for _ in thread_counts]
    failures = [IOBuffer() for _ in thread_counts]
    processes = map(thread_counts, outputs, failures) do threads, output, failure
        run(
            pipeline(equivalence_command(threads); stdout = output, stderr = failure);
            wait = false,
        )
    end
    foreach(wait, processes)
    return map(thread_counts, processes, outputs, failures) do threads, process, output, failure
        process.exitcode == 0 || error(
            "the $(threads)-thread equivalence run exited with ", process.exitcode,
            ":\n", String(take!(output)), String(take!(failure)),
        )
        parse_records(String(take!(output)))
    end
end

"""Print what one process ran, so a CI log shows the cost per thread count."""
function report_run(threads, cases, allocations)
    println("threading equivalence on ", threads, " threads")
    for name in sort!(collect(keys(cases)))
        case = cases[name]
        println(
            "  ", rpad(name, 14), " threads=", case.threads, " steps=", case.steps,
            " mesh_updates=", case.mesh_updates, " blocks=", case.blocks,
            " series=", case.series, " seconds=", case.seconds,
        )
    end
    for name in sort!(collect(keys(allocations)))
        println("  ", rpad(name, 14), " warm_step_bytes=", allocations[name])
    end
    flush(stdout)
    return nothing
end

@testset "bitwise equivalence across thread counts" begin
    thread_counts = parse.(
        Int, split(get(ENV, "THREADING_EQUIVALENCE_THREADS", EQUIVALENCE_THREADS), ','),
    )
    @test length(thread_counts) >= 2
    runs = equivalence_runs(thread_counts)
    for (threads, (cases, allocations)) in zip(thread_counts, runs)
        report_run(threads, cases, allocations)
        @test sort!(collect(keys(cases))) == sort!(collect(EQUIVALENCE_CASES))
        @test length(allocations) == 2
        for (name, case) in cases
            # A `--threads` flag the child ignored would make the comparison
            # pass trivially.
            @test case.threads == threads
            @test case.steps > 0
            @test case.blocks > 0
        end
        for (name, bytes) in allocations
            @test bytes <= warm_step_budget(name, threads)
        end
    end
    # Only this case is required to adapt; a run that stops adapting would
    # silently stop covering the transfer and rebuild paths.
    @test runs[1][1][ADAPTED_CASE].mesh_updates > 0
    # Four cases record diagnostics, whose reductions run in the threaded
    # kernels; a case that stops recording would weaken the comparison.
    for name in ("capillary", "droplet", "stokes3_two", "cylinder")
        @test runs[1][1][name].series > 0
    end
    # The Taylor--Green fixture records no diagnostics, like the allocation
    # flows, so their warm-step budgets measure the solver alone.
    @test runs[1][1]["taylor_green"].series == 0
    reference, reference_allocations = runs[1]
    for (cases, allocations) in runs[2:end]
        for (name, expected) in reference
            actual = cases[name]
            @test actual.steps == expected.steps
            @test actual.mesh_updates == expected.mesh_updates
            @test actual.blocks == expected.blocks
            @test actual.series == expected.series
            @test actual.diagnostics == expected.diagnostics
            @test actual.sha256 == expected.sha256
        end
        @test sort!(collect(keys(allocations))) ==
            sort!(collect(keys(reference_allocations)))
    end
end
