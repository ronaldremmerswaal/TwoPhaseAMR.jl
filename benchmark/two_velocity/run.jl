# Two-velocity verification suite: compares the two-velocity projections on
# the properties that matter (benchmark/README.md, docs page "Two-velocity
# verification"). From the repository root, in the benchmark environment:
#
#   julia --project=benchmark benchmark/two_velocity/run.jl [options]
#
#   --tier=1,2,3          tiers to run (default 1)
#   --methods=a,b,...     cmdgfm, cmdgfm_constant, mimetic, one_velocity (default all;
#                         tier 1 has no one-velocity cases)
#   --resolutions=32,64   tier-1 resolutions (default 32,64,128)
#   --impact=32:1,64:1    tier-3 impact cases n:member (default 32:1,...,32:5,64:1)
#   --output=DIR          results.toml and results.md (default benchmark/two_velocity/output)
#   --compare[=PATH]      compare with the baseline (default baseline.toml); exit 1 on a
#                         metric worse beyond its tolerance, a missing metric or a failed case
#   --record[=PATH]       write the run's tiers and methods into the baseline
#   --update-docs         rewrite the generated tables of docs/src/two-velocity-verification.md
#                         from the baseline (after --record: the new one; --tier= runs nothing)
#
# Exit status 1 also when a case failed (exception or non-converged projection).
include("common.jl")
include("tier1.jl")
include("tier2.jl")
include("tier3.jl")

const OPTIONS = ("tier", "methods", "resolutions", "impact", "output", "compare", "record", "update-docs")

function parse_options(args)
    options = Dict{String, String}()
    for arg in args
        startswith(arg, "--") || error("unknown argument $arg")
        key, value = occursin('=', arg) ? split(arg[3:end], '='; limit = 2) : (arg[3:end], "")
        key in OPTIONS || error("unknown option --$key (options: $(join(OPTIONS, ", ")))")
        options[key] = value
    end
    return options
end

parse_list(f, text) = [f(strip(item)) for item in split(text, ',') if !isempty(strip(item))]

function main(args)
    options = parse_options(args)
    tiers = parse_list(x -> parse(Int, x), get(options, "tier", "1"))
    all(in((1, 2, 3)), tiers) || error("tiers are 1, 2 and 3")
    names = parse_list(Symbol, get(options, "methods", join((m.name for m in METHODS), ',')))
    methods = method.(names)
    resolutions = parse_list(x -> parse(Int, x), get(options, "resolutions", "32,64,128"))
    impact = parse_list(get(options, "impact", join(("$n:$m" for (n, m) in IMPACT_CASES), ','))) do item
        n, member = parse.(Int, split(item, ':'))
        (n, member)
    end
    suite = Suite(tiers, methods, resolutions)
    @printf(
        "Two-velocity verification: tiers %s, methods %s, Julia %s, %d thread(s)%s\n",
        join(tiers, ","), join(names, ","), VERSION, Threads.nthreads(), PERTURBED ? ", PERTURBED" : "",
    )
    flush(stdout)
    elapsed = @elapsed for tier in sort(tiers)
        tier == 1 && run_tier1!(suite)
        tier == 2 && run_tier2!(suite)
        tier == 3 && run_tier3!(suite; cases = impact)
    end
    @printf("Ran in %.1f s\n", elapsed)
    output = get(options, "output", "")
    output = isempty(output) ? joinpath(@__DIR__, "output") : output
    mkpath(output)
    println("Results: ", write_toml(joinpath(output, "results.toml"), suite))
    println("Tables:  ", write_markdown(joinpath(output, "results.md"), suite))
    baseline = joinpath(@__DIR__, "baseline.toml")
    if haskey(options, "record")
        isempty(options["record"]) || (baseline = options["record"])
        println("Baseline recorded: ", record_baseline!(baseline, suite))
    end
    if haskey(options, "update-docs")
        page = joinpath(@__DIR__, "..", "..", "docs", "src", "two-velocity-verification.md")
        println("Docs tables: ", update_docs!(normpath(page), baseline))
    end
    ok = isempty(suite.failures)
    if haskey(options, "compare")
        ok &= compare(suite, isempty(options["compare"]) ? baseline : options["compare"])
    end
    println(ok ? "PASS" : "FAIL")
    return ok
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main(ARGS) ? 0 : 1)
