# Shared parts of the two-velocity verification suite: the compared methods,
# the measured metrics, their TOML and Markdown output, and the comparison
# against a recorded baseline. The suite measures and reports; the contracts
# of the package stay in test/.
using TwoPhaseAMR, LinearAlgebra, Printf, TOML
BLAS.set_num_threads(1)

const TP = TwoPhaseAMR

"""A compared method: a two-velocity pressure projection, or the
one-velocity model (`projection === nothing`) as a reference."""
struct Method
    name::Symbol
    label::String
    projection::Union{Nothing, AbstractTwoVelocityProjection}
end

const METHODS = [
    Method(:cmdgfm, "CMDGFM-L", CMDGFMProjection()),
    Method(:cmdgfm_constant, "CMDGFM-C", CMDGFMProjection(jump_inference = ConstantJumpInference())),
    Method(:mimetic, "MM", MimeticProjection()),
    Method(:one_velocity, "1-vel", nothing),
]

method(name::Symbol) = only(m for m in METHODS if m.name === name)
two_velocity(m::Method) = !isnothing(m.projection)
velocity_model(m::Method) = two_velocity(m) ?
    TwoVelocityModel(; pressure_projection = m.projection) : OneVelocityModel()

"""Projection solver and workspace of a two-velocity method on `interface`."""
function projection_solver(m::Method, interface, density)
    if m.projection isa MimeticProjection
        solver = MimeticProjectionSolver(interface; density)
        return solver, MimeticProjectionWorkspace(solver)
    end
    solver = CMDGFMProjectionSolver(interface; density, jump_inference = m.projection.jump_inference)
    return solver, CMDGFMProjectionWorkspace(solver)
end

"""Projection tolerance (on `Δt max|div u_mix|`) and iteration limit of the
single projections: the audit laboratory's 1e-8. The round-off floor of the
defect grows with the pressure scale and the resolution: the hydrostatic
pressure of the T3 gravity case at DR 1000 puts it at 2e-10 / 7e-10 / 3e-9
(MM) and 2e-9 / 3e-9 / 5e-8 (CMDGFM, stalled BiCGSTAB) at n = 32 / 64 / 128,
so that case uses `HYDROSTATIC_TOLERANCE` = 1e-7; its metrics agree to eight
digits between 1e-7 and 1e-8 and to four down to 1e-11 (they are not solver
limited). `TWO_VELOCITY_PERTURB=1` loosens both to 1e-3 to exercise the
failure path of `--compare` with a genuinely degraded computation."""
const PERTURBED = get(ENV, "TWO_VELOCITY_PERTURB", "0") == "1"
const PROJECTION_TOLERANCE = PERTURBED ? 1.0e-3 : 1.0e-8
const HYDROSTATIC_TOLERANCE = PERTURBED ? 1.0e-3 : 1.0e-7
const PROJECTION_ITERATIONS = 5000

"""Relative tolerance of a `--compare` per tier (the absolute tolerance is
set per metric). Tier 1 is a set of single projections, deterministic up to
the projection tolerance; tiers 2 and 3 integrate in time, tier 3 through
the impact singularity."""
const TIER_RTOL = Dict(1 => 1.0e-3, 2 => 2.0e-2, 3 => 0.25)

"""One measured quantity. `goal` is `:min` (an error: smaller is better),
`:max` (an observed order: larger is better) or `:info` (reported only).
A value is worse than its baseline beyond `max(rtol |baseline|, atol)`."""
struct Metric
    tier::Int
    group::String
    case::String
    n::Int
    method::Symbol
    name::String
    value::Float64
    goal::Symbol
    rtol::Float64
    atol::Float64
end

metric_key(m::Metric) = join((m.group, m.case, m.n == 0 ? "-" : "n$(m.n)", m.method, m.name), "/")

mutable struct Suite
    tiers::Vector{Int}
    methods::Vector{Method}
    resolutions::Vector{Int}
    metrics::Vector{Metric}
    failures::Vector{Pair{String, String}}
end

Suite(tiers, methods, resolutions) = Suite(tiers, methods, resolutions, Metric[], Pair{String, String}[])

function record!(
        suite::Suite, tier, group, case, n, m::Method, name, value;
        goal = :min, rtol = TIER_RTOL[tier], atol = 0.0,
    )
    # `/` separates the parts of the TOML keys.
    any(part -> occursin('/', part), (group, case)) && error("'/' in the test or case name $group/$case")
    push!(suite.metrics, Metric(tier, group, case, n, m.name, name, Float64(value), goal, rtol, atol))
    return value
end

function fail!(suite::Suite, tier, group, case, n, m::Method, message)
    key = join((group, case, n == 0 ? "-" : "n$n", m.name), "/")
    println("FAILED ", key, ": ", message)
    flush(stdout)
    push!(suite.failures, "tier$tier/" * key => message)
    return nothing
end

"""Run `f()` for one case and method; an exception or a non-converged
projection is recorded as a failure (and fails `--compare`)."""
function guarded(f, suite::Suite, tier, group, case, n, m::Method)
    started = time()
    try
        f()
    catch err
        err isa InterruptException && rethrow()
        fail!(suite, tier, group, case, n, m, first(sprint(showerror, err), 400))
    end
    @printf("  %-6s %-32s n=%-4d %-16s %7.1f s\n", group, case, n, m.name, time() - started)
    flush(stdout)
    return nothing
end

checked(stats) = stats.converged ? stats : error("projection not converged: $stats")

"""Observed orders of the `:min` metrics measured on a resolution ladder
(consecutive doublings), recorded at the finer resolution as
`order(<name>)`. Values within `100 atol` of zero (the round-off or solver
level of an exact case) or below `floor` have no order. Orders are reported
only (`:info`): the values themselves are compared, and an order also rises
when its coarse value gets worse."""
function record_orders!(suite::Suite, tier; floor = 1.0e-12)
    series = Dict{Tuple{String, String, Symbol, String}, Vector{Metric}}()
    for m in suite.metrics
        m.tier == tier && m.goal === :min && m.n > 0 || continue
        push!(get!(series, (m.group, m.case, m.method, m.name), Metric[]), m)
    end
    for ((group, case, method_name, name), entries) in sort!(collect(series); by = first)
        sort!(entries; by = e -> e.n)
        for (coarse, fine) in zip(entries[1:(end - 1)], entries[2:end])
            fine.n == 2coarse.n || continue
            threshold = max(floor, 100 * max(coarse.atol, fine.atol))
            coarse.value > threshold && fine.value > threshold || continue
            order = log2(coarse.value / fine.value)
            push!(
                suite.metrics, Metric(
                    tier, group, case, fine.n, method_name, "order($name)", order, :info, 0.0, 0.02,
                ),
            )
        end
    end
    return suite
end

# ------------------------------------------------------------------ output

"""The metrics as TOML tables `tier<k>` of `key => value`, added to `data`."""
function metric_data!(data, metrics)
    for m in metrics
        get!(data, "tier$(m.tier)", Dict{String, Any}())[metric_key(m)] = m.value
    end
    return data
end

function write_toml(path, suite::Suite)
    data = metric_data!(Dict{String, Any}("meta" => run_metadata()), suite.metrics)
    isempty(suite.failures) || (data["failures"] = Dict(first(f) => last(f) for f in suite.failures))
    open(path, "w") do io
        println(io, "# Two-velocity verification suite (benchmark/two_velocity/run.jl).")
        println(io, "# Keys: <test>/<case>/<resolution>/<method>/<metric>.")
        TOML.print(io, data; sorted = true)
    end
    return path
end

function run_metadata()
    revision = try
        readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    catch
        "unknown"
    end
    return Dict{String, Any}(
        "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "revision" => revision, "perturbed" => PERTURBED,
    )
end

format_value(x) = isnan(x) ? "NaN" : iszero(x) ? "0" : abs(x) >= 1.0e4 || abs(x) < 1.0e-3 ?
    @sprintf("%.2e", x) : @sprintf("%.3g", x)
format_order(x) = @sprintf("%.2f", x)

const METRIC_DESCRIPTIONS = Dict{String, String}()

"""Marks of summary-table cells whose value is exact, or limited, by
construction: `(group, case pattern, metric pattern, method, mark)`, the
mark explained by `MARK_TEXT[mark]` under the table."""
const TABLE_MARKS = Tuple{String, Regex, Regex, Symbol, String}[]
const MARK_TEXT = Dict{String, String}()

function cell_marks(group, case, name, method_name)
    return unique(
        mark for (g, c, n, m, mark) in TABLE_MARKS if g == group && m === method_name && occursin(c, case) && occursin(n, name)
    )
end

"""Detailed Markdown tables of one tier: one table per test, case and
metric, rows by resolution, columns by method; the observed order of a
`:min` metric follows its value in parentheses."""
function markdown(io, suite::Suite, tier)
    metrics = filter(m -> m.tier == tier, suite.metrics)
    isempty(metrics) && return
    methods = [m for m in METHODS if any(x -> x.method === m.name, metrics)]
    lookup = Dict(metric_key(m) => m for m in metrics)
    groups = unique((m.group, m.case, m.name) for m in metrics if !startswith(m.name, "order("))
    current = ""
    for (group, case, name) in groups
        if group != current
            current = group
            println(io, "### $group\n")
        end
        description = get(METRIC_DESCRIPTIONS, "$group/$name", "")
        println(io, "**$case**, `$name`", isempty(description) ? "" : " ($description)", "\n")
        resolutions = sort!(unique(m.n for m in metrics if (m.group, m.case, m.name) == (group, case, name)))
        println(io, "| n | ", join((m.label for m in methods), " | "), " |")
        println(io, "|---|", repeat("---|", length(methods)))
        for n in resolutions
            cells = map(methods) do m
                key = join((group, case, n == 0 ? "-" : "n$n", m.name, name), "/")
                haskey(lookup, key) || return ""
                text = format_value(lookup[key].value)
                order_key = join((group, case, "n$n", m.name, "order($name)"), "/")
                haskey(lookup, order_key) && (text *= " (" * format_order(lookup[order_key].value) * ")")
                text
            end
            println(io, "| ", n == 0 ? "–" : string(n), " | ", join(cells, " | "), " |")
        end
        println(io)
    end
    return
end

"""Compact table of one tier: a row per test, case and metric; per method
the values over the resolutions, the observed order of the finest pair in
parentheses and the marks of `TABLE_MARKS` in brackets; then the metric and
mark legends."""
function summary(io, metrics::Vector{Metric}, tier)
    metrics = filter(m -> m.tier == tier, metrics)
    isempty(metrics) && return
    methods = [m for m in METHODS if any(x -> x.method === m.name, metrics)]
    lookup = Dict(metric_key(m) => m for m in metrics)
    rows = unique((m.group, m.case, m.name) for m in metrics if !startswith(m.name, "order("))
    used = Set{String}()
    println(io, "| test | case | metric | n | ", join((m.label for m in methods), " | "), " |")
    println(io, "|---|---|---|---|", repeat("---|", length(methods)))
    for (group, case, name) in rows
        resolutions = sort!(unique(m.n for m in metrics if (m.group, m.case, m.name) == (group, case, name)))
        cells = map(methods) do m
            values = map(resolutions) do n
                key = join((group, case, n == 0 ? "-" : "n$n", m.name, name), "/")
                haskey(lookup, key) ? format_value(lookup[key].value) : "–"
            end
            all(==("–"), values) && return ""
            text = join(values, " / ")
            order_key = join((group, case, "n$(last(resolutions))", m.name, "order($name)"), "/")
            haskey(lookup, order_key) && (text *= " (" * format_order(lookup[order_key].value) * ")")
            marks = cell_marks(group, case, name, m.name)
            union!(used, marks)
            isempty(marks) || (text *= " [" * join(marks, ",") * "]")
            text
        end
        sizes = join((n == 0 ? "–" : string(n) for n in resolutions), " / ")
        println(io, "| $group | $case | `$name` | $sizes | ", join(cells, " | "), " |")
    end
    println(io)
    for (group, name) in unique((m.group, m.name) for m in metrics if !startswith(m.name, "order("))
        description = get(METRIC_DESCRIPTIONS, "$group/$name", "")
        isempty(description) || println(io, "- $group `$name`: $description")
    end
    for mark in sort!(collect(used))
        println(io, "- `[$mark]` ", MARK_TEXT[mark])
    end
    println(io)
    return
end

function write_markdown(path, suite::Suite)
    open(path, "w") do io
        meta = run_metadata()
        println(io, "# Two-velocity verification results\n")
        println(
            io, "Julia ", meta["julia_version"], ", ", meta["threads"], " thread(s), revision ",
            meta["revision"], meta["perturbed"] ? " (PERTURBED)" : "", ".\n",
        )
        for tier in sort(unique(m.tier for m in suite.metrics))
            println(io, "## Tier $tier\n")
            summary(io, suite.metrics, tier)
            markdown(io, suite, tier)
        end
        if !isempty(suite.failures)
            println(io, "## Failures\n")
            for (key, message) in suite.failures
                println(io, "- `", key, "`: ", replace(message, '\n' => ' '))
            end
        end
    end
    return path
end

"""Replace the generated table of each tier of the baseline (or results)
file `source` in the docs page `path`, between the lines
`<!-- tier<k> table: begin -->` and `<!-- tier<k> table: end -->`."""
function update_docs!(path, source)
    text = read(path, String)
    metrics = baseline_metrics(source)
    meta = get(TOML.parsefile(source), "meta", Dict{String, Any}())
    for tier in sort(unique(m.tier for m in metrics))
        begin_marker, end_marker = "<!-- tier$tier table: begin -->", "<!-- tier$tier table: end -->"
        first_index = findfirst(begin_marker, text)
        last_index = findfirst(end_marker, text)
        (isnothing(first_index) || isnothing(last_index)) && error("$path has no tier-$tier table markers")
        table = sprint() do io
            println(io, begin_marker)
            println(
                io, "\n_Generated from `benchmark/two_velocity/baseline.toml` by `run.jl --update-docs` (Julia ",
                get(meta, "julia_version", "?"), ", revision ", get(meta, "revision", "?"), ")._\n",
            )
            summary(io, metrics, tier)
        end
        text = text[1:(first(first_index) - 1)] * table * text[first(last_index):end]
    end
    write(path, text)
    return path
end

# ------------------------------------------------------------------ baseline

"""Values of a results or baseline TOML file by tier and key."""
function load_values(path)
    data = TOML.parsefile(path)
    return Dict(
        parse(Int, tier[5:end]) => Dict{String, Float64}(k => Float64(v) for (k, v) in table)
            for (tier, table) in data if startswith(tier, "tier")
    )
end

method_of_key(key) = Symbol(split(key, '/')[4])

"""Order of the tests in tables generated from a TOML file."""
const GROUP_ORDER = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "hydrostatic", "static drop", "capillary wave",
    "Kelvin-Helmholtz", "translation", "label symmetry", "impact",
]

"""The metrics of a baseline or results file, ordered by tier, test, case,
metric and resolution (goal and tolerances are not stored: `:info`, 0)."""
function baseline_metrics(path)
    metrics = Metric[]
    for (tier, table) in load_values(path), (key, value) in table
        group, case, size, method_name, name = split(key, '/'; limit = 5)
        n = size == "-" ? 0 : parse(Int, size[2:end])
        push!(metrics, Metric(tier, group, case, n, Symbol(method_name), name, value, :info, 0.0, 0.0))
    end
    rank(group) = something(findfirst(==(group), GROUP_ORDER), length(GROUP_ORDER) + 1)
    return sort!(metrics; by = m -> (m.tier, rank(m.group), m.group, m.case, m.name, m.n, string(m.method)))
end

"""Replace the entries of the run's tiers and methods in the baseline at
`path` (created if absent) by the run's values."""
function record_baseline!(path, suite::Suite)
    previous = isfile(path) ? load_values(path) : Dict{Int, Dict{String, Float64}}()
    names = Set(m.name for m in suite.methods)
    data = Dict{String, Any}("meta" => run_metadata())
    for (tier, table) in previous, (key, value) in table
        tier in suite.tiers && method_of_key(key) in names && continue
        get!(data, "tier$tier", Dict{String, Any}())[key] = value
    end
    metric_data!(data, suite.metrics)
    open(path, "w") do io
        println(io, "# Baseline of the two-velocity verification suite; re-record with")
        println(io, "# `run.jl --tier=... --record` only for an intentional change (see README).")
        TOML.print(io, data; sorted = true)
    end
    return path
end

"""Compare the run with the baseline: per metric better / same / worse
beyond its tolerance. Returns `true` when nothing is worse, no baseline
metric of a case that ran is missing (a derived order may vanish when its
values reach the round-off level; cases not run, e.g. other resolutions, are
not compared) and no case failed."""
function compare(suite::Suite, path; io = stdout)
    baseline = load_values(path)
    counts = Dict(:better => 0, :same => 0, :worse => 0, :new => 0, :changed => 0)
    rows = Tuple{Symbol, Metric, Float64}[]
    seen = Set{String}()
    for m in suite.metrics
        key = metric_key(m)
        push!(seen, "tier$(m.tier)/" * key)
        table = get(baseline, m.tier, Dict{String, Float64}())
        if !haskey(table, key)
            counts[:new] += 1
            push!(rows, (:new, m, NaN))
            continue
        end
        base = table[key]
        verdict = judge(m, base)
        counts[verdict] += 1
        verdict === :same || push!(rows, (verdict, m, base))
    end
    case_prefix(key) = rsplit(key, '/'; limit = 2)[1]
    ran = Set(case_prefix("tier$(m.tier)/" * metric_key(m)) for m in suite.metrics)
    union!(ran, first.(suite.failures))
    missing_keys = String[]
    for (tier, table) in baseline, key in keys(table)
        full = "tier$tier/" * key
        case_prefix(full) in ran && !occursin("/order(", key) && !(full in seen) && push!(missing_keys, full)
    end
    println(io, "\nComparison with ", path, " (tolerance max(rtol |baseline|, atol); rtol per tier ", TIER_RTOL, ")")
    for (verdict, m, base) in sort(rows; by = r -> (r[1] !== :worse, r[1], metric_key(r[2])))
        @printf(
            io, "  %-7s tier%d/%s: %s (baseline %s, tolerance %s)\n", verdict, m.tier, metric_key(m),
            format_value(m.value), format_value(base), format_value(max(m.rtol * abs(base), m.atol)),
        )
    end
    for key in sort!(missing_keys)
        println(io, "  missing ", key)
    end
    for (key, message) in suite.failures
        println(io, "  failed  ", key, ": ", message)
    end
    @printf(
        io, "Summary: %d better, %d same, %d worse, %d changed (info), %d new, %d missing, %d failed\n",
        counts[:better], counts[:same], counts[:worse], counts[:changed], counts[:new],
        length(missing_keys), length(suite.failures),
    )
    return counts[:worse] == 0 && isempty(missing_keys) && isempty(suite.failures)
end

function judge(m::Metric, base)
    tolerance = max(m.rtol * abs(base), m.atol)
    value = m.value
    isnan(value) && isnan(base) && return :same
    if m.goal === :info
        return isnan(value) != isnan(base) || abs(value - base) > tolerance ? :changed : :same
    end
    isnan(value) && return :worse
    isnan(base) && return :better
    difference = m.goal === :min ? value - base : base - value
    difference > tolerance && return :worse
    difference < -tolerance && return :better
    return :same
end
