# Summarize `--trace-compile=FILE` without confusing signature mentions with
# method ownership. This is a text-level lower bound, not every inferred method.
length(ARGS) == 1 || error("usage: trace_summary.jl TRACE_FILE")

function trace_summary(path)
    owners = Dict{String, Int}()
    targets = ("NavierStokesSystem", "NavierStokesWorkspace", "FlowSimulation")
    type_forms = Dict(t => Set{String}() for t in targets)
    mentions = Dict(t => 0 for t in targets)
    total = kw = kw_amr = recompiled = amr = 0
    for line in eachline(path)
        startswith(line, "precompile(") || continue
        total += 1
        m = match(r"^precompile\(Tuple\{(?:typeof\(|Type\{)?([A-Za-z_][A-Za-z_0-9]*)", line)
        owner = isnothing(m) ? "?" : m.captures[1]
        owners[owner] = get(owners, owner, 0) + 1
        is_amr = occursin("TwoPhaseAMR.", line)
        amr += is_amr
        is_kw = occursin("typeof(Core.kwcall)", line)
        kw += is_kw
        kw_amr += is_kw && is_amr
        recompiled += occursin("# recompile", line)
        for target in targets
            prefix = "TwoPhaseAMR." * target * "{"
            occursin(target, line) && (mentions[target] += 1)
            pos = firstindex(line)
            while true
                found = findnext(prefix, line, pos)
                isnothing(found) && break
                depth = 1
                cursor = nextind(line, last(found))
                while cursor <= lastindex(line) && depth > 0
                    char = line[cursor]
                    depth += (char == '{') - (char == '}')
                    cursor = nextind(line, cursor)
                end
                if depth == 0
                    push!(type_forms[target], line[first(found):prevind(line, cursor)])
                end
                pos = cursor
                pos > lastindex(line) && break
            end
        end
    end
    println(
        "Traced lines: ", total, "; mentioning TwoPhaseAMR: ", amr,
        "; Core.kwcall: ", kw, " (", kw_amr, " mention TwoPhaseAMR); recompile-tagged: ", recompiled
    )
    println("First signature-token prefixes (a textual ownership heuristic):")
    for (owner, count) in sort!(collect(owners); by = last, rev = true)
        println("  ", rpad(owner, 25), " ", count)
    end
    for target in targets
        println(
            target, ": ", mentions[target], " matching lines, ",
            length(type_forms[target]), " distinct syntactic type forms"
        )
    end
    return
end
trace_summary(ARGS[1])
