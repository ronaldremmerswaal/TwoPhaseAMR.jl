include("cases.jl")

baseline = TOML.parsefile(joinpath(@__DIR__, "baseline.toml"))
tol = 1.0e-10
mode = VERSION == VersionNumber(baseline["julia_version"]) && Threads.nthreads() == baseline["threads"] ? "bitwise" : "tolerance"
for arg in ARGS
    if startswith(arg, "--tol=")
        global tol = parse(Float64, split(arg, "="; limit = 2)[2])
    elseif startswith(arg, "--mode=")
        global mode = split(arg, "="; limit = 2)[2]
    else
        error("Unknown argument: $arg")
    end
end
mode in ("bitwise", "tolerance") || error("mode must be bitwise or tolerance")
isfinite(tol) && tol >= 0 || error("tolerance must be finite and nonnegative")
println("Comparison mode: ", mode, " (rtol=", tol, ")")

function close_value(a, b, tol)
    if a isa Number && b isa Number
        return isapprox(a, b; rtol = tol, atol = tol)
    elseif a isa AbstractVector && b isa AbstractVector
        return length(a) == length(b) && all(close_value(x, y, tol) for (x, y) in zip(a, b))
    end
    return a == b
end

function check_all()
    ok = true
    expected_names = Set(keys(baseline["cases"]))
    actual_names = Set(first(case) for case in CASES)
    if expected_names != actual_names
        println("case names: FAIL (expected ", expected_names, ", got ", actual_names, ")")
        return false
    end
    for (name, run) in CASES
        actual = capture(run(); perturb = get(ENV, "ORACLE_PERTURB", "0") == "1" && name == "taylor_green")
        expected = baseline["cases"][name]
        meta_ok = actual["steps"] == expected["steps"] && close_value(actual["time"], expected["time"], tol)
        diag_ok = keys(actual["diagnostics"]) == keys(expected["diagnostics"]) && all(
            close_value(value, expected["diagnostics"][key], tol) for (key, value) in actual["diagnostics"]
        )
        println(name, "/metadata+diagnostics: ", meta_ok && diag_ok ? "PASS" : "FAIL")
        ok &= meta_ok && diag_ok
        field_keys_ok = keys(actual["fields"]) == keys(expected["fields"])
        ok &= field_keys_ok
        field_keys_ok || println(name, "/field names: FAIL")
        for field in sort!(collect(union(keys(actual["fields"]), keys(expected["fields"]))))
            a = get(actual["fields"], field, nothing)
            b = get(expected["fields"], field, nothing)
            match = !isnothing(a) && !isnothing(b) && a["count"] == b["count"] && a["finite_count"] == b["finite_count"] &&
                (
                mode == "bitwise" ? a["sha256"] == b["sha256"] :
                    close_value(a["l2"], b["l2"], tol) && close_value(a["max"], b["max"], tol)
            )
            println(name, "/", field, ": ", match ? "PASS" : "FAIL")
            ok &= match
        end
    end
    return ok
end
ok = check_all()
println(ok ? "PASS" : "FAIL")
exit(ok ? 0 : 1)
