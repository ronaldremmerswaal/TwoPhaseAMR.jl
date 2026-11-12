# Preliminary comparison for one case: |η̂|(t) against the references, fitted growth
# rates per run and the 2×2 field snapshot.
#   julia --project=. prelim.jl <case> <n_unresolved> <n_resolved>
include(joinpath(@__DIR__, "plots.jl"))

function prelim(name, n_coarse, n_fine)
    out = joinpath(@__DIR__, "prelim")
    mkpath(out)
    c = CASES[name]
    r = read_table(joinpath(RESULTS, "reference_$(name).csv"))
    ref = (; t = r["t"], η = complex.(r["eta_re"], r["eta_im"]))
    a, b = window(c)
    s_ref, _ = fit(ref.t, ref.η, a, b)
    table = IOBuffer()
    @printf(table, "| model | λ/h | δ0/h | growth rate | rel. error | |η̂(T)|/|η̂_ref(T)| |\n|---|---:|---:|---:|---:|---:|\n")
    @printf(table, "| reference (IVP) | – | – | %.4f | – | 1 |\n", s_ref)
    for model in ("one", "two"), n in RESOLUTIONS
        path = joinpath(RUNS, "$(name)_$(model)_n$(n).csv")
        isfile(path) || continue
        run = read_series(path)
        s, _ = fit(run.t, run.η, a, b)
        @printf(
            table, "| %s | %d | %.3g | %.4f | %+.3f | %.3f |\n", model, n, minimum(initial_thicknesses(c)) * n,
            s, (s - s_ref) / s_ref, abs(run.η[end]) / abs(interpolate_reference(ref, run.t[end]))
        )
    end
    amplitude_figure(NamedTuple[]; cases = [name], out, filename = "amplitude_$(name).png")
    field_figure(name, n_coarse, n_fine; out)
    write(joinpath(out, "growth_$(name).md"), String(take!(table)))
    print(read(joinpath(out, "growth_$(name).md"), String))
    return nothing
end

prelim(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
