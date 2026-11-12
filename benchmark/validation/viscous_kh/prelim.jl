# Preliminary comparison for one case: |η̂|(t) against the linear reference, fitted
# growth rates per run and the 2×2 field snapshot.
#   julia --project=. prelim.jl <case> <n,n,…> <n_unresolved> <n_resolved>
include(joinpath(@__DIR__, "plots.jl"))

function prelim(name, resolutions, n_coarse, n_fine)
    out = joinpath(@__DIR__, "prelim")
    mkpath(out)
    c = CASES[name]
    r = read_table(joinpath(RESULTS, "reference_$(name).csv"))
    ref = (; t = r["t"], η = complex.(r["eta_re"], r["eta_im"]))
    a, b = window(c)
    s_ref, c_ref = fit(ref.t, ref.η, a, b)
    fig = Figure(; size = (900, 560), fontsize = 14)
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t ΔU/λ", ylabel = "|η̂| / λ",
        title = "$(CASE_TITLES[name]): interface mode amplitude")
    lines!(ax, ref.t, abs.(ref.η); color = :black, linewidth = 3, label = "linear IVP reference")
    lines!(ax, ref.t, c.amplitude .* exp.(imag(vortex_sheet_frequency(c)) .* ref.t); color = :gray60,
        linestyle = :dot, label = "vortex sheet (Lamb), δ → 0")
    table = IOBuffer()
    @printf(table, "| model | λ/h | δ0/h | growth rate | rel. error | |η̂(T)|/|η̂_ref(T)| |\n|---|---:|---:|---:|---:|---:|\n")
    @printf(table, "| reference (IVP) | – | – | %.4f | – | 1 |\n", s_ref)
    styles = (:dot, :dashdot, :dash, :solid, :solid, :solid)
    for model in ("one", "two"), (q, n) in enumerate(resolutions)
        path = joinpath(RUNS, "$(name)_$(model)_n$(n).csv")
        isfile(path) || continue
        run = read_series(path)
        s, _ = fit(run.t, run.η, a, b)
        lines!(ax, run.t, abs.(run.η); color = COLORS[model], linestyle = styles[q],
            linewidth = 1 + q / 2, label = @sprintf("%s, λ/h = %d (δ₀/h = %.2g)", LABELS[model], n, minimum(initial_thicknesses(c)) * n))
        @printf(table, "| %s | %d | %.3g | %.4f | %+.3f | %.3f |\n", model, n, minimum(initial_thicknesses(c)) * n,
            s, (s - s_ref) / s_ref, abs(run.η[end]) / abs(interpolate_reference(ref, run.t[end])))
    end
    vspan!(ax, a, b; color = (:gray, 0.1))
    Legend(fig[1, 2], ax; framevisible = false, labelsize = 12)
    save(joinpath(out, "amplitude_$(name).png"), fig; px_per_unit = 1)
    field_figure(name, n_coarse, n_fine; out)
    write(joinpath(out, "growth_$(name).md"), String(take!(table)))
    print(read(joinpath(out, "growth_$(name).md"), String))
    return nothing
end

prelim(ARGS[1], parse.(Int, split(ARGS[2], ',')), parse(Int, ARGS[3]), parse(Int, ARGS[4]))
