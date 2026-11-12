# Bulk momentum flux: MonotonizedCentral (MC) vs the default LaxWendroff (LW), both at
# the default ctu_threshold 0.5, both models, all cases.
#   julia --project=. mc_compare.jl run      # → results/mc_runs/*_MC.csv
#   julia --project=. mc_compare.jl report   # → results/mc_compare.md, figures/mc_vs_lw.png
using Printf
const MC_NS = (32, 64, 128, 256)
const MC_RUNS = joinpath(@__DIR__, "results", "mc_runs")

if ARGS[1] == "run"
    include(joinpath(@__DIR__, "kh_run.jl"))
    for name in ("r1_thin", "r10_thin", "r1_thick", "r10_thick"), n in MC_NS, model in (OneVelocityModel(), TwoVelocityModel())
        run_case(CASES[name], model, n; tag = "_MC", scheme = MonotonizedCentral(), runs = MC_RUNS, skip_existing = true)
    end
else
    include(joinpath(@__DIR__, "plots.jl"))
    function mc_report()
        io = IOBuffer()
        println(io, "| case | model | λ/h | \\|η̂(T)\\|/ref LW | \\|η̂(T)\\|/ref MC | growth err LW | growth err MC | MC flags |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---|")
        fig = Figure(; size = (900, 520), fontsize = 13)
        ax = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "λ/h", ylabel = "|η̂(T)| / |η̂_ref(T)|")
        hlines!(ax, [1.0]; color = :black)
        colors = Dict("r1_thin" => :royalblue3, "r10_thin" => :firebrick3, "r1_thick" => :seagreen, "r10_thick" => :darkorange2)
        for name in CASE_ORDER
            c = CASES[name]
            r = read_table(joinpath(RESULTS, "reference_$(name).csv"))
            ref = (; t = r["t"], η = complex.(r["eta_re"], r["eta_im"]))
            a, b = window(c)
            s_ref, _ = fit(ref.t, ref.η, a, b)
            for model in ("one", "two")
                pts = Dict("LW" => Tuple{Int, Float64}[], "MC" => Tuple{Int, Float64}[])
                for n in MC_NS
                    paths = ("LW" => joinpath(RUNS, "$(name)_$(model)_n$(n).csv"), "MC" => joinpath(MC_RUNS, "$(name)_$(model)_n$(n)_MC.csv"))
                    vals = Dict{String, Tuple{Float64, Float64, String}}()
                    for (s, p) in paths
                        isfile(p) || continue
                        run = read_series(p)
                        done = run.t[end] >= c.final_time - 1.0e-9
                        g, _ = done ? fit(run.t, run.η, a, b) : (NaN, NaN)
                        ratio = abs(run.η[end]) / abs(interpolate_reference(ref, run.t[end]))
                        m = run.meta
                        flags = join(
                            filter(
                                !isempty, [
                                    get(m, "failed", "no") == "yes" ? "FAILED" : "",
                                    !done ? "incomplete" : "",
                                    parse(Int, get(m, "infinite", "0")) > 0 ? "Inf" : "",
                                    haskey(m, "fmin") && (parse(Float64, m["fmin"]) < -1.0e-12 || parse(Float64, m["fmax"]) > 1 + 1.0e-12) ? "f∉[0,1]" : "",
                                    haskey(m, "umax") && max(abs(parse(Float64, m["umin"])), abs(parse(Float64, m["umax"]))) > 1 ? "|u|>ΔU" : "",
                                ]
                            ), ","
                        )
                        vals[s] = (ratio, abs(g - s_ref) / s_ref, flags)
                        push!(pts[s], (n, ratio))
                    end
                    haskey(vals, "MC") || continue
                    lw = get(vals, "LW", (NaN, NaN, ""))
                    mc = vals["MC"]
                    @printf(
                        io, "| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %s |\n", name, model, n, lw[1], mc[1], lw[2], mc[2],
                        isempty(mc[3]) ? "ok" : mc[3]
                    )
                end
                for (s, filled) in (("LW", true), ("MC", false))
                    isempty(pts[s]) && continue
                    scatterlines!(
                        ax, first.(pts[s]), last.(pts[s]); color = colors[name], linestyle = STYLES[model],
                        marker = :circle, markersize = 10, strokewidth = 1.5, strokecolor = colors[name],
                        markercolor = filled ? colors[name] : :white
                    )
                end
            end
        end
        keys = [
            [LineElement(; color = colors[n], linewidth = 3) for n in CASE_ORDER],
            [
                LineElement(; color = :gray30, linestyle = :solid), LineElement(; color = :gray30, linestyle = :dash),
                MarkerElement(; marker = :circle, color = :gray30), MarkerElement(; marker = :circle, color = :white, strokecolor = :gray30, strokewidth = 1.5),
            ],
        ]
        Legend(
            fig[1, 2], keys, [[CASE_TITLES[n] for n in CASE_ORDER], ["one-velocity", "two-velocity", "LaxWendroff (default)", "MC"]],
            ["case", "model / scheme"]; framevisible = false, labelsize = 11
        )
        save(joinpath(FIGURES, "mc_vs_lw.png"), fig; px_per_unit = 1)
        write(joinpath(RESULTS, "mc_compare.md"), String(take!(io)))
        print(read(joinpath(RESULTS, "mc_compare.md"), String))
        return nothing
    end
    mc_report()
end
