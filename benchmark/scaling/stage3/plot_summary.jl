# Plot accepted Stage 3 general-upwind absolute cost and efficiency.
include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport
using CairoMakie

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia plot_summary.jl MATRIX_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: julia plot_summary.jl MATRIX_DIRECTORY")
    directory = args[1]
    summary = readjson(joinpath(directory, "summary.json"))
    rows = summary["rows"]
    summary["raw_samples"] == 180 && length(rows) == 36 ||
        error("expected validated five-repeat primary matrix")
    index = Dict((r["n"], r["block"], r["workers"], r["variant"]) => r
                 for r in rows)
    sizes = (512, 1024, 2048)
    colors = Dict(1 => :gray30, 4 => :steelblue, 8 => :darkorange)
    styles = Dict("frozen" => :dash, "current" => :solid)
    figure = Figure(size=(1150, 800))
    for (column, (layout, block_for)) in enumerate(
            (("one block", identity), ("8×8-cell blocks", n->8)))
        cost = Axis(figure[1, column], title=layout,
                    ylabel="ns / unknown / complete step", yscale=log10,
                    xticks=(collect(sizes), collect(string.(sizes))))
        efficiency = Axis(figure[2, column], xlabel="Grid side n",
                          ylabel="parallel efficiency, strongest serial",
                          xticks=(collect(sizes), collect(string.(sizes))))
        for variant in ("frozen", "current"), p in (1, 4, 8)
            selected = [index[(n, block_for(n), p, variant)] for n in sizes]
            y = [r["median_ns_per_unknown_step"] for r in selected]
            label = "$variant, $p worker$(p == 1 ? "" : "s")"
            lines!(cost, collect(sizes), y; color=colors[p],
                   linestyle=styles[variant], linewidth=2, label)
            scatter!(cost, collect(sizes), y; color=colors[p])
            p == 1 && continue
            e = [r["strongest_serial_efficiency"] for r in selected]
            low = [r["strongest_serial_efficiency_ci95"][1] for r in selected]
            high = [r["strongest_serial_efficiency_ci95"][2] for r in selected]
            band!(efficiency, collect(sizes), low, high;
                  color=(colors[p], 0.12))
            lines!(efficiency, collect(sizes), e; color=colors[p],
                   linestyle=styles[variant], linewidth=2)
            scatter!(efficiency, collect(sizes), e; color=colors[p])
        end
        hlines!(efficiency, [0.5]; color=:gray, linestyle=:dot)
        ylims!(efficiency, 0, 1.18)
        column == 1 && axislegend(cost; position=:rt)
    end
    Label(figure[0, :], "General upwind: complete-step cost and five-process scaling")
    save(joinpath(directory, "main-absolute-efficiency.svg"), figure)
    save(joinpath(directory, "main-absolute-efficiency.png"), figure)
    println("wrote main-absolute-efficiency.svg/.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
