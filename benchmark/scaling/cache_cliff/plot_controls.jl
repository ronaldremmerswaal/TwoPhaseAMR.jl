"""CairoMakie figures from validated counter and padding diagnostic summaries."""
module CacheCliffControlsPlots

include(joinpath(@__DIR__, "..", "Support.jl"))
using .ScalingSupport: readjson
using CairoMakie

event(row, name) = only([item["median_events_per_unknown_step"] for item in
                         row["events"] if item["name"] == name])

function save_figure(figure, directory, stem)
    save(joinpath(directory, "$stem.svg"), figure)
    save(joinpath(directory, "$stem.png"), figure; px_per_unit=2)
end

function pmu_plots(directory)
    rows = readjson(joinpath(directory, "summary.json"))["rows"]
    sizes = [1024, 1280, 1408, 2048]
    demand = [only([row for row in rows if row["nx"] == n && row["workers"] == 16 &&
                    row["counter_mode"] == 1]) for n in sizes]
    prefetch = [only([row for row in rows if row["nx"] == n && row["workers"] == 16 &&
                      row["counter_mode"] == 2]) for n in sizes]
    figure = Figure(size=(880, 480))
    axis = Axis(figure[1,1], yscale=log10,
        title="p16 spread: memory-source events rise across the cache-capacity knee",
        xlabel="Grid width N (N×N)",
        ylabel="Zen 3 source-of-fill events / unknown / complete timestep",
        xticks=(sizes, string.(sizes)))
    for (series, name, label, color) in (
        (demand, "zen3_demand_mem_io_local", "Demand fills from local memory/IO", "#d24b28"),
        (prefetch, "zen3_hwprefetch_mem_io_local", "Hardware-prefetch fills from local memory/IO", "#2563eb"),
        (demand, "zen3_demand_internal_cache", "Demand fills from internal cache", "#a66a1e"),
        (prefetch, "zen3_hwprefetch_internal_cache", "Hardware-prefetch fills from internal cache", "#169087"))
        values = Float64[event(row, name) for row in series]
        lines!(axis, sizes, values; color, linewidth=2, label)
        scatter!(axis, sizes, values; color)
    end
    axislegend(axis)
    save_figure(figure, directory, "pmu-origin-p16")

    figure = Figure(size=(1050, 470))
    for (column, n) in enumerate((1024, 1280))
        axis = Axis(figure[1,column], title="$(n)×$(n), p8",
                    ylabel="HW-prefetch local-memory fills / unknown / timestep",
                    xticks=([1,2], ["compact", "spread"]))
        values = [event(only([row for row in rows if row["nx"] == n &&
            row["workers"] == 8 && row["placement"] == place &&
            row["counter_mode"] == 2]), "zen3_hwprefetch_mem_io_local")
            for place in ("compact", "spread")]
        barplot!(axis, [1,2], values; color=["#d24b28", "#2563eb"])
    end
    save_figure(figure, directory, "pmu-origin-placement-p8")
end

function padding_plot(directory)
    rows = readjson(joinpath(directory, "summary.json"))["rows"]
    figure = Figure(size=(1050, 470))
    for (column, n) in enumerate((1024, 2048))
        sequence = sort([row for row in rows if row["nx"] == n];
                        by=row->row["padding"])
        pads = Float64[row["padding"] for row in sequence]
        values = Float64[row["median_ns_per_unknown_step"] for row in sequence]
        bounds = [row["time_ci95"] for row in sequence]
        axis = Axis(figure[1,column], title="$(n)×$(n), p16 spread",
                    xlabel="Extra Float64 elements in leading dimension",
                    ylabel=column == 1 ? "ns per unknown per complete timestep" : "",
                    xticks=(pads, string.(Int.(pads))))
        errorbars!(axis, pads, values, values .- first.(bounds),
                   last.(bounds) .- values; color="#2563eb", whiskerwidth=8)
        lines!(axis, pads, values; color="#2563eb", linewidth=2)
        scatter!(axis, pads, values; color="#2563eb")
    end
    save_figure(figure, directory, "padding-p16")
end

function main(args=ARGS)
    length(args) == 1 || error("usage: julia plot_controls.jl RESULT_ROOT")
    root = abspath(args[1])
    pmu_plots(joinpath(root, "pmu"))
    padding_plot(joinpath(root, "padding"))
    println("wrote PMU and padding plots")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && CacheCliffControlsPlots.main()
