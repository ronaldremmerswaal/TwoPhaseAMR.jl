#!/usr/bin/env julia
module BlockInteriorPlot

include(joinpath(@__DIR__,"..","..","Support.jl"))
using .ScalingSupport
using CairoMakie

function main(args=ARGS)
    if any(arg -> arg in ("-h","--help"),args)
        println("usage: plot.jl COHORT_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: plot.jl COHORT_DIRECTORY")
    directory = abspath(args[1])
    data = readjson(joinpath(directory,"summary.json"))
    cases = data["config"]["cases"]
    rows = data["rows"]
    fig = Figure(size=(1120,280length(cases)))
    for (index,(n,b)) in enumerate(cases)
        selected = sort!([row for row in rows if row["n"] == n && row["block"] == b];
                         by=row -> row["workers"])
        length(selected) == 3 || error("missing worker rows for $n:$b")
        workers = [row["workers"] for row in selected]
        title = b == n ? "$(n)², one block" : "$(n)², $(b)×$(b)-cell blocks"
        ax_cost = Axis(fig[index,1],title=title,
            xlabel="Julia workers",ylabel="ns per scalar cell per complete step",
            xticks=(workers,string.(workers)))
        baseline = [row["baseline_median_ns_per_unknown_step"] for row in selected]
        current = [row["current_median_ns_per_unknown_step"] for row in selected]
        lines!(ax_cost,workers,baseline;color=:steelblue,linewidth=2.5,label="committed block interior")
        scatter!(ax_cost,workers,baseline;color=:steelblue,markersize=11)
        lines!(ax_cost,workers,current;color=:darkorange,linewidth=2.5,label="checked narrow IDs")
        scatter!(ax_cost,workers,current;color=:darkorange,markersize=11)
        index == 1 && axislegend(ax_cost;position=:rt)
        displayed_upper = maximum(begin
            interval = row["current_efficiency_ci95"]
            self_reference = row["workers"] == 1 &&
                             row["strongest_serial_variant"] == "current"
            interval === nothing || self_reference ?
                row["current_strongest_serial_efficiency"] :
                max(row["current_strongest_serial_efficiency"],interval[2])
        end for row in selected)
        ax_eff = Axis(fig[index,2],title="Strongest-serial efficiency",
            xlabel="Julia workers",ylabel="efficiency",xticks=(workers,string.(workers)),
            limits=(nothing,(0,max(1.15,1.08displayed_upper))))
        hlines!(ax_eff,[0.5];color=:gray,linestyle=:dash,linewidth=1.5)
        efficiency = [row["current_strongest_serial_efficiency"] for row in selected]
        lines!(ax_eff,workers,efficiency;color=:darkorange,linewidth=2.5)
        scatter!(ax_eff,workers,efficiency;color=:darkorange,markersize=11)
        for (x,row) in zip(workers,selected)
            interval = row["current_efficiency_ci95"]
            self_reference = row["workers"] == 1 &&
                             row["strongest_serial_variant"] == "current"
            if interval !== nothing && !self_reference
                y = row["current_strongest_serial_efficiency"]
                errorbars!(ax_eff,[x],[y],[max(0,y-interval[1])],
                           [max(0,interval[2]-y)];color=:darkorange,whiskerwidth=10)
            end
        end
    end
    save(joinpath(directory,"summary.svg"),fig)
    save(joinpath(directory,"summary.png"),fig)
    println("saved summary.svg and summary.png to $directory")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
