# Generate standalone scaling and working-set figures from summary.json.
include(joinpath(@__DIR__, "Support.jl"))
using .ScalingSupport
using CairoMakie

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("usage: julia plot.jl RESULTS_DIRECTORY")
        return
    end
    length(args) == 1 || error("usage: julia plot.jl RESULTS_DIRECTORY")
    destination = args[1]
    rows = readjson(joinpath(destination, "summary.json"))
    groups = Dict{Tuple, Vector{Any}}()
    for row in rows
        key = (row["case"], row["backend"], row["placement"])
        push!(get!(groups, key, Any[]), row)
    end
    for ((case, backend, placement), values) in groups
        by_size = Dict{Tuple, Vector{Any}}()
        for row in values
            push!(get!(by_size, (row["nx"], row["ny"]), Any[]), row)
        end
        figure = Figure(size=(1200, 840))
        axes = [Axis(figure[i, j], xlabel="Requested Julia workers",
                     ylabel=label) for (i, j, label) in
                ((1, 1, "Wall time (s)"), (1, 2, "Speedup T₁/Tₚ"),
                 (2, 1, "Efficiency Sₚ/p"), (2, 2, "Probe item imbalance: max/mean"))]
        for (size, series) in sort!(collect(by_size); by=first)
            sort!(series; by=row->row["workers"])
            workers = [row["workers"] for row in series]
            label = "$(size[1])×$(size[2])"
            for (axis, field) in zip(axes,
                    ("time_median_seconds", "speedup", "efficiency",
                     "worker_probe_imbalance_median"))
                lines!(axis, workers, [row[field] for row in series]; label)
                scatter!(axis, workers, [row[field] for row in series])
            end
        end
        axes[1].yscale = log10
        hlines!(axes[3], [0.5]; color=:gray, linestyle=:dash)
        for axis in axes
            axislegend(axis; position=:rt)
        end
        Label(figure[0, :], "$case / $backend / $placement; fixed trajectory within each size")
        stem = "$case-$backend-$placement-scaling"
        save(joinpath(destination, stem * ".svg"), figure)
        save(joinpath(destination, stem * ".png"), figure)

        working = Figure(size=(1200, 480))
        throughput = Axis(working[1, 1], xlabel="Working set (MiB)",
                          ylabel="Cell steps/s (million)")
        errors = Axis(working[1, 2], xlabel="Wall time (s)",
                      ylabel="Max RMS solution error across runs")
        by_worker = Dict{Int, Vector{Any}}()
        for row in values
            push!(get!(by_worker, row["workers"], Any[]), row)
        end
        for (worker, series) in sort!(collect(by_worker); by=first)
            sort!(series; by=row->row["working_set_bytes"])
            lines!(throughput,
                   [row["working_set_bytes"] / 2^20 for row in series],
                   [row["throughput_cells_per_second"] / 1e6 for row in series];
                   label="p=$worker")
        end
        ordered = sort(values; by=row->(row["nx"], row["workers"]))
        scatter!(errors, [row["time_median_seconds"] for row in ordered],
                 [row["l2_error_max"] for row in ordered];
                 color=[row["workers"] for row in ordered], colormap=:viridis)
        axislegend(throughput)
        Label(working[0, :], "$case / $backend / $placement; error is trajectory-dependent")
        stem = "$case-$backend-$placement-working-set"
        save(joinpath(destination, stem * ".svg"), working)
        save(joinpath(destination, stem * ".png"), working)
    end
    println("Wrote figures for $(length(groups)) case/backend/placement groups to $destination")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
