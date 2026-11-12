# Figures of the viscous Kelvin–Helmholtz study (run analyze.jl first).
#   julia --project=benchmark/validation/viscous_kh benchmark/validation/viscous_kh/plots.jl
using CairoMakie
using Printf
include(joinpath(@__DIR__, "analyze.jl"))

const FIGURES = joinpath(@__DIR__, "figures")
const FIELDS = joinpath(RESULTS, "fields")
const COLORS = Dict("one" => :royalblue3, "two" => :firebrick3)
const LABELS = Dict("one" => "one-velocity", "two" => "two-velocity")
const CASE_TITLES = Dict(
    "r1_thin" => "ρ_l/ρ_g = 1, Re = 1000, δ₀ = λ/128",
    "r1_thick" => "ρ_l/ρ_g = 1, Re = 100, δ₀ = λ/32",
    "r10_thin" => "ρ_l/ρ_g = 10, Re = 1000, δ₀,g = λ/128",
    "r10_thick" => "ρ_l/ρ_g = 10, Re = 100, δ₀,g = λ/32",
)
const CASE_ORDER = ["r1_thin", "r10_thin", "r1_thick", "r10_thick"]

function slope_guide!(ax, xs, ys, p; label)
    x0, x1 = extrema(xs)
    y0 = maximum(ys)
    lines!(ax, [x0, x1], [y0, y0 * (x1 / x0)^(-p)]; color = :gray50, linestyle = :dash, label)
    return nothing
end

"Error vs δ0/h for every case (rows) and metric (columns)."
function refinement_figure(rows)
    metrics = (
        (:growth_err, "growth-rate rel. error"), (:speed_err, "phase-speed error / ΔU"),
        (:amp_err, "η̂(T) rel. error"), (:profile_err, "ū(y,T) RMS error / ΔU"),
    )
    cases = filter(c -> any(r -> r.case == c, rows), CASE_ORDER)
    fig = Figure(; size = (1250, 250 * length(cases) + 60), fontsize = 13)
    for (i, name) in enumerate(cases), (j, (field, label)) in enumerate(metrics)
        ax = Axis(fig[i, j]; xscale = log2, yscale = log10, xlabel = "δ₀,l / h (liquid layer)",
            ylabel = j == 1 ? label * "\n" * CASE_TITLES[name] : label, title = i == 1 ? label : "")
        j == 1 && (ax.ylabel = CASE_TITLES[name])
        for model in ("one", "two")
            ms = sort(filter(r -> r.case == name && r.model == model, rows); by = r -> r.n)
            isempty(ms) && continue
            x = [r.delta0_over_h for r in ms]
            y = [max(getfield(r, field), 1.0e-6) for r in ms]
            scatterlines!(ax, x, y; color = COLORS[model], label = LABELS[model])
        end
        xs = [r.delta0_over_h for r in rows if r.case == name]
        ys = [getfield(r, field) for r in rows if r.case == name]
        slope_guide!(ax, xs, ys, 1; label = "O(h)")
        slope_guide!(ax, xs, ys, 2; label = "O(h²)")
        vlines!(ax, [1.0]; color = :gray80)
        i == 1 && j == 1 && axislegend(ax; position = :lb, framevisible = false)
    end
    save(joinpath(FIGURES, "refinement.png"), fig; px_per_unit = 1)
    return fig
end

"Error in η̂(T) against wall-clock time and cell count."
function efficiency_figure(rows)
    cases = filter(c -> any(r -> r.case == c, rows), CASE_ORDER)
    fig = Figure(; size = (1100, 520), fontsize = 13)
    for (j, (field, xlabel)) in enumerate(((:wall, "wall-clock time [s]"), (:cells, "cells")))
        ax = Axis(fig[1, j]; xscale = log10, yscale = log10, xlabel, ylabel = "η̂(T) rel. error")
        for (k, name) in enumerate(cases), model in ("one", "two")
            ms = sort(filter(r -> r.case == name && r.model == model, rows); by = r -> r.n)
            isempty(ms) && continue
            scatterlines!(ax, [Float64(getfield(r, field)) for r in ms], [r.amp_err for r in ms];
                color = COLORS[model], marker = (:circle, :rect, :utriangle, :diamond)[k],
                label = "$(LABELS[model]), $(name)")
        end
        j == 2 && Legend(fig[1, 3], ax; framevisible = false)
    end
    save(joinpath(FIGURES, "efficiency.png"), fig; px_per_unit = 1)
    return fig
end

"""
x-averaged tangential velocity against the distance to the (mean) interface at the
final time, for all resolutions of both models and the exact base flow.
"""
function profile_figure(name; ylim = 0.06)
    c = CASE_ORDER[1] == name ? CASES[name] : CASES[name]
    base = base_flow(c)
    T = c.final_time
    fig = Figure(; size = (1100, 560), fontsize = 13)
    yy = range(-ylim, ylim; length = 400)
    exact = [base_profile(base, y < 0 ? -1 : 1, y, T)[1] for y in yy]
    for (j, model) in enumerate(("one", "two"))
        ax = Axis(fig[1, j]; xlabel = "tangential velocity u", ylabel = j == 1 ? "distance to interface y" : "",
            title = "$(LABELS[model]) — $(CASE_TITLES[name]), t = $(T)")
        lines!(ax, exact, collect(yy); color = :black, linewidth = 2.5, label = "reference (erf, t = T)")
        hlines!(ax, [0.0]; color = :gray60, linestyle = :dot)
        files = sort(filter(f -> occursin(Regex("^$(name)_$(model)_n(\\d+)_profile\\.csv\$"), f), readdir(RUNS));
            by = f -> parse(Int, match(r"_n(\d+)_", f).captures[1]))
        nres = length(files)
        for (q, f) in enumerate(files)
            n = parse(Int, match(r"_n(\d+)_", f).captures[1])
            prof = read_table(joinpath(RUNS, f))
            y = prof["y"]
            shade = 0.25 + 0.75 * (q - 1) / max(nres - 1, 1)
            col = model == "one" ? RGBf(0.1, 0.2 + 0.3 * (1 - shade), shade) : RGBf(shade, 0.15 + 0.3 * (1 - shade), 0.1)
            lab = @sprintf("λ/h = %d (δ_l/h = %.2g)", n, thickness(base, 1, T) * n)
            if model == "one"
                sel = findall(v -> abs(v) <= ylim, y)
                scatterlines!(ax, prof["u"][sel], y[sel]; color = col, markersize = 5, label = lab)
            else
                below = findall(v -> -ylim <= v < 0, y)
                above = findall(v -> 0 < v <= ylim, y)
                # each phase on its own side, continued to the interface with its value in
                # the adjacent row to show the jump
                ul = [prof["u_liquid"][below]; prof["u_liquid"][below[end]]]
                ug = [prof["u_gas"][above[1]]; prof["u_gas"][above]]
                scatterlines!(ax, ul, [y[below]; 0.0]; color = col, markersize = 5, label = lab)
                scatterlines!(ax, ug, [0.0; y[above]]; color = col, markersize = 5, marker = :utriangle)
            end
        end
        ylims!(ax, -ylim, ylim)
        axislegend(ax; position = :lt, framevisible = false, labelsize = 11)
    end
    save(joinpath(FIGURES, "profile_$(name).png"), fig; px_per_unit = 1)
    return fig
end

"2×2 panels of the perturbation tangential velocity u − ū(y) near the interface."
function field_figure(name, n_unresolved, n_resolved; halfwidth = 0.08)
    c = CASES[name]
    fig = Figure(; size = (1100, 640), fontsize = 13)
    for (i, n) in enumerate((n_unresolved, n_resolved)), (j, model) in enumerate(("one", "two"))
        path = joinpath(FIELDS, "$(name)_$(model)_n$(n)_field.csv")
        isfile(path) || continue
        f = read_table(path)
        x, y = f["x"], f["y"]
        u = haskey(f, "u") ? f["u"] : [isfinite(gl) && (yy > 0 || !isfinite(ll)) ? gl : ll for (yy, ll, gl) in zip(y, f["u_liquid"], f["u_gas"])]
        ys = sort(unique(y))
        mean = Dict(v => sum(u[y .== v]) / count(==(v), y) for v in ys)
        up = [u[q] - mean[y[q]] for q in eachindex(u)]
        sel = findall(v -> abs(v) <= halfwidth, y)
        xs = sort(unique(x[sel]))
        yv = sort(unique(y[sel]))
        Z = fill(NaN, length(xs), length(yv))
        ix = Dict(v => q for (q, v) in enumerate(xs))
        iy = Dict(v => q for (q, v) in enumerate(yv))
        for q in sel
            Z[ix[x[q]], iy[y[q]]] = up[q]
        end
        lim = maximum(abs, filter(isfinite, Z))
        series = read_series(joinpath(RUNS, "$(name)_$(model)_n$(n).csv"))
        ηT = series.η[end]
        ax = Axis(fig[i, 2j - 1]; xlabel = "x/λ", ylabel = "y/λ", aspect = DataAspect(),
            title = @sprintf("%s, λ/h = %d (δ_l/h = %.2g)", LABELS[model], n, thickness(base_flow(c), 1, c.final_time) * n))
        hm = heatmap!(ax, xs, yv, Z; colormap = :balance, colorrange = (-lim, lim))
        xx = range(0, 1; length = 200)
        lines!(ax, xx, [real(ηT * cis(WAVENUMBER * v)) for v in xx]; color = :black, linewidth = 1.5)
        Colorbar(fig[i, 2j], hm; label = "u − ū(y)")
    end
    Label(fig[0, :], "$(CASE_TITLES[name]): perturbation tangential velocity at t = $(c.final_time), interface (black, true scale)")
    save(joinpath(FIGURES, "fields_$(name).png"), fig; px_per_unit = 1)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(FIGURES)
    rows = collect_errors()
    refinement_figure(rows)
    efficiency_figure(rows)
    for name in CASE_ORDER
        any(r -> r.case == name, rows) && profile_figure(name)
    end
    length(ARGS) >= 3 && field_figure(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end
