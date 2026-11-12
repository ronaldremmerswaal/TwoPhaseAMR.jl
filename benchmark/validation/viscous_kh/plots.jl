# Figures of the viscous Kelvin–Helmholtz study (run analyze.jl first).
#   julia --project=. plots.jl [<case> <n_unresolved> <n_resolved>]
# Style of all model comparisons: colour = interface resolution λ/h (sequential map,
# same in every figure), line style = model (one-velocity solid, two-velocity dashed),
# references black (viscous linear IVP / exact profile solid, inviscid Lamb dashed),
# drawn last.
using CairoMakie
using Printf
include(joinpath(@__DIR__, "analyze.jl"))

const FIGURES = joinpath(@__DIR__, "figures")
const FIELDS = joinpath(RESULTS, "fields")
const LABELS = Dict("one" => "one-velocity", "two" => "two-velocity")
const STYLES = Dict("one" => :solid, "two" => :dash)
const RESOLUTIONS = [16, 32, 64, 128, 256, 512]
const CASE_TITLES = Dict(
    "r1_thin" => "ρ_l/ρ_g = 1, Re = 1000, δ₀ = λ/128",
    "r1_thick" => "ρ_l/ρ_g = 1, Re = 100, δ₀ = λ/32",
    "r10_thin" => "ρ_l/ρ_g = 10, Re = 1000, δ₀,g = λ/128",
    "r10_thick" => "ρ_l/ρ_g = 10, Re = 100, δ₀,g = λ/32",
)
const CASE_ORDER = ["r1_thin", "r10_thin", "r1_thick", "r10_thick"]

"Colour of interface resolution λ/h = n (coarse light → fine dark)."
resolution_color(n) = cgrad(:viridis; rev = true)[0.12 + 0.88 * (log2(n) - 4) / 5]

"Compact legend: one colour per resolution plus the model/reference style key."
function style_legend!(position, resolutions; references = (:viscous, :inviscid), labels = Dict{Symbol, String}())
    res = [LineElement(; color = resolution_color(n), linewidth = 3) for n in resolutions]
    keys = [LineElement(; color = :gray30, linestyle = :solid), LineElement(; color = :gray30, linestyle = :dash)]
    names = ["one-velocity", "two-velocity"]
    if :viscous in references
        push!(keys, LineElement(; color = :black, linestyle = :solid, linewidth = 3))
        push!(names, get(labels, :viscous, "Orr–Sommerfeld linear IVP"))
    end
    if :inviscid in references
        push!(keys, LineElement(; color = :black, linestyle = :dash, linewidth = 2))
        push!(names, get(labels, :inviscid, "vortex sheet (Lamb)"))
    end
    Legend(
        position, [res, keys], [["λ/h = $n" for n in resolutions], names], ["resolution", "model / reference"];
        framevisible = false, labelsize = 12, titlesize = 12, gridshalign = :left
    )
    return nothing
end

function slope_guide!(ax, xs, ys, p; label)
    x0, x1 = extrema(xs)
    y0 = maximum(ys)
    lines!(ax, [x0, x1], [y0, y0 * (x1 / x0)^(-p)]; color = :gray60, linestyle = :dot, label)
    text!(ax, x1, y0 * (x1 / x0)^(-p); text = label, fontsize = 11, color = :gray40, align = (:left, :center))
    return nothing
end

"Error vs δ0,l/h for every case (rows) and metric (columns)."
function refinement_figure(rows)
    metrics = (
        (:growth_err, "growth-rate rel. error"), (:speed_err, "phase-speed error / ΔU"),
        (:amp_err, "η̂(T) rel. error"), (:profile_err, "ū(y,T) RMS error / ΔU"),
    )
    cases = filter(c -> any(r -> r.case == c, rows), CASE_ORDER)
    fig = Figure(; size = (1350, 255 * length(cases) + 40), fontsize = 13)
    for (i, name) in enumerate(cases), (j, (field, label)) in enumerate(metrics)
        ax = Axis(
            fig[i, j]; xscale = log2, yscale = log10, xlabel = i == length(cases) ? "δ₀,l / h" : "",
            ylabel = j == 1 ? CASE_TITLES[name] : "", title = i == 1 ? label : "", ylabelsize = 11
        )
        for model in ("one", "two")
            ms = sort(filter(r -> r.case == name && r.model == model, rows); by = r -> r.n)
            isempty(ms) && continue
            x = [r.delta0_over_h for r in ms]
            y = [max(getfield(r, field), 1.0e-6) for r in ms]
            lines!(ax, x, y; color = :gray30, linestyle = STYLES[model])
            scatter!(
                ax, x, y; color = resolution_color.([r.n for r in ms]), marker = model == "one" ? :circle : :utriangle,
                markersize = 11, strokewidth = 0.5
            )
        end
        xs = [r.delta0_over_h for r in rows if r.case == name]
        ys = [getfield(r, field) for r in rows if r.case == name]
        slope_guide!(ax, xs, ys, 1; label = "h¹")
        slope_guide!(ax, xs, ys, 2; label = "h²")
        vlines!(ax, [1.0]; color = :gray85)
    end
    style_legend!(fig[1:length(cases), 5], RESOLUTIONS; references = ())
    save(joinpath(FIGURES, "refinement.png"), fig; px_per_unit = 1)
    return fig
end

"Error in η̂(T) against wall-clock time and cell count."
function efficiency_figure(rows)
    cases = filter(c -> any(r -> r.case == c, rows), CASE_ORDER)
    fig = Figure(; size = (1200, 330 * cld(length(cases), 2) + 40), fontsize = 13)
    for (q, name) in enumerate(cases)
        ax = Axis(
            fig[cld(q, 2), 2 - q % 2]; xscale = log10, yscale = log10, xlabel = "wall-clock time [s]",
            ylabel = "η̂(T) rel. error", title = CASE_TITLES[name]
        )
        for model in ("one", "two")
            ms = sort(filter(r -> r.case == name && r.model == model, rows); by = r -> r.n)
            isempty(ms) && continue
            x, y = [r.wall for r in ms], [r.amp_err for r in ms]
            lines!(ax, x, y; color = :gray30, linestyle = STYLES[model])
            scatter!(
                ax, x, y; color = resolution_color.([r.n for r in ms]), marker = model == "one" ? :circle : :utriangle,
                markersize = 11, strokewidth = 0.5
            )
        end
    end
    style_legend!(fig[1:cld(length(cases), 2), 3], RESOLUTIONS; references = ())
    save(joinpath(FIGURES, "efficiency.png"), fig; px_per_unit = 1)
    return fig
end

"""
x-averaged tangential velocity against the distance to the (mean) interface at the
final time, for all resolutions of both models and the exact base flow. Two-velocity:
liquid velocity on the rows below, gas velocity on the rows above the interface, each
continued to the interface with its value in the adjacent row, so the horizontal gap at
y = 0 is the velocity jump carried across the interface.
"""
function profile_figure(name; ylim = 3 * thickness(base_flow(CASES[name]), 2, CASES[name].final_time))
    c = CASES[name]
    base = base_flow(c)
    T = c.final_time
    fig = Figure(; size = (1200, 620), fontsize = 13)
    yy = range(-ylim, ylim; length = 400)
    exact = [base_profile(base, y < 0 ? -1 : 1, y, T)[1] for y in yy]
    axes = Axis[]
    present = Int[]
    for (j, model) in enumerate(("one", "two"))
        ax = Axis(
            fig[1, j]; xlabel = "x-averaged tangential velocity ū / ΔU",
            ylabel = j == 1 ? "distance to the interface y / λ" : "", title = "$(LABELS[model]), t = $(T)"
        )
        push!(axes, ax)
        hlines!(ax, [0.0]; color = :gray50, linestyle = :dot)
        files = filter(f -> occursin(Regex("^$(name)_$(model)_n(\\d+)_profile\\.csv\$"), f), readdir(RUNS))
        for f in sort(files; by = f -> parse(Int, match(r"_n(\d+)_", f).captures[1]))
            n = parse(Int, match(r"_n(\d+)_", f).captures[1])
            push!(present, n)
            prof = read_table(joinpath(RUNS, f))
            y = prof["y"]
            col = resolution_color(n)
            # rows within the window plus the first row outside on each side (clipped)
            lo = maximum(filter(v -> v < -ylim, y); init = -Inf)
            hi = minimum(filter(v -> v > ylim, y); init = Inf)
            if model == "one"
                sel = findall(v -> lo <= v <= hi, y)
                scatterlines!(ax, prof["u"][sel], y[sel]; color = col, markersize = 6, linewidth = 2)
            else
                below = findall(v -> lo <= v < 0, y)
                above = findall(v -> 0 < v <= hi, y)
                ul = [prof["u_liquid"][below]; prof["u_liquid"][below[end]]]
                ug = [prof["u_gas"][above[1]]; prof["u_gas"][above]]
                scatterlines!(ax, ul, [y[below]; 0.0]; color = col, markersize = 6, linewidth = 2, linestyle = :dash)
                scatterlines!(
                    ax, ug, [0.0; y[above]]; color = col, markersize = 6, linewidth = 2, linestyle = :dash,
                    marker = :utriangle
                )
            end
        end
        lines!(ax, exact, collect(yy); color = :black, linewidth = 2.5)
        ylims!(ax, -ylim, ylim)
        text!(ax, 0.98, 0.02; text = "interface", space = :relative, align = (:right, :bottom), color = :gray40, fontsize = 11)
    end
    linkaxes!(axes...)
    resolutions = sort(unique(present))
    res = [LineElement(; color = resolution_color(n), linewidth = 3) for n in resolutions]
    keys = [
        LineElement(; color = :gray30, linestyle = :solid), LineElement(; color = :gray30, linestyle = :dash),
        LineElement(; color = :black, linewidth = 3),
    ]
    Legend(
        fig[1, 3], [res, keys],
        [
            [@sprintf("λ/h = %d (δ_l/h = %.2g)", n, thickness(base, 1, T) * n) for n in resolutions],
            ["one-velocity", "two-velocity (● liquid, ▲ gas)", "exact U(y, T)"],
        ],
        ["resolution", "model / reference"]; framevisible = false, labelsize = 12, titlesize = 12
    )
    Label(fig[0, :], "$(CASE_TITLES[name]): tangential velocity across the shear layer")
    save(joinpath(FIGURES, "profile_$(name).png"), fig; px_per_unit = 1)
    return fig
end

"|η̂|(t) of every run against the references, one panel per case."
function amplitude_figure(
        rows; cases = filter(c -> any(r -> r.case == c, rows), CASE_ORDER),
        out = FIGURES, filename = "amplitude.png"
    )
    ncol = min(2, length(cases))
    fig = Figure(; size = (560 * ncol + 220, 400 * cld(length(cases), ncol) + 20), fontsize = 13)
    present = Int[]
    for (q, name) in enumerate(cases)
        c = CASES[name]
        ax = Axis(
            fig[cld(q, ncol), mod1(q, ncol)]; yscale = log10, xlabel = "t ΔU/λ", ylabel = "|η̂| / λ",
            title = CASE_TITLES[name]
        )
        r = read_table(joinpath(RESULTS, "reference_$(name).csv"))
        for model in ("one", "two"), n in RESOLUTIONS
            path = joinpath(RUNS, "$(name)_$(model)_n$(n).csv")
            isfile(path) || continue
            push!(present, n)
            run = read_series(path)
            lines!(ax, run.t, abs.(run.η); color = resolution_color(n), linewidth = 2, linestyle = STYLES[model])
        end
        vspan!(ax, window(c)...; color = (:gray, 0.08))
        lines!(
            ax, r["t"], c.amplitude .* exp.(imag(vortex_sheet_frequency(c)) .* r["t"]); color = :black,
            linestyle = :dash, linewidth = 2
        )
        lines!(ax, r["t"], abs.(complex.(r["eta_re"], r["eta_im"])); color = :black, linewidth = 3)
        ylims!(ax, c.amplitude / 20, nothing)
    end
    style_legend!(fig[1:cld(length(cases), ncol), ncol + 1], sort(unique(present)))
    save(joinpath(out, filename), fig; px_per_unit = 1)
    return fig
end

"""
2×2 panels (one/two-velocity × unresolved/resolved mesh) of the perturbation
tangential velocity u − ū(y) (ū: row mean) of the x-faces near the interface, each face
drawn as its control volume; for the two-velocity model the velocity of the phase at the
face centre is shown.
"""
function field_figure(
        name, n_unresolved, n_resolved; out = FIGURES,
        halfwidth = 4 * thickness(base_flow(CASES[name]), 2, CASES[name].final_time)
    )
    c = CASES[name]
    fig = Figure(; size = (1150, 640), fontsize = 13)
    for (i, n) in enumerate((n_unresolved, n_resolved)), (j, model) in enumerate(("one", "two"))
        path = joinpath(FIELDS, "$(name)_$(model)_n$(n)_field.csv")
        isfile(path) || continue
        series = read_series(joinpath(RUNS, "$(name)_$(model)_n$(n).csv"))
        ηT = series.η[end]
        interface(x) = real(ηT * cis(WAVENUMBER * x))
        f = read_table(path)
        x, y, h = f["x"], f["y"], f["h"]
        u = haskey(f, "u") ? f["u"] :
            [
                yy > interface(xx) ? (isfinite(g) ? g : l) : (isfinite(l) ? l : g)
                for (xx, yy, l, g) in zip(x, y, f["u_liquid"], f["u_gas"])
            ]
        rowsum = Dict{Float64, Tuple{Float64, Float64}}()
        for q in eachindex(u)
            s, w = get(rowsum, y[q], (0.0, 0.0))
            rowsum[y[q]] = (s + h[q] * u[q], w + h[q])
        end
        up = [u[q] - rowsum[y[q]][1] / rowsum[y[q]][2] for q in eachindex(u)]
        sel = findall(q -> abs(y[q]) - h[q] / 2 <= halfwidth, eachindex(u))
        rects = [Rect2f(x[q] - h[q] / 2, y[q] - h[q] / 2, h[q], h[q]) for q in sel]
        lim = maximum(abs, up[sel])
        ax = Axis(
            fig[i, 2j - 1]; xlabel = "x/λ", ylabel = "y/λ",
            title = @sprintf("%s, λ/h = %d (δ_l(T)/h = %.2g)", LABELS[model], n, thickness(base_flow(c), 1, c.final_time) * n)
        )
        p = poly!(ax, rects; color = up[sel], colormap = :balance, colorrange = (-lim, lim), strokewidth = 0)
        xx = range(0, 1; length = 200)
        lines!(ax, xx, interface.(xx); color = :black, linewidth = 1.5)
        limits!(ax, 0, 1, -halfwidth, halfwidth)
        Colorbar(fig[i, 2j], p; label = "u − ū(y)")
    end
    Label(fig[0, :], "$(CASE_TITLES[name]): perturbation tangential velocity at t = $(c.final_time); black: interface (true scale, y axis stretched)")
    save(joinpath(out, "fields_$(name).png"), fig; px_per_unit = 1)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(FIGURES)
    rows = collect_errors()
    refinement_figure(rows)
    efficiency_figure(rows)
    amplitude_figure(rows)
    for name in CASE_ORDER
        any(r -> r.case == name, rows) && profile_figure(name)
    end
    length(ARGS) >= 3 && field_figure(ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end
