# Tables and figure of the CTU-threshold study (after ctu_study.jl reproducer / kh).
#   julia --project=.. ctu_report.jl
include(joinpath(@__DIR__, "..", "plots.jl"))

const CTU = @__DIR__
const BOUND_TOL = 1.0e-12
const THRESHOLDS = (0.5, 0.1, 0.01, 1.0e-12)   # as in ctu_study.jl
const SCHEME_ORDER = ("Upwind", "LaxWendroff", "MC")
const KH_NS = (64, 128, 256)

function read_reproducer()
    lines = collect(eachline(joinpath(CTU, "reproducer.csv")))
    rows = NamedTuple[]
    for l in lines[2:end]
        a = split(l, ','; limit = 11)
        push!(
            rows, (;
                scheme = a[1], θ = parse(Float64, a[2]), n = parse(Int, a[3]),
                dh = parse(Float64, a[4]), j1 = parse(Float64, a[5]), j160 = parse(Float64, a[6]),
                umin = parse(Float64, a[7]), umax = parse(Float64, a[8]), fmin = parse(Float64, a[9]),
                fmax = parse(Float64, a[10]), failure = strip(a[11], '"'),
            )
        )
    end
    return rows
end

bounds_note(r) = join(
    filter(
        !isempty, [
            !isempty(r.failure) ? "FAILED: $(r.failure)" : "",
            !(isfinite(r.umin) && isfinite(r.umax)) ? "non-finite" : "",
            r.umax > 0.5 + BOUND_TOL || r.umin < -0.5 - BOUND_TOL ? @sprintf("u ∈ [%.4f, %.4f]", r.umin, r.umax) : "",
            r.fmin < -BOUND_TOL || r.fmax > 1 + BOUND_TOL ? @sprintf("f ∈ [%.1e, 1%+.1e]", r.fmin, r.fmax - 1) : "",
        ]
    ), "; "
)

function report()
    rows = read_reproducer()
    io = IOBuffer()
    println(io, "## Part 1 — jump reproducer (`../jump_reproducer.jl`, uniform mesh, no hanging faces)\n")
    println(io, "max|u_g − u_l| over x-faces carrying both phases, after 1 and 160 steps; order = log₂ of the ratio between successive δ/h (1.6 → 3.2 → 6.4). Bounds: initial |u| ≤ 0.5, f ∈ [0, 1].\n")
    println(io, "| scheme | ctu_threshold | jump(1) δ/h=1.6, 3.2, 6.4 | order | jump(160) δ/h=1.6, 3.2, 6.4 | order | bounds / failures |")
    println(io, "|---|---:|---|---|---|---|---|")
    for s in SCHEME_ORDER, θ in THRESHOLDS
        rs = sort(filter(r -> r.scheme == s && r.θ == θ, rows); by = r -> r.n)
        isempty(rs) && continue
        f(v) = join((@sprintf("%.3f", x) for x in v), ", ")
        o(v) = join((@sprintf("%.2f", log2(v[k] / v[k + 1])) for k in 1:(length(v) - 1)), ", ")
        j1, j160 = [r.j1 for r in rs], [r.j160 for r in rs]
        notes = join(filter(!isempty, [isempty(bounds_note(r)) ? "" : "δ/h=$(r.dh): $(bounds_note(r))" for r in rs]), "; ")
        @printf(io, "| %s | %g | %s | %s | %s | %s | %s |\n", s, θ, f(j1), o(j1), f(j160), o(j160), isempty(notes) ? "ok" : notes)
    end
    # figure
    fig = Figure(; size = (1150, 470), fontsize = 13)
    colors = Dict(0.5 => :black, 0.1 => :royalblue3, 0.01 => :darkorange2, 1.0e-12 => :firebrick3)
    styles = Dict("Upwind" => :dot, "LaxWendroff" => :solid, "MC" => :dash)
    for (j, (field, title)) in enumerate(((:j1, "after 1 step"), (:j160, "after 160 steps")))
        ax = Axis(fig[1, j]; xscale = log2, yscale = log10, xlabel = "δ / h", ylabel = "max |u_g − u_l| / ΔU", title)
        for s in SCHEME_ORDER, θ in THRESHOLDS
            rs = sort(filter(r -> r.scheme == s && r.θ == θ, rows); by = r -> r.n)
            isempty(rs) && continue
            scatterlines!(
                ax, [r.dh for r in rs], [max(getfield(r, field), 1.0e-8) for r in rs];
                color = colors[θ], linestyle = styles[s], markersize = 8
            )
        end
        lines!(ax, [1.6, 6.4], [0.6, 0.15]; color = :gray60, linestyle = :dot)
        text!(ax, 6.4, 0.15; text = "h¹", color = :gray40, fontsize = 11)
    end
    th = [LineElement(; color = colors[θ], linewidth = 3) for θ in THRESHOLDS]
    sc = [LineElement(; color = :gray30, linestyle = styles[s]) for s in SCHEME_ORDER]
    Legend(
        fig[1, 3], [th, sc], [["$(θ)" for θ in THRESHOLDS], collect(SCHEME_ORDER)], ["ctu_threshold", "bulk scheme"];
        framevisible = false
    )
    save(joinpath(CTU, "jump_vs_h.png"), fig; px_per_unit = 1)

    # part 2: KH runs
    runs = joinpath(CTU, "runs")
    if isdir(runs)
        c = CASES["r1_thin"]
        r = read_table(joinpath(RESULTS, "reference_r1_thin.csv"))
        ref = (; t = r["t"], η = complex.(r["eta_re"], r["eta_im"]))
        a, b = window(c)
        s_ref, _ = fit(ref.t, ref.η, a, b)
        println(io, "\n## Part 2 — r1_thin (ρ ratio 1, Re = 1000, δ₀ = λ/128), two-velocity model\n")
        @printf(io, "Reference growth rate (fit over [T/3, T]) %.4f. Baseline = main study (LaxWendroff, ctu_threshold 0.5). The graded mesh has its level transitions at |y| = 1/32 and beyond, i.e. no hanging faces at the interface.\n\n", s_ref)
        println(io, "| scheme | ctu_threshold | λ/h | δ₀/h | growth rate | rel. error | \\|η̂(T)\\|/ref | max jump at T | bounds / failures |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---|")
        entries = Tuple{String, Float64, Int, String}[]
        for n in KH_NS
            push!(entries, ("LaxWendroff (baseline)", 0.5, n, joinpath(RUNS, "r1_thin_two_n$(n).csv")))
        end
        for f in sort(readdir(runs))
            m = match(r"^r1_thin_two_n(\d+)_(\w+)_ctu([0-9.e+-]+)\.csv$", f)
            isnothing(m) && continue
            push!(entries, (m.captures[2], parse(Float64, m.captures[3]), parse(Int, m.captures[1]), joinpath(runs, f)))
        end
        for (s, θ, n, path) in sort(entries; by = e -> (e[1], -e[2], e[3]))
            isfile(path) || continue
            run = read_series(path)
            prof = read_table(replace(path, ".csv" => "_profile.csv"))
            done = run.t[end] >= c.final_time - 1.0e-9
            g, _ = done ? fit(run.t, run.η, a, b) : (NaN, NaN)
            ratio = abs(run.η[end]) / abs(interpolate_reference(ref, run.t[end]))
            m = run.meta
            note = haskey(m, "umin") ? join(
                    filter(
                        !isempty, [
                            get(m, "failed", "no") == "yes" ? "FAILED" : "",
                            parse(Int, get(m, "infinite", "0")) > 0 ? "±Inf entries" : "",
                            parse(Float64, m["fmin"]) < -BOUND_TOL || parse(Float64, m["fmax"]) > 1 + BOUND_TOL ? "f ∈ [$(m["fmin"]), $(m["fmax"])]" : "",
                            "u ∈ [$(m["umin"]), $(m["umax"])]",
                        ]
                    ), "; "
                ) : "(baseline run, bounds not recorded)"
            @printf(
                io, "| %s | %g | %d | %.3g | %.4f | %+.3f | %.3f | %.3f | %s |\n", s, θ, n, n / 128, g, (g - s_ref) / s_ref,
                ratio, interface_jump(prof), note
            )
        end
    end
    write(joinpath(CTU, "results.md"), String(take!(io)))
    print(read(joinpath(CTU, "results.md"), String))
    return nothing
end

report()
