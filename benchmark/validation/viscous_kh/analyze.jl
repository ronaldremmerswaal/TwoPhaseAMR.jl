# Error tables and refinement plots of the viscous Kelvin–Helmholtz study.
#   julia --project=benchmark/validation/viscous_kh benchmark/validation/viscous_kh/analyze.jl
# Reads results/reference_<case>.csv and results/runs/*.csv, writes
# results/errors.csv, results/tables.md and figures/*.png.
using Printf
include(joinpath(@__DIR__, "kh_cases.jl"))

const RESULTS = joinpath(@__DIR__, "results")
const RUNS = joinpath(RESULTS, "runs")

function read_series(path)
    meta = Dict{String, String}()
    t, η = Float64[], ComplexF64[]
    for line in eachline(path)
        if startswith(line, "#")
            for kv in split(line[2:end])
                k, v = split(kv, '=')
                meta[k] = v
            end
        elseif !startswith(line, "t,")
            a = parse.(Float64, split(line, ','))
            push!(t, a[1])
            push!(η, complex(a[2], a[3]))
        end
    end
    return (; meta, t, η)
end

function read_table(path)
    lines = collect(eachline(path))
    header = split(lines[1], ',')
    data = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    return Dict(h => [d[j] for d in data] for (j, h) in enumerate(header))
end

"Unwrapped phase of a complex series."
function unwrapped_phase(η)
    φ = angle.(η)
    for j in 2:length(φ)
        φ[j] -= 2pi * round((φ[j] - φ[j - 1]) / 2pi)
    end
    return φ
end

"Least-squares slope of y(t) over t ∈ [a, b]."
function slope(t, y, a, b)
    idx = findall(τ -> a - 1.0e-12 <= τ <= b + 1.0e-12, t)
    tt, yy = t[idx], y[idx]
    tm, ym = sum(tt) / length(tt), sum(yy) / length(yy)
    return sum((tt .- tm) .* (yy .- ym)) / sum((tt .- tm) .^ 2)
end

"Growth rate and phase speed fitted over the window [a, b]: η̂ ∝ exp(s t − i ω_r t)."
function fit(t, η, a, b)
    s = slope(t, log.(abs.(η)), a, b)
    ωr = -slope(t, unwrapped_phase(η), a, b)
    return s, ωr / WAVENUMBER
end

"Linear interpolation of the complex reference at time τ (log-amplitude and phase)."
function interpolate_reference(ref, τ)
    j = clamp(searchsortedlast(ref.t, τ), 1, length(ref.t) - 1)
    θ = (τ - ref.t[j]) / (ref.t[j + 1] - ref.t[j])
    la = (1 - θ) * log(abs(ref.η[j])) + θ * log(abs(ref.η[j + 1]))
    ph = unwrapped_phase(ref.η[j:(j + 1)])
    return exp(la + im * ((1 - θ) * ph[1] + θ * ph[2]))
end

window(c::KHCase) = (c.final_time / 3, c.final_time)

"Mean tangential velocity profile error (RMS over the rows with |y| ≤ 0.1) at T."
function profile_error(c::KHCase, prof)
    base = base_flow(c)
    T = c.final_time
    y = prof["y"]
    u = haskey(prof, "u") ? prof["u"] : [yy < 0 ? ul : ug for (yy, ul, ug) in zip(y, prof["u_liquid"], prof["u_gas"])]
    idx = findall(yy -> abs(yy) <= 0.1, y)
    exact = [base_profile(base, yy < 0 ? -1 : 1, yy, T)[1] for yy in y[idx]]
    return sqrt(sum(abs2, u[idx] .- exact) / length(idx))
end

"Velocity jump between the phases on the rows where both are present (two-velocity)."
function interface_jump(prof)
    haskey(prof, "u_liquid") || return NaN
    both = findall(j -> isfinite(prof["u_liquid"][j]) && isfinite(prof["u_gas"][j]), eachindex(prof["y"]))
    isempty(both) && return NaN
    return maximum(prof["u_gas"][both] .- prof["u_liquid"][both])
end

function collect_errors()
    rows = NamedTuple[]
    for name in sort(collect(keys(CASES)))
        c = CASES[name]
        refpath = joinpath(RESULTS, "reference_$(name).csv")
        isfile(refpath) || continue
        r = read_table(refpath)
        ref = (; t = r["t"], η = complex.(r["eta_re"], r["eta_im"]))
        a, b = window(c)
        s_ref, c_ref = fit(ref.t, ref.η, a, b)
        for file in sort(readdir(RUNS))
            m = match(Regex("^$(name)_(one|two)_n(\\d+)\\.csv\$"), file)
            isnothing(m) && continue
            run = read_series(joinpath(RUNS, file))
            run.t[end] >= c.final_time - 1.0e-9 || continue
            n = parse(Int, m.captures[2])
            s, cr = fit(run.t, run.η, a, b)
            ηT = run.η[end]
            ηT_ref = interpolate_reference(ref, run.t[end])
            prof = read_table(joinpath(RUNS, replace(file, ".csv" => "_profile.csv")))
            δT = thickness(base_flow(c), 1, c.final_time)   # liquid (thinner) layer at T
            push!(rows, (;
                case = name, model = m.captures[1], n,
                delta0_over_h = minimum(initial_thicknesses(c)) * n,
                deltaT_over_h = δT * n,
                cells = parse(Int, run.meta["cells"]), steps = parse(Int, run.meta["steps"]),
                wall = parse(Float64, run.meta["wall"]),
                growth = s, growth_ref = s_ref, growth_err = abs(s - s_ref) / s_ref,
                speed = cr, speed_ref = c_ref, speed_err = abs(cr - c_ref),
                amp_err = abs(ηT - ηT_ref) / abs(ηT_ref),
                profile_err = profile_error(c, prof), jump = interface_jump(prof),
            ))
        end
    end
    return rows
end

order(e1, e2) = log2(e1 / e2)

function write_tables(rows)
    open(joinpath(RESULTS, "errors.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        for r in rows
            println(io, join(values(r), ","))
        end
    end
    open(joinpath(RESULTS, "tables.md"), "w") do io
        for name in sort(unique(r.case for r in rows))
            c = CASES[name]
            sub = filter(r -> r.case == name, rows)
            r0 = sub[1]
            @printf(io, "\n### %s — ρ_l/ρ_g = %g, Re = %g, δ0_g = 1/%d, T = %g; reference: growth %.5f, phase speed %+.5f (fit over [T/3, T])\n\n",
                name, c.density[1] / c.density[2], c.reynolds, round(Int, 1 / c.thickness), c.final_time, r0.growth_ref, r0.speed_ref)
            println(io, "| model | λ/h | δ0_l/h | cells | steps | wall [s] | growth err | order | phase-speed err | order | η̂(T) err | order | ū(y,T) RMS err | order | jump |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for model in ("one", "two")
                ms = sort(filter(r -> r.model == model, sub); by = r -> r.n)
                for (j, r) in enumerate(ms)
                    o(f) = j == 1 ? "" : @sprintf("%.2f", order(getfield(ms[j - 1], f), getfield(r, f)))
                    @printf(io, "| %s | %d | %.3g | %d | %d | %.1f | %.2e | %s | %.2e | %s | %.2e | %s | %.2e | %s | %s |\n",
                        model, r.n, r.delta0_over_h, r.cells, r.steps, r.wall, r.growth_err, o(:growth_err),
                        r.speed_err, o(:speed_err), r.amp_err, o(:amp_err), r.profile_err, o(:profile_err),
                        isnan(r.jump) ? "–" : @sprintf("%.3f", r.jump))
                end
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = collect_errors()
    write_tables(rows)
    print(read(joinpath(RESULTS, "tables.md"), String))
end
