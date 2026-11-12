# Linear reference solutions for the viscous Kelvin–Helmholtz cases.
#   julia --project=benchmark/validation/viscous_kh benchmark/validation/viscous_kh/make_reference.jl
# Writes results/reference_<case>.csv (t, Re η̂, Im η̂ of the linear initial-value
# problem) and results/reference_summary.txt (eigenvalues, quasi-steady growth rates,
# discretisation checks).
using Printf
include(joinpath(@__DIR__, "kh_cases.jl"))

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)
const OUT = IOBuffer()
report(args...) = (println(args...); println(OUT, args...))

const DT = 2.5e-3
const DT_OUT = 5.0e-3

function study(c::KHCase)
    T = c.final_time
    times = collect(0:DT_OUT:T)
    N, β = c.collocation
    PRIMARY = (; N, stretch = β, dt = DT)
    CHECK_DT = (; N, stretch = β, dt = DT / 2)
    CHECK_N = (; N = N + 20, stretch = β, dt = DT)
    ref = reference_history(c; times, PRIMARY...)
    chk_dt = reference_history(c; times, CHECK_DT...)
    chk_n = reference_history(c; times, CHECK_N...)
    # time integrator check: frozen base flow, the eigenmode must grow as exp(−iω0 t)
    p, ω0, X0 = initial_mode(c)
    frozen = integrate(p, base_flow(c; diffusing = false), X0, [T]; dt = DT)[end, 1]
    sheet = vortex_sheet_frequency(c)
    report(@sprintf("== %s: ρ = %s, ν = %s, δ0 = %s, U∞ = %s, σ = %.6g, T = %g, |η̂(0)| = %g",
        c.name, c.density, kinematic_viscosities(c), initial_thicknesses(c), far_velocities(c),
        surface_tension(c), T, c.amplitude))
    report(@sprintf("   vortex sheet (Lamb):     ω = %+.6f %+.6fi", real(sheet), imag(sheet)))
    report(@sprintf("   OS eigenvalue at t = 0:  ω = %+.6f %+.6fi", real(ref.ω0), imag(ref.ω0)))
    for t in range(0, T; length = 5)
        ω, _ = leading_mode(ref.problem, base_flow(c); t)
        report(@sprintf("   quasi-steady OS at t = %.3f (δ_g = %.5f): ω = %+.6f %+.6fi", t,
            thickness(base_flow(c), 2, t), real(ω), imag(ω)))
    end
    rel(chk) = maximum(abs.(ref.eta .- chk.eta) ./ abs.(chk.eta))
    report(@sprintf("   IVP: |η̂(T)|/|η̂(0)| = %.5f, phase(T) = %+.5f rad", abs(ref.eta[end]) / c.amplitude, angle(ref.eta[end] / ref.eta[1])))
    for (label, chk, setting) in (("time step", chk_dt, CHECK_DT), ("collocation", chk_n, CHECK_N))
        report(@sprintf("   IVP %s check (N,β,dt) = %s vs %s: max_t |Δη̂|/|η̂| = %.2e", label, values(PRIMARY), values(setting), rel(chk)))
    end
    report(@sprintf("   Radau IIA frozen-base check: |η̂(T)/(η̂(0)e^{-iω0 T}) − 1| = %.2e",
        abs(frozen / (X0[end] * exp(-im * ω0 * T)) - 1)))
    open(joinpath(RESULTS, "reference_$(c.name).csv"), "w") do io
        println(io, "t,eta_re,eta_im")
        for (t, η) in zip(times, ref.eta)
            @printf(io, "%.6f,%.12e,%.12e\n", t, real(η), imag(η))
        end
    end
    return nothing
end

for name in sort(collect(keys(CASES)))
    study(CASES[name])
end
write(joinpath(RESULTS, "reference_summary.txt"), String(take!(OUT)))
