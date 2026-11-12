# Verification of the two-fluid Orr–Sommerfeld solver against published values.
#   julia --project=benchmark/validation/viscous_kh benchmark/validation/viscous_kh/verify_reference.jl
# Writes results/verify_reference.txt.
using Printf
include(joinpath(@__DIR__, "os_reference.jl"))

const OUT = IOBuffer()
report(args...) = (println(args...); println(OUT, args...))

# 1. Plane Poiseuille flow, Re = 10^4, α = 1, split into two identical layers
#    (Orszag 1971: c = 0.23752649 + 0.00373967i).
function poiseuille(N)
    base = FunctionBaseFlow((s, y, t) -> (1 - y^2, -2y, -2.0))
    lay(s) = Layer(s, N, 1.0; density = 1.0, viscosity = 1.0e-4)
    p = TwoLayerProblem(; liquid = lay(-1), gas = lay(1), wavenumber = 1.0, wall = :no_slip)
    ω, _ = eigenmodes(p, base)
    c = ω ./ p.wavenumber
    return c[argmin(abs.(c .- (0.2375 + 0.0037im)))], c[1]
end

# 2. Bagué et al. (2010), Table I/II: composite erf profile, δ_l = δ_g = 1, U_g = 1,
#    no-slip walls at L = 6δ, Re_g = 2000; growth rate α c_i δ_g / U_g at α δ_g = π/2.
function bague(case, N; α = pi / 2, stretch = 0.0)
    r, m, We_g = case === :A ? (1.0, 0.1, Inf) : case === :B ? (1.0, 0.1, 10.0) :
        case === :C ? (0.1, 0.99, Inf) : (0.1, 0.99, 10.0)
    ρg, μg = 1.0, 1 / 2000
    ρl, μl = ρg / r, μg / m
    σ = isinf(We_g) ? 0.0 : ρg / We_g
    base = ErfBaseFlow(; U∞ = (m, 1.0), δ0 = (1.0, 1.0), ν = (μl / ρl, μg / ρg), diffusing = false)
    p = TwoLayerProblem(;
        liquid = Layer(-1, N, 6.0; stretch, density = ρl, viscosity = μl),
        gas = Layer(1, N, 6.0; stretch, density = ρg, viscosity = μg),
        wavenumber = α, surface_tension = σ, wall = :no_slip,
    )
    ω, _ = leading_mode(p, base)
    return ω
end

# 3. Near-inviscid checks with a very thin frozen layer (vortex sheet + capillarity/
#    gravity) between free-slip walls at |y| = H (Lamb 1932, §267–268):
#    ρ_l(ω − kU_l)² + ρ_g(ω − kU_g)² = (σk³ + g k (ρ_l − ρ_g)) tanh(kH).
#    Residual differences are O(kδ) (layer thickness) and O(√(νk²/|ω|)) (viscous
#    interfacial boundary layer), i.e. ≈ 1e-3 here.
function vortex_sheet(; ρ = (1.0, 0.1), U = (-0.1, 0.9), σ = 0.01, g = 0.0, δ = 2.0e-4, ν = 1.0e-9, N = 120, H = 0.5)
    k = 2pi
    base = ErfBaseFlow(; U∞ = (-U[1], U[2]), δ0 = (δ, δ), ν = (ν, ν), diffusing = false)
    p = TwoLayerProblem(;
        liquid = Layer(-1, N, H; stretch = 9.0, density = ρ[1], viscosity = ρ[1] * ν),
        gas = Layer(1, N, H; stretch = 9.0, density = ρ[2], viscosity = ρ[2] * ν),
        wavenumber = k, surface_tension = σ, gravity = g, wall = :free_slip,
    )
    ω, _ = eigenmodes(p, base)
    # (ρ_l+ρ_g)ω² − 2k(ρ_lU_l+ρ_gU_g)ω + k²(ρ_lU_l²+ρ_gU_g²) − (σk³ + gk(ρ_l−ρ_g))tanh(kH) = 0
    a = sum(ρ)
    b = -2k * (ρ[1] * U[1] + ρ[2] * U[2])
    c = k^2 * (ρ[1] * U[1]^2 + ρ[2] * U[2]^2) - (σ * k^3 + g * k * (ρ[1] - ρ[2])) * tanh(k * H)
    d = sqrt(complex(b^2 - 4a * c))
    exact = ((-b + d) / 2a, (-b - d) / 2a)
    found = map(e -> ω[argmin(abs.(ω .- e))], exact)
    return exact, found
end

report("== 1. Plane Poiseuille, Re = 1e4, α = 1 (Orszag 1971: c = 0.23752649 + 0.00373967i)")
for N in (40, 60, 80)
    c, _ = poiseuille(N)
    report(@sprintf("  N = %3d per layer: c = %.8f %+.8fi", N, real(c), imag(c)))
end

report("\n== 2. Bagué et al. 2010, growth rate α c_i δ_g/U_g at αδ_g = π/2, L = 6δ")
report("   published (Table II, converged digits): A 0.1087018, B 0.0937734, C 0.200692, D 0.188538")
for case in (:A, :B, :C, :D), (N, β) in ((70, 0.0), (100, 0.0), (70, 3.0), (100, 3.0), (130, 3.0))
    ω = bague(case, N; stretch = β)
    report(@sprintf("  case %s N = %3d β = %.0f: α c_i = %.8f  (α c_r = %+.8f)", case, N, β, imag(ω), real(ω)))
end

report("\n== 3. Near-inviscid vortex-sheet / capillary / Rayleigh–Taylor limit (kδ = 1.3e-3, ν = 1e-9)")
for (label, kw) in (
        ("capillary wave, no shear", (; U = (0.0, 0.0), σ = 0.01)),
        ("KH, We = 2-ish", (; σ = 0.0045)),
        ("RT, heavy on top (g < 0)", (; U = (0.0, 0.0), σ = 0.001, g = -1.0)),
        ("KH, ρ ratio 1", (; ρ = (1.0, 1.0), U = (-0.5, 0.5), σ = 1 / (8pi))),
    )
    exact, found = vortex_sheet(; kw...)
    report("  ", label)
    for (e, f) in zip(exact, found)
        report(@sprintf("    exact %.6f %+.6fi   OS %.6f %+.6fi", real(e), imag(e), real(f), imag(f)))
    end
end

mkpath(joinpath(@__DIR__, "results"))
write(joinpath(@__DIR__, "results", "verify_reference.txt"), String(take!(OUT)))
