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

# 3. Near-inviscid limit: Lamb (1932, §267–268) for a vortex sheet with capillarity and
#    gravity between free-slip walls at |y| = H,
#    ρ_l(ω − kU_l)² + ρ_g(ω − kU_g)² = (σk³ + g k (ρ_l − ρ_g)) tanh(kH).
#    (a) no shear, small ν: capillary wave / Rayleigh–Taylor (signs of σ, g, ρ);
#    (b) frozen erf layer of thickness δ → 0 at fixed Re_δ = ΔUδ/ν: ω → Lamb.
function lamb(; ρ, U, σ, g = 0.0, H = 0.5)
    k = 2pi
    a = sum(ρ)
    b = -2k * (ρ[1] * U[1] + ρ[2] * U[2])
    c = k^2 * (ρ[1] * U[1]^2 + ρ[2] * U[2]^2) - (σ * k^3 + g * k * (ρ[1] - ρ[2])) * tanh(k * H)
    d = sqrt(complex(b^2 - 4a * c))
    return (-b + d) / 2a
end

function thin_layer(; ρ, U, σ, g = 0.0, δ, ν, N = 100, β = 2.0, H = 0.5)
    base = ErfBaseFlow(; U∞ = (-U[1], U[2]), δ0 = (δ, δ), ν = (ν, ν), diffusing = false)
    p = TwoLayerProblem(;
        liquid = Layer(-1, N, H; stretch = β, density = ρ[1], viscosity = ρ[1] * ν),
        gas = Layer(1, N, H; stretch = β, density = ρ[2], viscosity = ρ[2] * ν),
        wavenumber = 2pi, surface_tension = σ, gravity = g, wall = :free_slip,
    )
    ω, _ = eigenmodes(p, base)
    exact = lamb(; ρ, U, σ, g, H)
    return exact, ω[argmin(abs.(ω .- exact))]
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

report("\n== 3a. No shear, ν = 1e-6 (Lamb: neutral capillary wave, unstable RT); difference = viscous damping")
for (label, kw) in (
        ("capillary wave, ρ_l/ρ_g = 10", (; ρ = (1.0, 0.1), U = (0.0, 0.0), σ = 0.01)),
        ("Rayleigh–Taylor, heavy on top", (; ρ = (1.0, 0.1), U = (0.0, 0.0), σ = 0.001, g = -1.0)),
    )
    exact, found = thin_layer(; kw..., δ = 0.01, ν = 1.0e-6, β = 0.0)
    report(@sprintf("  %-32s Lamb ω = %+.6f %+.6fi   OS ω = %+.6f %+.6fi", label, real(exact), imag(exact), real(found), imag(found)))
end
report("\n== 3b. Shear layer δ → 0 at Re_δ = 2000, We = 2 (Lamb vortex sheet as the limit)")
for (label, kw) in (
        ("ρ_l/ρ_g = 1", (; ρ = (1.0, 1.0), U = (-0.5, 0.5), σ = 1 / (8pi))),
        ("ρ_l/ρ_g = 10", (; ρ = (1.0, 0.1), U = (-0.5, 0.5), σ = 0.1 / 1.1 / (4pi))),
    )
    for δ in (1 / 32, 1 / 64, 1 / 128, 1 / 256), (N, β) in ((100, 2.0), (120, 3.0))
        exact, found = thin_layer(; kw..., δ, ν = δ / 2000, N, β)
        report(@sprintf("  %s kδ = %.4f N = %d β = %g: OS ω = %+.5f %+.5fi   (Lamb %+.5f %+.5fi)",
            label, 2pi * δ, N, β, real(found), imag(found), real(exact), imag(exact)))
    end
end

mkpath(joinpath(@__DIR__, "results"))
write(joinpath(@__DIR__, "results", "verify_reference.txt"), String(take!(OUT)))
