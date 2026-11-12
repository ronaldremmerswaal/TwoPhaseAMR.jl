# Physical cases of the viscous Kelvin–Helmholtz study and their linear reference.
#
# Domain (0, λ) × (−H, H), periodic in x, free-slip walls at y = ±H, liquid (phase 1)
# below the interface y = η(x, t), gas (phase 2) above. λ = 1 (k = 2π), H = 1/2,
# ΔU = U∞_l + U∞_g = 1 (liquid moves in −x, gas in +x, interface at rest).
# Base flow: the diffusing composite error-function profile (Rayleigh's impulsively
# started shear layer between two fluids), U_π = U∞_π erf(y/δ_π(t)),
# δ_π(t)² = δ0_π² + 4ν_π t, δ0_l/δ0_g = √(ν_l/ν_g), √(ρ_lμ_l)U∞_l = √(ρ_gμ_g)U∞_g.
# Surface tension from the vortex-sheet Weber number We = ρ_lρ_gΔU²/(kσ(ρ_l+ρ_g)).

include(joinpath(@__DIR__, "os_reference.jl"))

Base.@kwdef struct KHCase
    name::String
    density::NTuple{2, Float64}          # (ρ_l, ρ_g)
    viscosity_ratio::Float64 = 1.0       # ν_g/ν_l
    thickness::Float64                   # δ0_g (gas layer thickness at t = 0)
    reynolds::Float64                    # Re = ΔU δ0_g / ν_g
    weber::Float64 = 2.0
    amplitude::Float64 = 2.0e-4          # initial interface amplitude |η̂(0)|
    final_time::Float64
    height::Float64 = 0.5                # walls at y = ±H
    band::Float64 = 1 / 32               # finest mesh level within |y| ≤ band (kh_run.jl)
    collocation::Tuple{Int, Float64} = (100, 2.0)   # reference (N per layer, stretch β)
end

const WAVELENGTH = 1.0
const WAVENUMBER = 2pi / WAVELENGTH

kinematic_viscosities(c::KHCase) = (c.thickness / c.reynolds / c.viscosity_ratio, c.thickness / c.reynolds)
dynamic_viscosities(c::KHCase) = c.density .* kinematic_viscosities(c)
initial_thicknesses(c::KHCase) = (c.thickness / sqrt(c.viscosity_ratio), c.thickness)

function far_velocities(c::KHCase)
    ρ, μ = c.density, dynamic_viscosities(c)
    w = (sqrt(ρ[1] * μ[1]), sqrt(ρ[2] * μ[2]))   # U∞_l/U∞_g = w_g/w_l
    Ug = 1 / (1 + w[2] / w[1])
    return (1 - Ug, Ug)                            # magnitudes (U∞_l, U∞_g), sum = ΔU = 1
end

surface_tension(c::KHCase) = prod(c.density) / (WAVENUMBER * c.weber * sum(c.density))

base_flow(c::KHCase; diffusing = true) = ErfBaseFlow(;
    U∞ = far_velocities(c), δ0 = initial_thicknesses(c), ν = kinematic_viscosities(c), diffusing,
)

"Velocity-jump (vortex-sheet) dispersion relation with surface tension between walls at ±H."
function vortex_sheet_frequency(c::KHCase)
    k, ρ, H = WAVENUMBER, c.density, c.height
    U = (-far_velocities(c)[1], far_velocities(c)[2])
    a = sum(ρ)
    b = -2k * (ρ[1] * U[1] + ρ[2] * U[2])
    cc = k^2 * (ρ[1] * U[1]^2 + ρ[2] * U[2]^2) - surface_tension(c) * k^3 * tanh(k * H)
    d = sqrt(complex(b^2 - 4a * cc))
    return (-b + d) / 2a
end

"Two-layer Orr–Sommerfeld problem for the case (`N` nodes per layer, stretch `β`)."
function reference_problem(c::KHCase; N = 100, stretch = 0.0)
    ρ, μ = c.density, dynamic_viscosities(c)
    return TwoLayerProblem(;
        liquid = Layer(-1, N, c.height; stretch, density = ρ[1], viscosity = μ[1]),
        gas = Layer(1, N, c.height; stretch, density = ρ[2], viscosity = μ[2]),
        wavenumber = WAVENUMBER, surface_tension = surface_tension(c), wall = :free_slip,
    )
end

"""
    initial_mode(c; N, stretch) -> (problem, ω0, X0)

Most unstable eigenmode of the frozen base flow at t = 0, scaled to η̂(0) = amplitude.
"""
function initial_mode(c::KHCase; N = c.collocation[1], stretch = c.collocation[2])
    p = reference_problem(c; N, stretch)
    ω, X = leading_mode(p, base_flow(c); t = 0.0)
    return p, ω, X .* c.amplitude
end

"""
    reference_history(c; times, N, stretch, dt) -> (; times, eta, states, problem, ω0)

Linear initial-value reference: the t = 0 eigenmode evolved with the diffusing base
flow. `eta[m]` is the complex interface Fourier coefficient η̂(t_m) (η = Re(η̂ e^{ikx})).
"""
function reference_history(c::KHCase; times, N = c.collocation[1], stretch = c.collocation[2], dt = 2.5e-3)
    p, ω0, X0 = initial_mode(c; N, stretch)
    states = integrate(p, base_flow(c), X0, times; dt)
    return (; times, eta = states[end, :], states, problem = p, ω0)
end

# Collocation settings from convergence checks (results/reference_summary.txt).
const CASES = Dict(
    "r1_thin" => KHCase(; name = "r1_thin", density = (1.0, 1.0), thickness = 1 / 128, reynolds = 1000.0, final_time = 1.5, collocation = (100, 2.0)),
    "r1_thick" => KHCase(; name = "r1_thick", density = (1.0, 1.0), thickness = 1 / 32, reynolds = 100.0, final_time = 3.0, band = 1 / 8, collocation = (80, 0.0)),
    "r10_thin" => KHCase(; name = "r10_thin", density = (1.0, 0.1), viscosity_ratio = 4.0, thickness = 1 / 128, reynolds = 1000.0, final_time = 2.5, collocation = (120, 3.0)),
    "r10_thick" => KHCase(; name = "r10_thick", density = (1.0, 0.1), viscosity_ratio = 4.0, thickness = 1 / 32, reynolds = 100.0, final_time = 4.0, band = 1 / 8, collocation = (80, 0.0)),
)
