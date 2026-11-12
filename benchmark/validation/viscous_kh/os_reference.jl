# Two-fluid Orr–Sommerfeld reference solver (eigenvalue and initial-value problem).
#
# Linearised incompressible Navier–Stokes about a parallel base flow U_π(y,t) e_x in
# two layers separated by a flat interface at y = 0: liquid (π = l) in −H_l < y < 0,
# gas (π = g) in 0 < y < H_g. A single Fourier mode exp(ikx) of the stream function
# ψ = φ_π(y,t) exp(ikx) (u = ∂ψ/∂y, v = −∂ψ/∂x) and of the interface displacement
# η(t) exp(ikx) satisfies (Boomkamp et al. 1997, Bagué et al. 2010)
#
#   ∂t (D² − k²)φ = −ik U (D² − k²)φ + ik U'' φ + ν (D² − k²)² φ        in each layer,
#
# with interface conditions linearised at y = 0
#   φ_l = φ_g                                      (normal velocity)
#   Dφ_l + U_l' η = Dφ_g + U_g' η                  (tangential velocity)
#   μ_l (D²φ_l + k²φ_l + U_l'' η) = μ_g (…)_g      (tangential stress)
#   S_g − S_l + ik g (ρ_g − ρ_l) η − ik σ k² η = 0  (normal stress), with
#   S_π = ρ_π ∂t Dφ_π + ik ρ_π (U_π Dφ_π − U_π' φ_π) − μ_π (D³ − 3k² D) φ_π
#   ∂t η + ik U(0) η = −ik φ(0)                    (kinematic)
# and walls at y = −H_l, H_g: φ = 0 plus D²φ = 0 (free slip) or Dφ = 0 (no slip).
#
# Each layer is discretised by Chebyshev collocation on its own (optionally
# interface-stretched) grid; the boundary and interface conditions replace the two
# collocation rows next to each end. This gives the descriptor system M Ẋ = A(t) X,
# X = [φ_l; φ_g; η]. With a frozen base flow, exp(−iωt) modes give the generalised
# eigenproblem A x = −iω M x; for the diffusing base flow the system is integrated in
# time with the three-stage Radau IIA method (order 5, stiffly accurate, L-stable).

using LinearAlgebra
using SpecialFunctions: erf

"Chebyshev–Gauss–Lobatto nodes x_j = cos(πj/N) and the differentiation matrix (Trefethen 2000)."
function chebyshev(N)
    x = [cos(pi * j / N) for j in 0:N]
    c = [(j == 0 || j == N ? 2.0 : 1.0) * (-1)^j for j in 0:N]
    D = zeros(N + 1, N + 1)
    for i in 1:(N + 1), j in 1:(N + 1)
        i == j || (D[i, j] = c[i] / c[j] / (x[i] - x[j]))
    end
    for i in 1:(N + 1)
        D[i, i] = -sum(D[i, j] for j in 1:(N + 1) if j != i)
    end
    return x, D
end

"""
    Layer(side, N, H; stretch=0, density, viscosity)

One fluid layer: `side = -1` (liquid, −H < y < 0) or `+1` (gas, 0 < y < H). Nodes run
from the wall (index 1) to the interface (index N+1). `stretch = β > 0` uses the map
|y| = H sinh(β(1+ξ)/2)/sinh(β), which refines the node spacing at the interface by
≈ β/sinh(β) (entire map, so spectral convergence is retained); `β = 0` is linear.
"""
struct Layer
    side::Int
    N::Int
    H::Float64
    stretch::Float64
    ξ::Vector{Float64}
    y::Vector{Float64}
    D1::Matrix{Float64}
    D2::Matrix{Float64}
    D3::Matrix{Float64}
    D4::Matrix{Float64}
    density::Float64
    viscosity::Float64
end

_map(ξ, H, β) = iszero(β) ? H * (1 + ξ) / 2 : H * sinh(β * (1 + ξ) / 2) / sinh(β)
_dmap(ξ, H, β) = iszero(β) ? H / 2 : H * β * cosh(β * (1 + ξ) / 2) / (2sinh(β))
_imap(r, H, β) = iszero(β) ? 2r / H - 1 : 2asinh(r * sinh(β) / H) / β - 1

function Layer(side, N, H; stretch = 0.0, density, viscosity)
    ξ, Dξ = chebyshev(N)
    r = _map.(ξ, H, stretch)
    # Derivatives of ξ(r) for the chain rule (Faà di Bruno), applied to the exact
    # polynomial derivatives Dξ^m; composing mapped first-derivative matrices instead
    # produces spurious growing eigenvalues ∝ ν.
    q = if iszero(stretch)
        (fill(2 / H, N + 1), zeros(N + 1), zeros(N + 1), zeros(N + 1))
    else
        s, β = sinh(stretch), stretch
        a, c = 2s / (β * H), s / H
        z = r .* c
        w = 1 .+ z .^ 2
        (a ./ sqrt.(w), -a * c .* z ./ w .^ 1.5, a * c^2 .* (2z .^ 2 .- 1) ./ w .^ 2.5,
            a * c^3 .* (9z .- 6z .^ 3) ./ w .^ 3.5)
    end
    q1, q2, q3, q4 = q
    Dξ2 = Dξ * Dξ
    Dξ3 = Dξ2 * Dξ
    Dξ4 = Dξ3 * Dξ
    D1 = Diagonal(side .* q1) * Dξ
    D2 = Diagonal(q1 .^ 2) * Dξ2 + Diagonal(q2) * Dξ
    D3 = side .* (Diagonal(q1 .^ 3) * Dξ3 + Diagonal(3q1 .* q2) * Dξ2 + Diagonal(q3) * Dξ)
    D4 = Diagonal(q1 .^ 4) * Dξ4 + Diagonal(6q1 .^ 2 .* q2) * Dξ3 +
        Diagonal(3q2 .^ 2 .+ 4q1 .* q3) * Dξ2 + Diagonal(q4) * Dξ
    return Layer(side, N, H, stretch, ξ, side .* r, D1, D2, D3, D4, density, viscosity)
end

kinematic_viscosity(layer::Layer) = layer.viscosity / layer.density

"""
    ErfBaseFlow(; U∞=(U_l, U_g), δ0=(δ_l, δ_g), ν=(ν_l, ν_g), diffusing=true, U_i=0)

Composite error-function base flow U_π(y,t) = U_i + U∞_π erf(y/δ_π(t)) with
δ_π(t) = sqrt(δ0_π² + 4 ν_π t) (diffusing) or δ_π = δ0_π (frozen). With
√(ρ_l μ_l) U∞_l = √(ρ_g μ_g) U∞_g and δ0_g/δ0_l = √(ν_g/ν_l) the diffusing profile is
an exact solution of the two-layer diffusion problem in an unbounded domain (both
tangential velocity and stress continuous at y = 0). U∞_l > 0 means the liquid moves
in −x far below the interface.
"""
Base.@kwdef struct ErfBaseFlow
    U∞::NTuple{2, Float64}
    δ0::NTuple{2, Float64}
    ν::NTuple{2, Float64}
    diffusing::Bool = true
    U_i::Float64 = 0.0
end

thickness(b::ErfBaseFlow, phase, t) = b.diffusing ? sqrt(b.δ0[phase]^2 + 4b.ν[phase] * t) : b.δ0[phase]
_phase(side) = side < 0 ? 1 : 2

"Base velocity and its first two derivatives at `y` (evaluated with the formula of the layer on `side`)."
function base_profile(b::ErfBaseFlow, side, y, t)
    p = _phase(side)
    δ = thickness(b, p, t)
    U = b.U_i + b.U∞[p] * erf(y / δ)
    U1 = b.U∞[p] * 2 / sqrt(pi) / δ * exp(-(y / δ)^2)
    return U, U1, -2y / δ^2 * U1
end

"Generic base flow from a function `(side, y, t) -> (U, U', U'')`."
struct FunctionBaseFlow{F}
    f::F
end
base_profile(b::FunctionBaseFlow, side, y, t) = b.f(side, y, t)

"""
    TwoLayerProblem(liquid, gas; wavenumber, surface_tension=0, gravity=0, wall=:free_slip)

Discretised two-layer linear stability problem for one wavenumber.
"""
Base.@kwdef struct TwoLayerProblem
    liquid::Layer
    gas::Layer
    wavenumber::Float64
    surface_tension::Float64 = 0.0
    gravity::Float64 = 0.0
    wall::Symbol = :free_slip
end

nunknowns(p::TwoLayerProblem) = p.liquid.N + p.gas.N + 3

"Descriptor matrices (M, A) of M Ẋ = A X at time `t` for base flow `base`."
function descriptor(p::TwoLayerProblem, base, t)
    n = nunknowns(p)
    M = zeros(ComplexF64, n, n)
    A = zeros(ComplexF64, n, n)
    descriptor!(M, A, p, base, t)
    return M, A
end

function descriptor!(M, A, p::TwoLayerProblem, base, t)
    fill!(M, 0)
    fill!(A, 0)
    k = p.wavenumber
    ik = im * k
    n = nunknowns(p)
    l, g = p.liquid, p.gas
    offsets = (0, l.N + 1)
    ends = Vector{NTuple{3, Float64}}(undef, 2)   # base (U, U', U'') at y = 0 per layer
    for (layer, o) in zip((l, g), offsets)
        np = layer.N + 1
        cols = (o + 1):(o + np)
        ν = kinematic_viscosity(layer)
        L = layer.D2 - k^2 * I
        L2 = layer.D4 - 2k^2 * layer.D2 + k^4 * I
        for j in 3:(np - 2)
            U, _, U2 = base_profile(base, layer.side, layer.y[j], t)
            M[o + j, cols] .= L[j, :]
            A[o + j, cols] .= -ik * U .* L[j, :] .+ ν .* L2[j, :]
            A[o + j, o + j] += ik * U2
        end
        A[o + 1, o + 1] = 1                                            # φ = 0 at the wall
        A[o + 2, cols] .= p.wall === :free_slip ? layer.D2[1, :] :     # D²φ = 0 (free slip)
            p.wall === :no_slip ? layer.D1[1, :] :                     # Dφ = 0 (no slip)
            throw(ArgumentError("wall must be :free_slip or :no_slip"))
        ends[_phase(layer.side)] = base_profile(base, layer.side, 0.0, t)
    end
    nl, ng = l.N + 1, g.N + 1
    lc, gc = 1:nl, (nl + 1):(nl + ng)
    li, gi = nl, nl + ng                                               # interface nodes
    Ul, Ul1, Ul2 = ends[1]
    Ug, Ug1, Ug2 = ends[2]
    # liquid block: continuity of normal and tangential velocity
    r = nl - 1
    A[r, li] = 1
    A[r, gi] = -1
    r = nl
    A[r, lc] .= l.D1[end, :]
    A[r, gc] .= -g.D1[end, :]
    A[r, n] = Ul1 - Ug1
    # gas block: tangential and normal stress
    r = nl + ng - 1
    A[r, lc] .= l.viscosity .* l.D2[end, :]
    A[r, li] += l.viscosity * k^2
    A[r, gc] .= -g.viscosity .* g.D2[end, :]
    A[r, gi] -= g.viscosity * k^2
    A[r, n] = l.viscosity * Ul2 - g.viscosity * Ug2
    r = nl + ng
    M[r, gc] .= g.density .* g.D1[end, :]
    M[r, lc] .= -l.density .* l.D1[end, :]
    A[r, gc] .= -(ik * g.density * Ug .* g.D1[end, :] .- g.viscosity .* g.D3[end, :] .+ 3g.viscosity * k^2 .* g.D1[end, :])
    A[r, gi] += ik * g.density * Ug1
    A[r, lc] .= ik * l.density * Ul .* l.D1[end, :] .- l.viscosity .* l.D3[end, :] .+ 3l.viscosity * k^2 .* l.D1[end, :]
    A[r, li] -= ik * l.density * Ul1
    A[r, n] = -ik * p.gravity * (g.density - l.density) + ik * p.surface_tension * k^2
    # kinematic condition
    M[n, n] = 1
    A[n, n] = -ik * Ug
    A[n, gi] = -ik
    return M, A
end

"""
    eigenmodes(p, base; t=0, cap=1e4) -> (ω, X)

All finite eigenpairs with |ω| < `cap` of the frozen problem at time `t`, sorted by
decreasing growth rate Im ω (mode ∝ exp(i(kx − ωt))). The cap removes the spurious
O(N⁴) eigenvalues of the collocation discretisation. Eigenvectors are normalised to
η = 1 when possible.
"""
function eigenmodes(p::TwoLayerProblem, base; t = 0.0, cap = 1.0e4)
    M, A = descriptor(p, base, t)
    F = eigen(A, M)
    λ = F.values
    keep = findall(z -> isfinite(z) && abs(z) < cap, λ)
    ω = im .* λ[keep]
    X = F.vectors[:, keep]
    for j in axes(X, 2)
        η = X[end, j]
        abs(η) > 1.0e-12 * norm(X[:, j]) && (X[:, j] ./= η)
    end
    order = sortperm(imag.(ω); rev = true)
    return ω[order], X[:, order]
end

"""
Most unstable eigenpair (ω, X) of the frozen problem. Of a pair of (nearly) equally
unstable modes travelling in opposite directions, the one with Re ω ≥ 0 is returned.
"""
function leading_mode(p::TwoLayerProblem, base; t = 0.0, cap = 1.0e4)
    ω, X = eigenmodes(p, base; t, cap)
    j = length(ω) > 1 && imag(ω[1]) - imag(ω[2]) < 1.0e-3 * abs(imag(ω[1])) &&
        real(ω[2]) > real(ω[1]) ? 2 : 1
    return ω[j], X[:, j]
end

const RADAU_C = ((4 - sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1.0)
const RADAU_A = [
    (88 - 7sqrt(6)) / 360 (296 - 169sqrt(6)) / 1800 (-2 + 3sqrt(6)) / 225
    (296 + 169sqrt(6)) / 1800 (88 + 7sqrt(6)) / 360 (-2 - 3sqrt(6)) / 225
    (16 - sqrt(6)) / 36 (16 + sqrt(6)) / 36 1 / 9
]

"""
    integrate(p, base, X0, times; dt) -> Matrix (one column per entry of `times`)

Integrate M(t) Ẋ = A(t) X from X(0) = X0 with three-stage Radau IIA and return the
state at the requested `times` (sorted, ≥ 0). M does not depend on t here.
"""
function integrate(p::TwoLayerProblem, base, X0, times; dt)
    n = nunknowns(p)
    out = zeros(ComplexF64, n, length(times))
    X = copy(X0)
    M = zeros(ComplexF64, n, n)
    Aj = [zeros(ComplexF64, n, n) for _ in 1:3]
    S = zeros(ComplexF64, 3n, 3n)
    rhs = zeros(ComplexF64, 3n)
    t = 0.0
    for (m, target) in enumerate(times)
        while t < target - 1.0e-12 * max(1, target)
            h = min(dt, target - t)
            for j in 1:3
                descriptor!(M, Aj[j], p, base, t + RADAU_C[j] * h)
            end
            for i in 1:3
                rows = ((i - 1) * n + 1):(i * n)
                for j in 1:3
                    cols = ((j - 1) * n + 1):(j * n)
                    S[rows, cols] .= -h * RADAU_A[i, j] .* Aj[j]
                    i == j && (S[rows, cols] .+= M)
                end
                rhs[rows] .= M * X
            end
            Y = S \ rhs
            X .= Y[(2n + 1):(3n)]
            t += h
        end
        out[:, m] .= X
    end
    return out
end

"Barycentric Chebyshev–Lobatto interpolation of nodal values `f` at ξ ∈ [−1, 1]."
function chebinterp(ξn, f, ξ)
    N = length(ξn) - 1
    num = zero(eltype(f))
    den = 0.0
    for j in 0:N
        w = (j == 0 || j == N ? 0.5 : 1.0) * (-1)^j
        d = ξ - ξn[j + 1]
        iszero(d) && return f[j + 1]
        num += w / d * f[j + 1]
        den += w / d
    end
    return num / den
end

"Indices of layer `phase` (1 liquid, 2 gas) in the state vector."
layer_range(p::TwoLayerProblem, phase) = phase == 1 ? (1:(p.liquid.N + 1)) :
    ((p.liquid.N + 2):(p.liquid.N + p.gas.N + 2))
layer(p::TwoLayerProblem, phase) = phase == 1 ? p.liquid : p.gas

"""
    perturbation(p, X, phase, y) -> (φ, Dφ, D²φ)

Stream-function amplitude of layer `phase` and its derivatives at `y`. Inside the
layer this is Chebyshev interpolation; on the far side of the interface (|y| small,
used when the perturbed interface crosses y = 0) a cubic Taylor extension from y = 0.
"""
function perturbation(p::TwoLayerProblem, X, phase, y)
    lay = layer(p, phase)
    φ = X[layer_range(p, phase)]
    if lay.side * y >= 0
        r = min(abs(y), lay.H)
        ξ = _imap(r, lay.H, lay.stretch)
        return chebinterp(lay.ξ, φ, ξ), chebinterp(lay.ξ, lay.D1 * φ, ξ),
            chebinterp(lay.ξ, lay.D2 * φ, ξ)
    end
    d = [φ[end], (lay.D1 * φ)[end], (lay.D2 * φ)[end], (lay.D3 * φ)[end], (lay.D4 * φ)[end]]
    return d[1] + y * d[2] + y^2 / 2 * d[3] + y^3 / 6 * d[4],
        d[2] + y * d[3] + y^2 / 2 * d[4],
        d[3] + y * d[4]
end
