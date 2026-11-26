using LinearAlgebra: eigvals
using SpecialFunctions: erfc

function _quartic_roots(coefficients::NTuple{5, Float64})
    coefficients[1] == 1 ||
        throw(ArgumentError("the quartic must be monic"))
    companion = zeros(ComplexF64, 4, 4)
    companion[2, 1] = companion[3, 2] = companion[4, 3] = 1
    companion[:, 4] .= -reverse(collect(coefficients[2:end]))
    return eigvals(companion)
end

"""Prosperetti's equal-kinematic-viscosity linear amplitude solution.

This is the formula reproduced in the thesis, with the polynomial coefficient
`-4beta*sqrt(nu*k^2)` used by its reference implementation. The returned
quantity is normalized by the initial amplitude.
"""
function prosperetti_amplitude(
        time; liquid_density, gas_density, kinematic_viscosity,
        gravity, surface_tension, wavenumber = 2pi
    )
    rho_l, rho_g = liquid_density, gas_density
    nu, k = kinematic_viscosity, wavenumber
    omega0_squared = (
        gravity * k * (rho_l - rho_g) + surface_tension * k^3
    ) / (rho_l + rho_g)
    omega_nu = nu * k^2
    beta = rho_l * rho_g / (rho_l + rho_g)^2
    roots = _quartic_roots(
        (
            1.0,
            -4beta * sqrt(omega_nu),
            2(1 - 6beta) * omega_nu,
            4(1 - 3beta) * omega_nu^(3 / 2),
            (1 - 4beta) * omega_nu^2 + omega0_squared,
        )
    )
    products = [
        prod(roots[j] - roots[i] for j in eachindex(roots) if i != j)
            for i in eachindex(roots)
    ]
    result = 4(1 - 4beta) * omega_nu^2 /
        (8(1 - 4beta) * omega_nu^2 + omega0_squared) *
        erfc(sqrt(omega_nu * time))
    for i in eachindex(roots)
        shifted = roots[i]^2 - omega_nu
        result += roots[i] / products[i] * omega0_squared / shifted *
            exp(shifted * time) * erfc(roots[i] * sqrt(time))
    end
    return real(result)
end
