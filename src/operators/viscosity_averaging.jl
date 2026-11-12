"""Abstract supertype of a phase-viscosity averaging rule for the two-phase
stress operator `div(2 mu D(u))`.

The rule averages the phase viscosities at two places: in every cell, from the
conditional liquid fraction (the normal-stress and wall rows), and at every
shear-stress vertex, from the cell quarters that the vertex row spans. Both
velocity models share the operator (the two-velocity step diffuses the
mass-weighted mixture velocity with it), so one rule covers both. See
[`HarmonicViscosityAveraging`](@ref) and [`GeometricViscosityAveraging`](@ref).
"""
abstract type AbstractViscosityAveraging end

"""
    HarmonicViscosityAveraging()

Average phase viscosities by their series (harmonic) conductance.

In a cell of conditional liquid fraction `c` the viscosity is
`1 / (c / mu_liquid + (1 - c) / mu_gas)`, and at a shear-stress vertex that
spans several cell quarters with weights `w`, summing to one, it is
`1 / sum(w / mu_cell)`. This is the coefficient that keeps the tangential
traction continuous across an interface aligned with the shear, where the
centred difference puts the two phases in series; the geometric mean
overestimates that flux by a factor `2 sqrt(mu1 mu2) / (mu1 + mu2)` (6.07 % for
`mu = (2, 1)`). It is the default; the thesis uses the geometric mean
(`GeometricViscosityAveraging`, see the deviation in the technical reference).
"""
struct HarmonicViscosityAveraging <: AbstractViscosityAveraging end

"""
    GeometricViscosityAveraging()

Average phase viscosities by their phase-fraction-weighted geometric mean,
`mu_liquid^c mu_gas^(1-c)` in a cell of conditional liquid fraction `c` and
`prod(mu_cell^w)` at a shear-stress vertex. This is the diffuse model of the
thesis, which smears the viscosity jump over the interface cells; it is kept
for reproducing its results. It gives first-order shear across an aligned
viscosity jump, because the vertex coefficient is not the series conductance
the centred difference needs.
"""
struct GeometricViscosityAveraging <: AbstractViscosityAveraging end

"""One phase-viscosity mixture in a cell, or `1 / mu` of it, from the
conditional liquid fraction."""
@inline function _phase_dynamic_viscosity(
        c, viscosity::NTuple{2}, ::HarmonicViscosityAveraging,
    )
    liquid, gas = viscosity
    return inv(muladd(c, inv(liquid) - inv(gas), inv(gas)))
end

@inline function _phase_dynamic_viscosity(
        c, viscosity::NTuple{2}, ::GeometricViscosityAveraging,
    )
    liquid, gas = viscosity
    return exp(muladd(c, log(liquid) - log(gas), log(gas)))
end
