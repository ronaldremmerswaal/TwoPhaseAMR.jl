"""
    SinglePhase(; density, dynamic_viscosity)

Physical properties for an incompressible single-phase flow. Numerical and
linear-solver choices are supplied separately through [`NavierStokesOptions`](@ref).
"""
struct SinglePhase{R <: Real, M <: Real}
    density::R
    dynamic_viscosity::M
end

function SinglePhase(; density, dynamic_viscosity)
    density isa Real || throw(
        ArgumentError("single-phase density must be real"),
    )
    dynamic_viscosity isa Real || throw(
        ArgumentError("single-phase dynamic_viscosity must be real"),
    )
    isfinite(density) && density > 0 || throw(
        ArgumentError("density must be finite and strictly positive"),
    )
    isfinite(dynamic_viscosity) && dynamic_viscosity > 0 || throw(
        ArgumentError(
            "dynamic_viscosity must be finite and strictly positive",
        ),
    )
    return SinglePhase(density, dynamic_viscosity)
end
