# Types and policies owned by the intrinsic two-velocity transport model.
"""
    TwoPhaseVelocity

A liquid/gas pair of existing [`FaceVelocity`](@ref) fields, stored as the
ordinary tuple `(liquid_velocity, gas_velocity)`. The alias adds dispatch and
documentation without introducing another field container or duplicating
velocity storage.

`FlowSimulation` and two-velocity updates store NaN for an absent phase
(exactly zero open staggered phase volume). Stencil extensions are computed
from valid phase values in working storage only. Phase-weighted operations
must exclude absent values before arithmetic, not multiply them by zero.
"""
const TwoPhaseVelocity = Tuple{<:FaceVelocity, <:FaceVelocity}

"""
    ProjectedLiquidVelocity()

Two-velocity interface-advection policy which extends and discretely projects
the liquid velocity before transporting the physical liquid interface. The gas
velocity advances a separate auxiliary gas carrier.
"""
struct ProjectedLiquidVelocity{T} <: AbstractInterfaceTransportMethod
    extension_layers::Int
    tolerance::T
    max_cycles::Int

    function ProjectedLiquidVelocity(extension_layers, tolerance, max_cycles)
        extension_layers >= 1 || throw(
            ArgumentError("extension_layers must be positive"),
        )
        isfinite(tolerance) && tolerance > 0 || throw(
            ArgumentError("projection tolerance must be finite and positive"),
        )
        max_cycles >= 1 || throw(
            ArgumentError("projection max_cycles must be positive"),
        )
        converted_tolerance = float(tolerance)
        return new{typeof(converted_tolerance)}(
            Int(extension_layers), converted_tolerance, Int(max_cycles),
        )
    end
end

function ProjectedLiquidVelocity(;
        extension_layers = 3, tolerance = sqrt(eps(Float64)),
        max_cycles = 200,
    )
    return ProjectedLiquidVelocity(extension_layers, tolerance, max_cycles)
end

"""
    FrozenInterfaceFluxes()

Two-velocity interface-advection policy which freezes the time-step-start
liquid/gas face apertures and extrudes each subface with its own phase
velocity. With a solid, subface ordering is frozen as well.
"""
struct FrozenInterfaceFluxes <: AbstractInterfaceTransportMethod end

const _TwoVelocityInterfaceAdvection = Union{
    ProjectedLiquidVelocity, FrozenInterfaceFluxes,
}

_select_interface_transport(interface_transport, default) =
    isnothing(interface_transport) ? default : interface_transport
