# Types and policies owned by the intrinsic two-velocity transport model.
"""
    TwoPhaseVelocity

A liquid/gas pair of existing [`FaceVelocity`](@ref) fields, stored as the
ordinary tuple `(liquid_velocity, gas_velocity)`. The alias adds dispatch and
documentation without introducing another field container or duplicating
velocity storage.
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
end

function ProjectedLiquidVelocity(;
        extension_layers = 3, tolerance = sqrt(eps(Float64)),
        max_cycles = 200,
    )
    extension_layers >= 1 || throw(
        ArgumentError(
            "extension_layers must be positive",
        )
    )
    isfinite(tolerance) && tolerance > 0 || throw(
        ArgumentError(
            "projection tolerance must be finite and positive",
        )
    )
    max_cycles >= 1 || throw(
        ArgumentError(
            "projection max_cycles must be positive",
        )
    )
    return ProjectedLiquidVelocity(
        Int(extension_layers), float(tolerance), Int(max_cycles),
    )
end

"""
    FrozenInterfaceFluxes()

Two-velocity interface-advection policy which freezes the time-step-start
liquid/gas subdivision of every open face and extrudes each subface with its
own phase velocity.
"""
struct FrozenInterfaceFluxes <: AbstractInterfaceTransportMethod end

const _TwoVelocityInterfaceAdvection = Union{
    ProjectedLiquidVelocity, FrozenInterfaceFluxes,
}

function _select_interface_transport(
        interface_transport, interface_method, default,
    )
    !isnothing(interface_transport) && !isnothing(interface_method) && throw(
        ArgumentError(
            "pass interface_transport, not both interface_transport and " *
                "the compatibility keyword interface_method",
        ),
    )
    return isnothing(interface_transport) ?
        (isnothing(interface_method) ? default : interface_method) :
        interface_transport
end
