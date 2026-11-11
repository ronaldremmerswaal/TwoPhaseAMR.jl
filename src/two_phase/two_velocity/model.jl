"""
    TwoVelocityModel(; interface_transport=FrozenInterfaceFluxes(),
                     pressure_projection=CMDGFMProjection())

Select separate intrinsic phase velocities and configure their interface
transport and pressure-projection methods.
"""
struct TwoVelocityModel{
        I <: AbstractInterfaceTransportMethod,
        P <: AbstractTwoVelocityProjection,
    } <: AbstractVelocityModel
    interface_transport::I
    pressure_projection::P

    function TwoVelocityModel(interface_transport, pressure_projection)
        interface_transport isa _TwoVelocityInterfaceAdvection || throw(
            ArgumentError(
                "interface_transport must be FrozenInterfaceFluxes() or " *
                    "ProjectedLiquidVelocity()",
            ),
        )
        pressure_projection isa AbstractTwoVelocityProjection || throw(
            ArgumentError(
                "pressure_projection must be a CMDGFMProjection()",
            ),
        )
        return new{typeof(interface_transport), typeof(pressure_projection)}(
            interface_transport, pressure_projection,
        )
    end
end

function TwoVelocityModel(;
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    )
    return TwoVelocityModel(interface_transport, pressure_projection)
end
