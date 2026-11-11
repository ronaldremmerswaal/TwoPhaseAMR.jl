"""
    TwoVelocityModel(; pressure_projection=CMDGFMProjection())

Select separate intrinsic phase velocities and configure their
pressure-projection method.
"""
struct TwoVelocityModel{P <: AbstractTwoVelocityProjection} <: AbstractVelocityModel
    pressure_projection::P

    function TwoVelocityModel(pressure_projection)
        pressure_projection isa AbstractTwoVelocityProjection || throw(
            ArgumentError(
                "pressure_projection must be a CMDGFMProjection()",
            ),
        )
        return new{typeof(pressure_projection)}(pressure_projection)
    end
end

function TwoVelocityModel(; pressure_projection = CMDGFMProjection())
    return TwoVelocityModel(pressure_projection)
end
