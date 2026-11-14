"""
    TwoVelocityModel(; pressure_projection=CMDGFMProjection(),
                     interphase_drag=InterphaseDrag())

Select separate intrinsic phase velocities and configure their
pressure-projection method and their viscous coupling across the interface
([`InterphaseDrag`](@ref), or `nothing` for none). The drag is independent of
the projection and acts only with nonzero viscosities.
"""
struct TwoVelocityModel{
        P <: AbstractTwoVelocityProjection, D <: Union{Nothing, InterphaseDrag},
    } <: AbstractVelocityModel
    pressure_projection::P
    interphase_drag::D

    function TwoVelocityModel(pressure_projection, interphase_drag = InterphaseDrag())
        pressure_projection isa AbstractTwoVelocityProjection || throw(
            ArgumentError(
                "pressure_projection must be a CMDGFMProjection()",
            ),
        )
        _check_interphase_drag(interphase_drag)
        return new{typeof(pressure_projection), typeof(interphase_drag)}(
            pressure_projection, interphase_drag,
        )
    end
end

function TwoVelocityModel(;
        pressure_projection = CMDGFMProjection(),
        interphase_drag = InterphaseDrag(),
    )
    return TwoVelocityModel(pressure_projection, interphase_drag)
end

_check_interphase_drag(drag) = drag isa Union{Nothing, InterphaseDrag} || throw(
    ArgumentError("interphase_drag must be an InterphaseDrag() or nothing"),
)
