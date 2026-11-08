"""
    SurfaceTension(coefficient=0; method=CGFM())

Surface-tension coefficient and its discrete force method.
"""
struct SurfaceTension{T <: Real, M <: AbstractSurfaceTensionMethod}
    coefficient::T
    method::M
end

function SurfaceTension(coefficient = 0.0; method = CGFM())
    coefficient isa Real || throw(
        ArgumentError("surface-tension coefficient must be real"),
    )
    isfinite(coefficient) && coefficient >= 0 || throw(
        ArgumentError(
            "surface-tension coefficient must be finite and nonnegative",
        ),
    )
    method isa AbstractSurfaceTensionMethod || throw(
        ArgumentError("surface-tension method must be CGFM() or CSF()"),
    )
    return SurfaceTension(coefficient, method)
end

"""
    TwoPhase(interface; density, dynamic_viscosity,
             surface_tension=SurfaceTension(), gravity=(0, 0), scalars=())

Physical properties and phase fields shared by the one- and two-velocity
two-phase models. A nonnegative numeric `surface_tension` is shorthand for
`SurfaceTension(value; method=CGFM())`.
"""
struct TwoPhase{
        I <: InterfaceField, R <: Tuple, M, S <: SurfaceTension,
        G <: Tuple, Q <: Tuple,
    }
    interface::I
    density::R
    dynamic_viscosity::M
    surface_tension::S
    gravity::G
    scalars::Q
end

function TwoPhase(
        interface::InterfaceField; density, dynamic_viscosity,
        surface_tension = SurfaceTension(), gravity = (0.0, 0.0),
        scalars = (),
    )
    density isa NTuple{2} || throw(
        ArgumentError("density must be a (liquid, gas) tuple"),
    )
    viscosity = dynamic_viscosity isa Number ?
        (dynamic_viscosity, dynamic_viscosity) : dynamic_viscosity
    viscosity isa NTuple{2} || throw(
        ArgumentError(
            "dynamic_viscosity must be a (liquid, gas) tuple or scalar",
        ),
    )
    tension = surface_tension isa Real ?
        SurfaceTension(surface_tension) : surface_tension
    tension isa SurfaceTension || throw(
        ArgumentError(
            "surface_tension must be a nonnegative number or " *
                "SurfaceTension(...)"
        ),
    )
    gravity isa NTuple{2} || throw(
        ArgumentError("gravity must be a two-tuple"),
    )
    phase_scalars = _configuration_tuple(scalars)
    all(rho -> rho isa Real && isfinite(rho) && rho > 0, density) || throw(
        ArgumentError("phase densities must be finite and positive"),
    )
    all(mu -> mu isa Real && isfinite(mu) && mu >= 0, viscosity) || throw(
        ArgumentError("phase dynamic viscosities must be finite and nonnegative"),
    )
    all(iszero, viscosity) || all(>(0), viscosity) || throw(
        ArgumentError("phase dynamic viscosities must both be positive or both zero"),
    )
    all(value -> value isa Real && isfinite(value), gravity) || throw(
        ArgumentError("gravity must contain finite real values"),
    )
    all(
        scalar -> scalar isa PhaseScalar && scalar.interface === interface,
        phase_scalars,
    ) || throw(
        ArgumentError("scalars must belong to the supplied interface"),
    )
    return TwoPhase(
        interface, density, viscosity, tension, gravity, phase_scalars,
    )
end
