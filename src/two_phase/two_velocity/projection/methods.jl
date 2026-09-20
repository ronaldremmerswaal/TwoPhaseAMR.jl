"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select the symmetric mimetic two-velocity pressure projection."""
struct MimeticProjection <: AbstractTwoVelocityProjection end

"""Compatibility alias; `MimeticProjection` is the canonical selector name."""
const MimeticGradient = MimeticProjection

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
