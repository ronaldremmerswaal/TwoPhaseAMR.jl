"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select the symmetric mimetic two-velocity pressure projection."""
struct MimeticProjection <: AbstractTwoVelocityProjection end

"""Compatibility alias; `MimeticProjection` is the canonical selector name."""
const MimeticGradient = MimeticProjection

"""Select the symmetric projection with face-local aperture-normal constraints.

The local rows use the unnormalised vector `I*D(a_l*n)` and therefore retain
the interface-measure scaling supplied by the phase apertures.  Vanishing
phase masses are removed from the jump space and dependent local rows are
discarded before the symmetric KKT system is assembled.
"""
struct LocalMimeticProjection{P, O, A, R} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
    active_mass_fraction::A
    rank_tolerance::R
end

LocalMimeticProjection(;
    preconditioner = :cholesky, preconditioner_options = (;),
    active_mass_fraction = nothing, rank_tolerance = nothing,
) = LocalMimeticProjection(
    preconditioner, preconditioner_options, active_mass_fraction,
    rank_tolerance,
)

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
