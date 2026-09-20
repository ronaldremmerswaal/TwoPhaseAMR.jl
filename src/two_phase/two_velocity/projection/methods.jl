"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select an energy-preserving mimetic two-velocity projection.

`jump_interpolant` selects the original centred aperture-divergence model
(`:divergence`), the face-local aperture-normal model (`:face`), or the
cell-local aperture-normal model (`:cell`). The local models default to
`interpolation_weights=:c1mc`; `:geometric` selects the comparison weights.
All variants share one active set, rank-safe symmetric KKT projection, and
phase kinetic-energy inner product.
"""
struct MimeticProjection{J, P, O, A, R, W} <: AbstractTwoVelocityProjection
    jump_interpolant::J
    preconditioner::P
    preconditioner_options::O
    active_mass_fraction::A
    rank_tolerance::R
    interpolation_weights::W
end

MimeticProjection(;
    jump_interpolant = :divergence,
    preconditioner = :cholesky, preconditioner_options = (;),
    active_mass_fraction = nothing, rank_tolerance = nothing,
    interpolation_weights = :c1mc,
) = MimeticProjection(
    jump_interpolant, preconditioner, preconditioner_options,
    active_mass_fraction, rank_tolerance, interpolation_weights,
)

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
