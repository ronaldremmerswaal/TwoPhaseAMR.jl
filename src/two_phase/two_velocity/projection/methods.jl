"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select the symmetric mimetic two-velocity pressure projection."""
struct MimeticProjection <: AbstractTwoVelocityProjection end

"""Compatibility alias; `MimeticProjection` is the canonical selector name."""
const MimeticGradient = MimeticProjection

"""Select the symmetric projection with local aperture-normal constraints.

With the default `constraint_location=:cell`, the rows use the unnormalised
centred vector `D(a_l*n)` contracted with a staggered-to-centre jump
interpolant. `constraint_location=:face` retains the earlier face-local
variant. The default `interpolation_weights=:c1mc` weights contributing faces
by `c*(1-c)` (and face measure for cell-centred interpolation), so their
contribution vanishes continuously when the moving interface leaves a face.
Use `:geometric` for the purely geometric comparison interpolant. Vanishing
phase masses are removed from the jump space and dependent local rows are
discarded before the symmetric KKT system is assembled.
"""
struct LocalMimeticProjection{P, O, A, R, L, W} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
    active_mass_fraction::A
    rank_tolerance::R
    constraint_location::L
    interpolation_weights::W
end

LocalMimeticProjection(;
    preconditioner = :cholesky, preconditioner_options = (;),
    active_mass_fraction = nothing, rank_tolerance = nothing,
    constraint_location = :cell,
    interpolation_weights = :c1mc,
) = LocalMimeticProjection(
    preconditioner, preconditioner_options, active_mass_fraction,
    rank_tolerance, constraint_location, interpolation_weights,
)

"""Convenience constructor for `LocalMimeticProjection(constraint_location=:cell)`."""
CellLocalMimeticProjection(; options...) =
    LocalMimeticProjection(; options..., constraint_location = :cell)

"""Convenience constructor for `LocalMimeticProjection(constraint_location=:face)`."""
FaceLocalMimeticProjection(; options...) =
    LocalMimeticProjection(; options..., constraint_location = :face)

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
