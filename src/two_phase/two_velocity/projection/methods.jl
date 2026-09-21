"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select an energy-preserving mimetic two-velocity projection.

`jump_interpolant` selects the original centred aperture-divergence model
(`:divergence`), the face-local aperture-normal model (`:face`), or the
cell-local aperture-normal model (`:cell`). `:commuting` first modifies the
geometric tangential observation `T` so its response to the production
pressure block is first-order accurate while reproducing affine velocity
fields. It then constructs `Q=N'` in the kernel of `T*inv(K_xi)`, subject to
exact aperture-normal moments. A patch grows through the connected interface
component only when a smaller patch cannot meet the pressure or multiplier
certificate. It always uses geometric face weights; `c(1-c)` weights belong
only to the legacy `:cell` and `:face` interpolants. All variants share one
active set, rank-safe symmetric KKT projection, and phase kinetic-energy inner
product.
"""
struct MimeticProjection{J, P, O, A, R, W, C, M, L, F, E} <:
        AbstractTwoVelocityProjection
    jump_interpolant::J
    preconditioner::P
    preconditioner_options::O
    active_mass_fraction::A
    rank_tolerance::R
    interpolation_weights::W
    commuting_patch_radius::C
    commuting_max_patch_radius::M
    commuting_amplification_limit::L
    commuting_accuracy_factor::F
    commuting_sliver_epsilon::E
end

MimeticProjection(;
    jump_interpolant = :divergence,
    preconditioner = :cholesky, preconditioner_options = (;),
    active_mass_fraction = nothing, rank_tolerance = nothing,
    interpolation_weights = nothing,
    commuting_patch_radius = 2, commuting_max_patch_radius = nothing,
    commuting_amplification_limit = 50.0,
    commuting_accuracy_factor = 1.0,
    commuting_sliver_epsilon = 0.05,
) = begin
    weights = isnothing(interpolation_weights) ?
        (jump_interpolant === :commuting ? :geometric : :c1mc) :
        interpolation_weights
    jump_interpolant === :commuting && weights !== :geometric && throw(
        ArgumentError(
            "the commuting interpolant is defined by aperture geometry " *
                "and does not use c(1-c) weights",
        ),
    )
    MimeticProjection(
        jump_interpolant, preconditioner, preconditioner_options,
        active_mass_fraction, rank_tolerance, weights,
        commuting_patch_radius, commuting_max_patch_radius,
        commuting_amplification_limit, commuting_accuracy_factor,
        commuting_sliver_epsilon,
    )
end

"""Pressure-projection policy for the conservative multidimensional GFM."""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
