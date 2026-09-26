"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Compatibility name retained for code written before the typed model API."""
const AbstractTwoVelocityPressureProjection = AbstractTwoVelocityProjection

"""Select an energy-preserving mimetic two-velocity projection.

`jump_interpolant` selects the original centred aperture-divergence model
(`:divergence`), the face-local aperture-normal model (`:face`), or the
cell-local aperture-normal model (`:cell`). `:infer` leaves tangential jump
directions free to be inferred from stationarity and constrains only normal
directions. Its default `infer_face_selection=:paired` uses disjoint
aperture-normal cell pairs with a K-orthogonal tangent/normal split;
`:smoothed` averages aperture normals over a one-ring patch before pairing;
`:plic` uses the reconstructed cell and staggered PLIC normals with the same
pairing and KKT blocks;
the experimental `:stencil` alternative classifies faces using CMDGFM's
staggered-normal beta and imposes its normal-dominant rows;
`:smoothed_stencil` uses the same facewise rule with one-ring aperture
normals interpolated to faces. `:commuting` first modifies the
geometric tangential observation `T` so its response to the production
pressure block is first-order accurate while reproducing affine velocity
fields. It then constructs `Q=N'` in the kernel of `T*inv(K_xi)`, subject to
exact aperture-normal moments. A patch grows through the connected interface
component only when a smaller patch cannot meet the pressure or multiplier
certificate. It always uses geometric face weights; `c(1-c)` weights belong
only to the legacy `:cell` and `:face` interpolants.
`commuting_sliver_epsilon` restricts the commuting jump space to faces whose
staggered liquid fraction lies in `[epsilon, 1-epsilon]` (default `0.05`). A
sliver is tied by constant extrapolation to the eligible parallel face at one
of its two tangential endpoints whose fraction is closest to `1/2`; it is set
to zero if neither endpoint supplies a donor. Set the value to zero to recover
the unreduced commuting interpolant. All variants share one active set,
rank-safe symmetric KKT projection, and phase kinetic-energy inner product.
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
    infer_face_selection::Symbol
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
    infer_face_selection = :paired,
) = begin
    weights = isnothing(interpolation_weights) ?
        (jump_interpolant in (:commuting, :infer) ? :geometric : :c1mc) :
        interpolation_weights
    jump_interpolant in (:commuting, :infer) && weights !== :geometric && throw(
        ArgumentError(
            "the commuting and infer interpolants do not accept " *
                "c(1-c) weights",
        ),
    )
    infer_face_selection in (
        :paired, :smoothed, :plic, :stencil, :smoothed_stencil,
    ) || throw(
        ArgumentError(
            "infer_face_selection must be :paired, :smoothed, :plic, " *
                ":stencil, or :smoothed_stencil",
        ),
    )
    MimeticProjection(
        jump_interpolant, preconditioner, preconditioner_options,
        active_mass_fraction, rank_tolerance, weights,
        commuting_patch_radius, commuting_max_patch_radius,
        commuting_amplification_limit, commuting_accuracy_factor,
        commuting_sliver_epsilon, infer_face_selection,
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
