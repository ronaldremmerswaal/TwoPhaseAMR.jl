"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

abstract type _MimeticVariant end
struct _MimeticJointSmoothed <: _MimeticVariant end
struct _MimeticJointFace <: _MimeticVariant end

function _mimetic_variant(variant)
    variant === :joint_smoothed && return _MimeticJointSmoothed()
    variant === :joint_face && return _MimeticJointFace()
    throw(
        ArgumentError(
            "mimetic variant must be :joint_smoothed or :joint_face",
        )
    )
end

@inline _mimetic_variant_symbol(::_MimeticJointSmoothed) = :joint_smoothed
@inline _mimetic_variant_symbol(::_MimeticJointFace) = :joint_face

"""Select a jointly fitted mimetic two-velocity projection.

The default `variant=:joint_smoothed` reconstructs the normal jump from
one-ring-smoothed aperture vectors. `variant=:joint_face` instead starts from
CMDGFM's normal-dominant face layout. A local fit of both normal and
tangential observations keeps exact constant-jump moments and approximately
matches the pressure and multiplier
commutativity relations. Only the fitted normal rows enter the symmetric
energy-compatible KKT projection; the tangential rows guide their fit.
The layout is encoded in the concrete projection type and a singleton variant.
"""
struct MimeticProjection{V <: _MimeticVariant, P, O, A, R} <: AbstractTwoVelocityProjection
    variant::V
    preconditioner::P
    preconditioner_options::O
    active_mass_fraction::A
    rank_tolerance::R
    patch_radius::Int
    pressure_weight::Float64
    commutator_weight::Float64
    correction_cap::Float64
    fit_sweeps::Int

    function MimeticProjection(
            variant, preconditioner, preconditioner_options,
            active_mass_fraction, rank_tolerance, patch_radius,
            pressure_weight, commutator_weight, correction_cap, fit_sweeps,
        )
        selected_variant = _mimetic_variant(variant)
        preconditioner isa Symbol &&
            !(preconditioner in (:cholesky, :amg, :ilu, :ichol)) &&
            throw(
            ArgumentError(
                "mimetic preconditioner must be :cholesky, :amg, :ilu, :ichol, nothing, a factory, or a pre-built preconditioner",
            )
        )
        all(
            x -> isfinite(x) && x > 0, (
                pressure_weight, commutator_weight, correction_cap,
            )
        ) || throw(
            ArgumentError(
                "mimetic fit weights and correction cap must be finite and positive",
            )
        )
        patch_radius >= 0 || throw(
            ArgumentError(
                "patch_radius must be nonnegative",
            )
        )
        fit_sweeps >= 1 || throw(
            ArgumentError(
                "fit_sweeps must be positive",
            )
        )
        return new{
            typeof(selected_variant), typeof(preconditioner),
            typeof(preconditioner_options), typeof(active_mass_fraction),
            typeof(rank_tolerance),
        }(
            selected_variant, preconditioner, preconditioner_options,
            active_mass_fraction, rank_tolerance, Int(patch_radius),
            Float64(pressure_weight), Float64(commutator_weight),
            Float64(correction_cap), Int(fit_sweeps),
        )
    end
end

MimeticProjection(;
    variant = :joint_smoothed,
    preconditioner = :cholesky, preconditioner_options = (;),
    active_mass_fraction = 1.0e-6, rank_tolerance = nothing,
    patch_radius = 2, pressure_weight = 1000.0,
    commutator_weight = 1000.0, correction_cap = 1.0,
    fit_sweeps = 3,
) = MimeticProjection(
    variant, preconditioner, preconditioner_options,
    active_mass_fraction, rank_tolerance, patch_radius,
    pressure_weight, commutator_weight, correction_cap, fit_sweeps,
)

"""Pressure-projection policy for the conservative multidimensional ghost-fluid method.

The supported preconditioners are `:block` (the default), `:ilu`, and `nothing`;
unsupported choices are rejected at construction.
"""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O

    function CMDGFMProjection(preconditioner, preconditioner_options)
        preconditioner in (:block, :ilu, nothing) || throw(
            ArgumentError("CMDGFM preconditioner must be :block, :ilu, or nothing"),
        )
        return new{typeof(preconditioner), typeof(preconditioner_options)}(
            preconditioner, preconditioner_options,
        )
    end
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
