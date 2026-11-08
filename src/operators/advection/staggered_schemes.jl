"""Interface for staggered face-flux interpolation schemes."""
abstract type AbstractStaggeredAdvectionScheme end

"""First-order upwind interpolation on staggered control-volume faces."""
struct Upwind <: AbstractStaggeredAdvectionScheme end

"""Equal-weight interpolation on staggered control-volume faces."""
struct Centred <: AbstractStaggeredAdvectionScheme end

"""Second-order directionally split Lax--Wendroff transport."""
struct LaxWendroff <: AbstractStaggeredAdvectionScheme end

abstract type AbstractThreePointFluxScheme <:
AbstractStaggeredAdvectionScheme end

"""Unrestricted three-point Fromm flux."""
struct Fromm <: AbstractThreePointFluxScheme end

"""Three-point flux with the minmod limiter."""
struct Minmod <: AbstractThreePointFluxScheme end

"""Three-point flux with the monotonized-central (MC) limiter."""
struct MonotonizedCentral <: AbstractThreePointFluxScheme end

"""Three-point flux with the van Leer limiter."""
struct VanLeer <: AbstractThreePointFluxScheme end

"""Three-point flux with the superbee limiter."""
struct Superbee <: AbstractThreePointFluxScheme end

"""
    default_amr_interface_scheme(scheme)

Choose the robust default flux on non-regular AMR and wall-adjacent
connections. `Upwind()` remains upwind everywhere, while `Centred()` and
`LaxWendroff()` use `MonotonizedCentral()` there. Another three-point scheme
is used unchanged. The corresponding
scheme symbols are accepted as well.
"""
@inline default_amr_interface_scheme(scheme::Upwind) = scheme
@inline default_amr_interface_scheme(::Centred) = MonotonizedCentral()
@inline default_amr_interface_scheme(::LaxWendroff) = MonotonizedCentral()
@inline default_amr_interface_scheme(scheme) = scheme
@inline function default_amr_interface_scheme(scheme::Symbol)
    if scheme === :upwind
        return Upwind()
    elseif scheme === :centred || scheme === :centered || scheme === :central ||
            scheme === :lw || scheme === :lax_wendroff
        return MonotonizedCentral()
    end
    return scheme
end

@inline _is_upwind_scheme(scheme) =
    scheme isa Upwind || scheme === :upwind

@inline _canonical_staggered_scheme(scheme::AbstractStaggeredAdvectionScheme) = scheme
@inline function _canonical_staggered_scheme(scheme::Symbol)
    scheme === :upwind && return Upwind()
    (scheme === :centred || scheme === :centered || scheme === :central) &&
        return Centred()
    (scheme === :lw || scheme === :lax_wendroff) && return LaxWendroff()
    scheme === :fromm && return Fromm()
    scheme === :minmod && return Minmod()
    (
        scheme === :mc || scheme === :monotonized_central ||
            scheme === :monotonised_central
    ) && return MonotonizedCentral()
    (scheme === :van_leer || scheme === :vanleer) && return VanLeer()
    scheme === :superbee && return Superbee()
    throw(ArgumentError("unsupported staggered advection scheme: $scheme"))
end
_canonical_staggered_scheme(scheme) = throw(
    ArgumentError(
        "unsupported staggered advection scheme: $scheme"
    )
)

function _validate_amr_interface_scheme(scheme, amr_interface_scheme)
    if _is_upwind_scheme(amr_interface_scheme) && !_is_upwind_scheme(scheme)
        throw(
            ArgumentError(
                "upwind AMR-interface fluxes require scheme=Upwind(); " *
                    "use a limited interface scheme such as MonotonizedCentral() otherwise"
            )
        )
    end
    return amr_interface_scheme
end
