@inline _configuration_tuple(::Nothing) = ()
@inline _configuration_tuple(value::Tuple) = value
@inline _configuration_tuple(value::AbstractVector) = Tuple(value)
@inline _configuration_tuple(value) = (value,)

"""
    Adaptivity(; criteria=(), fields=())

Runtime AMR configuration for an equation system. `criteria` contains
refinement criteria and `fields` contains additional fields that must be
transferred whenever the mesh changes. Adaptation cadence remains a property
of [`MeshAdaptationOptions`](@ref).
"""
struct Adaptivity{C <: Tuple, F <: Tuple}
    criteria::C
    fields::F
end

function Adaptivity(; criteria = (), fields = ())
    return Adaptivity(
        _configuration_tuple(criteria), _configuration_tuple(fields),
    )
end
