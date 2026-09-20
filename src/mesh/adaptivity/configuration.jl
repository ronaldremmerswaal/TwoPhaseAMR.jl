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
    function Adaptivity(criteria::C, fields::F) where {C <: Tuple, F <: Tuple}
        all(
            criterion -> criterion isa AbstractRefinementIndicator,
            criteria,
        ) || throw(
            ArgumentError("criteria must contain refinement criteria"),
        )
        all(field -> field isa AbstractMeshField, fields) || throw(
            ArgumentError("fields must contain mesh fields"),
        )
        return new{C, F}(criteria, fields)
    end
end

function Adaptivity(; criteria = (), fields = ())
    configured_criteria = _configuration_tuple(criteria)
    configured_fields = _configuration_tuple(fields)
    return Adaptivity(configured_criteria, configured_fields)
end
