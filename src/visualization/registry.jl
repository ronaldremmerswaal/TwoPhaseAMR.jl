function _register_overlay_mesh!(registry, mesh)
    if isnothing(registry.mesh)
        registry.mesh = mesh
    elseif registry.mesh !== mesh
        throw(ArgumentError("all visualization fields and overlays must share one mesh"))
    end
    return registry
end

"""
    register_interface!(registry, interface)

Always draw the reconstructed PLIC curve of `interface`.  If the interface is
attached to an embedded geometry, its filled PLIC solid is registered too.
"""
function register_interface!(
        registry::VisualizationFieldRegistry, interface::InterfaceField,
    )
    _register_overlay_mesh!(registry, interface.mesh)
    any(item -> item === interface, registry.interfaces) ||
        push!(registry.interfaces, interface)
    isnothing(interface.geometry) ||
        register_geometry!(registry, interface.geometry)
    return registry
end

"""Always draw `geometry` as a filled, cell-local PLIC solid."""
function register_geometry!(
        registry::VisualizationFieldRegistry, geometry::GeometryField,
    )
    _register_overlay_mesh!(registry, geometry.mesh)
    any(item -> item === geometry, registry.geometries) ||
        push!(registry.geometries, geometry)
    return registry
end

function _register_source!(
        registry::VisualizationFieldRegistry, name::Symbol,
        source::AbstractVisualizationFieldSource;
        label = string(name), units = "", colormap = :viridis,
        colorrange = nothing
    )
    haskey(registry.indices, name) &&
        throw(ArgumentError("visualization field :$name is already registered"))
    mesh = _source_mesh(source)
    if isnothing(registry.mesh)
        registry.mesh = mesh
    elseif registry.mesh !== mesh
        throw(ArgumentError("all visualization fields must share one mesh"))
    end
    entry = VisualizationFieldEntry(
        name, string(label), string(units), source, colormap, colorrange
    )
    push!(registry.entries, entry)
    registry.indices[name] = length(registry.entries)
    return registry
end

"""Register a primitive cell- or directional face-field source."""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        field::CellField; kwargs...
    )
    return _register_source!(
        registry, name, CellVisualizationSource(field);
        kwargs...
    )
end

"""
    register_field!(registry, name, velocity::TwoPhaseVelocity, interface,
                    Face{Axis}(); kwargs...)
    register_field!(registry, name, velocity::TwoPhaseVelocity, interface,
                    derived; kwargs...)

Register a component, magnitude, or vorticity of a discontinuous intrinsic
velocity. Liquid and gas values share one field entry and are drawn on the
regions defined by the interface's PLIC reconstruction.
"""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
        ::Face{Axis}; kwargs...
    ) where {Axis}
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    source = PhaseVelocityComponentVisualizationSource{
        Axis, typeof(velocity), typeof(interface),
    }(velocity, interface)
    _register_source!(registry, name, source; kwargs...)
    return register_interface!(registry, interface)
end

function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
        derived::Union{CentredMagnitude, Vorticity}; kwargs...
    )
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    source = PhaseVelocityDerivedVisualizationSource(
        derived, velocity, interface,
    )
    _register_source!(registry, name, source; kwargs...)
    return register_interface!(registry, interface)
end

function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        field::FaceField{Axis}; kwargs...
    ) where {Axis}
    return _register_source!(
        registry, name, FaceVisualizationSource{Axis, typeof(field)}(field);
        kwargs...
    )
end

"""Register one directional component of a `FaceVelocity` as a face scalar."""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::FaceVelocity, ::Face{Axis}; kwargs...
    ) where {Axis}
    source = FaceComponentVisualizationSource{Axis, typeof(velocity)}(velocity)
    return _register_source!(registry, name, source; kwargs...)
end

"""Register a derived centred magnitude or vorticity cell scalar."""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        field::FaceVelocity,
        derived::Union{CentredMagnitude, Vorticity}; kwargs...
    )
    source = DerivedFaceVisualizationSource(derived, field)
    return _register_source!(registry, name, source; kwargs...)
end

"""
    register_field!(registry, name, scalar::PhaseScalar; kwargs...)
    register_field!(registry, name, scalar::PhaseFaceScalar; kwargs...)

Register both intrinsic phase values as one discontinuous field. The liquid
and gas values are drawn only on their respective PLIC regions, including for
staggered dual control volumes and embedded cut-cell geometry.
"""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        scalar::Union{PhaseScalar, PhaseFaceScalar};
        kwargs...
    )
    source = PhaseVisualizationSource(scalar)
    _register_source!(registry, name, source; kwargs...)
    return register_interface!(registry, scalar.interface)
end

function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        fields::Tuple{X, Y},
        derived::Union{CentredMagnitude, Vorticity};
        kwargs...
    ) where {X <: FaceField{1}, Y <: FaceField{2}}
    fields[1].mesh === fields[2].mesh ||
        throw(ArgumentError("face-field components must share one mesh"))
    source = DerivedFaceVisualizationSource(derived, fields)
    return _register_source!(registry, name, source; kwargs...)
end

_default_field_label(name::Symbol) =
    uppercasefirst(replace(string(name), '_' => ' '))

_default_face_field_names(name::Symbol) = (
    Symbol(name, "_x"), Symbol(name, "_y"),
    Symbol(name, "_magnitude"), Symbol(name, "_vorticity"),
)

_default_field_names(
    name::Symbol, ::Union{CellField, FaceField, PhaseScalar, PhaseFaceScalar},
) = (name,)
_default_field_names(name::Symbol, ::InterfaceField) = (name,)
_default_field_names(name::Symbol, ::FaceVelocity) =
    _default_face_field_names(name)
_default_field_names(name::Symbol, ::Tuple{X, Y}) where
{X <: FaceField{1}, Y <: FaceField{2}} = _default_face_field_names(name)
function _default_field_names(name::Symbol, source)
    throw(
        ArgumentError(
            "default visualization source :$name must be a CellField, " *
                "FaceField, PhaseScalar, PhaseFaceScalar, FaceVelocity, " *
                "or (FaceField{1}, FaceField{2}) tuple; " *
                "got $(typeof(source))"
        )
    )
end

_default_field_mesh(field::AbstractMeshField) = field.mesh
_default_field_mesh(field::Union{PhaseScalar, PhaseFaceScalar}) =
    field.interface.mesh
function _default_field_mesh(fields::Tuple{X, Y}) where
    {X <: FaceField{1}, Y <: FaceField{2}}
    fields[1].mesh === fields[2].mesh ||
        throw(ArgumentError("face-field components must share one mesh"))
    return fields[1].mesh
end

function _register_default_field!(registry, name, field::CellField)
    return register_field!(
        registry, name, field; label = _default_field_label(name)
    )
end

function _register_default_field!(registry, name, field::InterfaceField)
    register_field!(
        registry, name, field.fraction;
        label = _default_field_label(name), colorrange = (0, 1),
    )
    return register_interface!(registry, field)
end

function _register_default_field!(registry, name, field::FaceField)
    return register_field!(
        registry, name, field; label = _default_field_label(name),
        colormap = :balance
    )
end

function _register_default_field!(
        registry, name, field::Union{PhaseScalar, PhaseFaceScalar}
    )
    return register_field!(
        registry, name, field; label = _default_field_label(name)
    )
end

function _register_default_field!(registry, name, velocity::FaceVelocity)
    x_name, y_name, magnitude_name, vorticity_name =
        _default_face_field_names(name)
    label = _default_field_label(name)
    register_field!(
        registry, x_name, velocity, Face{1}();
        label = "$label x", colormap = :balance
    )
    register_field!(
        registry, y_name, velocity, Face{2}();
        label = "$label y", colormap = :balance
    )
    register_field!(
        registry, magnitude_name, velocity, CentredMagnitude();
        label = "$label magnitude"
    )
    register_field!(
        registry, vorticity_name, velocity, Vorticity();
        label = "$label vorticity", colormap = :balance
    )
    return registry
end

function _register_default_field!(registry, name, fields::Tuple{X, Y}) where
    {X <: FaceField{1}, Y <: FaceField{2}}
    x_name, y_name, magnitude_name, vorticity_name =
        _default_face_field_names(name)
    label = _default_field_label(name)
    register_field!(
        registry, x_name, fields[1]; label = "$label x", colormap = :balance
    )
    register_field!(
        registry, y_name, fields[2]; label = "$label y", colormap = :balance
    )
    register_field!(
        registry, magnitude_name, fields, CentredMagnitude();
        label = "$label magnitude"
    )
    register_field!(
        registry, vorticity_name, fields, Vorticity();
        label = "$label vorticity", colormap = :balance
    )
    return registry
end

"""
    register_default_fields!(registry, name, source, ...)

Register alternating names and field sources using standard visualization
names and styles. A `CellField`, directional `FaceField`, `PhaseScalar`, or
`PhaseFaceScalar` produces one entry.
A `FaceVelocity` or `(FaceField{1}, FaceField{2})` tuple named `:velocity`
produces `:velocity_x`, `:velocity_y`, `:velocity_magnitude`, and
`:velocity_vorticity`. All arguments are validated before the registry is
modified. Use [`register_field!`](@ref) when custom names or metadata are
needed.
"""
function register_default_fields!(
        registry::VisualizationFieldRegistry,
        arguments...
    )
    iseven(length(arguments)) || throw(
        ArgumentError(
            "register_default_fields! expects alternating name and source arguments"
        )
    )
    registrations = Tuple{Symbol, Any}[]
    expanded_names = Symbol[]
    target_mesh = registry.mesh

    for index in 1:2:length(arguments)
        name = arguments[index]
        name isa Symbol || throw(
            ArgumentError(
                "visualization field names must be Symbols; got $(repr(name))"
            )
        )
        source = arguments[index + 1]
        names = _default_field_names(name, source)
        mesh = _default_field_mesh(source)
        if isnothing(target_mesh)
            target_mesh = mesh
        elseif target_mesh !== mesh
            throw(ArgumentError("all visualization fields must share one mesh"))
        end
        append!(expanded_names, names)
        push!(registrations, (name, source))
    end

    length(unique(expanded_names)) == length(expanded_names) ||
        throw(ArgumentError("default visualization field names overlap"))
    for name in expanded_names
        haskey(registry.indices, name) && throw(
            ArgumentError(
                "visualization field :$name is already registered"
            )
        )
    end
    for (name, source) in registrations
        _register_default_field!(registry, name, source)
    end
    return registry
end

"""
    register_default_fields!(registry, name, velocity::TwoPhaseVelocity,
                             interface)

Register x, y, magnitude, and vorticity views of a discontinuous intrinsic
velocity. Each view contains both phases, sharply separated by the interface's
PLIC reconstruction. For example, `:velocity` produces `:velocity_x` through
`:velocity_vorticity`.
"""
function register_default_fields!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
    )
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    _register_overlay_mesh!(registry, interface.mesh)
    names = collect(_default_face_field_names(name))
    length(unique(names)) == length(names) || throw(
        ArgumentError(
            "default two-velocity visualization field names overlap",
        )
    )
    any(field_name -> haskey(registry.indices, field_name), names) && throw(
        ArgumentError("a default two-velocity visualization field already exists"),
    )
    x_name, y_name, magnitude_name, vorticity_name = names
    label = _default_field_label(name)
    register_field!(
        registry, x_name, velocity, interface, Face{1}();
        label = "$label x", colormap = :balance,
    )
    register_field!(
        registry, y_name, velocity, interface, Face{2}();
        label = "$label y", colormap = :balance,
    )
    register_field!(
        registry, magnitude_name, velocity, interface,
        CentredMagnitude(); label = "$label magnitude",
    )
    register_field!(
        registry, vorticity_name, velocity, interface, Vorticity();
        label = "$label vorticity", colormap = :balance,
    )
    return registry
end

"""Return registered field names in display order."""
field_names(registry::VisualizationFieldRegistry) =
    [entry.name for entry in registry.entries]

function _field_entry(registry::VisualizationFieldRegistry, name::Symbol)
    index = get(registry.indices, name, 0)
    iszero(index) && throw(ArgumentError("unknown visualization field :$name"))
    return registry.entries[index]
end

_source_mesh(source::CellVisualizationSource) = source.field.mesh
_source_mesh(source::FaceVisualizationSource) = source.field.mesh
_source_mesh(source::FaceComponentVisualizationSource) = source.field.mesh
_source_mesh(source::DerivedFaceVisualizationSource{D, F}) where
{D, F <: FaceVelocity} = source.field.mesh
_source_mesh(source::DerivedFaceVisualizationSource{D, F}) where
{D, F <: Tuple} = source.field[1].mesh
_source_mesh(source::PhaseVisualizationSource) =
    source.quantity.interface.mesh
_source_mesh(source::PhaseVelocityComponentVisualizationSource) =
    source.interface.mesh
_source_mesh(source::PhaseVelocityDerivedVisualizationSource) =
    source.interface.mesh

_source_placement(::CellVisualizationSource) = Cell()
_source_placement(::FaceVisualizationSource{Axis}) where {Axis} = Face{Axis}()
_source_placement(::FaceComponentVisualizationSource{Axis}) where {Axis} = Face{Axis}()
_source_placement(::DerivedFaceVisualizationSource) = Cell()
_source_placement(::PhaseVisualizationSource{<:PhaseScalar}) = Cell()
_source_placement(
    ::PhaseVisualizationSource{<:PhaseFaceScalar{Axis}}
) where {Axis} = Face{Axis}()
_source_placement(
    ::PhaseVelocityComponentVisualizationSource{Axis}
) where {Axis} = Face{Axis}()
_source_placement(::PhaseVelocityDerivedVisualizationSource) = Cell()

@inline _visualization_replacement(replacements, field) =
    get(replacements, field, field)

function _visualization_sampling_interface(interface::I, replacements) where {I <: InterfaceField}
    supplied = get(replacements, interface, interface)
    supplied === interface || return supplied
    isempty(replacements) && return interface
    # Exact keyframes replace individual interface fields, not necessarily
    # the InterfaceField itself. Bind those fields without copying mesh/halos
    # or reconstructing geometry. This view is used only during capture.
    fields = ntuple(fieldcount(I)) do index
        field = getfield(interface, index)
        get(replacements, field, field)
    end
    return I(fields...)
end

const _DiscontinuousVisualizationSource = Union{
    PhaseVisualizationSource,
    PhaseVelocityComponentVisualizationSource,
    PhaseVelocityDerivedVisualizationSource,
}

_is_discontinuous_source(source) = source isa _DiscontinuousVisualizationSource
_source_interface(source::PhaseVisualizationSource) = source.quantity.interface
_source_interface(
    source::Union{
        PhaseVelocityComponentVisualizationSource,
        PhaseVelocityDerivedVisualizationSource,
    }
) = source.interface
