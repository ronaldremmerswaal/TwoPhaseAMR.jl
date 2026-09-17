"""
    AbstractVisualization

Supertype for optional simulation observers and interactive viewers. Numerical
code can dispatch on this interface without depending on a graphics package.
"""
abstract type AbstractVisualization end

_supports_dynamic_topology(::AbstractVisualization) = true

"""A zero-cost visualization target for code paths that do not need output."""
struct NoVisualization <: AbstractVisualization end

"""
    AMRCellGeometry

Backend-independent triangulation of the owned cells in a `BlockTreeMesh`.
Every cell has four private vertices and two triangles. Repeating vertices is
intentional: it permits a discontinuous, piecewise-constant color per cell and
makes a picked mesh vertex map back to a cell by integer arithmetic.

`block_segments` contains pairs of endpoints for drawing leaf-block outlines.
No halo cells or `RegionTrees` objects are included.
"""
struct AMRCellGeometry{T}
    points::Vector{NTuple{2, T}}
    triangles::Vector{NTuple{3, Int}}
    cells::Vector{CellRef}
    block_segments::Vector{NTuple{2, T}}
    block_size::NTuple{2, Int}
end

"""Stable reference to an entry in an interior- or boundary-face schedule."""
struct FaceRef
    boundary::Bool
    index::Int
end

"""
    AMRFaceGeometry{Axis}

Backend-independent filled dual-control-volume tiles for owned faces normal to
`Axis`. Each tile has four private vertices and two triangles. `faces` maps
tiles back to the immutable face schedule, while `value_indices` maps tiles to
directional field storage. A periodic seam can produce two tiles for one face.
"""
struct AMRFaceGeometry{Axis, T}
    points::Vector{NTuple{2, T}}
    triangles::Vector{NTuple{3, Int}}
    faces::Vector{FaceRef}
    value_indices::Vector{Int}
end

"""Derived cell-centred magnitude of a two-component face field."""
struct CentredMagnitude end

"""US-spelling alias for [`CentredMagnitude`](@ref)."""
const CenteredMagnitude = CentredMagnitude

"""Derived cell-centred scalar curl of a two-component 2-D face field."""
struct Vorticity end

abstract type AbstractVisualizationFieldSource end

struct CellVisualizationSource{F <: CellField} <: AbstractVisualizationFieldSource
    field::F
end

struct FaceVisualizationSource{Axis, FF <: FaceField{Axis}} <:
    AbstractVisualizationFieldSource
    field::FF
end

struct FaceComponentVisualizationSource{Axis, V <: FaceVelocity} <:
    AbstractVisualizationFieldSource
    field::V
end

struct DerivedFaceVisualizationSource{D, F} <:
    AbstractVisualizationFieldSource
    derived::D
    field::F
end

"""Intrinsic phase scalar with regions lacking that phase masked for display."""
struct PhaseVisualizationSource{Q} <: AbstractVisualizationFieldSource
    quantity::Q
    liquid::Bool
end

"""One masked directional component of an intrinsic phase velocity."""
struct PhaseVelocityComponentVisualizationSource{Axis, V, I} <:
    AbstractVisualizationFieldSource
    velocity::V
    interface::I
    liquid::Bool
end

"""Masked centred magnitude or vorticity of one intrinsic phase velocity."""
struct PhaseVelocityDerivedVisualizationSource{D, V, I} <:
    AbstractVisualizationFieldSource
    derived::D
    velocity::V
    interface::I
    liquid::Bool
end

struct VisualizationFieldEntry{S <: AbstractVisualizationFieldSource, C, R}
    name::Symbol
    label::String
    units::String
    source::S
    colormap::C
    colorrange::R
end

struct VisualizationDiagnosticEntry{F, C}
    name::Symbol
    label::String
    units::String
    reduction::F
    color::C
end

"""
    VisualizationFieldRegistry()

Registry of named scalar fields available to a visualization. Entries retain
references to their source arrays, but values are copied whenever they are
sampled for rendering or future snapshot storage.
"""
mutable struct VisualizationFieldRegistry
    mesh::Union{Nothing, BlockTreeMesh}
    entries::Vector{VisualizationFieldEntry}
    indices::Dict{Symbol, Int}
    diagnostics::Vector{VisualizationDiagnosticEntry}
    diagnostic_indices::Dict{Symbol, Int}
    interfaces::Vector{Any}
    geometries::Vector{Any}
end

VisualizationFieldRegistry() = VisualizationFieldRegistry(
    nothing, VisualizationFieldEntry[], Dict{Symbol, Int}(),
    VisualizationDiagnosticEntry[], Dict{Symbol, Int}(), Any[], Any[]
)

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
                    Face{Axis}(); phase, kwargs...)
    register_field!(registry, name, velocity::TwoPhaseVelocity, interface,
                    derived; phase, kwargs...)

Register a masked component, magnitude, or vorticity of one intrinsic phase
velocity. Values on dual volumes or cells where that phase is absent are
reported as `NaN`, following the same rule as `PhaseFaceScalar`.
"""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
        ::Face{Axis}; phase, kwargs...
    ) where {Axis}
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    source = PhaseVelocityComponentVisualizationSource{
        Axis, typeof(velocity), typeof(interface),
    }(velocity, interface, _visualization_phase(phase))
    _register_source!(registry, name, source; kwargs...)
    return register_interface!(registry, interface)
end

function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
        derived::Union{CenteredMagnitude, Vorticity}; phase, kwargs...
    )
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    source = PhaseVelocityDerivedVisualizationSource(
        derived, velocity, interface, _visualization_phase(phase),
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
        derived::Union{CenteredMagnitude, Vorticity}; kwargs...
    )
    source = DerivedFaceVisualizationSource(derived, field)
    return _register_source!(registry, name, source; kwargs...)
end

function _visualization_phase(phase::Symbol)
    phase in (:liquid, :gas) || throw(
        ArgumentError("phase must be :liquid or :gas; got $(repr(phase))")
    )
    return phase === :liquid
end

"""
    register_field!(registry, name, scalar::PhaseScalar;
                    phase, kwargs...)
    register_field!(registry, name, scalar::PhaseFaceScalar;
                    phase, kwargs...)

Register one intrinsic phase value for visualization. Values in cells or
staggered control volumes where the selected phase fraction is no larger than
the interface's global `fraction_floor` are returned as `NaN`. This masks
stored values which have no physical meaning because that phase is absent,
while leaving the numerical field unchanged.
"""
function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        scalar::Union{PhaseScalar, PhaseFaceScalar};
        phase,
        kwargs...
    )
    liquid = _visualization_phase(phase)
    source = PhaseVisualizationSource(scalar, liquid)
    _register_source!(registry, name, source; kwargs...)
    return register_interface!(registry, scalar.interface)
end

function register_field!(
        registry::VisualizationFieldRegistry, name::Symbol,
        fields::Tuple{X, Y},
        derived::Union{CenteredMagnitude, Vorticity};
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

_default_field_names(name::Symbol, ::Union{CellField, FaceField}) = (name,)
_default_field_names(name::Symbol, ::InterfaceField) = (name,)
_default_field_names(name::Symbol, ::FaceVelocity) =
    _default_face_field_names(name)
_default_field_names(name::Symbol, ::Tuple{X, Y}) where
{X <: FaceField{1}, Y <: FaceField{2}} = _default_face_field_names(name)
function _default_field_names(name::Symbol, source)
    throw(
        ArgumentError(
            "default visualization source :$name must be a CellField, " *
                "FaceField, FaceVelocity, or (FaceField{1}, FaceField{2}) tuple; " *
                "got $(typeof(source))"
        )
    )
end

_default_field_mesh(field::AbstractMeshField) = field.mesh
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
names and styles. A `CellField` or directional `FaceField` produces one entry.
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

Register x, y, magnitude, and vorticity views for both intrinsic phase
velocities. Names are prefixed with `liquid_` and `gas_`; for example,
`:velocity` produces `:liquid_velocity_x` and `:gas_velocity_vorticity`.
"""
function register_default_fields!(
        registry::VisualizationFieldRegistry, name::Symbol,
        velocity::TwoPhaseVelocity, interface::InterfaceField,
    )
    _validate_two_phase_velocity(
        velocity, interface.mesh, eltype(interface),
    )
    _register_overlay_mesh!(registry, interface.mesh)
    names = Symbol[]
    for phase in (:liquid, :gas)
        append!(names, _default_face_field_names(Symbol(phase, "_", name)))
    end
    length(unique(names)) == length(names) || throw(ArgumentError(
        "default two-velocity visualization field names overlap",
    ))
    any(field_name -> haskey(registry.indices, field_name), names) && throw(
        ArgumentError("a default two-velocity visualization field already exists"),
    )
    for phase in (:liquid, :gas)
        phase_name = Symbol(phase, "_", name)
        x_name, y_name, magnitude_name, vorticity_name =
            _default_face_field_names(phase_name)
        label = _default_field_label(phase_name)
        register_field!(
            registry, x_name, velocity, interface, Face{1}();
            phase, label = "$label x", colormap = :balance,
        )
        register_field!(
            registry, y_name, velocity, interface, Face{2}();
            phase, label = "$label y", colormap = :balance,
        )
        register_field!(
            registry, magnitude_name, velocity, interface,
            CentredMagnitude(); phase, label = "$label magnitude",
        )
        register_field!(
            registry, vorticity_name, velocity, interface, Vorticity();
            phase, label = "$label vorticity", colormap = :balance,
        )
    end
    return registry
end

"""
    register_diagnostic!(registry, name, reduction;
                         label=name, units="", color=nothing)

Register a scalar diagnostic sampled from completed simulation states.
`reduction` is called as
`reduction(velocity, pressure, time, step, solver)` and must return a real
scalar. Diagnostic sampling is configured independently on
[`VisualizationSession`](@ref).
"""
function register_diagnostic!(
        registry::VisualizationFieldRegistry,
        name::Symbol, reduction;
        label = _default_field_label(name), units = "",
        color = nothing
    )
    haskey(registry.diagnostic_indices, name) && throw(
        ArgumentError(
            "visualization diagnostic :$name is already registered"
        )
    )
    entry = VisualizationDiagnosticEntry(
        name, string(label), string(units), reduction, color
    )
    push!(registry.diagnostics, entry)
    registry.diagnostic_indices[name] = length(registry.diagnostics)
    return registry
end

"""Return registered field names in display order."""
field_names(registry::VisualizationFieldRegistry) =
    [entry.name for entry in registry.entries]

"""Return registered scalar-diagnostic names in display order."""
diagnostic_names(registry::VisualizationFieldRegistry) =
    [entry.name for entry in registry.diagnostics]

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

_source_values(source::CellVisualizationSource) =
    visualization_values(source.field)
_source_values(source::FaceVisualizationSource) =
    visualization_values(source.field)
_source_values(source::FaceComponentVisualizationSource{Axis}) where {Axis} =
    visualization_values(source.field, Face{Axis}())
_source_values(source::DerivedFaceVisualizationSource) =
    visualization_values(source.field, source.derived)

@inline _visualization_replacement(replacements, field) =
    get(replacements, field, field)

function _mask_absent_phase!(values, fractions, liquid, fraction_floor;
        open_volumes = nothing)
    length(values) == length(fractions) || throw(
        DimensionMismatch("phase values and fractions have different sizes")
    )
    isnothing(open_volumes) || length(open_volumes) == length(values) || throw(
        DimensionMismatch("phase values and open volumes have different sizes")
    )
    invalid = convert(eltype(values), NaN)
    if liquid
        @inbounds for index in eachindex(values, fractions)
            open = isnothing(open_volumes) ||
                open_volumes[index] > zero(eltype(open_volumes))
            if !open || !(fractions[index] > fraction_floor)
                values[index] = invalid
            end
        end
    else
        upper = one(eltype(fractions)) - fraction_floor
        @inbounds for index in eachindex(values, fractions)
            open = isnothing(open_volumes) ||
                open_volumes[index] > zero(eltype(open_volumes))
            if !open || !(fractions[index] < upper)
                values[index] = invalid
            end
        end
    end
    return values
end

function _phase_visualization_values(
        source::PhaseVisualizationSource{<:PhaseScalar}, replacements
    )
    scalar = source.quantity
    field = source.liquid ? scalar.liquid : scalar.gas
    values = visualization_values(
        _visualization_replacement(replacements, field)
    )
    fractions = visualization_values(
        _visualization_replacement(
            replacements, scalar.interface.conditional_fraction
        )
    )
    geometry = scalar.interface.geometry
    open_volumes = isnothing(geometry) ? nothing :
        visualization_values(geometry.open_fraction)
    return _mask_absent_phase!(
        values, fractions, source.liquid, scalar.interface.fraction_floor;
        open_volumes
    )
end

function _phase_visualization_values(
        source::PhaseVisualizationSource{<:PhaseFaceScalar{Axis}},
        replacements
    ) where {Axis}
    scalar = source.quantity
    field = source.liquid ? scalar.liquid : scalar.gas
    values = visualization_values(
        _visualization_replacement(replacements, field)
    )
    fractions = visualization_values(
        _visualization_replacement(replacements, scalar.fraction)
    )
    geometry = scalar.interface.geometry
    open_volumes = if isnothing(geometry)
        nothing
    else
        _geometry_phase_open_volumes(geometry, Val(Axis))
    end
    return _mask_absent_phase!(
        values, fractions, source.liquid, scalar.interface.fraction_floor;
        open_volumes
    )
end

_source_values(source::PhaseVisualizationSource) =
    _phase_visualization_values(source, IdDict{Any, Any}())

function _phase_velocity_visualization_values(
        source::PhaseVelocityComponentVisualizationSource{Axis}, replacements,
    ) where {Axis}
    interface = source.interface
    phase = source.liquid ? 1 : 2
    velocity = _visualization_replacement(
        replacements, source.velocity[phase],
    )
    values = visualization_values(velocity, Face{Axis}())
    fraction = _phase_face_average!(
        FaceField{Axis}(interface.mesh, eltype(interface)),
        _visualization_replacement(
            replacements, interface.conditional_fraction,
        ),
        interface.geometry,
    )
    fractions = visualization_values(fraction)
    geometry = interface.geometry
    open_volumes = isnothing(geometry) ? nothing :
        _geometry_phase_open_volumes(geometry, Val(Axis))
    return _mask_absent_phase!(
        values, fractions, source.liquid, interface.fraction_floor;
        open_volumes,
    )
end

function _phase_velocity_visualization_values(
        source::PhaseVelocityDerivedVisualizationSource, replacements,
    )
    interface = source.interface
    phase = source.liquid ? 1 : 2
    velocity = _visualization_replacement(
        replacements, source.velocity[phase],
    )
    values = visualization_values(velocity, source.derived)
    fractions = visualization_values(
        _visualization_replacement(
            replacements, interface.conditional_fraction,
        ),
    )
    geometry = interface.geometry
    open_volumes = isnothing(geometry) ? nothing :
        visualization_values(geometry.open_fraction)
    return _mask_absent_phase!(
        values, fractions, source.liquid, interface.fraction_floor;
        open_volumes,
    )
end

_source_values(source::Union{
    PhaseVelocityComponentVisualizationSource,
    PhaseVelocityDerivedVisualizationSource,
}) = _phase_velocity_visualization_values(source, IdDict{Any, Any}())

"""
    visualization_geometry(mesh)

Materialize plotting geometry from the compact leaf-block records in `mesh`.
The ordering is block-major, then `j`, then `i`, matching
`visualization_values`.
"""
function visualization_geometry(mesh::BlockTreeMesh{T}) where {T}
    nx, ny = mesh.block_size
    count = ncells(mesh)
    points = Vector{NTuple{2, T}}(undef, 4count)
    triangles = Vector{NTuple{3, Int}}(undef, 2count)
    cells = Vector{CellRef}(undef, count)
    block_segments = Vector{NTuple{2, T}}(undef, 8nblocks(mesh))

    cell_index = 0
    for (block_index, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        cell_index += 1
        xlo = block.origin[1] + (i - 1) * block.spacing[1]
        xhi = xlo + block.spacing[1]
        ylo = block.origin[2] + (j - 1) * block.spacing[2]
        yhi = ylo + block.spacing[2]
        first_vertex = 4cell_index - 3
        points[first_vertex] = (xlo, ylo)
        points[first_vertex + 1] = (xhi, ylo)
        points[first_vertex + 2] = (xhi, yhi)
        points[first_vertex + 3] = (xlo, yhi)
        first_triangle = 2cell_index - 1
        triangles[first_triangle] =
            (first_vertex, first_vertex + 1, first_vertex + 2)
        triangles[first_triangle + 1] =
            (first_vertex, first_vertex + 2, first_vertex + 3)
        cells[cell_index] = CellRef(block_index, i, j)
    end

    for (block_index, block) in pairs(mesh.blocks)
        xlo, ylo = block.origin
        xhi = xlo + block.extent[1]
        yhi = ylo + block.extent[2]
        first_point = 8block_index - 7
        block_segments[first_point] = (xlo, ylo)
        block_segments[first_point + 1] = (xhi, ylo)
        block_segments[first_point + 2] = (xhi, ylo)
        block_segments[first_point + 3] = (xhi, yhi)
        block_segments[first_point + 4] = (xhi, yhi)
        block_segments[first_point + 5] = (xlo, yhi)
        block_segments[first_point + 6] = (xlo, yhi)
        block_segments[first_point + 7] = (xlo, ylo)
    end
    return AMRCellGeometry(
        points, triangles, cells, block_segments,
        mesh.block_size
    )
end

function _face_normal_intervals(
        mesh::BlockTreeMesh{T}, face::InteriorFace,
        axis::Int
    ) where {T}
    minus_width = mesh.blocks[face.minus.block].spacing[axis] / 2
    plus_width = mesh.blocks[face.plus.block].spacing[axis] / 2
    lower = face.center[axis] - minus_width
    upper = face.center[axis] + plus_width
    domain_lower = mesh.origin[axis]
    domain_upper = domain_lower + mesh.extent[axis]
    if !mesh.periodic[axis]
        return [(max(lower, domain_lower), min(upper, domain_upper))]
    elseif lower < domain_lower
        return [
            (domain_lower, min(upper, domain_upper)),
            (domain_upper - (domain_lower - lower), domain_upper),
        ]
    elseif upper > domain_upper
        return [
            (max(lower, domain_lower), domain_upper),
            (domain_lower, domain_lower + (upper - domain_upper)),
        ]
    end
    return [(lower, upper)]
end

function _face_normal_intervals(
        mesh::BlockTreeMesh{T}, face::BoundaryFace,
        axis::Int
    ) where {T}
    half_width = mesh.blocks[face.cell.block].spacing[axis] / 2
    lower = face.side < 0 ? face.center[axis] :
        face.center[axis] - half_width
    upper = face.side < 0 ? face.center[axis] + half_width :
        face.center[axis]
    return [(lower, upper)]
end

function _push_face_tile!(
        geometry_points, geometry_triangles,
        geometry_faces, value_indices, reference,
        value_index, axis, normal_lower, normal_upper,
        tangent_lower, tangent_upper
    )
    normal_upper > normal_lower || return
    tangent_upper > tangent_lower || return
    xlower, xupper, ylower, yupper = if axis == 1
        (normal_lower, normal_upper, tangent_lower, tangent_upper)
    else
        (tangent_lower, tangent_upper, normal_lower, normal_upper)
    end
    first_vertex = length(geometry_points) + 1
    push!(
        geometry_points,
        (xlower, ylower), (xupper, ylower),
        (xupper, yupper), (xlower, yupper)
    )
    push!(
        geometry_triangles,
        (first_vertex, first_vertex + 1, first_vertex + 2),
        (first_vertex, first_vertex + 2, first_vertex + 3)
    )
    push!(geometry_faces, reference)
    push!(value_indices, value_index)
    return
end

"""Construct filled staggered dual-cell geometry for one face direction."""
function visualization_geometry(mesh::BlockTreeMesh{T}, ::Face{Axis}) where {T, Axis}
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    faces = FaceRef[]
    value_indices = Int[]
    value_index = 0
    tangent_axis = 3 - Axis
    tangent_domain_lower = mesh.origin[tangent_axis]
    tangent_domain_upper =
        tangent_domain_lower + mesh.extent[tangent_axis]

    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        value_index += 1
        tangent_lower = max(
            tangent_domain_lower, face.center[tangent_axis] - face.area / 2
        )
        tangent_upper = min(
            tangent_domain_upper, face.center[tangent_axis] + face.area / 2
        )
        for (normal_lower, normal_upper) in
            _face_normal_intervals(mesh, face, Axis)
            _push_face_tile!(
                points, triangles, faces, value_indices,
                FaceRef(false, global_index), value_index, Axis,
                normal_lower, normal_upper, tangent_lower, tangent_upper
            )
        end
    end
    for (global_index, face) in pairs(mesh.boundary_faces)
        face.axis == Axis || continue
        value_index += 1
        tangent_lower = max(
            tangent_domain_lower, face.center[tangent_axis] - face.area / 2
        )
        tangent_upper = min(
            tangent_domain_upper, face.center[tangent_axis] + face.area / 2
        )
        for (normal_lower, normal_upper) in
            _face_normal_intervals(mesh, face, Axis)
            _push_face_tile!(
                points, triangles, faces, value_indices,
                FaceRef(true, global_index), value_index, Axis,
                normal_lower, normal_upper, tangent_lower, tangent_upper
            )
        end
    end
    return AMRFaceGeometry{Axis, T}(
        points, triangles, faces, value_indices
    )
end

"""
    visualization_values(field)

Copy the owned (non-halo) values of a cell field into the same block-major,
`j`, `i` ordering as `visualization_geometry(mesh).cells`.
"""
function visualization_values(field::CellField{T}) where {T}
    nx, ny = field.mesh.block_size
    values = Vector{T}(undef, ncells(field.mesh))
    index = 0
    for block in eachindex(field.blocks)
        owned = interior(field, block)
        for j in 1:ny, i in 1:nx
            index += 1
            @inbounds values[index] = owned[i, j]
        end
    end
    return values
end

"""Copy owned directional face values in the order used by face geometry."""
function visualization_values(field::FaceField{Axis, T}) where {Axis, T}
    values = Vector{T}(undef, length(field.interior) + length(field.boundary))
    copyto!(values, 1, field.interior, 1, length(field.interior))
    copyto!(
        values, length(field.interior) + 1,
        field.boundary, 1, length(field.boundary)
    )
    return values
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::Face{Axis}
    ) where {T, Axis}
    count = _nfaces(velocity.mesh, Axis) +
        _nboundary_faces(velocity.mesh, Axis)
    values = Vector{T}(undef, count)
    local_index = 0
    for (global_index, face) in pairs(velocity.mesh.faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.interior[global_index]
    end
    for (global_index, face) in pairs(velocity.mesh.boundary_faces)
        face.axis == Axis || continue
        local_index += 1
        @inbounds values[local_index] = velocity.boundary[global_index]
    end
    return values
end

@inline function _visualization_cell_index(mesh::BlockTreeMesh, cell::CellRef)
    nx, ny = mesh.block_size
    return (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
end

@inline _visualization_side_slot(axis::Int, side::Int) =
    2axis - (side < 0 ? 1 : 0)

function _cell_centered_velocity(velocity::FaceVelocity{T}) where {T}
    mesh = velocity.mesh
    count = ncells(mesh)
    weighted = zeros(T, 4, count)
    measures = zeros(T, 4, count)
    for (face_index, face) in pairs(mesh.faces)
        value = @inbounds velocity.interior[face_index]
        for (cell, side) in ((face.minus, 1), (face.plus, -1))
            index = _visualization_cell_index(mesh, cell)
            slot = _visualization_side_slot(face.axis, side)
            @inbounds weighted[slot, index] += value * face.area
            @inbounds measures[slot, index] += face.area
        end
    end
    for (face_index, face) in pairs(mesh.boundary_faces)
        index = _visualization_cell_index(mesh, face.cell)
        slot = _visualization_side_slot(face.axis, face.side)
        value = @inbounds velocity.boundary[face_index]
        @inbounds weighted[slot, index] += value * face.area
        @inbounds measures[slot, index] += face.area
    end

    x = Vector{T}(undef, count)
    y = Vector{T}(undef, count)
    for index in 1:count
        all(@view(measures[:, index]) .> zero(T)) ||
            throw(ArgumentError("incomplete cell-face schedule"))
        @inbounds x[index] = (
            weighted[1, index] / measures[1, index] +
                weighted[2, index] / measures[2, index]
        ) / 2
        @inbounds y[index] = (
            weighted[3, index] / measures[3, index] +
                weighted[4, index] / measures[4, index]
        ) / 2
    end
    return x, y
end

function visualization_values(
        velocity::FaceVelocity{T},
        ::CenteredMagnitude
    ) where {T}
    x, y = _cell_centered_velocity(velocity)
    return hypot.(x, y)
end

function visualization_values(
        fields::Tuple{X, Y},
        derived::Union{CenteredMagnitude, Vorticity}
    ) where
    {X <: FaceField{1}, Y <: FaceField{2}}
    fields[1].mesh === fields[2].mesh ||
        throw(ArgumentError("face-field components must share one mesh"))
    return visualization_values(FaceVelocity(fields...), derived)
end

function _cell_centers(mesh::BlockTreeMesh{T}) where {T}
    nx, ny = mesh.block_size
    centers = Vector{NTuple{2, T}}(undef, ncells(mesh))
    index = 0
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        index += 1
        centers[index] = cellcenter(mesh, block, i, j)
    end
    return centers
end

function visualization_values(velocity::FaceVelocity{T}, ::Vorticity) where {T}
    mesh = velocity.mesh
    xvelocity, yvelocity = _cell_centered_velocity(velocity)
    centers = _cell_centers(mesh)
    count = length(centers)
    a11 = zeros(T, count)
    a12 = zeros(T, count)
    a22 = zeros(T, count)
    bx1 = zeros(T, count)
    bx2 = zeros(T, count)
    by1 = zeros(T, count)
    by2 = zeros(T, count)

    for face in mesh.faces
        minus = _visualization_cell_index(mesh, face.minus)
        plus = _visualization_cell_index(mesh, face.plus)
        dx = T(
            _wrapped_delta(
                centers[plus][1] - centers[minus][1],
                mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                centers[plus][2] - centers[minus][2],
                mesh.extent[2], mesh.periodic[2]
            )
        )
        distance_squared = dx^2 + dy^2
        distance_squared > zero(T) || continue
        weight = T(face.area) / distance_squared
        du = xvelocity[plus] - xvelocity[minus]
        dv = yvelocity[plus] - yvelocity[minus]
        for index in (minus, plus)
            @inbounds a11[index] += weight * dx^2
            @inbounds a12[index] += weight * dx * dy
            @inbounds a22[index] += weight * dy^2
            @inbounds bx1[index] += weight * du * dx
            @inbounds bx2[index] += weight * du * dy
            @inbounds by1[index] += weight * dv * dx
            @inbounds by2[index] += weight * dv * dy
        end
    end

    result = Vector{T}(undef, count)
    for index in 1:count
        determinant = @inbounds a11[index] * a22[index] - a12[index]^2
        scale = @inbounds max(abs(a11[index] * a22[index]), a12[index]^2)
        determinant > 128eps(T) * scale ||
            throw(ArgumentError("cell-neighbor geometry is rank deficient"))
        dvdx = @inbounds (
            a22[index] * by1[index] -
                a12[index] * by2[index]
        ) / determinant
        dudy = @inbounds (
            a11[index] * bx2[index] -
                a12[index] * bx1[index]
        ) / determinant
        @inbounds result[index] = dvdx - dudy
    end
    return result
end

function _visualization_vertex_values(field::CellField)
    cell_values = visualization_values(field)
    values = Vector{eltype(cell_values)}(undef, 4length(cell_values))
    for index in eachindex(cell_values)
        value = @inbounds cell_values[index]
        first_vertex = 4index - 3
        @inbounds for vertex in first_vertex:(first_vertex + 3)
            values[vertex] = value
        end
    end
    return values
end

function _visualization_vertex_values(values::AbstractVector)
    result = Vector{eltype(values)}(undef, 4length(values))
    for index in eachindex(values)
        first_vertex = 4index - 3
        @inbounds result[first_vertex] = values[index]
        @inbounds result[first_vertex + 1] = values[index]
        @inbounds result[first_vertex + 2] = values[index]
        @inbounds result[first_vertex + 3] = values[index]
    end
    return result
end

function _visualization_face_vertex_values(
        values::AbstractVector, geometry::AMRFaceGeometry
    )
    result = Vector{eltype(values)}(undef, 4length(geometry.faces))
    for tile in eachindex(geometry.faces)
        value = values[geometry.value_indices[tile]]
        first_vertex = 4tile - 3
        @inbounds result[first_vertex] = value
        @inbounds result[first_vertex + 1] = value
        @inbounds result[first_vertex + 2] = value
        @inbounds result[first_vertex + 3] = value
    end
    return result
end

"""
    cell_for_vertex(geometry, vertex_index)

Return the cell owning a picked mesh vertex, or `nothing` for an invalid
index. Makie reports the largest vertex index of a picked mesh triangle, which
is compatible with this four-private-vertices-per-cell mapping.
"""
function cell_for_vertex(geometry::AMRCellGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.cells[cld(vertex_index, 4)]
end

"""Return the face owning a picked dual-cell vertex, or `nothing`."""
function face_for_vertex(geometry::AMRFaceGeometry, vertex_index::Integer)
    1 <= vertex_index <= length(geometry.points) || return nothing
    return @inbounds geometry.faces[cld(vertex_index, 4)]
end

function _face_outline(
        geometry::AMRFaceGeometry,
        vertex_index::Integer
    )
    1 <= vertex_index <= length(geometry.points) ||
        return eltype(geometry.points)[]
    first_vertex = 4cld(vertex_index, 4) - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end

function _cell_outline(geometry::AMRCellGeometry, cell::CellRef)
    nx, ny = geometry.block_size
    1 <= cell.block || return eltype(geometry.points)[]
    1 <= cell.i <= nx || return eltype(geometry.points)[]
    1 <= cell.j <= ny || return eltype(geometry.points)[]
    cell_index = (cell.block - 1) * nx * ny + (cell.j - 1) * nx + cell.i
    cell_index <= length(geometry.cells) || return eltype(geometry.points)[]
    first_vertex = 4cell_index - 3
    southwest = geometry.points[first_vertex]
    southeast = geometry.points[first_vertex + 1]
    northeast = geometry.points[first_vertex + 2]
    northwest = geometry.points[first_vertex + 3]
    return [
        southwest, southeast, southeast, northeast,
        northeast, northwest, northwest, southwest,
    ]
end

"""
    visualize(source; kwargs...)

Create an interactive AMR field viewer from a `CellField` or
`VisualizationFieldRegistry`. This method is supplied by the optional Makie
extension; load GLMakie or WGLMakie before calling it.
"""
function visualize end

"""Initialize a visualization observer for one simulation run."""
function initialize_visualization! end

"""Replace the values displayed by an existing visualization."""
function update_visualization! end

"""Return the cell selected in a visualization, or `nothing`."""
selected_cell(::NoVisualization) = nothing

"""Return the face selected in a visualization, or `nothing`."""
selected_face(::NoVisualization) = nothing

"""Switch an interactive viewer to another registered scalar field."""
function select_visualization_field! end

"""Return the backend-specific figure owned by a visualization."""
function visualization_figure end

# Optional backend hooks used by a session to bind GUI controls and refresh
# lifecycle/checkpoint state without exposing solver-owned arrays.
_bind_visualization_session!(::AbstractVisualization, session) = nothing
_update_visualization_state!(::AbstractVisualization, session) = nothing
_update_visualization_diagnostics!(::AbstractVisualization, session) = nothing

update_visualization!(visualization::NoVisualization, args...; kwargs...) =
    visualization

function initialize_visualization!(
        visualization::AbstractVisualization;
        step_procedure = nothing, kwargs...
    )
    return update_visualization!(visualization; event = :initial, kwargs...)
end
