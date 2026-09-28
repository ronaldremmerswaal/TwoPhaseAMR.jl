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

"""
    PLICVisualizationGeometry{Owner,T}

Triangulation of both fluid phases, clipped sharply by the cell-local PLIC
reconstruction. `owners`, `value_indices`, and `liquid` are stored per vertex
because clipped polygons do not have a fixed number of vertices. `Owner` is
`CellRef` for centred quantities and `FaceRef` for staggered quantities.
"""
struct PLICVisualizationGeometry{Owner, T}
    points::Vector{NTuple{2, T}}
    triangles::Vector{NTuple{3, Int}}
    owners::Vector{Owner}
    value_indices::Vector{Int}
    liquid::BitVector
end

"""Intrinsic liquid/gas values before majority selection or PLIC rendering."""
struct TwoPhaseVisualizationValues{T}
    liquid::Vector{T}
    gas::Vector{T}
end

Base.copy(values::TwoPhaseVisualizationValues) = TwoPhaseVisualizationValues(
    copy(values.liquid), copy(values.gas),
)
Base.:(==)(a::TwoPhaseVisualizationValues, b::TwoPhaseVisualizationValues) =
    a.liquid == b.liquid && a.gas == b.gas
Base.isequal(a::TwoPhaseVisualizationValues, b::TwoPhaseVisualizationValues) =
    isequal(a.liquid, b.liquid) && isequal(a.gas, b.gas)

struct PLICVisualizationSnapshot{V, G}
    values::V
    geometry_cache::G
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

"""Both intrinsic values of a phase scalar, subject to the display policy."""
struct PhaseVisualizationSource{Q} <: AbstractVisualizationFieldSource
    quantity::Q
end

"""Both phase values of one directional intrinsic-velocity component."""
struct PhaseVelocityComponentVisualizationSource{Axis, V, I} <:
    AbstractVisualizationFieldSource
    velocity::V
    interface::I
end

"""Centred magnitude or vorticity of both intrinsic phase velocities."""
struct PhaseVelocityDerivedVisualizationSource{D, V, I} <:
    AbstractVisualizationFieldSource
    derived::D
    velocity::V
    interface::I
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
    VisualizationFieldRegistry(; phase_rendering=:cell)

Registry of named scalar fields available to a visualization. Entries retain
references to their source arrays, but values are copied whenever they are
sampled for rendering or future snapshot storage.

Discontinuous fields use the liquid value in cells with at least 50% liquid
and the gas value otherwise. Face values are first centred within each phase.
Use `phase_rendering=:plic` to display both values on clipped PLIC regions.
"""
mutable struct VisualizationFieldRegistry
    mesh::Union{Nothing, BlockTreeMesh}
    entries::Vector{VisualizationFieldEntry}
    indices::Dict{Symbol, Int}
    diagnostics::Vector{VisualizationDiagnosticEntry}
    diagnostic_indices::Dict{Symbol, Int}
    interfaces::Vector{Any}
    geometries::Vector{Any}
    phase_rendering::Symbol
    sampling_workspaces::Dict{Tuple{Any, Any, Symbol}, Any}
end

function _check_phase_rendering(mode)
    mode in (:cell, :plic) || throw(ArgumentError(
        "phase_rendering must be :cell or :plic",
    ))
    return mode
end

VisualizationFieldRegistry(; phase_rendering = :cell) = VisualizationFieldRegistry(
    nothing, VisualizationFieldEntry[], Dict{Symbol, Int}(),
    VisualizationDiagnosticEntry[], Dict{Symbol, Int}(), Any[], Any[],
    _check_phase_rendering(phase_rendering), Dict{Tuple{Any, Any, Symbol}, Any}(),
)
