"""Intrinsic liquid/gas values of a phase field before phase selection."""
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
Base.hash(values::TwoPhaseVisualizationValues, seed::UInt) =
    hash(values.gas, hash(values.liquid, hash(TwoPhaseVisualizationValues, seed)))

"""Derived cell-centred magnitude of a two-component face field."""
struct CentredMagnitude end

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

"""Both intrinsic values of a phase scalar."""
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

"""
    VisualizationFieldRegistry()

Registry of named scalar fields recorded by a [`Recorder`](@ref) or drawn by
the static Makie functions. Entries retain references to their source arrays;
values are copied whenever they are captured.

Phase fields store both intrinsic values. Whether a display shows the
majority phase of each cell or splits mixed cells at the PLIC plane is chosen
when drawing (`phase_rendering=:cell` or `:plic`), not when recording.
"""
mutable struct VisualizationFieldRegistry
    mesh::Union{Nothing, BlockTreeMesh}
    entries::Vector{VisualizationFieldEntry}
    indices::Dict{Symbol, Int}
    interfaces::Vector{Any}
    geometries::Vector{Any}
    sampling_workspaces::Dict{Tuple{Any, Any, Symbol}, Any}
end

VisualizationFieldRegistry() = VisualizationFieldRegistry(
    nothing, VisualizationFieldEntry[], Dict{Symbol, Int}(), Any[], Any[],
    Dict{Tuple{Any, Any, Symbol}, Any}(),
)

function _check_phase_rendering(mode)
    mode in (:cell, :plic) || throw(
        ArgumentError(
            "phase_rendering must be :cell or :plic",
        )
    )
    return mode
end
