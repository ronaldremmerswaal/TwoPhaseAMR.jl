abstract type AbstractCheckpointSpecification end

struct CheckpointCount <: AbstractCheckpointSpecification
    count::Int

    function CheckpointCount(count::Integer)
        count >= 1 ||
            throw(ArgumentError("the checkpoint count must be positive"))
        return new(Int(count))
    end
end

struct CheckpointInterval{T <: Real} <: AbstractCheckpointSpecification
    interval::T

    function CheckpointInterval(interval::T) where {T <: Real}
        isfinite(interval) && interval > zero(T) ||
            throw(
            ArgumentError(
                "the checkpoint time step must be finite and positive"
            )
        )
        return new{T}(interval)
    end
end

struct ExplicitCheckpointTimes{T <: Real} <: AbstractCheckpointSpecification
    times::Vector{T}
end

mutable struct LatestFrameMailbox{F}
    condition::Base.GenericCondition{ReentrantLock}
    pending::Union{Nothing, F}
    state_pending::Bool
    diagnostics_pending::Bool
    closed::Bool
    dropped::Int

    LatestFrameMailbox{F}() where {F} =
        new{F}(Threads.Condition(), nothing, false, false, false, 0)
end


function _publish_latest!(mailbox::LatestFrameMailbox{F}, frame::F) where {F}
    Base.@lock mailbox.condition begin
        mailbox.closed && return false
        isnothing(mailbox.pending) || (mailbox.dropped += 1)
        mailbox.pending = frame
        notify(mailbox.condition)
        return true
    end
end

function _publish_state!(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        mailbox.closed && return false
        mailbox.state_pending = true
        notify(mailbox.condition)
        return true
    end
end

function _publish_diagnostics!(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        mailbox.closed && return false
        mailbox.diagnostics_pending = true
        notify(mailbox.condition)
        return true
    end
end

function _take_latest!(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        while isnothing(mailbox.pending) && !mailbox.state_pending &&
                !mailbox.diagnostics_pending && !mailbox.closed
            wait(mailbox.condition)
        end
        frame = mailbox.pending
        state_pending = mailbox.state_pending
        diagnostics_pending = mailbox.diagnostics_pending
        mailbox.pending = nothing
        mailbox.state_pending = false
        mailbox.diagnostics_pending = false
        return frame, state_pending, diagnostics_pending
    end
end

function _discard_pending!(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        mailbox.pending = nothing
    end
    return mailbox
end

function _close_mailbox!(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        mailbox.closed = true
        notify(mailbox.condition)
    end
    return mailbox
end

function _dropped_frames(mailbox::LatestFrameMailbox)
    Base.@lock mailbox.condition begin
        return mailbox.dropped
    end
end

"""Immutable geometry and face schedules shared by frames of one topology."""
struct VisualizationTopology{T}
    cell_geometry::AMRCellGeometry{T}
    face_geometries::Tuple{AMRFaceGeometry{1, T}, AMRFaceGeometry{2, T}}
    interior_faces::Vector{InteriorFace{T}}
    boundary_faces::Vector{BoundaryFace{T}}
end

"""
    VisualizationFrame

Snapshot metadata, immutable topology, copied primitive data, and cached
registered scalar views for one completed solver state. Arrays never alias
solver-owned storage, and old frames remain valid after live-mesh adaptation.
Registered scalar views are accessed with `frame[name]`.
"""
struct VisualizationFrame{T}
    time::T
    step::Int
    registry::VisualizationFieldRegistry
    primitives::IdDict{Any, Any}
    values::Dict{Symbol, Any}
    topology::VisualizationTopology{T}
    phase_rendering::Symbol
end

Base.haskey(frame::VisualizationFrame, name::Symbol) =
    haskey(frame.registry.indices, name)

Base.getindex(frame::VisualizationFrame, name::Symbol) = frame.values[name]

_frame_values(frame::VisualizationFrame, source::CellVisualizationSource) =
    frame.primitives[source.field]

_frame_values(frame::VisualizationFrame, source::FaceVisualizationSource) =
    visualization_values(frame.primitives[source.field])

function _frame_values(
        frame::VisualizationFrame,
        source::FaceComponentVisualizationSource{Axis}
    ) where {Axis}
    velocity = frame.primitives[source.field]
    return visualization_values(velocity, Face{Axis}())
end

function _frame_values(
        frame::VisualizationFrame,
        source::DerivedFaceVisualizationSource
    )
    field = source.field
    if field isa FaceVelocity
        return visualization_values(
            frame.primitives[field], source.derived
        )
    end
    fields = (frame.primitives[field[1]], frame.primitives[field[2]])
    return visualization_values(fields, source.derived)
end

_frame_values(frame::VisualizationFrame, source::PhaseVisualizationSource) =
    frame.primitives[source].values
_frame_values(
    frame::VisualizationFrame,
    source::Union{
        PhaseVelocityComponentVisualizationSource,
        PhaseVelocityDerivedVisualizationSource,
    },
) = frame.primitives[source].values

function _frame_phase_geometry(frame::VisualizationFrame, source)
    frame.phase_rendering == :plic || throw(ArgumentError(
        "cell-rendered frames have no phase-clipped geometry",
    ))
    return _cached_phase_geometry!(
        frame.primitives[source].geometry_cache, _source_placement(source),
    )
end

"""A retained frame together with the time at which it was requested."""
struct VisualizationCheckpoint{T}
    requested_time::T
    frame::VisualizationFrame{T}
end

"""Scalar diagnostic values sampled at one completed simulation state."""
struct VisualizationDiagnosticSample{T}
    time::T
    step::Int
    values::Dict{Symbol, T}
end

Base.haskey(sample::VisualizationDiagnosticSample, name::Symbol) =
    haskey(sample.values, name)
Base.getindex(sample::VisualizationDiagnosticSample, name::Symbol) =
    sample.values[name]

mutable struct VisualizationDiagnosticHistory{T}
    every::Union{Nothing, Int}
    interval::Union{Nothing, T}
    next_time::T
    last_time::T
    last_step::Int
    samples::Vector{VisualizationDiagnosticSample{T}}
end
