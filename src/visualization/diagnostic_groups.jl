"""
    diagnostic_info(diagnostic)::DiagnosticInfo

Display metadata of a diagnostic. Built-in diagnostics report their physical
dimension (two-dimensional quantities per unit depth), vector components, and
kind; a custom [`Diagnostic`](@ref) reports the metadata given at construction.
Other diagnostics default to a name-derived label without a dimension.
"""
diagnostic_info(diagnostic::AbstractDiagnostic) = DiagnosticInfo(;
    label = _default_diagnostic_label(diagnostic_name(diagnostic)),
)
diagnostic_info(diagnostic::Diagnostic) = diagnostic.info

const _ENERGY_DIMENSION = PhysicalDimension(1, 1, -2)
const _VELOCITY_DIMENSION = PhysicalDimension(0, 1, -1)
const _DIMENSIONLESS = PhysicalDimension(0, 0, 0)

diagnostic_info(::KineticEnergy) =
    DiagnosticInfo(; label = "Kinetic energy", dimension = _ENERGY_DIMENSION)
diagnostic_info(::PotentialEnergy) =
    DiagnosticInfo(; label = "Potential energy", dimension = _ENERGY_DIMENSION)
diagnostic_info(::SurfaceEnergy) =
    DiagnosticInfo(; label = "Surface energy", dimension = _ENERGY_DIMENSION)
diagnostic_info(::TotalEnergy) =
    DiagnosticInfo(; label = "Total energy", dimension = _ENERGY_DIMENSION)
diagnostic_info(::LinearMomentum) = DiagnosticInfo(;
    label = "Linear momentum", dimension = PhysicalDimension(1, 0, -1),
    components = ["x", "y"],
)
diagnostic_info(::PhaseVolume) = DiagnosticInfo(;
    label = "Phase volume", dimension = PhysicalDimension(0, 2, 0),
    components = ["liquid", "gas"],
)
diagnostic_info(::PhaseMass) = DiagnosticInfo(;
    label = "Phase mass", dimension = PhysicalDimension(1, -1, 0),
    components = ["liquid", "gas"],
)
diagnostic_info(::InterfaceMeasure) = DiagnosticInfo(;
    label = "Interface length", dimension = PhysicalDimension(0, 1, 0),
)
diagnostic_info(::MaximumVelocity) = DiagnosticInfo(;
    label = "Maximum velocity", dimension = _VELOCITY_DIMENSION,
)
diagnostic_info(::CFLTimeStep) = DiagnosticInfo(;
    label = "CFL time step", dimension = PhysicalDimension(0, 0, 1),
    kind = :solver,
)
diagnostic_info(::DivergenceNorm) = DiagnosticInfo(;
    label = "Divergence norm", dimension = _VELOCITY_DIMENSION,
    kind = :defect, scale = :log,
)
diagnostic_info(::LeafBlockCount) = DiagnosticInfo(;
    label = "Leaf blocks", dimension = _DIMENSIONLESS, kind = :count,
)
diagnostic_info(::CellCount) = DiagnosticInfo(;
    label = "Cells", dimension = _DIMENSIONLESS, kind = :count,
)

"""Return `name => DiagnosticInfo` pairs of the diagnostics in a collection."""
diagnostic_info(collection::DiagnosticCollection) = Pair{Symbol, DiagnosticInfo}[
    diagnostic_name(series.diagnostic) => diagnostic_info(series.diagnostic)
        for series in collection.series
]

const _SUPERSCRIPT_DIGITS = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')

function _superscript(exponent::Integer)
    exponent == 1 && return ""
    digits = [_SUPERSCRIPT_DIGITS[d + 1] for d in reverse(Base.digits(abs(exponent)))]
    return (exponent < 0 ? "⁻" : "") * String(digits)
end

# `print`/`string` give the compact form `M L T⁻²` (dimensionless: `1`);
# `show` gives the constructor call.
function Base.print(io::IO, dimension::PhysicalDimension)
    parts = String[]
    for (symbol, exponent) in (
            ("M", dimension.mass), ("L", dimension.length), ("T", dimension.time),
        )
        iszero(exponent) || push!(parts, symbol * _superscript(exponent))
    end
    return print(io, isempty(parts) ? "1" : join(parts, " "))
end

Base.show(io::IO, dimension::PhysicalDimension) = print(
    io, "PhysicalDimension(", dimension.mass, ", ", dimension.length, ", ",
    dimension.time, ")",
)

const _DIMENSION_QUANTITIES = Dict(
    _ENERGY_DIMENSION => "Energy",
    PhysicalDimension(1, 0, -1) => "Momentum",
    PhysicalDimension(0, 2, 0) => "Area",
    PhysicalDimension(1, -1, 0) => "Mass",
    PhysicalDimension(0, 1, 0) => "Length",
    _VELOCITY_DIMENSION => "Velocity",
    PhysicalDimension(0, 0, 1) => "Time",
)

"""
    DiagnosticGroup

Diagnostics that share one plot axis: the same kind, physical dimension, and
scale. `title` names the shared quantity, `names` lists the members in
registration order.
"""
struct DiagnosticGroup
    title::String
    kind::Symbol
    dimension::Union{Nothing, PhysicalDimension}
    scale::Symbol
    units::String
    names::Vector{Symbol}
end

function _group_title(info::DiagnosticInfo, name::Symbol)
    isempty(info.units) || return info.units
    dimension = info.dimension
    isnothing(dimension) && return info.label
    info.kind === :count && return "Count"
    info.kind === :defect && return "$(info.label) ($(string(dimension)))"
    quantity = get(_DIMENSION_QUANTITIES, dimension, "")
    return isempty(quantity) ? string(dimension) :
        "$quantity ($(string(dimension)))"
end

"""
    diagnostic_groups(infos)

Group diagnostics for plotting. `infos` is a vector of `name =>
DiagnosticInfo` pairs, a `DiagnosticCollection`, or a `SnapshotStore`.
Diagnostics share a group when kind, physical dimension, and scale agree.
Diagnostics without a dimension share a group only with diagnostics that have
the same nonempty `units`; otherwise each forms its own group.
"""
function diagnostic_groups(infos::AbstractVector{<:Pair{Symbol, DiagnosticInfo}})
    groups = DiagnosticGroup[]
    keys = Any[]
    for (name, info) in infos
        key = if isnothing(info.dimension)
            isempty(info.units) ? (:single, name) :
                (:units, info.kind, info.units, info.scale)
        else
            (:dimension, info.kind, info.dimension, info.scale)
        end
        index = findfirst(isequal(key), keys)
        if isnothing(index)
            push!(keys, key)
            push!(
                groups, DiagnosticGroup(
                    _group_title(info, name), info.kind, info.dimension, info.scale,
                    info.units, [name],
                )
            )
        else
            push!(groups[index].names, name)
        end
    end
    return groups
end

diagnostic_groups(collection::DiagnosticCollection) =
    diagnostic_groups(diagnostic_info(collection))
