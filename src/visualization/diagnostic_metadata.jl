"""
    PhysicalDimension(mass, length, time)

Exponents of mass, length, and time of a diagnostic. Two-dimensional
quantities are per unit depth: an energy is `PhysicalDimension(1, 1, -2)`.
Used to group diagnostics with the same physical meaning; see
[`diagnostic_groups`](@ref).
"""
struct PhysicalDimension
    mass::Int8
    length::Int8
    time::Int8
end

PhysicalDimension(mass::Integer, length::Integer, time::Integer) =
    PhysicalDimension(Int8(mass), Int8(length), Int8(time))

const _DIAGNOSTIC_KINDS = (:physical, :count, :solver, :defect)
const _DIAGNOSTIC_SCALES = (:linear, :log)

"""
    DiagnosticInfo(; label, units="", dimension=nothing, components=String[],
                   kind=:physical, scale=:linear)

Display metadata of a diagnostic: a `label`, optional display `units` (for
example `"J/m"`; empty for nondimensional runs), its [`PhysicalDimension`](@ref)
or `nothing`, names of vector `components` (for example `["x", "y"]`), the
`kind` (`:physical`, `:count`, `:solver`, or `:defect`), and the preferred
axis `scale` (`:linear` or `:log`). See [`diagnostic_info`](@ref).
"""
struct DiagnosticInfo
    label::String
    units::String
    dimension::Union{Nothing, PhysicalDimension}
    components::Vector{String}
    kind::Symbol
    scale::Symbol

    function DiagnosticInfo(label, units, dimension, components, kind, scale)
        kind in _DIAGNOSTIC_KINDS || throw(
            ArgumentError("diagnostic kind must be one of $(_DIAGNOSTIC_KINDS)"),
        )
        scale in _DIAGNOSTIC_SCALES || throw(
            ArgumentError("diagnostic scale must be :linear or :log"),
        )
        return new(
            string(label), string(units), dimension,
            String[string(component) for component in components], kind, scale,
        )
    end
end

DiagnosticInfo(;
    label, units = "", dimension = nothing, components = String[],
    kind = :physical, scale = :linear,
) = DiagnosticInfo(label, units, dimension, components, kind, scale)

Base.:(==)(a::DiagnosticInfo, b::DiagnosticInfo) =
    a.label == b.label && a.units == b.units && a.dimension == b.dimension &&
    a.components == b.components && a.kind == b.kind && a.scale == b.scale
Base.hash(info::DiagnosticInfo, seed::UInt) = hash(
    (info.label, info.units, info.dimension, info.components, info.kind, info.scale),
    hash(DiagnosticInfo, seed),
)

_default_diagnostic_label(name::Symbol) =
    uppercasefirst(replace(string(name), '_' => ' '))
