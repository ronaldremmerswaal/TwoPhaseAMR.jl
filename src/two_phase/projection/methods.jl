"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Pressure-projection policy for the conservative multidimensional ghost-fluid method.

The supported preconditioners are `:block` (the default), `:ilu`, and `nothing`;
unsupported choices are rejected at construction.
"""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O

    function CMDGFMProjection(preconditioner, preconditioner_options)
        preconditioner in (:block, :ilu, nothing) || throw(
            ArgumentError("CMDGFM preconditioner must be :block, :ilu, or nothing"),
        )
        return new{typeof(preconditioner), typeof(preconditioner_options)}(
            preconditioner, preconditioner_options,
        )
    end
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)

"""Interface for the local kinetic-metric block of a [`MimeticProjection`](@ref)."""
abstract type AbstractCutCellMetric end

"""
    MFDCutCellMetric(; small_cell_stabilization=true, nonsimple_cells=:error)

Mimetic-finite-difference kinetic metric on the liquid and gas polygons of the
cut cells (candidate MM of the mimetic-projection audit): one block
`R Rᵀ/|P| + (1/2) D Q D` per polygon `P`, the chord flux slaved to closure, the
diagonal phase masses away from the interface. A phase value on a face that the
phase does not reach follows a donor. `small_cell_stabilization` adds the
audit's ad hoc tie `(1 - 2a) m (u - u_donor)²` for apertures `0 < a < 1/2` on
mixed faces.

A cut cell whose aperture polygon is not simple (more than one chord,
degenerate, or undetermined wet face end) throws `ArgumentError` for
`nonsimple_cells = :error`. `:lumped` instead gives its face values the lumped
mass `ρh²/2`, the fallback of the audit's laboratory; this loses the metric's
consistency in that cell and is reported in `MimeticProjectionStats`.
"""
struct MFDCutCellMetric <: AbstractCutCellMetric
    small_cell_stabilization::Bool
    nonsimple_cells::Symbol

    function MFDCutCellMetric(small_cell_stabilization::Bool, nonsimple_cells::Symbol)
        nonsimple_cells in (:error, :lumped) || throw(
            ArgumentError("nonsimple_cells must be :error or :lumped"),
        )
        return new(small_cell_stabilization, nonsimple_cells)
    end
end

MFDCutCellMetric(; small_cell_stabilization::Bool = true, nonsimple_cells::Symbol = :error) =
    MFDCutCellMetric(small_cell_stabilization, nonsimple_cells)

"""
    MimeticProjection(; metric=MFDCutCellMetric())

Kinetic-energy-orthogonal two-velocity projection onto vanishing
aperture-weighted mixture divergence in the non-diagonal cut-cell metric
`metric` (no normal-jump rows: in a cut cell the mixture divergence imposes the
normal velocity jump). Surface tension and the pressure gradient use the same
map `𝓜⁻¹Dᵀ`; gravity is the uniform acceleration of both phases. An interior
face between a pure liquid and a pure gas cell carries one shared phase value.
Supported on uniform square-cell meshes without embedded solids,
for interfaces whose cut cells are simple and do not touch a domain boundary;
other configurations throw `ArgumentError`. [`CMDGFMProjection`](@ref) remains
the default.
"""
struct MimeticProjection{M <: AbstractCutCellMetric} <: AbstractTwoVelocityProjection
    metric::M
end

function MimeticProjection(; metric = MFDCutCellMetric())
    metric isa AbstractCutCellMetric || throw(
        ArgumentError("metric must be an AbstractCutCellMetric, e.g. MFDCutCellMetric()"),
    )
    return MimeticProjection(metric)
end
