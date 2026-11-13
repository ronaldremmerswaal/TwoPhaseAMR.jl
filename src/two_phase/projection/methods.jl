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
    MFDCutCellMetric(; small_cell_stabilization=true)

Mimetic-finite-difference kinetic metric of the cut cells (candidate MMd of
the mimetic-projection audit's note on kinetic metrics). Per cut cell and
phase the block is `R G⁺ Rᵀ + S` with the Gram tensor
`G = Σ_k b_k h n_k (x_k - x*)ᵀ` of the wetted sides (wetted lengths `b_k h`
from the shared face apertures, outward normals `n_k`, wetted midpoints `x_k`)
and the rows `R = [b_k h (x_k - x*)ᵀ]`. The pressure point `x*` lies on the
line where `G` is symmetric, next to the interface of the data, and `S` is the
MFD stabilisation of the wetted sides closed by one slaved edge. Away from the
interface the metric is the diagonal phase mass. No phase polygon is formed:
the shared apertures need not bound one. Where they do, `G = |P| I` and the
block is the classical MFD block of the polygon (candidate MM). A cell whose
Gram tensor has no consistent block gets least-squares consistent blocks
(eigenvalues raised to `0.02h²`) and is counted in
`MimeticProjectionStats.defective_cells`.

A phase value on a face that the phase does not reach follows a donor.
`small_cell_stabilization` adds the audit's ad hoc tie
`(1 - 2a) m (u - u_donor)²` for apertures `0 < a < 1/2` on mixed faces.
"""
struct MFDCutCellMetric <: AbstractCutCellMetric
    small_cell_stabilization::Bool
end

MFDCutCellMetric(; small_cell_stabilization::Bool = true) =
    MFDCutCellMetric(small_cell_stabilization)

"""
    MimeticProjection(; metric=MFDCutCellMetric())

Kinetic-energy-orthogonal two-velocity projection onto vanishing
aperture-weighted mixture divergence in the non-diagonal cut-cell metric
`metric` (no normal-jump rows: in a cut cell the mixture divergence imposes the
normal velocity jump). Surface tension and the pressure gradient use the same
map `𝓜⁻¹Dᵀ`; gravity is the uniform acceleration of both phases. An interior
face between a pure liquid and a pure gas cell carries one shared phase value.
Supported on uniform square-cell meshes without embedded solids, for
interfaces whose cut cells do not touch a domain boundary; other
configurations throw `ArgumentError`. [`CMDGFMProjection`](@ref) remains
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
