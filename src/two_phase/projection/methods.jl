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

"""
    MimeticProjection()

Kinetic-energy-orthogonal two-velocity projection onto vanishing
aperture-weighted mixture divergence in a non-diagonal cut-cell kinetic metric
(no normal-jump rows: in a cut cell the mixture divergence imposes the normal
velocity jump). The pressure gradient uses the map `𝓜⁻¹Dᵀ`; surface tension
is a per-phase force through the same map, with the capillary potentials
`σκH` (gas) and `σκH - σκ` (liquid) of a cell curvature `κ` that is defined on
every cell carrying both phases, so that a constant curvature is exactly
balanced and a varying one drives the liquid only. Gravity is the uniform
acceleration of both phases, and the viscous increment of the mixture enters
as the impulse of the staggered phase masses through the metric, `𝓜⁻¹WPδ`.
An interior face between a pure liquid and a pure gas cell carries one
shared phase value.

The metric is built per cut cell and phase from the Gram tensor
`G = Σ_k b_k h n_k (x_k - x*)ᵀ` of the wetted sides (wetted lengths `b_k h`
from the shared face apertures, outward normals `n_k`, wetted midpoints `x_k`):
the block is `R G⁺ Rᵀ + S` with the rows `R = [b_k h (x_k - x*)ᵀ]`. The
pressure point `x*` lies on the line where `G` is symmetric, next to the
interface of the data and inside the interval of that line where both phases'
Gram tensors are positive semi-definite (for cells whose wetted parts balance,
`η = Σ_k b_k h n_k = 0`: where `ker G ⊂ ker R`), and `S` is the MFD
stabilisation of the wetted sides closed by one slaved edge. No phase polygon
is formed: the shared apertures need not bound one. Where they do, `G = |P| I`
and the block is the classical MFD block of the polygon. A cell without a
consistent block (no admissible pressure point) gets least-squares consistent
blocks (eigenvalues raised to `0.02h²`) and is counted in
[`MimeticProjectionStats`](@ref) `.defective_cells`. Away from the interface
the metric is the diagonal phase mass. A phase value on a face that the phase
does not reach follows a donor, and a value with aperture `0 < a < 1/2` on a
mixed face is tied to a donor with weight `(1 - 2a)` times its mass
(small-cell stabilisation).

Supported on uniform square-cell meshes without embedded solids, for
interfaces whose cut cells do not touch a domain boundary; other
configurations throw `ArgumentError`. [`CMDGFMProjection`](@ref) remains
the default.
"""
struct MimeticProjection <: AbstractTwoVelocityProjection end
