"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Rule by which a CMDGFM jump row infers the phase velocities of its mixed
face from the neighbouring faces. See [`LinearJumpInference`](@ref) (the
default) and [`ConstantJumpInference`](@ref) (the published rows)."""
abstract type AbstractJumpInference end

"""
    LinearJumpInference()

The CMDGFM jump inference (the default, issue #171). Compared with the
published rows ([`ConstantJumpInference`](@ref)):

- each phase is extrapolated linearly from two donors on its side of the
  interface, `u(h₁) + θ (u(h₁) - u(h₂))`, in the velocity and in the matching
  jump and pressure-gradient columns; for a smooth single-valued field and
  `θ = 1` the inferred jump is `O(h³)` instead of `O(h)`;
- the slope weight `θ ∈ [0, 1]` of a row is one minus a smoothstep of the
  interface crossing speed `w/U` over the ramp `[1/16, 3/16]`, with `w` the
  normal component of the mixture velocity at the face and `U` the local
  mixture speed: an interface that moves through the face gets the one-donor
  rule, a quasi-static one (a shear layer) the linear extrapolation;
- a mixed face without a regular row gets a row that makes the interpolated
  normal jump of the projected state vanish where `|n_t| ≤ |n_a|`, and the
  tangential row below where `|n_t| > |n_a|`, instead of keeping its jump;
- a mixed face with an exactly tangential staggered normal gets the row with
  `β = 0` and no transverse term, imposed as an increment of the face's own
  jump.

Where a row would use a face across a refinement-level transition or a face
reduced by a solid, or where a second donor is missing, does not carry its
phase or is itself a mixed face, that face falls back to the rule of
[`ConstantJumpInference`](@ref); the counts are reported in
`CMDGFMProjectionStats.jump_inference`.
"""
struct LinearJumpInference <: AbstractJumpInference end

"""
    ConstantJumpInference()

The CMDGFM jump rows as published (the thesis reference). A row whose
staggered normal is closer to the transverse axis infers each phase from one
same-axis donor face next to the mixed face (a constant extrapolation of each
phase); faces without a row keep their jump. In a resolved shear layer the
inferred jump is the two-point difference `2U'h` of the shear (issue #171).
"""
struct ConstantJumpInference <: AbstractJumpInference end

"""Pressure-projection policy for the conservative multidimensional ghost-fluid method.

The supported preconditioners are `:block` (the default), `:ilu`, and `nothing`;
`jump_inference` is [`LinearJumpInference()`](@ref LinearJumpInference)
(the default) or the published rows
[`ConstantJumpInference()`](@ref ConstantJumpInference).
Unsupported choices are rejected at construction.
"""
struct CMDGFMProjection{P, O, J <: AbstractJumpInference} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O
    jump_inference::J

    function CMDGFMProjection(
            preconditioner, preconditioner_options,
            jump_inference = LinearJumpInference(),
        )
        preconditioner in (:block, :ilu, nothing) || throw(
            ArgumentError("CMDGFM preconditioner must be :block, :ilu, or nothing"),
        )
        _validate_jump_inference(jump_inference)
        return new{
            typeof(preconditioner), typeof(preconditioner_options), typeof(jump_inference),
        }(
            preconditioner, preconditioner_options, jump_inference,
        )
    end
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
    jump_inference = LinearJumpInference(),
) = CMDGFMProjection(preconditioner, preconditioner_options, jump_inference)

_validate_jump_inference(jump_inference) =
    jump_inference isa AbstractJumpInference || throw(
    ArgumentError(
        "CMDGFM jump_inference must be LinearJumpInference() or ConstantJumpInference()",
    ),
)

"""
    MimeticProjection()

Kinetic-energy-orthogonal two-velocity projection onto vanishing
aperture-weighted mixture divergence in a non-diagonal cut-cell kinetic metric
(no normal-jump rows: in a cut cell the mixture divergence imposes the normal
velocity jump). The pressure gradient uses the map `𝓜⁻¹Dᵀ`; surface tension
is a per-phase force through the same map, with the capillary potentials
`σκH` (gas) and `σκH - σκ` (liquid) of a cell curvature `κ` that is defined on
every cell carrying both phases, so that a constant curvature is exactly
balanced and a varying one drives the liquid only. Gravity is the Gram moment
of the uniform acceleration of both phases, and the viscous step of the mixture stress is
an implicit midpoint in the metric (non-increasing in `𝓜` for every time step).
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
the metric is the diagonal phase mass, except in a cell next to two finer
cells on an adaptive mesh: its side of two faces makes it a polygon of five
to eight edges, whose Gram block (`G = h² I`, centre `x*`) reproduces constant
fields, so a hydrostatic or translating state stays at rest across a level
transition. A phase value on a face that the phase does not reach follows a
donor with the tie weight `ρ_π V_f` (`V_f` the staggered dual volume), and a
value with aperture `0 < a < 1/2` on a mixed face is tied to a donor with
weight `(1 - 2a) ρ_π V_f` (small-cell stabilisation): the ties vanish on
constants, are continuous in `a` and keep `𝓜 ⪰ c diag(ρ_π V_f)` with `c`
independent of the apertures.

A cut cell at a domain wall keeps its wall side as a side of the Gram tensor;
the prescribed wall flux is not a value of the metric, so the block is
restricted to the interior sides. Supported on uniform and adaptive (2:1
graded) meshes of square cells without embedded solids; other configurations
throw `ArgumentError`. [`CMDGFMProjection`](@ref) remains the default.
"""
struct MimeticProjection <: AbstractTwoVelocityProjection end
