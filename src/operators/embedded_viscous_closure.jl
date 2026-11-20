"""Abstract supertype of the cut-cell closure of the single-phase viscous
operator next to an embedded solid ([`GeometryField`](@ref)). See
[`WallDistanceClosure`](@ref) (the default) and [`LSStagClosure`](@ref).
"""
abstract type AbstractEmbeddedViscousClosure end

"""
    WallDistanceClosure()

The default cut-cell closure of the single-phase viscous operator: every
velocity value sits at its face centre, the interior contacts of a cut cell
use rebuilt open measures, and the wall flux of a cut-cell PLIC segment is
split among the dual faces, each piece divided by the perpendicular distance
from the face centre to the reconstructed plane, floored at one sixty-fourth
of the cell width. On a wall that is not grid aligned the velocity is first
order (#110): a face whose centre lies in the solid stays an unknown and its
wall conductance has the wrong sign for the open part it represents.
"""
struct WallDistanceClosure <: AbstractEmbeddedViscousClosure end

"""
    LSStagClosure()

LS-STAG cut-cell closure of the single-phase viscous operator (Cheny and
Botella, J. Comput. Phys. 229 (2010) 1043, doi:10.1016/j.jcp.2009.10.007),
built from the PLIC data of the [`GeometryField`](@ref). It replaces the rows
of the velocity values on and next to cut cells; the rest of the operator,
the mass (the open staggered volumes), the divergence and the gradient are
unchanged.

- Every face with a positive open aperture stays an unknown. Its value is the
  velocity at the midpoint of the open part of the face, which always lies in
  the fluid.
- The normal viscous stress of a cut cell is its Green gradient,
  `(1/|c|) Σ_f ±a_f|f| u_f` with the wall value zero, so its trace is the
  open divergence.
- The shear stress along the grid line of a face differences the open
  midpoints of neighbouring faces; at a cut end it is the one-sided quotient
  to the wall at half the open length, and the wall pieces close every half
  control volume, so an affine velocity that vanishes on the wall has zero
  residual. A fine line that ends at a coarser cell takes the linear-exact
  derivative through the two faces of that cell.

The stiffness stays symmetric positive definite. The velocity is second order
at oblique walls, uniform and adaptive (2:1) meshes. No distance floor is
used: a thin fluid layer has the large conductance of its width. It applies to
the single-phase operator; the two-phase stress operator keeps its closure.
"""
struct LSStagClosure <: AbstractEmbeddedViscousClosure end
