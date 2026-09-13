# Staggered advection

## Discrete construction

`FaceField{Axis}` contains one scalar per owned face normal to `Axis`. A
staggered control volume surrounds each such face. Its boundary fluxes are not
sampled or reconstructed independently: they are the algebraic, area-weighted
average of the two neighbouring centred face-normal velocities. For a uniform
Cartesian grid, for example,

```math
F^u_{i,j+1/2} = \frac{\Delta x}{2}
                 \left(v_{i-1/2,j+1/2}+v_{i+1/2,j+1/2}\right).
```

This is the centred-to-staggered flux construction selected in the literature
review. Every dual face is stored once and contributes equal-and-opposite
integrated fluxes to its two neighbouring staggered volumes, so each component
of linear momentum is conserved to roundoff on a periodic mesh.

For an oriented dual-face flux `F`, neighbouring states `q⁻` and `q⁺`, normal
speed `a`, spacing `h`, and time step `Δt`, the upwind and centred face states
are

```math
q_\mathrm{up} = \begin{cases}q^- & F \ge 0,\\q^+ & F < 0,\end{cases}
\qquad
q_\mathrm{cen} = \frac{q^-+q^+}{2},
```

The Lax--Wendroff state follows the multidimensional construction in
Remmerswaal's thesis. It first linearly interpolates the upwind state in the
tangential direction to the line through the donating-region centroid, then
interpolates that value and the downwind state in the normal direction. When
the tangential velocity is zero, this reduces to

```math
q_\mathrm{LW} = \frac{q^-+q^+}{2}
                 - \frac{\Delta t\,a}{2h}(q^+-q^-),
```

the classical one-dimensional Lax--Wendroff state. The local staggered CFL
bound controls both interpolation fractions.

The public scheme values are `Upwind()`, `Centred()`, and `LaxWendroff()`;
`Centered()` and `LW()` are aliases. `convection!` evaluates the operator,
while `advect!` applies its forward-Euler conservative update. A
`StaggeredVelocityAdvectionWorkspace` advances both velocity components from
the same old-time advecting field.

## Skew-symmetry invariant

Let `D_c` be the cell divergence and `I_F` the algebraic flux interpolation.
The dual divergence satisfies

```math
D_f I_F = I_C D_c.
```

With the centred face state, summation by parts on a periodic uniform mesh
therefore gives

```math
\langle p,C(u)q\rangle_V = -\langle C(u)p,q\rangle_V
\quad\text{when}\quad D_c u=0.
```

The tests assemble the complete single-block convection matrices for both
directions and check `C ≈ -transpose(C)`. They also verify the bilinear identity
with a nonconstant, discretely divergence-free vortex and confirm that
`⟨q,Cq⟩` vanishes to roundoff. This is a semi-discrete property; a forward
Euler step with the centred operator does not by itself preserve kinetic
energy in time.

## Current scope

The dual schedule currently supports uniform-resolution meshes with periodic
boundaries in both directions, including any equivalent uniform tree-block
partition. Workspace construction rejects physical boundaries and mixed-level
trees. Those cases require boundary dual volumes and conforming coarse--fine
face constraints; silently applying the uniform stencil would violate the
documented ownership and compatibility invariants.
