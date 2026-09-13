# TwoPhaseAMR.jl

This package is being developed as a sequence of tested vertical slices. The
public API covers the centred single-phase foundation, uniform/mixed-level
staggered operators, pressure projection, wall diffusion, and the fixed-tree
single-phase Navier--Stokes solver from milestones 1--4 of `DEVELOPMENT.md`.

## Data model

Each `RegionTrees.Cell` leaf represents one fixed-size Cartesian block. A
`CellField` owns one dense Julia `Matrix` per leaf, including a configurable
halo. Mesh construction turns the mutable tree into immutable leaf, face, and
halo schedules. Numerical kernels traverse those schedules and arrays only.

The face schedule subdivides a coarse face where necessary and gives each
resulting physical segment exactly one owner. Conservative updates add one
integrated flux to one cell and subtract the same value from its neighbor.

The [compatible pressure gradient](pressure-gradient.md) leaves this primal
Gauss divergence unchanged. A precomputed inverse face Riesz map is diagonal
on conforming regions and adds sparse affine-exact couplings at coarse--fine
vertices, making the composite pressure operator symmetric and stable.

`FaceField{Axis}` stores compact directional face degrees of freedom. The
staggered-advection workspace builds dual-control-volume connectivity from the
immutable centred face schedule. Its upwind, centred, and Lax--Wendroff kernels
are allocation-free after setup. A separate `amr_interface_scheme` choice
inherits the bulk `scheme` when omitted, so the public default is
Lax--Wendroff throughout the mesh. See
[Staggered advection](staggered-advection.md) for the discrete construction,
the divergence-on-the-diagonal identity, and the interface stability results.

The [geometric multigrid preconditioner](poisson.md) coarsens dyadic cell geometry
inside and across blocks, inherits symmetric Galerkin pressure operators,
fixes the periodic/all-Neumann pressure nullspace with a volume-weighted zero
mean, and projects a mixed-level face velocity to the requested residual
tolerance. Krylov.jl owns the iteration, with CG for pressure and BiCGSTAB for
staggered diffusion by default. Geometric MG, AlgebraicMultigrid.jl,
IncompleteLU.jl, and LimitedLDLFactorizations.jl preconditioners share one
interface; direct LU in geometric MG is confined to the bottom levels.

The [single-phase Navier--Stokes solver](navier-stokes.md) applies one
second-order LW convection update, one Crank--Nicolson/implicit-midpoint
diffusion solve, and exactly one pressure projection per global time step.
No-permeation is automatic on non-periodic sides, while the viscous wall
closure is no-slip by default and can be changed to free-slip.

The [implicit-diffusion solver review](implicit-diffusion-solver-review.md)
compares FAC/AMR geometric multigrid, AMG, incomplete factorizations,
matrix-free alternatives, and L-stable time integrators. It also records the
measured hierarchy complexity and the proposed stable-core preconditioner.

An optional [interactive visualization](visualization.md) extension renders
owned cell fields and mixed-level block geometry with either GLMakie or
WGLMakie without adding a graphics dependency to headless solver runs.

## Scope

The package is 2-D, CPU-only, serial, refinement-ratio-two, and globally
time-stepped. The compatible gradient supports uniform and isotropic
mixed-level meshes with periodic or homogeneous-Neumann pressure-correction
boundaries. Staggered advection and diffusion support 2:1-balanced
mixed-level meshes and uniform multi-block partitions. Coarse--fine transfer
is encoded directly by the single-owner segment fractions in immutable dual
schedules; no duplicate coarse face unknown is synchronized afterward. The
tree is fixed during a Navier--Stokes solve. The literature and architecture
documents state the remaining compatibility conditions for VOF, dynamic AMR,
and distributed execution.
