# TwoPhaseAMR.jl

This package is developed as a sequence of tested vertical slices. The public
API covers centred and staggered operators, pressure projection, wall
diffusion, dynamic refinement, geometric VOF, and a one-velocity
Navier--Stokes solver for both single- and two-phase flow.

For standard use, `FlowSimulation` owns the evolving fields, solver,
workspace, and continuation counters:

```julia
mesh = BlockTreeMesh(block_size=(32, 32))
flow = FlowSimulation(
    mesh; viscosity=1e-3, velocity=initial_velocity, dt=1e-3)
result = simulate!(flow, 1.0)
```

The complete operator-level interface remains public for numerical research
and custom integrators. Start with [Getting started](getting-started.md), then
use [Advanced workflows](advanced-workflows.md) when explicit ownership or
solver composition is needed.

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
and its Fromm, minmod, MC, van Leer, and superbee three-point kernels are
allocation-free after setup. A separate `amr_interface_scheme` choice uses a
scheme-aware default: Upwind remains global Upwind, centred/LW use MC, and
other limited schemes retain themselves. Interface-only Upwind is rejected. See
[Staggered advection](staggered-advection.md) for the discrete construction,
the divergence-on-the-diagonal identity, and the interface stability results.

The [geometric multigrid preconditioner](poisson.md) coarsens dyadic cell geometry
inside and across blocks, inherits symmetric Galerkin pressure operators,
fixes the periodic/all-Neumann pressure nullspace with a volume-weighted zero
mean, and projects a mixed-level face velocity to the requested residual
tolerance. Krylov.jl owns the iteration. The automatic methods are CG and
BiCGSTAB for the original single-phase pressure and diffusion systems, and
FGMRES for the variable-coefficient two-phase systems. Geometric MG, AlgebraicMultigrid.jl,
IncompleteLU.jl, and LimitedLDLFactorizations.jl preconditioners share one
interface; direct LU in geometric MG is confined to the bottom levels.

The [Navier--Stokes solver](navier-stokes.md) applies one transport update, one
Crank--Nicolson/implicit-midpoint diffusion solve, and exactly one pressure
projection per global time step. When an interface is present, transport uses
shared Weymouth--Yue phase-volume fluxes, diffusion uses the coupled
variable-viscosity stress, and the same step adds balanced phase forces before
projection. Persistent density, scalars, and velocity remain intrinsic values.
No-permeation is automatic on non-periodic sides, while the viscous wall
closure is no-slip by default and can be changed to free-slip.

The [dynamic-refinement framework](adaptive-refinement.md) combines centred
block indicators with refine/keep/coarsen precedence, temporal hysteresis, 2:1
grading, conservative cell/face transfer, and divergence-preserving staggered
velocity prolongation. It mutates the same `BlockTreeMesh` and rebuilds solver
operators only after an actual topology change. A localized manufactured
vortex validation demonstrates near-second-order maximum-level convergence
and the same error as a uniform finest mesh with 57.8% of its active cells.

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
schedules; no duplicate coarse face unknown is synchronized afterward.
Dynamic refinement is available for both models with one global time step;
level subcycling and its flux-register synchronization are not. Two-phase
adaptation provides interface-band, curvature, and vorticity criteria and
geometric PLIC fraction transfer. Distributed execution is not implemented.
