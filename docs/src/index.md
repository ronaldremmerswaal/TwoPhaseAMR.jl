# TwoPhaseAMR.jl

TwoPhaseAMR is a two-dimensional incompressible-flow package for
block-structured quadtree meshes. It provides single-phase flow, geometric
VOF, one- and two-velocity two-phase models, embedded PLIC solids, dynamic
mesh adaptation, and tree-geometric multigrid.

The high-level interface separates five concerns:

1. [`BlockTreeMesh`](@ref) describes topology and AMR policy;
2. [`SinglePhase`](@ref) or [`TwoPhase`](@ref) describes physics;
3. [`OneVelocityModel`](@ref) or [`TwoVelocityModel`](@ref) selects the
   two-phase kinematics;
4. [`NavierStokesOptions`](@ref) and [`Adaptivity`](@ref) select numerics;
5. [`FlowSimulation`](@ref) owns evolving state, workspace, time, and
   diagnostics.

```julia
using TwoPhaseAMR

mesh = BlockTreeMesh(block_size = (16, 16))
physics = SinglePhase(density = 1.0, dynamic_viscosity = 0.01)
flow = FlowSimulation(
    mesh, physics;
    velocity = (x, y, time, axis) ->
        axis == 1 ? sinpi(2x) * cospi(2y) :
                    -cospi(2x) * sinpi(2y),
    diagnostics = (KineticEnergy(), DivergenceNorm()),
)
result = simulate!(flow, 0.1)
```

Start with [Getting started](getting-started.md). The [Public API](api.md)
contains the complete configuration matrix, [Diagnostics](diagnostics.md)
shows simulation-owned histories, and [Recording and visualization](visualization.md)
describes live Studio and static output.

## What is implemented

- uniform and balanced mixed-level block meshes;
- conservative cell and staggered-face operators across coarse--fine faces;
- geometric interface initialization, PLIC reconstruction, and
  Weymouth--Yue transport;
- single-, one-velocity two-phase, and two-velocity Navier--Stokes evolution;
- CSF and CGFM surface tension;
- the CMDGFM two-velocity pressure projection;
- embedded fixed solids with no-slip or free-slip walls;
- dynamic interface, curvature, geometry, and vorticity refinement criteria;
- reusable Krylov and tree-geometric multigrid workspaces;
- simulation-owned diagnostics, recorded AMR frames, a browser-based Studio
  workspace, and optional Makie figures and animation export.

The package exposes its operator-level interfaces as well. Numerical-method
pages document conservation, compatible gradients, staggered transport,
diffusion, pressure projection, and multigrid in detail.

## Scope

The current implementation is 2-D and shared-memory. It uses one global time
step. Three-dimensional meshes, distributed execution, level subcycling, and
general pressure Dirichlet data are outside the current scope.
