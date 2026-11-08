# Single-phase flow tutorial

This tutorial solves the periodic Taylor--Green vortex and checks velocity
error and incompressibility. It is deliberately small enough to run while
editing a model.

## Problem

On ``[0,2\pi]^2``, the exact decaying velocity is

```math
u=e^{-2\nu t}\sin(x)\cos(y), \qquad
v=-e^{-2\nu t}\cos(x)\sin(y).
```

Create the mesh and initialize the high-level simulation directly from this
function:

```@example taylor-green
using TwoPhaseAMR

viscosity = 0.02
exact = (x, y, time, axis) -> exp(-2viscosity * time) *
    (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))

mesh = BlockTreeMesh(block_size=(12, 12), extent=(2pi, 2pi))
dt = 0.02 / 12
flow = FlowSimulation(
    mesh, SinglePhase(density=1.0, dynamic_viscosity=viscosity);
    velocity=exact, boundary=FreeSlip(), dt,
    diagnostics=(KineticEnergy(), DivergenceNorm()),
    numerics=NavierStokesOptions(
        multigrid=MultigridOptions(coarse_size=16)))
result = simulate!(
    flow, 0.02; tolerance=2e-10, max_cycles=80)
result.pressure.converged
```

## Check divergence

The projection uses the same finite-volume divergence as the pressure
operator. Compute it explicitly when validating a case:

```@example taylor-green
divergence = CellField(mesh)
divergence!(divergence, flow.velocity)
maximum_divergence = maximum(
    maximum(abs, interior(divergence, block))
    for block in eachindex(mesh.blocks))
maximum_divergence
```

## Check the velocity error

The staggered components have different dual volumes. Use the component data
already held by the diffusion operator when computing an ``L^2`` error:

```@example taylor-green
function velocity_error(flow, exact, time)
    squared_error = 0.0
    total_volume = 0.0
    for component in (flow.solver.diffusion.x, flow.solver.diffusion.y)
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            face = flow.solver.mesh.faces[global_index]
            difference = flow.velocity.interior[global_index] -
                exact(face.center[1], face.center[2], time, face.axis)
            squared_error += component.volumes[local_index] * difference^2
            total_volume += component.volumes[local_index]
        end
    end
    return sqrt(squared_error / total_volume)
end
velocity_l2 = velocity_error(flow, exact, result.time)
```

The package test suite repeats this problem on two uniform resolutions and a
static mixed-level mesh. See `examples/01_taylor_green/taylor_green.jl` for a command-line
version and [Navier--Stokes method](navier-stokes.md) for the split algorithm.
