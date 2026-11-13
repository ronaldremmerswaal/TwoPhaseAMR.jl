# Choosing a flow model

Choose the physical model first and the numerical methods second. The
high-level API makes this distinction explicit so that an initializer or an
unrelated keyword cannot silently change the equations being solved.

## Single-phase flow

Use [`SinglePhase`](@ref) when one density and dynamic viscosity describe the
whole fluid domain:

```julia
physics = SinglePhase(density=1.0, dynamic_viscosity=1e-3)
flow = FlowSimulation(mesh, physics)
```

## One-velocity two-phase flow

Use [`TwoPhase`](@ref) with the default [`OneVelocityModel`](@ref) when both
phases share one velocity field. This is the default for two-phase problems:

```julia
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1e-3, 1.8e-5),
    surface_tension=0.072,
)
flow = FlowSimulation(mesh, physics; velocity_model=OneVelocityModel())
```

The numeric surface-tension shorthand selects the default conservative
ghost-fluid method, [`CGFM`](@ref).

## Two-velocity two-phase flow

Use [`TwoVelocityModel`](@ref) when the liquid and gas need separate intrinsic
velocities, for example across a tangential velocity jump:

```julia
model = TwoVelocityModel(
    pressure_projection=CMDGFMProjection(),
)
flow = FlowSimulation(mesh, physics; velocity_model=model)
```

A shared initializer is copied to both phase fields. To initialize different
velocities, pass a `(liquid, gas)` tuple of callables or `FaceVelocity`
objects. The tuple never selects the velocity model implicitly.

`TwoVelocityModel(pressure_projection=MimeticProjection())` selects the
optional mimetic cut-cell projection: the kinetic-energy-orthogonal
projection in a non-diagonal metric built, per cut cell and phase, on the
Gram tensor of the wetted face parts (the shared face apertures and their
midpoints; no phase polygon is needed). It runs on uniform meshes of square
cells without solids and throws for adaptive meshes, solids and cut cells at
a wall. A cut cell whose Gram tensor admits no consistent block gets a
least-squares fallback and is counted in
`MimeticProjectionStats.defective_cells`, which the run summary prints. Each
projection solves a conjugate-gradient system with a sparse factorization of
the cut-cell block, so it is more expensive than the CMDGFM.
See the technical reference, section
[Mimetic cut-cell projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:projection:mimetic).

## Option matrix

| Decision | Choices | Default | Applies to |
| --- | --- | --- | --- |
| velocity formulation | `OneVelocityModel()`, `TwoVelocityModel(...)` | one velocity | two-phase flow |
| surface-tension force | `CGFM()`, `CSF()` | CGFM | either two-phase velocity model |
| pressure projection | `CMDGFMProjection()`, `MimeticProjection()` | CMDGFM | two velocity only |

Surface tension is configured by [`SurfaceTension`](@ref), while the
projection is a field of [`TwoVelocityModel`](@ref). This ownership makes
invalid combinations visible at construction time. See
[Two-phase flow](two-phase-flow.md) and
[Pressure gradient](pressure-gradient.md) for the numerical definitions.
