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

The projection decides the pressure and force discretisation; nothing else
has to be selected. `CMDGFMProjection()` (the default) is the thesis'
conservative multidimensional ghost-fluid projection.
`TwoVelocityModel(pressure_projection=MimeticProjection())` selects the
mimetic projection: the kinetic-energy-orthogonal projection in a
non-diagonal metric built, per cut cell and phase, on the Gram tensor of the
wetted face parts (the shared face apertures and their midpoints; no phase
polygon is needed). Surface tension goes through the same gradient map as a
per-phase force: each phase has its own capillary potential, so a drop of
constant curvature is exactly at rest and a varying curvature drives the
liquid, not a thin gas layer next to it. Gravity is the uniform acceleration
of both phases, and the viscous increment enters as an impulse through the
metric (the common increment of both phases away from the interface). It
takes no options and works with either surface-tension formulation; it
builds its own pressure
solver, so non-default `pressure_*` solver settings of the system throw
`ArgumentError`. It runs on uniform meshes of
square cells without solids and throws for adaptive meshes, solids and cut
cells at a wall. A cut cell whose Gram tensor admits no consistent block gets
a least-squares fallback and is counted in
`MimeticProjectionStats.defective_cells`; the run summary prints the totals of
a `simulate!` call. Each projection solves a conjugate-gradient system with a
sparse factorization of the cut-cell block, so it is more expensive than the
CMDGFM.
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
