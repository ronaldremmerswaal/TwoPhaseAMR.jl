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
    interface_transport=FrozenInterfaceFluxes(),
    pressure_projection=CMDGFMProjection(),
)
flow = FlowSimulation(mesh, physics; velocity_model=model)
```

A shared initializer is copied to both phase fields. To initialize different
velocities, pass a `(liquid, gas)` tuple of callables or `FaceVelocity`
objects. The tuple never selects the velocity model implicitly.

## Option matrix

| Decision | Choices | Default | Applies to |
| --- | --- | --- | --- |
| velocity formulation | `OneVelocityModel()`, `TwoVelocityModel(...)` | one velocity | two-phase flow |
| surface-tension force | `CGFM()`, `CSF()`, `GFM(conservative=false)` | CGFM | either two-phase velocity model |
| interface carrier | `FrozenInterfaceFluxes()`, `ProjectedLiquidVelocity()` | frozen | two velocity only |
| pressure projection | `CMDGFMProjection()`, `MimeticProjection()` | CMDGFM | two velocity only |

Surface tension is configured by [`SurfaceTension`](@ref), while the carrier
and projection are fields of [`TwoVelocityModel`](@ref). This ownership makes
invalid combinations visible at construction time. See
[Two-phase flow](two-phase-flow.md) and
[Pressure gradient](pressure-gradient.md) for the numerical definitions.
