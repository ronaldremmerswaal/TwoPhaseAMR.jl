# Refactor API migration

The typed high-level API separates physics, velocity formulation, numerical
policy, and adaptivity. Existing low-level operator constructors remain
available. The keyword-only `FlowSimulation(mesh; density=...,
dynamic_viscosity=...)` path translates to the new implementation and emits an
actionable deprecation warning.

Legacy high-level spellings will remain available throughout the `0.2.x`
series and will not be removed before `0.3`. Compatibility aliases that are
useful to low-level research code may remain longer.

## Constructor migration

```julia
# Before
flow = FlowSimulation(
    mesh;
    density=1.0,
    dynamic_viscosity=1e-3,
    velocity=initial_velocity,
)

# Now
physics = SinglePhase(density=1.0, dynamic_viscosity=1e-3)
flow = FlowSimulation(mesh, physics; velocity=initial_velocity)
```

For two phases:

```julia
physics = TwoPhase(
    interface;
    density=(rho_liquid, rho_gas),
    dynamic_viscosity=(mu_liquid, mu_gas),
    surface_tension=SurfaceTension(sigma; method=CGFM()),
    gravity=(0.0, -gravity_acceleration),
)
flow = FlowSimulation(
    mesh, physics;
    velocity_model=TwoVelocityModel(
        interface_transport=FrozenInterfaceFluxes(),
        pressure_projection=CMDGFMProjection(),
    ),
)
```

## Name and keyword mapping

| Legacy spelling | Preferred spelling |
| --- | --- |
| `NavierStokesSolver(...)` | `NavierStokesSystem(...)` for direct equation-system construction |
| `NavierStokesSimulation(...)` | `FlowSimulation(...)` |
| keyword-only `FlowSimulation(mesh; density=..., ...)` | `FlowSimulation(mesh, SinglePhase(...); ...)` or `FlowSimulation(mesh, TwoPhase(...); ...)` |
| `surface_tension_formulation=CGFM()` | `SurfaceTension(coefficient; method=CGFM())`; a number is also accepted |
| `interface_method=FrozenInterfaceFluxes()` | `TwoVelocityModel(interface_transport=FrozenInterfaceFluxes())` |
| `two_velocity_projection=CMDGFMProjection()` | `TwoVelocityModel(pressure_projection=CMDGFMProjection())` |
| `GeometricMultigrid*` | corresponding `TreeGeometricMultigrid*` name |

Do not pass both `interface_transport` and the compatibility keyword
`interface_method`. A liquid/gas velocity tuple no longer selects the model;
pass `velocity_model=TwoVelocityModel()` explicitly. New code should group
advection, pressure, diffusion, and multigrid choices in
[`NavierStokesOptions`](@ref), and refinement criteria/additional fields in
[`Adaptivity`](@ref).

## Diagnostics migration

Viewer-local five-argument diagnostic callbacks remain accepted, but reusable
diagnostics should be owned by the simulation:

```julia
flow = FlowSimulation(mesh, physics; diagnostics=(KineticEnergy(),))
register_diagnostic!(registry, flow, :kinetic_energy)
```

This keeps the history available in headless runs and prevents the viewer from
becoming the source of truth.
