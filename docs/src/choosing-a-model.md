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
of both phases (with the Gram-moment correction in cut cells at a wall), and the viscous step is an implicit midpoint in the same
metric, stable for every time step (the common increment of both phases away
from the interface). It
takes no options and works with either surface-tension formulation; it
builds its own pressure
solver, so non-default `pressure_*` solver settings of the system throw
`ArgumentError`. It runs on uniform and adaptive (2:1 graded) meshes of
square cells, cut cells at the domain walls (contact lines included) and at
embedded solids among them, and throws for non-square cells. A cut cell whose Gram tensor admits no consistent block gets
a least-squares fallback and is counted in
`MimeticProjectionStats.defective_cells`; the run summary prints the totals of
a `simulate!` call. Each projection solves a conjugate-gradient system with a
sparse factorization of the cut-cell block, so it is more expensive than the
CMDGFM.
See the technical reference, section
[Mimetic cut-cell projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:projection:mimetic).

The viscous step diffuses the phase-mass weighted mixture velocity and gives
both phases the same increment, which leaves their difference untouched.
`TwoVelocityModel(interphase_drag=InterphaseDrag())`, the default for both
projections, adds a Navier-slip shear between the phases across the
interface: on every face whose staggered volume holds both phases the jump
`j` decays as `dj/dt = -lambda j / (1 + Re_j)`, solved exactly at fixed
mixture momentum, with `lambda = W (1/m_liquid + 1/m_gas)`, the no-slip
conductance `W` the interface viscosity times the PLIC interface length in
the face's control volume over the distance of the two phase centroids, and
`Re_j = |j| h / nu` the cell Reynolds number of the jump. A slowly slipping
layer (`Re_j << 1`, a resolved shear layer) is removed, while a vortex sheet
with `Re_j >> 1`, by which the two-velocity model represents an unresolved
shear layer, is kept. Without viscosity, at rest or for equal phase velocities
the drag does nothing. By default only the share of each face component
tangential to the interface is coupled; `InterphaseDrag(component=:full)`
couples the whole component, and `interphase_drag=nothing` keeps free slip
between the phases.
See [Interphase drag](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=eq:viscosity:drag).

### Which projection?

| Flow | Recommended | Why |
| --- | --- | --- |
| capillary-dominated and moderate flows: drops at rest, sessile drops, capillary waves, interfacial instabilities at moderate speeds | `MimeticProjection()` | energy-neutral in its metric; the capillary force pays the surface energy and the contact line along domain walls the wall energy |
| violent flows: impacts, splashes, thin films | `CMDGFMProjection()` (default) | bounds the light-phase values next to the interface independently of the density ratio |

The mimetic projection's limitation is measured on a drop impact at density
ratio 1000 (a drop of radius 0.08 at speed 2 onto a pool, `n = 32` and 64):
the median largest velocity is 50 and 111, against 3.5 and 5.0 with the
CMDGFM, and grows under refinement. It is structural: in an energy-neutral
projection with homogeneous constraints, controlling the near-empty
light-phase values and keeping the tangential slip between the phases
conflict. The CMDGFM obtains both only through parts that are not
energy-neutral: its unresolved mixed faces keep the predictor's velocity jump,
and its jump rows infer the jump from neighbouring faces with an update that
is not their adjoint, so it conserves momentum but not kinetic energy. See
the technical reference, section
[Scope and limitations](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:projection:mimetic-scope).

## Option matrix

| Decision | Choices | Default | Applies to |
| --- | --- | --- | --- |
| velocity formulation | `OneVelocityModel()`, `TwoVelocityModel(...)` | one velocity | two-phase flow |
| surface-tension force | `CGFM()`, `CSF()` | CGFM | either two-phase velocity model |
| pressure projection | `CMDGFMProjection()`, `MimeticProjection()` | CMDGFM | two velocity only |
| interphase drag | `InterphaseDrag()`, `InterphaseDrag(component=:full)`, `nothing` | `InterphaseDrag()` | two velocity only |

Surface tension is configured by [`SurfaceTension`](@ref), while the
projection is a field of [`TwoVelocityModel`](@ref). This ownership makes
invalid combinations visible at construction time. See
[Two-phase flow](two-phase-flow.md) and
[Pressure gradient](pressure-gradient.md) for the numerical definitions.
