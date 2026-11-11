# Two-phase flow

Two-phase simulations use the same [`FlowSimulation`](@ref) entry point as
single-phase flow. A [`TwoPhase`](@ref) object contains the interface and
material properties; the velocity model independently selects one shared
velocity or separate intrinsic liquid and gas velocities.

```julia
mesh = BlockTreeMesh(block_size=(16, 16), max_level=2)
interface = InterfaceField(
    mesh, (x, y) -> hypot(x - 0.5, y - 0.6) - 0.15;
    lipschitz=1,
)
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1.0e-3, 1.8e-5),
    surface_tension=0.072,
    gravity=(0.0, -9.81),
)
flow = FlowSimulation(
    mesh, physics;
    velocity_model=OneVelocityModel(),
    boundary=FreeSlip(),
)
```

A numeric `surface_tension` uses [`CGFM`](@ref), the default conservative
ghost-fluid formulation. Use a [`SurfaceTension`](@ref) object to select
another method:

```julia
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1.0e-3, 1.8e-5),
    surface_tension=SurfaceTension(0.072; method=CSF()),
)
```

## Choosing the velocity model

[`OneVelocityModel`](@ref) stores one face velocity and is the default. It is
appropriate when a shared interfacial velocity is the intended closure.

[`TwoVelocityModel`](@ref) stores separate intrinsic phase velocities. Its two
independent numerical choices are explicit:

```julia
velocity_model = TwoVelocityModel(
    interface_transport=FrozenInterfaceFluxes(),
    pressure_projection=CMDGFMProjection(),
)
flow = FlowSimulation(mesh, physics; velocity_model)
```

The alternative interface carrier is [`ProjectedLiquidVelocity`](@ref). It is
the only choice besides the default carrier; see
[Two-velocity advection](two-velocity-advection.md) for its discrete
definition. A shared velocity initializer works with either model. To start
the phases differently, pass a liquid/gas pair of fields or callables together
with `TwoVelocityModel()`.

## Interface initialization and reconstruction

[`InterfaceField`](@ref) stores the physical liquid volume fraction. The
level-set constructor treats `phi(x,y) <= 0` as liquid and integrates it over
each active cell. The function need not be a signed distance. When it is, pass
`lipschitz=1` to make pure-cell exclusion more reliable.

```julia
interface = InterfaceField(
    mesh, (x, y) -> ((x - 0.5) / 0.2)^2 +
        ((y - 0.5) / 0.1)^2 - 1;
    atol=1.0e-10,
)
```

For a piecewise-linear boundary, construct a [`PolygonIndicator`](@ref)
instead of approximating the polygon by a nonsmooth level-set function. Cell
fractions then come from exact polygon clipping and do not depend on a
quadrature tolerance:

```julia
polygon = PolygonIndicator([
    (0.2, 0.1), (0.8, 0.1), (0.7, 0.6), (0.3, 0.7),
])
interface = InterfaceField(mesh, polygon)
```

[`reconstruct!`](@ref) builds local PLIC geometry and curvature. For ordinary
mixed cells, the default `reconstruction_method=:lhf` uses local
height-function normals when available and Weymouth--Yue (WY) normals
otherwise. `:lhf_ghf` also uses a generalized height-function (GHF) normal
when the local height function is unavailable; `:wy` uses WY normals.
Curvature uses height functions and GHF fits independently of the selected
normal method and is `NaN` when neither is available.

In mixed cells cut by an embedded solid, every method instead uses an
unconstrained solid-aware GHF fit for the PLIC normal, provided it does not
oppose a WY estimate that excludes fully covered neighbours. Otherwise the
nearest valid LHF normal from a fully open mixed cell in the 5×5 neighbourhood
is used if it lies within 45° of the WY estimate; if none qualifies, the WY
estimate is used. A separate fit applies the embedded contact-angle constraint
to curvature when available; if it fails, the unconstrained curvature is used.
Both embedded and domain-wall contact angles constrain curvature only, not
the advected PLIC plane.
The reconstruction, force, contact-angle, and refinement choices are discussed in
[Remmerswaal (2023)](https://doi.org/10.33612/diss.562133319).

For a single-valued interface, [`interface_height`](@ref) returns its physical
coordinate along an adaptive-mesh column. This is useful for wave diagnostics:

```julia
amplitude = Diagnostic(
    :amplitude,
    state -> interface_height(state.interface; direction=:y, position=0) -
        mean_height,
)
```

## Transport and material properties

The interface uses conservative directional Weymouth--Yue transport. Liquid
and gas face volumes add to the total open-face volume, and
[`stable_timestep`](@ref) includes the interface-volume CFL restriction and,
for cut donors, bounds each outgoing segment by its open strip and separates
opposing swept open regions. This includes hanging faces on adaptive meshes.
[`PhaseScalar`](@ref) and phase momentum reuse the same phase fluxes rather
than reconstructing a separate carrier.

The sweep checks liquid volume against the open cell volume and gas volume
against zero before clamping and snapping. Its round-off allowance is scaled
by the full cell volume, so division by a small open fraction does not turn a
round-off-sized volume error into a false bound violation. Volume violations
larger than the allowance still raise an error; the auxiliary gas carrier has
no upper bound.

For one-velocity momentum on embedded AMR meshes, the primal-to-dual phase
fluxes use the open strip volume behind each face segment. This keeps the
staggered density update compatible with the VOF update even where a cut
coarse cell meets two fine face segments. As elsewhere, momentum conservation
is subject to fraction clamping and snapping, boundary fluxes, and the
discrete divergence of the transporting velocity.

Density is linearly volume averaged at the locations needed by projection.
Dynamic viscosity is geometrically averaged at normal- and shear-stress
locations. Set both phase viscosities to zero for inviscid flow; otherwise
both must be positive.

Surface tension choices are:

| Method | Use |
| --- | --- |
| [`CGFM`](@ref) | default conservative ghost-fluid force |
| [`CSF`](@ref) | continuous-surface-force formulation |

When no curvature estimate is available on a face with a nonzero surface-tension
jump, the force uses zero curvature there. Consequently under-resolved films,
ligaments or satellites experience locally no capillary force on those faces;
mesh refinement is needed to resolve them. Reconstruction itself retains
unavailable curvature as `NaN`. Monitor the latest step with
`flow.workspace.unresolved_curvature_faces[]`, or record it in a `Diagnostic`
using `state.workspace.unresolved_curvature_faces[]`. The `simulate!` result
reports the sum over the call as `unresolved_curvature_faces` and, for a
`FlowSimulation`, the cumulative sum as `total_unresolved_curvature_faces`.
The simulation summary displays nonzero counts.

Gravity and surface tension use the same compatible pressure-gradient
locations as the projection. Nonzero gravity is not supported along a
periodic direction.

## Embedded solids and contact angles

[`GeometryField`](@ref) represents a fixed solid with `phi(x,y) <= 0`. Pass it
to both the interface (when the phases touch the solid) and the simulation:

```julia
geometry = GeometryField(mesh, wall_levelset)
interface = InterfaceField(
    mesh, phase_levelset;
    geometry,
    contact_angle=(nothing, nothing, pi / 2, nothing),
)
physics = TwoPhase(
    interface;
    density=(1000.0, 1.0),
    dynamic_viscosity=(1.0e-3, 1.8e-5),
    surface_tension=0.072,
)
flow = FlowSimulation(
    mesh, physics;
    geometry,
    boundary=FreeSlip(),
)
```

The interface keeps its geometry association for its lifetime.
`InterfaceField{T,M}` annotations accept interfaces both with and without a
solid geometry. Copying an interface copies its fraction data but shares the
same `GeometryField`; construct another interface with a copied geometry when
independent solid state is needed. A `GeometryField`'s own solid interface has
no geometry.

The four contact-angle entries correspond to left, right, bottom, and top
domain walls. Use `embedded_contact_angle` for the solid; the default is
`pi / 2`. Passing `nothing` disables its contact-angle row, not the
solid-aware reconstruction. For compatibility, a scalar `contact_angle`
supplied with a geometry selects the embedded angle instead. Closed solid
faces carry no phase, surface-tension, or gravity flux.

## Adaptive refinement

Use [`InterfaceRefinementIndicator`](@ref) for a fixed band around the phase
interface and [`CurvatureRefinementIndicator`](@ref) where the local curvature
needs more resolution. Add them through [`Adaptivity`](@ref):

```julia
criteria = (
    InterfaceRefinementIndicator(interface; level=2),
    CurvatureRefinementIndicator(interface; cells_per_radius=8),
)
flow = FlowSimulation(
    mesh, physics;
    adaptivity=Adaptivity(criteria=criteria),
)
```

The curvature criterion considers liquid–gas mixed cells in the open fluid;
liquid-filled solid cut cells alone do not trigger it. The fixed interface
band can still include wetted solid boundaries because covered cells have
zero conditional liquid fraction. See [Dynamic adaptive refinement](adaptive-refinement.md)
for the band and coarsening rules.

Regridding conserves phase volume and rebuilds interface, transport, pressure,
and diffusion workspaces after a real topology change. The original level-set
function is used only for initialization; later refinement transfers the
reconstructed physical interface.

## Practical limits

- Interface transport currently supports impermeable or periodic boundaries.
- The solver uses one global time step; there is no level subcycling.
- Curvature and contact-angle accuracy should be checked by mesh refinement.
- Tiny phase fractions below `fraction_floor` are removed to keep intrinsic
  scalar and velocity recovery well-defined.
- The package is two-dimensional and uses shared-memory execution.

The runnable [examples](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/tree/main/examples)
include capillary waves, interfacial instability, sessile droplets, and
embedded-wall cases. Every Navier--Stokes two-phase example accepts either
velocity model; the instability example defaults to two velocity.

## References

- R. Remmerswaal, [*Numerical modelling of variability in liquid
  impacts*](https://doi.org/10.33612/diss.562133319), PhD thesis, University
  of Groningen (2023).
- G. D. Weymouth and D. K.-P. Yue, [*Conservative volume-of-fluid method for
  free-surface simulations on Cartesian-grids*](https://doi.org/10.1016/j.jcp.2009.12.018),
  Journal of Computational Physics 229 (2010), 2853--2865.
- S. Popinet, [*An accurate adaptive solver for surface-tension-driven
  interfacial flows*](https://doi.org/10.1016/j.jcp.2009.04.042), Journal of
  Computational Physics 228 (2009), 5838--5866.
- R. A. Remmerswaal and A. E. P. Veldman, [*Towards a sharp, structure
  preserving two-velocity model for two-phase flow: transport of mass and
  momentum*](https://arxiv.org/abs/2209.14934).
- R. I. Saye, [*High-order quadrature methods for implicitly defined surfaces
  and volumes in hyperrectangles*](https://doi.org/10.1137/140966290), SIAM
  Journal on Scientific Computing 37 (2015), A993--A1019.
- J. Kromer and D. Bothe, [*Third-order accurate initialization of VOF volume
  fractions on unstructured meshes with arbitrary polyhedral
  cells*](https://arxiv.org/abs/2111.01073).
- J. U. Brackbill, D. B. Kothe, and C. Zemach, [*A continuum method for
  modeling surface tension*](https://doi.org/10.1016/0021-9991(92)90240-Y),
  Journal of Computational Physics 100 (1992), 335--354.
- P. H. A. J. M. van der Plas, [*Local grid refinement for free-surface flow
  simulations*](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/),
  PhD thesis, University of Groningen (2017).
- R. W. C. P. Verstappen and M. Dröge, [*A symmetry-preserving Cartesian grid
  method for computing a viscous flow past a circular
  cylinder*](https://doi.org/10.1016/j.crme.2004.09.021),
  Comptes Rendus Mécanique 333 (2005), 51--57.
