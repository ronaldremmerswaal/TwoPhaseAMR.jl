# TwoPhaseAMR.jl

`TwoPhaseAMR.jl` is a tree-native, block-structured finite-volume solver for
single- and two-phase incompressible flow in Julia. Both models use the same
high-level API and staged Navier--Stokes step on uniform and mixed-level 2-D
quadtree meshes.

Cell kernels work on dense, haloed block arrays; owned face kernels work on
compact contiguous arrays and immutable dual-face schedules. Tree traversal
happens only while constructing or regridding a mesh. Coarse--fine centred
faces have one owner, so the flux applied to the coarse cell is exactly the sum
of the equal-and-opposite fluxes applied to adjacent fine cells.

## Implemented

- one-block, uniform multi-block, and balanced two-level meshes;
- explicit `Cell` and `Face{Axis}` placement types;
- automatic full 2:1 balancing around `RegionTrees.jl` leaves;
- same-level, periodic, coarse--fine, and physical halo filling;
- conservative average restriction and limited linear prolongation;
- prescribed face-normal velocities and first-order upwind scalar advection;
- owned directional `FaceField{Axis}` degrees of freedom;
- algebraically interpolated dual-control-volume fluxes for upwind, centred,
  Lax--Wendroff, Fromm, minmod, MC, van Leer, and superbee staggered advection
  on uniform and mixed-level meshes;
- a scheme-aware AMR-interface default: Upwind for global Upwind, MC for
  centred/LW transport, and the selected scheme for other limited fluxes;
- simultaneous conservative transport of both staggered velocity components;
- exact recovery of the interpolated primary divergence from constant-state
  convection and twice the all-centred convection-matrix diagonal;
- a centred convection operator that is volume-weighted skew-symmetric for
  discretely divergence-free velocity on periodic mixed-level trees;
- a divergence-derived pressure gradient whose sparse SPD vertex-patch Riesz
  map is affine-exact on isotropic mixed-level meshes;
- rediscretized block-tree multigrid for single- and two-phase pressure and
  diffusion, with physical coarse operators and no intermediate `P'AP`
  products;
- reusable shifted elliptic operators and pressure projection with variable
  face coefficients;
- Krylov.jl CG for pressure and BiCGSTAB for staggered diffusion, with solver
  workspaces cleanly separated from their preconditioners;
- optional AlgebraicMultigrid.jl AMG, IncompleteLU.jl ILU, and
  LimitedLDLFactorizations.jl incomplete-Cholesky preconditioners;
- automatic no-permeation on every non-periodic boundary and the corresponding
  homogeneous-Neumann pressure-correction closure;
- no-slip (default), moving-wall, and free-slip staggered diffusion closures;
- widened, convergent AMR-interface diffusion stencils and component-wise
  staggered geometric multigrid, with LU used only on the bottom grids;
- an optionally adaptive incompressible Navier--Stokes solver with one LW convection
  update, one implicit-midpoint diffusion solve, and one projection per step;
- geometric VOF with level-set initialization, LHF/GHF interface geometry,
  and directionally split Weymouth--Yue transport;
- compatible intrinsic centred-scalar and staggered momentum transport using
  the same liquid and gas volume fluxes as the interface;
- the coupled variable-viscosity stress operator and variable-density pressure
  projection used by the one-velocity two-phase model;
- balanced CSF/GFM surface tension and gravity, plus interface-, curvature-,
  and vorticity-based refinement indicators;
- centred block indicators, 2:1-graded in-place regridding, coarsening
  hysteresis, and conservative cell/face field transfer;
- divergence-preserving staggered-velocity prolongation and automatic rebuild
  of topology-dependent solver workspaces;
- Taylor--Green-vortex and lid-driven-cavity validation cases;
- allocation-free single-phase time-step, halo-fill, and CFL kernels after
  setup, with two-phase kernel allocations tracked by dedicated benchmarks; and
- correctness, conservation, reference, allocation, and benchmark coverage.

General pressure Dirichlet data, 3-D meshes, time subcycling, and distributed
execution are outside the current scope. The two-phase construction and its
discrete invariants are recorded in the
[implementation record](docs/src/two-phase-development.md).

## Quick start

```julia
using TwoPhaseAMR

mesh = BlockTreeMesh(block_size=(16, 16), extent=(2pi, 2pi))
viscosity = 0.02
velocity = (x, y, time, axis) -> exp(-2viscosity * time) *
    (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))

flow = FlowSimulation(
    mesh; viscosity, velocity, boundary=FreeSlip(), dt=0.001)
result = simulate!(flow, 0.02)
@assert result.pressure.converged
```

The same constructor enables two-phase flow when an interface is supplied.
Initialize physical test geometry from a level set so cut-cell fractions are
integrated rather than sampled at cell centres:

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
levelset = (x, y) -> hypot(x - 0.5, y - 0.6) - 0.15
interface = InterfaceField(mesh, levelset)

flow = FlowSimulation(
    mesh;
    interface,
    density=(1000.0, 1.0),             # (liquid, gas)
    viscosity=(1e-3, 1.8e-5),          # dynamic viscosities
    surface_tension=0.072,
    gravity=(0.0, -9.81),
    surface_tension_formulation=GFM(),
    dt=1e-4,
)
result = simulate!(flow, 1e-3)
```

`flow.interface` stores geometric liquid volume fractions, and
`flow.diffusion_operator` exposes the active constant- or variable-coefficient
stress operator. Density, phase scalars, and velocity remain intrinsic fields;
phase-weighted quantities are formed only temporarily inside conservative
updates.

The [Stokes-3 wave example](examples/stokes3_wave.jl) uses the thesis
parameters and combines interface-, curvature-, and vorticity-based AMR. It
advances to `t=2` by default, keeps the interface band at level 1, and opens a
live volume-fraction viewer. Run it with multiple threads so the viewer stays
responsive:

```sh
julia --threads=auto --project=. examples/stokes3_wave.jl
```

Three validation cases also have live examples. Each displays the geometric
volume fraction together with velocity, pressure where applicable, and the
current AMR blocks:

```sh
julia --threads=auto --project=. examples/vortex_reverse.jl
julia --threads=auto --project=. examples/capillary_wave.jl
julia --threads=auto --project=. examples/sessile_droplet.jl
```

The vortex example uses 8-by-8 blocks, up to five refinement levels, an
eight-unit reversal period, and the curvature criterion with a protected
level-one interface band and records its live block count. The wave viewer
compares the measured amplitude magnitude with Prosperetti's finite-viscosity
linear solution. The sessile drop starts as an equal-volume circular cap and
visibly relaxes toward its gravity-deformed Young--Laplace equilibrium while
recording its kinetic energy.

Fixed arbitrary solids use the same level-set-to-fraction and PLIC data model
as a phase interface. The solver derives open cell volumes, open face
apertures, and staggered cut control volumes from that reconstruction:

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
solid_levelset = (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2
geometry = GeometryField(mesh, solid_levelset; lipschitz=1)
flow = FlowSimulation(mesh; geometry, viscosity=0.01)
```

The pressure gradient is the negative adjoint of the aperture-weighted
divergence, centred convection is skew symmetric, and cut-cell diffusion is
symmetric positive definite. The same `geometry` keyword works with an
`InterfaceField`. Its stored liquid fraction is measured against the complete
Cartesian cell and is therefore bounded by the geometry's open fraction. PLIC
reconstruction and material laws derive the conditional liquid fraction within
that open volume. Triple-point curvature uses the thesis GHF contact-angle equation.
Both overlays are geometric: phase interfaces are drawn as PLIC segments and
solids as filled PLIC polygons.

Two examples exercise the coupled functionality:

```sh
julia --threads=auto --project=. examples/cylinder_flow.jl
julia --threads=auto --project=. examples/sloped_sessile_droplet.jl
```

The cylinder example implements the Reynolds-100 domain and boundary
conditions of Verstappen and Dröge (2005), including inflow, zero-gradient
lateral boundaries, and a flux-balanced outflow. The sloped sessile drop uses
interface-, curvature-, and solid-boundary AMR and records kinetic energy as it
relaxes from a circular cap. Its embedded wall is impermeable and free-slip,
so the contact line can move without grid-dependent no-slip regularization.
The long cylinder refinement validation is
`test/integration/single_phase_cylinder.jl`; its default run checks the
reported second-order pressure-drag convergence and the paper's bulk values.

`FlowSimulation` owns the velocity, pressure, solver, reusable workspace,
current time, default time step, and cumulative counters. Supplying `dt` at
construction also avoids building an intermediate zero-shift diffusion
hierarchy. Calling `simulate!` again continues the same calculation. Every
lower-level object remains available through
`flow.velocity`, `flow.pressure`, `flow.solver`, and `flow.workspace`.

For operator development, the lower-level API can be used directly. This
example checks conservative cell-centred advection on a static mixed mesh:

```julia
mesh = BlockTreeMesh(
    block_size=(16, 16), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0))
q = CellField(mesh, (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y))

u = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
workspace = AdvectionWorkspace(q)
dt = stable_timestep(u, workspace; cfl=0.5)

mass_before = integrate(q)
advect!(q, u, dt, workspace)
@assert integrate(q) ≈ mass_before
```

Directional staggered transport uses the same owned face-normal velocity:

```julia
staggered_mesh = BlockTreeMesh(block_size=(16, 16))
staggered_velocity = FaceVelocity(
    staggered_mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
momentum_x = FaceField{1}(
    staggered_mesh, (x, y) -> sinpi(2x) * cospi(2y))
staggered_workspace = StaggeredAdvectionWorkspace(momentum_x)
dt = stable_timestep(staggered_velocity, staggered_workspace; cfl=0.5)

advect!(momentum_x, staggered_velocity, dt, staggered_workspace)
```

`Upwind()`, `Centred()` (also `Centered()`), `LaxWendroff()` (also `LW()`),
`Fromm()`, `Minmod()`, `MonotonizedCentral()` (also `MC()`), `VanLeer()`, and
`Superbee()` are supported for both `scheme` and `amr_interface_scheme`.
The bulk scheme defaults to `LW()`. Its AMR-interface default is selected by
`default_amr_interface_scheme`: Upwind maps to Upwind, Centred and LW map to
MC, and the other three-point schemes map to themselves. Interface-only
Upwind is rejected because its unmatched first-order closure has poor global
accuracy; choose global `scheme=Upwind()` when upwinding is required. LW and
three-point schemes are directionally
split: x- and y-oriented graph fluxes are never combined in one update. Each
time step has two full sweeps, publishes the intermediate field, and alternates
x/y with y/x ordering on successive steps. The LW predictor also includes the
directional divergence required for a variable transporting velocity. The
limited schemes suppress the isolated interface-extremum defect while
retaining better than first-order convergence in the mixed-level manufactured
test.
Staggered advection accepts uniform and 2:1-balanced mixed-level meshes.
Non-periodic sides have zero normal advective flux. Cell-centred scalar
advection remains the separate first-order upwind method described above.

The equivalent fully explicit single-phase solver setup is:

```julia
mesh = BlockTreeMesh(block_size=(32, 32), periodic=(false, false))
lid = NoSlip((x, y, time, axis) -> axis == 1 && y > 0.5 ? 1.0 : 0.0)
solver = NavierStokesSolver(mesh; viscosity=0.01, boundary=lid)
dt = 0.002
workspace = NavierStokesWorkspace(solver; dt)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)

simulate!(velocity, pressure, solver, 1.0, workspace; dt)
```

Convection defaults to bulk `LW()` with `MC()` on non-regular AMR connections.
Diffusion is Crank--Nicolson/implicit
midpoint and is solved by standalone or BiCGStab-accelerated staggered
geometric multigrid according to the measured stiffness crossover. Each
`step!` then performs exactly one compatible pressure projection.

Dynamic refinement is opt-in. Mesh options control its cadence and cooldown;
the solver receives indicators and any additional fields that must follow the
mesh. Velocity, pressure, and indicator fields are registered automatically:

```julia
mesh = BlockTreeMesh(
    block_size=(16, 16), max_level=4,
    adaptation=MeshAdaptationOptions(
        update_interval=5, coarsen_delay=20))
marker = CellField(mesh)
temperature = CellField(mesh)
velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
pressure = CellField(mesh)
criterion = InterpolationErrorIndicator(
    marker; E_refine=0.08, E_coarsen=0.025)
solver = NavierStokesSolver(
    mesh; viscosity=1e-3, refinement_indicators=criterion,
    adaptive_fields=(temperature,))
workspace = NavierStokesWorkspace(solver)
result = simulate!(velocity, pressure, solver, 1.0, workspace)
```

See the [adaptive-refinement design and literature review](docs/src/adaptive-refinement.md)
for label precedence, conservative transfer, divergence preservation, memory
reuse, and current limitations.

The [getting-started guide](docs/src/getting-started.md),
[single-phase tutorial](docs/src/single-phase-tutorial.md), and
[advanced workflows](docs/src/advanced-workflows.md) separate the standard and
low-level APIs.

Run the serial tests with `julia --project -e 'using Pkg; Pkg.test()'`, or the
parallel suite with
`julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'`. Full two-phase
refinement studies live under `test/integration`; CI runs each study with four
Julia threads. Build the documentation with
`julia --project=docs docs/make.jl`. Benchmark environment
instructions are in [benchmark/README.md](benchmark/README.md). The detailed
[Stokes-3 performance analysis](benchmark/stokes3-performance-results.md)
includes a per-step timing and allocation profiler for the adaptive two-phase
example. The standalone/PCG comparison is recorded in
[benchmark/pcg-multigrid-results.md](benchmark/pcg-multigrid-results.md).
The staggered-diffusion BiCGStab comparison is in
[benchmark/bicgstab-multigrid-results.md](benchmark/bicgstab-multigrid-results.md).
The [block-locality experiment](benchmark/data-locality-results.md) compares
global and block-first traversal before further parallel scheduling changes.

## Interactive visualization

Interactive visualization is optional and does not add Makie to the solver's
mandatory dependencies. Install the latest registered desktop or browser
backend in the environment where it will be used:

```julia
using Pkg
Pkg.add("GLMakie")       # native OpenGL window
# or: Pkg.add("WGLMakie") # WebGL in a browser
```

After loading either backend, register cell fields, directional face fields,
and derived velocity scalars on a mixed-level mesh. The menu switches between
them without changing solver state:

```julia
using TwoPhaseAMR, GLMakie # replace GLMakie with WGLMakie for a browser

mesh = BlockTreeMesh(
    block_size=(12, 12), periodic=(false, false), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)
pressure = CellField(mesh)
initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
velocity = FaceVelocity(
    mesh, (x, y, time, axis) -> axis == 1 ?
        sinpi(x) * cospi(y) : -cospi(x) * sinpi(y))

fields = VisualizationFieldRegistry()
register_default_fields!(fields, :pressure, pressure, :velocity, velocity)
register_diagnostic!(
    fields, :maximum_velocity,
    (velocity, pressure, time, step, solver) ->
        maximum(abs, velocity.interior);
    label="Maximum velocity")
register_diagnostic!(
    fields, :total_pressure,
    (velocity, pressure, time, step, solver) -> integrate(pressure);
    label="Total pressure")
viewer = visualize(fields; field=:pressure, show_blocks=true)

initialize!(pressure, (x, y) -> cospi(2x) * sinpi(2y))
update_visualization!(viewer) # refresh the field currently shown
select_visualization_field!(viewer, :velocity_vorticity)
```

The default registration expands a two-component face field into its x and y
components, centred magnitude, and vorticity. Use `register_field!` directly
when custom names, units, or plotting styles are required.

The viewer supports piecewise-constant cell colors, gap-free filled dual cells
for directional face fields, velocity magnitude and vorticity, pan/zoom, block
outlines, optional cell outlines, and click selection of cells or faces. The
original
`visualize(field)` and `update_visualization!(viewer, field)` API remains
available for a single cell field. Live solver observation, checkpoints,
programmatic historical/live switching, asynchronous publication, GUI
checkpoint navigation, pause/resume, and independently sampled scalar
diagnostics are implemented.

For a solver run, wrap the registry and viewer in a session and pass it to
`simulate!`. Checkpoints can be requested in three ways:

```julia
session = VisualizationSession(
    fields; viewer, checkpoints=10, live_every=5,
    diagnostic_every=1, asynchronous=true)
# or: VisualizationSession(fields; viewer, checkpoint_dt=0.1)
# or: VisualizationSession(fields; viewer, checkpoint_times=[0.0, 0.2, 0.7, 1.0])

simulation = Threads.@spawn simulate!(
    velocity, pressure, solver, final_time, workspace;
    visualization=session)

request_pause!(session)
wait_until_paused(session) # the latest completed step is now stationary
resume_simulation!(session)

result = fetch(simulation)
wait_visualization!(session)
```

The nearest completed solution is retained by default. Pass
`exact_checkpoints=true` to take an additional side step on copied state for
each off-grid requested time. The copied result is retained without changing
the actual simulation trajectory. The session requests these side steps
through an independent procedure supplied at initialization; `simulate!`
itself remains unaware of checkpoint times. Use
`visualization_checkpoints(session)`, `select_checkpoint!(session, i)`, and
`live_mode!(session)` to inspect retained states and return to the latest live
frame while a run continues. `live_every=n` or `live_dt=interval` controls
transient field sampling independently of checkpoint storage. Asynchronous
delivery retains only the newest waiting live frame; use
`dropped_visualization_frames(session)` to inspect replacements and
`wait_visualization!(session)` after completion when the final rendered frame
must be synchronized. The Makie viewer's checkpoint menu selects retained
frames, and its **Live** button returns immediately to the newest sampled
state without changing the running simulation. Its **Pause** button requests a
pause at the next completed-step boundary, displays **Resume** while the
request is pending or reached, and resumes or cancels the request when clicked
again. `simulation_state(session)` distinguishes `:pause_requested` from
`:paused`; pause latency is at most one numerical step. Control is implemented
inside the visualization update, so `simulate!` contains no
visualization-specific control logic. Run the GUI and solver on separate tasks
in a multi-threaded Julia process so the interface remains responsive.
Registered diagnostics are retained losslessly and displayed together in a
time-series panel. They use `diagnostic_every` or `diagnostic_dt`, independently
of the mesh-frame sampling policy. Checkpoints appear as vertical lines and
clickable markers; clicking one selects the same historical frame as the
checkpoint menu.

Nearest-solution checkpoints retain immutable topology snapshots and therefore
remain valid through dynamic regrids. Exact off-grid checkpoints currently
require a fixed topology; `simulate!` rejects that combination when refinement
indicators are enabled.

An end-to-end GLMakie demonstration initializes a projected, band-limited
random velocity and evolves freely decaying two-dimensional isotropic
turbulence with dynamic refinement:

```sh
julia --project=. --threads=auto examples/visualize_decaying_turbulence.jl
```

The mesh starts uniformly at the finest 128-by-128 equivalent resolution.
Interpolation error in the live pressure field then coarsens smooth regions,
while a protected 32-by-32 base level, update cadence, and coarsening cooldown
prevent over-coarsening and chatter. Block outlines change with the topology,
and an active-cell fraction starts at 100% and falls as resolution is removed.
It is plotted alongside normalized energy and enstrophy, peak speed, and
divergence. The viewer also exercises field switching, block/face selection,
live updates, pause/resume, checkpoint history, and linked checkpoint markers.
This is a 2-D decaying-turbulence showcase, not a model of the three-dimensional
HIT energy cascade. Increase `max_level` to 5 explicitly for a substantially
larger run.

## Development status

| Milestone | Status |
|---|---|
| 1. Centred single-phase foundation | Implemented for first-order upwind transport |
| 2. Staggered fields and compatible operators | Implemented for static, globally stepped uniform and mixed-level meshes |
| 3. Pressure projection | Implemented for periodic and impermeable-wall uniform/isotropic mixed-level meshes |
| 4. Wall diffusion and single-phase Navier--Stokes | Implemented on static or dynamically adapted uniform/mixed-level trees |
| 5. Geometric VOF | Implemented with level-set initialization, LHF/GHF/WY reconstruction, compatible phase transport, and geometric AMR transfer |
| 6. Two-phase Navier--Stokes | Implemented with coupled variable stress diffusion, variable-density projection, balanced surface tension/gravity, and interface-aware AMR |
| 7. Interactive visualization | Dynamic-topology field rendering, live/nearest-checkpoint navigation, pause/resume, and scalar diagnostics implemented |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the complete roadmap.
