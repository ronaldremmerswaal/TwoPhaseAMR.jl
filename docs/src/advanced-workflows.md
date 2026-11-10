# Advanced workflows

`FlowSimulation` is an owner, not a second solver implementation. Its fields
are the same low-level objects exposed by the package, so standard and advanced
workflows can be mixed without data conversion.

## Explicit object construction

The high-level constructor retains the typed physics and numerical options on
its equation system. The equivalent explicit construction is:

```julia
velocity = FaceVelocity(mesh, initial_velocity)
pressure = CellField(mesh)
system = NavierStokesSystem(
    mesh, physics; options=numerics, adaptivity, boundary, geometry)
workspace = NavierStokesWorkspace(system; dt=fixed_dt)

result = simulate!(
    velocity, pressure, system, final_time, workspace; run_options...)
```

For two-phase flow, construct the interface and its physical configuration
before constructing the system:

```julia
interface = InterfaceField(mesh, levelset)
physics = TwoPhase(
    interface;
    density=(rho_liquid, rho_gas),
    dynamic_viscosity=(mu_liquid, mu_gas),
    surface_tension=SurfaceTension(sigma; method=CGFM()),
    gravity=(0.0, -gravity_acceleration),
)
system = NavierStokesSystem(mesh, physics; options=numerics, geometry=interface.geometry)
workspace = NavierStokesWorkspace(system; dt=fixed_dt)
```

Use this path to manage several states with one discretization, integrate a
custom outer algorithm, or call `step!` yourself. The fully assembled objects
can also be wrapped without copying:

```julia
flow = FlowSimulation(
    velocity, pressure, system; workspace, time=0, dt=fixed_dt)
```

## Solver composition

Pressure, single-phase staggered diffusion, and coupled two-phase stress
diffusion have symmetric positive (semi-)definite integrated operators and
default to preconditioned Krylov.jl CG. All use rediscretized tree-geometric
preconditioners by default.
Single-phase diffusion has independent x/y tree hierarchies, while two-phase
diffusion keeps its coupled stress system.
Solver and preconditioner choices are independent. The hierarchy prunes the
tree and then agglomerates 2×2 base roots where possible; the last available
sparse operator is factorized as the bottom solve:

```julia
import Krylov

numerics = NavierStokesOptions(
    pressure=LinearSolverOptions(
        krylov=Krylov.GmresWorkspace,
        krylov_options=(; restart=true, memory=30),
        preconditioner=:tree_geometric,
    ),
    diffusion=LinearSolverOptions(preconditioner=:tree_geometric),
)
flow = FlowSimulation(
    mesh, SinglePhase(density=1.0, dynamic_viscosity=1e-3);
    numerics,
)
```

Optional `:amg`, `:ilu`, and `:ichol` preconditioners require importing their
weak dependency first. Pre-built Krylov workspaces, preconditioner factories,
and objects implementing three-argument `ldiv!` remain accepted by the
low-level constructors.

## Adaptive runs

Indicators and additional transferred fields are solver configuration, while
adaptation cadence and hysteresis belong to the mesh:

```julia
mesh = BlockTreeMesh(
    block_size=(16, 16), max_level=4,
    adaptation=MeshAdaptationOptions(
        update_interval=5, coarsen_delay=20, minimum_level=1))
marker = CellField(mesh, initial_marker)
criterion = InterpolationErrorIndicator(
    marker; E_refine=0.08, E_coarsen=0.025)

flow = FlowSimulation(
    mesh, SinglePhase(density=1.0, dynamic_viscosity=1e-3);
    adaptivity=Adaptivity(
        criteria=(criterion,), fields=(temperature,)))
```

Velocity, pressure, the two-phase interface and phase scalars,
refinement-criterion fields, and registered extra fields are updated in place.
Topology-dependent operators are rebuilt only after a real mesh change.

## Allocation contract

Mesh, operator, and workspace construction may allocate. After construction,
the single-phase core `advect!`, `convection!`, `divergence!`, `gradient!`, and
halo kernels are allocation-free; fixed-topology `step!` reuses its main
arrays and solver objects. Two-phase reconstruction and sparse coefficient
refreshes still allocate. The current benchmark suite separates flow setup,
adaptivity, transport, and complete stepping so construction and regridding
are not accidentally included in steady-step measurements. Reproduction
commands live in the [benchmark guide](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/main/benchmark/README.md).

`InterfaceField` records the concrete embedded-geometry type (or `Nothing`)
in its type, while existing `InterfaceField{T,M}` dispatch continues to match
either case. Mesh-only phase schedules and velocity-boundary extrapolation
schedules are reused until the mesh topology generation changes. Phase graphs
are cached for the mesh precision and for `Float32`/`Float64`; other foreign
precisions are constructed without caching. Staggered convection takes the
current boundary and time for each call rather than retaining them in its
workspace.

When a fixed step is known, pass it to `FlowSimulation(...; dt)` or
`NavierStokesWorkspace(system; dt)` so setup constructs the shifted diffusion
hierarchy at the coefficient used by the first step.
