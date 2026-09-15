# Advanced workflows

`FlowSimulation` is an owner, not a second solver implementation. Its fields
are the same low-level objects exposed by the package, so standard and advanced
workflows can be mixed without data conversion.

## Explicit object construction

The high-level constructor is equivalent to:

```julia
velocity = FaceVelocity(mesh, initial_velocity)
pressure = CellField(mesh)
solver = NavierStokesSolver(mesh; viscosity, solver_options...)
workspace = NavierStokesWorkspace(solver; dt=fixed_dt)

result = simulate!(
    velocity, pressure, solver, final_time, workspace; run_options...)
```

Use this path to manage several states with one discretization, integrate a
custom outer algorithm, or call `step!` yourself. The fully assembled objects
can also be wrapped without copying:

```julia
flow = FlowSimulation(
    velocity, pressure, solver; workspace, time=0, dt=fixed_dt)
```

## Solver composition

Pressure defaults to Krylov.jl CG and staggered diffusion to Krylov.jl
BiCGSTAB. Both use geometric multigrid by default. Solver and preconditioner
choices are independent:

```julia
import Krylov

flow = FlowSimulation(
    mesh; viscosity=1e-3,
    pressure_krylov=Krylov.GmresWorkspace,
    pressure_krylov_options=(; restart=true, memory=30),
    pressure_preconditioner=:geometric,
    diffusion_preconditioner=:geometric,
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
    mesh; viscosity=1e-3,
    refinement_indicators=criterion,
    adaptive_fields=(temperature,))
```

Velocity, pressure, indicator fields, and registered extra fields are updated
in place. Topology-dependent operators are rebuilt only after a real mesh
change.

## Allocation contract

Mesh, operator, and workspace construction may allocate. After construction,
the core `advect!`, `convection!`, `divergence!`, `gradient!`, and halo kernels
are allocation-free; fixed-topology `step!` reuses its main arrays and solver
objects. Benchmark setup/regridding separately from stepping. Reproduction
commands and machine-specific reference results live in `benchmark/README.md`.
When a fixed step is known, pass it to `FlowSimulation(...; dt)` or
`NavierStokesWorkspace(solver; dt)` so setup constructs the shifted diffusion
hierarchy at the coefficient used by the first step.
