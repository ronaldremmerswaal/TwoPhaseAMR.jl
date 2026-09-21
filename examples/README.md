# Examples

Every example is a standalone Julia file with one entry point:

```julia
include("examples/capillary_wave.jl")
run = run_example(
    max_level = 1,
    final_time = 0.01,
    velocity_model = TwoVelocityModel(),
    visualize = false,
)
```

The named physical parameters and `max_level` form the introductory API.
`visualize=false` is the reusable/headless default. Other keywords are
forwarded to `FlowSimulation`, so typed model and numerical options remain
available without adding plumbing to each script. Navier--Stokes examples let
`simulate!` select a stable time step automatically. Running a file directly
passes `visualize=true`, opens a live GLMakie view, and prints a detailed model,
mesh, solver, AMR, and diagnostics report when the run completes.

| Command | Default model, cost, and purpose |
| --- | --- |
| `julia --project=. examples/taylor_green.jl` | single phase; seconds; periodic verification on mixed levels |
| `julia --project=. examples/lid_driven_cavity.jl` | single phase; minutes; moving-wall boundary conditions |
| `julia --project=. examples/cylinder_flow.jl` | single phase; long; cut-cell geometry and pressure drag |
| `julia --project=. examples/decaying_turbulence.jl` | single phase; minutes; pressure-driven dynamic AMR |
| `julia --project=. examples/capillary_wave.jl` | one velocity; minutes; numerical/linear amplitude comparison |
| `julia --project=. examples/sessile_droplet.jl` | one velocity; long; contact angle and gravity |
| `julia --project=. examples/sloped_sessile_droplet.jl` | one velocity; long; embedded inclined wall |
| `julia --project=. examples/stokes3_wave.jl` | one velocity; long; nonlinear wave with three AMR criteria |
| `julia --project=. examples/lgpi.jl` | one velocity; very long; large gas-pocket impact |
| `julia --project=. examples/interfacial_instability.jl` | **two velocity**; long; KH/RT numerical/linear amplitude comparison |
| `julia --project=. examples/vortex_reverse.jl` | prescribed carrier; minutes; standalone VOF transport |
| `julia --project=. examples/manufactured_poisson.jl` | **two velocity**; seconds; projected pressure, phase pressure gradient, and their errors |

All Navier--Stokes two-phase examples also accept:

```julia
velocity_model = TwoVelocityModel(
    interface_transport = FrozenInterfaceFluxes(),
    pressure_projection = CMDGFMProjection(),
)
```

The instability exposes `velocity_model` directly and uses that model by
default because its physical initial state contains a tangential velocity
jump. Passing `OneVelocityModel()` selects the documented density-weighted
mean closure. The vortex-reversal example has no Navier--Stokes velocity model
because its carrier velocity is prescribed.

Use `max_level=0` and a short `final_time` for a quick first run. Full
dispersion relations, convergence studies, and literature comparisons live in
`test/integration/`, keeping the executable scripts focused on setting up and
running one case.

The manufactured Poisson example is a standalone projection rather than a
time-dependent flow. Its field menu contains the gauge-aligned pressure after
projection, both physical phase pressure-gradient components, the pressure and
phase-gradient errors, and each component of the gradient-jump error on mixed
faces. The returned `errors` named tuple also reports volume-weighted `l2` and
pointwise `linf` values.
Pass `method=:cmdgfm` to compare CMDGFM with the default commuting mimetic
projection.

Direct execution requires GLMakie and benefits from multiple Julia threads:

```sh
julia --project=. --threads=auto examples/capillary_wave.jl
```
