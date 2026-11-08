# 04 – Cylinder flow

A uniform inflow passes an embedded circular cylinder while adaptive blocks follow the solid and vorticity.

![Vorticity and embedded cylinder with block boundaries](assets/vorticity.png)

## At a glance

| | |
| --- | --- |
| **Physics** | Single phase, density 1, `reynolds=100`, viscosity 0.01, unit free-stream speed and unit cylinder diameter. The domain is `[-4, 10] × [-4, 4]`; left inflow is `(1, 0)`, the right is outflow, and top and bottom use zero-gradient velocity. |
| **Algorithms** | Embedded `GeometryField` with PLIC solid geometry, staggered Lax–Wendroff advection, implicit-midpoint diffusion and pressure projection. Pressure uses CG/tree multigrid; diffusion uses tree multigrid and selects CG or BiCGSTAB by matrix symmetry. Geometry and vorticity indicators drive AMR above the base level, with update interval and coarsen delay 5. |
| **Worth varying** | `reynolds=100` changes viscosity; `max_level=3` changes finest resolution and AMR availability; `final_time=40` changes the simulated duration; `dt=nothing` selects automatic steps. At `max_level ≤ 1`, the example disables dynamic criteria. |
| **Validated against** | Exact incompressibility gives a zero reference for `:divergence_norm`; the docs run ends at `4.30412e-4` in the volume-weighted norm. Its instantaneous `:pressure_drag` is `0.604378` at `t=10`. This short transient is not a quantitative comparison with the longer-time [Verstappen and Dröge benchmark](https://doi.org/10.1016/j.crme.2004.09.021). |
| **Cost** | Default: level 3, `T=40`, runtime and final cells unmeasured. Docs: level 2, `T=10`, 2368 final cells and 3804 steps; setup and simulation took 27.42 s in a single-thread shared gallery process, including incremental compilation. |

## Setup

The solid is initialized from `hypot(x, y) - 0.5`, with `4 × 4`-cell blocks on a `28 × 16` base mesh. Initially the fluid velocity is the unit inflow. Geometry is refined to `max_level`, while the vorticity criterion uses thresholds 0.2 for refinement and `0.2/3` for coarsening. The pressure-only drag diagnostic reports

```math
C_{D,p}=\frac{F_{p,x}}{\tfrac12\rho U^2D}=2F_{p,x}
\quad (\rho=U=D=1).
```

It does not include viscous drag. See the [method reference](../../docs/src/method.md) for cut cells, pressure projection and refinement criteria.

## Run

From the repository root, use the gallery settings headlessly:

```julia
using TwoPhaseAMR
include("examples/04_cylinder_flow/cylinder_flow.jl")
run = run_example(max_level = 2, final_time = 10.0)
drag = diagnostic_history(run.flow, :pressure_drag)
```

The direct example records keyframes in `run.recorder` (accessible via `snapshot_store(run.recorder)`) with the longer defaults. The second command renders the smaller gallery run using CairoMakie:

```sh
julia --project=examples examples/04_cylinder_flow/cylinder_flow.jl
julia --project=examples examples/04_cylinder_flow/make_figures.jl
```

Use `make_figures(; outdir=..., size=:default)` for level 3 and `T=40`. Both sizes print the diagnostic summary and runtime.

## Results

The hero shows final vorticity, gray leaf-block outlines and the reconstructed solid cylinder in black.

![Pressure drag coefficient as a function of time](assets/pressure_drag.png)

The initial `:pressure_drag` is deliberately `NaN`, because the impulsive-start state is not a developed force. At `t=10` the pressure-only coefficient is `0.604378`; it is not total drag or a Strouhal validation.

## Variations

After including the example, try:

```julia
run_example(max_level = 1, final_time = 10.0)  # base-level mesh, no dynamic criteria
run_example(max_level = 2, final_time = 20.0)  # longer adaptive wake
run_example(reynolds = 50, max_level = 2, final_time = 10.0)  # more viscous flow
```

## Files

- [cylinder_flow.jl](cylinder_flow.jl) — embedded geometry, AMR and pressure-drag diagnostic.
- [make_figures.jl](make_figures.jl) — headless solid/vorticity and drag plots.
