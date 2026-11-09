# 04 – Cylinder flow

A uniform inflow passes an embedded circular cylinder while adaptive blocks follow the solid and vorticity.

![Vorticity and embedded cylinder with block boundaries](assets/vorticity.png)

## At a glance

| | |
| --- | --- |
| **Physics** | Single phase, density 1, unit free-stream speed and unit cylinder diameter. The default is `reynolds=100` (viscosity 0.01); the gallery uses `reynolds=200` (viscosity 0.005). The domain is `[-4, 10] × [-4, 4]`; left inflow is `(1, 0)`, the right is outflow, and top and bottom use zero-gradient velocity. |
| **Algorithms** | Embedded `GeometryField` with PLIC solid geometry, staggered Lax–Wendroff advection, implicit-midpoint diffusion and pressure projection. Pressure uses CG/tree multigrid; diffusion uses tree multigrid and selects CG or BiCGSTAB by matrix symmetry. Geometry and vorticity indicators drive AMR above the base level, with update interval and coarsen delay 5. |
| **Worth varying** | `reynolds=100` changes viscosity; `max_level=3` changes finest resolution and AMR availability; `final_time=40` changes the simulated duration; `dt=nothing` selects automatic steps. At `max_level ≤ 1`, the example disables dynamic criteria. |
| **Validated against** | In the `t=30–50` window, the mean pressure-only drag coefficient is `1.02563`, and the Strouhal number from upward zero crossings of low-pass-filtered pressure lift is `0.2303`. [Qu et al. (2013), Table 3](https://doi.org/10.1016/j.jfluidstructs.2013.02.007) report Re = 200 values of 0.1958 for Strouhal, 1.068 for mean pressure drag and 1.316 for mean **total** drag. This under-resolved calculation in a small domain is not a grid- or domain-independent benchmark, and the pressure-only diagnostic excludes viscous drag. |
| **Cost** | Default: level 3, `T=40`, runtime and final cells unmeasured. The gallery uses level 3 and `T=50`, with 83169 steps and 6160 final cells; setup and simulation took 1349 s on one thread. The larger figure setting uses level 4 and `T=50`. |

## Setup

The solid is initialized from `hypot(x, y) - 0.5`, with `4 × 4`-cell blocks on a `28 × 16` base mesh. The default initial fluid velocity is the unit inflow. The gallery adds a localized divergence-free initial vortex with streamfunction `0.1 exp(-((x-1)^2+(y-0.5)^2)/(2×0.5^2))` to seed the wake early; the inflow boundary stays `(1,0)`. Re = 200 was chosen for comparison with a published two-dimensional benchmark while encouraging shedding. Geometry is refined to `max_level`, while the vorticity criterion uses thresholds 0.2 for refinement and `0.2/3` for coarsening. The pressure-only drag diagnostic reports

```math
C_{D,p}=\frac{F_{p,x}}{\tfrac12\rho U^2D}=2F_{p,x}
\quad (\rho=U=D=1).
```

The analogous `:pressure_lift` uses the vertical force. Neither diagnostic includes viscous stresses. See the [method reference](../../docs/src/method.md) for cut cells, pressure projection and refinement criteria.

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

Use `make_figures(; outdir=..., size=:default)` for level 4 and `T=50`. Both sizes print the diagnostic summary and runtime.

## Results

The hero shows final vorticity, gray leaf-block outlines and the reconstructed solid cylinder in black.

![Pressure drag coefficient as a function of time](assets/pressure_drag.png)

![Vortex shedding behind the cylinder](assets/shedding.gif)

The last 20 time units of the docs run (48 retained keyframes from approximately `t=30.42` to `t=50`): vorticity with a fixed range of ±3, AMR block outlines and the embedded cylinder in gray.

For the frequency estimate, lift was sampled every 0.05 time units, averaged over an 11-sample window, and its upward mean crossings used to estimate the period. The residual of the raw sampled lift minus the smoothed signal has RMS `0.05367`, about 5.13% of the smoothed lift amplitude; this noise and the limited averaging window should be considered in the comparison. Initial pressure force samples are deliberately `NaN`, because the impulsive-start state is not a developed force. These diagnostics exclude viscous stresses.

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
