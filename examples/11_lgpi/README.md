# 11 – Large gas-pocket impact

A steep liquid front evolves over an embedded elliptical bathymetry in a scaled two-phase impact benchmark.

![Interface, bathymetry and vorticity at three times](assets/hero.png)

## At a glance

| | |
| --- | --- |
| **Physics** | Liquid/gas densities `(1000, 1)`, dynamic viscosities `(10⁻³, 1.75×10⁻⁵)`, surface tension `0.073` and gravity `(0, -9.81)`. At `scale=5` the nonperiodic domain is `4 × 2.4`, with free-slip outer boundaries and an embedded elliptical solid. |
| **Algorithms** | One velocity by default; WY geometric VOF, `:lhf` PLIC with WY fallback, CGFM surface tension, Lax–Wendroff momentum advection, implicit-midpoint viscosity, and CG/tree-geometric multigrid for pressure and two-phase diffusion. Interface and geometry indicators refine to `min(1,max_level)`; vorticity refines up to `max_level` at `0.25` and coarsens below `0.25/3`. AMR is checked every 5 steps with coarsen delay 10. |
| **Worth varying** | `scale=5` changes the prescribed domain and initial geometry while leaving the listed physical coefficients fixed; `max_level=3` changes the vorticity-refinement ceiling; `final_time=nothing` selects `0.6` gravity periods. Other supported `FlowSimulation` options include `velocity_model` and `numerics`. |
| **Validated against** | Liquid volume should remain constant: `:phase_volume` changes by `-1.11522e-9` at docs size, and the final volume-weighted `:divergence_norm` is `3.67718e-6` against zero. No experimental impact or interface-position error is claimed. |
| **Cost** | Default: `max_level=3`, `T≈0.9613`, runtime and final cells unmeasured. Docs: `max_level=2` at the same time, 3516 steps, 177 regrids and 6480 final cells; setup and simulation took 613.77 s in a single-thread shared gallery process, including incremental compilation. |

## Setup

The liquid initially lies below the tanh-shaped profile, while the negative side of the ellipse is solid. At the default scale,

```math
\eta(x)=1.52+0.72\tanh\!\bigl(2.2(x-3.1)\bigr),\qquad
\phi_s(x,y)=\sqrt{(x/3.6)^2+(y/0.56)^2}-1.
```

The initial velocity is zero. For wavelength `λ=4`, the script uses the gravity-wave timescale

```math
t_g=\sqrt{\frac{\rho_l+\rho_g}{\rho_l-\rho_g}\frac{2\pi\lambda}{g}},
\qquad T=0.6t_g.
```

See the [method reference](../../docs/src/method.md) for embedded cut cells, two-phase transport, forcing and adaptive refinement.

## Run

From the repository root, a short headless run is:

```julia
include("examples/11_lgpi/lgpi.jl")
run = run_example(max_level=1, final_time=0.1)
print_simulation_summary(run.flow, run.result)
```

```sh
julia --project=examples examples/11_lgpi/make_figures.jl
julia --project=examples examples/11_lgpi/lgpi.jl
```

## Results

The hero displays vorticity, the PLIC interface, filled bathymetry and block outlines at states nearest `0`, `0.3t_g` and `0.6t_g`; the titles give the actual times. The three color ranges are selected independently.

![Vorticity and gas-pocket evolution](assets/impact.gif)

The GIF uses 21 nearest completed states and an independently scaled vorticity color range per frame. The run ends with kinetic energy `3702.43` and liquid-volume change `-1.11522e-9`. A historical audit saw one `diffusion matrix and stiffness sparsity differ` failure at level 2; two later runs completed, and a multigrid cache-key defect was subsequently fixed, though its connection to that one-off failure was not established.

## Variations

```julia
run_example(max_level=1, final_time=0.1)           # inexpensive short evolution
run_example(max_level=2, final_time=0.3)           # shorter level-2 evolution
run_example(scale=10, max_level=1, final_time=0.1) # rescale initial geometry
```

## Files

- [lgpi.jl](lgpi.jl): benchmark geometry and simulation.
- [make_figures.jl](make_figures.jl): three snapshots and animation.
- `assets/`: generated PNG and GIF.
