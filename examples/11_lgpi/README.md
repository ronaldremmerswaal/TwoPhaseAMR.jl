# 11 – Large gas-pocket impact

A steep liquid front evolves over an embedded elliptical bathymetry in a scaled two-phase impact benchmark.

![Interface, bathymetry and vorticity at three times](assets/hero.png)

## At a glance

| | |
| --- | --- |
| **Physics** | Liquid/gas densities `(1000, 1)`, dynamic viscosities `(10⁻³, 1.75×10⁻⁵)`, surface tension `0.073` and gravity `(0, -9.81)`. At `scale=5` the nonperiodic domain is `4 × 2.4`, with free-slip outer boundaries and an embedded elliptical solid. |
| **Algorithms** | Two velocities by default (`TwoVelocityModel()`), with `FrozenInterfaceFluxes()` transport and `CMDGFMProjection()` (a block-preconditioned BiCGSTAB solve); WY geometric VOF, `:lhf` PLIC with WY fallback, CGFM surface tension, Lax–Wendroff momentum advection, implicit-midpoint viscosity, and CG/tree-geometric multigrid for diffusion. The interface indicator targets `min(1,max_level)` and the liquid-velocity vorticity indicator refines up to `max_level` at `0.25` and coarsens below `0.25/3`. Both use `ymin=refinement_ymin`; AMR is checked every 5 steps with coarsen delay 10. |
| **Worth varying** | `scale=5` changes the prescribed domain and initial geometry while leaving the listed physical coefficients fixed; `max_level=4` changes the vorticity-refinement ceiling; `final_time=nothing` selects `1.2` gravity periods. `refinement_ymin` overrides the default solid-adjacent exclusion height; `velocity_model=OneVelocityModel()` selects a shared velocity. Other supported `FlowSimulation` options include `numerics`. |
| **Validation status** | In the documented level-2 run, liquid volume changes by `−2.22×10⁻⁸` from `2.918817` over the full simulation. There are 1752 cumulative unresolved-curvature face events; the later fragmented interface is not resolved well enough to validate entrainment. No experimental impact or interface-position error is claimed. |
| **Cost** | The documented, instrumented two-velocity run used `max_level=2`, `T≈1.92265` (1.2 gravity periods), 8414 steps, 867 mesh updates and 5088 final cells; setup and simulation took about 1704 s on one thread in a shared process. The example default uses `max_level=4`. |

## Setup

The liquid initially lies below the tanh-shaped profile, while the negative side of the ellipse is solid. At the default scale,

```math
\eta(x)=1.52+0.72\tanh\!\bigl(2.2(x-3.1)\bigr),\qquad
\phi_s(x,y)=\sqrt{(x/3.6)^2+(y/0.56)^2}-1.
```

The ellipse reaches its highest point at `y=2.8/scale=0.56`. Adding two
level-zero cell heights (`2 × 0.1`) gives `0.76`; the next base-block boundary
is at `0.8`, and half a coarse cell above it gives `refinement_ymin=0.85`.
The geometry indicator is omitted because it would refine the solid boundary.
Blocks below this threshold request coarsening, but 2:1 grading can still
refine solid-adjacent blocks by one level next to refined blocks.

The initial velocity is zero. For wavelength `λ=4`, the script uses the gravity-wave timescale

```math
t_g=\sqrt{\frac{\rho_l+\rho_g}{\rho_l-\rho_g}\frac{2\pi\lambda}{g}},
\qquad T=1.2t_g.
```

See the [method reference](../../docs/src/method.md) for embedded cut cells, two-phase transport, forcing and adaptive refinement.

The vorticity plots show the intrinsic value of the majority phase in each
cell, with the reconstructed interface drawn on top. The vorticity refinement
indicator uses the liquid velocity, not the gas velocity.

## Run

From the repository root, a short headless run is:

```julia
include("examples/11_lgpi/lgpi.jl")
run = run_example(max_level=1, final_time=0.1)
print_simulation_summary(run.flow, run.result)
```

The first command runs the full two-velocity level-2 calculation and regenerates
the docs assets, but displays only its earlier frames; the second runs the
level-4 default without exporting figures. Both use the full end time and are
subject to the validation caveat above:

```sh
julia --project=examples -t 1 examples/11_lgpi/make_figures.jl
julia --project=examples -t 1 examples/11_lgpi/lgpi.jl
```

The figure generator also accepts `final_time` to change the simulated end
time, `display_time` to limit the exported frames without shortening the
simulation, and `recording` to load an existing serialized snapshot store:

```julia
include("examples/11_lgpi/make_figures.jl")
make_figures(recording="/path/to/recording.jls", display_time=0.9)
```

The default `display_time=0.97` limits the published figures while retaining
the full simulated recording. Set `display_time=nothing` to export all recorded
frames up to the simulation end time; this does not improve their resolution.

## Results

The hero displays two-velocity vorticity, the PLIC interface, AMR blocks and
filled bathymetry at three times from the level-2 run.

![Vorticity and gas-pocket evolution](assets/impact.gif)

The GIF shows the same run with block outlines and a synchronized
kinetic-energy cursor; the energy plot includes the full simulation history.
Vorticity uses a symmetric range of ±38.87, the 99.9th percentile of
rasterized magnitudes in the displayed frames. Larger values saturate: the
sampled peak is 589.85 near `(x,y)=(0.005,1.385)` at `t≈0.9613`.

The last displayed frame is at `t≈0.9613`: at the next saved frame
(`t≈1.0095`) detached liquid components occupy only 2 and 3 cells, and
unresolved-curvature events have begun. These late structures and their wall
interaction are not reliably resolved at this level, so the published
animation stops before them rather than implying validated entrainment. The
full simulation remains available through `run_example(final_time=nothing)`
and the full recorded animation through `make_figures(display_time=nothing)`.

## Variations

```julia
run_example(max_level=1, final_time=0.1)           # inexpensive short evolution
run_example(max_level=2, final_time=0.3)           # shorter level-2 evolution
run_example(scale=10, max_level=1, final_time=0.1) # rescale initial geometry
run_example(max_level=1, final_time=0.1, velocity_model=OneVelocityModel()) # shared velocity
```

## Files

- [lgpi.jl](lgpi.jl): benchmark geometry and simulation.
- [make_figures.jl](make_figures.jl): three snapshots and animation.
- `assets/`: generated PNG and GIF.
