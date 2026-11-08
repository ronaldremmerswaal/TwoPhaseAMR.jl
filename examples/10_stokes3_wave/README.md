# 10 – Third-order Stokes wave

A steep, finite-depth wave propagates for one nominal period on an adaptively refined two-phase mesh.

![Computed and third-order reference profiles](assets/hero.png)

## At a glance

| | |
| --- | --- |
| **Physics** | Two phases, density ratio `850`, `g=1`, Bond number `1000`, Reynolds number `4×10⁴`, steepness `0.55` and depth `0.5`. The domain is `[-0.5,0.5]²`, periodic in `x` with free-slip `y` boundaries. |
| **Algorithms** | One velocity by default; WY geometric VOF, `:lhf` PLIC with WY fallback, CGFM surface tension, Lax–Wendroff momentum advection and implicit-midpoint viscosity. Pressure and two-phase diffusion use CG with tree-geometric multigrid. AMR combines the interface indicator through `t=0.01`, curvature at 8 cells per radius, and vorticity thresholds `0.2` and `0.2/3`; adaptation is attempted every step with coarsen delay 2. |
| **Worth varying** | `steepness=0.55` changes the initialized wave and frequency; `bond=1000` changes surface tension; `reynolds=4e4` changes viscosity; `density_ratio=850` changes gas density; `max_level=5` sets the refinement ceiling; `final_time=nothing` selects one nominal period; `visualize=false` disables the viewer and `phase_rendering=:cell` selects the default phase display. |
| **Validated against** | At docs size, the final interface at 200 evenly spaced `x` positions differs from the analytical third-order *initial* profile by sample RMS `0.0258386` and maximum `0.0580631`. This approximate Stokes profile is not an exact solution of the viscous, capillary, two-phase problem. |
| **Cost** | Default: `max_level=5`, one period, runtime and final cells unmeasured. Docs: `max_level=3`, `T≈2.19852`, 250 steps, 15 regrids and 2368 final cells; setup and simulation took 15.34 s in a single-thread shared gallery process, including incremental compilation. |

## Setup

The liquid occupies the region below the third-order profile returned by `stokes3_initial_condition().elevation`; its initial analytical velocity is projected before time integration. With `k=2π`, `h=0.5`, `ε=0.55` and `χ=coth(kh)`, the initialization uses

```math
\omega^2=\frac{gk}{\chi}\left[1+\epsilon^2\left(\frac98(\chi^2-1)^2+\chi^2\right)\right],
\qquad T=\frac{2\pi}{\omega},\qquad
\sigma=\frac{(\rho_l-\rho_g)g}{Bo\,k^2}.
```

The viscosity pair is `(1/Re, 1/(51.15Re))`. The plotted reference is the third-order initialization expression evaluated at the same `x` values, not a fitted or shifted curve. See the [method reference](../../docs/src/method.md) for VOF, projection, viscosity and AMR.

## Run

From the repository root:

```julia
include("examples/10_stokes3_wave/stokes3_wave.jl")
run = run_example(max_level=3)  # one nominal wave period
print_simulation_summary(run.flow, run.result)
```

```sh
julia --project=examples examples/10_stokes3_wave/make_figures.jl
julia --project=examples examples/10_stokes3_wave/stokes3_wave.jl
```

## Results

After one nominal period, the sampled interface differs from the third-order initialization by RMS `0.0258386` and maximum `0.0580631`; these differences include physical model differences as well as numerical error.

![Final vorticity with interface and AMR blocks](assets/vorticity.png)

Vorticity at the final retained state, with the PLIC interface and leaf-block boundaries. The signed vorticity color range is selected automatically.

![Vorticity animation](assets/wave.gif)

The animation contains 21 retained states nearest equally spaced requested times; each frame's vorticity scale is chosen independently. The summary also prints kinetic energy, phase volumes, divergence, momentum, regrids, final cell count and elapsed setup/simulation time.

## Variations

```julia
run_example(steepness=0.35, max_level=3)          # reduce wave nonlinearity
run_example(max_level=4)                           # increase spatial refinement
run_example(max_level=3, velocity_model=TwoVelocityModel()) # liquid wave, gas initially at rest
```

The two-velocity option uses the script's phase-specific initializer; the displayed profile comparison above uses the default one-velocity model.

## Files

- [stokes3_wave.jl](stokes3_wave.jl): analytical initialization and simulation.
- [make_figures.jl](make_figures.jl): profile, vorticity and animation generation.
- `assets/`: generated PNGs and GIF.
