# Examples gallery

These standalone examples each define `run_example` and run headlessly. The tables on each page
describe physical parameters, algorithms, validation and cost. The
manufactured Poisson case is a static projection rather than a time-dependent
flow.

| # | Example | Physics | Default model | Default cost |
| --- | --- | --- | --- | --- |
| 01 | [Taylor–Green vortex](01_taylor_green/) | Periodic vortex decay | Single phase | Level 2, T=0.2; 0.04 s warm† |
| 02 | [Lid-driven cavity](02_lid_driven_cavity/) | Moving-wall recirculation | Single phase | Level 1, T=15; unmeasured |
| 03 | [Decaying turbulence](03_decaying_turbulence/) | Random vortices and adaptive refinement | Single phase | Level 4, T=10; unmeasured |
| 04 | [Cylinder flow](04_cylinder_flow/) | Cut-cell wake and drag | Single phase | Level 3, T=40; unmeasured |
| 05 | [Vortex reversal](05_vortex_reverse/) | Geometric VOF transport | Prescribed velocity | Level 5, T=8; unmeasured |
| 06 | [Capillary wave](06_capillary_wave/) | Viscous wave damping | One velocity | Level 2, one period; 1.3 s warm† |
| 07 | [Interfacial instability](07_interfacial_instability/) | Kelvin–Helmholtz and Rayleigh–Taylor growth | Two velocity | Level 1, until doubled amplitude; unmeasured |
| 08 | [Sessile droplet](08_sessile_droplet/) | Contact-angle relaxation | One velocity | Level 3, eight capillary times; unmeasured |
| 09 | [Sloped sessile droplet](09_sloped_sessile_droplet/) | Embedded inclined wall | One velocity | Level 2, four capillary times; 6.6 s warm† |
| 10 | [Stokes wave](10_stokes3_wave/) | Nonlinear free-surface wave | One velocity | Level 5, one period; unmeasured |
| 11 | [Large gas-pocket impact](11_lgpi/) | Gravity wave over bathymetry | One velocity | Level 3, 0.6 gravity periods; unmeasured |
| 12 | [Manufactured Poisson](12_manufactured_poisson/) | Pressure projection with density jump | Two velocity | 32×32; unmeasured |

† Historical Julia 1.12.7 single-thread warm timings from the documentation
audit, before the current precompilation workload; hardware and setup affect
cost. First-call compilation is additional. See each page for docs-size runs
and validation details.

Current gallery measurements were taken with Julia 1.12.7 on an AMD Ryzen 9
5950X, using one Julia thread and one BLAS thread. Setup/simulation timings
exclude package loading and may include just-in-time compilation; individual
pages identify the settings and whether a measurement is warm or shared.

## Running

From the repository root, first prepare the environment with
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`. Include an
example for headless use:

```julia
include("examples/06_capillary_wave/capillary_wave.jl")
run = run_example(max_level=1, final_time=0.01)
```

To run an example directly, for example,
`julia --project=examples examples/06_capillary_wave/capillary_wave.jl`.
Most flow examples return `(; flow, result, recorder)`; keyframes are available
via `snapshot_store(run.recorder)`. Vortex reversal returns its interface,
prescribed velocity, result and recorder, while manufactured Poisson returns
its projection fields, errors and a `registry`. For most cases, a low `max_level`
and short `final_time` make a quick initial run.

## Reproducing figures

With CairoMakie installed in the examples environment, regenerate the static
assets:

```sh
julia --project=examples examples/make_all_figures.jl
julia --project=examples examples/make_all_figures.jl --only 06,07
```

The default `--size docs` uses the smaller documented configurations and
enforces the 150 kB PNG / 600 kB GIF limits. Use `--size default` for the
example defaults; rerun at docs size before committing those assets. Generated
assets are committed so a documentation build need not rerun the simulations.
