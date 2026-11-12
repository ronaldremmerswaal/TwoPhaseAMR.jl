# Examples gallery

These standalone examples each define `run_example` and run headlessly by
default. The tables on each page describe physical parameters, algorithms,
validation and cost. The manufactured Poisson case is a static projection
rather than a time-dependent flow.

| # | Example | Physics | Default model | Default cost |
| --- | --- | --- | --- | --- |
| 01 | [Taylor–Green vortex](01_taylor_green/) | Periodic vortex decay | Single phase | Level 2, T=0.2; 7 steps, 0.13 s warm |
| 02 | [Lid-driven cavity](02_lid_driven_cavity/) | Moving-wall recirculation | Single phase | Level 1, T=15; 1900 automatic steps, 2.41 s warm |
| 03 | [Decaying turbulence](03_decaying_turbulence/) | Random vortices and adaptive refinement | Single phase | Level 4, T=10; 18213 steps, 106.4 s |
| 04 | [Cylinder flow](04_cylinder_flow/) | Cut-cell wake and drag | Single phase | Level 3, T=40; unmeasured (gallery T=50: 83169 steps, 930.7 s) |
| 05 | [Vortex reversal](05_vortex_reverse/) | Geometric VOF transport | Prescribed velocity | Level 5, T=8; unmeasured |
| 06 | [Capillary wave](06_capillary_wave/) | Viscous wave damping | One velocity | Level 4, one period; 75.49 s, two velocities 119.34 s |
| 07 | [Interfacial instability](07_interfacial_instability/) | Kelvin–Helmholtz and Rayleigh–Taylor growth | Two velocity | Level 4, until doubled amplitude; 68.58 s, one velocity 19.79 s |
| 08 | [Sessile droplet](08_sessile_droplet/) | Contact-angle relaxation | One velocity | Level 3, eight capillary times; 32.11 s |
| 09 | [Sloped sessile droplet](09_sloped_sessile_droplet/) | Embedded inclined wall | One velocity | Level 2, four capillary times; unmeasured at this size (level-3 gallery run 127.77 s) |
| 10 | [Stokes wave](10_stokes3_wave/) | Nonlinear free-surface wave | One velocity | Level 5, one period; unmeasured |
| 11 | [Large gas-pocket impact](11_lgpi/) | Gravity wave over bathymetry | One velocity | Level 4, 1.2 gravity periods; unmeasured (gallery level-3 run 5778.1 s) |
| 12 | [Manufactured Poisson](12_manufactured_poisson/) | Pressure projection with density jump | Two velocity | 32×32; 11.86 s setup and solve |

Warm timings are second calls in one process; first-call compilation is
additional. See each page for docs-size runs and validation details.

Current gallery measurements were taken with Julia 1.12.7 on an AMD Ryzen 9
5950X, using one Julia thread and one BLAS thread. Setup/simulation timings
exclude package loading and may include just-in-time compilation; individual
pages identify the settings and whether a measurement is warm or shared.

## Running

From the repository root, first prepare the environment with
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`. The example
environment resolves TwoPhaseAMR and TwoPhaseAMRStudio from this checkout.
From inside `examples/`, use `--project=.` and omit the `examples/` prefix in
the paths below. Include an example for headless use:

```julia
include("examples/06_capillary_wave/capillary_wave.jl")
run = run_example(max_level=1, final_time=0.01)
```

To run an example directly, for example,
`julia --project=examples examples/06_capillary_wave/capillary_wave.jl`.
Most flow examples return `(; flow, result, recorder, app)`; keyframes are
available via `snapshot_store(run.recorder)`. Vortex reversal returns its
interface, prescribed velocity, result, recorder and app. LGPI returns its
flow, result and recorder, while manufactured Poisson returns its projection
fields, errors and a `registry`. For most cases, a low `max_level` and short
`final_time` make a quick initial run.

## Studio

Examples 01–10 can opt into live browser viewing from the command line:

```sh
julia -t 2,1 --project=examples examples/06_capillary_wave/capillary_wave.jl --studio
```

The last thread is an interactive thread, recommended for a responsive server
during a solve (especially on Julia 1.11). The command keeps the server open
after the solve; press Ctrl+C to stop it. Without `--studio` the script runs
headlessly and does not load Studio.

From a Julia session using the examples environment, pass Studio's constructor
to start the server before stepping:

```julia
using TwoPhaseAMRStudio
include("examples/06_capillary_wave/capillary_wave.jl")
run = run_example(max_level=1, final_time=0.01, start_studio=studio)
# The browser can inspect the completed run until the server is closed.
close(run.app)
```

To avoid launching a browser, use
`start_studio = recorder -> studio(recorder; open=:none)` and open the printed
URL. Treat that tokenized URL as a credential. Interactive runs record
wall-clock-throttled previews by default; `recorder_options` overrides Recorder
settings, e.g. `(; preview=EveryStep())`. Headless runs default to keyframes
only. For LGPI, `studio(run.recorder)` opens its completed recording; the
static Poisson example does not have a recorder.

## Reproducing figures

With CairoMakie installed in the examples environment, regenerate the static
assets:

```sh
julia --project=examples examples/make_all_figures.jl
julia --project=examples examples/make_all_figures.jl --only 06,07
```

The default `--size docs` uses the smaller documented configurations and
enforces the 150 kB PNG / 2 MB GIF limits. Use `--size default` for the
example defaults; rerun at docs size before committing those assets. Generated
assets are committed so a documentation build need not rerun the simulations.
