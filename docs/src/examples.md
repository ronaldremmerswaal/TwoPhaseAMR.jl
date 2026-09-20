# Example catalogue

The files under `examples/` are complete, standalone applications rather than
hidden test helpers. Each exposes `run_example`, runs headlessly by default
when included, enables visualization when executed from a terminal, and lets
`simulate!` choose a stable timestep.

| Example | Physics and representative result | Default model | Alternate model | Typical cost |
| --- | --- | --- | --- | --- |
| `taylor_green.jl` | periodic vortex decay and energy history | single phase | — | seconds |
| `lid_driven_cavity.jl` | moving-wall recirculation | single phase | — | minutes |
| `cylinder_flow.jl` | cut-cell wake and pressure drag | single phase | — | long |
| `decaying_turbulence.jl` | pressure-driven dynamic refinement | single phase | — | minutes |
| `capillary_wave.jl` | numerical and linear wave amplitude | one velocity | `TwoVelocityModel()` | minutes |
| `sessile_droplet.jl` | contact-angle relaxation under gravity | one velocity | `TwoVelocityModel()` | long |
| `sloped_sessile_droplet.jl` | contact on an embedded inclined wall | one velocity | `TwoVelocityModel()` | long |
| `stokes3_wave.jl` | nonlinear free-surface wave with AMR | one velocity | `TwoVelocityModel()` | long |
| `lgpi.jl` | large gas-pocket impact | one velocity | `TwoVelocityModel()` | very long |
| `interfacial_instability.jl` | KH/RT amplitude against linear theory | **two velocity** | `OneVelocityModel()` | long |
| `vortex_reverse.jl` | prescribed-carrier VOF reversal | prescribed velocity | — | minutes |

For a quick first run, set `max_level=0` and a short `final_time`. For example:

```julia
include("examples/interfacial_instability.jl")
run = run_example(
    max_level=0,
    final_time=0.02,
    velocity_model=OneVelocityModel(),
)
```

The returned value contains the flow and simulation result, so diagnostics can
be inspected without visualization. Direct execution prints
[`print_simulation_summary`](@ref). Full convergence studies and literature
comparisons live under `test/integration/`; the examples
are intentionally focused on one inviting physical setup.
