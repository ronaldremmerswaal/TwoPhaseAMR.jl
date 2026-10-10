# Research-only main-package cell-centred LW prototype

This directory holds the uniform `CellField` Lax–Wendroff prototype removed from
production source. It evaluates a research method into the loaded `TwoPhaseAMR`
module without changing the package's source files or exports:

```julia
using TwoPhaseAMR
include("research/UniformLWAdvection/experiments/main_cell_lw/MainCellLW.jl")
using .MainCellLW

mesh = BlockTreeMesh(base_mesh=(64, 64), block_size=(8, 8))
field = CellField(mesh, Float64)
velocity = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.7 : -0.3)
workspace = UniformCellLWWorkspace(field, velocity)
advect!(field, velocity, 0.001, workspace)
```

The prototype accepts Float64 cell averages on a uniform periodic unit square
with a spatially constant registered carrier. Use `set_uniform_lw_velocity!`
when that carrier changes; direct mutation of the carrier arrays is outside its
contract. It applies `x(dt/2), y(dt), x(dt/2)` and has its own workspace, so it
is separate from production upwind advection and staggered LW.

Run focused correctness checks with the Julia environment used by the project:

```sh
julia --project=. research/UniformLWAdvection/experiments/main_cell_lw/validate.jl
```

`validate.jl` compares full fields with a scalar periodic reference, checks
seams, conservation, smooth order, workspace isolation, and topology validity.
