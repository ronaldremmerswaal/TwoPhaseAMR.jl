# Affine-indexed fused prototype

`fused_affine.jl` reuses the frozen `fused.jl` workspace, chunk ownership,
validation, and final copy-back barrier. It replaces only the tile kernel:
interior cell loops use affine global indices and `@simd`, while small scalar
sections handle physical periodic seams. Each chunk still recomputes the
required x-half and y-full halo values in local scratch, then writes its
uniquely owned final output tile. The step has one tile-computation barrier
and one copy-back barrier; `State.q` keeps its identity. It still stores a
full-grid `next` buffer, and it has the same numerical-array footprint as the
original fused prototype.
When Polyester is on the load path, `fused_affine_polyester.jl` supplies
typed, direct chunk loops for both passes. This avoids per-tile argument
boxing through Polyester's nested callback while keeping the same chunk
ownership and numerical kernel.

`fused_simulate_amortized!` is a separate trajectory experiment. It swaps
local source/destination references after each completed tile pass and copies
back to the original `State.q` only once when the step count is odd. This
preserves `State.q` identity on return, but it does not expose intermediate
states during the trajectory. `fused_step!` still performs a complete
copy-back every step. Report single-step and amortized-trajectory timings
separately; a trajectory gain does not imply a faster individual step.

For untimed worker participation evidence, construct `WorkerProbe(work;
nsteps)` and call `fused_step!(state,work,dt,probe)` with `nsteps=1`, or
`fused_simulate_amortized!(state,work,dt,nsteps,probe)`. `probe.compute` stores
one actual worker ID per chunk and step; `probe.copyback` stores IDs for the
final copy-back pass when it occurs. `active_workers(probe,pass)` and
`active_copyback_workers(probe)` summarize those caller-owned records.

This experiment tests whether the original fused kernel's gathered periodic
indices, rather than fusion itself, caused its serial slowdown. Run
`test_fused_affine.jl` with the package project active, and the optional
`test_fused_affine_polyester.jl` when Polyester is available. Time identical
grid, timestep, tile, backend, and step counts against the frozen fused and
core recomputed-flux paths; include the complete copy-back in the timed step.
