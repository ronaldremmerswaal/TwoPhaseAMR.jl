# Full-Strang tile fusion candidate

`fused.jl` is an isolated Stage 2 research candidate, outside the installed
toy package. It is **not selected** as the package backend and is not used by
the main CFD package. Load it with `include("research/UniformLWAdvection/experiments/fused.jl")`
after loading `UniformLWAdvection`, then construct `FusedWorkspace(state)`
and call `fused_step!` or `fused_simulate!`.

The three sequential LW stages have a compact dependency cone. For a tile of
final outputs, the first x-half stage needs one extra cell in both x and y;
its reads of the old global field reach two cells beyond the output in x and
one in y. The y stage keeps one x-halo; the final x stage writes only its
owned output tile. Each bounded chunk owns both intermediate tile arrays and
periodic index lists. Chunk ownership survives Julia task migration. Face
fluxes on duplicated halos use the same positive orientation and arithmetic
as the package's recompute kernel; no concurrent output or scratch is shared.

One launch computes all final tiles into a distinct global `next` array. A
barrier then precedes a second, parallel tile copy into the original `state.q`
array. This preserves `q` identity for callers holding an external reference.
Swapping `state.q` with `next` could eliminate the copy but would change that
identity; it is deliberately absent from this prototype. The two launches
replace the three global sweep launches of recomputed flux. The base package's
stored-flux candidate uses six launches.

The numerical-array footprint is the existing three state arrays (`q,u,v`),
one global `next`, and per-chunk scratch of
`8[(tx+2)(ty+2)+(tx+2)ty]` bytes plus short periodic-index vectors. Thus
the fixed global footprint is four Float64 arrays rather than five for the
base recompute policy or six for stored flux. This is only a storage count,
not a measured DRAM-traffic claim. Fusion removes global materialization of
the two intermediate states but adds a full-grid read and write in copy-back,
duplicates halo work, and revisits the velocity arrays in all three local
stages. The scratch must fit a useful cache level for its tile size. At small
grids or excessive chunk counts, scratch and scheduling may erase the gain.
`fused_memory_bytes` reports numerical arrays and chunk index storage for a
specific workspace.

The experiment should be judged only after the ordinary toy's numerical and
scaling gates are measured. Compare the same constant-carrier timestep and
trajectory against the best serial package path at 1, 4, 8, and 16 workers,
with the already declared large cache-transition and DRAM fixtures. Include
both complete-step time and the copy-back component, worker participation,
allocations, error, mass, and numerical-array footprint. Vary only a small
predeclared tile list. Reject fusion if its serial cost or total step time
misses the plan's gate; a faster tile calculation alone is insufficient.

`test_fused.jl` supplies odd-size/seam, signed/zero velocity, multi-step
agreement, identity, mass, and smooth-order checks. It must pass before this
candidate enters timing. No claim is made yet about measured performance.
