# Periodic halo-copy layout experiment

`halo.jl` compares padded storage with the core's explicit periodic seam
loops. `PaddedState(core_state)` makes one full-grid copy during construction;
all five numerical arrays then have `(nx+2) × (ny+2)` storage. A timed
`padded_step!` refreshes only `O(nx+ny)` halo entries before each of its three
Strang sweeps. The prescribed constant face-speed arrays are copied once and
read in every sweep. Each sweep has one tile barrier, just like the core's
recomputed-flux path. The physical `q` array keeps its identity and no
full-grid copy occurs during a step.

This is a constant-carrier layout comparison, with serial, Julia Threads, and
optional Polyester scheduling. It is deliberately separate from the core
package and the temporal-fusion experiment. Include `test_halo.jl` with the
package project active; include `test_halo_polyester.jl` only when Polyester is
loaded in the environment. Benchmark initialization and halo setup outside the
timed loop, and compare against the core recomputed-flux path at the same grid,
step count, and backend.
