# Milestone 1 benchmark results

Results are pending execution on the current implementation. Run
`benchmark/benchmarks.jl` as described in `benchmark/README.md` and record the
Julia version, CPU, thread count, median time, memory, and allocations here.

The milestone-1 performance contract enforced by tests is zero allocations for
`advect!`, `fill_halo!`, and `stable_timestep` after mesh/workspace setup.
Pressure-solve comparison against `IncompressibleNavierStokes.jl` belongs to
milestone 3 and has no fabricated result here.
