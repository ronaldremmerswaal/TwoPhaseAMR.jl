# Stokes3 time-step selection: zero-allocation plan

This is point 3 of [the performance study](PLAN.md). The target is **zero
allocations for a warmed, complete `stable_timestep` query**, not merely for
its final rate scan. Measure both the one-velocity and CMDGFM paths. Include
the first query after an AMR topology change, once the replacement workspace
has been built; one-time AMR workspace construction is charged to mesh
updates, not hidden inside the CFL result.

## What currently happens

With automatic `dt`, `simulate!` enforces velocity boundary values and calls
`stable_timestep(...; cfl=0.45)` before each step. The one-velocity path scans
incoming/outgoing fluxes and face extrusion rates, then applies capillary and
gravity limits. The CMDGFM `FrozenInterfaceFluxes` path additionally calls
`reconstruct!`, freezes both phase apertures, and scans both phase fluxes.
The integrated step reuses that prepared interface geometry. A direct `step!`
or fixed-`dt` run instead checks `cfl=0.5` and prepares its own geometry;
standalone `advect!` also checks its bound. These calls must remain correct
even if there is no preceding outer query.

The original full Stokes3 profile attributed about 0.03 s to one-velocity
outer selection and 8.33 s to CMDGFM. The latest post-transport/diagnostics
`t=0.5` profiles found **0.033 s / 206 steps** and **4.78 s / 219 steps**
respectively. Within the latter sampled outer bucket, reconstruction accounted
for about 2.03 s, aperture freezing 0.29 s, and the frozen CFL scan 0.15 s;
other time remains to be attributed. These are whole-run, different-history
figures, not comparable per-query costs. The first measurement task below is
to establish a matched, allocation-profiled per-query baseline.

## Measurement and diagnosis

1. Add a focused benchmark script beside `benchmark.jl`. Use the default
   Stokes3 setup and a short advance through **at least one AMR change**.
   Capture identical saved states for isolated one- and two-velocity queries,
   and report block/face counts, calls, median and tail latency, `@allocated`,
   and returned `dt`. Benchmark a warm unchanged-topology state and the first
   query on a rebuilt topology separately. Also run the default `t=0.5`
   benchmark for integrated cost.
2. Split the measured CMDGFM query into boundary enforcement, PLIC
   reconstruction, phase-aperture freeze, the frozen-phase CFL scan, and the
   capillary/gravity calculation. Split the one-velocity query into its rate
   scan and force limits. Attribute inner `cfl=0.5` transport guards and WY
   safety checks separately from the outer `cfl=0.45` call. Count calls and
   report whether the winning limit is transport, capillary or gravity. Keep
   the measured implementation path intact: isolated kernels are explanatory,
   not substitutes for end-to-end measurements.
3. Use allocation profiling on the first *warmed* call at each state and
   verify hot-path dispatch. In particular, check PLIC scratch, aperture
   freezing, `_parallel_block_columns` closures/tasks, the incoming/outgoing
   field scan, `minimum` over block spacing, keyword dispatch, and any
   topology-dependent arrays. Distinguish real heap allocations from profiler
   bookkeeping and compile-time allocations with a repeated `@allocated`
   gate. Record outputs before optimization to detect numerical drift.

## Changes to try, in measured order

1. Remove allocation from the complete query using existing workspace
   arrays, concrete function barriers, and in-place reconstruction/aperture
   updates. Reuse any mesh-only minimum spacing through a mesh-owned cache if
   its rebuild/invalidation contract is clear; do not add a stale interface
   cache. Avoid making the one-velocity path slower or allocating.
2. If reconstruction or aperture preparation dominates time, separate
   *preparation* from the rate scan. The preparation may be reused only across
   calls that see the same interface, mesh and relevant velocity state. A
   step-local token owned by the integrated stepping path could document this
   validity; direct public calls must prepare independently. Confirm whether
   the current outer-to-transport reuse already eliminates the duplicate in
   the usual automatic-`dt` path before adding any new mechanism.
3. Check whether an inner guard can reuse an outer bound by converting the
   transport rate from `cfl=0.45` to `0.5`. A single scalar final bound is
   insufficient if a capillary or gravity limit won: retain the individual
   transport and force bounds, or simply rescan when validity is uncertain.
   Never weaken the safety check or rely on a bound from before velocity,
   interface, or mesh changes.

## Validation gates

- **Allocation:** after warm-up, `@allocated(stable_timestep(...)) == 0` for
  both models on uniform and AMR states, including the first query after a
  prepared AMR rebuild. Test the full solver-level query, not only the CFL
  kernel. Any intentional one-time workspace rebuild allocation is measured
  and reported separately. Keep repeated-query results identical.
- **Numerical safety:** compare transport, capillary, gravity and final bounds
  against the current implementation on frozen states, immediately after AMR,
  and across density ratios/forces. Test `dt` just below and above each active
  bound. Existing direct `step!` and public `advect!` safety checks must still
  reject an unsafe step.
- **Integration:** run the one- and two-velocity transport tests, AMR and
  embedded-geometry tests, Stokes3 example tests, then the full suite. In
  short and `t=0.5` Stokes3 runs, compare step count, AMR history, diagnostics
  and final fields to the baseline. If the bound policy is intentionally
  changed, explain any trajectory difference rather than accepting it as
  performance noise.
- **Performance:** paired warm short runs should show lower time and bytes in
  the outer time-step bucket without moving equal work into transport or AMR.
  Confirm the gain in the default `t=0.5` run and report per-query cost as
  well as whole-run totals; differing step counts must not masquerade as a
  faster kernel.

This file is a plan only; it does not change time-step selection.
