# Diagnostics cost: focused study plan

This is the next workstream after transport. The first measurements and
codebase conclusions are in
[`DIAGNOSTICS_FINDINGS.md`](DIAGNOSTICS_FINDINGS.md), generated with
[`diagnostics_analysis.jl`](diagnostics_analysis.jl). This document retains
the measurement and implementation gates; the diagnostics implementation now
uses mesh-/geometry-owned face-volume caches and in-place per-face fraction
evaluation. Keep
the default Stokes3 diagnostics in the reference runs: `KineticEnergy`,
`LinearMomentum`, `PhaseVolume`, `DivergenceNorm`, and `LeafBlockCount`, all
scheduled every step by default.

## What the current profiles say

The warm, post-transport-change `t=0.05` runs both made four AMR updates and
ended at 169 blocks / 21,832 faces (21 one-velocity steps, 22 CMDGFM steps):

| Sampled diagnostic cost | One velocity | CMDGFM two velocities |
| --- | ---: | ---: |
| Time | 1.64 s | 2.62 s |
| Allocation | ~1.46 GB | ~5.24 GB |
| Nested time in `_diagnostic_face_volumes` | ~1.63 s | ~1.33 s |

At `t=0.5`, the diagnostic buckets were 7.77 s and 19.23 s respectively;
`_diagnostic_face_volumes` accounted for roughly 7.69 s and 11.58 s. The
late runs had different AMR histories and final meshes (97 versus 166 blocks),
so the apparent ~2.5x whole-run diagnostic ratio is **not** a same-mesh
per-evaluation ratio. Even the short runs have one step of difference. These
are sampled estimates from one warm run and require direct timing.

## Codebase findings and expected problems

1. **Face dual volumes are rebuilt twice per diagnostic event.** Energy and
   momentum each call `_diagnostic_face_volumes`
   ([`interface.jl`](../../src/simulation/diagnostics/interface.jl)). In
   Stokes3, the no-solid branch calls `_uncut_face_dual_volumes`, which builds
   an entire `_amr_dual_schedule` for *each axis* merely to take its `.volumes`
   ([`embedded_operators.jl`](../../src/indicator/embedded_operators.jl),
   [`staggered_schedules.jl`](../../src/operators/advection/staggered_schedules.jl)).
   Each schedule also constructs dual-flux connectivity and a coefficient
   dictionary—work not needed to compute a diagnostic weight. Thus four full
   schedule constructions can occur per step, on an unchanged mesh. The
   global-face conversion additionally constructs axis index maps and scans
   faces. This is the strongest candidate for both models' diagnostic time.
   Face dual volumes depend on topology (and fixed solid geometry), not on
   pressure or moving fluid fractions, so they should not be rebuilt every
   step. Verify the embedded-geometry invalidation rules before caching.
2. **One-velocity face density is recomputed twice.** Energy and momentum each
   call `mixture_density`, allocating a face vector and traversing the
   interface. Fractions change after transport, so the values need refreshing
   *per completed state*, but they can be shared between these diagnostics
   within that state. This is likely smaller than schedule construction but
   should be timed separately.
3. **Two-velocity phase fractions are recomputed at least three times.** Energy
   and momentum each call `_staggered_liquid_fraction` for every face.
   `DivergenceNorm` calls `_mass_weighted_one_velocity!`, which repeats that
   fraction calculation to build a mixture velocity. Each calculation is
   individually cheap without embedded solids but traverses all faces, and
   the three traversals occur at the same completed interface state. With
   cut-cell geometry, the interpolation is more expensive. Measure the phase
   fraction work separately from the extra velocity arithmetic.
4. **Divergence scratch is rebuilt every evaluation.** `DivergenceNorm`
   allocates a new `CellField` on either model. On two velocities it also
   allocates a `FaceVelocity` and a density vector, then fills the mixture
   velocity before taking divergence. The density vector is an output of the
   general `_mass_weighted_one_velocity!` helper but is not used by this
   diagnostic afterward. Reusable scratch or a diagnostics-specific overload
   could remove these allocations without changing the reduction.
5. **Scheduling/history overhead is probably secondary.**
   [`history.jl`](../../src/simulation/diagnostics/history.jl) checks each
   schedule and appends a scalar/small vector once per event. Its collection
   stores abstract series, so dynamic dispatch exists, but there are only
   five diagnostics per step. Measure it before attempting a structural
   rewrite. `PhaseVolume` and `LeafBlockCount` should also be timed rather
   than assumed negligible.

## Measurement sequence

1. Add a focused development benchmark that uses the existing Stokes3
   initialization, warms compilation, and reports exact `@timed`/`@allocated`
   totals for each of the five `diagnostic_value` calls on a frozen completed
   state. Use an initial fine mesh and a state after at least one AMR topology
   change. Evaluate each diagnostic repeatedly without changing the state,
   reporting median/p95 wall time and bytes per call. Record blocks, faces,
   cells, density model, and the diagnostic value itself.
2. Instrument complete short `t=0.05` runs with default diagnostics and with
   diagnostics disabled, for both velocity models. Count initial, step and
   completed evaluations. This measures the actual impact on simulation wall
   time and separates diagnostic code from unrelated mesh/pressure costs.
   Keep default diagnostics in the final performance reference.
3. Inside energy/momentum/divergence, time `_uncut_face_dual_volumes`, the two
   `_amr_dual_schedule` builds, global-face mapping, mixture density or
   staggered phase-fraction traversal, mixture-velocity fill, divergence, and
   history append. Pair time measurements with sampled allocation stacks.
   Confirm whether repeated schedule construction is indeed the leading
   cause of the `face_volumes` bucket.
4. Compare on genuinely matched mesh/interface snapshots, not only the two
   evolving runs. Report absolute time per call and time/bytes per face. The
   short runs happen to end with equal block/face counts, but identical counts
   do not prove identical topology. This comparison will quantify the true
   model-dependent overhead after removing the common topology work.
5. Trial one change at a time: first obtain/cache the face dual volumes once
   per topology (preferably without constructing the unused full dual-flux
   schedule), then share face fractions/density within a diagnostic event,
   then reuse divergence scratch. Re-run the frozen-state and whole-run
   measurements after each change. Keep direct public `diagnostic_value`
   correct even when called outside a simulation with no cache.

## Validation and acceptance gates

- Every optimized diagnostic must return the same value as the current
  implementation on fixed uniform, mixed AMR, and embedded-solid meshes,
  including immediately after refinement/coarsening. Histories must retain
  the same times, steps, lengths and values. Compare at tight floating-point
  tolerances; preserve the same reduction order where practical.
- Any topology-only cache must be invalidated at every AMR rebuild and when
  solid geometry changes. Per-state fraction/density scratch must be
  refreshed after fluid transport and never reused across distinct states.
- Run [`test/diagnostics.jl`](../../test/diagnostics.jl),
  [`test/examples/stokes3_example.jl`](../../test/examples/stokes3_example.jl),
  relevant AMR/geometry tests, and eventually the full suite. Both default
  Stokes3 models must still reach `t=0.5` with unchanged step/AMR counts and
  conservation histories.
- Accept a change only when the default-diagnostics full run, not just an
  isolated microbenchmark, becomes faster and allocations fall without moving
  equivalent work into another stage. Report the new one- versus two-velocity
  *same-state* ratio as well as the evolving-run totals.
