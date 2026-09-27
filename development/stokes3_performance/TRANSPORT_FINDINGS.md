# Focused Stokes3 transport analysis and optimization

The initial measurements below motivated production changes to both velocity
models. The reproducible short AMR benchmark and exploratory
no-solid-geometry kernel are in
[`transport_analysis.jl`](transport_analysis.jl). The post-change results are
reported at the end. Timings are warm single runs, not statistical estimates.

## Comparable short runs with AMR

Both runs used the default Stokes3 configuration, `max_level=5`, no
visualization, and `final_time=0.05`. Compilation was warmed by a separate
short run; time and allocation profiling then used separate fresh flows. Julia
1.12.7 used one Julia thread and 16 BLAS threads. Both modes made four AMR
topology changes and ended at 169 blocks / 21,832 faces, although the
two-velocity path took one extra time step. Times below are **sampled
estimates**, and nested substage rows overlap. Allocations are estimates from
0.5% allocation sampling; whole-run byte totals come from `@timed`.

| Measure to `t=0.05` | One velocity | CMDGFM two velocities |
| --- | ---: | ---: |
| Steps / AMR changes | 21 / 4 | 22 / 4 |
| Whole advance | 5.86 s | 16.63 s |
| Whole advance allocations | 8.43 GB | 29.90 GB |
| Transport stage | 1.78 s | 6.71 s |
| Transport allocations | ~1.61 GB | ~11.16 GB |
| Outer time-step bound | negligible sampled time | 0.98 s |
| Outer time-step allocations | negligible | ~1.67 GB |

The comparable final mesh and near-equal step counts make this a much cleaner
comparison than the earlier `t=0.5` runs. Per step, transport is approximately
0.085 s versus 0.305 s; including the outer CFL work makes the two-velocity
figure approximately 0.349 s/step. The difference is not just AMR history.

The nested two-velocity time samples show about 5.61 s in `_advect_wy!`,
including 3.33 s in phase-face momentum transport. Reconstruction accounts
for about 1.34 s of the transport samples. Aperture freezing accounts for
about 1.04 s, and the inner CFL check for about 1.25 s; the last two overlap
with other nested rows. The largest transport allocation buckets were:

| Two-velocity transport allocation source | Estimated allocation |
| --- | ---: |
| Phase-face momentum loops | 4.81 GB |
| Aperture freezes within transport | 4.22 GB |
| Directional WY flux construction | 1.76 GB |
| Reconstruction within transport | 0.32 GB |

The outer CFL query adds another ~1.67 GB, mostly through its own aperture
preparation. One-velocity transport allocates ~1.45 GB in its WY flux
construction, but has no corresponding two-phase aperture-freeze or intrinsic
phase-face momentum costs. The sampled allocation totals are close to the
separately measured whole-run byte totals, but these are still profiles of one
warm run, not confidence intervals.

## Repeated work in the current path

For the default `FrozenInterfaceFluxes()` path, source inspection gives this
call sequence for one automatically selected physical step:

| Location | Reconstruction | Aperture freeze |
| --- | ---: | ---: |
| Outer `simulate!` CFL query | 1 | 1 |
| `_transport_substep!` stability guard | 1 | 1 |
| `advect!` entry | 1 | 1 |
| `_advect_wy!` stability guard | 1 | 1 |
| `_advect_wy!` initial geometry | 1 | 0 |
| After first and second directional sweeps | 2 | 0 |
| **Total** | **7** | **4** |

The two sweeps need a reconstruction of their current fraction state; the
final reconstruction is also needed by subsequent physics. That calls for
three reconstructions per step: initial state, after sweep 1, and after sweep
2. The apertures represent the *time-step-start* interface and should be
frozen once before either sweep. Thus four reconstructions and three aperture
freezes in the table are repeated preparation. The four pre-sweep
reconstructions all see the same fraction state. On the initial Stokes3 state,
two successive calls produced identical normal, intercept, curvature and
reconstruction fields, including NaNs in curvature where appropriate. This
idempotence must also be checked after AMR and during deformed-interface
states before removing calls.

A same-state microbenchmark on the initial 214-block mesh found:

| One geometry call, median of six | Wall time | Allocated |
| --- | ---: | ---: |
| `reconstruct!` | 9.19 ms | 2.82 MB |
| Current aperture freeze | 12.91 ms | 74.56 MB |
| Development-only no-solid function barrier | 2.42 ms | 0 B |

The function-barrier result exactly reproduced both aperture arrays. It is
exploratory: its loop is serial and is only valid without embedded solids.
The production kernel needs the same specialization for both solid and
no-solid geometry and must retain its parallel behavior. Nonetheless, it
shows that the current per-face allocation is avoidable, not intrinsic to
calculating phase apertures.

## Why the two-velocity core is also slower

Even if all repeated reconstruction/aperture work vanished, the current
two-velocity core would remain substantially more expensive. Two-phase
momentum is transported on both face orientations for both phases, rather
than transporting one face velocity. That is real extra work, but the
~4.81 GB sampled allocation in its inner loops is not obviously necessary:
[`PhaseFaceScalar.workspace`](../../src/two_phase/phase_face_fields.jl) is
stored as `StaggeredAdvectionWorkspace{Axis,T,M}` without its concrete
schedule type parameter. The hottest sampled allocation sites are the
per-dual-face schedule access and `_phase_transport` calls in
[`phase_face_transport.jl`](../../src/two_phase/phase_face_transport.jl).
This is a strong type-instability hypothesis, not yet a proven full-run
speedup. The interface and WY workspace also store geometry as `Any`; the
aperture function-barrier experiment directly demonstrates the cost of
repeatedly entering a face kernel with that erased type.

The frozen-subface flux path in
[`interface_transport.jl`](../../src/two_phase/two_velocity/interface_transport.jl)
creates/clips polygons even when there is no embedded solid. The one-velocity
path has an analytic `_plane_fraction` fast path and returns immediately for
pure-phase donor cells. A no-solid two-velocity path can use the analytic
rectangle/PLIC formula on its *phase subface interval* without changing the
conservative flux definition; equivalence must be tested at mixed and AMR
faces, especially near pure fractions and reversed flow.

## Initial proposed change order

1. **Prepare geometry once per physical step.** Carry an internal,
   step-local prepared-state token from the first CFL evaluation through the
   transport guard and `advect!`. For a prescribed `dt`, prepare during the
   guard instead. The token must be invalidated by interface, velocity or
   mesh changes. Keep standalone `advect!` and `stable_timestep` independently
   safe. Preserve the initial/between-sweep/final reconstruction sequence
   (three calls) and the single time-step-start aperture freeze. Avoid
   recomputing a weaker CFL guard if a stronger bound on the same state was
   already checked. Verify that reconstruction is idempotent on the relevant
   states before expecting bitwise agreement.
2. **Make aperture freezing type-stable.** Put the face loop behind a
   geometry-specialized function barrier, with separate `Nothing` and
   embedded-geometry paths. The development-only `Nothing` result gives a
   concrete performance and exact-output target. This can be done without
   changing the public interface representation.
3. **Make phase-face momentum schedule concrete.** Add the schedule/workspace
   type parameter to `PhaseFaceScalar` (or use an equally effective function
   barrier), then remeasure allocations at the reported hot sites. Do not
   fuse liquid/gas updates until type instability is removed and the
   remaining work is measured.
4. **Add the no-solid analytic flux fast path.** Replace polygon allocation
   with `_plane_fraction` on the exact frozen subface rectangle, plus a
   pure-donor shortcut where its result is unambiguous. Retain the polygon
   path for embedded solids. Compare phase volumes and individual fluxes,
   not only final Stokes3 error.
5. Re-profile after each change. If a substantial gap remains, study whether
   phase-face fractions, predicted fractions and divergence rates can be
   shared between the two velocity components within a sweep. Some of their
   present recomputation is mathematical work needed after the first sweep;
   only same-state duplicates should be removed.

At the original baseline, the 10%-over-one-velocity aim was a stretch target,
not a measured prediction. At those per-step costs it meant reducing two-velocity
transport from ~0.305 s to at most ~0.093 s on this interval. Removing the
four extra reconstructions and three extra freezes, even with the fast
aperture kernel, would save only on the order of 0.09 s/step on the initial
mesh. The large phase-face momentum cost and remaining flux work must also
fall substantially. No production change should be declared successful
from allocation reduction alone; wall time, numerical fluxes and full-run
behavior must all improve.

Validation gates: compare phase fractions, both phase velocities, face fluxes
and frozen apertures after one step on unchanged and freshly adapted meshes;
exercise independent `advect!` with an unsafe step; run
[`test/two_phase/two_velocity/transport/two_velocity.jl`](../../test/two_phase/two_velocity/transport/two_velocity.jl),
[`test/integration/two_velocity_transport.jl`](../../test/integration/two_velocity_transport.jl)
and the Stokes3 example tests; then repeat this four-AMR-update `t=0.05`
benchmark and the default `t=0.5` run. Require no worsened mass/momentum
diagnostics or solver failures, and report median warm times and allocations
over repeated runs.

## Changes implemented and conclusions

The integrated frozen-interface step now reuses geometry prepared by its CFL
query. A direct/fixed-step call prepares geometry in its own guard, and public
stand-alone `advect!` still prepares and checks independently. The WY path
keeps the three necessary reconstructions (initial state, after each sweep)
and one time-step-start aperture freeze. AMR rebuilds the workspace, so a
prepared state is never carried across a topology change. Aperture freezing,
the CFL scan, and frozen-subface flux construction dispatch on the geometry
type *outside* their face/cell loops. The no-solid frozen-subface flux uses
the analytic PLIC rectangle fraction; embedded-solid clipping is retained.
`PhaseFaceScalar` now retains its concrete advection schedule type.

After seeing the two-velocity speedup, the same geometry-specialized flux
sweep was applied to one velocity. Its no-solid flux was already analytic,
but the geometry type had been erased inside the face loop. This removed the
one-velocity flux allocation hotspot without changing its formula.

| `t=0.05` (four AMR changes in each mode) | One velocity before | One velocity after | CMDGFM before | CMDGFM after |
| --- | ---: | ---: | ---: | ---: |
| Advance wall time | 5.86 s | 5.42 s | 16.63 s | 9.38 s |
| Sampled transport time | 1.78 s | 0.76 s | 6.71 s | 0.61 s |
| Sampled transport allocation | ~1.61 GB | ~0.16 GB | ~11.16 GB | ~0.11 GB |
| Sampled outer-CFL time | ~0.00 s | ~0.00 s | 0.98 s | 0.47 s |
| Sampled outer-CFL allocation | negligible | negligible | ~1.67 GB | ~0.16 GB |
| Whole-run allocation | 8.43 GB | 6.95 GB | 29.90 GB | 16.93 GB |

The final short runs had 21 one-velocity and 22 CMDGFM steps, as before, and
both ended with 169 blocks / 21,832 faces. A single production aperture
freeze on the initial no-solid mesh now takes a median **1.43 ms and 0 B**, down
from 12.91 ms and 74.56 MB. The former phase-face momentum and frozen-subface
allocation hotspots no longer appear in the post-change transport profile;
remaining sampled transport allocation is mostly interface reconstruction.

The larger benchmark uses the default Stokes3 physics/AMR and `t=0.5`.
Different late-time mesh histories mean these are whole-run stage comparisons,
not matched-state kernel ratios. Steps and AMR changes remained unchanged
within each model.

| `t=0.5` | One velocity before | One velocity after | CMDGFM before | CMDGFM after |
| --- | ---: | ---: | ---: | ---: |
| Advance wall time | 39.90 s | 36.62 s | 143.19 s | 83.50 s |
| Sampled transport time | 8.44 s | 5.12 s | 57.56 s | 5.42 s |
| Sampled outer-CFL time | ~0.03 s | ~0.03 s | 8.33 s | 4.32 s |
| Whole-run allocation | 56.10 GB | 46.27 GB | 255.04 GB | 141.49 GB |
| Steps / AMR changes | 206 / 45 | 206 / 45 | 219 / 32 | 219 / 32 |

**Conclusion:** in the full reference run, CMDGFM's transport stage is now
about **5.8% more expensive** than optimized one-velocity transport, meeting
the requested within-10% goal. One-velocity transport is itself about 39%
faster than before. The outer two-velocity CFL path remains a separate
~4.32 s cost, so the combined transport-plus-CFL ratio is less favorable.
The overall CMDGFM advance is still slower because pressure, diagnostics and
other stages remain; transport is no longer the dominant reason.

Targeted two-velocity WY/reference tests and one-velocity transport plus
embedded-geometry tests passed. The complete suite finished with **13,509
passes, two failures, and one broken test**. Both failures were reproduced
in isolation on the pre-optimization revision (`HEAD^` of the transport
change): `mimetic_projection_variants.jl` reports a pressure-commuting
residual of 0.9 against a `< 0.2` assertion, and `cylinder_example.jl`
reports a non-finite pressure-drag history. Thus the full suite is not green,
but these two failures predate the transport changes. All transport suites
and the Stokes3 example tests passed in the full run. These timings are
single warm runs, so smaller differences should not be overinterpreted
without repetitions.
