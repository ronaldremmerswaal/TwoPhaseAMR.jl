# Stokes3: AMR connectivity reuse

The proposed optimization is now implemented. Pressure discretization, ILU,
tolerances, phase-extension formulas and refinement decisions are unchanged.
The physical inactive phase velocities remain NaN; only transfer/curl scratch
is extended. The investigation and its original measurements are retained
below, explicitly as pre-optimization evidence.

## Implementation and full-length benchmark

- `BlockTreeMesh` owns a lazy topology cache: compressed cell neighbours and
  centers, and directional phase-transport schedules/adjacency/mappings.
  The generation changes after face/halo schedules are rebuilt, not when the
  tree is first edited: old-mesh transfer still needs the old connectivity.
  Same-size topology changes invalidate it too.
- Select the actual periodic/bounded/solid schedule before construction.
  Uncut phase transfer, intrinsic momentum and shared momentum borrow the
  same read-only directional graph for one mesh epoch. Geometry-dependent
  schedules remain independently constructed; they are not cached under an
  insufficient mesh-only key. Real periodic advection stencils are retained.
- Flux vectors, boundary-condition references, donor masks and other evolving
  numerical data are not shared. `AMRWorkspace` retains packing/extension and
  vorticity fit scratch. Fractions and donors are recomputed on every call.
- Phase-aware curl and block-label evaluation dispatch once on the concrete
  geometry instead of through `interface.geometry::Any` in each face/cell.
  They reuse mesh connectivity and numerical scratch. Allocating public
  visualization calls still return independently owned output.
- The shared-velocity momentum workspace is retained for existing typed API
  and reconfiguration compatibility; its graph now aliases the intrinsic
  graph instead of being reconstructed. Removing that entire workspace,
  and changing pressure/diffusion topology construction, are not part of this
  optimization.

As requested, the default `t=0.5` benchmarks were run before the remaining
validation. Three unprofiled production repetitions per model, isolated
Julia 1.11.9 / normal optimization / BLAS 1 / Julia thread 1, warmup and initial
setup outside the advance:

| Complete production advance | One: before | One: now | CMDGFM: before | CMDGFM: now |
| --- | ---: | ---: | ---: | ---: |
| Wall time, median | 15.759 s | 15.728 s | 42.612 s | 25.621 s |
| Wall time, range | 15.637–16.091 s | 15.640–16.144 s | 42.491–42.680 s | 25.503–25.650 s |
| Allocation, median | 12.717 GB | 12.532 GB | 61.749 GB | 26.146 GB |
| Steps / AMR changes / final faces | 206 / 45 / 12,504 | 206 / 45 / 12,504 | 219 / 79 / 9,016 | 219 / 79 / 9,016 |

CMDGFM is 1.66× faster, allocating 57.7% fewer bytes. The separate 4 ms
profiles put adaptation at **17.294 → 2.254 s** and topology rebuild at
**14.488 → 9.262 s**; their sum drops 63.8%, from 3.32× to 1.26× the
one-velocity subtotal. Full before/after component tables are in
[`PLAN.md`](PLAN.md). Numerical equivalence and suite status must be checked
separately; matching mesh/step counts alone are insufficient.

Raw log: `/tmp/stokes3_amr_optimized_full_benchmark.log`.
[`amr_validation.jl`](amr_validation.jl) reproduces the unprofiled runs and
captures/compares plain field arrays, NaN masks, topology and all default
diagnostic histories using independent before/after source environments.

## Pre-optimization investigation

### Conclusion

**Part of the increase is a timing-bucket shift.** Before pressure reuse,
AMR discarded the intrinsic-flow workspace and the next timestep constructed
it again. Now AMR retains the pressure workspace and reconfigures intrinsic
transport immediately. Approximately 5.7 s of existing setup work therefore
appears in topology rebuilding instead of the next step's overhead. This
should have been made explicit in the earlier comparison.

That does not account for the entire remaining gap. Its main sources are:

1. The corrected two-velocity run changes topology more often: 79 changes,
   versus 45 for one velocity. The older two-velocity run had only 32 changes,
   but incorrectly used dormant liquid velocities in absent-liquid regions
   when deciding whether to coarsen. That is not a valid performance baseline
   to recover by undoing NaN handling.
2. NaN-safe phase-velocity transfer repeatedly constructs full advection
   schedules. On Stokes3's bounded mesh, it then discards the general schedule
   and replaces it with a bounded schedule. Intrinsic transport reconfiguration
   repeats this pattern on the new mesh. The general constructor computes
   three-point advection stencils that neither operation uses.

The affine phase extension itself is cheap. **Cache/reuse connectivity; do not
remove the extension or replace physical NaNs with invented velocities.**

The expensive pressure assembly itself has not been relocated into AMR, and
the matched whole advance is faster. The AMR transfer, phase-activity,
field-transfer and vorticity source files are unchanged between pre-pressure
commit `bdc9ac2` and pressure-reuse commit `9262848`. The like-for-like
pre-pressure profile below identifies the workspace-reconfiguration shift
separately.

## Complete-production profile

Default Stokes3, `final_time=0.5`, `max_level=5`, default diagnostics, no
visualization. Isolated Julia 1.11.9, normal optimization, package images
disabled, one Julia/BLAS thread. Warmup to `t=0.05` and initial setup/projection
are outside the advance timer. No production methods are replaced.

The following entries are **disjoint 2 ms sampling estimates** normalized to
the measured wall time, with GC included. They come from a new run and should
not be mixed with the earlier 4 ms component table in `PLAN.md`. These are not
independent stopwatch measurements. Small entries have substantial sampling
uncertainty.

| AMR work over the entire `t=0.5` advance | One velocity (s) | CMDGFM (s) |
| --- | ---: | ---: |
| Phase-transfer schedule construction, before regrid | — | 11.190 |
| Phase-transfer affine/constant extension | — | 0.118 |
| Other phase-transfer preparation | — | 0.077 |
| Vorticity: face-to-cell interpolation | 0.047 | 3.433 |
| Vorticity: stencil extension | not resolved | 0.072 |
| Vorticity: remaining curl calculation | 0.062 | 0.044 |
| Remaining indicator work | 0.157 | 1.019 |
| Mesh face/halo schedule rebuild | 0.542 | 0.651 |
| Finish interface prolongation/reconstruction | 0.683 | 0.598 |
| Face-field transfer | 0.114 | 0.254 |
| Other adaptation/transfer work | 0.101 | 0.136 |
| **Adaptation subtotal** | **1.706** | **17.591** |
| Intrinsic transport reconfiguration | — | 5.724 |
| Shared-velocity transport reconfiguration | 0.978 | 1.054 |
| Diffusion-operator construction | 3.380 | 4.184 |
| Diffusion-solver/preconditioner reconfiguration | 1.049 | 1.240 |
| Scalar pressure-operator construction | 1.306 | 1.639 |
| Scalar pressure-multigrid reconfiguration | 1.205 | 1.590 |
| Other solver/workspace rebuild | 0.047 | 0.072 |
| **Topology-dependent rebuild subtotal** | **7.965** | **15.503** |
| **AMR + topology total** | **9.671** | **33.094** |

The combined factor is 3.42 in this profile; the previous profile gave 3.32.
Topology rebuild alone is 1.95 times as expensive in total, but only 1.11 times
as expensive per actual topology change (177 versus 196 ms). The earlier
4 ms profile gave 176 versus 183 ms. Adaptation includes indicator evaluation
on unchanged attempts, so dividing that entire bucket by the number of
*changed* meshes would not be a proper per-call comparison.

| Run metadata | One velocity | CMDGFM |
| --- | ---: | ---: |
| Whole profiled advance (s) | 17.337 | 47.322 |
| Allocated bytes (decimal GB) | 12.719 | 61.749 |
| GC time, included in wall time (s) | 2.126 | 6.695 |
| Steps / changed meshes | 206 / 45 | 219 / 79 |
| Final blocks / faces | 97 / 12,504 | 70 / 9,016 |
| Classified / total samples | 8,069 / 8,084 | 21,602 / 21,647 |

Ordinary pressure refresh/projection is 1.852 versus 2.171 s in this profile.
Pressure-related topology construction is explicitly listed separately above;
it has not been hidden in the ordinary pressure timing. Its per-change cost
is actually lower for the CMDGFM mesh: approximately 41 versus 56 ms.

## Did pressure reuse move work into AMR?

The archived `bdc9ac2` source and current source were run sequentially with the
same isolated Julia 1.11.9 configuration, defaults and 2 ms profiling. Both
give 219 steps, 79 mesh changes and 9,016 final faces. This is a fresh paired
measurement, not a comparison with the invalid pre-NaN mesh history or a
different Julia version.

| Complete CMDGFM advance (s) | Before pressure reuse | Current |
| --- | ---: | ---: |
| Adaptation / transfer | 17.743 | 17.591 |
| Topology-dependent rebuild | 8.976 | 15.503 |
| Other step overhead | 10.366 | 4.951 |
| Ordinary pressure refresh + projection | 12.538 | 2.171 |
| **Whole profiled wall time** | **56.815** | **47.322** |
| Whole allocation (decimal GB) | 85.372 | 61.749 |

In the old `TwoPhaseFlowWorkspace` constructor, `two_velocity` was reset to
`Ref{Any}(nothing)` after AMR. `_two_velocity_flow_workspace!` then recreated
`PhaseFaceScalar` transport and projection storage during the following step.
The current constructor calls `_reconfigure_two_velocity_flow_workspace`
inside the AMR rebuild instead. Its intrinsic transport portion costs 5.724 s
in the current profile, while other-step overhead falls by 5.415 s. Those are
the corresponding setup costs, not evidence of a new physical computation.
The sampling table does not separately resolve every old lazy-construction
suboperation, so this is not an exact one-to-one stopwatch accounting.

The correct interpretation is therefore: **a real 9.493 s whole-run reduction,
plus a substantial relabeling of existing transport setup**. Ordinary pressure
work falls by about 10.367 s. The expensive phase-transfer schedule construction
is essentially unchanged: 11.200 s before versus 11.190 s after. Fixing it is
an additional optimization opportunity, not undoing a pressure regression.

## Which work is repeated?

`_prepare_adaptive_phase_velocities!` must prepare finite **transfer scratch**
from valid phase donors before prolongation. Its
`_extend_phase_velocity_for_transfer!` helper currently does this for each of
two phases and each of two coordinate components:

1. Allocate a directional field and a general `StaggeredAdvectionWorkspace`.
2. Build a full general dual schedule, including three-point stencils.
3. Because the domain is bounded, discard that schedule and construct a
   bounded schedule and another advection workspace.
4. Extend the phase values using only the bounded dual-face adjacency.

That is **eight schedule/workspace constructions per changed mesh on the old
mesh**, although the two phases have identical connectivity. The actual
extension does not use advection flux weights, three-point stencils, tendency,
rates or boundary-condition scratch.

After regridding, `_finish_phase_face_transfer!` does another general-then-
bounded construction for each component: **four more constructions on the
new mesh**. `TwoPhaseFlowWorkspace` also rebuilds the shared-velocity momentum
workspace, including **two bounded constructions**, even when intrinsic
two-velocity transport is selected. Thus this path invokes 14 schedule builds
per changed mesh, versus two for the one-velocity transport rebuild. The old
and new meshes are different: one cannot share a graph *across* that change
without refreshing it.

Across the CMDGFM AMR buckets, `_three_point_stencils` alone accounts for
approximately 10.13 s (an inclusive, overlapping profile row). Most of this
work is used to construct schedules that are immediately discarded. It is not
required by the pressure method or the phase extension.

Vorticity has a different problem. It calls the allocating visualization
routine on every adaptation attempt. The phase path repeatedly reads
`interface.geometry::Any` in face/cell activity checks, rather than dispatching
once to a geometry-specialized loop. It also rebuilds mesh centers and a
vector-of-vectors cell adjacency, then allocates the centered velocities and
least-squares accumulators. The profile attributes much more time to phase
activity/interpolation than to the actual one-cell affine extension. Keep the
current activity checks and donor rules while removing repeated dispatch and
mesh-only construction.

## Frozen-state experiment: schedule reuse without changing the extension

On the final CMDGFM state (`t=0.5`, 9,016 faces), the development experiment
copies each phase velocity, then uses the already-existing directional
transport schedules **for that same mesh**. It calls the identical production
extension routine, with identical original donors, ring limits, affine fit,
rank-deficient fallback and boundary treatment. It does not write the physical
velocities or change the interface.

Five warmed repetitions, resetting scratch outside each timed region:

| Preparing both phase velocities for transfer | Median time | Allocated bytes per call |
| --- | ---: | ---: |
| Current production helper | 92.584 ms | 118,745,008 B |
| Development cached-schedule helper | 1.129 ms | 1,593,664 B |

The resulting interior and boundary values agree **exactly** (`Linf=0` for
both). This is an 82-fold kernel speedup and 98.66% allocation reduction, not
an 82-fold whole-simulation speedup. This is a frozen-state experiment, not a
full AMR implementation or a complete accuracy gate. The
remaining allocations are directional packing, masks, original values,
centers and candidate scratch, which can also be retained.

The short run also validates schedule reuse on two larger, earlier states.
All interior/boundary errors are again exactly zero:

| Frozen state | Faces | Current transfer, median | Cached schedule, median | Current / cached allocation |
| --- | ---: | ---: | ---: | ---: |
| `t=0.05` | 21,832 | 231.201 ms | 2.513 ms | 294.439 / 4.099 MB |
| `t=0.08` | 22,232 | 237.662 ms | 2.535 ms | 297.569 / 4.122 MB |
| `t=0.50` | 9,016 | 92.584 ms | 1.129 ms | 118.745 / 1.594 MB |

The `t=0.08` run still has the expected 35 steps and seven topology changes.
It is an allocation-profiled **production advance**, not an advance using the
development cached helper. The latter is tested on independent scratch only.

### Phase-activity checks: an independent type-specialization experiment

The development script also scans liquid activity over all interior faces and
cells. The reference uses the current convenience overload, which reads
`interface.geometry::Any` for each entry. The alternative dispatches once on
`geometry`, then calls the existing, identical geometry-explicit overload in
both loops. No activity formula, threshold or phase value changes.

| Frozen state | Current activity scan, median | Geometry-specialized scan, median | Current / specialized allocation |
| --- | ---: | ---: | ---: |
| `t=0.05` | 13.695 ms | 0.153 ms | 101,318,800 / 3,040 B |
| `t=0.08` | 11.868 ms | 0.146 ms | 103,160,048 / 3,040 B |

The active face/cell counts are identical: 9,340/4,527 and 11,093/5,403,
respectively. This directly supports a geometry-specialized vorticity loop.
These are timings of activity checks, **not of the complete curl or AMR
criterion**. This experiment covers the uncut Stokes3 mesh; solid cases remain
an implementation validation gate.

### Allocation attribution on the short production run

At `t=0.08`, sampled allocation profiling reports 15,236 allocation samples
with probability 0.0002. Exact whole-advance allocation is 13.292 GB. The main
sampled AMR contributors are:

| AMR region | Sampled allocation events | Estimated allocated bytes (GB) |
| --- | ---: | ---: |
| Vorticity face-to-cell interpolation | 678 | 3.948 |
| Remaining indicator work | 191 | 1.392 |
| Old-mesh phase-transfer schedule construction | 4,493 | 0.750 |
| New-mesh intrinsic transport reconfiguration | 2,103 | 0.356 |
| Vorticity stencil extension | 163 | 0.045 |

Allocation sampling is uniform by event, not byte size. At this low probability
rare large arrays are often missed, and stage-byte estimates do not sum to the
exact total. In particular, do not replace the exact 294–298 MB per frozen
transfer call with the smaller sampled schedule estimate. The sampled event
counts identify substantial small-object churn in schedule construction,
while activity interpolation is a major byte contributor. Both the exact
frozen measurements and the complete production profiles are needed.

## Proposed implementation, in order

1. **Never build the schedule that will be discarded.** Choose periodic,
   bounded or geometry-aware construction before calling the constructor.
   Apply this to transfer preparation and `_finish_phase_face_transfer!`.
   This is the smallest, independently verifiable improvement.
2. **Store reusable mesh connectivity centrally.** Extend the existing
   mesh-only cache beyond face-dual volumes: component face mappings/centers,
   compressed dual adjacency and cell adjacency. Build directional transport
   connectivity once per topology and share it among phase transfer and
   transport workspaces. Keep evolving phase fractions, donor masks, fluxes
   and numerical work arrays separate. Solid-geometry-dependent graphs require
   their own validity key, not just a mesh key.
3. **Retain transfer scratch.** Put packed values, original donor values,
   masks, next-layer values and candidate scratch in an AMR-owned workspace.
   Reuse the old topology's connectivity until transfer preparation completes;
   refresh the mesh cache after rebuilding mesh schedules. Use an explicit
   topology generation: equal face counts do not imply equal connectivity.
4. **Avoid duplicate intrinsic/shared transport graph construction.** Reuse
   the same new-mesh graph with independent numerical scratch. Longer term,
   do not construct a shared-velocity momentum workspace when it is unused,
   provided this does not compromise supported model/workspace reconfiguration.
5. **Make phase-vorticity labels in-place and type-stable.** Dispatch once on
   geometry and phase, cache mesh centers/adjacency, retain the curl fit arrays,
   and compute block maxima without allocating visualization output. Preserve
   the existing NaN/absent-phase and unresolved-stencil decisions exactly.
6. Only then revisit diffusion/pressure topology templates: they are real
   costs in *both* models. Do not change ILU or weaken the projection merely
   because they appear inside the AMR timer.

The first priorities address approximately 16.9 s in the two schedule-heavy
CMDGFM buckets, with a further 3.4 s in vorticity interpolation. These are the
measured opportunity, not a promised subtraction from whole-run time. New
topologies still require connectivity construction and associated solver
reconfiguration, and GC attribution will change when allocations fall.

### Validation gates for a later implementation

- Compare the transfer scratch including boundaries and NaN masks against the
  current reference on early/late Stokes3 states, periodic/bounded/solid
  domains and hanging-face AMR meshes. Test affine exactness and the existing
  divergence-preserving prolongation.
- Compare vorticity values and refinement labels, including absent-phase
  blocks and rank-deficient stencils. No inactive physical phase velocity may
  become finite or serve as a donor.
- Exercise refine/coarsen cycles and same-size topology changes; no cached
  graph, fraction or mask may refer to the wrong mesh epoch. Check independent
  numerical scratch when graph storage is shared.
- Run a short benchmark with topology changes, then default Stokes3 to `t=0.5`.
  Require the current histories (206/45/12,504 and 219/79/9,016), comparable
  fields, unchanged pressure convergence, and conservation diagnostics. Record
  allocations, GC and event-normalized costs as well as whole wall time.
- Run the relevant phase-velocity, AMR, reconstruction, transport, diffusion
  and projection checks, followed by the full suite. Compare known baseline
  failures rather than weakening assertions.

## Reproduction

```sh
julia --project=. development/stokes3_performance/amr_analysis.jl one 0.5
julia --project=. development/stokes3_performance/amr_analysis.jl cmdgfm 0.5
julia --project=. development/stokes3_performance/amr_analysis.jl cmdgfm 0.08 allocs
```

For these measurements use the same isolated-runtime/depot invocation as in
`PLAN.md`; the installed Julia 1.12.7 has known runtime crashes in this session.
The allocation profiler samples uniformly by allocation, not by bytes. Its
per-stage byte estimates are approximate, especially for rare large arrays;
the frozen-state `@timed` byte totals above are exact for those calls.

Raw logs: `/tmp/stokes3_amr_current.log` and
`/tmp/stokes3_amr_pre_pressure.log`; early-state and sampled-allocation results
are in `/tmp/stokes3_amr_allocations.log`.
