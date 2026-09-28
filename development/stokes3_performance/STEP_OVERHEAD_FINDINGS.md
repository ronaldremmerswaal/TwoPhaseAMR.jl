# Stokes3: remaining step overhead

This work targets the residual `step!`/`simulate!` row in `PLAN.md`, after AMR
connectivity reuse. The pressure equations, K, ILU, solver tolerances, transport,
diffusion, refinement decisions and inactive-phase storage contract are unchanged.
The external-force callback API remains `force(x,y,time,axis)`.

## Analysis before implementation

The residual category is not necessarily bookkeeping: it owns samples inside
`step!`/`simulate!` not already charged to a named stage. The read-only
[`step_overhead_analysis.jl`](step_overhead_analysis.jl) distinguishes the
original inline predictor loop by source location and the remaining work by
its actual call stack. Named helpers identify the optimized predictor path.

Default CMDGFM Stokes3 to `t=0.5`, Julia 1.11.9 / normal optimization / no
package images / one Julia and BLAS thread / 2 ms sampling:

| Residual work | Samples | Sampled time |
| --- | ---: | ---: |
| External-force predictor loop | 3,449 | 7.542 s |
| Physical phase masks | 60 | 0.131 s |
| Projection wrapper outside `project!` | 5 | 0.011 s |
| Other step controls | 2 | 0.004 s |
| Other simulation controls | 2 | 0.004 s |
| Boundary enforcement, excluding phase masks | 1 | 0.002 s |
| Velocity copies / intrinsic-workspace lookup | not resolved | not resolved |

The external-force predictor owns **98.0% of the residual samples**. The
profile includes real transport, diffusion, pressure, AMR and default
diagnostics; it is not a substitute force-free timestep wrapper. Its complete
wall time was 33.888 s / 26.148 GB, with 219 steps, 79 AMR changes and 9,016
final faces. Another user Julia job overlapped part of this investigation, so
its wall time must not replace the exclusive unprofiled medians in `PLAN.md`.
Small entries are noisy; absence of a classified sample does not mean zero
cost. The listed sampled seconds are disjoint estimates, not exact timers.

The sampled short `t=0.08` allocation run gives 7,348 predictor allocation
samples, estimating 2.552 GB there against an exact 5.270 GB whole advance.
Allocation sampling is uniform by event, not bytes; that is an estimate of
stage allocation, not an exact subtraction. It independently identifies the
same dominant problem.

### Cause

`TwoPhaseFlowWorkspace.two_velocity` is a `Ref{Any}` to allow lazy construction
and changes of concrete transport/solver workspace type at AMR. The actual
force is not stored as `Any`: the default force is `nothing::Nothing`, and a
supplied callable has its own concrete type. The original intrinsic loop
repeatedly reads `state.convected[phase].interior` through the dynamic cache
inside the face loop. Its scalar access/update therefore dispatches and boxes
per face. It does this even when the force is absent and all it adds is zero.

The frozen 496-face probe repeats that original path explicitly: **1,667,376 B
per call**, versus **784 B** for identical arithmetic behind a typed function
boundary. The latter fixed overhead includes extracting the velocity tuple
from the dynamic caller, not a per-face allocation. Frozen values and NaN
masks agree exactly.

## Proposed and implemented changes

1. Put external-force updates behind one shared `_body_force_substep!` function
   boundary, receiving the actual velocity fields rather than the cache.
2. Give `force=nothing` a direct no-op method: no mesh traversal, field writes
   or callback evaluation. This also applies to one-velocity stepping.
3. Specialize the real-force helper on the concrete callable type. Bind the
   velocity array outside the face loop and update it in place, without
   allocating force vectors or changing the callback API.
4. Preserve the original sequential phase-major/face-major callback order,
   midpoint time, arithmetic, and interior-only update. Stateful callbacks
   are supported, so this loop is not parallelized or shared between phases.
   Intrinsic NaNs are skipped without evaluating the callback; boundary values
   remain the responsibility of the existing boundary stage.
5. Keep phase masking, boundary enforcement and velocity copies. Their
   measured cost is small, and removing them would require a separate proof
   that interface movement, public field edits and boundary updates cannot
   invalidate their checks. Keep the AMR-compatible dynamic cache too: dispatch
   once outside the loop is sufficient; a broad workspace type rewrite is
   unnecessary for the measured problem.

## Measured result

Exclusive, sequential before/after runs use the same isolated Julia 1.11.9
environment, normal optimization, no package images and one Julia/BLAS thread.
Default Stokes3 advances to `t=0.5`; setup, warmup and snapshot I/O are outside
the timer. The before source is commit `41d13bf`, after AMR reuse and before
this force-loop change. Each entry is a median of three production runs.
Allocation is cumulative decimal GB, not peak RAM.

| Whole production advance | One: before | One: after | CMDGFM: before | CMDGFM: after |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 15.836 s | 15.767 s | 25.692 s | 19.570 s |
| Wall-time range | 15.759–16.180 s | 15.738–16.122 s | 25.672–25.888 s | 19.567–19.723 s |
| Allocated | 12.533 GB | 12.533 GB | 26.146 GB | 16.287 GB |
| GC time, included above | 1.914 s | 1.883 s | 3.715 s | 2.131 s |
| Steps / AMR changes | 206 / 45 | 206 / 45 | 219 / 79 | 219 / 79 |
| Final blocks / faces | 97 / 12,504 | 97 / 12,504 | 70 / 9,016 | 70 / 9,016 |

CMDGFM saves **6.121 s (23.8%)** and **9.859 GB (37.7%)**. One velocity is
essentially unchanged. CMDGFM's whole advance now costs 1.24 times the
one-velocity time and 1.30 times its allocations; this is not a matched-mesh
comparison, since the two models have different step counts and AMR histories.

The new 4 ms whole-component profile puts the CMDGFM residual at **0.102 s**,
versus the preceding AMR result's **6.923 s**. Separate 2 ms profiling resolves
0.102 s of phase masking and 0.004 s of remaining step/simulation controls;
the optimized external-force path is no longer resolved in those samples.
See `PLAN.md` for the complete component table. These are sampling estimates,
not exact stage timers; do not add their times to the unprofiled medians.

The frozen probe also measures the unchanged side work. On its 496-face
state, copies allocate zero bytes; phase masking allocates **3,616 B** and
takes about **4 microseconds**, as does boundary enforcement including that
mask. These two latter measurements overlap and must not be added. Retaining
them protects the physical NaN contract without material overhead. The
production force kernel allocates **zero bytes** with no force and with the
nonzero test callback. No-force performs no face traversal; the nonzero kernel
takes approximately 0.8 microseconds on this state. User callbacks may of
course allocate internally.

## Validation gates and current status

- All **3,595 new focused checks pass** under Julia 1.11.9 / generic `-O1` /
  forced bounds. They cover Float32/Float64, shared/intrinsic velocities,
  uniform/hanging-face periodic/bounded meshes, arithmetic, boundary ownership,
  inactive NaNs, true no-force no-op semantics (including signed zeros),
  stateful callback order and an analytically known midpoint-forced flow with
  one velocity, CMDGFM and mimetic projection.
- Warmed inner force helpers allocate **zero bytes** for both absent and real
  allocation-free callable forces. This is a kernel claim, not a zero-allocation
  whole-step/whole-run claim; allocating user callbacks remain the user's choice.
- Exclusive paired three-repetition `t=0.5` production benchmarks and the
  whole-component and allocation profiles are complete.
- Independent before/after comparisons at **`t=0.08` and `t=0.5` pass for both
  models**: fraction, pressure and velocity errors are exactly `Linf=0`;
  physical NaN masks, topology, every default diagnostic history and final
  pressure iteration counts match. Final pressure solves converge. Snapshots
  originate before the AMR reuse changes and thus also cover that work. These
  checks establish numerical equality, not a bitwise claim about signed zeros.
- The full declared-dependency `Pkg.test` suite completes after performance
  measurement with **20,595 passes, seven failures, no errors/broken tests**
  in 6m17.4s, under Julia 1.11.9 / generic `-O1` / forced bounds. All 3,595
  new force checks and all 298 AMR regression checks are included and pass;
  existing CMDGFM projection/reuse and inactive-velocity checks also pass.
  This is a completed suite, **not a clean suite**. The failures match the
  preceding documented baseline; no numerical assertion or tolerance has
  been weakened.

| Existing failing assertion | Observed result |
| --- | --- |
| `two_phase/transport.jl:48` | warmed reconstruction: 3,072 B, required zero |
| `navier_stokes/two_phase_navier_stokes.jl:58,72` | warmed timestep selection: 176 B each, required zero |
| `diagnostics.jl:234,248` | warmed cut-cell timestep selection: 16 B each, required zero |
| `mimetic_projection_variants.jl:65` | pressure-commuting residual 0.9, required below 0.2 |
| `examples/cylinder_example.jl:13` | nonfinite pressure-drag history |

These are the same five allocation and two numerical assertions recorded in
[`AMR_FINDINGS.md`](AMR_FINDINGS.md) and
[`PRESSURE_REUSE_FINDINGS.md`](PRESSURE_REUSE_FINDINGS.md). The latter contains
the independent archived-baseline reproductions; the AMR report records the
reconstruction allocation's earlier 3,008 to 3,072 B change. None is fixed or
hidden by this force-loop optimization.

No-force is now a true no-op; unlike adding `+0`, it preserves a signed zero.
The intended physical result is identical. Independent full-flow comparisons
find no downstream numerical or diagnostic-history difference.

The short allocation-profile repeat has the same 35 steps, 7 AMR changes and
22,232 final faces, with **2.548 GB** allocated across the complete advance,
versus **5.270 GB** before. The optimized predictor has no sampled allocation
events. This reinforces the full-run result; it does not make sampled
allocations exact per-stage measurements or show that every other stage is
allocation-free.

## Reproduction

```sh
julia --project=. development/stokes3_performance/step_overhead_analysis.jl cmdgfm 0.5
julia --project=. development/stokes3_performance/step_overhead_analysis.jl cmdgfm 0.08 allocs
julia --project=. development/stokes3_performance/step_overhead_probe.jl
julia --project=. development/stokes3_performance/amr_validation.jl bench /tmp/reference 0.5
```

Use the isolated Julia/runtime/depot configuration in `PLAN.md` for comparable
measurements. The before source is archived commit `41d13bf` in a separate
temporary environment; the live source and repository dependencies were not
rewound. Initial logs: `/tmp/stokes3_step_overhead_before.log` and
`/tmp/stokes3_step_overhead_focused_tests.log`. Paired before/after logs:
`/tmp/stokes3_step_matched_before.log`, `/tmp/stokes3_step_matched_after.log`.
The full-suite log is `/tmp/stokes3_step_overhead_full_suite.log`.
