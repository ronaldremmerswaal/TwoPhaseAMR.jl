# Stokes3 performance study: transport, pressure, time-step selection, diagnostics

This is the study plan and running status. The reference is
[`examples/stokes3_wave.jl`](../../examples/stokes3_wave.jl) with its default
physics and AMR settings, `max_level=5`, `final_time=0.5`, no visualization,
and only `velocity_model` changed: `OneVelocityModel()` versus
`TwoVelocityModel(pressure_projection=CMDGFMProjection())`. Keep the default
`FrozenInterfaceFluxes()` transport policy for two velocities.

Transport, diagnostics, time-step selection and shared phase-aperture reuse
have now been implemented and benchmarked; see
[`TRANSPORT_FINDINGS.md`](TRANSPORT_FINDINGS.md),
[`DIAGNOSTICS_FINDINGS.md`](DIAGNOSTICS_FINDINGS.md) and
[`TIMESTEP_FINDINGS.md`](TIMESTEP_FINDINGS.md). Pressure optimization
is now implemented and validated, with known baseline suite failures recorded;
see
[`PRESSURE_REUSE_FINDINGS.md`](PRESSURE_REUSE_FINDINGS.md). ILU and the
discretization, including K, are unchanged. The original pressure plan and
pre-NaN timing baseline are retained as historical measurements.

Run the accompanying sampling benchmark from the repository root:

```sh
julia --project=. development/stokes3_performance/benchmark.jl one
julia --project=. development/stokes3_performance/benchmark.jl cmdgfm
```

It warms each model with a short run, constructs a fresh flow (including the
initial AMR and pressure projection), and profiles only the subsequent advance
to `t=0.5`. Run the two commands as separate processes. This reproduces the
measurement *procedure* below; timings will depend on the machine and Julia
environment. The script reports the final pressure iteration count, **not** a
distribution over the run. `STAGE` rows are disjoint sampled estimates;
`SUBSTAGE` rows are nested and must not be added.

## Baseline already measured

One warm, single-threaded-Julia run with Julia 1.12.7, 16 BLAS threads and a
4 ms sampling interval gave the following. Stage times are approximate
wall-time allocations from profiler samples, not separately timed regions.
Compilation during the measured advances was about 0.1 s. GC is included in
the advance time and is reported separately, not as an additional stage.

| Quantity | One velocity | Two velocities, CMDGFM |
| --- | ---: | ---: |
| Advance to `t=0.5` | 39.90 s | 143.19 s |
| Steps | 206 | 219 |
| AMR topology changes | 45 | 32 |
| Final blocks / faces | 97 / 12,504 | 166 / 21,448 |
| Allocated during advance | 56.1 GB | 255.0 GB |
| GC during advance | 6.90 s | 28.26 s |
| Final pressure solve only | 10 CG cycles | 7 BiCGSTAB iterations |
| Final CMDGFM system | — | 10,623 pressure + 491 jump unknowns |
| Final CMDGFM reuse flags | — | pressure hierarchy: yes; jump factor: no |

Both flows began with 214 blocks and 27,576 faces. Approximate stage costs:

| Sampled stage | One velocity | Two velocities, CMDGFM |
| --- | ---: | ---: |
| Transport, including checks within its call | 8.44 s | 57.56 s |
| Pressure-operator refresh + projection | 2.36 s | 25.32 s |
| Diagnostics | 7.46 s | 21.40 s |
| Outer time-step selection | 0.03 s | 8.33 s |
| Diffusion | 3.91 s | 7.17 s |
| Phase forces | 7.68 s | 9.00 s |
| AMR adaptation + topology rebuild | 10.04 s | 9.30 s |
| Other step work | approximately 0 s | 5.11 s |

## Current `t=0.5` status against the original baseline

**Baseline boundary:** this timing table predates the correction of dormant
phase velocities in the vorticity AMR criterion. The corrected CMDGFM
validation has 219 steps, 79 AMR changes and 9,016 final faces, versus
219 / 32 / 21,448 here. See [PHASE_VELOCITY_FINDINGS.md](PHASE_VELOCITY_FINDINGS.md).
Pressure performance must therefore be remeasured on that corrected
trajectory, with matched-state checks separating reuse from the mesh change.

The current runs include the transport, diagnostics, zero-allocation
time-step and shared face-aperture changes. Each stage entry is sampled, not
a precise sum of non-overlapping timers. Both runs use the default Stokes3
settings and have the same step/AMR histories as their initial counterparts.

| Quantity | One: initial | One: current | CMDGFM: initial | CMDGFM: current |
| --- | ---: | ---: | ---: | ---: |
| Advance to `t=0.5` | 39.90 s | 20.30 s | 143.19 s | 51.13 s |
| Allocated during advance | 56.1 GB | 11.4 GB | 255.0 GB | 58.7 GB |
| GC during advance | 6.90 s | 3.33 s | 28.26 s | 10.46 s |
| Steps / AMR changes | 206 / 45 | 206 / 45 | 219 / 32 | 219 / 32 |
| Final blocks / faces | 97 / 12,504 | 97 / 12,504 | 166 / 21,448 | 166 / 21,448 |
| Transport | 8.44 s | 3.58 s | 57.56 s | 5.25 s |
| Pressure refresh + projection | 2.36 s | 2.30 s | 25.32 s | 21.66 s |
| Diagnostics | 7.46 s | 0.046 s | 21.40 s | 0.192 s |
| Outer time-step selection | 0.03 s | 0.062 s | 8.33 s | 0.221 s |
| Diffusion | 3.91 s | 3.96 s | 7.17 s | 6.73 s |
| Phase forces | 7.68 s | 0.378 s | 9.00 s | 0.710 s |
| AMR adaptation + topology rebuild | 10.04 s | 9.96 s | 9.30 s | 8.45 s |

The wall-time improvement is 1.97× for one velocity and 2.80× for CMDGFM.
The two methods still have different meshes and step counts, so the table
compares each method with its own initial run, not their kernels on one state.

### Pressure status after the NaN/AMR correction

The following runs use the corrected phase-storage contract and a different,
explicit runtime configuration: Julia 1.12.7, no package images, generic CPU,
`-O1`, forced bounds, one Julia and one BLAS thread. They must not be pooled
with the native timing table above. Stage entries remain sampled estimates.

| Quantity at `t=0.5` | One velocity | CMDGFM before direct incidence | CMDGFM with direct incidence |
| --- | ---: | ---: | ---: |
| Advance | 42.96 s | 101.55 s | 69.82 s |
| Allocation, whole advance | 11.30 GB | 57.47 GB | 57.30 GB |
| Steps / AMR changes | 206 / 45 | 219 / 79 | 219 / 79 |
| Final faces | 12,504 | 9,016 | 9,016 |
| Pressure refresh + projection | 8.03 s | 44.52 s | 11.94 s |
| Within projection: CMDGFM assembly | — | 34.25 s | 1.32 s |
| AMR adaptation + topology rebuild | 14.72 s | 33.18 s | 33.15 s |

The pressure stage is now about 1.49× the one-velocity stage on these different
trajectories. Direct incidence preserves the same signed face-area entries;
ILU, K and the stopping conditions are unchanged. The entire two-velocity
advance still allocates substantially more, so exact pressure-allocation
attribution is kept separate in
[PRESSURE_REUSE_FINDINGS.md](PRESSURE_REUSE_FINDINGS.md).

At the intermediate typed solve/recovery barrier stage, exact timers gave
**10.40 s versus 7.69 s** for pressure refresh + projection (1.35×), with
1.273 GB versus 0.119 GB allocated in those regions. The timing gap is much
smaller; the pressure-allocation target was still open at that stage. Julia
runtime/compiler crashes hindered its full-suite rerun. The completed 15,707-pass
suite predates the final changes; all 499 focused reuse/solution checks pass
with the final barrier. See the pressure findings for the precise validation
status and next allocation candidates.

The final pooled/unboxed assembly and gauge-buffer reuse was validated in an
isolated Julia 1.11.9 environment because Julia 1.12.7 also crashed in
package-free startup. The installed runtime and repository dependency files
remain unchanged. Keep these results separate from the Julia 1.12/native
tables above:

| Exact pressure regions to `t=0.5`, Julia 1.11.9 / generic `-O1` / forced bounds / BLAS 1 | One velocity | CMDGFM final |
| --- | ---: | ---: |
| Coefficient/GMG refresh | 2.349 s / 119.390 MB | 1.995 s / 111.159 MB |
| Projection | 17.311 s / 0.224 MB | 22.196 s / 57.488 MB |
| Total pressure | 19.660 s / 119.615 MB | 24.191 s / 168.647 MB |
| Iterations ordinary, median/p95/max | 11 / 13 / 13 CG | 8 / 11.1 / 18 BiCGSTAB |
| Iterations post-AMR, median/p95/max | 11 / 13 / 14 CG | 8 / 14.1 / 16 BiCGSTAB |
| Steps / AMR / final faces | 206 / 45 / 12,504 | 219 / 79 / 9,016 |

Pressure time and allocated bytes are now within factors **1.23 and 1.41**
respectively. ILU, K, equations and stopping conditions are unchanged. This
does not mean equal per-call allocation or equal whole-advance allocation.
The remaining whole-run gap is outside these pressure regions; pressure-related
construction inside AMR remains a separate attribution item. All 1,220 focused
checks and 88 Poisson/KH checks pass. The final full `Pkg.test` completed with
16,702 passes and seven baseline failures: two numerical failures and five
small allocation checks, all reproduced on the pre-pressure commit in Julia
1.11. No assertions were relaxed; details and runtime caveats
are in the pressure findings.

Three normal-optimization production repetitions per model also completed
under isolated Julia 1.11.9 / BLAS 1, with identical physical histories:

| Production `simulate!` to `t=0.5` | One velocity | CMDGFM final |
| --- | ---: | ---: |
| Wall time, median (range) | 15.759 s (15.637–16.091) | 42.612 s (42.491–42.680) |
| Allocation, median | 12.717 GB | 61.749 GB |
| GC time, median | 1.905 s | 6.025 s |
| Maximum compilation per run | 0.105 s | 0.050 s |

This is not comparable to the Julia 1.12 initial/native numbers without a
runtime-matched baseline. The remaining whole-advance ratio is 2.70× in time
and 4.86× in allocation; it must not be presented as the pressure-stage ratio.

Exact pressure traces under this normal-optimization configuration give:

| Pressure regions to `t=0.5`, Julia 1.11.9 / normal optimization / BLAS 1 | One velocity | CMDGFM final |
| --- | ---: | ---: |
| Refresh | 0.316 s / 119.316 MB | 0.382 s / 111.071 MB |
| Projection | 1.368 s / 0.224 MB | 1.680 s / 57.467 MB |
| Total pressure | 1.684 s / 119.540 MB | 2.062 s / 168.538 MB |

The pressure ratio is **1.22× time / 1.41× allocation**. These regions have
zero compilation and all projections converge. The exact-stage wrapper is
not the production whole-advance benchmark; pressure topology construction
inside AMR is outside these regions for both models. Per-call allocation
still differs, partly from the unchanged jump ILU.

The phase-force difference was not simply a mesh effect. On one fixed
27,576-face Stokes3 state, the former uncached face-opening loop took 31--32 ms
and allocated 175 MB **per call**. Gravity now reads the interface-owned
aperture cache shared with the divergence/projection assembly, avoiding that
clipping loop. The matched-state [`force_analysis.jl`](force_analysis.jl)
benchmark also exposed an abstract-workspace capture in the one-velocity
acceleration update: a typed face-loop helper reduced its complete force call
from about 6.2 ms / 15.6 MB to 0.73 ms / 1.50 MB on the same state. The
two-velocity call was about 1.43 ms / 1.50 MB. Most remaining force
allocations arise inside surface-tension and weighted-gradient evaluation;
at under 2 MB per call, they are no longer a leading full-run contributor.

Within two-velocity transport, the previous nested sample estimates were
about 46.6 s in `_advect_wy!`, including about 29.3 s in phase-face momentum
transport and 12.2 s in interface reconstruction. These nested figures are
only clues; the next study must use non-overlapping timers. The raw 3.6x wall
ratio is **not** a same-state algorithmic ratio: the methods follow different
time-step counts and AMR histories. In particular, the two-velocity run ends
on a mesh with roughly 1.7x as many faces.

## 1. Transport: why is the two-velocity path so expensive?

The transport is similar at the level of the WY interface update, but not at
the level of face momentum, geometric preparation, or number of sampled faces.
[`phase_face_transport.jl`](../../src/two_phase/phase_face_transport.jl) advects
intrinsic face momentum for each phase; the two-velocity call goes through
[`interface_transport.jl`](../../src/two_phase/two_velocity/interface_transport.jl)
and the shared [`vof_fluxes.jl`](../../src/indicator/phase/vof_fluxes.jl). The
current stage bucket also contains an inner stability check, so its 57.56 s
cannot be assigned to advection alone.

1. Add non-overlapping per-call timers and allocation counters around: the
   transport wrapper and its CFL check; reconstruction; aperture freezing;
   directional WY flux construction and VOF update; each phase's face-momentum
   transport; scalar/velocity recovery; and remaining transport work. Count
   calls as well as time, because repeated work can dominate a cheap kernel.
2. Record step number, physical time, `dt`, blocks, faces, mixed faces and AMR
   update markers. Report total, per-call median/p95, and cost per face per
   call. Attribute GC and allocations to the same regions where practical.
3. Compare both complete trajectories, **then** use identical frozen mesh and
   interface snapshots to benchmark each model's transport on matched states.
   Compare at least an early fine-mesh state and a late coarse/fine state.
   This separates algorithmic cost from different mesh histories. Keep input
   velocities and interface snapshots unchanged between repetitions.
4. Test the likely contributors separately: two intrinsic phase-face momentum
   passes, repeated interface reconstruction/aperture work, split-sweep flux
   work, and allocation/GC. Do not infer a cause solely from a package-frame
   profiler percentage.

**Gate:** the instrumented default runs reach `t=0.5` with the same step and
AMR counts as their uninstrumented counterparts, within floating-point timing
noise; the measured subregions account for at least 90% of the transport
bucket; and the matched-state benchmark identifies a stable per-face ratio
over repeated warm runs. No transport rewrite until this decomposition exists.

## 2. Pressure: assembly, solve, and reuse

The focused, current code-based analysis and validation sequence is in
[`PRESSURE_PLAN.md`](PRESSURE_PLAN.md). In particular, the latest profile
places coupled assembly well ahead of jump-ILU construction as a likely
source of the CMDGFM cost. The outline below is the original broad plan.

The one-velocity pressure projection is a symmetric scalar Poisson solve with
CG and a tree-geometric multigrid preconditioner
([`projection.jl`](../../src/navier_stokes/projection.jl),
[`elliptic_solver.jl`](../../src/solvers/multigrid/elliptic_solver.jl)). On an
unchanged mesh, the scalar operator is refreshed from a template with new
interface-dependent inverse-density coefficients, and the multigrid hierarchy
is refreshed numerically while retaining its topology and work buffers.
Across AMR, the pressure solver tries to resize and reuse its Krylov/vector
storage and to reuse multigrid construction where possible
([`stepping.jl`](../../src/two_phase/stepping.jl),
[`workspace.jl`](../../src/navier_stokes/workspace.jl)).

CMDGFM additionally constructs a nonsymmetric coupled pressure/jump sparse
system for every projection. It uses BiCGSTAB with a block preconditioner:
tree multigrid for the reduced pressure block and ILU for the active jump
block ([`cmdgfm_projection.jl`](../../src/two_phase/two_velocity/projection/cmdgfm_projection.jl)).
The BiCGSTAB vectors and residual buffer are reused/resized. On a fixed mesh,
the pressure hierarchy topology is reusable, but changing interface
coefficients trigger a numerical refresh. The jump ILU is reused only when
the **entire** jump block (including values) is identical, which is unusual
for a moving interface. AMR rebuilds the projection workspace. An earlier
Stokes3 solver study found that blindly reusing stale numerical pressure
multigrid coefficients caused divergence; see
[`STOKES3_CMDGFM_SOLVER.md`](../STOKES3_CMDGFM_SOLVER.md). Thus reuse must be
split into safe symbolic/topological reuse and necessary numerical updates.

1. Instrument every projection, including the initial projection and those
   following AMR, with separate timings/allocations for scalar pressure
   refresh, CMDGFM face data/stencils, incidence and active-cell/gauge work,
   block assembly and sparse concatenation, multigrid numerical refresh,
   jump ILU construction, BiCGSTAB, residual checks, and gradient recovery.
   Record matrix dimensions, nonzeros, pattern hash, mixed-face count and
   `pressure_hierarchy_reused`/`jump_factor_reused`.
2. Collect **all** one-velocity CG cycle counts and all CMDGFM BiCGSTAB
   iteration counts, not just the last solve. Report mean, median, p95, max,
   iteration count versus time and mesh size, and seconds per iteration. Also
   record any BiCGSTAB correction restarts or unpreconditioned fallback. CG
   cycles and BiCGSTAB iterations are not interchangeable units: compare
   solve time and preconditioner applications as well.
3. Partition results by unchanged mesh, first step after AMR, and repeated
   projections of an unchanged interface. This shows whether the present
   workspaces and preconditioners actually reuse storage in the situations
   where they can. Keep a controlled same-matrix repeated-solve benchmark to
   isolate pure Krylov/preconditioner cost from matrix assembly.
4. Evaluate candidate improvements one at a time: precompute fixed-mesh
   incidence/topology and indexing; refill numerical entries of a fixed
   sparsity pattern instead of rebuilding sparse blocks; retain assembly
   scratch; avoid duplicating the scalar pressure operator where provably
   possible; and, only if justified by defects/iterations, test cheaper
   numerical preconditioner refresh strategies. Measure whether assembly or
   solve time falls and whether allocation/GC falls. Any jump-ILU refactor
   must respect changed values; equal sparsity alone is not enough.

**Gate:** both solvers reach `t=0.5` with unchanged convergence criteria and
pressure/normal-velocity defects; iteration distributions do not develop an
AMR-triggered or late-time tail; momentum/conservation diagnostics stay at
baseline accuracy. Accept a reuse change only if it is faster in warm full
runs, not merely in a frozen-matrix microbenchmark. Record the first changed
step and numerical defect if a proposed reuse policy fails.

## 3. Time-step selection: separate the bound from its preparation

The focused follow-up [time-step plan](TIMESTEP_PLAN.md) sets a zero-allocation
target for the complete warmed solver-level query, including the first query
after a prepared AMR rebuild, and details the measurement and validation gates.
Its implementation measurements and reconstruction call-site audit are in
[the time-step findings](TIMESTEP_FINDINGS.md).

The outer `simulate!` loop asks for `stable_timestep` before every step.
`_transport_substep!` asks again with a different CFL, and `_advect_wy!`
performs its own safety check. For `FrozenInterfaceFluxes`, a two-velocity
`stable_timestep` reconstructs the interface and freezes phase apertures
before scanning cell/face rates
([`simulate.jl`](../../src/navier_stokes/simulate.jl),
[`two_velocity/stepping.jl`](../../src/two_phase/two_velocity/stepping.jl),
[`interface_transport.jl`](../../src/two_phase/two_velocity/interface_transport.jl),
[`transport_geometry.jl`](../../src/two_phase/two_velocity/transport_geometry.jl)).
The transport entry point also reconstructs/freezes to remain correct when
called independently. Therefore the 8.33 s outer-CFL cost may mostly be
geometry preparation, and further CFL work is hidden inside transport.
The stated call count is a source-level expectation; instrumentation must
confirm which branches run and how often.

1. Count and time every `stable_timestep`, `reconstruct!`, aperture freeze and
   CFL scan by call site: outer loop, transport guard, and advection guard.
   Record the chosen bound from each, the phase/force limiter that wins, and
   whether interface or velocity state changed between calls.
2. Benchmark the pure rate scan separately from reconstruction and aperture
   preparation on identical states. Measure allocations. Check whether the
   inner `cfl=0.5` guard can reuse the outer `cfl=0.45` result without losing
   a stronger capillary/gravity restriction or allowing an invalid step.
3. If redundant, prototype a step-local *prepared-geometry/bound* token for
   the integrated stepping path. Invalidate it whenever the interface,
   velocity or mesh changes; leave public `advect!` and `stable_timestep`
   correct for independent callers. Avoid a global stale cache.

**Gate:** every accepted step still satisfies the original transport and force
limits, including AMR transitions; stand-alone `advect!` still rejects unsafe
steps; Stokes3 step counts and diagnostics are unchanged (or differences are
explained by a deliberate bound policy change). Confirm the outer and inner
CFL buckets both shrink rather than merely moving cost between stages.

## 4. Diagnostics: why nearly three times as expensive?

The follow-up [`DIAGNOSTICS_PLAN.md`](DIAGNOSTICS_PLAN.md) expands this section
with a staged same-state/AMR validation strategy. Its first measurements and
codebase findings are in
[`DIAGNOSTICS_FINDINGS.md`](DIAGNOSTICS_FINDINGS.md).

Stokes3 enables `KineticEnergy`, `LinearMomentum`, `PhaseVolume`,
`DivergenceNorm` and `LeafBlockCount` at every step. The measured total was
about 7.46 s for one velocity and 21.40 s for two velocities. Those are
whole-run totals on different mesh/time-step histories, so this is not yet a
same-mesh 3x kernel comparison. Still, the difference deserves its own
breakdown: the two-velocity energy and momentum calculations each traverse
faces and compute a staggered liquid fraction; `DivergenceNorm` constructs a
new mixture-velocity field, density vector and divergence field before
integrating the norm
([`interface.jl`](../../src/simulation/diagnostics/interface.jl)). The
one-velocity calculations also do nontrivial density and face-volume work,
but do not need two intrinsic phase fields. This source inspection identifies
possible repeated work, not a measured cause.

1. Time and count each of the five diagnostics separately, including the
   initial, per-step and completed events. Record allocations, GC, faces and
   cells per evaluation. Report per-face cost and full-run cost, and compare
   early/late matched mesh snapshots so AMR history cannot masquerade as a
   diagnostic-kernel difference.
2. Within energy, momentum and divergence, isolate face-volume construction,
   mixture density/fraction evaluation, mixture-velocity construction,
   divergence kernel, and history append. Count how many times each full face
   traversal occurs at the same step. Check whether a diagnostics-only shared
   scratch/cache can safely reuse face volumes, fractions and temporary
   fields after the completed projection, invalidating on every interface or
   mesh change. Do not cache a quantity across changing states merely because
   the mesh is unchanged.
3. Run the same `t=0.5` comparison once with default diagnostics and once
   with diagnostics disabled to measure their impact on total wall time. Use
   diagnostic-off only as an isolation experiment; default diagnostics remain
   the performance and accuracy reference. Try any caching/fusion candidate
   separately from transport or pressure changes.

**Gate:** diagnostic values and recorded times/steps are unchanged to their
existing numerical tolerances, including after AMR; default diagnostics are
still produced; shared scratch does not retain stale interface data; and the
warm full-run diagnostic bucket (not only a microbenchmark) shrinks without
raising transport or pressure cost.

## Measurement and validation sequence

1. Re-run this benchmark at least three times per model after warmup and
   report medians and spread. Capture Julia/BLAS thread counts, version,
   commit, machine, compilation time, profiler coverage and total GC. Keep
   visual output disabled and all Stokes3 defaults fixed. Reject timings if
   compilation is material or run-to-run spread obscures the effect.
2. Add low-overhead structured per-step counters/timers for the four target
   contributors, taking care not to perturb the solver. Verify their sums
   against wall time. Use the current sampled profile as a guide, not as
   exact accounting.
3. Perform the frozen-state transport and pressure benchmarks to separate
   per-operation cost from the differing AMR/step histories. Then trial one
   candidate optimization at a time, retaining a before/after profile.
4. Run relevant transport, projection and AMR tests, including
   [`test/examples/stokes3_example.jl`](../../test/examples/stokes3_example.jl),
   followed by both default `t=0.5` Stokes3 runs. Check pressure defects,
   mass/phase volume, momentum, energy diagnostics, final mesh/step counts,
   and no solver failures.
5. Deliver a table of baseline versus each accepted change: transport,
   pressure assembly/preconditioner/Krylov, outer/inner CFL, each diagnostic,
   AMR, allocation/GC, total wall time, and full pressure-iteration
   distributions. Explicitly distinguish same-state speedups from the final
   physically evolving run.

Different mesh histories remain an important control for all four sections.
