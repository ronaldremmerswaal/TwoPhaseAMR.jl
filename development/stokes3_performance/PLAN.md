# Stokes3 performance study: transport, pressure, time-step selection, diagnostics

This is the study plan and running status. The reference is
[`examples/stokes3_wave.jl`](../../examples/stokes3_wave.jl) with its default
physics and AMR settings, `max_level=5`, `final_time=0.5`, no visualization,
and only `velocity_model` changed: `OneVelocityModel()` versus
`TwoVelocityModel(pressure_projection=CMDGFMProjection())`. Keep the default
`FrozenInterfaceFluxes()` transport policy for two velocities.

The transport workstream has now been implemented and benchmarked; see
[`TRANSPORT_FINDINGS.md`](TRANSPORT_FINDINGS.md) and its focused
[`transport_analysis.jl`](transport_analysis.jl) benchmark. The next
workstream, diagnostics, has a detailed code-based
[`DIAGNOSTICS_PLAN.md`](DIAGNOSTICS_PLAN.md) and measured
[`DIAGNOSTICS_FINDINGS.md`](DIAGNOSTICS_FINDINGS.md). Pressure optimization
remains planned, not implemented here.

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

The current runs include the transport, diagnostics, zero-allocation
time-step and shared face-aperture changes. Each stage entry is sampled, not
a precise sum of non-overlapping timers. Both runs use the default Stokes3
settings and have the same step/AMR histories as their initial counterparts.

| Quantity | One: initial | One: current | CMDGFM: initial | CMDGFM: current |
| --- | ---: | ---: | ---: | ---: |
| Advance to `t=0.5` | 39.90 s | 20.72 s | 143.19 s | 51.13 s |
| Allocated during advance | 56.1 GB | 13.1 GB | 255.0 GB | 58.7 GB |
| GC during advance | 6.90 s | 3.45 s | 28.26 s | 10.46 s |
| Steps / AMR changes | 206 / 45 | 206 / 45 | 219 / 32 | 219 / 32 |
| Final blocks / faces | 97 / 12,504 | 97 / 12,504 | 166 / 21,448 | 166 / 21,448 |
| Transport | 8.44 s | 3.73 s | 57.56 s | 5.25 s |
| Pressure refresh + projection | 2.36 s | 2.42 s | 25.32 s | 21.66 s |
| Diagnostics | 7.46 s | 0.096 s | 21.40 s | 0.192 s |
| Outer time-step selection | 0.03 s | 0.029 s | 8.33 s | 0.221 s |
| Diffusion | 3.91 s | 3.59 s | 7.17 s | 6.73 s |
| Phase forces | 7.68 s | 0.883 s | 9.00 s | 0.710 s |
| AMR adaptation + topology rebuild | 10.04 s | 9.97 s | 9.30 s | 8.45 s |

The wall-time improvement is 1.93× for one velocity and 2.80× for CMDGFM.
The two methods still have different meshes and step counts, so the table
compares each method with its own initial run, not their kernels on one state.

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
