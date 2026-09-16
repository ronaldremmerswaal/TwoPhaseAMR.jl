# Stokes-3 performance analysis

## Scope and method

This report analyzes the default Stokes-3 discretization through physical time
`t=1`. It uses the default 8×8-cell blocks, maximum AMR level 5, density ratio
850, Reynolds number `4e4`, Bond number `1e3`, conservative GFM surface
tension, gravity, curvature refinement, and vorticity refinement. The fixed
interface criterion targets level 5 through `t=0.01`.

The run was recorded on 2026-09-16 with Julia 1.12.7 on an AMD Ryzen 9 5950X.
Julia had 32 threads and OpenBLAS had 16 threads. The implementation's mesh and
finite-volume loops are serial; the thread count should therefore not be read
as 32-way solver parallelism.

[`stokes3_performance.jl`](stokes3_performance.jl) executes the same numerical
substeps as `step!`, but places `@timed` measurements around each substep. A
one-step comparison against the production `simulate!` path gave bitwise equal
velocity, pressure, and volume fractions. The benchmark records:

- elapsed, GC, and first-use compilation time;
- allocated bytes and allocation count;
- pressure and diffusion Krylov iterations;
- block, cell, face, and maximum-level counts;
- AMR refinement/coarsening counts and retained solver state; and
- a CSV row with every metric for every time step.

The measured run excludes Makie sampling and rendering. This isolates the
simulation slowdown and means that visualization cannot explain the AMR
spikes reported below. It is a single diagnostic trace, rather than a
statistical timing distribution. Times remain machine dependent.

## Initialization

The script first compiles a small level-0 problem, invokes GC, and then measures
construction of the default case with `final_time=0`. Construction includes
level-set initialization, the initial AMR passes and analytical resampling,
solver and workspace construction, multigrid hierarchy construction, and the
initial one-velocity pressure projection.

| metric | value |
|---|---:|
| warmed setup time | 7.973 s |
| allocated during setup | 4.355 GiB |
| allocations during setup | 178,087,082 |
| setup GC time | 0.714 s |
| retained solver/workspace state | 143.629 MiB |
| initial active blocks | 214 |
| initial active cells | 13,696 |

Some mixed-level methods were first compiled during the physical run despite
the small warm-up. The per-step trace attributes 1.140 s to this first-use
compilation. It is less than 0.6% of total measured step time and does not
explain the persistent slowdown.

## End-to-end result through `t=1`

| metric | value |
|---|---:|
| physical steps | 448 |
| topology changes | 71 (15.8% of steps) |
| measured substep time | 205.812 s |
| instrumented wall time | 212.407 s |
| median step | 120.329 ms |
| p95 step | 2,296.199 ms |
| maximum step | 5,203.162 ms |
| mean without topology change | 120.971 ms |
| mean with topology change | 2,256.420 ms |
| allocated during physical steps | 211.856 GiB |
| allocations during physical steps | 3,780,058,116 |
| GC time during physical steps | 31.064 s |
| final active blocks/cells | 115 / 7,360 |
| final retained solver/AMR state | 82.449 MiB |

A topology-changing step is **18.7× slower** than a fixed-topology step. It
allocates 1,897 MiB on average, compared with 218 MiB on an ordinary step.
The maximum retained state was 143.807 MiB, so most of this traffic is
short-lived construction rather than useful persistent state.

The 6.595 s difference between summed substep measurements and instrumented
wall time is profiler bookkeeping, dictionary updates, and retained-size
sampling after regrids. It is 3.1% of wall time and does not alter the main
ranking.

## Cost by stage

| stage | total time | share | mean/step | allocated |
|---|---:|---:|---:|---:|
| topology-dependent solver/workspace rebuild | 103.374 s | 50.2% | 230.746 ms | 63.994 GiB |
| AMR tree/schedule/field transfer | 47.465 s | 23.1% | 105.948 ms | 51.217 GiB |
| diffusion matrix and preconditioner refresh | 25.521 s | 12.4% | 56.967 ms | 67.550 GiB |
| pressure preconditioner refresh | 7.497 s | 3.6% | 16.734 ms | 16.194 GiB |
| diffusion Krylov solve | 6.187 s | 3.0% | 13.810 ms | 1.224 GiB |
| WY interface and momentum advection | 5.157 s | 2.5% | 11.511 ms | 0.340 GiB |
| pressure projection solve and correction | 2.656 s | 1.3% | 5.929 ms | 0.010 GiB |
| dynamic-viscosity stress operator | 2.300 s | 1.1% | 5.135 ms | 3.595 GiB |
| variable-density pressure operator | 2.248 s | 1.1% | 5.017 ms | 4.745 GiB |
| AMR indicator evaluation | 2.036 s | 1.0% | 4.545 ms | 0.910 GiB |
| gravity | 0.788 s | 0.4% | 1.760 ms | 1.039 GiB |
| surface tension | 0.433 s | 0.2% | 0.967 ms | 0.988 GiB |
| staggered mixture density | 0.057 s | <0.1% | 0.128 ms | 0.049 GiB |
| CFL, state copies, and the empty user-force loop | 0.092 s | <0.1% | 0.205 ms | <0.001 GiB |

AMR field transfer and subsequent solver reconstruction consume **73.3% of
the entire run**. They consume 115.211 GiB of allocation traffic. On the 71
changing steps, transfer averages 0.667 s and the subsequent solver/workspace
rebuild averages 1.456 s. The remaining physical work on those steps is close
to the ordinary-step cost.

The allocation rate also causes a secondary bottleneck. GC consumes 31.064 s,
or 15.1% of measured stage time. Its largest sources are diffusion hierarchy
refresh (9.833 s), topology rebuild (8.960 s), AMR transfer (7.655 s), and
pressure hierarchy refresh (3.144 s).

## Fixed-topology steps

Removing the 71 topology-changing steps leaves 377 ordinary steps and 45.606 s
of work:

| ordinary-step stage | time | share of ordinary-step time |
|---|---:|---:|
| diffusion matrix/preconditioner refresh | 20.929 s | 45.9% |
| pressure preconditioner refresh | 6.125 s | 13.4% |
| diffusion solve | 5.119 s | 11.2% |
| advection | 4.287 s | 9.4% |
| pressure projection | 2.238 s | 4.9% |
| pressure coefficient/operator refresh | 1.961 s | 4.3% |
| dynamic-viscosity operator | 1.955 s | 4.3% |
| AMR indicators | 1.693 s | 3.7% |

Rebuilding the diffusion and pressure preconditioners therefore consumes
59.3% of a normal step. Adding their coefficient/operator work raises this to
67.9%. The actual diffusion and pressure solves consume only 16.1%.

The linear iteration counts remain stable: pressure uses 10–14 iterations
(median 11), and diffusion uses 5–7 (median 6). Regridding does not cause a
Krylov convergence collapse. The slowdown comes from rebuilding sparse
operators, Krylov workspaces, and multigrid hierarchies, rather than from
additional solve iterations.

## AMR behavior

The first physical topology change occurs at step 7, `t=0.016865`. Initial
adaptation uses pass indices 1–6, which leave `last_update_step=6`, while the
physical simulation restarts its completed-step counter at zero. Consequently
the first six physical AMR calls cannot attempt an update. By the first eligible
update, the interface criterion's `t_end=0.01` has passed. The mesh immediately
coarsens from 214 to 190 blocks; that step is the 5.203 s maximum.

The later trace contains repeated refinement/coarsening reversals. Examples
include:

- steps 99–105: 121 → 142 → 121 → 142 → 124 → 121 blocks;
- steps 196–205: 103 → 124 → 106 → 124 → 106 → 97 → 106 blocks; and
- several one-family 97 ↔ 100 and 103 ↔ 106 transitions.

There are only 6.14 steps on average between topology changes, and consecutive
changes occur. `update_interval=1` permits this. Although value hysteresis
exists inside the curvature and vorticity indicators, the combined criteria,
field transfer, and immediate reevaluation still permit topology churn.

The mesh shrinks from 13,696 to 7,360 cells, so later ordinary steps are faster
than early ones. However, repeated topology construction more than consumes
the savings. The selected time step is monotonically nonincreasing by design;
it starts near `2.406e-3` and falls toward `1.5e-3` as the velocity grows, which
produces 448 steps through `t=1`.

## Bottlenecks and optimization priorities

### 1. Rebuild only the state invalidated by a topology change

`_rebuild_topology_dependent_state!` constructs a complete new
`NavierStokesSolver` and then a complete `NavierStokesWorkspace`. For the
two-phase case this reconstructs pressure and diffusion operators,
preconditioners, Krylov storage, advection storage, and projection storage.
This path costs half of the whole run. Replace it with targeted rebuilds that
reuse buffers and preserve configuration objects. In particular, avoid
constructing single-phase placeholder diffusion state that the active
two-phase path does not use.

### 2. Reduce topology churn

Test a larger `update_interval`, a longer coarsening delay or dwell time, and
stronger temporal hysteresis. The current trace spends 150.839 s in AMR
transfer plus reconstruction to save roughly 46% of the initial cells. A
criterion should need to retain the resulting topology long enough to repay
that cost.

The initial AMR pass counter and physical completed-step counter should share a
consistent epoch. At present the intended interface criterion expires before
the first physical AMR attempt. Correcting that mismatch will also make
`t_init`/`t_end` behavior easier to reason about.

### 3. Reuse coefficient-dependent hierarchy structure

On every ordinary step, variable diffusion rebuilds its sparse matrix, Krylov
workspace, and complete geometric preconditioner. Variable pressure rebuilds
its coefficient and preconditioner too. Preserve the sparsity pattern,
aggregation, and transfer structure while updating numerical coefficients and
coarse operators in place.

A refresh cadence based on coefficient change or Krylov iteration growth is
also worth testing. A permanently frozen variable-density pressure hierarchy
is known to become ineffective at high density ratio, so this optimization
needs residual and iteration safeguards. The stable 10–14 and 5–7 iteration
ranges provide room to test controlled reuse.

### 4. Remove AMR transfer allocation hot spots

The AMR path allocates 51.217 GiB in this run, while retained state never
exceeds 144 MiB. Reuse tree/schedule dictionaries and vectors, face-transfer
geometry, and field-transfer scratch wherever topology permits. A follow-up
allocation profile should split tree mutation, schedule construction,
geometric interface prolongation, and staggered field transfer. Allocation
reduction also directly reduces the 7.655 s of GC attributed to AMR transfer.

### 5. Keep the physical kernels lower priority

Advection, diffusion solve, pressure projection, surface tension, and gravity
together account for about 7.5% of total time. Optimizing these kernels before
the hierarchy and AMR construction paths cannot materially resolve the
observed pauses.

## Reproduction

Run the default `t=1` trace with:

```bash
STOKES3_PROFILE_CSV=/tmp/stokes3.csv \
  julia --project=. --threads=auto benchmark/stokes3_performance.jl
```

The benchmark prints `step=<n>, time=<t>` immediately after every completed
step. It then prints initialization, aggregate, stage, and per-step tables.
The optional CSV contains elapsed time, bytes, allocations, GC time, and
compilation time for every stage of every step. Short exploratory runs can set
`STOKES3_PROFILE_FINAL_TIME`, `STOKES3_PROFILE_BASE_CELLS`, and
`STOKES3_PROFILE_MAX_LEVEL`.
