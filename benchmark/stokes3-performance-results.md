# Stokes-3 performance optimization

## Scope and comparison contract

This report compares the Stokes-3 simulation before and after optimizing its
topology-dependent state construction. Both runs use the default case through
physical time `t=1`: 8×8-cell blocks, maximum AMR level 5, density ratio 850,
Reynolds number `4e4`, Bond number `1e3`, conservative GFM surface tension,
gravity, curvature refinement, and vorticity refinement. The fixed interface
criterion targets level 5 through `t=0.01`.

The runs were recorded on 2026-09-16 with Julia 1.12.7 on an AMD Ryzen 9
5950X. Julia had 32 threads and OpenBLAS had 16 threads. The mesh and main
finite-volume loops are serial, so the thread count does not imply 32-way
solver parallelism.

The optimization deliberately leaves the refinement criteria, thresholds,
update interval, and mesh schedule unchanged. Each CSV row contains the sorted
set of active block keys. The comparison checks these keys together with the
block, cell, face, and level counts and every refine/coarsen decision.

The two traces have:

- 448 physical steps each;
- 71 topology-changing steps each; and
- identical active block keys and AMR decisions at every step.

The maximum difference between corresponding physical times is `1.255e-14`,
and both traces finish at exactly `t=1`. This is roundoff from changing
coefficient assembly from sparse products to in-place numerical updates.
Pressure and diffusion iteration counts match at every step.

[`stokes3_performance.jl`](stokes3_performance.jl) executes the production
substeps with `@timed` measurements around each stage. Its validation case
reproduces one production step bit for bit. The benchmark records elapsed,
allocation, GC, compilation, solver-iteration, topology, and retained-state
metrics for every step. Visualization is excluded.

## Result

| metric | before | after | speedup | reduction |
|---|---:|---:|---:|---:|
| measured substep time | 203.553 s | 69.764 s | 2.92× | 65.7% |
| instrumented wall time | 209.640 s | 75.710 s | 2.77× | 63.9% |
| median step | 121.762 ms | 98.779 ms | 1.23× | 18.9% |
| p95 step | 2,275.689 ms | 459.668 ms | 4.95× | 79.8% |
| maximum step | 5,241.905 ms | 1,121.596 ms | 4.67× | 78.6% |
| mean topology-changing step | 2,218.179 ms | 440.327 ms | 5.04× | 80.1% |
| mean fixed-topology step | 122.180 ms | 102.124 ms | 1.20× | 16.4% |
| allocated during physical steps | 211.856 GiB | 107.729 GiB | 1.97× | 49.2% |
| final retained state | 82.449 MiB | 70.628 MiB | 1.17× | 14.3% |
| maximum retained state | 143.807 MiB | 113.857 MiB | 1.26× | 20.8% |

The pauses associated with adaptation are substantially shorter. The p95 and
maximum improve by roughly the same factor as topology-changing steps, while
fixed-topology steps improve by 16%. This directly addresses the observed
slowdown without changing when or where the mesh adapts.

## Initialization

The benchmark compiles a small level-0 case, runs GC, and then measures
construction of the default case with `final_time=0`. Construction includes
level-set initialization, initial AMR passes and analytical resampling, solver
and workspace construction, multigrid hierarchies, and the initial projection.

| metric | before | after | reduction |
|---|---:|---:|---:|
| warmed setup time | 7.902 s | 2.686 s | 66.0% |
| allocated during setup | 4.355 GiB | 0.980 GiB | 77.5% |
| setup allocations | 178,087,091 | 7,483,733 | 95.8% |
| setup GC time | 0.710 s | 0.232 s | 67.3% |
| retained setup state | 143.629 MiB | 113.679 MiB | 20.9% |

## Stage breakdown

| stage | before time | after time | speedup | before allocation | after allocation |
|---|---:|---:|---:|---:|---:|
| topology-dependent solver/workspace rebuild | 101.404 s | 22.095 s | 4.59× | 63.994 GiB | 28.784 GiB |
| AMR tree/schedule/field transfer | 47.161 s | 1.738 s | 27.14× | 51.217 GiB | 1.467 GiB |
| diffusion matrix and preconditioner refresh | 25.554 s | 20.449 s | 1.25× | 67.550 GiB | 53.849 GiB |
| pressure preconditioner refresh | 7.180 s | 6.272 s | 1.14× | 16.194 GiB | 14.205 GiB |
| diffusion Krylov solve | 6.349 s | 6.188 s | 1.03× | 1.224 GiB | 1.183 GiB |
| WY interface and momentum advection | 5.220 s | 4.795 s | 1.09× | 0.340 GiB | 0.340 GiB |
| pressure projection solve and correction | 2.650 s | 2.597 s | 1.02× | 0.010 GiB | 0.010 GiB |
| dynamic-viscosity stress operator | 2.283 s | 0.353 s | 6.46× | 3.595 GiB | 0.207 GiB |
| variable-density pressure operator | 2.268 s | 2.089 s | 1.09× | 4.745 GiB | 4.745 GiB |
| AMR indicator evaluation | 2.095 s | 1.776 s | 1.18× | 0.910 GiB | 0.910 GiB |
| gravity | 0.835 s | 0.857 s | 0.97× | 1.039 GiB | 1.039 GiB |
| surface tension | 0.431 s | 0.437 s | 0.99× | 0.988 GiB | 0.988 GiB |
| staggered mixture density | 0.039 s | 0.037 s | 1.06× | 0.049 GiB | 0 GiB |
| CFL, copies, and empty user-force loop | 0.083 s | 0.081 s | 1.03× | <0.001 GiB | <0.001 GiB |

Before optimization, AMR transfer plus topology-dependent reconstruction
consumed 148.566 s, or 73.0% of measured time. They now consume 23.833 s, or
34.2%. Their combined allocation falls from 115.211 GiB to 30.251 GiB.

The largest remaining cost is numerical coefficient refresh. Diffusion and
pressure hierarchy refresh together consume 26.720 s, or 38.3% of the new
total. Their Krylov solves consume 8.785 s. Iteration counts remain stable:
pressure uses 10–14 iterations and diffusion uses 5–7, so convergence has not
degraded to obtain the speedup.

## What changed

### Targeted topology rebuild

The former topology hook constructed a complete `IncompressibleNavierStokesSolver`
and `NavierStokesWorkspace`. In a two-phase run this rebuilt inactive
single-phase state as well as the pressure solver, phase advection, coupled
stress diffusion, projection, and all scratch arrays.

The two-phase hook now rebuilds only objects whose dimensions or connectivity
depend on the new mesh:

- the variable-density pressure operator and solver;
- the two-phase advection and variable-stress workspace; and
- the projection workspace.

The old velocity, convected velocity, and diffusion work arrays retain their
objects and resize in place. Two-phase solver construction also omits inactive
single-phase advection and diffusion hierarchies.

### Indexed variable-stress construction

The dominant topology constructor was the coupled variable-viscosity stress
operator. Its shear-stress stencil builder previously scanned every face for
every mesh vertex. It now builds sorted spatial indexes and queries only faces
near a vertex. Candidate and stencil buffers are reused.

An exhaustive comparison on every shear stencil of the initial Stokes-3 mesh
matched the former brute-force construction exactly. On a representative
214-to-190-block topology change, variable-stress operator construction fell
from 2.549 s to 0.332 s. The complete targeted object rebuild measured 0.519 s
and 685 MiB in the same isolated profile.

### Reused hierarchy and coefficient storage

Geometric pressure and diffusion preconditioners preserve their aggregation,
transfer operators, and Krylov storage while refreshing level matrices and
coefficients. The variable diffusion matrix preserves its sparsity pattern and
updates numerical values and inertia in place. Dynamic-viscosity assembly
precomputes the mapping from reconstructed stresses to sparse entries and
updates `nzval` storage directly.

The in-place stress matrix was compared with the former sparse triple-product
assembly; the maximum entry difference was `1.06e-22`, with identical sparsity.

### Reused AMR transfer storage

AMR transfer now reuses parent maps, refinement sets, face-line buffers, cell
and face staging arrays, and PLIC plane storage. Height-cache ghost lookup uses
the block tree to query overlapping donors instead of scanning every active
block. Donor ordering remains deterministic. This removes nearly all of the
short-lived allocation that made field transfer and GC dominate regrid steps.

## Remaining bottlenecks

Topology rebuild remains the largest individual stage at 22.095 s and 28.784
GiB. A changed topology necessarily changes degrees of freedom, pressure
connectivity, staggered control volumes, advection schedules, and variable
stress stencils. The optimized path avoids inactive and reusable state, but it
still constructs those connectivity-dependent objects.

The next largest opportunity is fixed-topology coefficient refresh:

1. diffusion matrix and multigrid refresh: 20.449 s and 53.849 GiB;
2. pressure multigrid refresh: 6.272 s and 14.205 GiB; and
3. topology rebuild: further reduce retained temporary geometry in coupled
   stress and staggered transfer construction.

A less frequent coefficient refresh might reduce this further, but it would
change solver behavior and requires residual-based safeguards at density ratio
850. It was not introduced in this comparison.

## Reproduction

Record a trace with:

```bash
STOKES3_PROFILE_CSV=/tmp/stokes3.csv \
  julia --project=. --threads=auto benchmark/stokes3_performance.jl
```

The loop prints `step=<n>, time=<t>` after every completed step. Short runs can
set `STOKES3_PROFILE_FINAL_TIME`, `STOKES3_PROFILE_BASE_CELLS`, and
`STOKES3_PROFILE_MAX_LEVEL`.

Compare two full traces with:

```bash
julia --project=. benchmark/compare_stokes3_performance.jl \
  /tmp/stokes3-before.csv /tmp/stokes3-after.csv
```

The comparator fails before printing timings if the step count, block keys,
mesh sizes, maximum level, AMR attempts, or refine/coarsen decisions differ.
