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

The main comparison below ends with the targeted-rebuild implementation at
commit `53b14f0`. The final grow-only construction-cache experiment is reported
separately under [Grow-only topology construction cache](#grow-only-topology-construction-cache).

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

## Grow-only topology construction cache

### Design decision

The follow-up investigation tested two kinds of caching. Caching complete
topology-dependent solvers is a poor fit for this run: its 72 visited states
contain 66 unique topologies, and only 6 of the 71 rebuild destinations repeat.
A complete rebuilt pressure/two-phase workspace retains about 100 MiB for a
representative topology. Even a two-entry cache would therefore retain roughly
200 MiB while producing only five hits.

The implemented cache instead retains small construction buffers and grows
them only when a later topology needs more capacity. It reuses:

- strain and stress-interpolation coordinate buffers;
- vertex-reference lists and sorted face-search storage;
- derivative reconstruction scratch arrays;
- the sparse-entry refresh map; and
- the previous diffusion operator's measures, weights, work arrays, volumes,
  and refresh-map vectors.

Sparse matrices, connectivity schedules, and multigrid hierarchies are still
rebuilt. Their patterns change with the topology, and the existing sparse and
solver constructors do not accept reusable symbolic/numeric storage.

### Exact-mesh comparison

The cache trace uses the same benchmark and is compared with the `53b14f0`
trace. Both contain 448 steps and 71 topology changes. Active block keys, mesh
sizes, AMR actions, pressure iterations, diffusion iterations, and physical
times match at every step; the maximum time-grid difference is zero.

| metric | without cache | grow-only cache | change |
|---|---:|---:|---:|
| topology-rebuild allocation | 28.784 GiB | 26.121 GiB | −9.25% |
| allocation per topology change | 415.14 MiB | 376.73 MiB | −38.41 MiB |
| topology-rebuild allocation count | 153,055,016 | 146,748,714 | −4.12% |
| allocation during all physical steps | 107.729 GiB | 105.065 GiB | −2.47% |
| retained setup state | 113.679 MiB | 133.189 MiB | +19.510 MiB |
| final retained state | 70.628 MiB | 101.801 MiB | +31.173 MiB |
| maximum retained state | 113.857 MiB | 133.367 MiB | +19.510 MiB |

The larger final-state increase includes both the construction cache and
diffusion arrays whose capacity remains sized for an earlier, larger mesh.
The maximum retained-state increase is the relevant peak-memory cost.

This change has no demonstrated wall-time benefit. Measured stage time changed
from 69.764 s to 71.665 s, and topology rebuild changed from 22.095 s to
22.531 s. Unrelated stages moved in both directions in the same run (for
example, diffusion refresh was 4.9% slower while gravity was 18.7% faster), so
the timing delta cannot be assigned to the cache. The allocation reduction is
deterministic and appears on every changed-topology step; timing remains
subject to machine load and GC scheduling.

### Rejected broader cache

An isolated 214-to-190-block rebuild allocated 684.7 MiB while the resulting
objects retained 99.6 MiB. The largest components were:

| component | allocated | retained |
|---|---:|---:|
| variable diffusion solver | 260.4 MiB | 36.3 MiB |
| variable diffusion operator | 169.6 MiB | 25.1 MiB |
| momentum-advection workspace | 94.4 MiB | 23.8 MiB |
| pressure operator and solver | 156.8 MiB | 26.2 MiB |

Most of the gap is transient sparse-matrix multiplication and multigrid
construction rather than resizable plain arrays. A direct coordinate-format
assembly of the variable-diffusion stiffness matrix was also tested and
rejected: it increased operator allocation from 169.6 to 196.2 MiB and needed
three additional retained coordinate buffers.

Further material reduction requires APIs that separate symbolic sparse and
multigrid construction from numerical refresh, or topology-keyed symbolic
objects with a strict memory budget. The low topology-reuse rate makes an
unbounded solver cache unsuitable for this Stokes-3 mesh evolution.

## Matrix-free follow-up: short-run diagnosis

### Scope

The matrix-free geometric-preconditioning follow-up was profiled only through
`t=0.1`, as requested. The new trace reaches `t=0.1` in 42 steps. For an exact
before/after comparison, the table below uses the first 41 steps of both
traces: the historical trace was recorded through `t=1`, so its step 42 is a
full CFL step past `t=0.1`, whereas the short trace clips step 42 to end exactly
at `t=0.1`.

The compared 41 steps contain eight topology changes. Active block keys,
cell/face counts, AMR actions, and maximum level agree at every step. The
maximum physical-time difference is `2.568e-11`.

### Failed compact refresh schedule

The first attempt precomputed every scalar contribution to every Galerkin
entry. This appeared to turn coefficient refresh into an allocation-free loop,
but the representation was not compact:

- pressure stored 54,086,426 contribution targets;
- diffusion stored 17,063,458 contribution targets;
- the pressure preconditioner retained about 892 MiB;
- the diffusion preconditioner retained about 173 MiB; and
- complete setup state retained about 1.1 GiB.

Its `t=0.1` measured-stage time was 23.350 s. Topology reconstruction alone
took 13.042 s and allocated 14.258 GiB; pressure hierarchy refresh took
4.517 s. Scalar Julia loops over the expanded products were slower than the
optimized sparse matrix products they replaced. This is the direct cause of
the extreme slowdown observed during the initial implementation.

The contribution schedules were removed. The current implementation retains
the exact factorized fine operators and geometric transfers, while refreshing
intermediate Galerkin matrices with optimized sparse products.

### Coupled-hierarchy coarsening

The first full-vector diffusion hierarchy still had 14 levels on the initial
Stokes-3 mesh:

```text
27576, 7500, 2598, 1431, 1185, 1137, 1119, 682,
549, 540, 484, 247, 226, 113
```

Exact four-sibling coarsening was making almost no progress on several levels
near hanging interfaces. The hierarchy now switches to an irregular,
preconditioner-only aggregation when either velocity component contracts by
less than 25%. The physical fine operator is unchanged. The initial hierarchy
is reduced to eight levels:

```text
27576, 7500, 2598, 1431, 751, 355, 173, 85
```

This removes six Galerkin constructions and makes the bottom dense solve
smaller.

### Exact-mesh result through `t≈0.1`

| metric | grow-cache baseline | factorized/full-vector | speedup | reduction |
|---|---:|---:|---:|---:|
| accounted stage time | 12.906 s | 11.143 s | 1.16× | 13.7% |
| median step | 187.986 ms | 129.863 ms | 1.45× | 30.9% |
| p95 step | 767.201 ms | 827.807 ms | 0.93× | −7.9% |
| mean topology-changing step | 772.085 ms | 745.386 ms | 1.04× | 3.5% |
| mean fixed-topology step | 203.923 ms | 156.976 ms | 1.30× | 23.0% |
| allocated during compared steps | 16.762 GiB | 12.261 GiB | 1.37× | 26.8% |

The complete 42-step short run, including the clipped final step, initializes
in 3.000 s, allocates 1.117 GiB during setup, and retains 129.985 MiB of solver
and workspace state.

| stage, first 41 steps | baseline | current | speedup | reduction |
|---|---:|---:|---:|---:|
| diffusion hierarchy refresh | 3.449 s | 1.679 s | 2.05× | 51.3% |
| pressure operator refresh | 0.495 s | 0.241 s | 2.05× | 51.3% |
| pressure hierarchy refresh | 1.186 s | 1.085 s | 1.09× | 8.5% |
| diffusion Krylov solve | 0.978 s | 0.898 s | 1.09× | 8.1% |
| pressure projection | 0.583 s | 0.690 s | 0.84× | −18.4% |
| topology-dependent rebuild | 4.321 s | 4.695 s | 0.92× | −8.6% |

The pressure solve becomes slower because the exact factorized
`jump' * metric * jump` action has more loop and scratch traffic than one CSC
matrix-vector multiply. The isolated extension benchmark now reports both
paths, plus one pressure and diffusion V-cycle, so this cost cannot be hidden
inside an end-to-end result. Diffusion iteration counts improve or remain
unchanged with the coupled hierarchy; the reduced hierarchy also makes its
solve faster in this trace.

### Remaining bottleneck

Topology-dependent reconstruction remains the largest stage. In the complete
42-step run it takes 4.695 s, accounts for 41.8% of measured stage time, and
allocates 5.521 GiB. It constructs new pressure connectivity, staggered strain
connectivity, advection schedules, and coupled Galerkin matrices because all
of their dimensions and sparsity change with the mesh.

The result is now faster overall and uses substantially less transient memory,
but it does not yet realize the report's all-level matrix-free target.
Eliminating the remaining topology and coefficient-refresh allocation requires
a local rediscretized or BoxMG-style coarse operator. The failed expanded
schedule demonstrates that caching generic sparse multiplication term by term
increases retained memory and runtime.

## Rediscretized block-tree hierarchy

The next prototype constructs a genuine coarse `BlockTreeMesh` at every
multigrid level and discretizes the physical operator directly on it. It keeps
the exact factorized fine-grid actions, but it does not require every level to
be matrix-free: coarse sparse matrices are inexpensive enough to apply and are
useful to the smoother. The essential change is that no intermediate level
forms `P' * A * P`.

For pressure, inverse face density is area averaged onto coincident coarse
faces and the compatible elliptic operator is rebuilt on the coarse mesh. Its
cell transfer is coefficient smoothed and preserves constants. For diffusion,
cell `log(mu)` is volume averaged before reconstructing the thesis's geometric
viscosity mixture at coarse stress locations. Physical staggered density is
area averaged, and the complete coupled `E' W_mu E` operator is discretized on
each coarse tree. The staggered transfer uses tangential face overlap and
linear interpolation between coarse normal-face planes. It preserves dual
volume exactly, including next to a non-periodic normal wall.

### Which leaves are removed?

Two choices were implemented and retained as the `pruning` option:

- `deepest` caps the global deepest generation; and
- `all` simultaneously collapses every complete terminal sibling family,
  after which normal tree balancing restores the 2:1 constraint.

Neither mode removes an arbitrary leaf. A parent replaces its children only
when all four terminal siblings are present. The `all` mode is the default. On
a representative 28-block mixed mesh, the diffusion face hierarchies were

```text
deepest: 3568, 2016, 496, 120
all:     3568,  888, 496, 120
```

The deepest-only hierarchy was stronger for one synthetic high-contrast
right-hand side (19 instead of 28 Krylov iterations), but retained a much
larger second level. In the short Stokes-3 comparison both pruning modes used
the same diffusion iteration counts, while deepest-only made topology rebuild
slower and retained more memory. Coarsening all terminal families is therefore
the better default for this block-tree strategy; `deepest` remains available
for problems where the stronger intermediate correction offsets its cost.

### Exact-mesh `t=0.1` result

The following comparison uses all 42 steps of the clipped `t=0.1` runs. The
block-key sequence and every AMR action are identical, and the maximum physical
time-grid difference is `2.855e-12`.

| metric | Galerkin hierarchy | rediscretized tree | speedup | reduction |
|---|---:|---:|---:|---:|
| accounted stage time | 11.238 s | 9.333 s | 1.20x | 16.9% |
| median step | 129.642 ms | 83.349 ms | 1.56x | 35.7% |
| mean fixed-topology step | 155.144 ms | 105.870 ms | 1.47x | 31.8% |
| mean topology-changing step | 745.386 ms | 716.734 ms | 1.04x | 3.8% |
| allocated during steps | 12.409 GiB | 4.146 GiB | 2.99x | 66.6% |
| diffusion hierarchy refresh | 1.707 s | 0.095 s | 18.00x | 94.4% |
| pressure hierarchy refresh | 1.100 s | 0.243 s | 4.54x | 78.0% |
| diffusion Krylov solve | 0.912 s | 1.370 s | 0.67x | -50.2% |
| pressure projection | 0.701 s | 0.735 s | 0.95x | -4.8% |
| topology-dependent rebuild | 4.695 s | 4.599 s | 1.02x | 2.0% |

Warmed setup time falls from 3.000 s to 2.531 s and setup allocation from
1.117 GiB to 0.885 GiB. Retained state rises from 129.985 MiB to
155.361 MiB because the tree hierarchy stores a physical operator on every
coarse mesh and retains grow-only diffusion-construction scratch.

### Reuse across topology changes

The first rediscretized prototype still rebuilt every coarse tree, solver, and
workspace after each AMR change. The current implementation walks the new
pruned hierarchy until its mesh topology matches an old coarse level, then
retains that level and the complete suffix below it. It refreshes their
restricted density and viscosity, operator values, smoother diagonals, and
bottom factorization in place. The pressure and diffusion solver objects and
package-owned vectors are also retained and resized. The external Krylov
workspace is reconstructed because its basis and residual vectors have the
fine-grid dimension.

All eight topology changes in this trace find a reusable suffix. Seven reuse
five pressure and five diffusion levels; one refinement reuses three of each.
Relative to the otherwise identical tree run without this reuse, the topology
stage falls from 5.989 s and 4.501 GiB to 4.599 s and 3.135 GiB. This is a
23.2% time reduction and a 30.3% allocation reduction for that stage. Total
step allocation falls by 24.8% relative to the no-reuse tree prototype.

This result confirms the original performance intuition only for numerical
refresh. Sparse triple products were the reason coefficient refresh allocated
heavily, and rediscretization removes that cost. They were not the only cost in
the application. A changed fine mesh still requires new pressure connectivity,
staggered strain connectivity, VOF and momentum schedules, projection fields,
and Krylov storage. The unmatched fine prefix of the tree hierarchy also has
to be constructed. Reusing the matching coarse suffix removes enough work to
make topology-changing steps slightly faster than the Galerkin baseline, but
the rebuild remains the largest stage.

The coarse operators are no longer variational projections of the fine
operator. Pressure needs a few more iterations late in the run, and diffusion
uses 18--19 iterations on several coarsened topologies where the Galerkin
hierarchy uses 5--7. As a result, diffusion solve time rises by 50.2% and
pressure projection time by 4.8%, partially offsetting the hierarchy-refresh
saving.

The implementation is matrix-free on the fine-grid Krylov action: pressure is
applied as the compatible incidence/metric factorization and diffusion as
`M_rho x + dt/2 E' W_mu E x`. The new multigrid hierarchy is deliberately
assembled and rediscretized. Its defining property is the absence of `P'AP`,
not the absence of every sparse matrix. Further work should target the quality
of coefficient-aware tree transfers and cache the remaining unmatched-prefix
construction, rather than replacing these small sparse level applications
with scalar loops.

### Source-level profile of the remaining rebuild

Julia's built-in sampling profiler and `Profile.Allocs` allocation profiler
were applied to the first topology-changing step of the default `t=0.1` case.
The exact step was executed once before recording so topology-specific
compilation was excluded. The profiled step changed the mesh at step 7 and the
ordinary stage counter measured 0.487 s, 231.2 MiB, and no compilation in the
topology rebuild. The allocation profiler sampled 5,144 rebuild allocation
events at rate 0.002. Sampling is used for attribution only; the ordinary
counter remains the source for absolute allocated bytes.

The warmed CPU samples inside the rebuild have the following inclusive
distribution:

| path | share of rebuild samples |
|---|---:|
| two-phase diffusion workspace | 89.0% |
| coupled staggered diffusion operator | 65.8% |
| shear derivative construction | 52.9% |
| tree diffusion preconditioner | 17.9% |
| tree pressure preconditioner | 7.2% |
| pressure operator | 3.8% |

The entries are inclusive: the diffusion-operator and preconditioner rows are
both contained in the workspace row. Allocation-event attribution is even
more concentrated. The coupled diffusion operator accounts for 82.0% of
sampled rebuild events, and `_vertex_derivative` alone accounts for 76.1%.
Within that routine, repeated spatial candidate searches, sorting, distance
tests, and construction of temporary three-tuples dominate. The remaining
rebuild is therefore no longer primarily sparse pressure-matrix formation.
The next useful target is a persistent, fixed-type shear-patch schedule keyed
by local tree geometry, with unchanged patches reused across adaptation.

The same profile exposed one measurement cost outside the rebuild:
`Base.summarysize((solver, workspace, amr_workspace))` accounts for 28.9% of
active samples in the instrumented step. It computes retained-state size after
every topology change and is absent from production `simulate!`. It must not be
interpreted as AMR or solver time; stage totals already exclude it. The
reproducible source-level profiler is
[`stokes3_hotspot_profile.jl`](stokes3_hotspot_profile.jl).

## Reproduction

Record a trace with:

```bash
STOKES3_PROFILE_CSV=/tmp/stokes3.csv \
  julia --project=. --threads=auto benchmark/stokes3_performance.jl
```

The loop prints `step=<n>, time=<t>` after every completed step. Short runs can
set `STOKES3_PROFILE_FINAL_TIME`, `STOKES3_PROFILE_BASE_CELLS`, and
`STOKES3_PROFILE_MAX_LEVEL`. The rediscretized tree hierarchy is the default.
The `geometric` selector is now a compatibility alias for the same
implementation. The Galerkin columns in this report are retained historical
measurements; reproduce them from the commit preceding removal of that
implementation, then compare its CSV with a current trace.

Compare two full traces with:

```bash
julia --project=. benchmark/compare_stokes3_performance.jl \
  /tmp/stokes3-before.csv /tmp/stokes3-after.csv
```

The comparator fails before printing timings if the step count, block keys,
mesh sizes, maximum level, AMR attempts, or refine/coarsen decisions differ.
