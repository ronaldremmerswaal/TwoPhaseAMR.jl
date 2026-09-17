# Benchmarks

## Two-velocity passive transport

Compare the one-velocity WY step with both two-velocity interface policies on
uniform and fixed mixed-level meshes:

```bash
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/two_velocity_transport.jl")'
```

The benchmark reports workspace construction, the projected-liquid auxiliary
Poisson step, and a complete interface plus centred-scalar advection step. It
uses the same interface, scalar fields, time step, and sweep settings for all
three paths, and reports median time, allocated bytes, and allocation count.
Set `TWO_VELOCITY_BENCHMARK_SAMPLES` to change the default 10 samples.
Recorded baseline results and their interpretation are in
[`two-velocity-transport-results.md`](two-velocity-transport-results.md).

## Fixed cut-cell overhead

Compare the new fixed-geometry kernels with the corresponding uncut
single-phase kernels:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/cut_cell_extension.jl")'
```

The benchmark separates geometry construction and refresh, staggered
advection, diffusion setup and application, Poisson setup and application,
and one tree-geometric pressure V-cycle. It reports median time, allocated
bytes, and allocation count for each operation. Set
`CUTCELL_BENCHMARK_SAMPLES` to change the default 20 samples.

## Serial block-locality experiment

Before changing parallel schedules, compare global and block-first traversal
for the compatible staggered two-phase momentum kernel:

```bash
TWOPHASE_LOCALITY_SAMPLES=15 \
  julia --project=. --threads=1 benchmark/staggered_two_phase_locality.jl
```

The experiment separates block-backed centred-cell loops from flat face and
dual-volume loops, verifies bitwise-identical outputs, and records median
time, allocated bytes, and allocation count. The recorded runs and conclusion
are in
[`data-locality-results.md`](data-locality-results.md).

The production-wide serial comparison uses the same kernel suite as the
thread-scaling driver. Its before/after CSV files are stored in
[`data-locality-results`](data-locality-results), alongside the isolated
staggered-advection measurements.

## Shared-memory core-kernel scaling

Before profiling Stokes-3, run every core operation in isolated Julia
processes with controlled thread counts:

```bash
TWOPHASE_THREAD_COUNTS=1,2,4,8 \
  julia --project=. benchmark/thread_scaling.jl
```

The suite covers uniform and mixed-level mesh construction, initialization,
PLIC/LHF/GHF reconstruction, height sampling, halo exchange, centred,
staggered, WY scalar, and compatible momentum advection, pressure-gradient
construction/application, pressure Poisson construction/application/solve,
variable diffusion construction/coefficient refresh/application/solve,
surface tension, gravity, adaptive refine/coarsen with field transfer, and
direct restriction/prolongation at two block sizes. It records median wall
time, allocated bytes, allocation count, checksums, speedup, and parallel
efficiency. Cross-thread checksum mismatches abort the driver.

Set `TWOPHASE_KERNEL_SAMPLES` (default `5`) to control repetitions and
`TWOPHASE_KERNEL_CASE=uniform` or `mixed` to isolate one topology. The design
and literature basis are described in
[`shared-memory-parallelism.md`](../docs/src/shared-memory-parallelism.md).
Set `TWOPHASE_REUSE_RESULTS=true` to rebuild only the combined scaling table
from already completed per-thread CSV files.
The recorded 1/2/4/8-thread run and bottleneck analysis are in
[`threaded-kernel-results.md`](threaded-kernel-results.md).
Its current raw data and independent eight-thread repeat are in
[`threaded-kernel-results-block-local`](threaded-kernel-results-block-local).
The complete less-than-4x audit and optimization research are in
[`parallel-scaling-audit.md`](parallel-scaling-audit.md); its implementation
priorities should be read together with the serial locality experiment above.

After the isolated kernels pass, run the default adaptive Stokes-3 case through
`t=1` in separate 1/2/4/8-thread processes:

```bash
STOKES3_THREAD_COUNTS=1,2,4,8 \
  julia --project=. benchmark/stokes3_thread_scaling.jl
```

The driver forces BLAS to one thread and aborts unless every run has exactly
the same time-step sequence, AMR decisions, complete block-key trace, pressure
and diffusion iteration counts, and multigrid reuse history. Raw per-step
traces and aggregate CSV files are written to `/tmp/stokes3-thread-scaling` by
default; select another directory with `STOKES3_THREAD_OUTPUT`. Set
`STOKES3_THREAD_REUSE_RESULTS=true` to analyze completed traces without
rerunning the simulation. Recorded results and the load-balancing decision are
in [`stokes3-thread-scaling-results.md`](stokes3-thread-scaling-results.md).
The current aggregate tables, full traces, and eight-thread repeat are in
[`stokes3-thread-results-block-local`](stokes3-thread-results-block-local).

## Stokes-3 end-to-end performance

Profile the default Stokes-3 example through `t=1` with a per-step breakdown
of wall time, allocated bytes, allocation count, garbage-collection time,
linear-solver iterations, AMR work, and retained solver state:

```bash
julia --project=. --threads=auto benchmark/stokes3_performance.jl
```

The benchmark separates coefficient construction and multigrid refresh from
the diffusion and pressure Krylov solves. AMR indicator evaluation, field
transfer, and reconstruction of topology-dependent solver state are reported
independently. Set `STOKES3_PROFILE_FINAL_TIME`,
`STOKES3_PROFILE_BASE_CELLS`, or `STOKES3_PROFILE_MAX_LEVEL` for shorter
diagnostic runs. Set `STOKES3_PROFILE_CSV=/path/to/trace.csv` to retain every
metric for every step. The analysis and a recorded reference run are in
[`stokes3-performance-results.md`](stokes3-performance-results.md).
The benchmark defaults to the rediscretized `tree_geometric` pressure and
diffusion hierarchies, matching the Stokes-3 example and both solver modes.
The `geometric` compatibility selector resolves to the same tree hierarchy;
the former Galerkin implementation has been removed.
Its pruning mode is controlled independently by
`STOKES3_PROFILE_PRESSURE_TREE_PRUNING` and
`STOKES3_PROFILE_DIFFUSION_TREE_PRUNING` (`all` by default, or `deepest`).
The pressure and variable-diffusion Krylov actions remain factorized, while
all multigrid levels retain their assembled sparse matrices. The former
`matrix_free_fine` switch was removed after a matched `t=0.1` comparison found
lower memory use but slower fixed-topology steps; see the
[`matrix_free_fine` reassessment](stokes3-performance-results.md#reassessment-of-matrix_free_fine).
The literature review and implementation plan for removing the remaining
sparse pressure/diffusion hierarchy refresh are in
[`matrix-free-geometric-preconditioning.md`](../docs/src/matrix-free-geometric-preconditioning.md).

Compare traces from two implementations with:

```bash
julia --project=. benchmark/compare_stokes3_performance.jl \
  /tmp/stokes3-before.csv /tmp/stokes3-after.csv
```

The comparison aborts unless the complete block-key sequence and every AMR
decision match at every step. It then reports total, topology-changing, and
fixed-topology costs together with a stage-by-stage speedup table.

For source-level attribution of the first topology-changing step, use Julia's
built-in sampling and allocation profilers:

```bash
julia --project=. --threads=auto benchmark/stokes3_hotspot_profile.jl
```

The script first executes the exact step once so compilation is outside the
profile. It then writes separate flat CPU and allocation-stack reports. The
allocation profiler samples events (`0.002` by default); change this with
`STOKES3_HOTSPOT_ALLOCATION_RATE` and select the output prefix with
`STOKES3_HOTSPOT_PREFIX`. It filters the allocation report to stacks below
`_rebuild_topology_dependent_state!`, while retaining the ordinary benchmark's
exact byte and allocation totals.

Compare the two-phase extension with the corresponding original single-phase
kernels for interface/cell advection, reconstruction, staggered momentum
advection, diffusion, and Poisson setup, coefficient refresh, current-
coefficient preconditioner refresh, assembled/factorized operator application,
one multigrid V-cycle, and solve with:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/two_phase_extension.jl")'
```

The script reports uniform and mixed-level cases, median time, allocations,
and the two-phase/single-phase ratio. Setup, operator application, V-cycle, and
solve costs are measured separately.

For a representative end-to-end single-phase workflow, including high-level
setup, low-level stepping, driver overhead, and flat sampling profiles, run:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/single_phase_workflow.jl")'
julia --project=. benchmark/profile_single_phase.jl
```

The benchmark reports setup, low-level stepping, driver overhead, and the
first-step cost both with and without a known workspace `dt`. The profile
script warms compilation first and reports mixed-level solver setup separately
from 200 fixed-mesh time steps.

A recorded run and interpretation are in
[`single-phase-workflow-results.md`](single-phase-workflow-results.md).

`staggered_advection_interface.jl` reproduces the mixed-level LW/interface
convergence comparison for Fromm, minmod, MC, van Leer, superbee, and LW; the
frozen interface-extremum regression; and the linear one-step amplification
investigation for uniform LW, AMR LW, and global AMR Upwind. It reports
interface-adjacent and regular RMS errors separately.

```sh
julia --project=. benchmark/staggered_advection_interface.jl
```

To isolate the cost of augmenting the conservative AMR staggered schedule
with the predecessor metadata required by three-point fluxes, run:

```bash
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/staggered_advection_setup.jl")'
```

The reported two-point baseline builds the same AMR connection graph and
final workspace layout, but skips the three-point predecessor search. It is a
setup-cost comparison, not a comparison of retained workspace sizes.

Reference result (2026-09-15, Julia 1.12.7, one thread, AMD Ryzen 9 5950X,
200 samples):

| Active cells | Two-point setup | Three-point setup | Time ratio | Allocated-byte ratio |
|---:|---:|---:|---:|---:|
| 448 | 0.451 ms | 0.348 ms | 0.77x | 1.041x |
| 1,792 | 1.215 ms | 1.391 ms | 1.15x | 1.036x |
| 7,168 | 5.256 ms | 5.885 ms | 1.12x | 1.034x |

The smallest timing is below half a millisecond and is noise-sensitive. The
two larger cases show a 12--15% setup-time cost and 3--4% more allocated bytes
for predecessor construction. Allocation counts rise by only 12: from 3,706
to 3,718 at 1,792 cells and from 14,480 to 14,492 at 7,168 cells. A compressed
temporary adjacency representation replaced the former per-degree-of-freedom
vectors; before that change, the three-point cases allocated 9,136 and 36,086
objects and 10--11% more bytes at those two sizes.

Run from the package root:

```sh
julia --project -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/benchmarks.jl")'
```

Setup/regridding and time-stepping kernels are measured separately. Cell
advection, one-component staggered advection, and simultaneous two-component
staggered advection are reported independently. Pressure-gradient setup and
application are reported for uniform and mixed-level meshes. Results for the
current implementation and machine are recorded in `results.md`. The
mixed-level case separately measures AMR schedule construction and allocation-
free one- and two-component updates for both upwind and bulk-LW/interface-MC.

The milestone-3 pressure comparison additionally requires
`IncompressibleNavierStokes.jl`. Run it in a temporary environment so the
reference package does not become a runtime dependency:

```sh
julia --project -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add(["BenchmarkTools", "IncompressibleNavierStokes"]); include("benchmark/poisson_comparison.jl")'
```

It compares a 64×64 periodic Float64 single block with the same deterministic
broadband right-hand side, zero initial guess, relative tolerance, Julia
process, CPU thread count, and warm-median protocol. Setup is excluded. The
iterative `psolver_cg` reference is used because, unlike the package's FFT
direct solver, it exposes the tolerance/iteration comparison required by
`DEVELOPMENT.md`.

Unpreconditioned and geometric-MG-preconditioned Krylov.jl CG can be compared
without extra dependencies:

```sh
julia --project=. benchmark/pcg_multigrid.jl
```

The old standalone-multigrid measurements in
[pcg-multigrid-results.md](pcg-multigrid-results.md) predate the separation of
Krylov iteration and preconditioning and are retained as historical results.

The nonsymmetric staggered implicit-diffusion comparison is reproduced with:

```sh
julia --project=. benchmark/bicgstab_multigrid.jl
```

The old standalone-multigrid results in
[bicgstab-multigrid-results.md](bicgstab-multigrid-results.md) are historical.

Compare all applicable preconditioners on the same graded pressure and
staggered-diffusion systems in a temporary environment:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add(["AlgebraicMultigrid", "IncompleteLU", "Krylov", "LimitedLDLFactorizations"]); include("benchmark/preconditioner_comparison.jl")'
```

The table reports setup time, warm median solve time, Krylov iterations, final
strong residual, and retained preconditioner bytes. Pressure uses CG except
for ILU, which is paired with BiCGSTAB because ILU need not be SPD. Staggered
diffusion uses BiCGSTAB throughout. Its named AMG, ILU, and incomplete-
Cholesky options are built from the symmetric stable core while the exact
corrected matrix remains the Krylov operator. One recorded run is in
[preconditioner-results.md](preconditioner-results.md).
The same run compares BiCGSTAB with restarted GMRES and FGMRES using the
geometric diffusion preconditioner.

Measure adaptive transfer kernels and full topology changes with:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/adaptivity.jl")'
```

This reports direct cell restriction/prolongation separately from refinement
and coarsening of a representative multilevel mesh carrying multiple cell
fields, generic face fields, and a staggered velocity. A recorded run is in
[adaptive-refinement-results.md](adaptive-refinement-results.md).

The end-to-end usefulness study advances an analytical, forced localized
vortex on successively deeper adaptive meshes and compares its deepest mesh
with a nearly equal-cell-count uniform mesh:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/amr_usefulness.jl")'
```

It reports velocity error, observed max-level convergence, active cells,
divergence, and full setup-plus-simulation cost. Recorded values and the
manufactured equations are in
[amr-usefulness-results.md](amr-usefulness-results.md).

Inspect the grid, operator, and transfer complexity of the corresponding
pressure and staggered geometric hierarchies with:

```sh
julia --project=. benchmark/hierarchy_complexity.jl
```

This also distinguishes cumulative setup allocation from retained hierarchy
storage. The interpretation and the recorded reference-mesh values are in the
[implicit-diffusion solver review](../docs/src/implicit-diffusion-solver-review.md).
