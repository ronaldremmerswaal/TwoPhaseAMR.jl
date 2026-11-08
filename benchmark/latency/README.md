# First-run latency investigation (work plan PR 3.1)

This is a machine-specific investigation, not a performance guarantee. First
set up the benchmark environment as described in [`../README.md`](../README.md). Run each example in a **new** Julia
process with one Julia and one BLAS thread:

```sh
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_HEAP_SIZE_HINT=3G
for example in taylor_green capillary_wave stokes3_wave; do
    julia --heap-size-hint=3G -t 1 --project=benchmark \
        benchmark/latency_breakdown.jl "$example"
done
```

The probe times `using TwoPhaseAMR` with `@time_imports`, example inclusion,
setup including initial AMR, the Stokes initial pressure projection, and two
separate `simulate!` calls. It mirrors the example setup rather than calling
the monolithic `run_example`; the helper's closures consequently have their
own types. `max_level=1`, `dt=0.001` and two steps are the defaults; use
`--max-level=0`, `--dt=...`, or `--two-velocity` (CMDGFM) to change them.
LinearAlgebra is loaded before timing the package to set BLAS threads, so its
own import is excluded. `@timed` compile and recompile times are compiler
accounting, with recompile a subset of compile, not separate costs.

The following single fresh-process samples used Julia 1.12.7, AMD Ryzen 9
5950X, one thread, pre-existing package images and `--trace-compile`. Each
cell is **wall / compile / recompile seconds**. Zeroes are rounded; the
projection phase is a no-op outside Stokes. Times include tracing overhead and
can vary with shared-machine load.

| Phase | Taylor–Green | Capillary (one velocity) | Stokes-3 (one velocity) |
|---|---:|---:|---:|
| `using TwoPhaseAMR` | 0.627 / 0.206 / 0.000 | 0.666 / 0.225 / 0.000 | 0.667 / 0.222 / 0.000 |
| Include example | 0.045 / 0.036 / 0.000 | 0.097 / 0.049 / 0.000 | 0.058 / 0.038 / 0.000 |
| Setup + initial AMR | 18.710 / 18.702 / 0.000 | 41.607 / 41.592 / 0.000 | 47.097 / 47.084 / 0.000 |
| Initial projection | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 2.463 / 2.463 / 0.000 |
| First `simulate!` step | 6.268 / 6.267 / 0.000 | 32.072 / 32.069 / 0.000 | 33.658 / 33.655 / 0.000 |
| Second `simulate!` step | 0.000 / 0.000 / 0.000 | 0.003 / 0.000 / 0.000 | 0.002 / 0.000 / 0.000 |

Allocated MiB for those phases:

| Phase | Taylor–Green | Capillary | Stokes-3 |
|---|---:|---:|---:|
| Package load | 73.37 | 73.39 | 73.37 |
| Include | 0.30 | 5.10 | 0.70 |
| Setup + initial AMR | 3434.54 | 6362.56 | 6570.05 |
| Initial projection | 0.00 | 0.00 | 376.06 |
| First step | 1107.00 | 1905.47 | 1488.34 |
| Second step | 0.01 | 1.55 | 0.92 |

The process wall times including Julia startup, reporting, and the second step
were 26.89, 75.79, and **85.36 s**, respectively. The Stokes command required
by the probe is `julia --heap-size-hint=3G -t 1 --project=benchmark
benchmark/latency_breakdown.jl stokes3_wave` (the sample above additionally
enabled `--trace-compile`). All wall/compile/recompile entries are seconds,
not percentages. In the Stokes `@time_imports` report, the larger entries were
SuiteSparse_jll 185 ms, SparseArrays 147 ms and StaticArrays 93 ms; the
TwoPhaseAMR module itself was 15 ms. The prior oracle's 16.7 s Taylor–Green
and 58.4 s Stokes first-run figures are not directly comparable: it includes examples first and
runs Taylor–Green before Stokes in the same process, so Stokes can reuse its
compilation.

Running that exact Stokes command without tracing produced this phase table;
`/usr/bin/time` reported **84.12 s wall** (the script's own pre-output wall
was 83.134 s):

```text
Phase                       wall (s)    compile  recompile  alloc (MiB)
using TwoPhaseAMR              0.670      0.226      0.000        73.67
include example                0.060      0.040      0.000         0.70
setup + initial AMR           46.114     46.101      0.000      6569.98
initial projection             2.515      2.515      0.000       376.12
first simulate! step          33.429     33.425      0.000      1488.33
second simulate! step          0.002      0.000      0.000         0.92
```

To reproduce the diagnostic runs on Julia 1.12+:

```sh
julia --heap-size-hint=3G -t 1 --project=benchmark \
    benchmark/latency/inference.jl stokes3_wave
julia --heap-size-hint=3G -t 1 --project=benchmark \
    benchmark/latency/inference.jl capillary_wave
julia --heap-size-hint=3G -t 1 --project=benchmark \
    benchmark/latency/invalidations.jl
julia --heap-size-hint=3G -t 1 --project=benchmark \
    --trace-compile=/tmp/stokes-trace.jl benchmark/latency_breakdown.jl stokes3_wave
julia --heap-size-hint=3G -t 1 --project=benchmark \
    benchmark/latency/trace_summary.jl /tmp/stokes-trace.jl
```

The inference probes use the smallest `max_level=0` setup and one step, so
their AMR paths differ from the timings above. They load SnoopCompileCore
before, and SnoopCompile after, recording. `accumulate_by_source` aggregates
specializations of each method; the displayed inclusive inference times
exclude LLVM and overlap across nested methods, so they must not be summed.
The native trace counts emitted top-level `precompile` signatures, not all
inlined or inferred methods, and the owner labels are a first-token heuristic.

## Inference and invalidations

For Stokes, the profiler recorded 49,054 inference nodes and 48,624 distinct
method instances. The top 20 source methods are below. `#...#` entries are
compiler-generated keyword bodies; entries in the same call chain overlap.
Inferred branches need not all execute at runtime.

| Rank | Method (file:line) | Inclusive inference (s) |
|---:|---|---:|
| 1 | `Main.inference_workload` (`inference.jl:14`) | 30.156 |
| 2 | `Main.setup_case` (`cases.jl:63`) | 15.371 |
| 3 | `Main.#setup_case#19` (`cases.jl:63`) | 15.371 |
| 4 | `Main.one_step!` (`cases.jl:110`) | 13.127 |
| 5 | `simulate!` (`flow_simulation.jl:433`) | 13.125 |
| 6 | `#simulate!#849` (`flow_simulation.jl:433`) | 13.123 |
| 7 | `simulate!` (`simulate.jl:20`) | 13.106 |
| 8 | `#simulate!#836` (`simulate.jl:20`) | 13.099 |
| 9 | `step!` (`single_phase/stepping.jl:67`) | 8.854 |
| 10 | `#step!#820` (`single_phase/stepping.jl:67`) | 8.850 |
| 11 | `#FlowSimulation#848` (`flow_simulation.jl:380`) | 7.392 |
| 12 | `_flow_simulation_from_keywords` (`flow_simulation.jl:249`) | 7.365 |
| 13 | `#_flow_simulation_from_keywords#840` (`flow_simulation.jl:249`) | 7.358 |
| 14 | `FlowSimulation` (`flow_simulation.jl:380`) | 7.075 |
| 15 | `IncompressibleNavierStokesSolver` (`system.jl:106`) | 6.190 |
| 16 | `#IncompressibleNavierStokesSolver#800` (`system.jl:106`) | 6.183 |
| 17 | `_transport_substep!` (`single_phase/stepping.jl:15`) | 5.330 |
| 18 | `advect!` (`staggered_update.jl:256`) | 5.329 |
| 19 | `#advect!#169` (`staggered_update.jl:256`) | 5.328 |
| 20 | `_dispatch_convection!` (`staggered_fluxes.jl:410`) | 5.322 |

For capillary, 48,788 inference nodes and 48,471 distinct method instances were
recorded. The corresponding top 20 inclusive inference times, again excluding
LLVM and overlapping across nested methods, were:

| Rank | Method (file:line) | Inclusive inference (s) |
|---:|---|---:|
| 1 | `Main.inference_workload` (`inference.jl:14`) | 28.796 |
| 2 | `Main.one_step!` (`cases.jl:110`) | 16.134 |
| 3 | `simulate!` (`flow_simulation.jl:433`) | 16.131 |
| 4 | `#simulate!#849` (`flow_simulation.jl:433`) | 16.129 |
| 5 | `simulate!` (`simulate.jl:20`) | 16.109 |
| 6 | `#simulate!#836` (`simulate.jl:20`) | 16.102 |
| 7 | `Main.setup_case` (`cases.jl:23`) | 14.116 |
| 8 | `Main.#setup_case#8` (`cases.jl:23`) | 14.116 |
| 9 | `step!` (`single_phase/stepping.jl:67`) | 9.419 |
| 10 | `#step!#820` (`single_phase/stepping.jl:67`) | 9.415 |
| 11 | `#FlowSimulation#848` (`flow_simulation.jl:380`) | 8.295 |
| 12 | `_flow_simulation_from_keywords` (`flow_simulation.jl:249`) | 8.261 |
| 13 | `#_flow_simulation_from_keywords#840` (`flow_simulation.jl:249`) | 8.253 |
| 14 | `FlowSimulation` (`flow_simulation.jl:380`) | 6.803 |
| 15 | `_transport_substep!` (`single_phase/stepping.jl:15`) | 5.903 |
| 16 | `advect!` (`staggered_update.jl:256`) | 5.902 |
| 17 | `#advect!#169` (`staggered_update.jl:256`) | 5.900 |
| 18 | `_dispatch_convection!` (`staggered_fluxes.jl:410`) | 5.895 |
| 19 | `_dispatch_interface_convection!` (`staggered_fluxes.jl:343`) | 5.884 |
| 20 | `Polyester.batch` (`batch.jl:289`) | 5.852 |

Capillary likewise showed one concrete type of each of the three target
containers, 2,355 keyword-like instances under the same broad heuristic, and
1,439 instances stale after the analyzer was loaded.

The Stokes recording contained one concrete solver type, one concrete workspace type
and one concrete flow type in inferred signatures. The solver has 21 type
parameters, the workspace only two, and the flow five. A single case therefore
does **not** establish that a combinatorial explosion of solver types is the
dominant cost. The profiler found 2,404 keyword-*like* method instances using
a broad name heuristic that also counts closures; it is not a count of unique
keyword sorters. The count of actual `Core.kwcall` signatures in the native
trace below is narrower. The inference tree had 1,495 stale instances when
examined *after* importing SnoopCompile; these can include invalidations caused
by loading the analysis package, so they are not a package-load invalidation
count.

`@snoop_invalidations using TwoPhaseAMR`, after loading LinearAlgebra and
SnoopCompileCore but before loading SnoopCompile, recorded **1,059 unique
invalidated MethodInstances** (excluding method-table-cache-only entries and
Core.Compiler), in 70 invalidating trees. This includes dependencies loaded
alongside TwoPhaseAMR, not just methods owned by it. The largest trees, ranked
by reachable nodes including roots, were:

| Inserting method / package | Reachable nodes |
|---|---:|
| `any(f::Function, ::StaticArray; dims)` / StaticArrays | 720 |
| `convert(::Type{T}, ::Despec4)` / CommonWorldInvalidations | 373 |
| `==(::Nothing, ::Despec1)` / CommonWorldInvalidations | 153 |
| `step(::ORDespec4)` / CommonWorldInvalidations | 115 |
| `>(::Despec4, ::Despec1)` / CommonWorldInvalidations | 93 |
| `(::Colon)(::Integer, ::StaticInt)` / Static | 88 |

Trees can overlap, so their counts cannot be summed into the unique-victim
count. Despite the nonzero load-time invalidations, the separately timed
example phases rounded to 0.000 s of recompilation; this experiment does not
show invalidation-driven recompilation as the main first-step cost.

| Native trace count | Taylor–Green | Capillary | Stokes-3 |
|---|---:|---:|---:|
| All emitted signatures | 395 | 639 | 653 |
| First token `Base` / `Polyester` / `TwoPhaseAMR` / `Core` | 202 / 61 / 24 / 23 | 310 / 145 / 61 / 27 | 308 / 150 / 61 / 37 |
| Mentions `TwoPhaseAMR` anywhere in signature | 156 | 351 | 372 |
| `Core.kwcall` signatures (of which mention TwoPhaseAMR) | 23 (21) | 26 (24) | 36 (33) |
| Tagged `# recompile` | 2 | 4 | 4 |

The Stokes trace mentioned the solver on 19 lines, workspace on 18 and flow on
7. Balanced-brace extraction found respectively one, one and two distinct
*textual* type forms (the flow count includes a generic form). Across all
three case traces there were three solver, one workspace and four flow
textual forms in total. These are not complete counts of compiler
specializations: native tracing omits inlined methods and does not measure
their inference cost.

## Ranked causes and proposed follow-up

1. **First-use compilation dominates.** Stokes setup plus initial projection
   spent 49.547 of 49.560 s in compilation; its first step spent 33.655 of
   33.658 s. The second step took 0.002 s. These are measured phase totals,
   not a claim that physical AMR or the linear solvers are intrinsically slow.
2. **A large, layered specialization graph is compiled on first use.** The
   Stokes inference graph contains about 48,600 distinct method instances;
   the `simulate!` and `FlowSimulation`/solver keyword chains have large
   *overlapping* inclusive costs. The trace also contains many Polyester
   generated closures. In particular, the compiler inferred the `::Nothing`
   transport branch even for the two-phase example. [INFERENCE] A narrower
   typed constructor/step boundary may reduce inference of irrelevant
   branches, but this has not yet been tested.
3. **Configuration round-trips are a plausible contributor, not an isolated
   measurement.** `options.jl` expands typed options into a wide NamedTuple;
   `flow_simulation.jl` forwards it through keyword constructors and
   `independent_step.jl` reconstructs a similar NamedTuple. Stokes emitted 36
   distinct native `Core.kwcall` signatures, 33 mentioning this package. The
   independent visualization path was not timed by this probe. The 21-parameter
   solver is a concern across configurations, but one concrete solver type
   was seen per measured case, so parameter count alone is not a demonstrated
   cause of an explosion.
4. **Dependency invalidations exist, but are not the observed dominant
   latency.** Loading the dependency graph invalidated 1,059 unique instances,
   predominantly in the trees listed above. The timed phases nevertheless
   reported essentially no recompilation; package load was under 0.7 s in the
   baseline runs. Resolving dependency invalidations alone cannot explain an
   84 s cold Stokes run.

**PR 3.2, low-risk code changes:** replace `_navier_stokes_keywords` and the
constructor's NamedTuple splat with an internal typed configuration path while
preserving the public constructors; reuse that representation for
`_navier_stokes_configuration` instead of unpack/repack where practical.
Introduce function barriers after configuration selection, and experiment with
`@nospecialize` only on cold configuration/diagnostics inputs, never blindly on
the numerical kernels. Examine whether non-hot solver options can share a
concrete representation without erasing types of fields used inside hot
loops. [INFERENCE] A **5–15% Stokes TTFS reduction** (roughly 4–13 s on the
83.943 s sum of baseline phases) is a working target, not a measured gain; the
overlapping 7.392 s constructor-chain inference number cannot be added to
other rows. Benchmark warm stepping and allocations as well as cold latency,
and reject any despecialization that harms numerical-kernel performance.

## PrecompileTools experiment

Experiments used a disposable copy of `src/`, `ext/`, and `Project.toml`, with
PrecompileTools 1.3.4 added **only to the copy**. The helper
`benchmark/latency/precompile_workloads.jl` was copied into the copy's `src/`
as `latency_workloads.jl` and included before its module's final `end`; a
separate environment develops that copy. `Pkg.precompile("TwoPhaseAMR"; strict=true, timing=true)`
generates a package image, and the same phase probe is then run in fresh
processes with that environment. The probe prints the package source path so
the copy can be distinguished from the checkout. Workload selection uses
`TWOPHASE_LATENCY_WORKLOAD`; change the copied workload file between selections,
because changing an environment variable alone does **not** invalidate an
existing image. Use `JULIA_NUM_PRECOMPILE_TASKS=1`, `JULIA_NUM_THREADS=1`,
`JULIA_HEAP_SIZE_HINT=3G` and `OPENBLAS_NUM_THREADS=1` for the precompile
coordinator and its worker.

For example, after creating the copy, adding
`PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"` to its `[deps]`,
and including the helper as described, use a separate environment `ENV_DIR`:

```sh
export JULIA_PKG_PRECOMPILE_AUTO=0 TWOPHASE_LATENCY_WORKLOAD=all
mkdir -p "$ENV_DIR"
julia --heap-size-hint=3G -t 1 --project="$ENV_DIR" -e \
    'using Pkg; Pkg.develop(path=ENV["COPY_DIR"]); Pkg.add("SpecialFunctions")'
julia --heap-size-hint=3G -t 1 --project="$ENV_DIR" -e \
    'using LinearAlgebra; BLAS.set_num_threads(1); using Pkg; Pkg.precompile("TwoPhaseAMR"; strict=true, timing=true)'
julia --heap-size-hint=3G -t 1 --project="$ENV_DIR" \
    benchmark/latency_breakdown.jl stokes3_wave
```

Set `COPY_DIR` to the absolute path of the copy and export it before these
commands; use `taylor_green` for the other measured after-case.

The tiny `tg`, `one`, and `two` workloads construct and execute one
`simulate!` step for single-phase, one-velocity and CMDGFM respectively, with
diagnostics; `all` includes all three. `one_amr` additionally forces an
initial AMR pass and calls the topology rebuild after a step. `tg_stable`
constructs the flow and calls `stable_timestep` instead of stepping.

TTFS in the table is the **sum of the timed import, example include, setup,
initial projection, and first-step phases**, excluding Julia startup, harness
overhead and the second step. A dash means that combination was not timed.
Package-image time is the TwoPhaseAMR line from the serial precompile, with
dependencies already cached; it is not the time for a clean depot.

| Julia | Package-image workload | Image time (s) | TG TTFS (s) | Stokes TTFS (s) | Result |
|---|---|---:|---:|---:|---|
| 1.12.7 | Stock, no workload | 4.596 | 25.202 | 82.788 | Success |
| 1.12.7 | `tg` | 24.878 | 3.203 | 61.221 | Success |
| 1.12.7 | `one` | 53.582 | — | 20.261 | Success |
| 1.12.7 | `two` | 61.736 | — | — | Success |
| 1.12.7 | `one_amr` | 56.798 | — | — | Success |
| 1.12.7 | `tg_stable` | 20.979 | — | — | Success |
| 1.12.7 | `all` | 69.248 | 3.442 | 19.511 | Success |
| 1.13.2 | Stock, no workload | — | 23.956 | 66.206 | Success |
| 1.13.2 | `one_amr` | 38.8 | — | — | Success |
| 1.13.2 | `tg_stable` | 17.0 | — | — | Success |
| 1.13.2 | `all` | 52.4 | 2.705 | 16.097 | Success |

These TTFS comparisons are all untraced, separate-process samples on a shared
machine, not statistically controlled speedup estimates. The earlier phase
breakdown and native trace table use separate traced samples.
Julia 1.13.2 emitted a warning that the copied experiment manifest had been
resolved under 1.12.7; its results should be repeated with a 1.13-resolved
manifest before treating them as a release gate.

The earlier `claudit:audit/audit-perf.md` audit reported Julia 1.12.7 SIGSEGVs
in LLVM package-image emission for `stable_timestep(flow)` and for natural
two-phase `simulate!`/AMR-rebuild/diagnostic workloads; its
`stable_timestep` failure was reported twice. This investigation's smaller
`tg_stable`, `one_amr`, and `all` runs each completed on both 1.12.7 and
1.13.2, so they did **not** reproduce that report or crash on 1.13.2. The
three individual `tg`/`one`/`two` modes were not timed separately on 1.13.2;
all three ran successfully as part of `all`. Different workload composition
may matter; success here is not proof that the earlier crash is fixed. No
failing MWE can be supplied without a reproduced failure.

**PR 3.3, workload proposal:** precompile small single-phase TG, one-velocity
and CMDGFM constructions and fixed-step `simulate!`, including representative
diagnostics; add `stable_timestep`, an actual AMR pass and a topology rebuild
only after checking package-image reliability on each supported Julia.
For this machine and these exact 1.12.7 fixtures, `all` changed TG TTFS from
25.202 to 3.442 s (−86.3%) and Stokes from 82.788 to 19.511 s (−76.4%),
while increasing the isolated package-image build from 4.596 to 69.248 s.
The example closures and the AMR/diagnostic mix do not exactly match the tiny
workload; Stokes construction and stepping still compile. [INFERENCE] This
mismatch likely contributes to the residual latency. Similar reductions are a
reasonable target for the eventual workload, not a promise for other Julia
versions or cases; measure both image build/load cost and cold examples in CI.

## After PR 3.3 (PrecompileTools workload)

`src/precompile.jl` precompiles small Taylor–Green, Stokes-3-like (interface,
curvature and vorticity AMR criteria, topology rebuild), and capillary
one-velocity and CMDGFM two-velocity runs, including `stable_timestep`,
diagnostics and the simulation summary. The mimetic projection is not part of
the workload. Disable it with the standard PrecompileTools preference
(`PrecompileTools.Preferences.set_preferences!(TwoPhaseAMR, "precompile_workload" => false; force=true)`).

Single fresh-process samples, AMD Ryzen 9 5950X, one Julia and BLAS thread,
`max_level=1`, `dt=0.001`. TTFS is the sum of import, example inclusion,
setup/initial AMR, initial projection and first step (excluding Julia startup).

| Example | Julia 1.12.7 TTFS before → after (s) | first step before → after (s) | Julia 1.13.2 TTFS before → after (s) | first step before → after (s) |
|---|---:|---:|---:|---:|
| Taylor–Green | 27.1 → 2.06 | 6.66 → 0.011 | 23.4 → 1.45 | 5.49 → 0.011 |
| Capillary, one velocity | 80.0 → 6.80 | 33.8 → 1.96 | 65.3 → 4.41 | 28.1 → 1.34 |
| Capillary, CMDGFM | 97.9 → 9.07 | 52.9 → 4.43 | 75.2 → 6.07 | 40.5 → 2.82 |
| Stokes-3 | 85.1 → 3.13 | 35.4 → 0.013 | 70.3 → 2.15 | 27.9 → 0.012 |

Stokes-3 phases on 1.12.7, before → after: load 0.66 → 1.48 s, setup and
initial AMR 46.3 → 1.56 s, initial projection 2.69 → 0.004 s, first step
35.4 → 0.013 s.

Package-image time: 4.6 → 89.7 s on 1.12.7 (+85 s) and 4.6 → 74.0 s on
1.13.2 (+69 s). No crash during image emission on either version.
