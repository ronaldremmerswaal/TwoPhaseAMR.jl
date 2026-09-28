# CMDGFM pressure reuse

This implementation leaves CMDGFM's equations, pressure inverse metric,
BiCGSTAB stopping conditions and jump ILU unchanged. Measurements use the
**corrected inactive-velocity NaN implementation**, not the earlier dormant
velocity baseline in [PRESSURE_FINDINGS.md](PRESSURE_FINDINGS.md).

Final normal-optimization pressure-stage measurements to `t=0.5` (isolated
Julia 1.11.9, one Julia/BLAS thread) give
**2.062 s / 168.538 MB for CMDGFM versus 1.684 s / 119.540 MB for one velocity**:
factors 1.22 in time and 1.41 in allocated bytes. These include coefficient/hierarchy refresh
and projection, on each method's default evolving mesh. Ordinary projection
calls still allocate more than scalar CG; this is not a zero-allocation claim.

All 1,220 new reuse checks and 88 Poisson/KH checks pass. The final full suite
completed with **16,702 passes and seven baseline failures**. Five are small
allocation failures reproduced on the pre-pressure commit in the isolated
Julia 1.11 runtime; the two numerical failures also reproduce on that snapshot.
The installed Julia 1.12 startup/compiler crashes are not fixed by this work.

## Changes and invalidation

- Cache mesh incidence, pressure component gauges, incident-face lists,
  vertex patches and the reduced-pressure slice. Invalidate on block-key
  changes or solid-geometry revision, not just array dimensions.
- Construct CMDGFM's incidence directly in CSC form after AMR. Its two
  signed face-area entries are known in advance; no general triplet
  conversion is needed. The original shared incidence builder remains the
  independent reference for this change.
- Refresh density coefficients every projection. Keep the inverse metric,
  coefficient-refresh map, face-data storage and elliptic scratch.
- For compatible uncut flows, share the already-refreshed scalar pressure
  operator and matching multigrid hierarchy used by balanced forces. Never
  share the one-velocity cut-cell `A²/rho` operator with CMDGFM's `A/rho`.
  Custom hierarchy settings must match before hierarchy sharing is allowed.
- Assemble only the two common-gradient rows used by each inference term,
  directly from `sqrt(kappa) K sqrt(kappa) I`. The full sparse-product path
  remains the independent reference; **K is not modified**.
- Refill the coupled CSC buffers even when the active jump set changes.
  No fixed jump pattern is assumed. Assembly views, like other workspace
  scratch, remain valid only until the next assembly; copy snapshots.
- Retain the two coupling blocks, jump block and their triplet/sort buffers.
  Jump equations are emitted in row order; counting-sort by column combines
  duplicate coefficients in that same order and preserves explicit zeros.
  This changes storage only, not the matrix sent to ILU.
- Reuse transverse stencil/weight vectors and the resolved-face lists. Their
  contents, PLIC normals, beta and inference donors are recomputed each step;
  no moving-interface coefficient or stencil selection is frozen.
- Replace the extra cell-adjacency and geometric vertex-object lists with
  incidence columns and compact endpoint-to-face connectivity. Reuse these
  arrays after AMR. Component gauges still walk every open-face component,
  including disconnected solids and isolated cells; no connectivity or
  interface-topology assumption is introduced.
- Reuse RHS, cell-flux and gradient scratch. Compute only the mixture
  constraint needed by CMDGFM; avoid constructing an unused centred normal
  constraint and avoid allocating `matrix-matrix'` for its asymmetry scalar.
- Put solve/recovery behind a typed function barrier on the actual assembled
  system. The cached operator may change concrete type at AMR; dispatch once
  instead of allocating boxed values in every subsequent face/cell loop.
- Bind finalized connectivity tables without reassigning variables captured
  by the incident-face accessor. Allocation profiling found that Julia's
  boxed capture allocated 1.64 MB per stencil call even with pooled vectors.
  Read compact vertex ranges directly, avoiding escaping endpoint views.
- A new gauge wrapper after AMR retains its full-cell input/output vectors,
  coefficient snapshot and index/key arrays. Only their contents and sizes
  are reset; topology and numerical-validity decisions precede this reset.
- Retain the intrinsic-flow pressure/Krylov workspace across AMR. A standalone
  pressure hierarchy keeps a coefficient snapshot, so aliasing an in-place
  operator cannot hide a necessary numerical refresh. Jump-ILU reuse still
  requires identical numerical jump matrices, not merely identical sparsity.
- Check a fitted initial guess against the true coupled residual. For zero
  or previous pressure, use the **current, unchanged jump ILU** to fit
  `J xi = rhs_jump - jump_pressure*p`. Accept only finite guesses that reduce
  both residual L2 and the maximum of the fractional-volume and jump defects.
  Otherwise start from zero. `project!(...; warm_start=false)` preserves the
  zero-start reference. No equation, tolerance, metric or ILU is altered.

## Measurements obtained so far

GB/MB below are decimal allocated bytes, not resident memory. Initial AMR
and initial projection are excluded from the measured advance. The two models
have different physical AMR histories, so whole-run comparisons are not
matched-mesh pressure-kernel comparisons.

| Stokes3 advance to `t=0.5` | Wall time (s) | Allocation (GB) | Steps / AMR | Final faces |
| --- | ---: | ---: | ---: | ---: |
| One velocity, corrected NaN baseline, native / BLAS 16 | 19.182 | 11.242 | 206 / 45 | 12,504 |
| CMDGFM before pressure reuse, native / BLAS 16 | 49.315 | 78.977 | 219 / 79 | 9,016 |
| CMDGFM first reuse stage, native / BLAS 16 | 40.241 | 59.617 | 219 / 79 | 9,016 |
| CMDGFM donor-row and buffer reuse, native / BLAS 1 | 39.826 | 57.540 | 219 / 79 | 9,016 |
| Same final implementation, generic / `-O1` / forced bounds / BLAS 16 | 104.328 | 57.596 | 219 / 79 | 9,016 |
| One velocity, generic / `-O1` / forced bounds / BLAS 1, sampled profile | 42.957 | 11.300 | 206 / 45 | 12,504 |
| CMDGFM guarded guess, before direct incidence, same configuration | 101.549 | 57.465 | 219 / 79 | 9,016 |
| CMDGFM guarded guess and direct incidence, before typed solve barrier, same configuration | 69.822 | 57.303 | 219 / 79 | 9,016 |

Do not compare the `-O1` time against native optimization. The BLAS-1 native
sample is recorded explicitly, not silently pooled with BLAS-16 samples.
All completed CMDGFM advances had converged final projections. Repeated Julia
1.12 native timing remains incomplete because of intermittent runtime/compiler
crashes (including one system-image restoration crash before package import).
Minimal-compilation reference checks and the generic, bounds-checked full
Stokes3 run completed. CPU affinity did not eliminate the crashes.

| Frozen pressure work | Before reuse | After reuse |
| --- | ---: | ---: |
| Initial 27,576-face assembly allocation | 162.194 MB | 3.802 MB |
| Initial assembly time | 51.35 ms, including 8.22 ms GC | 3.34 ms (native / BLAS 1) |
| Late 9,016-face assembly allocation | fresh-reference path, see probe | 1.560 MB |
| Late cached assembly time | — | 1.21 ms (native / BLAS 1) |
| Same late mesh: scalar coefficient + GMG refresh | — | 0.84 ms / 0.398 MB |
| Coupled CSC refill | fresh buffer allocation | 0 allocated bytes |
| Matrix-asymmetry diagnostic | sparse difference allocation | 0 allocated bytes |
| Apply pressure GMG, jump ILU, full block preconditioner | already allocation-free | still 0 allocated bytes |

The late cached substage breakdown is in `pressure_probe.jl`, emitted as
`CACHED_ASSEMBLY`. Its original `ASSEMBLY` rows deliberately retain the fresh
reference decomposition; only `whole_assembly_check` there uses production
cached assembly. Do not sum that check with the reference rows.

The whole advance is **not yet comparable** to one velocity: the remaining
gap must be attributed rather than assumed to be pressure. In particular,
the NaN correction changed CMDGFM from 32 to 79 AMR updates, exposing the cost
of phase-aware transfer and topology rebuilds. The old AMR history is not an
acceptable performance target because its coarsening was wrong.

The matched-configuration profiles now put the total pressure stage at
approximately **11.94 s for CMDGFM versus 8.03 s for one velocity** (1.49×).
Projection alone is 11.19 s versus 7.17 s; coefficient/hierarchy refresh is
0.76 s versus 0.86 s. CMDGFM assembly is now 1.32 s, versus 34.25 s before
direct incidence in the same compiler configuration. These are sampled
whole-trajectory estimates, not matched-mesh or exact region timers. The
remaining total wall-time gap includes 33.1 s in CMDGFM AMR/adaptation versus
14.7 s for one velocity, with 79 versus 45 topology changes. Allocation
attribution is still required: the 57 GB whole advance is not 57 GB of pressure
allocation.

### Historical exact pressure-stage attribution, Julia 1.12

[pressure_stage_trace.jl](pressure_stage_trace.jl) mirrors the example's
force-free stage wrapper, with actual transport, diffusion, forces, AMR and
diagnostics. It separately times pressure refresh and projection at each
step. All runs below have zero compilation inside those measured regions.
Their step/AMR/final-face histories match `simulate!`. Because the wrapper is
instrumented and skips the zero external-force loop, its **whole-advance**
time/bytes must not replace the production advance table above.

| Exact pressure work to `t=0.5`, generic `-O1` / forced bounds / BLAS 1 | One velocity | CMDGFM before typed solve barrier | CMDGFM with typed solve barrier |
| --- | ---: | ---: | ---: |
| Coefficient/GMG refresh time | 0.840 s | 0.760 s | 0.737 s |
| Projection time | 6.852 s | 10.350 s | 9.663 s |
| Total pressure time | 7.692 s | 11.111 s | 10.400 s |
| Refresh allocation | 118.471 MB | 110.177 MB | 110.159 MB |
| Projection allocation | 0.737 MB | 2,802.098 MB | 1,163.075 MB |
| Ordinary projection allocation, median | 0.003616 MB | 9.051 MB | 1.830 MB |
| Post-AMR projection allocation, median | 0.003616 MB | 15.143 MB | 9.454 MB |
| Ordinary iterations, median/p95/max | 11 / 13 / 13 CG | 8 / 11.1 / 18 BiCGSTAB | 8 / 11.1 / 18 BiCGSTAB |
| Post-AMR iterations, median/p95/max | 11 / 13 / 14 CG | 8 / 14.1 / 16 BiCGSTAB | 8 / 14.1 / 16 BiCGSTAB |

CMDGFM accepted 183 guarded guesses in 219 steps. At this intermediate stage,
the pressure timing ratio was 1.35×, but the total pressure-allocation ratio
was still 10.7×: the allocation objective was not achieved. Unlike one velocity,
CMDGFM charges its new mesh/gauge/vertex cache to the first projection after AMR;
the scalar solver's mesh/operator construction is charged to topology rebuild.
That attribution difference must be kept visible, not used to disguise the
remaining allocation. At that stage ordinary coupled assembly still allocated
temporary stencils, sparse triplets and coupling blocks.

The next candidates were reusable triplet/CSC work for the changing
jump blocks, reusable stencil storage, and compact/reused vertex connectivity.
These must preserve the independent fresh matrix/action/RHS gates, including
changed jump sets and cut/periodic/coarse–fine configurations. Keep ILU and K
unchanged. Separate pressure-related topology construction for **both** models
when assessing total pressure allocation across AMR.

### Pooled assembly and runtime validation

The candidates above are now implemented. All **1,057** intermediate checks passed
under Julia 1.12.7 / generic `-O1` / forced bounds. This includes 502 compact
connectivity/disconnected-gauge checks and 56 pooled-CSC checks in addition to
the previous 499 reference/solution checks. The following Stokes3 benchmark
aborted in height-function ghost reconstruction with an invalid donor index.
The subsequent isolated-runtime validation below completed.

The runtime investigation also reproduced a system-image restoration crash
in a package-free `julia -e 'println(VERSION)'` launch. Thus startup failures
are independent of the pressure changes; that does **not** establish the
cause of the later invalid donor index. Avoiding global package-cache builds
did not resolve all failures.

An isolated Julia 1.11.9 binary was downloaded into
`/tmp/cmdgfm-julia-validation.caGIcA`, with its archive SHA-256 checked against
the [official release list](https://julialang.org/downloads/oldreleases/):
`b36363356d7a05eaf8b7b9e7a91c710f6bd3d2940be4d4e6d14b9a9f2927de35`.
Its depot path is the temporary directory followed by the existing dependency
source depot. The repository manifest's PrecompileTools 1.3.4 requires Julia
1.12, so a separate environment was resolved in the temporary directory,
developing the live repository source. IncompleteLU 0.2.1 and Krylov 0.10.9
remain at their original versions; the temporary environment selects compatible
precompilation support. The installed Julia and repository project/manifest
have not been replaced or resolved. Timings must not be pooled with Julia 1.12.

Both `t=0.5` traces completed in Julia 1.11.9, with the same histories as before:
CMDGFM 219 steps / 79 AMR / 9,016 final faces; one velocity 206 / 45 / 12,504.
All **1,220** final focused checks pass, including the stencil allocation gate
and retention of pressure-preconditioner buffers across uniform/mixed AMR.

| Exact pressure stage, Julia 1.11.9 / generic `-O1` / forced bounds / BLAS 1 | One velocity (final comparison) | CMDGFM pooled assembly | CMDGFM unboxed traversal, before gauge-buffer reuse | CMDGFM final |
| --- | ---: | ---: | ---: | ---: |
| Refresh time | 2.349 s | 2.021 s | 2.015 s | 1.995 s |
| Projection time | 17.311 s | 22.304 s | 22.123 s | 22.196 s |
| Total pressure time | 19.660 s | 24.324 s | 24.138 s | 24.191 s |
| Refresh allocation | 119.390 MB | 111.150 MB | 111.152 MB | 111.159 MB |
| Projection allocation | 0.224 MB | 294.284 MB | 80.080 MB | 57.488 MB |
| Total pressure allocation | 119.615 MB | 405.434 MB | 191.232 MB | 168.647 MB |
| Ordinary projection allocation, median | 0.001088 MB | 1.204 MB | 0.256 MB | 0.256 MB |
| Post-AMR projection allocation, median | 0.001088 MB | 1.186 MB | 0.447 MB | 0.201 MB |

Final pressure refresh + projection is **1.23× the one-velocity time and 1.41×
the allocation**, on their respective default Stokes3 trajectories. This is
much closer than the original pressure gap, not identical allocation: ordinary
CMDGFM calls still allocate more than scalar CG, including the unchanged jump
ILU factorization and its bookkeeping. Iteration distributions are unchanged
by the storage changes: ordinary median/p95/max 8/11.1/18 BiCGSTAB versus
11/13/13 CG; post-AMR 8/14.1/16 versus 11/13/14. No compilation occurred in
these pressure regions; all projections converged, with 183 accepted guarded
guesses. Both full histories are unchanged.

The final manual advance still allocates 53.284 GB for CMDGFM versus 13.477 GB
for one velocity. **Do not assign that total to projection**: only 0.169 GB
and 0.120 GB respectively occurred in the exact pressure regions. Further
whole-algorithm allocation work must distinguish other stages, especially
AMR, from pressure-related topology construction. The scalar solver's
pressure topology construction occurs during the AMR rebuild; these exact
region timers do not include it for either model. The large whole-run gap
does not implicate the unchanged, independently measured jump ILU.

The allocation profile at the short physical predictor reduced stencil work
from 1.637 MB to 0.047 MB by removing the boxed capture, before removing
escaping endpoint views. Pressure-jump construction and coupled-block refill
are allocation-free on that warmed state. Gauge-buffer reuse saved another
22.6 MB of projection allocation over the full trajectory.

### Repeated production advance, Julia 1.11.9

`pressure_probe.jl` now completed three production `simulate!` repetitions
per model at `t=0.5`, after a short warmup, using normal optimization/CPU
targeting and one Julia/BLAS thread. Package images were disabled. This is
the same isolated dependency environment as the bounds-checked comparisons,
but a different compiler configuration: keep its times separate.
Initial AMR/projection and frozen-state decompositions are outside the timer.
The probe adds a small pressure-iteration/structure diagnostic to the example's
default diagnostics; its measured allocation is included.

| Production advance | One velocity | CMDGFM |
| --- | ---: | ---: |
| Wall time, median (range) | 15.759 s (15.637–16.091) | 42.612 s (42.491–42.680) |
| Allocated bytes, median | 12.717 GB | 61.749 GB |
| GC time, median (range) | 1.905 s (1.773–2.066) | 6.025 s (5.911–6.102) |
| Compilation per run, maximum | 0.105 s | 0.050 s |
| Steps / AMR / final faces, every run | 206 / 45 / 12,504 | 219 / 79 / 9,016 |
| Final pressure converged, every run | yes | yes |

Both models' second and third runs have zero measured compilation. These
repeat timings confirm a remaining **2.70× whole-advance time and 4.86×
allocation** gap; they do not attribute it to pressure. On the final frozen
CMDGFM state, jump ILU construction takes 0.033 ms / 95,296 B; application,
GMG application and the frozen BiCGSTAB solve allocate zero bytes. The pooled
late assembly has allocation-free pressure-jump and coupled-CSC refill,
0.153 ms / 18,528 B stencil traversal, and 0.059 ms / 1,056 B donor-row assembly.
These are warmed microbenchmarks, not sums over the evolving trajectory.

The exact-stage wrapper was then run with the same normal optimization
configuration, on both complete trajectories:

| Exact pressure regions, Julia 1.11.9 / normal optimization / BLAS 1 | One velocity | CMDGFM |
| --- | ---: | ---: |
| Coefficient/GMG refresh time | 0.316 s | 0.382 s |
| Projection time | 1.368 s | 1.680 s |
| Total pressure time | 1.684 s | 2.062 s |
| Refresh allocation | 119.316 MB | 111.071 MB |
| Projection allocation | 0.224 MB | 57.467 MB |
| Total pressure allocation | 119.540 MB | 168.538 MB |
| Ordinary projection allocation, median | 1,088 B | 256,056 B |
| Post-AMR projection allocation, median | 1,088 B | 201,072 B |

The pressure ratios are **1.22× time / 1.41× allocation**, consistent with
the bounds-checked comparison, with zero compilation in either pressure
region. Every projection converged and 183 guarded guesses were accepted.
Iteration distributions and full step/AMR/final-face histories match the
bounds-checked stage traces. Pressure-related scalar topology construction
inside AMR is excluded for both models. This is one exact-stage trajectory
per model, not three region-timing repetitions. The whole-wrapper totals
(18.247 s / 12.724 GB versus 40.796 s / 52.322 GB) are not the production
`simulate!` totals: the wrapper skips the zero external-force loop.

An initial suite attempt in this runtime was stopped to restart on the final
source revision. The final complete suite traversed the cut-cell WY geometry
tests without the previously corrupted donor index; its result is below.

## Validation

- 248 independent checks compare full matrices/actions, RHS, gauges,
  retained faces and recovered gradients against fresh assembly, on uniform
  and mixed-level meshes, with and without solids, at density ratios 1 and
  1000. Both native and minimal-compilation runs passed those checks.
- Six additional checks cover compatible scalar/hierarchy sharing and custom
  hierarchy settings. The corrected minimal-compilation run passed all 254.
- Seventeen further checks cover periodic uniform ↔ mixed-level AMR, including
  changing the transport schedule's concrete type while retaining pressure
  storage. These revealed and fixed the schedule-type conversion in the reuse
  path; this does not change Stokes3's bounded transport schedule.
- 156 warm-start checks cover equal/reversed/high density ratios, pure phase,
  moved interfaces, identical projected active velocities, identical inactive
  NaN masks, conservation, accepted good pressure, and rejected bad/nonfinite
  pressure guesses. All 427 pressure-reuse tests pass with forced bounds.
- After direct incidence, all **499** focused checks passed with minimal
  compilation and forced bounds: 48 additional periodic/bounded Float32 and
  Float64 incidence checks and 24 adaptive/cut-mesh structural comparisons.
- The same 499 checks passed again with the final typed solve barrier under
  generic `-O1` and forced bounds. The full `t=0.5` exact pressure-stage trace
  then completed with the unchanged corrected AMR history and all projections
  converged.
- The final pooled/unboxed/gauge-buffer implementation passes all **1,220**
  focused checks under the isolated runtime. The short KH and manufactured
  Poisson checks were rerun on the final source: all 88 checks pass. KH errors
  match the earlier table below to the shown digits. Gradient Linf convergence
  is still not claimed or changed.
- The matrix maximum difference on completed Stokes3 frozen states was at most
  `7.28e-12`, relative `3.37e-17`; RHS differences were zero. Differences
  reflect coefficient-refresh/accumulation roundoff, not another operator.
- Existing CMDGFM tests check moving-interface numerical hierarchy refresh.
  The AMR test additionally requires the pressure workspace to survive and
  its key snapshot to match the new topology.
- Final `Pkg.test`, including declared test dependencies, completed under
  Julia 1.11.9 / generic `-O1` / forced bounds: **16,702 passes, seven failures,
  no errors, no broken tests**, in 6m03.7s. All 1,220 reuse tests, 1,056
  existing CMDGFM tests and 1,736 inactive-velocity tests pass. This is a
  completed suite, **not a clean suite**.

The seven suite failures are:

| Check | Observed failure | Baseline evidence |
| --- | --- | --- |
| `two_phase/transport.jl:48` | warmed `reconstruct!`: 3,008 B rather than 0 | Identical on archived `bdc9ac2`, same Julia 1.11 configuration |
| `navier_stokes/two_phase_navier_stokes.jl:58,72` | warmed timestep selection: 176 B rather than 0, two checks | Both identical on archived `bdc9ac2` |
| `diagnostics.jl:234,248` | warmed cut-cell timestep selection: 16 B rather than 0, two checks | Both identical on archived `bdc9ac2` |
| `mimetic_projection_variants.jl:65` | pressure-commuting residual 0.9, required below 0.2 | Identical on archived `bdc9ac2`; also previously recorded |
| `examples/cylinder_example.jl:13` | nonfinite pressure-drag history | Identical on archived `bdc9ac2`; also previously recorded |

The archived baseline was developed in a separate temporary environment;
neither the live worktree nor its dependency files were changed for that
comparison. No assertion was relaxed. The earlier Julia 1.12 suite had 15,707
passes, two failures and one broken test, but preceded the final changes and
is not the final validation result. An invalid cell index in an interrupted
Julia 1.12 cut-cell WY test remains unexplained; it is not reclassified as a
baseline numerical failure or a proven runtime effect.

## Initial-guess experiment and guarded implementation

The scalar solver already packs the previous pressure as its initial guess;
CMDGFM previously discarded it. [pressure_initial_guess.jl](pressure_initial_guess.jl)
assembles a physical Stokes3 predictor at `t=0.03` and solves the same frozen
matrix/RHS with four guesses. Stopping conditions and ILU are unchanged.

| Initial guess | Initial pressure defect | Initial jump defect | Initial residual L2 | Iterations | Warm time (ms), generic `-O1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero | 0.05613 | 0.09694 | 123.388 | 14 | 95.03 |
| Previous pressure and face jump | 0.08082 | 0.11225 | 108.287 | 14 | 94.20 |
| Previous pressure; current jump fitted with current ILU | 0.07846 | `1.18e-5` | 0.754 | 8 | 53.46 |
| Zero pressure; current jump fitted with current ILU | 0.06505 | `1.16e-5` | 0.866 | 8 | 53.21 |

The fitted jump approximates the existing jump-row equation
`J xi = rhs_jump - jump_pressure*p`. It is an initial guess, not a new jump
condition. Its pressure defect is initially worse, while its combined maximum
defect and L2 residual are better. The production implementation therefore
checks the **true full residual**, rejects poor/nonfinite guesses, retains zero
as a fallback, and requires the same final physical defects.

The short Stokes3 trace to `t=0.08` completed 35 steps and seven AMR changes:
34 accepted guesses, median/p95/max total iterations 8/9/14; maximum volume
defect `1.57e-9`, normal defect `2.46e-12`, and momentum identity error
`8.88e-16`. The unchanged default tolerance is `sqrt(eps(Float64))`.
Short-run wall time was 25.98 s under generic `-O1`/forced bounds; it is not a
native-performance comparison.

[pressure_accuracy.jl](pressure_accuracy.jl) adds explicit solution gates.
Six manufactured Poisson states (`n=8,16,32`, density ratios 1000 and 1/1000)
pass all 36 warm/cold checks, including pressure/gradient L2 and Linf. Maximum
solution difference is `2.77e-11`. This validates an unchanged solution, not
Linf convergence of the discretization: **the gradient Linf error still does
not converge** and this performance work does not attempt to fix it.

Sixteen short KH runs to `t=0.2`, at density ratios 1 and 1000, Weber numbers
0.5 and 5, meshes 8 and 16 and timestep factors 1, 1/2 and 1/4, pass 52 checks
for solver/NaN-storage robustness and bounded temporal error. They are not
long-time dispersion-convergence gates. Set `PRESSURE_WAVE_TIME=default` to use
the existing integration problem's original final times instead.

| Short KH case | `N=8`, full step | `N=16`, full step | `N=16`, half step | `N=16`, quarter step |
| --- | ---: | ---: | ---: | ---: |
| DR=1, We=0.5 | 0.028404 | 0.006769 | 0.006705 | 0.006731 |
| DR=1, We=5 | 0.090639 | 0.037160 | 0.022646 | 0.015926 |
| DR=1000, We=0.5 | 0.100242 | 0.037139 | 0.022976 | 0.017124 |
| DR=1000, We=5 | 0.203982 | 0.069689 | 0.047631 | 0.037444 |

Entries are the existing normalized amplitude-history error at `t=0.2`.

## First-after-AMR attribution

The repeated frozen assembly timings alone do not describe topology rebuild
cost. [pressure_assembly_trace.jl](pressure_assembly_trace.jl) mirrors the
physical predictor, then measures the first cache preparation at each new
topology. An extra staged assembly is intentional: it is an attribution
script, **not** an uninstrumented advance benchmark.

Before the direct-CSC incidence change, generic `-O1` / forced-bounds results
were 9.02 s in pressure-cache preparation during the `t=0.08` trace, mostly
seven post-AMR rebuilds at 1.25–1.52 s each; unchanged-topology refresh was
about 0.3 ms. Other measured assembly/recovery substages summed to under 0.2 s.
The history matched the ordinary trace: 35 steps / seven AMR changes / 22,232
final faces.

`decompose_pressure_cache` in [pressure_probe.jl](pressure_probe.jl) split the
cold cache on the 27,576-face initial mesh. The original incidence builder
alone cost 1.903 s / 4.802 MB, versus 1.915 s / 27.766 MB for the complete
cold cache. Gauge, adjacency, vertex patches, metric transpose and stiffness
slice individually took 0.07–3.53 ms. This identified the next reuse-only
optimization; it did **not** implicate the ILU or require changing K.

### Runtime-validation caveat

Repeated Julia 1.12.7 attempts have crashed in GC or compiler internals,
including system-image restoration with zero allocations, before importing
this package. Neither package-image disabling, generic CPU targeting, forced
bounds nor CPU affinity is a proven general fix. There are related upstream
[GC/code-generation regression reports](https://github.com/JuliaLang/julia/issues/62524),
but they do not establish the cause of these crashes. Compiler-aborted runs
are not numerical test passes or failures. Completed run results and the
runtime configuration must be reported separately.

## Reproduction entry points

From the repository root, with a working compatible Julia environment:

```sh
julia --project=. development/stokes3_performance/pressure_probe.jl one 0.5 3
julia --project=. development/stokes3_performance/pressure_probe.jl cmdgfm 0.5 3
julia --project=. development/stokes3_performance/pressure_stage_trace.jl one 0.5
julia --project=. development/stokes3_performance/pressure_stage_trace.jl cmdgfm 0.5
julia --project=. development/stokes3_performance/pressure_accuracy.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Use a single Julia and BLAS thread for the reported pressure measurements.
The isolated Julia 1.11 runs instead used a separate `--project` in `/tmp`
developing this repository; the repository's Julia 1.12 manifest was not
rewritten for them. Add generic targeting, `-O1` and forced bounds only to
reproduce the explicitly labelled bounds-checked measurements.
