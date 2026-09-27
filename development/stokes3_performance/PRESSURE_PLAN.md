# Pressure-projection performance: analysis plan only

The investigative phase has been carried out; see
[PRESSURE_FINDINGS.md](PRESSURE_FINDINGS.md) for measurements and attribution.
No solver or preconditioner changes have been made.

Scope: explain the roughly tenfold Stokes3 pressure-stage cost of CMDGFM
relative to one velocity, then evaluate lower-allocation alternatives without
weakening convergence or changing the discretization. **This document does not
authorize or implement a solver change.** Use the default Stokes3 problem to
`t=0.5` and retain the current transport, aperture and diagnostics changes.

## What the current measurements do and do not say

| Sampled `t=0.5` pressure work | One velocity | CMDGFM |
| --- | ---: | ---: |
| Pressure-operator refresh | 0.402 s | 0.677 s |
| Projection (including assembly, solve and recovery) | 1.900 s | 20.986 s |
| Total of those sampled stages | 2.302 s | 21.663 s |
| Within CMDGFM projection: coupled assembly | — | 13.579 s |
| Within CMDGFM projection: BiCGSTAB | — | 3.844 s |
| Within CMDGFM projection: preconditioner construction/refresh | — | 1.834 s |
| Steps; final faces | 206; 12,504 | 219; 21,448 |

These are profiler-derived whole-trajectory estimates, not non-overlapping
timers or a same-mesh kernel comparison. CMDGFM takes more steps and retains
more faces. Nevertheless, coupled assembly is about 63% of its measured
pressure-stage time, while **all** preconditioner work is about 8%. Thus the
current evidence does not support jump-ILU construction as the main cause of
the tenfold gap. It may still be worth optimizing once exact timings and
allocations are known. The final solve's 10 CG cycles versus 7 BiCGSTAB
iterations is not an iteration-history comparison.

## Code-based cost model and hypotheses

The one-velocity path refreshes its existing scalar
[`PoissonOperator`](../../src/two_phase/stepping.jl) by replacing
coefficient-dependent values on the current topology; the fixed inverse-face
metric and coefficient refresh map survive. It numerically refreshes the
tree-GMG hierarchy and solves a scalar symmetric problem with CG
([`elliptic_solver.jl`](../../src/solvers/multigrid/elliptic_solver.jl)).

The CMDGFM path first refreshes that scalar operator for Stokes3's gravity
and surface forces. Its projection then separately constructs a pressure
operator **without a template** in
[`_mimetic_pressure_operator`](../../src/two_phase/two_velocity/projection/mimetic_foundation.jl),
followed by a coupled pressure/jump system in
[`_assemble_cmdgfm_system!`](../../src/two_phase/two_velocity/projection/cmdgfm_projection.jl).
Every step constructs face fractions/masses, an incidence sparse matrix,
active-cell/gauge data, vertex/face adjacency for jump stencils, multiple
sparse matrix products and slices for the common gradient and coupling
blocks, and a fresh concatenated system matrix/RHS. Its matrix dimensions
and jump-face set may change even without AMR. The assembled system is stored
in the workspace *after* the solve, but its sparse storage is not currently
refilled in place. The projection also forms `matrix - matrix'` for an
asymmetry statistic and allocates a new `EllipticWorkspace` during gradient
recovery. These are likely allocation sources beyond the Krylov solve.

The default preconditioner is **not** whole-system ILU. It is a lower block
preconditioner: tree-GMG for reduced pressure and `IncompleteLU.ilu` for the
smaller jump block. The pressure hierarchy and work arrays are reused if the
gauge remains the same, but its numerical coefficients must be refreshed for
a moving interface. Jump-ILU factors are reused only when the complete jump
matrix, including numerical values, is identical; `jump_factor_reused=false`
at the final Stokes3 step. The installed IncompleteLU implementation offers
a drop tolerance `τ`; setting `τ=0` gives **full LU**, not ILU(0)/no-fill.
A true no-fill candidate therefore needs a separate implementation or a
verified package capability, not a parameter change to the current call.

Hypotheses to test, in descending order of expected impact:

1. Full two-velocity scalar-operator and sparse block assembly dominate both
   time and allocations; unchanged-topology symbolic work is repeated.
2. New intermediate sparse products/slices and the final block concatenation
   dominate the allocation within assembly, particularly on large AMR meshes.
3. Preconditioner numerical refresh is necessary but modest in total time;
   jump ILU may have a useful cheaper substitute, provided it does not make
   BiCGSTAB slower or fragile.
4. Gradient recovery, asymmetry diagnostics, and small scratch arrays explain
   part of the remaining projection cost.
5. Different mesh histories magnify the whole-run ratio but do not by
   themselves explain the much larger CMDGFM assembly path.

## Measurement sequence

1. **Make a trustworthy baseline.** Extend the development benchmark with
   non-overlapping timers and `@allocated`/`@timed` samples around the
   pressure-stage calls, separately from GC and compilation. Run at least
   three warm `t=0.5` repetitions per model, recording spread, step/AMR
   histories, faces/cells/mixed faces and final defects. Keep the present
   sampled profiler only as a cross-check. Do not charge initial projection
   or AMR workspace construction to a regular fixed-topology step; report
   those separately.
2. **Decompose assembly.** Time and count, individually,
   `_mimetic_face_data`, `_mimetic_pressure_operator`, incidence creation,
   active-cell/gauge partition, incident-face and vertex-patch construction,
   jump-stencil selection, `_cmdgfm_common_gradient_matrix`, sparse
   slicing/products, jump-block triplet generation, sparse conversion,
   final block concatenation, RHS, and post-solve gradient recovery.
   Include the one-velocity coefficient update/GMG refresh/CG path. Report
   bytes, matrix dimensions and nonzeros, cost per face and per unknown.
   Attribute the `matrix - matrix'` diagnostic separately; it is outside
   `_assemble_cmdgfm_system!` but still inside projection.
3. **Decompose preconditioner and solve.** Record pressure GMG refresh time
   and bytes, jump ILU factorization time/bytes, factor nonzeros and fill
   ratio, both `ldiv!` costs, BiCGSTAB matvecs and iterations, correction
   restarts, non-finite fallback, and all residual/constraint defects for
   *every* step. Count `pressure_hierarchy_reused` and `jump_factor_reused`
   on unchanged topology and directly after AMR. Include jump-block diagonal
   margin and zero/tiny diagonal counts as possible predictors for Jacobi.
4. **Compare on identical states.** Capture representative initial fine,
   ordinary fixed-topology, first-post-AMR and late states. For each frozen
   mesh/interface, evaluate both pressure paths without evolving either
   flow. Time first assembly, repeated same-state assembly, numerical-value
   changes with unchanged topology, solve with prebuilt matrix, and full
   projection. This separates algorithmic cost from the two methods' very
   different late meshes. Verify that any repeated-state benchmark refreshes
   the coefficient-dependent hierarchy when the values change.
5. **Map structural stability.** Per step, record hashes/lengths of cell
   gauge, mixed-face set, sparse column pointers/row indices and values.
   Determine what fraction of steps could reuse incidence, stencil graph,
   sparse pattern, or numerical jump factors. Include AMR transitions and
   cases where faces enter/leave the active jump set. Do not equate equal
   sparsity with equal numerical preconditioner validity.

The planned output is one table of end-to-end pressure time/bytes per model,
one CMDGFM substage breakdown, per-step iteration and defect distributions
(median/p95/max), and matched-state cost curves versus face/mixed-face count.
Add a small diagnostic script in this development directory only when the
analysis begins; this plan deliberately adds no instrumentation or solver
code.

## Candidate changes to evaluate *after* attribution

Assess one candidate at a time, first on frozen matrices and then on full
trajectories. Revert or reject any candidate that fails the gates below.

| Candidate | Expected gain | Main risk/check |
| --- | --- | --- |
| Reuse the compatible pressure-operator template and inverse metric on unchanged topology; avoid duplicate scalar construction where the one- and two-velocity coefficients genuinely agree | Large assembly/geometry savings | Embedded cut-cell coefficients differ (`A/ρ` versus `A²/ρ`); never reuse the wrong metric or a stale density. |
| Cache mesh-only incidence, face/cell adjacency and vertex patches | Less repeated assembly and allocation | Invalidate exactly on AMR; verify periodic and coarse/fine connectivity. |
| Retain sparse block patterns and refill values while the gauge/jump active set is unchanged | Potentially large allocation reduction | Pattern changes as mixed faces appear/disappear; rebuilding must be safe and not slower on unstable patterns. |
| Reuse projection `EllipticWorkspace` and scalar/vector scratch | Modest, low-risk allocation reduction | Resize after AMR; no aliasing during Krylov or force evaluation. |
| Replace jump ILU by protected diagonal Jacobi, block Jacobi on jump-graph components, or a true ILU(0); compare several drop tolerances | Less factorization memory/time | Extra BiCGSTAB iterations, stagnation at high density ratio, small/zero diagonals or loss of robustness. |
| Matrix-free/fused coupled action after simpler reuse is exhausted | Avoid sparse products/concatenation | Larger implementation/verification burden; preconditioner still needs meaningful blocks. |

For the preconditioner comparison, keep pressure GMG fixed. Compare *jump*
preconditioners first; replacing the full coupled preconditioner by plain
Jacobi is a separate, higher-risk experiment. Time factor **construction**
and repeated **application**, then compare total solve time and allocation,
not factorization alone. A cheaper factor that doubles iterations or triggers
extra correction solves is not an improvement. The current `:ilu` option
factors the **whole** coupled matrix and is not the proposed no-fill jump
alternative.

## Validation gates before accepting any future implementation

- Exactly the same discrete operator and right-hand side for reuse-only
  changes, verified on representative uniform/AMR and cut-cell snapshots
  (matrix/action products, pressure/jump residuals, gradient and aperture
  constraints). A deliberate preconditioner change may alter iterations,
  not the converged physical solution.
- CMDGFM conservation/normal-jump defects, manufactured Poisson errors, and
  two-velocity wave/instability tests remain at baseline tolerances. No new
  solver breakdown or late-time failure at several density ratios, Weber and
  Bond numbers, including Stokes3 `t=0.5` and AMR transitions.
- Compare the full suite with its current baseline rather than assuming a
  clean baseline: it currently has two known failures (the mimetic-variant
  pressure-commuting threshold and non-finite cylinder pressure drag) and
  one broken test. A candidate must not add failures or worsen those defects.
- Record per-step iteration p95/max and correction restarts. Reject a faster
  setup if end-to-end pressure time, tail latency or robustness worsens.
  Test at the configured tolerance and a tighter reference tolerance so a
  weak preconditioner cannot conceal an inaccurate solve.
- Measure warm full-run wall time, bytes and GC at least three times per model
  with unchanged step/AMR histories. Report fixed-mesh versus first-post-AMR
  costs separately. Aim first to remove avoidable assembly allocations;
  only then choose the simplest jump preconditioner that passes all gates.

No pressure optimization should be implemented until these measurements are
reviewed with the user, particularly any proposal to change the pressure
metric or the block preconditioner.
