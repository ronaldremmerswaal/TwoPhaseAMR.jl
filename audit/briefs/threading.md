# Brief: Phase 4E.4 — threading

Owner decision (2026-11-11): run this as **one Opus agent on high effort**
(`model = "anthropic/claude-opus-5-5:high"` and **no `effort` parameter**: `effort = "hi"`
overrides the suffix and runs the model at max reasoning, verified 2026-11-11), started by the
orchestrator without further owner approval once the prerequisites below are met.

## Start condition (orchestrator checks, no owner approval needed)

- PR #139 (cut-donor transport) and the `ProjectedLiquidVelocity` removal are merged
  into `main`, so the refactor does not race the transport changes.
- No other agent is editing `src/threading.jl`, the elliptic solver, `diffusion!`,
  `_phase_face_average!` or `surface_tension!`.

## Decisions already taken (do not ask the owner again)

1. **Determinism:** results must be **bitwise identical for every thread count**
   (fixed chunking and fixed reduction order in dot products, norms and sums).
   The regression oracle stays a single bitwise record; the equivalence test
   asserts bitwise equality between 1, 2 and 4 threads.
   If a specific reduction cannot reach the speed-up target this way, the agent
   reports the measured cost; tolerance-based equivalence is not accepted without
   a new owner decision.
2. **Machine time:** the agent may run speed-up measurements with up to
   8 threads on this shared machine (Ryzen 9 5950X, 16 physical cores), without
   asking, when `free -g` available ≥ 8 GB and the 1-minute load average is below 4
   (check `uptime` right before); at most one multi-threaded process at a time, each
   run ≤ 10 minutes; otherwise wait and retry. All correctness work stays
   single-threaded except the equivalence tests. Confirm the final 1/2/4-thread
   numbers once on a CI runner (`gh workflow run`; standard runners have 4 vCPUs,
   so 8 threads is measured locally only), reporting both.

## Scope (WORKPLAN 4E.4; audit C15, C16, C20, F11, F23, E-7 in `audit/audit-core.md`,
`audit/audit-twophase.md`, `audit/audit-perf.md`, `audit/AUDIT.md`)

1. **Re-measure first** on current `main`: 1/2/4/8 threads on Taylor–Green 128²
   (hierarchical), the cylinder, and the Stokes-3 wave (`examples/10_stokes3_wave`)
   with both `OneVelocityModel()` and the two-velocity model, at the default resolution
   and one level finer (the capillary wave is dropped as a target case: owner, 2026-11-12,
   its functionality is covered by Stokes-3);
   profile the serial fraction of a step. Audit baseline (pre-4B–4E.3): 4 threads gave
   TG 1.17× (uniform), 1.53× (hierarchical), CMDGFM 1.16×. Order step 4 by the
   profile, not by the audit's guesses.
2. **One threading layer:** five launch primitives (`@_parallel`, `_parallel_for`,
   `_parallel_columns`, `_parallel_block_columns`, `@_parallel_tasks`) → at most two;
   fix the `_parallel_for` docstring (says task scheduler, is Polyester); remove the
   double-expanded loop body of `@_parallel`; make the threading grain overridable
   so tests can force the threaded path on small meshes. Clean cutover, no aliases.
3. **Errors and callbacks:** verify how Polyester surfaces an exception thrown in a
   worker (≥ 2 threads); move `throw`/`error` out of parallel bodies (flag + throw
   after the loop): `_update_elliptic_activity!`, `_add_existing_sparse!`,
   field-transfer bodies. User callbacks (`initialize!`, `fill_halo!` boundary
   values) either run serially or are documented as required to be thread-safe.
4. **Parallelise serial loops:** RHS scaling, residuals, norms and the strong-residual
   callback in every Krylov iteration; field↔vector copies; `apply!`'s final loop;
   `diffusion!` gather/scatter (use the existing transpose product);
   `_phase_face_average!` (serial over all faces); `surface_tension!` curvature
   lookup (scans all samples per face); plus what the profile shows (likely CMDGFM
   BiCGSTAB vector operations and the WY/VOF sweeps).
5. **Tests and CI:** serial-vs-threaded bitwise equivalence test (1/2/4 threads) on
   one-phase AMR, two-phase CMDGFM with a solid and an AMR regrid; a CI job running
   it at 2 threads; no new allocations in warm steps.

## Acceptance

- Owner decision (2026-11-12):
  - **Hard acceptance:** 4 threads ≥ 2× on TG 128² (hierarchical), the cylinder and
    the Stokes-3 wave with the one- and the two-velocity model, results bitwise
    identical for every thread count.
  - **Stretch goal:** 8 threads ≥ 4× (≥ 50 % parallel efficiency) on the same cases
    one level finer than their default/benchmark resolution (Stokes-3: one level finer
    than the example default). Where it is missed, report the profile showing what
    remains serial or memory-bound.
  - **Smoothers/preconditioners:** if the profile shows that inherently sequential
    parts (ILU triangular solves, Gauss–Seidel-type smoothers, regrid/tree operations)
    block a target, do **not** replace them with parallel variants: stop that line of
    work and report the measured cost; switching methods changes iteration counts and
    results and is an owner decision.
  - Report a 1/2/4/8-thread table for all target cases at both resolutions.
- Oracle: one bitwise record valid for every thread count. One re-record is allowed in
  the PR introducing fixed-chunk reductions (separate commit, every changed case listed
  with its max relative change vs the old record, ≤ 1e-10); otherwise unchanged.
  Equivalence test bitwise at 1/2/4 threads (CI) and 8 threads (locally, reported).
- Focused groups, techref ch. 11/14 threading statements, docs (`docs/src`
  performance/threading section), CHANGELOG updated; Runic 1.10.0 on all `.jl`.
- One PR per step (1 is a report, 2–5 are PRs), each green on CI with the
  `run-integration` label; the orchestrator reviews with `julia-perf-reviewer`
  and `numerics-reviewer` before merging.
