# Audit: `threading` branch (2026-10-20)

Scope: `origin/main..origin/threading` (17 commits, 2026-09-29 → 2026-10-10, base `157448c` = current `main`). Static review of the diff; nothing re-run for this report except where stated.
Evidence tags: **[obs]** observed in the diff or a run, **[INF]** inference.

## 1. Summary

- The branch is a performance-research effort ("phased optimization plan for shared-memory parallelism"). Of 34 150 added lines, **≈ 350 are package source** (`src/`, 4 files), 675 are tests, ≈ 5 000 are Documenter pages (12 `multithreading-*.md` execution reports), ≈ 10 000 are benchmark drivers/probes under `benchmark/scaling/` (+ two Manifests), and ≈ 7 000 are a standalone research package `research/UniformLWAdvection/` (Julia + Fortran/OpenMP).
- The production-relevant result is small: (a) a **bug fix** for `PhaseFaceScalar` across uniform ↔ mixed-level AMR schedules, (b) a **compact cell-advection workspace** (first-order upwind `advect!` on `CellField`) that is 1.4–2.3× faster serially, (c) a stale-workspace guard and a boxed-closure fix.
- The optimized operator (`AdvectionWorkspace` + `advect!(::CellField, …)`) is **not used by any Navier–Stokes or two-phase stepping path** on `main` (only tests, `test/threading.jl`, the performance contracts and the cut-cell scalar `advect!`) [obs: grep of `AdvectionWorkspace` in `src/`, `examples/`, `ext/`]. The branch therefore does not address the bottlenecks measured in `audit-perf.md` (base-level coarse LU, per-step coarse refactorization, `reconstruct!`, mimetic assembly).
- The stated goal (≥ 50 % parallel efficiency at 4/8 workers on 2048²/4096²) was **not met**; the reports say so explicitly and attribute much of the large-grid loss to an L3 cache-capacity transition on the two-CCD Ryzen 5950X.
- An unresolved **intermittent memory-corruption crash** during 2048² mesh construction is documented (invalid `InteriorFace` fields, signal 11), not attributed to code, runtime or hardware. Together with the intermittent LLVM/GC crashes seen during the main audit on the same machine, this warrants a hardware/runtime check before trusting long runs on this workstation.

## 2. What the branch contains

| Area | Files | Lines | Nature |
|---|---|---|---|
| Package source | `src/operators/advection/cell_advection.jl`, `src/indicator/embedded_operators.jl`, `src/two_phase/phase_face_fields.jl`, `src/two_phase/phase_face_transport.jl` | +350 / −76 | compact upwind workspace; schedule-union fix |
| Tests | `test/operators/advection/cell_advection_compact.jl` (542), `test/two_phase/phase_face_lifecycle.jl` (133), `test/runtests.jl` | +677 | bitwise equivalence vs old operator; AMR lifecycle |
| Example | `examples/cylinder_flow.jl` | 1 line | removes the `force > 100 ? NaN` guard on the t = 0 drag sample (unrelated to threading) |
| Docs | `docs/src/multithreading-{optimization-plan, literature, execution, stage2-report, stage3-*-report ×8, cache-cliff-investigation, julia-fortran-comparison, waterlily-reference}.md`, `docs/make.jl` (+30), `docs/export_stage2_assets.jl` | ≈ 5 000 | plans, gate ledgers, measurement receipts with SHA-256 hashes |
| Benchmarks | `benchmark/scaling/**`, `benchmark/correctness/**` | ≈ 10 000 + 2 Manifests | drivers, probes, copied-source studies, plots |
| Research | `research/UniformLWAdvection/**` | ≈ 7 000 | standalone dense uniform LW toy package, Polyester ext, Fortran/OpenMP reference, perf-counter C helpers |

## 3. Source changes, judged

### 3.1 `PhaseFaceScalar` schedule union + function barrier — **take (bug fix)**

`main` declares `PhaseFaceScalar{Axis,T,M,S,W}.workspace::W` (concrete), but AMR transfer reassigns it: `src/two_phase/phase_face_transfer.jl:43` `q.workspace = work`, where `work` may carry a `UniformDualAdvectionSchedule` or an `AMRDualAdvectionSchedule` depending on whether the new mesh is uniform. When the mesh switches between uniform and mixed-level, the assignment cannot convert to the original `W` [INF: `convert` MethodError]. The branch changes the field to a two-member `Union` of the concrete workspace types and adds a function barrier (`_advect_phase_faces_workspace!`) so the hot loop still specializes. This is the right pattern (small `Union`, union-splitting, barrier once per sweep).

The branch's new test `test/two_phase/phase_face_lifecycle.jl` (refine 1 → 4 → 7 blocks, mixed coarsen/refine, back to 1 block, with conservation and constant-preservation checks) **errored twice when run against `main`** (114 pass, 2 errors; run before this static-only review was requested; error text not captured) [obs]. Whether the production two-velocity path hits this depends on whether user-visible `PhaseFaceScalar`s are registered with `AMRWorkspace`: `TwoVelocityMomentumAdvectionWorkspace` recreates its scalars when the uniform flag changes (`src/two_phase/two_velocity/momentum_workspace.jl:28-32`), so the internal path is probably safe [INF]; the public `AMRWorkspace(mesh, interface, qx, qy)` path is not.

Recommendation: cherry-pick the two `src/two_phase/*` hunks and the lifecycle test. Effort S. Also replace `typeof(q).parameters[1]` in the test with a small accessor (reflection on type parameters is brittle).

### 3.2 Compact cell-advection workspace — **do not merge now; keep as reference**

What it does [obs]:
- Intra-block faces get dense x/y slots (`x_face_ids`, `y_face_ids`, per-block constant `x_areas`/`y_areas`); fluxes are computed per block column from contiguous halo-padded arrays.
- Cells not touching a block edge sum four arithmetic neighbours; edge cells keep the CSR incidence path. A packed `UInt8` (`interior_order`) encodes the original CSR summation order so results are **bitwise identical** to the old operator.
- Face IDs stored as `Int32` when the face count fits (`_cell_advection_face_id_type`), else `Int`.
- `generation::UInt` stamp + `_validate_advection_workspace_generation` rejects a workspace built for an older mesh topology.
- `_stable_timestep_promoted` passes the promoted type as a method parameter (`::Type{T}`) to avoid boxing `zero(T)` in the column closure.

Measured on the branch (stage-3 report; Julia 1.10.12, 5 processes per cell): ns per unknown per step, frozen → compact, 1 worker: 15.1 → 10.8 (512², 1 block), 26.8 → 17.1 (2048², 1 block), 41.7 → 17.9 (2048², 8×8-cell blocks); 4 workers 2048² 1 block: 11.7 → 6.7. All 180 output fields bitwise equal to the frozen oracle.

Judgement:
- **Benefit is off the critical path.** The operator is first-order upwind for passive `CellField`s; no flow solver step uses it. +260 net lines in a 330-line file for a test/utility operator conflicts with the audit's bloat targets.
- **Complexity is driven by a bitwise-reproducibility constraint** (`_cell_advection_order`, packed order byte, separate edge/interior paths, five `face_kinds`). If kept, a fixed summation order plus tolerance-based tests would remove ≈ 60 lines.
- **Unchecked indexing risk.** `x_face_ids`/`y_face_ids` start as zeros and are filled only for faces classified as regular; nothing asserts that every slot was filled before `velocity.interior[face_id]` is read under `@inbounds`. A zero ID is an out-of-bounds read. Add `all(!iszero, …)` at construction if this is ever merged.
- x and y loops are duplicated in `_cell_advection_regular_fluxes!` and `_cell_advection_regular_rates!`.
- The techniques (owned face lists, dense per-block slots, narrow IDs, block-column tiling) are relevant for the kernels that matter (staggered LW sweeps, WY fluxes, `_phase_face_average!`), but must be re-measured there.

Recommendation: leave on the branch. Port two small pieces independently: the **generation guard** (generalize it to every workspace that caches mesh topology: `StaggeredAdvectionWorkspace`, `WYAdvectionWorkspace`, multigrid hierarchies — the main audit found several "is this workspace still valid?" ad-hoc checks), and the **`::Type{T}` closure fix** pattern (grep for other closures capturing a runtime `T`). Effort S each.

### 3.3 `examples/cylinder_flow.jl` — **revert or justify**

The `force > 100 ? NaN : 2force` guard skipped the impulsive-start drag spike at t = 0; the branch records it unconditionally. Unrelated to threading and changes the example's diagnostic history and plots [obs].

## 4. Research, benchmarks and reports, judged

| Item | Value | Recommendation |
|---|---|---|
| `docs/src/multithreading-literature.md`, `multithreading-waterlily-reference.md` | Literature review and comparison with WaterLily's threading model | Condense into the technical reference (parallelization appendix) or a short `performance.md` page; keep citations |
| `docs/src/multithreading-optimization-plan.md` (950 lines), `-execution.md` (684) | Staged plan (stages 1–8: uniform LW → non-uniform → AMR → staggered → Poisson/GMG/CMDGFM → integration), acceptance contract, gate ledger | Keep the **measurement contract** (fresh processes, 5 repeats, bootstrap CIs, bitwise oracles, efficiency definition) as a short developer page; move the ledger out of user docs. Stages 4–8 should be re-prioritized against `audit-perf.md`: stage 7 (Poisson/GMG) is where the measured time goes |
| `multithreading-stage2-report.md`, `-julia-fortran-comparison.md`, `-cache-cliff-investigation.md` | Dense uniform LW toy vs Fortran/OpenMP; diagnosis of the 1024² → 2048² cliff as an L3-capacity transition on the 2-CCD CPU | Key finding worth one paragraph in the docs: large-grid efficiency on this CPU is bandwidth/L3-limited; per-unknown cost of the production operator (≈ 6.7 ns at 4 workers) is ≈ 4–5× the dense toy (1.3–1.8 ns) |
| `multithreading-stage3-*-report.md` (8 files) | Rejected candidates (ordered gather: −42 % on many-block at 8 workers; rolling flux: slower everywhere; scalar selector), narrow IDs accepted | Negative results are useful; keep as one summarized table, not 8 pages in the user site |
| `multithreading-stage3-mesh-crash-report.md` | Intermittent memory corruption (high bits set in `InteriorFace` `Int` fields, e.g. `plus.block = 262145` on a one-block mesh) right after `_build_face_schedules` at 2048²; not reproduced with `--check-bounds=yes` or in later instrumented runs | See §5 |
| `benchmark/scaling/**` (+ `Manifest.toml`, `stage3/narrow_ids/numerical_project/Manifest.toml`), `benchmark/correctness/**` | Probe generators that copy source trees, receipt writers, plotting | Do not merge into the package. These generate the 4.7 GB untracked `benchmark/scaling/results` found in the main audit. Move to a separate repository or keep on the branch; if any driver is kept, it should benchmark the public API without copying `src/` |
| `research/UniformLWAdvection/**` | Standalone toy package + Fortran/OpenMP + perf-counter C | Separate repository (or keep on branch). Not package code |
| `docs/make.jl` (+30), `docs/export_stage2_assets.jl` | Adds the 12 report pages to the nav, exports benchmark figures into `docs/src/assets/` | Do not merge; conflicts with the "light API site" target (`audit-docs.md`) and is the origin of the untracked `docs/src/assets/` content |

## 5. Mesh-construction corruption: risk assessment

Observed on the branch (Julia 1.10.12, one thread, pinned): two fresh processes found 226 and 263 invalid faces immediately after `_build_face_schedules` for a 2048² single-block periodic mesh (8 388 608 faces); other runs crashed with signal 11 in `_face_dual_volumes` (unchecked indexing through a bad `CellRef`); several later runs, including `--check-bounds=yes` and instrumented checkpoints, completed cleanly. Kernel logs and EDAC counters were not accessible. During the main audit on the same workstation (Julia 1.12.7), the performance and docs auditors independently hit intermittent LLVM and GC segfaults (3 of ≈ 40 fresh processes; two JIT aborts), none reproducible on rerun.

Assessment [INF]: corrupted upper bits in otherwise plausible integers, intermittency across two Julia versions, and unrelated crash sites are more consistent with a hardware/memory or runtime issue than a deterministic logic error in `_build_face_schedules`, but a code cause is not excluded. Recommendations:
1. Run a memory test (e.g. memtest86+ or `stressapptest`) on the workstation; reproduce the 2048² construction on a different machine/CI runner.
2. Add a cheap O(faces) invariant check of the face schedule at construction (block indices in range, `i, j` within block size, axis ∈ {1,2}), enabled in tests; it turns silent corruption into an early, attributable error. Effort S.
3. Until resolved, treat crashes and unexplained numerical outliers from long runs on this machine with suspicion.

## 6. Reuse plan

| Action | Items | Effort |
|---|---|---|
| Cherry-pick now | `src/two_phase/phase_face_fields.jl` + `phase_face_transport.jl` hunks; `test/two_phase/phase_face_lifecycle.jl` | S |
| Port as small independent changes | workspace generation guard (generalized); `::Type{T}` closure fix; face-schedule invariant check (§5) | S each |
| Keep on branch as reference | compact cell-advection workspace and its 542-line equivalence test; narrow-ID and block-tiling techniques for later staggered/WY kernel work | — |
| Condense into docs / tech ref | literature review, measurement contract, cache-cliff finding, table of accepted/rejected candidates with numbers | M |
| Do not merge | `benchmark/scaling/**`, `benchmark/correctness/**`, `research/**`, 12 report pages, `docs/make.jl` nav additions, `docs/export_stage2_assets.jl`, cylinder example change | — |
| Re-plan | Future threading work should start from the measured hot spots: base-level multigrid coarsening (serial UMFPACK), coarse refactorization per step, serial streaming loops, `reconstruct!` | — |

Once the cherry-picks are on `main`, the branch can be archived (tag it, e.g. `archive/threading-2026-10`, then delete the branch) so its history stays reachable without remaining an active line of development.
