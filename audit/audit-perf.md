# TwoPhaseAMR.jl — time-stepping performance audit (slice `perf`)

Scratch, scripts, raw logs: `/tmp/audit-perf/` (`cases.jl`, `stages.jl`, `prof.jl`, `alloc_*.jl`, `jet_run.jl`, `ths.jl`, `latency.jl`, `lu_bench.jl`, `exp1.jl`, `exp2.jl`, `*.log`). Prototype copies: `/tmp/audit-perf/pkg_b` (function-barrier patch), `/tmp/audit-perf/pkg` (PrecompileTools workload). No tracked file was modified.

**Machine / settings.** AMD Ryzen 9 5950X (16C/32T), 31 GB RAM, Julia 1.12.7, repo `Manifest.toml` untouched (own `/tmp/audit-perf/env`: TwoPhaseAMR dev + BenchmarkTools + JET), BLAS 1 thread, `-t 1` unless stated (thread scaling: `-t 1/2/4`). **The machine was shared**: load average ≈ 5 (docs build + other Julia jobs) during all runs → absolute numbers carry ±10 % noise; shares/ratios are more reliable than absolutes. Warm = ≥5 untimed steps after the first-call compile.

## 1. Summary

1. **Single-phase TG is allocation-free (4–5 KB/step) and cheap (0.8–1.1 µs/cell/step), but 90 % of it is two linear solves whose coarse level is a *direct LU of the whole base mesh*** (the tree GMG only coarsens through AMR levels, never the base level): a flat 128² mesh = one 16 384-unknown UMFPACK solve per V-cycle = 49.6 % of the step. The identical problem built as 1 root block + 3 uniform levels is **1.6× faster** (18.4 → 11.5 ms) with no code change. The same effect makes two-phase cost scale ≈ N^1.5 (k=2→4: 3456→7680 cells, 16.8→54.7 ms).
2. **Two-phase steps are dominated by per-step rebuild of coefficient-dependent preconditioners** (`lu` of the coarse level for diffusion + pressure every step; 4 of the 5.5 MB/step allocated; ≈ 30 % of the one-velocity step) and by `reconstruct!` (≈ 20 %). Mimetic projection is 5× slower than CMDGFM (106 vs 21 ms/step, 135 MB/step, ~323 MINRES its/step, 84 % of its step in `project!`).
3. **Latency is the largest user-visible cost**: README quick start = 25.6 s to first result, two-phase = 69.5 s (99 % compilation). A PrecompileTools workload cut this to 0.8 s / 4.0 s (measured in a /tmp copy) — but several natural workloads (`simulate!`, `stable_timestep(flow)`, AMR rebuild + diagnostics) SIGSEGV Julia 1.12.7's LLVM during pkgimage emission (each reproduced 1–2×), and 3 of ~40 fresh runs crashed in the GC while compiling.
4. **Thread scaling is poor** (4 threads: TG 1.17× flat / 1.53× hierarchical; CMDGFM 1.16×): UMFPACK factor/solve and many `eachindex` streaming loops are serial; allocations rise 5 → 35 KB/step with threads.
5. The "≈1 KB/cell/step single-phase" figure is **not reproduced**: periodic single-phase = 4.3 KB/step total; non-periodic = 105–208 KB/step, all from a `Ref{Any}` boxing in `_add_boundary_convection!` (fix: 105 KB → 9.8 KB, measured). The ~1.7 KB/cell figure is real for two-phase (5.7 MB/step @3456 cells) and is mostly `lu` factor storage.

## 2. Measurement tables

Cell counts: TG 64² = 4096 cells/16 blocks, 128² = 16 384 cells/64 blocks (blocks 16², periodic, README parameters, ν=0.02). Two-phase: capillary-wave setup of `examples/capillary_wave.jl` (ρ ratio 10³, σ=1, free-slip, periodic-x, interface-AMR `max_level=2`, update interval 5), base mesh scaled by k: k=2 → 3456 cells/54 blocks, k=4 → 7680 cells/120 blocks. dt = 0.5×CFL bound (two-phase), 0.45-CFL (TG, “CFL dt”). `run_example(max_level=2,final_time=…)` itself = 960 cells (k=1); I used the same setup with scaled base mesh (`cases.jl:cap_flow`) to get moderate sizes. No AMR topology change occurred in any timed window (0/40 steps changed); AMR cost measured separately.

### 2.1 Per case: time, allocations, solver iterations (1 thread, warm, one `simulate!` call per step)

| case | cells | median ms/step | alloc/step | pressure its/step | notes |
|---|---:|---:|---:|---:|---|
| (a) TG, README dt=1e-3 | 4096 | 2.33 | 4.25 KB | **0** (residual already below tol) | README example never runs CG |
| (a) TG, README dt=1e-3 | 16384 | 12.56 | 4.25 KB | 0 | |
| (a) TG, CFL dt (0.022 / 0.011) | 4096 | 3.54 (3.69 with KE+div diagnostics) | 4.25 KB (7.6 KB w/ diag) | 1 | diffusion CG 1+1 |
| (a) TG, CFL dt | 16384 | 17.9 (18.98 w/ diag) | 4.25 KB (7.6 KB) | 1 | |
| (a') TG 128², **hierarchical** (1 root 16², 3 levels) | 16384 | 12.2 | 7.5 KB | 3 | same physics, no code change |
| (b) 1-velocity | 3456 | 15.64 (min 15.27) | **5.73 MB** (1657 B/cell) | 2 (CG) | |
| (c) 2-velocity CMDGFM | 3456 | 21.35 | 5.97 MB (1729 B/cell) | 5–6 | |
| (d) 2-velocity Mimetic | 3456 | **105.8** (min 70.8, mean 109.5) | **134.7 MB** (38 985 B/cell) | **323 median (130–401, cap 500)** MINRES | |
| (c) CMDGFM k=4 | 7680 | 70.9 | 21.7 MB (2819 B/cell) | 6–7 | |

Nonperiodic single phase (4096 / 16384 cells): 3.01 / 15.6 ms, **105 504 / 207 904 B/step**.

### 2.2 Phase breakdown (stage timers replicating `step!`; ms/step, % of step; `stages.jl`)

Single-phase TG, CFL dt (periodic):

| stage | 64² (4096) | | 128² flat (16384) | | 128² hierarchical |
|---|---:|---:|---:|---:|---:|
| momentum advection | 0.298 (8.9 %) | | 1.485 (8.5 %) | | 1.53 (13 %) |
| diffusion solve (x,y CG, 1+1 its) | 1.563 (46.5 %) | | 8.593 (49.0 %) | | 3.39 (29.5 %) |
| projection (div + CG + grad) | 1.444 (43.0 %) | | 7.24 (41.2 %) | | 6.53 (56.9 %) |
| diagnostics (KE + div norm) | 0.051 (1.5 %) | | 0.219 (1.2 %) | | 0.006 |
| everything else (copy, BCs) | <0.02 | | <0.02 | | |
| **total** | 3.36 | | 17.55 | | 11.47 |

Two-phase (k=2, 3456 cells; ms/step, % of stages; alloc KB/step in parentheses):

| stage | (b) 1-velocity | (c) CMDGFM | (d) Mimetic |
|---|---:|---:|---:|
| copy + BC old | 0.004 | 0.263 (3.5) | 0.256 |
| transport (VOF + momentum advection) | 5.897 (36.2 %, 330 KB) | 7.621 (33.5 %, 364 KB) | 7.411 (6.8 %) |
| BC convected | 0.000 | 0.277 | 0.251 |
| **diffusion solve (variable-ρ,µ)** | **7.176 (44.0 %, 2942 KB)** | 7.846 (34.5 %, 2955 KB) | 7.153 (6.6 %) |
| BC velocity | 0.000 | 0.244 | 0.243 |
| refresh pressure operator + PC | 1.913 (11.7 %, 1976 KB) | 1.957 (8.6 %) | 1.844 |
| phase forces (σ, g) | 0.144 (287 KB) | 0.449 (287 KB) | 0.403 |
| **projection** | 1.077 (6.6 %, 1.1 KB), 2 CG its | 4.011 (17.6 %, 168 KB), 5–6 its | **90.85 (83.7 %, 114 MB), ~311 MINRES its** |
| AMR indicator + `adapt!` attempt (no change) | 0.074 | 0.072 | 0.078 |
| diagnostics record (interface height) | 0.019 | 0.022 | 0.021 |
| **sum** | 16.30 | 22.76 | 108.5 |

Per-iteration: one-velocity CG ≈ 0.54 ms/it (3456 cells); CMDGFM ≈ 0.73 ms/it (incl. assembly); Mimetic ≈ 0.28 ms/MINRES it incl. assembly (90.9 ms/323).

Other per-step costs when `dt=nothing` (CFL query each step): `stable_timestep(flow)` 0.056 ms (1-vel) / 0.31 ms (2-vel); `_enforce_current_velocity_boundary!` 0.21 / 0.58 ms. Negligible.

AMR topology rebuild (`_rebuild_topology_dependent_state!`, mesh unchanged, 3456 cells, 2nd call): 45 ms (CMDGFM run), 51 ms (mimetic run), 238 ms (1-vel run — GC/load noise suspected) = 3–15 steps; 61.5 MB each; first call 3.3–4.9 s (compile). A real topology change additionally pays `adapt!` field transfer (not isolated: no change happened organically in 40 steps; indicator evaluation each step is 0.074 ms).

### 2.3 Profile (Profile.@profile, 1 ms, `prof.jl`; “incl.” = inclusive share of `simulate!`)

| case | top entries |
|---|---|
| TG 128² flat | `umfpack_dl_wsolve` 49.6 % (coarse solve inside V-cycle); `umfpack_dl_numeric` 10.6 % + `_symbolic` 2.9 % + `spdiagm` 2.6 % (dt-jitter refactorization, see F4); everything else <3 % each |
| (b) 1-vel | diffusion 46.6 % incl. (of which `_refresh_tree_variable_diffusion_preconditioner!` 29.6 %); `lu` 32.1 % incl. (`umfpack_dl_numeric` 15 % self, `decrement` 9.1 %, `_symbolic` 7.8 %); `reconstruct!` 23.3 % (`_reconstruct_interface_cell!` 8.3 % self, `_parallel_sparse_mul_from_transpose!` 9.6 % self) |
| (c) CMDGFM | diffusion 34.3 % (refresh 18.3 %), transport 34.2 %, `lu` 20.5 %, `reconstruct!` 19.1 %, V-cycle 18.7 %, `_mark_inactive_phase_velocities!` 5.1 % self (velocity_fields.jl:43–46) |
| (d) Mimetic | `project!` 82.1 %; `minres!` 48.2 %; block-preconditioner `ldiv!` (mimetic_projection.jl:123, `\` → CHOLMOD) 43 %, `cholmod_l_solve` 38.3 % self; `_assemble_mimetic_system!` 28.0 % (`PoissonOperator` rebuild 9.9 %, `_local_joint_constraint_operator` 10.7 %, `_mimetic_incidence` 5.0 % self at mimetic_foundation.jl:11) |

### 2.4 Steady-state allocation sites (`Profile.Allocs`, sample_rate=1, 1-vel k=2; `alloc_one.log`)

Allocs captured 1.9 MB/step of 5.5 MB/step measured with `@timed`; the rest is attributed by direct measurement: `lu(A)` of the diffusion coarse matrix allocates 2.93 MB and of the pressure coarse matrix 1.08 MB per call (`lu_bench.log`) = 4.0 MB/step.

| KB/step | allocs | site |
|---:|---:|---|
| 561 | 34 | `solvers/multigrid/geometric_preconditioner.jl:987` (`_refresh_geometric_multigrid!`) |
| 273 | 5 | `operators/elliptic.jl:31` (`_cell_references`, rebuilt per `PoissonOperator`) |
| 249 | 5 | `two_phase/variable_diffusion_preconditioner.jl:337` (`lu(coarse.matrix)`) |
| 102 | 5 | `geometric_preconditioner.jl:993` (`lu(augmented)`) |
| 54 ×3 | 1 each | `forces.jl:21` (`mixture_density`), `forces.jl:128` (`PoissonOperator`), `elliptic.jl:54` (`_face_coefficients`) |
| 42+18+12+9+12 | ~2500 | `indicator/phase/height_functions.jl:321–340` (`_lhf_geometry` tuples/Memory) and `:640`, `:743` |
| 34 | 16 | `indicator/phase/interface_field.jl:426` (`reconstruct!` closure tuple) |
| 17+17+11+10+10 | 10 | `indicator/phase/vof_fluxes.jl:337,390,491` (`_advect_wy!` closures / workspaces) |

Non-periodic single phase (105 KB/step @16 blocks; `alloc_np.jl`): `operators/advection/staggered_fluxes.jl:92` 71 KB (4560 boxed `Float64`/step), `:85` 24 KB (504 `Tuple{Float64,Float64}` + Float64), `:61` 4 KB — all in `_add_boundary_convection!` (reads `workspace.boundary_conditions[]` = `Ref{Any}` → `boundary, time` are `Any`). The profile of the same run also shows `_update_staggered_elliptic!` (implicit_diffusion.jl:428–451) rebuilding both component solvers on some steps (F4).

### 2.5 Type stability (`JET.@report_opt target_modules=(TwoPhaseAMR,) step!(…)` on a warm flow; `jet_*.log`)

`Base.return_types(step!)` is concrete in every case (`LinearSolverStats{Float64}`; two-velocity `Union{CMDGFMProjectionStats, MimeticProjectionStats}` = small union over the two projection variants — fine).

| case | runtime-dispatch reports | where (leaf frames) |
|---|---:|---|
| single phase periodic / non-periodic | 15 / 15 | `mesh/variables.jl:42,44` (`_nfaces`/`_nboundary_faces` `count` closure), `implicit_diffusion.jl:21–22` (`operator.x/.y` Any), `mesh/topology/mesh.jl:226–227,629`, `staggered_update.jl:324` (`_dual_volumes` → Any), kwcalls at `single_phase/stepping.jl:19,34,113` |
| (b) 1-velocity | 131 | `threading.jl:23` ×11 (`_parallel_block_columns#1`: closure type lost), ~20 sites in `indicator/phase/height_functions.jl:494–525,597–620` (`_embedded_ghf_constraint`, `_orient_ghf_geometry`: `field.geometry::Any`), `interface_field.jl:285,313,336–338,441` (geometry::Any), `two_phase/stepping.jl:41,60,64,74,106,146` (`pressure_solver.operator/preconditioner` abstract), `phase_face_fields.jl:134`, `phase_face_transport.jl:73` |
| (c)/(d) 2-velocity | 218 / 218 | all of the above + `threading.jl:23` ×16, `threading.jl:78` ×6 (`_parallel_for`), `two_velocity/stepping.jl:90,95,105,212,214,262,272,325` (`workspace.two_velocity[]` is `Ref{Any}` → `state.convected/.momentum` Any), `forces.jl:76,354` ×4, `gradient.jl:184` (`Dict{Any,Matrix{Float64}}` cache), `mimetic_joint_fit.jl:93–95`, `solvers/multigrid/cycles.jl:353` |

Measured consequence: these are per-call (not per-cell) dispatches except inside `_reconstruct_interface_cell!`; their measurable cost is the ≈100 KB/step of boxed tuples in the LHF path and — for the non-periodic case — the 100–200 KB/step in `_add_boundary_convection!` (F8). They are not the dominant time sink anywhere; `geometry::Any` cannot be shown to cost time while `geometry === nothing`.

### 2.6 Thread scaling (fresh process per point; `ths.jl`)

TG 128² (16 384 cells), ms/step by stage:

| mesh | threads | advection | diffusion | projection | step total | simulate!/call median | alloc/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat (64 roots) | 1 | 1.62 | 9.29 | 7.79 | 18.7 | 18.8 | 7.6 KB |
| | 2 | 1.17 | 9.33 | 6.94 | 17.5 | 17.4 | 34 KB |
| | 4 | 0.86 | 8.67 | 5.93 | 15.5 | 16.0 | 34 KB |
| hierarchical | 1 | 1.63 | 3.59 | 6.88 | 12.1 | 12.2 | 7.6 KB |
| | 2 | 1.34 | 3.27 | 4.88 | 9.5 | 8.4 | 38 KB |
| | 4 | 0.95 | 2.74 | 3.73 | 7.4 | 8.0 | 38 KB |

(c) CMDGFM, k=4 (7680 cells), ms/step:

| threads | transport | diffusion | refresh pressure | projection | step total | alloc/step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.57 | 31.34 | 7.53 | 10.83 | 68.8 | 21.1 MB |
| 2 | 13.84 | 36.64 | 8.28 | 9.85 | 70.9 | 21.3 MB |
| 4 | 9.02 | 30.86 | 8.82 | 8.62 | 59.4 | 21.3 MB |

Speed-up at 4 threads: TG flat 1.17×, TG hierarchical 1.53×, CMDGFM 1.16× (stages: transport 1.8×, projection 1.3–1.85×, diffusion 1.0–1.3×, pressure refresh 0.85×). Load ≈5 on the host may understate scaling; the *absence* of any gain in diffusion/refresh is nevertheless structural (serial UMFPACK, below).

### 2.7 Latency (fresh process, `latency.jl`; precompiled package, `-t 1`)

| item | stock | with PrecompileTools workload (prototype) |
|---|---:|---:|
| `using TwoPhaseAMR` | 0.56–0.58 s | 0.72–0.93 s |
| TG README: construct | 19.07 s (3.56 GB alloc, 99.9 % compile) | **0.38 s** |
| TG README: first `simulate!(flow,0.002)` | 6.50 s | **0.44 s** |
| TG second `simulate!` | 0.31 ms | 0.29 ms |
| two-phase 1-vel: construct (InterfaceField + AMR init) | 40.27 s (6.6 GB) | **1.72 s** |
| two-phase 1-vel: first step | 29.25 s (2.0 GB) | **2.23 s** |
| two-phase 1-vel: second step | 16.7 ms | 16.1 ms |
| CMDGFM / mimetic first step (stock only) | 40.7 s / 47.8 s (+ ~40 s construct) | not built (see F3) |
| package precompile | ≈5 s | 24 s (TG) / 48 s (TG + one-velocity) |

Time to first result: TG 25.6 s → 0.8 s; two-phase 69.5 s → 4.0 s. Remaining 2.2 s in the two-phase run = paths not in the prototype workload (`simulate!` driver, diagnostics, AMR transfer).

### 2.8 Prototype measurements (in `/tmp` copies; same results checked by identical iteration counts / no failing solve)

| prototype | before | after |
|---|---:|---:|
| function barrier in `_add_boundary_convection!` (non-periodic TG, 4096 / 16384 cells) | 105 504 / 207 904 B/step; 3.01 / 15.6 ms | **9 824 / 9 824 B/step**; 3.2 / 15.2 ms (time within noise) |
| coarse-level refactorization, diffusion matrix (n=1520 / 6112) | `lu` 2.42 / 11.58 ms, 2.9 / 12.2 MB | `lu!` (symbolic reuse) 1.83 / 8.86 ms; `cholesky` 1.41 / 5.55 ms; **`cholesky!` 0.84 / 2.83 ms** |
| coarse-level refactorization, pressure matrix (n=768 / 3072; singular → augmented LU) | `lu` 0.76 / 3.55 ms, 1.08 / 4.2 MB | `lu!` 0.51 / 2.45 ms |
| TG 128² mesh layout: 64 roots (flat) → 1 root + 3 levels (no code change) | 18.4 ms/step | **11.5 ms/step** (diffusion 9.0 → 3.4 ms, projection 7.7 → 6.5 ms but 1 → 3 CG its) |
| PrecompileTools workload (partial) | 25.6 s / 69.5 s to first result | 0.8 s / 4.0 s |

## 3. Findings (ranked)

| ID | severity | location | measurement | proposal | expected gain | effort |
|---|---|---|---|---|---|---|
| F1 | **High** | compile latency: no `precompile` workload anywhere (`src/TwoPhaseAMR.jl`) | TG 25.6 s, two-phase 69.5 s to first result; 99.5 % compile; package load only 0.6 s | add `PrecompileTools` `@compile_workload` covering README TG, one-velocity, CMDGFM, mimetic **and** `simulate!`; first resolve the LLVM crash (F2) | 25.6 → 0.8 s, 69.5 → 4.0 s (partial workload, measured) | M (blocked by F2) |
| F2 | **High** (reliability) | Julia 1.12.7 SIGSEGV during pkgimage emission when the workload contains `stable_timestep(flow)` (README flow) or `simulate!`/AMR-rebuild/diagnostics for two-phase | reproduced 2×: `stable_timestep(flow; cfl=0.45)` → `signal 11` in `llvm::CmpInst::CmpInst ← removeAddrspaces` (jitlayers.cpp:1459, llvm-remove-addrspaces.cpp:352) after 2 s; the component method `stable_timestep(velocity, workspace.xworkspace)`, `step!`, `project!` and `_accumulate_staggered_rates!` precompile fine. Also 3 of ~40 fresh runs died with `signal 11` in `gc_mark_objarray` during type inference (`batch_inline!`) of cmdgfm/mimetic first calls; not reproducible on rerun. No `unsafe_`/`ccall` in `src/` | continue the bisect (the crash lies between the component method and the flow-level call: candidate `stable_timestep(::FaceVelocity, ::StaggeredVelocityAdvectionWorkspace)`, `staggered_update.jl:433–442`, `min` over two `Axis` specializations — [INFERENCE], not isolated), then restructure or file upstream; run once with `--check-bounds=yes` to rule out out-of-bounds writes (not done: needs a separate package image build) | prerequisite for F1; removes flaky first-run crashes | S–M |
| F3 | **High** | `solvers/multigrid/geometric_preconditioner.jl:850–856,993–995`; `cycles.jl:50–73` (`_coarse_solve!`); `navier_stokes/implicit_diffusion.jl:68,412`; `two_phase/variable_diffusion_preconditioner.jl:262,337` | the GMG coarsens only through AMR tree levels; base-mesh cells are LU-factorized as a whole. Flat 128²: 1 level, n=16 385, nnz(L+U)=1.33 M, 1.5 ms per application, 49.6 % of the step in `umfpack_dl_wsolve`; factorization 49 ms. 64²: n=4097, 0.30 ms/solve. Capillary k=2: levels [3456, 1152, **768**]; k=4: [7680, 4608, **3072**] (diffusion [15424, 9216, **6112**]). Step 16.8 → 54.7 ms for 2.2× cells (N^1.5). Serial (no thread gain) | coarsen the base level too (2×2 agglomeration of root blocks by rediscretization on coarser uniform grids, as done for tree levels, or algebraic coarsening via the optional AMG extension) until ≤ `coarse_size`; interim: document/create uniform meshes as 1 root + `max_level` refinement | TG 128²: 1.6× (measured via mesh layout); two-phase: removes the coarse-solve share (est. 10–25 %) and the N^1.5 scaling | M–L |
| F4 | **High** (mimetic only) | `projection/mimetic_projection.jl:828–880` (`_assemble_mimetic_system!`), `:1075–1130` (`project!`), `:114–123` (`ldiv!`), `mimetic_foundation.jl:11` (`_mimetic_incidence`) | 90.9 of 108.5 ms/step (84 %), 114 of 120 MB/step; 323 MINRES its/step (130–401; cap 500); profile: MINRES 48 %, `cholmod_l_solve` 38 % self (preconditioner applied through allocating `\`), assembly 28 % (`PoissonOperator` rebuild 9.9 %, incidence 5.0 %, joint constraint 10.7 %); `maximum(abs, matrix - transpose(matrix))` (l.1091) and a fresh `MinresWorkspace` per refinement round (l.1109) and `zeros(...)` solution (l.1094) every step | cache `_mimetic_incidence` (topology only) and reuse the `PoissonOperator` built in `_refresh_pressure_substep!`; compute the symmetry error only when diagnostics ask; in-place `ldiv!(x, F, b)` for the block Cholesky; reuse one `MinresWorkspace`; warm-start from previous multipliers [INFERENCE: effect on iteration count not tested] | caching alone ≈ 15 % of step (incidence 5 % + operator 9.9 %); allocation 135 → well under 50 MB/step; iteration reduction is the large lever | M |
| F5 | **High/Med** | per-step refresh: `two_phase/variable_diffusion_preconditioner.jl:329–338`, `variable_diffusion_solver.jl:195–230`; `geometric_preconditioner.jl:919–998`; `two_phase/stepping.jl:41–77` (`_refresh_pressure_substep!`, builds a new `PoissonOperator` each step: `elliptic.jl:31,54`, `forces.jl:21,128`) | 1-vel: refresh of diffusion hierarchy 29.6 % of step, `lu` 32.1 % inclusive; CMDGFM: 18.3 % + 8.6 %; 4.0 of 5.5 MB/step is `lu` factor storage; k=4 (CMDGFM): diffusion 31.3 + pressure refresh 7.5 of 68.9 ms | (a) `cholesky!` (diffusion matrix SPD) and `lu!` (pressure) with symbolic reuse; (b) skip the coarse refactorization when the coefficient change is below a tolerance (a stale factorization is still a valid preconditioner; CG corrects it); (c) reuse `_cell_references` | (a) 11 % (k=2) – 18 % (k=4) of step, measured microbenchmarks; (b) 19–28 % | S–M |
| F6 | **Med** | `navier_stokes/simulate.jl` (`step_size = min(proposed, target_time - current_time)`, `selected_step` re-chosen per call when `dt=nothing`) + `implicit_diffusion.jl:428–433` (`value == solver.operator.diffusion` exact float compare → rebuild of both component hierarchies + Krylov workspaces) | bitwise change of dt ⇒ full viscous re-factorization. Flat TG 128²: a 25-step `simulate!` call costs 21.9 ms/step vs 17.7 (≈105 ms, 85 MB extra at the last step); with `dt=nothing` 37.3 ms/step (≈3 events, 272 MB) vs 17.5. Non-periodic 64²: mean 5.24 vs median 3.01 ms. Hierarchical meshes: 12.0 vs 11.6 ms (cheap). Chunked driving (checkpoint/visualization `live_dt`) pays this per chunk | (a) treat the last step as the regular `dt` when `remaining` is within 1e-9·dt; (b) update the operator shift in place and refactor the coarse level only when `|Δshift|/shift` > ~5 % | removes ≈100 ms per `simulate!` call at 16 k cells flat | S |
| F7 | **Med** | `indicator/phase/interface_field.jl:405–480` (`reconstruct!`, `passes = 3` at l.451, all-cell loops `_reconstruct_interface_cell!` l.325), `height_functions.jl:321–340` (`_lhf_geometry` returns allocated tuples/`Memory`), `two_velocity/velocity_fields.jl:35–50` (`_mark_inactive_phase_velocities!`, serial) | `reconstruct!` 23.3 % (1-vel), 19.1 % (CMDGFM) incl.; `_reconstruct_interface_cell!` 8.3/7.5 % self; ≈100 KB/step of boxed LHF tuples; `_mark_inactive_phase_velocities!` 5.1 % self (CMDGFM, serial) | iterate only over cells with 0<c<1 (a list, built once per step) instead of every cell × 3 passes; return `SVector`/NTuple from `_lhf_geometry` without heap boxing; `@_parallel` the face loop in `_mark_inactive_phase_velocities!`; concretize `InterfaceField.geometry` | est. 5–10 % of two-phase step [INFERENCE for the exact amount] | M |
| F8 | **Med** | threading: serial UMFPACK (F3/F5), serial streaming loops `projection.jl:72–79`, `implicit_diffusion.jl:643–646`, `variable_diffusion_solver.jl:203,242,269,343`, `elliptic.jl:688,776,823,849,859`, `staggered_update.jl:111,191–201,295–301,335,350`, `staggered_fluxes.jl:513–521`; known serial `_phase_face_average!` | 4 threads: TG flat 1.17×, TG hier 1.53×, CMDGFM k=4 1.16×; stage speed-ups: transport 1.8×, projection 1.3–1.85×, diffusion 1.0–1.3×, refresh 0.85×. Threaded runs allocate 34–39 KB/step vs 7.6 KB (Polyester tasks) | fix F3/F5 first (they are the serial 45–60 %); then wrap the loops above in `@_parallel` | with F3/F5 fixed, remaining serial fraction ≈ 10–15 % → ≥2.5× @4 threads expected [INFERENCE] | S (loops) + F3/F5 |
| F9 | **Low–Med** | `operators/advection/staggered_fluxes.jl:52–96` `_add_boundary_convection!` (`workspace.boundary_conditions[]::Any` → `boundary, time` boxed) | non-periodic single phase: 105 504 B/step @16 blocks, 207 904 B @64 blocks (4560 + 504 + 504 + 256 allocations/step) | function barrier (split into outer unpack + inner concrete method), prototype in `/tmp/audit-perf/pkg_b` | 105/208 KB → 9.8 KB/step (−91…−95 %); time ≈ −2 % | S |
| F10 | **Low–Med** | `navier_stokes/workspace.jl:68–` `_rebuild_topology_dependent_state!` | 45–238 ms and 61.5 MB per call at 3456 cells (3–15 steps of work); first-call 3.3–4.9 s compile; known earlier: ≈16× a pressure solve at 26 k cells | already identified (Dict pointer lookups, construction caches); not re-derived | — | M |
| F11 | **Low** | `adapt!` evaluates `refinement_labels!` every step before checking `update_interval` (`mesh/adaptivity/adapt.jl:88–100`, `_adapt_with_labels!` checks interval after labels) | 0.074 ms/step (0.5 % of a 3456-cell step); grows with interface-cell count | check the interval before computing labels | <0.5 % | S |
| F12 | Info | README quick start | with `dt=1e-3` the divergence is already below the tolerance (`cycles=0`): the README example exercises no pressure solve | use a CFL-scale dt in the example/benchmarks (`benchmark/common.jl` uses `dt=5e-4` and also does 0 cycles: `benchmark_step!` showed `cycles=0` in `a3.jl`) | benchmark validity | S |

### Top-5 hot spots (measured)

1. **Coarse-level direct LU / UMFPACK solve** — `geometric_preconditioner.jl:850–856,993–995`, `cycles.jl:50`: 49.6 % of single-phase step (flat 16 k), 20–32 % of two-phase steps; superlinear growth (F3).
2. **Mimetic `project!`** — `mimetic_projection.jl:828–880, 1075–1130`: 90.9 ms of 108.5 ms/step, 114 MB/step (F4).
3. **Per-step diffusion/pressure preconditioner refresh** — `variable_diffusion_preconditioner.jl:329–338`, `geometric_preconditioner.jl:919–998`: 30 % (1-vel) / 27 % (CMDGFM) of step, 4 MB/step (F5).
4. **`reconstruct!`** — `interface_field.jl:405–480`: 19–23 % of two-phase step (F7).
5. **Compile latency + dt-jitter refactorization** — whole-package first calls 25.6–69.5 s (F1); `implicit_diffusion.jl:428–433` ≈105 ms per `simulate!` call on a flat 16 k mesh (F6).

## 4. Details and caveats

* **Cause of the “≈1 KB/cell/step”.** Single phase: 4352 B/step constant for 4096 and 16 384 cells (periodic). Non-periodic: scales with the number of boundary faces (not cells) — F9. Two-phase: 1.66–1.73 KB/cell (k=2), 2.8 KB/cell (k=4) because `lu` + hierarchy refresh allocate O(coarse-level nnz), which itself grows ~N^1.5 (F3, F5). The earlier 116 KB@64 cells / 310 KB@256 / 1.03 MB@1024 figures were not reproduced with the benchmark fixture either (`benchmark/common.jl` periodic=(true,false): 17 KB@256 cells, 20.5 KB@448 mixed); I did not run them with multiple threads.
* **`enforce_velocity_boundary!` 43.8 KB/call** (Ref{Any} + key vector): not reproduced for `FreeSlip()`/`NoSlip()` (0.2 KB/call, 1 µs; `solver.boundary` is not a `VelocityBoundaryConditions` there). That path (`staggered_diffusion.jl:85–110`) is only taken with `Inflow/Outflow/ZeroGradientVelocity` and was not exercised.
* **Per-step time is not AMR-dominated in steady state**: 0 of 40 steps changed topology; `adapt!` attempt 0.074 ms/step; the topology rebuild (F10) is the AMR cost when it occurs.
* **Not measured:** (i) AMR field transfer cost for an organic topology change; (ii) `--check-bounds=yes` run (F2); (iii) a mimetic/CMDGFM workload in PrecompileTools (blocked by F2); (iv) thread scaling beyond 4 threads / at the mimetic case; (v) problem sizes >16 k cells (instruction cap ~30 k; flat n=192² would be 36 k).
* **Accounting detail.** Stage timers (`stages.jl`) re-implement `step!` from the package’s own internal functions; their sum matched the `simulate!`-per-call median within ≤5 % (e.g. 16.30 vs 15.64 ms; 3.36 vs 3.54 ms). Stage timing for mimetic had one non-converged-looking range (130–401 its) — all steps converged (the driver throws otherwise).
* **Run configuration:** TG 4096/16384 cells, two-phase 3456 cells (k=2) and 7680 cells (k=4, CMDGFM thread scaling only); threads 1/2/4; Julia 1.12.7; AMD Ryzen 9 5950X; shared machine (load ≈5).
