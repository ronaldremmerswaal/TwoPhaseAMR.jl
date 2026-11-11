# Thread scaling on `main` (phase 4E.4, step 1)

Baseline for the threading work: `main` at `6a7a723` (after #139 and #141),
measured 2026-11-11 on the shared AMD Ryzen 9 5950X (16C/32T, 31 GB), Julia
1.12.7, BLAS 1 thread, one fresh process per point. Multi-threaded runs started
only when available memory was ≥ 8 GB and the 1-minute load average was < 4
(`gate.sh`); the gate reported load 3.3–3.8, so absolute times carry the
noise of a shared host.

```sh
benchmark/threading/gate.sh   # before every multi-threaded run
OPENBLAS_NUM_THREADS=1 julia --project=. -t N benchmark/threading/scaling.jl CASE STEPS PROFILE_STEPS
```

`scaling.jl` builds the fixture, takes 5 warm-up steps, times `STEPS`
fixed-`dt` steps (one `simulate!` call each) and prints a checksum of the final
velocity and pressure. With `PROFILE_STEPS > 0` it profiles further steps and
splits the samples of the main thread into *serial* samples and samples inside
a Polyester batch (any Polyester/ThreadingUtilities frame on the stack).

## Fixtures

| case | setup | cells | blocks | dt |
|---|---|---:|---:|---:|
| `taylor_green` | 128² TG, hierarchical: one 16² root refined to level 3, ν = 0.02, periodic | 16 384 | 64 | 0.45 CFL = 1.105e-2 |
| `capillary` | capillary wave k = 4 (example 06 setup, base mesh 32×96 cells, 8² blocks, interface AMR `max_level = 2`), `TwoVelocityModel(CMDGFMProjection())` | 7 680 | 120 | 0.5 × stable = 9.74e-5 |
| `cylinder` | example 04 `run_example(final_time = 0)` (default `max_level = 3`, Re 100, embedded cylinder, wake AMR, example diagnostics) | 2 944 | 184 | 0.45 CFL = 6.48e-4 |

No topology change occurred in any timed window (block counts unchanged).

## Results (ms/step, median of the timed steps)

Raw output of every point, including per-step times and the profiles, is in
`logs/main/CASE_tN.log` (each starts with the `uptime` line read just before
the run).

| case | 1 thread | 2 threads | 4 threads | speed-up 2 / 4 | alloc/step 1 → 4 threads |
|---|---:|---:|---:|---:|---:|
| TG 128² hierarchical (40 steps) | 13.53 | 9.81 | 8.43 | 1.38× / **1.60×** | 6.4 KB → 39 KB |
| capillary CMDGFM k = 4 (30 steps) | 50.63 | 39.97 | 33.55 | 1.27× / **1.51×** | 1.97 MB → 2.17 MB |
| cylinder (100 steps) | 3.14 | 2.54 | 2.30 | 1.24× / **1.37×** | 81 KB → 119 KB |

Audit baseline (pre-4B–4E.3): TG hierarchical 1.53×, CMDGFM 1.16× at 4 threads.
A first, identical run of the same points an hour earlier gave TG 1.69×,
capillary 1.56×, cylinder 1.41× (load 3.3–3.8 both times); the spread between
the two runs is the noise of the shared host.

The capillary steps are not uniform, but the pattern is the same at every
thread count: steps 18, 25 and 29 of the timed window are about 20 % cheaper
than the other 27 (40.4 vs ≈ 50.5 ms at 1 thread, 26.2 vs ≈ 33.5 ms at
4 threads, see `step_ms` in the logs). The median is therefore the cost of a
regular step at every thread count; the mean (49.9 / 39.2 / 33.2 ms) gives the
same speed-ups (1.27× / 1.50×).

**Determinism:** the checksum of the final state is identical at 1, 2 and 4
threads for all three cases (`73b29d3e…`, `e67ec5aa…`, `9165022c…`): every
current parallel loop owns its outputs and all sums are serial.

The CI-runner confirmation of the brief is done once for the final numbers
(after step 4), together with these baselines.

## Serial fraction

Amdahl estimate from the 1- and 4-thread medians, s = (4/S₄ − 1)/3:
TG 0.50, capillary 0.55, cylinder 0.64. The 4-thread profile (main thread,
0.5 ms sampling; TG 150 steps, capillary 60, cylinder 400) agrees: the main
thread is outside any Polyester batch for **60.6 %** (TG), **65.1 %**
(capillary) and **75.2 %** (cylinder) of the wall time.

Serial samples grouped by cause, % of the 4-thread main-thread samples. Each
entry is the sum of the listed functions' "serial self by function" lines in
`logs/main/CASE_t4.log` (innermost package frame of a serial sample; Krylov.jl
vector operations are attributed to `_run_krylov!`, which calls them).

| cause (functions) | TG | capillary | cylinder |
|---|---:|---:|---:|
| ordered sums: `_multigrid_remove_component_means!`, `_multigrid_project_component_rhs!`, `_remove_component_volume_means!`, `_project_component_rhs!`, `_integrated_residual_norms`, `_cmdgfm_scaled_residual_norm`, `_run_krylov!` | 24.4 | 8.7 | 3.1 |
| multigrid transfer `mul!`: `_restrict!`, `_prolongate!` | 6.7 | 4.5 | 5.8 |
| other solver loops: `_roundoff_residual_floor!`, `_true_residual!`, `_add!`, `_run_krylov_reliably!`, `_apply_staggered_diffusion!`, `_apply_vcycle_preconditioner!`, `ldiv!` (implicit diffusion), `_copy_to_field!`/`_copy_from_field!`, `_parallel_csr_row_mul!` below the grain | 16.2 | 0.4 | 18.2 |
| stepping loops: `step!` (inlined), `copyto!`, `weighted_gradient!`, `divergence!`, `_copy_velocity!` | 9.3 | 0.3 | 1.4 |
| two-phase loops: `_mark_inactive_phase_velocities!` 8.8, `_frozen_interface_stable_timestep` 7.0, `_staggered_liquid_fraction` 3.2, `_projection_correct_phase_velocities!` 2.5, `_phase_face_average!` 1.8, `_extend_phase_face_values!` 1.8 | – | 25.1 | – |
| hierarchy refresh: `_refresh_multigrid_level!`, `_refresh_tree_variable_diffusion_preconditioner!`, `_refresh_elliptic_components!`, `_refresh_variable_diffusion_matrix_from_inertia!`, `_impose_inactive_constraints!`, `_jacobi_weight!` | 0.8 | 8.2 | – |
| CMDGFM projection: `mul!`, `ldiv!`, `_project_cmdgfm_assembled!`, `_cmdgfm_matrix_asymmetry`, `_cmdgfm_initial_solution!` | – | 4.3 | – |
| direct coarse solves / refactorization: `_coarse_solve!`, `_refactorize_coarse!`, ILU | 0.4 | 3.8 | 3.7 |
| diagnostics and AMR labels: `diagnostic_value` (example `pressure_drag`/`lift` callbacks), `_record_diagnostics!`, `_local_plic_segment`, `_vorticity_from_centered!`, `_cell_centered_velocity!`, `_protect_fixed_band_coarsening!` | – | – | 33.1 |
| sum of the rows | 57.8 | 55.3 | 65.3 |

The remaining serial samples (TG 2.8, capillary 9.8, cylinder 9.9 points) are
field accessors attributed to their callers' loops (`_get`: capillary 0.9,
cylinder 2.3) and entries below 0.6 % each.

## Consequences for steps 2–5

- The largest single serial item in TG is an *ordered* sum: the volume-weighted
  component mean removed on every multigrid level in every V-cycle. Together
  with residual norms and Krylov dots, ordered sums are about a quarter of the
  4-thread TG step. Parallelising them with thread-count-independent chunks
  changes the summation order also at 1 thread (one-time oracle re-record,
  orchestrator decision 2026-11-11); the cost/benefit is reported separately.
- Everything else in the table can be parallelised bitwise-neutrally: owned
  outputs, gathers that keep each output's accumulation order, and max
  reductions. Step 4 order, by measured share: (1) multigrid transfer `mul!`
  and solver elementwise/scatter loops (TG, cylinder), (2) two-phase
  `_mark_inactive_phase_velocities!`, `_frozen_interface_stable_timestep`,
  multigrid/variable-diffusion refresh, `_staggered_liquid_fraction`,
  `_phase_face_average!` and the phase-velocity corrections (capillary),
  (3) ordered sums with fixed chunking, (4) `surface_tension!` curvature lookup
  (not visible in this profile; kept for the audit item F23).
- The cylinder is dominated by small meshes (2 944 cells, mostly below the
  1024-cell threading grain per loop) and serial example diagnostics; it is
  reported but has no target.
