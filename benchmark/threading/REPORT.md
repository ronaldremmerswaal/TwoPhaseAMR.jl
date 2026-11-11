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

| case | 1 thread | 2 threads | 4 threads | speed-up 2 / 4 | alloc/step 1 → 4 threads |
|---|---:|---:|---:|---:|---:|
| TG 128² hierarchical (40 steps) | 13.56 | 9.71 | 8.02 | 1.40× / **1.69×** | 6.4 KB → 39 KB |
| capillary CMDGFM k = 4 (30 steps) | 51.05 | 38.79 | 32.76 | 1.32× / **1.56×** | 1.97 MB → 2.17 MB |
| cylinder (100 steps) | 3.15 | 2.55 | 2.23 | 1.24× / **1.41×** | 81 KB → 119 KB |

Audit baseline (pre-4B–4E.3): TG hierarchical 1.53×, CMDGFM 1.16× at 4 threads.
The capillary step alternates between cheap and expensive steps (min 40.7 ms,
median 51.0 ms at 1 thread), so its median is the noisier number.

**Determinism:** the checksum of the final state is identical at 1, 2 and 4
threads for all three cases (`73b29d3e…`, `e67ec5aa…`, `9165022c…`): every
current parallel loop owns its outputs and all sums are serial.

## Serial fraction

Amdahl estimate from the 1- and 4-thread medians, s = (4/S₄ − 1)/3:
TG 0.46, capillary 0.52, cylinder 0.61. The 4-thread profile (main thread,
0.5 ms sampling, TG 150 steps, capillary 60, cylinder 400) agrees: the main
thread is outside any Polyester batch for **59.7 %** (TG), **66.0 %**
(capillary) and **76.5 %** (cylinder) of the wall time.

Serial samples by cause, % of the 4-thread wall time (innermost package frame,
grouped):

| cause | TG | capillary | cylinder |
|---|---:|---:|---:|
| ordered sums: multigrid component means / RHS projection (`hierarchy.jl:102,121`), `_remove_component_volume_means!`, L2 residual norms, Krylov.jl vector ops (`_run_krylov!`) | ≈ 23 | ≈ 9 | ≈ 6 |
| sparse transfer `mul!` (`_restrict!`, `_prolongate!`) | 4.2 | 4.5 | 6.3 |
| other elementwise/scatter loops of the solvers: residual formation (`_true_residual!`), `_roundoff_residual_floor!`, `diffusion!`/`_apply_staggered_diffusion!`, `_add!`, field↔vector copies, `apply!` | ≈ 12 | ≈ 4 | ≈ 9 |
| inlined projection/stepping loops (`step!` → `stepping.jl:254`, `divergence!`) | 7.7 | – | 6.7 |
| two-phase loops: `_mark_inactive_phase_velocities!` | – | 12.2 | – |
| two-phase loops: `_frozen_interface_stable_timestep` (max reduction) | – | 8.8 | – |
| two-phase loops: `_refresh_multigrid_level!`, variable-diffusion refresh | – | ≈ 7 | – |
| two-phase loops: `_staggered_liquid_fraction`, `_phase_face_average!`, `_projection_correct_phase_velocities!`, `_extend_phase_face_values!`, CMDGFM residual/`mul!` | – | ≈ 9 | – |
| coarse direct solves / refactorization (UMFPACK, ILU) | 0.5 | 3.5 | 4.1 |
| diagnostics (example callbacks `pressure_drag/lift`, vorticity) and AMR labels | – | – | ≈ 37 |

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
