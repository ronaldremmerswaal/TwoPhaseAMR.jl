# Brief: regenerate the example gallery after the numerics fixes

Start after #141 (ProjectedLiquidVelocity removal) is merged. Default implementer model.

## Why
Since the gallery was last generated, the numerics changed: outflow projection (#93),
nondimensional two-velocity defects (#94), split-domain gauges (#95/#97), grid-aligned
embedded walls (#106), energy-contractive single-phase diffusion (#80), AMR indicator
fixed point and curvature in cut cells (#117/#118), cut-donor transport bound (#52/#56),
solver selection (#126/#128), one-phase cfl range (#121), removal of the mimetic
projection and of the projected-liquid carrier. README numbers and assets of examples
01–12 are stale (some README pages still carry "predates …" notes).

## Work
- Worktree from `origin/main` (e.g. `/home/ronald/Documents/wt-gallery`, not /tmp:
  a reboot wipes /tmp); branch `examples/regenerate-gallery`.
- For every example `examples/NN_*/make_figures.jl` (docs-size and default/README
  size as the scripts define): regenerate assets, update the README numbers
  (tables, measured values, run cost rows: Julia version, machine, 1 thread),
  remove stale provenance notes. LGPI: scale 20, level 3, MC, full length (~2 h).
- GIFs ≤ 2 MB; light and dark heroes where the example has them.
- Verify every GIF from the file itself with a contact sheet (frames sampled
  over the whole length) and every PNG by viewing it; write contact sheets to
  `/home/ronald/Documents/gallery-contact/` and list them in the report.
- Report per example: what changed visibly or numerically vs the previous assets
  (old values from git), anything that looks physically wrong.
- `examples` test group, docs build; Runic on changed `.jl`.

## Machine
One Julia process at a time, 1 thread (`OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1
julia --heap-size-hint=3G -t 1`), wait if `free -g` available < 6 GB.
Commit after each example (a reboot loses uncommitted files).
