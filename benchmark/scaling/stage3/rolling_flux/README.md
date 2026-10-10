# Copied-source rolling regular-flux experiment

This is a research-only experiment against the frozen checked narrow-ID
advection source (`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`).
`make_probe.jl` copies that source and changes ordinary `advect!` on every AMR
leaf. Production code, cut-cell advection, and `stable_timestep` remain
unchanged. The generated patch and source hashes identify the candidate;
timings must use that exact copied tree.

The candidate computes owned residual, leaf-perimeter, and tile-seam fluxes
from the unchanged scalar field. Each tile then calculates its remaining
regular faces once into one x row and two alternating y rows of
workspace-owned scratch. It stores the ordered integrated sum temporarily in
`workspace.increment`. A separate pass divides by the block volume, writes
the final increment, and applies the **unrounded promoted change** to the
scalar. This preserves the original Float32/Float64 update semantics. The
full x/y flux arrays remain allocated for the unchanged cut-cell and CFL
paths, but ordinary advection touches them only for perimeter/seam slots.
This is a working-set/array-access hypothesis, not a measured DRAM-byte or
speedup claim.

Source-level slot accounting illustrates the tradeoff. At 1024² in one
block with eight workers, the plain path touches 39,862 of 2,095,104 regular
full-array slots (about 1.9%) and allocates 98,272 extra `T` scratch slots
for 32 tiles. With 8×8 one-tile leaves it touches 52 of 112 regular slots
per leaf (about 46%); 28 of 64 cells use the edge incidence path. The new
path also writes integrated values, then reads and rewrites increments in a
second pass before the scalar update, with synchronization between passes.
It still loads the four operands of each cell sum, and cache behavior and
write allocation determine actual memory traffic.

`validation_driver.jl` loads the copied module, checks its source and method
paths, runs the existing independent compact reference suite, then includes
`focused_validation.jl`. The focused tests cover non-square one-leaf tile
partitions, 2×N and N×2 mixed-level all-edge leaves, physical and periodic
boundaries, live face values, exact scalar and increment fields, and a
Float32/Float64-volume witness where updating from rounded increment would
change bits. The frozen focused file is also exercised at three and seven
workers to cover partial final tiles.

`make_count_probe.jl` creates a **different, untimed copied tree** by adding
per-face atomic counters to the common ordinary regular-flux helper.
`count_validation.jl` requires exactly one call for every regular owned face
ID and zero for residual face IDs on small one-leaf and mixed-AMR meshes.
The instrumented source has a separate hash and is never used for timing.

The reusable Julia `../copied_source/study.jl` measures the uninstrumented
candidate against a fresh copied control, one case per fresh child process.
It checks mesh faces and halo entries before timing, uses a fresh control
oracle and bitwise full-field comparison, pins source/driver/project/input
hashes, and records adjacent randomized pairs at one, four, and eight
workers. Fixed-`dt` `advect!` and per-step-CFL workloads are distinct modes;
the latter includes `stable_timestep` inside every timed step. Accept a
cohort only when all planned rows and receipts pass the independent
`../copied_source/audit_study.jl` audit. Report nanoseconds per cell per
complete `advect!` or CFL-inclusive step as appropriate, plus efficiency
against the faster compatible one-worker median. The Stage 3 plan calls for
five independent process pairs and large-layout numerical/performance gates;
a shorter pilot is exploratory.

The audited three-pair fixed-`dt` pilots rejected this two-pass candidate.
For 1024² with 8×8 blocks, control→candidate median nanoseconds per cell
per `advect!` were 12.903→20.105 (p1), 4.959→6.192 (p4), and
3.067→5.315 (p8). With one 1024² block, they were 8.472→9.591,
2.125→2.357, and 1.585→1.785 respectively. Every one-block paired
comparison favored control; the eight-block p8 parallel-efficiency point
estimate was 30.3% against the faster serial control. Both completed cohorts
passed the independent receipt and bitwise-field audit. These are
fixed-`dt` complete `advect!` timings, not CFL-inclusive step timings. The
candidate remains research-only; no further two-pass timing is planned.
