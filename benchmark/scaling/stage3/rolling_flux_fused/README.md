# Row-fused rolling-flux copied-source probe

`make_probe.jl` derives this research-only ordinary general-AMR upwind
candidate from the frozen two-pass rolling source (advection SHA-256
`9383d5b7800300353b2f054efac51b14be48f16aeaabf3eec886f4dc1eeb85b6`).
Its receipt also pins the original checked narrow-ID control (advection
SHA-256 `047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`).
The generated candidate is advection SHA-256
`d3d8edeb29cef090ca0f4168ba9103f44346425c1d55be40f80905f4e2e21027`.
No production source is changed.

The two-pass parent's global prepass computes residual, physical/periodic,
leaf-perimeter, and cross-tile seam fluxes from the unchanged scalar field.
Within each disjoint tile, the fused loop computes a complete x-flux row and
the next y-flux row before writing scalar row `j`. The previous y-flux row is
already cached; a tile-boundary y face was frozen by the prepass. Thus the
write cannot change an old-state read needed later in that tile or another
tile. For each cell, the original packed/CSR flux order forms `integrated`,
then `change = integrated / volume` is stored in `workspace.increment` and
added directly to the scalar. This preserves promoted Float32/Float64
arithmetic rather than updating from the rounded stored increment. The
separate full-cell update pass and its barrier are removed. This is a
source-level dependency argument, supported by finite fixtures; it is not a
proof for every reachable mesh. Cut-cell advection and CFL are unchanged.

`validation_driver.jl` loads a copied source and runs the original compact
advection suite plus the shared `rolling_flux/focused_validation.jl`. The
focused suite checks exact fields and increments on non-square, mixed-AMR,
physical/periodic, precision-promotion, and partial-tile fixtures. The
candidate passed at one, three, four, seven, and eight workers; the copied
control passed at one worker. The frozen manifest is
`benchmark/scaling/results/20261010-rolling-flux-fused-candidate/validation-runs.toml`
(ignored raw artifact). `make_count_probe.jl` creates a **separate untimed**
instrumented copy. The shared `rolling_flux/count_validation.jl` requires
exactly one regular helper call per regular face ID and zero per residual
face, including mixed-AMR and partial-tile worker counts. Its one-, four-,
seven-, and eight-worker receipts are under
`benchmark/scaling/results/20261010-rolling-flux-fused-call-count/`.

`preflight.jl` checks the parent/control/source/suite chain without mesh
construction. `study.jl` wraps the archived generic copied-source runner,
checks parent identity before and after its fresh-process cohort, and writes
`parent-verification.json`. Any timings use the original checked narrow-ID
source as control, not the slower two-pass parent. The independent
`copied_source/audit_study.jl` checks completed receipts and full fields.

The first 1024² many-block fixed-`dt` cohort stopped at its first fused
child's workspace construction, after mesh validation and before a candidate
field or timing report. Its control oracle and one isolated control p8 row
completed, but no matched pair or study summary exists. The failure cause is
unknown; the candidate has **no accepted throughput result**. The incomplete
ignored receipt is
`benchmark/scaling/results/20261010-rolling-flux-fused-1024b8-fixed-r3/`.
No retry or production edit followed.
