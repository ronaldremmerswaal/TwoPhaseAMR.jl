# Current jump-projection development tools

Production supports `CMDGFMProjection()` and two layouts of the same
`MimeticProjection`: `variant=:joint_smoothed` (default) and
`variant=:joint_face`. Both mimetic layouts use the full active jump space,
local normal/tangential fitting, and the symmetric KKT projection.

The current accuracy comparison is
[`INSTABILITY_ACCURACY_MIMETIC_OVERVIEW.md`](INSTABILITY_ACCURACY_MIMETIC_OVERVIEW.md).
Its raw data are in
[`instability_accuracy_results_both_mimetic.tsv`](instability_accuracy_results_both_mimetic.tsv),
and [`instability_accuracy_overview.jl`](instability_accuracy_overview.jl)
resumes missing cases without rerunning completed ones. The report compares
CMDGFM with both mimetic layouts under mesh and time-step refinement and
includes energy history.

The maintained smaller probes are:

- [`kh_refinement.jl`](kh_refinement.jl): KH mesh/time sweeps;
- [`frozen_kh_response.jl`](frozen_kh_response.jl): frozen analytic and
  transported KH predictors;
- [`kh_joint_diagnostics.jl`](kh_joint_diagnostics.jl): stepwise KKT and
  energy diagnostics;
- [`kh_operator_continuity.jl`](kh_operator_continuity.jl): sensitivity to
  small interface shifts;
- [`normal_consistency.jl`](normal_consistency.jl),
  [`inspect_normal_rows.jl`](inspect_normal_rows.jl), and
  [`poisson_jump_audit.jl`](poisson_jump_audit.jl): local and manufactured
  pressure-jump checks.

Earlier model studies and intermediate raw sweeps are retained in
[`archive/`](archive/README.md). They describe superseded variants and are
not part of the current production API.
