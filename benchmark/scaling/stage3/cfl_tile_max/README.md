# Copied-source tile-max CFL study

This research-only study compares the original checked production source
(`047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541`)
with a separately generated tile-max candidate. `make_probe.jl` must produce
the candidate source, one patch, and `cfl-tile-max-receipt.json`. Its frozen
`validation-runs.toml` must include source-locked control and candidate p1
compact/focused suites. The suite reports use the actual
`tile_max_method_file` and `tile_max_expected` keys; the method must resolve
to the copied advection file in the candidate and be absent in control.

`setup_gate.jl` reuses the archived copied-source suite checks and checked
CFL driver. It launches one fresh setup process for each source at 1, 4, and
8 threads on a periodic 1024² one-block mesh. Every face/halo check and
workspace construction must pass. A failure leaves process receipts but no
passed gate, and stops this candidate cohort without a retry. The passed
gate pins the source trees, suite inputs, generator, driver, runtime, project,
and placement.

`study.jl` requires that exact passed gate and delegates the numerical
comparison to the archived `../copied_source/study.jl` with `--workload cfl`.
For the first pilot use `--case 1024:1024 --steps 256 --repeats 3`; it runs a
fresh control oracle and randomized adjacent control/candidate pairs at
1, 4, and 8 threads. Each timed step scans
`stable_timestep(velocity, workspace; cfl=0.9)`, chooses `dt=0.4*bound`,
then calls complete `advect!` on the prescribed constant carrier. The wrapper
requires every row's bound, dt, and full-field hash to match the oracle.
The fixed-`dt` trajectory is different and must not be pooled with this one.
Independently check a completed setup with `audit_study.jl setup PASSED_GATE`,
then a completed timed cohort with
`audit_study.jl measure COMPLETED_OUTPUT PASSED_GATE`.

If the complete three-pair pilot is promising and independently audited, a
separately predeclared two-pair cohort can use `--repeats 2` with the same
gate and source/driver/input identity. Pooling three plus two pairs requires
an independent cross-cohort audit. A smaller many-block layout, if later
chosen, is diagnostic and cannot replace the failed 1024:8 setup/performance
gate. No 2048 case or repeated setup-until-pass process is part of this
study. Generated data belongs under ignored `benchmark/scaling/results/`.
