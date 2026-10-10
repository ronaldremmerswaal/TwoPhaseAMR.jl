# Bounded ordinary-workspace setup check

The first fused rolling-flux timing attempt stopped during ordinary
`AdvectionWorkspace` construction, in `_grouped_output_schedule` before the
advection timer. The observed location does not establish a defect in
`sortperm`. Both the original control and fused copies build the incidence
vectors and call the same unchanged schedule routine before fused tile scratch
is allocated.

This fixture copies the frozen control and fused source trees, then adds gated
Julia assertions around the unchanged sort and CSR schedule construction. The
driver validates the periodic 1024² mesh with 8×8 leaves, checks aligned
incidence vectors and target bounds before sorting, and checks permutation
bijection, CSR groups, and stable within-cell incidence order afterward. It
constructs one workspace per fresh process. It does not advance a field or
time advection. The runner fixes eight Julia workers and the same CPU placement
`[0, 8, 1, 9, 2, 10, 3, 11]` used by the failed cohort, and launches exactly
one control and one fused child regardless of the first child's result.

The two predetermined children both passed with empty stderr and exit code 0.
Each observed 16,384 blocks, 2,097,152 faces, 1,048,576 cells, and 1,835,008
edge incidences; the target, permutation, group, and workspace assertions all
passed. The ignored result directory is
`benchmark/scaling/results/20261010-workspace-setup-p8-assertions-once/`;
its compact receipt is `setup-validation-receipt.json` (SHA-256
`44d33b3ce4e6822d9ee28bb4c6bd0d3b6b5eafaa5fc64ef19a78a2de3f3085b3`).
The two instrumented source copies and their patches have separate receipts
under `20261010-workspace-setup-control-checked/` and
`20261010-workspace-setup-fused-checked/` in the same ignored results tree.

These passes show that the asserted schedule invariants held in two
instrumented setup-only processes. They do not establish the reliability of
the original uninstrumented timing process or explain its earlier failure.
No production source or sorting algorithm was changed.
