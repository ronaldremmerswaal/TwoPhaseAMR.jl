# Stokes3 pressure investigation: measurements, no solver changes

This is the measurement phase of [PRESSURE_PLAN.md](PRESSURE_PLAN.md). The
development-only scripts are [pressure_probe.jl](pressure_probe.jl) and
[pressure_step_trace.jl](pressure_step_trace.jl). They use default Stokes3
physics and AMR, one Julia thread, 16 BLAS threads, Julia 1.12.7. Initial
AMR and initial projection are outside each measured `t=0` to `t=0.5`
advance. Each process first runs a short warm-up. The probe adds small custom
diagnostics so that matrix structure and one-velocity CG cycles can be
observed; its three runs per model retained identical trajectories.

From the repository root, reproduce the main runs with:

```sh
julia --project=. development/stokes3_performance/pressure_probe.jl one 0.5 3
julia --project=. development/stokes3_performance/pressure_probe.jl cmdgfm 0.5 3
julia --project=. development/stokes3_performance/pressure_step_trace.jl 0.5
```

## Full-trajectory baseline

| Model | Warm advance time, three repetitions (s) | Median (s) | Allocated, three repetitions (GB) | Steps / AMR changes | Final faces |
| --- | --- | ---: | --- | ---: | ---: |
| One velocity | 18.845, 18.394, 18.444 | 18.444 | 11.402, 11.395, 11.395 | 206 / 45 | 12,504 |
| CMDGFM | 50.022, 48.825, 49.028 | 49.028 | 58.733, 58.724, 58.724 | 219 / 32 | 21,448 |

The median whole-run CMDGFM cost is 2.66 times the one-velocity time and
5.15 times the allocated bytes. Its larger final mesh and different AMR
history account for some of that difference. The earlier sampled profile in
the plan estimates pressure-stage time at 21.66 s versus 2.30 s, including
13.58 s of CMDGFM assembly, 3.84 s in BiCGSTAB, and 1.83 s in
preconditioner work. Those profile categories are **estimates**, not
independent stopwatch timings; the frozen-state timings below provide the
non-overlapping decomposition.

## Frozen-state pressure costs

The staged assembly reconstructs the production matrix and right-hand side
and checks both for exact equality. `whole_assembly_check` is a separate
repeat, not an additional substage. The drill-down rows repeat work inside
`pressure_operator`, so they must not be added to the assembly total.
Times are single post-compilation samples and can include GC; allocated
bytes are more stable. Stokes3 has **no embedded cut-cell geometry**, so the
two scalar pressure formulations use the same uncut metric on a matched
state. On the default AMR mesh at initialization, the fresh CMDGFM operator
and the scalar operator had exactly equal face coefficients, stiffness, and
numerical inverse-gradient metric. At `t=0.02`, after one AMR change, the
coefficient and metric still agreed exactly; stiffness differed by at most
`9.09e-13` in absolute value. This is consistent with floating-point
accumulation differences between fresh assembly and coefficient refresh,
not a different discrete metric. The probe reports these differences on
each frozen snapshot. The metric wrapper itself does not implement
value-based `==`, so its underlying diagonal or sparse matrix is compared.

| State and operation | Time (ms) | Allocation (MB) |
| --- | ---: | ---: |
| Initial CMDGFM mesh, 27,576 faces: whole assembly | 41.21 | 162.21 |
| Initial: fresh pressure operator | 35.95 (6.43 GC) | 117.41 |
| Initial: gradient metric within that operator | 13.73 | 45.91 |
| Initial: elliptic operator from prepared metric | 19.71 | 71.06 |
| Late CMDGFM mesh, 21,448 faces: whole assembly | 42.42 (8.43 GC) | 128.72 |
| Late: fresh pressure operator | 23.81 | 94.84 |
| Late: jump stencils | 3.04 | 9.99 |
| Late: common-gradient sparse product | 2.17 | 7.46 |
| Late: final block concatenation | 0.36 | 2.32 |
| Late: jump-block generation | 0.55 | 1.67 |
| Late: `matrix - matrix'` asymmetry check (outside assembly) | 0.35 | 2.99 |
| Late: gradient recovery (outside assembly) | 0.37 | 0.51 |
| One-velocity final mesh, 12,504 faces: scalar operator refresh + GMG | 1.21 | 0.48 |
| Same 24,912-face CMDGFM mesh: scalar refresh + GMG | 1.93 | 0.78 |
| Same 24,912-face CMDGFM mesh: fresh CMDGFM assembly | 38.98 | 151.84 |

On the late CMDGFM mesh, the fresh pressure operator alone accounts for
roughly 74% of assembly allocations and 56% of its elapsed time, before GC
variation. The initial-state drill-down separates its allocation into the
mesh gradient metric (45.91 MB) and elliptic matrix/refresh-map construction
(71.06 MB). This is repeated on unchanged topology even though the
one-velocity path refreshes only coefficient-dependent values.

## Preconditioner and solve on a frozen 24,912-face state

The coupled system has 12,351 pressure unknowns and 592 jump unknowns;
its matrix has 68,042 nonzeros. The jump block has 1,836 nonzeros, and the
ILU factors have 3,237 combined nonzeros (fill ratio 1.76). The timings below
are repeated-state microbenchmarks, not an estimate of every moving-interface
step.

| Operation | Time (ms) | Allocation (MB) |
| --- | ---: | ---: |
| Build block preconditioner with no previous hierarchy | 33.18 | 102.22 |
| Reuse block preconditioner on exactly the same matrix | 0.15 | 0.008 |
| Refresh pressure GMG numerical values | 1.31 | 0.170 |
| Construct fresh jump ILU only | 0.089 | 0.199 |
| Apply pressure GMG / jump ILU | 0.631 / 0.007 | 0 / 0 |
| Apply full block preconditioner | 0.678 | 0 |
| BiCGSTAB on fixed RHS at relative tolerance `1e-8` | 4.48 (3 iterations) | 0 |

The default is pressure GMG plus jump-block ILU, **not** whole-system ILU.
Jump ILU is neither the largest setup cost nor a significant allocator on
this state. A new pressure hierarchy after AMR is costly (102 MB), but it is
reused on fixed topology. The installed IncompleteLU package treats `τ=0`
as full LU, not an ILU(0) switch.

## Structural reuse observed during the CMDGFM trajectory

The 219-step trajectory has 32 AMR updates. A matrix remained available to
the post-step diagnostic on 188 states (initial state plus 187 post-step
states); after AMR, the projection workspace is rebuilt and that step's
matrix is not retained there. Across the 187 consecutive observable pairs,
the retained mixed-face set and full sparse pattern both changed 187 times.
The pressure hierarchy was marked reused on 163 of the 188 states; the
jump factor was reused on **none**. This does not rule out caching the
mesh-only incidence, gradient metric, vertex patches, or a pressure
coefficient-refresh map: their topology is stable across many steps even
when active jump faces and numerical values change. It does make a single
fixed coupled-matrix pattern or long-lived numerical ILU factor unlikely to
pay off without a more flexible active-set representation.

## Per-step solver behavior

The development-only manual loop matched the ordinary `simulate!` trajectory
at both `t=0.02` (9 steps, 1 AMR update, 24,912 final faces) and `t=0.5`
(219 steps, 32 updates, 21,448 faces). It records the projection stats
returned by each `step!`; unlike the probe's post-step diagnostic, it also
captures the 32 projections immediately preceding an AMR rebuild.

| CMDGFM quantity over 219 steps | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Total BiCGSTAB iterations, including correction calls | 14 | 17 | 19 |
| Final algebraic residual | `8.98e-11` | `1.63e-7` | `2.96e-7` |
| Mixture-volume defect | `5.91e-13` | `1.27e-8` | `1.46e-8` |
| Normal-velocity defect | `5.42e-14` | `2.88e-12` | `6.32e-12` |
| Momentum identity error | `2.22e-16` | `4.44e-16` | `1.78e-15` |
| Active jump unknowns | 499 | 589 | 598 |

All projections converged. The total iteration count exceeded the last
Krylov call's count on 122/219 steps; this identifies repeated correction
calls (it does not separately detect a non-finite fallback). The pressure
hierarchy was reused on 187/219 actual projection steps and jump ILU on
0/219. The smaller 163/188 pressure-reuse count in the post-step structural
diagnostic arises because it cannot observe the just-discarded projection
workspace after AMR. Every step had some underresolved jump stencils, which
are not retained as jump unknowns; the script did not count their number
here. No jump-block diagonal was tiny relative to the chosen `1e-12` scale,
but its minimum diagonal-dominance margin was about `-2.2e-16`, effectively
zero. This is not evidence that Jacobi will be robust.

For context, the **whole physical step**, not merely pressure, had median
0.148 s / 187 MB on ordinary fixed-topology steps versus 0.248 s / 404 MB
on the 32 first steps after AMR. Those context figures must not be read as
pressure-only timings. The one-velocity CG cycle distribution from its
standard simulation diagnostic was median 10, p95 13, max 14 (206 steps).

## Interpretation and next decision

The main pressure bottleneck is repeated construction of the uncut
gradient metric and elliptic operator, followed by jump stencil/sparse
assembly. Reducing or replacing jump ILU first would save little setup time
and risks weakening a preconditioner whose jump block is not evidently
strongly diagonally dominant. The first candidate for a separate
implementation phase is to reuse the already-refreshed scalar pressure
operator in CMDGFM on this uncut case, or at least its compatible gradient
metric and coefficient-refresh map on unchanged topology. This would not
be bitwise identical to fresh assembly after coefficient updates; compare
operator actions, converged solutions, and the existing physical validation
gates with roundoff-aware tolerances. Cut-cell cases
need a separate equality check because their one- and two-velocity
coefficients differ. No optimization has been made in this investigation.

This is particularly concrete for Stokes3: its surface tension and gravity
cause `_refresh_pressure_substep!` to update the scalar operator immediately
before CMDGFM projection, so that operator is already live on the correct
mesh and interface state. A reuse change would still need to verify aliasing
and lifecycle across AMR, and compare the complete pressure/jump action and
gradient recovery before being accepted.

The per-stage timings are frozen-state samples rather than a pressure-only
stopwatch for every time step; the full-run pressure split remains based on
the earlier sampled profiler. The probe did not rerun a full projection on
the frozen post-projection velocity, since that nearly divergence-free RHS
would not represent a normal physical step. The trace measures the total
iteration count and final defects but does not count matrix-vector products
or explicitly flag the rare non-finite fallback branch. Alternative jump
preconditioners and reuse implementations remain intentionally untested
pending review of these measurements.
