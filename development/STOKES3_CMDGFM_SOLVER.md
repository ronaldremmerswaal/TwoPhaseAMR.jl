# CMDGFM Stokes3 solver benchmark

Run the reference problem with the default wave parameters, two velocities,
`max_level=5`, and `final_time=0.5`:

```sh
julia --project=. development/stokes3_cmdgfm_solver.jl default 0.5 5
```

The benchmark advances one physical step per `simulate!` call to retain the
projection statistics for every step. It keeps the minimum stable time step,
uses the normal AMR path, and reports the pressure and normal-velocity defects
separately. The `stale_pressure`, `pressure_refresh`, `fresh`, and `ilu` modes
select comparison preconditioners. Reported wall times include compilation and
depend on the machine; iteration counts and defects are the useful solver
comparisons.

## Failure and cause

The old default retained the numerical pressure multigrid hierarchy as the
interface moved on an unchanged mesh. Its Krylov iterations rose from 78 to
318 to 500; step 4 failed at `t=0.007206642` with a volume-scaled pressure
defect of `1.54e-4` against a target of `1.49e-8`. The normal-velocity defect
was only `6.71e-11`. The system at that step had 13,695 pressure unknowns,
598 active jump unknowns, and 74,639 nonzeros. This is a preconditioner
failure, not a failure to assemble or solve the jump condition.

Refreshing the pressure coefficients while retaining the hierarchy topology
reduced the first four step counts to 13, 6, 12, and 13. This mode completed
219 steps and 32 AMR updates through `t=0.5`; its final pressure-volume defect
was `6.21e-11` and normal defect `7.07e-14`. Rebuilding both pressure and jump
preconditioners also completed the reference run. Thus numerical pressure
refresh is essential, whereas rebuilding the entire hierarchy is unnecessary.
The final default (including safe jump-factor and Krylov-workspace reuse)
likewise completed all 219 steps and 32 AMR updates; its last solve took 7
iterations with the same final defects. The one-step driver measured 153 s in
physical steps, excluding setup and compilation, on this machine.
The uninstrumented `run_example` entry point also reached `t=0.5` in 219
steps; its wall time was 220 s including setup and compilation.

The old jump-ILU reuse test checked only sparsity. An ILU factor also depends
on values, so equal sparsity does not justify reuse after the interface moves.
The solver now reuses it only if the entire jump matrix, including its values,
is identical. This still permits factor reuse for repeated projections of a
stationary interface. IncompleteLU.jl does not expose an in-place numerical
refactorization API, so changed jump matrices require a new ILU factor.

A whole-system ILU was not a better fallback here: it failed on step 4 with a
`3.50e-8` volume-scaled pressure defect, despite a small Euclidean residual.
The block preconditioner is better matched to the two physical residuals.

The CMDGFM projection now retains its BiCGSTAB vectors and residual buffer
between solves, resizing them only if the number of unknowns changes. The
pressure hierarchy retains its mesh hierarchy and refreshes numerical levels
only when coefficients, volumes, or active-cell constraints change. AMR
topology changes still build a replacement projection workspace.
