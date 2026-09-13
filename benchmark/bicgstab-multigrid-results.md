# Plain versus geometric-preconditioned BiCGSTAB

Measured on 2026-09-13 with Julia 1.12.7, one Julia thread, and an AMD Ryzen 9
5950X. The benchmark uses 8×8-cell blocks and recursively refines the southwest
quarter, leaving coarse and fine regions adjacent through as many as six tree
refinement steps.

Both paths apply the same corrected staggered shifted operator

```math
A = M(I-\mu L_c)
```

to the same smooth two-component right-hand side from a zero initial guess.
The dimensionless relative stiffness is
`mu * max(abs-row-sum(L_c))`. Both paths use Krylov.jl BiCGSTAB; the geometric
row additionally applies one hybrid multigrid V-cycle as a right
preconditioner. The solve tolerance is `rtol=1e-9`, and setup is excluded.
Times are warm medians of seven samples through level 4 and three samples at
level 6.

| Max tree level | Staggered unknowns | Relative stiffness | Preconditioner | x/y iterations | Median time (ms) | Allocated bytes | LInf residual |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 912 | 0.45 | none | 8/8 | 0.201 | 0 | 6.103e-10 |
| 2 | 912 | 0.45 | geometric | 2/2 | 0.427 | 0 | 8.882e-16 |
| 2 | 912 | 4 | none | 15/15 | 0.346 | 0 | 8.252e-10 |
| 2 | 912 | 4 | geometric | 3/3 | 0.623 | 0 | 3.389e-11 |
| 2 | 912 | 40 | none | 31/36 | 0.719 | 0 | 5.350e-10 |
| 2 | 912 | 40 | geometric | 5/5 | 1.018 | 0 | 6.808e-10 |
| 4 | 14,400 | 0.45 | none | 10/9 | 2.290 | 0 | 7.777e-10 |
| 4 | 14,400 | 0.45 | geometric | 2/2 | 4.457 | 0 | 1.110e-15 |
| 4 | 14,400 | 4 | none | 17/17 | 3.775 | 0 | 3.512e-11 |
| 4 | 14,400 | 4 | geometric | 3/3 | 6.504 | 0 | 5.574e-11 |
| 4 | 14,400 | 40 | none | 36/39 | 7.854 | 0 | 8.332e-10 |
| 4 | 14,400 | 40 | geometric | 5/5 | 10.538 | 0 | 2.700e-10 |
| 6 | 159,232 | 0.45 | none | 32/33 | 84.631 | 0 | 6.082e-10 |
| 6 | 159,232 | 0.45 | geometric | 2/2 | 64.503 | 0 | 1.554e-15 |
| 6 | 159,232 | 4 | none | 42/42 | 109.336 | 0 | 8.876e-10 |
| 6 | 159,232 | 4 | geometric | 3/3 | 95.297 | 0 | 4.303e-11 |
| 6 | 159,232 | 40 | none | 64/59 | 156.840 | 0 | 4.410e-10 |
| 6 | 159,232 | 40 | geometric | 5/5 | 156.441 | 0 | 2.432e-10 |

The preconditioner is not worthwhile on the two smaller cases despite its
large iteration reduction. At 159,232 unknowns it is 24% faster at relative
stiffness 0.45, 13% faster at stiffness 4, and essentially tied at stiffness
40. The hierarchy therefore improves the large-problem scaling but does not
justify replacing plain BiCGSTAB for every small system. Both warmed paths
allocate zero bytes.

The public default remains BiCGSTAB with geometric multigrid; the benchmark
keeps `preconditioner=nothing` available as the small-problem reference. An
automatic crossover policy should include setup amortization and more mesh
patterns before it changes that default.

Reproduce with:

```sh
julia --project=. benchmark/bicgstab_multigrid.jl
```

References:

- [PETSc `KSPBCGS` documentation](https://petsc.org/release/manualpages/KSP/KSPBCGS/)
- [van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG*](https://doi.org/10.1137/0913035)
