# Staggered V-cycle versus V-cycle-preconditioned BiCGStab

Measured on 2026-09-13 with Julia 1.12.7, one Julia thread, and an AMD Ryzen 9
5950X. The benchmark uses 8×8-cell blocks and recursively refines the southwest
quarter, leaving coarse and fine regions adjacent through as many as six tree
refinement steps.

Both solvers apply the same staggered shifted operator

```math
A = M(I-\mu L)
```

to the same smooth two-component right-hand side from a zero initial guess.
The dimensionless relative stiffness in the table is
`mu * max(abs-row-sum(L))`. The solve tolerance is the implicit-diffusion
default, `rtol=64eps(Float64)`. Hierarchy setup is excluded; times are warm
medians of seven samples through level 4 and three samples at level 6.

| Max tree level | Staggered unknowns | Relative stiffness | Method | x/y iterations | Median time | Allocated bytes | LInf residual |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 912 | 0.45 | standalone V-cycle | 4/4 | 0.946 ms | 0 | 1.06e-15 |
| 2 | 912 | 0.45 | BiCGStab + V-cycle | 2/2 | 0.969 ms | 0 | 8.88e-16 |
| 2 | 912 | 4 | standalone V-cycle | 10/10 | 2.344 ms | 0 | 2.12e-15 |
| 2 | 912 | 4 | BiCGStab + V-cycle | 4/4 | 1.874 ms | 0 | 2.00e-15 |
| 2 | 912 | 40 | standalone V-cycle | 20/23 | 5.150 ms | 0 | 2.86e-14 |
| 2 | 912 | 40 | BiCGStab + V-cycle | 7/8 | 3.542 ms | 0 | 2.69e-14 |
| 4 | 14,400 | 0.45 | standalone V-cycle | 4/4 | 10.055 ms | 0 | 7.77e-16 |
| 4 | 14,400 | 0.45 | BiCGStab + V-cycle | 2/2 | 10.356 ms | 0 | 1.55e-15 |
| 4 | 14,400 | 4 | standalone V-cycle | 10/10 | 25.193 ms | 0 | 2.44e-15 |
| 4 | 14,400 | 4 | BiCGStab + V-cycle | 4/4 | 20.125 ms | 0 | 5.11e-15 |
| 4 | 14,400 | 40 | standalone V-cycle | 24/24 | 59.257 ms | 0 | 2.73e-14 |
| 4 | 14,400 | 40 | BiCGStab + V-cycle | 8/9 | 41.261 ms | 0 | 1.90e-14 |
| 6 | 159,232 | 0.45 | standalone V-cycle | 4/4 | 593.363 ms | 0 | 7.77e-16 |
| 6 | 159,232 | 0.45 | BiCGStab + V-cycle | 2/2 | 625.492 ms | 0 | 1.33e-15 |
| 6 | 159,232 | 4 | standalone V-cycle | 9/9 | 1,269.183 ms | 0 | 2.98e-14 |
| 6 | 159,232 | 4 | BiCGStab + V-cycle | 4/4 | 1,214.660 ms | 0 | 1.09e-14 |
| 6 | 159,232 | 40 | standalone V-cycle | 32/32 | 4,465.611 ms | 0 | 2.04e-14 |
| 6 | 159,232 | 40 | BiCGStab + V-cycle | 10/10 | 2,949.949 ms | 0 | 1.93e-14 |

At relative stiffness 0.45, standalone multigrid is 2--5% faster because it
already converges in four traversals. At stiffness 4, BiCGStab is 4--20%
faster; at stiffness 40 it is 30--34% faster. Both paths allocate zero bytes
after setup.

Consequently, `method=:auto` is the default and selects BiCGStab on a
multilevel hierarchy when the relative stiffness is at least 4. It retains
standalone multigrid below that crossover and on a one-level hierarchy.
Explicit `method=:bicgstab` and `method=:multigrid` choices remain available.

Reproduce the benchmark with:

```sh
julia --project=. benchmark/bicgstab_multigrid.jl
```

Reference:

- [PETSc `KSPBCGS` documentation](https://petsc.org/release/manualpages/KSP/KSPBCGS/)
- [van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG*](https://doi.org/10.1137/0913035)
