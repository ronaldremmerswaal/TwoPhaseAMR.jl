# V-cycle versus V-cycle-preconditioned CG

Measured on 2026-09-13 with Julia 1.12.7, one Julia thread, and an AMD Ryzen 9
5950X. The working tree was based on commit `6df17f7` plus the PCG changes.

The benchmark uses 8×8-cell blocks and refines the southwest quarter
recursively, producing adjacent coarse regions and fine regions through seven
tree levels. Both methods solve the same compatible periodic Poisson problem
from a zero initial guess to `rtol=1e-10`. The hierarchy uses
`coarse_size=32`, two pre-sweeps, two post-sweeps, Galerkin coarse operators,
and the same bottom solve. Setup is excluded. Reported times are warm medians
of seven samples through level 4, five at level 6, and three at level 7.

| Max tree level | Active cells | Solver levels | Method | Iterations/V-cycles | Median time | Allocated bytes | LInf residual |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 448 | 3 | standalone V-cycle | 13 | 0.251 ms | 0 | 2.57e-9 |
| 2 | 448 | 3 | PCG + V-cycle | 8 | 0.168 ms | 0 | 9.42e-9 |
| 4 | 7,168 | 5 | standalone V-cycle | 14 | 5.052 ms | 0 | 7.82e-9 |
| 4 | 7,168 | 5 | PCG + V-cycle | 9 | 3.406 ms | 0 | 5.45e-9 |
| 6 | 79,360 | 7 | standalone V-cycle | 17 | 90.622 ms | 0 | 2.29e-9 |
| 6 | 79,360 | 7 | PCG + V-cycle | 10 | 61.583 ms | 0 | 2.22e-9 |
| 7 | 289,024 | 8 | standalone V-cycle | 18 | 463.721 ms | 0 | 4.71e-9 |
| 7 | 289,024 | 8 | PCG + V-cycle | 11 | 295.817 ms | 0 | 4.56e-9 |

PCG reduces the number of hierarchy traversals by 36--41% and measured solve
time by 32--36%; its extra fine-grid matrix-vector product and vector updates
do not erase the gain at large size. It is therefore the default for the
symmetric cell-centred elliptic family. `method=:multigrid` retains the
standalone comparison path, while `method=:auto` falls back to it if pre- and
post-smoothing are not symmetric or the hierarchy has only one level.

CG is not used for the widened staggered AMR diffusion operator because its
consistency correction is mildly nonsymmetric. That operator continues to use
standalone geometric multigrid; a GMRES-family outer iteration would be the
mathematically appropriate Krylov experiment there.

The benchmark is reproduced with:

```sh
julia --project=. benchmark/pcg_multigrid.jl
```

References:

- [PETSc `KSPCG` documentation](https://petsc.org/release/manualpages/KSP/KSPCG/)
- [hypre BoomerAMG documentation](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
- [Tatebe, *The Multigrid Preconditioned Conjugate Gradient Method*](https://ntrs.nasa.gov/citations/19940017009)
