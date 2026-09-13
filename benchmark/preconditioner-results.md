# Krylov preconditioner comparison

Measured on 2026-09-13 with Julia 1.12.7 on an AMD Ryzen 9 5950X. Julia used
one Julia thread and the default 16 OpenBLAS threads. The mesh has 7,168 active
cells and 14,400 owned faces. Values are medians of five warmed solves at a
strong relative tolerance of `1e-9`; setup timings exclude JIT compilation.
`Stored bytes` is `Base.summarysize` of the retained preconditioner, including
its matrices and factors.

## Pressure Poisson

| Preconditioner | Krylov | Setup (ms) | Warm solve (ms) | Iterations | L∞ residual | Stored bytes |
|---|---|---:|---:|---:|---:|---:|
| none | CG | 0.225 | 11.741 | 296 | 8.745e-8 | 0 |
| geometric MG | CG | 8.420 | 3.280 | 8 | 7.600e-8 | 8,371,752 |
| AMG | CG | 3.206 | 12.630 | 33 | 5.403e-8 | 2,831,320 |
| incomplete Cholesky | CG | 2.817 | 10.666 | 97 | 9.433e-8 | 1,382,280 |
| ILU | BiCGSTAB | 15.714 | 5.358 | 10 | 1.856e-8 | 9,548,424 |

Geometric MG is the strongest pressure preconditioner in this case: it reduces
296 CG iterations to 8 and gives the fastest solve, at the cost of the largest
CG-compatible retained structure. AMG uses about 34% of that storage, but its
more expensive V-cycle makes it slightly slower than unpreconditioned CG here.
The default incomplete-Cholesky parameters are memory-efficient but weak. ILU
is fast after setup, but that row uses BiCGSTAB because a generic ILU is not an
SPD preconditioner and therefore is not a valid CG comparison.

## Staggered implicit diffusion

The dimensionless stiffness is 40. The nonsymmetric AMR operator is solved
with BiCGSTAB in every applicable row.

| Preconditioner | Setup (ms) | Warm solve (ms) | x/y iterations | L∞ residual | Stored bytes |
|---|---:|---:|---:|---:|---:|
| none | 0.158 | 7.883 | 36/39 | 8.332e-10 | 0 |
| geometric MG | 23.705 | 10.567 | 5/5 | 2.700e-10 | 8,008,592 |
| AMG | 8.572 | 55.702 | 27/29 | 9.977e-10 | 7,785,120 |
| incomplete Cholesky | 9.863 | 17.943 | 30/29 | 4.415e-10 | 3,405,584 |
| ILU | 2.428 | 9.683 | 38/36 | 1.054e-9 | 662,832 |

For this mesh, none of the tested diffusion preconditioners beats plain
BiCGSTAB in wall time. The refactored geometric hierarchy nevertheless cuts
the original baseline setup from 120.612 ms to 23.705 ms, solve time from
28.807 ms to 10.567 ms, and component storage from 39.7 MB to 8.0 MB while
reducing the solve to 5/5 iterations. Generic AMG and ILU still do not improve
iteration counts enough to pay for their application. Incomplete Cholesky is
now applicable because it is built from the symmetric stable core; BiCGSTAB
still applies the nonsymmetric corrected matrix.

These are baseline defaults, not tuned optima. In particular, AMG strength,
smoother and cycle parameters, ILU dropping, and incomplete-Cholesky memory
should be swept before making a final policy decision. The separate scaling
sweep shows that geometric diffusion preconditioning amortizes on the largest
test, while plain BiCGSTAB remains preferable for this smaller problem.

## Staggered Krylov methods

The same geometric hierarchy was also paired with the supported nonsymmetric
Krylov methods. GMRES and FGMRES use restart length 20 and
reorthogonalization.

| Krylov | Setup (ms) | Warm solve (ms) | x/y iterations | L∞ residual | Workspace bytes |
|---|---:|---:|---:|---:|---:|
| BiCGSTAB | 20.263 | 10.535 | 5/5 | 2.700e-10 | 922,870 |
| GMRES(20) | 20.340 | 11.784 | 10/10 | 1.270e-10 | 2,772,464 |
| FGMRES(20) | 20.288 | 10.821 | 10/10 | 1.270e-10 | 4,963,184 |

BiCGSTAB remains the best default at this point. FGMRES is close in wall time
but uses over five times the Krylov workspace storage, while fixed V-cycles do
not require its flexible-preconditioner semantics.

Reproduce with [`preconditioner_comparison.jl`](preconditioner_comparison.jl)
using the command in the [benchmark README](README.md).
