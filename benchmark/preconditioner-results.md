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
| none | CG | 0.355 | 11.815 | 296 | 8.745e-8 | 0 |
| geometric MG | CG | 9.612 | 3.461 | 8 | 7.600e-8 | 10,007,768 |
| AMG | CG | 2.779 | 12.545 | 33 | 5.403e-8 | 2,831,320 |
| incomplete Cholesky | CG | 2.717 | 10.679 | 97 | 9.433e-8 | 1,382,280 |
| ILU | BiCGSTAB | 14.383 | 5.357 | 10 | 1.856e-8 | 9,548,424 |

Geometric MG is the strongest pressure preconditioner in this case: it reduces
296 CG iterations to 8 and gives the fastest solve, at the cost of the largest
CG-compatible retained structure. AMG uses about 28% of that storage, but its
more expensive V-cycle makes it slightly slower than unpreconditioned CG here.
The default incomplete-Cholesky parameters are memory-efficient but weak. ILU
is fast after setup, but that row uses BiCGSTAB because a generic ILU is not an
SPD preconditioner and therefore is not a valid CG comparison.

## Staggered implicit diffusion

The dimensionless stiffness is 40. The nonsymmetric AMR operator is solved
with BiCGSTAB in every applicable row.

| Preconditioner | Setup (ms) | Warm solve (ms) | x/y iterations | L∞ residual | Stored bytes |
|---|---:|---:|---:|---:|---:|
| none | 0.375 | 7.970 | 36/39 | 8.332e-10 | 0 |
| geometric MG | 120.612 | 28.807 | 6/6 | 5.047e-11 | 39,749,152 |
| AMG | 6.774 | 68.082 | 38/37 | 1.042e-9 | 7,071,504 |
| ILU | 1.278 | 10.128 | 38/40 | 5.021e-10 | 662,832 |
| incomplete Cholesky | — | — | not applicable | — | — |

For this mesh, none of the tested diffusion preconditioners beats plain
BiCGSTAB in wall time. Geometric MG greatly reduces iterations but its two
staggered hierarchies are expensive to apply and retain about 39.7 MB. The
generic AMG and ILU defaults do not improve iteration counts. Incomplete
Cholesky is excluded because the staggered AMR matrix is nonsymmetric.

These are baseline defaults, not tuned optima. In particular, AMG strength,
smoother, and cycle parameters and the ILU drop tolerance should be swept
before making a final policy decision. The current results support retaining
geometric MG as the pressure default, while identifying staggered geometric
coarsening/application as the next performance target.

Reproduce with [`preconditioner_comparison.jl`](preconditioner_comparison.jl)
using the command in the [benchmark README](README.md).
