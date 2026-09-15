# Localized-vortex AMR usefulness study

Recorded 2026-09-15 with Julia 1.12.7 on an AMD Ryzen 9 5950X, using one
Julia thread. Reproduce it with the `amr_usefulness.jl` command in
[the benchmark guide](README.md).

## Manufactured solution

The periodic unit square uses the smooth localized streamfunction

```math
B_\kappa(s;s_0) = \exp\{\kappa[\cos(2\pi(s-s_0))-1]\},
\qquad
\psi(x,y,t) = a e^{-\lambda t} B_\kappa(x;x_0)B_\kappa(y;y_0),
```

and the staggered exact velocity

```math
\boldsymbol{u} = (\partial_y\psi,-\partial_x\psi).
```

It is analytically divergence-free. With exact pressure `p=0`, the body force
is manufactured as

```math
\boldsymbol{f} = \partial_t\boldsymbol{u}
  + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}
  - \nu\nabla^2\boldsymbol{u}.
```

The parameters are `(x0,y0)=(0.37,0.53)`, `kappa=8`, `a=0.05`, `lambda=1`,
and `nu=0.002`. The run advances to `t=0.006` with `dt=0.0005`. This is more
selective than Taylor--Green: the large derivatives that control spatial error
are confined to a small periodic neighbourhood of the vortex.

The indicator field is `B_kappa(x;x0) B_kappa(y;y0)`, with an 8-by-8 cell
base block, `E_refine=0.025`, `E_coarsen=0`, and `relative_floor=0.02`. The
mesh is adapted to a stable initial topology before sampling the exact face
velocity. This prevents a max-level convergence study from measuring lost
initial-condition information that was never present on the base mesh. A
separate integration test starts on the root mesh and exercises all three
runtime rebuilds through the uniform-to-mixed transition.

## Results

Times and allocation values are BenchmarkTools medians of three complete
setup-plus-simulation evaluations after compilation. The AMR cases change
only the maximum level; level 3 has the same finest spacing as 64-by-64.

| case | cells | blocks | velocity L2 error | max divergence | median time | allocated memory | allocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMR, max level 1 | 256 | 4 | 1.564381e-3 | 1.66e-15 | 6.647 ms | 11.27 MiB | 94,531 |
| AMR, max level 2 | 1,024 | 16 | 4.128123e-4 | 5.67e-15 | 22.621 ms | 46.56 MiB | 359,595 |
| AMR, max level 3 | 2,368 | 37 | 1.075527e-4 | 7.11e-15 | 106.377 ms | 249.00 MiB | 994,991 |
| uniform 48-by-48 | 2,304 | 1 | 1.890773e-4 | -- | 51.753 ms | 110.60 MiB | 785,590 |
| uniform 64-by-64 | 4,096 | 1 | 1.075491e-4 | -- | 82.753 ms | 188.01 MiB | 1,386,453 |

The observed rates from maximum levels 1 to 2 and 2 to 3 are 1.922 and 1.940.
At almost equal active-cell count, the deepest AMR mesh has 0.569 times the
error of the 48-by-48 uniform mesh. It also matches the 64-by-64 error to
0.004% while using 57.8% as many cells.

This confirms the numerical usefulness of refinement for a localized smooth
solution, but it also exposes a current performance limitation: at this small
problem size, mixed-level connectivity and elliptic-hierarchy setup make the
AMR run slower and more allocation-heavy than either uniform comparison.
Consequently this is an accuracy-per-cell success, not yet a wall-time or
memory win. Reusing topology-dependent solver objects across unchanged runs
and reducing mixed-level setup allocation remain optimization targets.
