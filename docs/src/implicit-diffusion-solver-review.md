# Implicit staggered-diffusion solver: literature review

This review covers the constant-viscosity, face-staggered Helmholtz systems
that arise in the fixed-tree Navier--Stokes step, their time integration, and
their solution on a 2:1 block-tree mesh. It also identifies the constraints
that variable two-phase viscosity will add later. The conclusions combine the
published literature with measurements of the current implementation; they do
not claim that a method used successfully for a cell-centred or finite-element
operator transfers automatically to this face graph.

## Executive conclusion

The implicit midpoint formulation is appropriate and must not have a diffusion
CFL restriction. Its linear system is the problem to optimize. For one
component it is

```math
A u^{n+1} = b, \qquad
A = M\left(I-\alpha L_c\right), \qquad
\alpha=\frac{\nu\Delta t}{2},
```

where `M` contains dual volumes and `L_c` is the corrected AMR Laplacian. The
conservative core has the symmetric positive-definite shifted form

```math
B = M + \alpha K, \qquad K=G^T W G \succeq 0.
```

The best next design is therefore:

1. keep BiCGSTAB as the default outer Krylov method for the nonsymmetric
   corrected equation;
2. apply the exact corrected operator on the fine grid, eventually without
   assembling a `SparseMatrixCSC`;
3. precondition it with a compact, symmetric stable-core hierarchy for `B`,
   rather than Galerkin-coarsening the full corrected matrix; and
4. rediscretize compact coarse operators or explicitly sparsify them, with
   face-topology-aware coarsening and scheduled transfer kernels.

This is a standard defect-correction/preconditioning separation, not a change
to the equation. PETSc explicitly permits the operator used by Krylov and the
matrix used to build a preconditioner to differ, including a lower-order
preconditioning matrix.[^9] Mature block-AMR solvers likewise retain geometric
Helmholtz operators through their hierarchy and often use a generic AMG or
Krylov package only at the bottom level.[^3][^5]

The current geometric preconditioner should not yet be removed: it demonstrates
that the low-frequency space is useful, cutting the benchmark from about
38 BiCGSTAB iterations to 6. Its implementation, however, has excessive grid
and operator complexity, so one V-cycle costs too much. Generic AMG and ILU are
valuable optional fallbacks, but the untuned benchmark does not support either
as the diffusion default.

## Time discretization: implicit does mean no diffusion CFL

For the scalar test equation `y'=lambda*y`, implicit midpoint/Crank--Nicolson
has amplification factor

```math
R(z)=\frac{1+z/2}{1-z/2}, \qquad z=\lambda\Delta t.
```

For a dissipative spatial operator with `Re(lambda) <= 0`, `abs(R(z)) <= 1`:
the method is A-stable. There is no viscous stability restriction of the form
`dt <= C*h^2/nu`, and none should be imposed by `simulate!`. A large `dt` can
still be inaccurate, and it makes the Helmholtz system more elliptic and often
harder to solve, but neither fact is a diffusion CFL.

Midpoint is not L-stable: `R(z)` tends to `-1`, rather than zero, as a negative
real `z` tends to minus infinity. Very stiff modes can therefore alternate in
sign and decay too slowly. Rannacher damping starts nonsmooth diffusion data
with backward-Euler half steps before switching to a second-order centred
method.[^19] Chombo instead uses the second-order, `L0`-stable Twizell--Gumel--
Arigu (TGA) scheme for parabolic AMR updates.[^4][^18] McCorquodale, Colella,
and Johansen report that Crank--Nicolson was unsuitable for their moving
embedded-boundary heat problem and used TGA instead.[^20]

Those observations do not justify a hidden timestep cap. They justify an
optional damping-capable integrator if discontinuous viscosity, regridding, or
sharp two-phase data produces visible stiff-mode ringing. TGA needs two shifted
Helmholtz solves per step. TR--BDF2 is another second-order L-stable candidate;
its two implicit stages share a diagonal coefficient and can reuse one shifted
operator/preconditioner.[^23] Midpoint should remain the default until a
targeted test demonstrates a damping problem because it needs only one solve
and is already second order for the present smooth fixed-tree validation cases.

## Spatial operator and future variable viscosity

The current component operator deliberately has two parts. Connections between
dual control volumes build the quadratic energy `K=G'WG`; widened
linear-exact gradients preserve symmetry and non-negative dissipation. Rows in
a narrow refinement-interface band are then replaced by quadratic-exact
Laplacian rows. This restores the measured spatial order but makes `L_c` mildly
nonsymmetric. Consequently:

- CG and incomplete Cholesky are valid for the stable-core preconditioner `B`,
  but not for the full corrected equation `A`;
- BiCGSTAB or GMRES is appropriate for `A`; and
- a preconditioner is not required to reproduce the correction exactly.

This last point is important. The present hierarchy uses stable-core smoothing
for transfer construction, but still forms `P_l' * A_l * P_l` for the complete
nonsymmetric matrix on every level. That is more exact than a preconditioner
needs to be and is the source of most of its cost.

Independent scalar component solves also have a scope limit. For constant
viscosity and a divergence-free velocity,
`div(mu*(grad(u)+grad(u)'))` reduces to `mu*Delta(u)`. With spatially varying or
discontinuous viscosity, gradients of `mu` remain and the symmetric stress
operator couples velocity components. Basilisk's implicit viscosity operator
solves the mutually coupled stress equations, and AMReX provides a tensor
operator for viscous terms.[^22][^5] Analysis of implicit MAC schemes on
nonuniform Cartesian grids likewise treats variable density and viscosity in
the full discrete problem.[^21]

The two-phase milestone should therefore introduce a block `ViscousOperator`
contract rather than extending the current scalar-Laplacian assumption by
coefficient multiplication. The constant-viscosity implementation can remain
the optimized separable specialization.

## What established adaptive solvers do

FAC was formulated to solve a composite adaptive elliptic problem through
interactions between global and local regular grids.[^1] Martin and Cartwright
adapted multigrid relaxation to block-structured AMR and emphasized explicit
coarse--fine matching needed for global second-order accuracy.[^2] These are
the closest ancestors of Chombo's `AMRMultiGrid`, whose operator factory builds
the appropriate Helmholtz operator on each level. Chombo offers Jacobi,
red-black Gauss--Seidel, and multicolour smoothers, and uses a smoother,
BiCGSTAB, or GMRES at the bottom.[^3]

AMReX follows the same separation. `MLMG` consumes a geometric linear-operator
class such as `MLABecLaplacian`; it geometrically coarsens as far as possible,
uses agglomeration when boxes become too small, and can call hypre or PETSc as
a bottom solver.[^5][^6] Afivo demonstrates that quadtree/octree geometric
multigrid with compact finite-volume stencils can handle very large adaptive
problems without constructing a generic Galerkin sparse hierarchy.[^7]

hypre makes the structural distinction explicit. Its SMG and PFMG solvers are
specialized for structured diffusion systems, while the semi-structured
interface covers problems with multiple structured parts and AMR.[^8] BoomerAMG
is the general sparse-matrix route and includes aggressive coarsening,
interpolation truncation, and non-Galerkin coarse-grid dropping to control
complexity.[^14]

The transferable lesson is not that rediscretization is always superior.
Pressure in this package uses a special compatible Riesz map, and blindly
replacing it by a five-point stencil would lose the selected discrete
invariants. The lesson is that a low-order block-AMR operator should not become
nearly dense merely because it is being used as a preconditioner. Its coarse
operator must either be rediscretized from the same conservative construction,
or sparsified while a spectral-equivalence/convergence test protects the
preconditioner.

## Audit of the current hierarchy

The audit uses the same graded mesh as `benchmark/preconditioner_comparison.jl`:
7,168 cells, 14,400 owned faces, `coarse_size=32`, and relative diffusion
stiffness 40. `A` is the shifted system matrix, `K` the stored stable stiffness,
and `P` the prolongation from that row to the next level.

### Pressure hierarchy

| Level | Unknowns | `nnz(A)` | `nnz(A)`/row | `nnz(P)` |
|---:|---:|---:|---:|---:|
| 1 | 7,168 | 36,480 | 5.09 | 21,888 |
| 2 | 1,792 | 24,632 | 13.75 | 11,328 |
| 3 | 448 | 17,976 | 40.13 | 6,272 |
| 4 | 112 | 10,368 | 92.57 | 2,944 |
| 5 | 28 | 784 | 28.00 | -- |

The grid complexity, `sum(n_l)/n_1`, is 1.332. The operator complexity,
`sum(nnz(A_l))/nnz(A_1)`, is 2.474. The retained pressure preconditioner is
10,007,768 bytes by `Base.summarysize`.

### One staggered component

The x and y results are identical for this symmetric test mesh; both sets are
stored independently.

| Level | Unknowns | `nnz(A)` | `nnz(A)`/row | `nnz(K)` | `nnz(P)` |
|---:|---:|---:|---:|---:|---:|
| 1 | 7,200 | 66,664 | 9.26 | 75,936 | 51,454 |
| 2 | 4,096 | 174,748 | 42.66 | 170,656 | 60,436 |
| 3 | 1,024 | 111,976 | 109.35 | 111,024 | 36,072 |
| 4 | 256 | 59,998 | 234.37 | 60,092 | 15,800 |
| 5 | 64 | 4,096 | 64.00 | 4,096 | 1,024 |
| 6 | 16 | 256 | 16.00 | 256 | -- |

The grid complexity is 1.758 and the `A` operator complexity is 6.266. The
first staggered coarsening only reduces 7,200 unknowns to 4,096, rather than
the roughly four-to-one reduction expected from regular 2-D coarsening. By
level 4, a 256 by 256 matrix contains 59,998 of 65,536 possible entries. It is
stored in a sparse format, but it is effectively dense.

No large dense finest-level matrix is used. The only intentional dense
conversion is the bottom `lu(Matrix(coarse.matrix))`, which is 16 by 16 per
component here (and a 29 by 29 augmented pressure solve). Those factorizations
are negligible. The memory comes from:

- a weak first face coarsening caused by coordinate-bin aggregation and delayed
  singletons;
- smoothed prolongation with 51,454 and then 60,436 entries;
- Galerkin triple products whose stencils widen on every coarse level;
- storing both `A_l` and `K_l`;
- storing restriction as a separate sparse copy of `P_l'`; and
- retaining two component hierarchies and their work vectors.

The x and y hierarchy components occupy 39,749,152 bytes together. Counting
the wrapper and its uniquely referenced operator data gives 44,843,296 bytes.
On a warmed construction, Julia reported 468,843,840 bytes of *cumulative
allocation traffic*, which is not peak resident memory. The earlier 2.5 GB
observation could not be reproduced as retained memory on this case; a cold
`@time` allocation total, including compilation and temporary sparse triple
products, must not be described as a 2.5 GB live peak.

The benchmark consequence is consistent with this structure. Geometric MG
reduces x/y iterations from 36/39 to 6/6, but setup rises from 0.375 ms to
120.612 ms and the warm solve from 7.970 ms to 28.807 ms. Generic smoothed-
aggregation AMG gives 38/37 iterations and ILU gives 38/40, so neither untuned
option improves on plain BiCGSTAB in that experiment. These measurements are a
single operating point, not an asymptotic scaling result.

Galerkin stencil growth is a known AMG problem rather than an unavoidable cost
of multigrid. Falgout and Schroder developed non-Galerkin coarse-grid
sparsification specifically because `P'AP` can increase coarse stencil sizes
and operator complexity.[^11] hypre exposes both interpolation truncation and
level-dependent non-Galerkin dropping for the same reason.[^14]

## Krylov and preconditioner assessment

### BiCGSTAB and GMRES

BiCGSTAB is a reasonable default for the corrected operator: it targets
nonsymmetric systems with short recurrences and low vector storage.[^15] Its
residual can be irregular and it can break down. GMRES minimizes the residual
over its Krylov subspace and is the appropriate first comparison when
BiCGSTAB stagnates, at the cost of an expanding orthogonal basis or restart
effects.[^16] FGMRES is only required if an application of the preconditioner
changes between outer iterations, for example because an inner coarse solve
uses a variable stopping tolerance; a fixed one-V-cycle map is linear and does
not require flexibility.[^9]

Recommendation: retain BiCGSTAB as the default, add benchmark coverage for
restarted GMRES and FGMRES, and record true residual histories and breakdowns.
Do not select a method by iteration count alone.

### Geometric stable-core multigrid

This has the strongest evidence for the long-term default. It exploits the
known tree geometry, can use the compact conservative operator, and already
finds a good coarse space. Matrix-free adaptive GMG has shown robust scaling
for variable-viscosity Stokes systems in a different finite-element setting,
including better refinement robustness than the tested AMG implementation.[^13]
That result supports the architecture, not a direct performance prediction for
this low-order 2-D face operator.

The immediate change should be to coarsen the SPD stable core only and use it
as a right preconditioner for the exact corrected `A`. This permits symmetric
smoothers, incomplete Cholesky at the bottom if desired, compact
rediscretization, and aggressive reuse. Because the correction is localized at
AMR interfaces, it is plausible that `B^-1 A` remains well clustered; this is
an implementation hypothesis that must be tested, not assumed.

### Generic AMG

Classical smoothed aggregation was developed for SPD elliptic systems.[^12]
Applying a default SA setup directly to the nonsymmetric corrected matrix has
no general robustness guarantee, and the present no-improvement result is not
surprising. Better experiments are:

1. SA or Ruge--Stuben AMG built from the symmetric stable-core matrix and used
   to precondition the corrected equation;
2. strength, coarsening, interpolation, truncation, and smoother sweeps; and
3. a nonsymmetric-capable implementation such as BoomerAMG if an additional
   native dependency is acceptable.

AMG remains valuable for future irregular coefficients and as a reference
against the package GMG. It should not replace geometry that the package
already knows.

### ILU and incomplete Cholesky

The current ILU point is untuned. Threshold ILU methods balance numerical
dropping with retained fill, so drop tolerance, fill cap, scaling, and ordering
are part of the method rather than incidental knobs.[^17] A sweep may make ILU
useful for small serial problems or as a bottom solve, but it is unlikely to be
the scalable default for dynamic block AMR.

Incomplete Cholesky cannot factor the full nonsymmetric matrix. It can,
however, be built from `M+alpha*K` and used as a right preconditioner for
BiCGSTAB/GMRES. The present API rejects it too early because it only offers
preconditioner construction from the Krylov operator itself. Separating
`Amat` from a preconditioning operator, as PETSc does,[^9] would enable both
this experiment and stable-core AMG without weakening the outer equation.

## Matrix-free feasibility

Matrix-free operation is feasible and well supported by established solver
interfaces. PETSc treats the Krylov operator as an abstract action, supports a
different assembled preconditioning matrix, and supports matrix-free residual
and transfer operations inside geometric multigrid.[^9][^10] Adaptive
spacetree multigrid implementations also store compact local stencil
corrections instead of global sparse matrices.[^24]

For this package, “matrix free” should mean:

- apply the fine corrected Laplacian from immutable connection/stencil
  schedules;
- cache only its diagonal and any row bound needed by a smoother;
- perform restriction and prolongation from aggregate index/weight schedules,
  not `SparseMatrixCSC` objects;
- rediscretize the stable-core Helmholtz action on every geometric level;
- keep a small assembled bottom matrix; and
- materialize a sparse matrix lazily only for AMG, ILU, debugging, eigenvalue
  tests, or export.

Making only the finest `A*x` matrix free would save its 66,664 entries per
component but would leave the 351,074 coarser `A` entries, the stable matrices,
and the transfer copies. It would not fix the main problem. Coarse operator and
transfer complexity must be addressed at the same time.

The low-order stencil also changes the expected payoff. Matrix-free methods
are especially advantageous for high-order finite elements, where sum
factorization makes operator application much cheaper than reading a large
matrix. Here the main benefit is eliminating pathological Galerkin fill and
temporary sparse products, not reducing arithmetic. A compact assembled
stencil or stencil schedule may perform as well as a purist matrix-free
implementation; both should satisfy the same operator-complexity target.

## Recommended development sequence

1. **Add reproducible complexity metrics.** For every hierarchy record grid
   complexity, `A`, `K`, and transfer nonzeros, setup allocation, retained
   bytes, V-cycle time, Krylov time, and true residual history. Sweep mesh size,
   refinement pattern, `alpha*lambda_max`, and wall condition.
2. **Build a stable-core preconditioner prototype.** Keep the exact corrected
   fine equation and replace all preconditioning levels by `M+alpha*K`. Compare
   BiCGSTAB and GMRES before changing any public default.
3. **Replace face binning.** Aggregate geometric dual cells by component
   topology so a regular 2-D region coarsens by approximately four. Preserve
   constants and wall constraints explicitly.
4. **Control coarse complexity.** First try conservative rediscretization.
   Where the selected AMR metric prevents it, truncate smoothed transfer or use
   tested non-Galerkin dropping. Target grid complexity near `4/3` and operator
   complexity below 2 for the compact stable hierarchy; reject a level that is
   nearly dense.
5. **Use scheduled kernels.** Share immutable x/y topology where it is truly
   identical, keep component workspaces separate, and replace explicit `R=P'`
   storage with adjoint transfer actions.
6. **Retune alternatives.** Compare stable-core AMG, ILUT parameter sweeps, and
   stable-core incomplete Cholesky against optimized GMG and no preconditioner.
   Use setup-amortized time over realistic sequences, not one solve only.
7. **Add an optional L-stable integrator only with a failing test.** Regridding,
   discontinuous two-phase coefficients, or nonsmooth wall changes are the
   relevant cases. Never express this as a diffusion CFL.
8. **Generalize to tensor viscosity before two-phase flow.** Define coupled
   stress application, coefficient restriction, and coarse--fine traction
   conservation before assuming scalar component solves extend to a viscosity
   jump.

Acceptance for a new diffusion default should require numerical equivalence to
the corrected solve, no loss of mixed-level convergence, robustness over the
stiffness sweep, retained hierarchy memory no more than 2.5 times the pressure
preconditioner on the reference mesh, and lower setup-amortized time than plain
BiCGSTAB. If the last condition is not met for weakly stiff or small problems,
an automatic `nothing` preconditioner is preferable there.

## Sources

[^1]: S. McCormick and J. Thomas, [*The Fast Adaptive Composite Grid (FAC)
    Method for Elliptic Equations*](https://doi.org/10.1090/S0025-5718-1986-0829618-X),
    *Mathematics of Computation* 46 (1986), 439--456.
[^2]: D. F. Martin and K. L. Cartwright, [*Solving Poisson's Equation Using
    Adaptive Mesh Refinement*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1996/3105.html),
    UCB/ERL M96/66 (1996).
[^3]: Chombo-discharge, [*Linear solvers*](https://chombo-discharge.github.io/chombo-discharge/Source/LinearSolvers.html),
    documentation for Helmholtz discretization and `AMRMultiGrid`.
[^4]: Chombo, [*Software Package for AMR Applications: Design Document*](https://crd.lbl.gov/assets/pubs_presos/chomboDesign.pdf),
    Section 4.7, parabolic equations and TGA.
[^5]: AMReX, [*MLMG and Linear Operator Classes*](https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html),
    official linear-solver documentation.
[^6]: W. Zhang et al., [*AMReX: Block-Structured Adaptive Mesh Refinement for
    Multiphysics Applications*](https://doi.org/10.1177/10943420211022811),
    *International Journal of High Performance Computing Applications* (2021).
[^7]: J. Teunissen and U. Ebert, [*Afivo: A Framework for Quadtree/Octree AMR
    with Shared-Memory Parallelization and Geometric Multigrid Methods*](https://arxiv.org/abs/1701.04329),
    *Computer Physics Communications* 233 (2018), 156--166.
[^8]: hypre, [*SMG and PFMG*](https://hypre.readthedocs.io/en/latest/solvers-smg-pfmg.html),
    official structured-grid solver documentation.
[^9]: PETSc, [*KSP: Linear System Solvers*](https://petsc.org/main/manual/ksp/),
    official manual, especially operator/preconditioner separation and
    matrix-free multigrid.
[^10]: PETSc, [`PCMG`](https://petsc.org/release/manualpages/PC/PCMG/), official
    geometric-multigrid preconditioner documentation.
[^11]: R. D. Falgout and J. B. Schroder, [*Non-Galerkin Coarse Grids for
    Algebraic Multigrid*](https://doi.org/10.1137/130931539), *SIAM Journal on
    Scientific Computing* 36 (2014), C309--C334.
[^12]: P. Vanek, J. Mandel, and M. Brezina, [*Algebraic Multigrid by Smoothed
    Aggregation for Second and Fourth Order Elliptic Problems*](https://doi.org/10.1007/BF02238511),
    *Computing* 56 (1996), 179--196.
[^13]: T. C. Clevenger and T. Heister, [*Comparison Between Algebraic and
    Matrix-Free Geometric Multigrid for a Stokes Problem on Adaptive Meshes
    with Variable Viscosity*](https://arxiv.org/abs/1907.06696) (2019).
[^14]: hypre, [*BoomerAMG*](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html),
    official documentation for interpolation truncation, non-Galerkin
    dropping, and smoothers.
[^15]: H. A. van der Vorst, [*Bi-CGSTAB: A Fast and Smoothly Converging Variant
    of Bi-CG for the Solution of Nonsymmetric Linear Systems*](https://doi.org/10.1137/0913035),
    *SIAM Journal on Scientific and Statistical Computing* 13 (1992), 631--644.
[^16]: Y. Saad and M. H. Schultz, [*GMRES: A Generalized Minimal Residual
    Algorithm for Solving Nonsymmetric Linear Systems*](https://doi.org/10.1137/0907058),
    *SIAM Journal on Scientific and Statistical Computing* 7 (1986), 856--869.
[^17]: Y. Saad, [*ILUT: A Dual Threshold Incomplete LU Factorization*](https://doi.org/10.1002/nla.1680010405),
    *Numerical Linear Algebra with Applications* 1 (1994), 387--402.
[^18]: E. H. Twizell, A. B. Gumel, and M. A. Arigu, [*Second-Order,
    `L0`-Stable Methods for the Heat Equation with Time-Dependent Boundary
    Conditions*](https://doi.org/10.1007/BF02127712), *Advances in
    Computational Mathematics* 6 (1996), 333--352.
[^19]: R. Rannacher, [*Finite Element Solution of Diffusion Problems with
    Irregular Data*](https://doi.org/10.1007/BF01390130), *Numerische
    Mathematik* 43 (1984), 309--327.
[^20]: P. McCorquodale, P. Colella, and H. Johansen, [*A Cartesian Grid
    Embedded Boundary Method for the Heat Equation on Irregular Domains*](https://doi.org/10.1006/jcph.2001.6900),
    *Journal of Computational Physics* 173 (2001), 620--635.
[^21]: L. Batteux, T. Gallouet, R. Herbin, J.-C. Latche, and P. Poullet,
    [*Convergence of the Implicit MAC-Discretized Navier--Stokes Equations with
    Variable Density and Viscosity on Non-Uniform Grids*](https://arxiv.org/abs/2107.10117)
    (2021).
[^22]: Basilisk, [*An Implicit Solver for the Viscous Diffusion Equation*](https://basilisk.fr/src/viscosity.h),
    executable source documentation.
[^23]: M. E. Hosea and L. F. Shampine, [*Analysis and Implementation of
    TR--BDF2*](https://doi.org/10.1016/0168-9274(95)00115-8), *Applied Numerical
    Mathematics* 20 (1996), 21--37.
[^24]: M. Weinzierl and T. Weinzierl, [*Quasi-Matrix-Free Hybrid Multigrid on
    Dynamically Adaptive Cartesian Grids*](https://arxiv.org/abs/1607.00648)
    (2016).
