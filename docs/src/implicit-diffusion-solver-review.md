# Implicit staggered-diffusion solver: literature review

This review covers the original constant-viscosity, face-staggered Helmholtz
systems, their time integration, and their solution on a 2:1 block-tree mesh.
It also records the variable-viscosity requirements that guided the later
two-phase implementation. Current two-phase details are in the
[implementation record](two-phase-development.md). The conclusions combine
published literature with measurements; they do not claim that a method used
successfully for a cell-centred or finite-element operator transfers
automatically to this face graph.

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

The implemented design is therefore:

1. keep BiCGSTAB as the default outer Krylov method for the nonsymmetric
   corrected equation;
2. prune the block tree and rediscretize the exact component operator on each
   coarse composite mesh;
3. interpolate staggered faces geometrically between nested tree meshes and
   use the adjoint action for restriction; and
4. cache each sparse hierarchy and factorize only the root-block operator.

This is a standard defect-correction/preconditioning separation, not a change
to the equation. PETSc explicitly permits the operator used by Krylov and the
matrix used to build a preconditioner to differ, including a lower-order
preconditioning matrix.[^9] Mature block-AMR solvers likewise retain geometric
Helmholtz operators through their hierarchy and often use a generic AMG or
Krylov package only at the bottom level.[^3][^5]

An earlier stable-core/Galerkin prototype is measured below because it
explains why a component block alone was insufficient. That implementation
has since been removed in favor of direct tree rediscretization.
Generic AMG, ILU, and incomplete Cholesky remain valuable optional fallbacks,
but the untuned benchmark does not support replacing the geometric default.

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

## Spatial operator and implemented variable viscosity

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

This last point motivated the former Galerkin prototype. The final hierarchy
instead rebuilds the corrected component operator on every pruned tree and
does not form `P_l' * A_l * P_l`.

Independent scalar component solves also have a scope limit. For constant
viscosity and a divergence-free velocity,
`div(mu*(grad(u)+grad(u)'))` reduces to `mu*Delta(u)`. With spatially varying or
discontinuous viscosity, gradients of `mu` remain and the symmetric stress
operator couples velocity components. Basilisk's implicit viscosity operator
solves the mutually coupled stress equations, and AMReX provides a tensor
operator for viscous terms.[^22][^5] Analysis of implicit MAC schemes on
nonuniform Cartesian grids likewise treats variable density and viscosity in
the full discrete problem.[^21]

The two-phase extension consequently implements a coupled staggered stress
operator for `div(2*mu*D(u))` rather than multiplying the scalar Laplacian by a
coefficient. It evaluates the dynamic viscosity at normal- and shear-stress
locations with the thesis geometric phase mixture and places density only in
the implicit inertia matrix. The original constant-viscosity component solver
remains the optimized separable specialization.

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

## Baseline audit and implementation outcome

The audit uses the same graded mesh as `benchmark/preconditioner_comparison.jl`:
7,168 cells, 14,400 owned faces, `coarse_size=32`, and relative diffusion
stiffness 40. `A` is the retained shifted matrix, `K` the stable stiffness used
during setup, and `P` the prolongation schedule from that row to the next
level.

### Rediscretized tree hierarchy

The current `benchmark/hierarchy_complexity.jl` run gives:

| operator | levels | grid complexity | operator complexity | setup allocation | setup time | retained wrapper |
|---|---:|---:|---:|---:|---:|---:|
| pressure | 4 | 1.429 | 1.430 | 45,995,144 B | 13.687 ms | 15,858,112 B |
| staggered diffusion | 4 | 1.429 | 1.457 | 79,459,520 B | 37.143 ms | 16,318,512 B |

The pressure levels contain 7,168, 1,792, 1,024, and 256 unknowns. Their
matrices contain 36,480, 9,280, 5,120, and 1,280 nonzeros. One staggered
component contains 7,200, 1,808, 1,024, and 256 unknowns, with 66,664, 24,088,
5,120, and 1,280 matrix nonzeros. The x and y counts are identical on this
symmetric reference mesh.

These levels are the operators obtained by pruning the AMR block tree and
rediscretizing on each resulting mesh. The former hierarchy produced growing
Galerkin stencils and ended at irregular 28-cell pressure and 129-face
component aggregates. That implementation and its dense bottom conversion
have been removed. The current root systems are sparse 256-unknown
factorizations. Setup allocation now comes from constructing physical coarse
meshes and their discrete operators; no sparse triple product is formed.

## Historical Krylov and preconditioner assessment

The measurements in this section describe the retired stable-core/Galerkin
prototype. They remain useful as evidence for the Krylov choices, but they are
not current tree-hierarchy timings.

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

BiCGSTAB remains the default for this nonsymmetric corrected operator.
Krylov.jl GMRES and FGMRES workspaces are
supported, including restart and reorthogonalization settings through
`krylov_options`, and regression tests compare their final true solutions. On
the reference hierarchy, BiCGSTAB takes 10.535 ms and 5/5 iterations;
GMRES(20) takes 11.784 ms and 10/10; FGMRES(20) takes 10.821 ms and 10/10 but
uses 4.96 MB of Krylov workspace versus 0.92 MB for BiCGSTAB. Broader residual-
history sweeps remain useful, but this point supports the short-recurrence
default.

### Geometric stable-core multigrid

This has the strongest evidence for the long-term default. It exploits the
known tree geometry, can use the compact conservative operator, and already
finds a good coarse space. Matrix-free adaptive GMG has shown robust scaling
for variable-viscosity Stokes systems in a different finite-element setting,
including better refinement robustness than the tested AMG implementation.[^13]
That result supports the architecture, not a direct performance prediction for
this low-order 2-D face operator.

The pure `B^-1 A` hypothesis did not hold strongly enough: an exact stable-core
factorization still required 26/27 BiCGSTAB iterations at relative stiffness
40. The deployed hybrid uses corrected-`A` Jacobi smoothing on the fine level
and SPD stable-core coarse correction. It reaches 5/5 iterations while keeping
the coarse hierarchy compact and leaves the outer equation unchanged.

### Generic AMG

Classical smoothed aggregation was developed for SPD elliptic systems.[^12]
Applying a default SA setup directly to the nonsymmetric corrected matrix has
no general robustness guarantee. The integration now builds named AMG from
the stable core and uses it against the corrected Krylov operator. On the
reference case this improves the old 38/37 result to 27/29 iterations, but its
55.529 ms solve remains much slower than both geometric MG and no
preconditioner. Remaining experiments are:

1. strength, coarsening, interpolation, truncation, and smoother sweeps; and
2. a nonsymmetric-capable implementation such as BoomerAMG if an additional
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

Incomplete Cholesky cannot factor the full nonsymmetric matrix. The public
`:ichol` path now builds it from `M+alpha*K` and uses it as a right
preconditioner for the corrected BiCGSTAB/GMRES equation. It takes 30/29
iterations and 17.797 ms in the reference run: applicable and memory-efficient,
but not competitive at its current parameters. The same operator/
preconditioning-operator separation is used for named AMG and ILU.

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
- perform restriction and prolongation from the stored geometric sparse
  interpolation and its adjoint;
- rediscretize the corrected Helmholtz action on every block-tree level;
- keep a small assembled bottom matrix; and
- materialize a sparse matrix lazily only for AMG, ILU, debugging, eigenvalue
  tests, or export.

Making only the finest `A*x` matrix free would save its 66,664 entries per
component. The refactor has already removed the larger avoidable costs:
Galerkin triple products, retained coarse stiffness duplicates, and explicit
restriction matrices. A fine stencil schedule may still reduce
memory traffic, but it is now an incremental optimization that should be
benchmarked against the compact assembled CSC action rather than assumed to
win.

The low-order stencil also changes the expected payoff. Matrix-free methods
are especially advantageous for high-order finite elements, where sum
factorization makes operator application much cheaper than reading a large
matrix. Here the main benefit is eliminating pathological Galerkin fill and
temporary sparse products, not reducing arithmetic. A compact assembled
stencil or stencil schedule may perform as well as a purist matrix-free
implementation; both should satisfy the same operator-complexity target.

## Development sequence and status

1. **Metrics: implemented, broader sweeps remain.** For every hierarchy record grid
   complexity, `A`, `K`, and transfer nonzeros, setup allocation, retained
   bytes, V-cycle time, Krylov time, and true residual history. Sweep mesh size,
   refinement pattern, `alpha*lambda_max`, and wall condition.
2. **Stable-core prototype: retired.** It established the convergence limits
   of a component approximation but depended on Galerkin coarse matrices.
3. **Block-tree rediscretization: implemented.** Each coarse level drops
   terminal block families and rebuilds the staggered operator with its wall
   and AMR closures.
4. **Geometric face transfer: implemented.** Coincident faces use overlap and
   new face planes interpolate between neighbouring coarse planes.
5. **Persistent hierarchy storage: implemented.** Fixed-topology updates
   refresh shifted matrices, and AMR rebuilds retain matching coarse levels.
6. **Alternative baseline: implemented; tuning remains.** Compare stable-core AMG, ILUT parameter sweeps, and
   stable-core incomplete Cholesky against optimized GMG and no preconditioner.
   Use setup-amortized time over realistic sequences, not one solve only.
7. **Add an optional L-stable integrator only with a failing test.** Regridding,
   discontinuous two-phase coefficients, or nonsmooth wall changes are the
   relevant cases. Never express this as a diffusion CFL.
8. **Coupled tensor viscosity: implemented separately for two-phase flow.**
   The scalar component hierarchy remains the constant-viscosity
   specialization.

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
