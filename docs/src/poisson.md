# Geometric multigrid and pressure projection

## Operator contract

The cellwise Gauss divergence remains the primal operator. Let `C` be the
oriented cell-to-face incidence matrix, `A` the diagonal matrix of face-segment
areas, `V` the diagonal cell-volume matrix, and `K` the inverse face Riesz map
constructed by `GradientOperator`. The existing operators are

```math
D = -V^{-1} C^T A, \qquad G = K A C.
```

The positive integrated Poisson stiffness is therefore

```math
S = -V D G = C^T A K A C.
```

`EllipticOperator` stores the canonical shifted form

```math
\mathcal L u = a u - b\,\nabla\cdot(\kappa\nabla u),
\qquad A_h = aV + bS_\kappa.
```

`PoissonOperator(mesh)` selects `a=0`, `b=1`. For implicit midpoint viscosity,
the `StaggeredEllipticOperator` uses the shifted coefficient `b=dt*nu/2`.

For a diagonal Cartesian face metric, a face coefficient simply multiplies
the ordinary normal gradient. At a refinement interface `K` is non-diagonal,
so left multiplication by a diagonal coefficient would generally destroy
symmetry. The implemented extension is

```math
S_\kappa = C^T A \sqrt{\kappa}\,K\sqrt{\kappa}\,A C.
```

It is symmetric positive semidefinite, preserves constants, and reduces to
the ordinary coefficient-weighted stencil in every conforming region. The
public `harmonic_face_coefficients` helper forms distance-weighted harmonic
means from positive cell values, including unequal half-cell distances at a
coarse--fine face. `weighted_gradient!` applies precisely the flux gradient
used in `S_κ`; the projection never reconstructs a second, inconsistent
coefficient-weighted gradient.

## Tree-geometric hierarchy

The hierarchy first follows the block tree. Each coarse mesh replaces terminal
sibling blocks with their parent while restoring 2:1 grading. With `pruning=:all`,
every available terminal family is removed in a pass; `pruning=:deepest`
removes only the deepest generation. Once the root level is reached, the
hierarchy continues by agglomerating 2×2 root blocks into coarser uniform
meshes with the same block size, extent, and periodicity. This is independent
of the simulation mesh: its permanent roots are not changed.

Coarsening stops once the target `coarse_size` or `max_levels` is reached, or
when either root-block count cannot be halved. Each periodic axis must also
retain at least four cells: with two cells, opposite centres are separated by
exactly half a period, and the current wrapped face displacement can make the
gradient metric's vertex patches degenerate. A nonperiodic axis may coarsen to
two cells. Consequently, the last mesh can have more than `coarse_size` cells
when no further base coarsening is available.

Every level rebuilds the compatible pressure operator on its own composite
mesh. Face coefficients are restricted by geometric overlap, then the coarse
stiffness and metric are assembled directly. The implementation never forms
`Pᵀ A P`. A coefficient-dependent smoothed cell interpolation preserves
constants, and its adjoint restricts integrated residuals.

V-cycles use pre- and post-damped-Jacobi relaxation. The last available sparse
operator is factorized as the bottom solver; periodic and all-Neumann pressure
use the volume-weighted nullspace augmentation. V-cycle work vectors are
cached. Across AMR updates, matching coarse tree and base-mesh levels,
transfers, operators, and work storage are retained and refreshed in place.
With embedded geometry, coarse mesh topology is reused while the
geometry-dependent quantities are restricted again.

`TreeGeometricMultigridPreconditioner` implements three-argument `ldiv!` and
contains no Krylov policy. `EllipticSolver` composes it with a Krylov.jl
workspace. The `:geometric` selector also selects this tree implementation.

## Staggered shifted hierarchy

The viscous problem has two separate sets of unknowns: face-normal velocities
on the x- and y-dual grids. Their integrated midpoint matrices are

```math
A_a = M_a\left(I-\frac{\Delta t\nu}{2}L_a\right),
\qquad a\in\{x,y\}.
```

`TreeStaggeredGeometricMultigridPreconditioner` uses the same tree pruning and
root-block agglomeration and reconstructs `StaggeredDiffusionOperator` on each
coarse mesh.
Face interpolation is geometric: coincident faces use tangential overlap,
and newly exposed faces interpolate between adjacent coarse face planes.
Both searches use sorted lattice-line indexes: binary interval lookup visits
only overlapping segments, and an outward walk finds the nearest bracketing
normal lines without rescanning every coarse face. Restriction is its adjoint.
Each coarse matrix contains the single-phase Laplacian and the same wall and
geometric AMR contact closures as that mesh's fine operator; no Galerkin
product is used.

The integrated AMR diffusion operator is symmetric positive definite for a
positive reaction term, so `StaggeredEllipticSolver` uses preconditioned CG by
default. Each component has its own V-cycle. Bottom systems with at most 256 unknowns use
retained dense LU; larger ones use sparse LU with symbolic reuse when their
pattern is unchanged. A timestep change refreshes the shifted matrices and
factors without rebuilding the hierarchy or Krylov workspaces. Matching levels
are reused after an AMR topology change.
The `:geometric` selector also selects this tree implementation.

## Preconditioner choices

The built-in and optional choices use one constructor interface:

| value | package | pressure | diffusion |
|---|---|---:|---:|
| `:tree_geometric` | TwoPhaseAMR | default | default; componentwise for one phase, coupled stress for two phases |
| `:geometric` | TwoPhaseAMR | compatibility alias | compatibility alias for `:tree_geometric` |
| `:amg` | AlgebraicMultigrid.jl | yes | yes |
| `:ichol` | LimitedLDLFactorizations.jl | yes | yes, from the stable core |
| `:ilu` | IncompleteLU.jl | use a nonsymmetric Krylov method | yes |
| `nothing` | none | yes | yes |

The optional packages are weak dependencies. Import the selected package before
constructing the solver, for example `import AlgebraicMultigrid`. For
staggered diffusion, the named `:amg`, `:ilu`, and `:ichol` choices are built
from the symmetric stable-core matrix but precondition a Krylov solve using
the exact corrected matrix. A custom factory or pre-built `(x=..., y=...)`
preconditioner remains available when direct access to the corrected matrices
is desired. AMG supports
smoothed aggregation and Ruge--Stuben setup; incomplete LU exposes its drop
tolerance; and incomplete Cholesky exposes limited fill, dropping, and a
diagonal shift. Options are passed through `preconditioner_options`.

ILU is intentionally not presented as a CG preconditioner because a generic
ILU factorization need not be symmetric positive definite. It can still be
used for pressure by explicitly selecting `Krylov.BicgstabWorkspace`.

This follows the established adaptive multigrid pattern. FAC introduced
coarse-grid correction on composite adaptive grids. Afivo demonstrates
geometric multigrid on fixed, even-sized quadtree/octree blocks, including
coarsening below the AMR level. AMReX likewise separates a multilevel solver
from canonical constant- and variable-coefficient elliptic operators and uses
agglomeration to continue geometric coarsening.

Method references and software documentation:

- S. F. McCormick and J. W. Thomas, [*The fast adaptive composite grid (FAC)
  method for elliptic equations*](https://doi.org/10.1090/S0025-5718-1986-0829618-X),
  Mathematics of Computation 46 (1986), 439--456.
- J. Teunissen and U. Ebert, [*Afivo: A framework for quadtree/octree AMR with
  shared-memory parallelization and geometric multigrid
  methods*](https://doi.org/10.1016/j.cpc.2018.06.018), Computer Physics
  Communications 233 (2018), 156--166.
- S. Popinet, [*Gerris: A tree-based adaptive solver for the incompressible
  Euler equations in complex
  geometries*](https://doi.org/10.1016/S0021-9991(03)00298-5), Journal of
  Computational Physics 190 (2003), 572--600.
- [AMReX linear-solver documentation](https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html)
- W. Zhang, A. Myers, K. Gott, A. Almgren, and J. Bell, [*AMReX:
  Block-structured adaptive mesh refinement for multiphysics
  applications*](https://doi.org/10.1177/10943420211022811), International
  Journal of High Performance Computing Applications 35 (2021), 508--526.
- [PETSc `KSPCG` documentation](https://petsc.org/release/manualpages/KSP/KSPCG/)
- [hypre BoomerAMG documentation](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
- P. Vaněk, J. Mandel, and M. Brezina, [*Algebraic multigrid by smoothed
  aggregation for second and fourth order elliptic
  problems*](https://doi.org/10.1007/BF02238511), Computing 56 (1996),
  179--196.
- O. Tatebe, [*The multigrid preconditioned conjugate gradient
  method*](https://ntrs.nasa.gov/citations/19940017009), Proceedings of the
  Sixth Copper Mountain Conference on Multigrid Methods, NASA Conference
  Publication 3224 (1993), 621--634.
- [PETSc `KSPBCGS` documentation](https://petsc.org/release/manualpages/KSP/KSPBCGS/)
- H. A. van der Vorst, [*Bi-CGSTAB: A fast and smoothly converging variant of
  Bi-CG for the solution of nonsymmetric linear
  systems*](https://doi.org/10.1137/0913035), SIAM Journal on Scientific and
  Statistical Computing 13 (1992), 631--644.
- Y. Saad and M. H. Schultz, [*GMRES: A generalized minimal residual algorithm
  for solving nonsymmetric linear systems*](https://doi.org/10.1137/0907058),
  SIAM Journal on Scientific and Statistical Computing 7 (1986), 856--869.
- Y. Saad, [*ILUT: A dual threshold incomplete LU
  factorization*](https://doi.org/10.1002/nla.1680010405), Numerical Linear
  Algebra with Applications 1 (1994), 387--402.
- T. C. Clevenger and T. Heister, [*Comparison between algebraic and
  matrix-free geometric multigrid for a Stokes problem on adaptive meshes with
  variable viscosity*](https://doi.org/10.1002/nla.2375), Numerical Linear
  Algebra with Applications 28 (2021), e2375.

## Nullspace and convergence

Pressure has a constant null mode on each connected fluid component that
has no positive open-face Dirichlet conductance and no inactive-cell
Dirichlet anchor. Components are connected through faces with positive
pressure coefficients; an embedded solid can separate them. On prescribed
velocity boundaries the normal pressure correction is homogeneous Neumann,
so it cannot change the prescribed normal flux.

`solve!` fixes the volume-weighted pressure mean on each unanchored component.
Its default `compatibility=:project` subtracts the volume mean of the strong
right-hand side on each such component.
`LinearSolverStats.removed_rhs_component_means` reports these means in the
order of `removed_rhs_component_anchors`, the smallest cell index in each
unanchored component. `removed_rhs_mean` is their volume-weighted mean and
may be zero even when the component means are not. `compatibility=:error`
rejects a component mean beyond its roundoff check. Projecting an incompatible
right-hand side makes the linear system solvable but cannot remove its
physical boundary-flux imbalance: immediately after the correction, the
remaining divergence includes the corresponding componentwise contribution
in addition to the residual.

The multigrid bottom solve uses one volume-weighted augmentation for each
unanchored component, generalizing

```math
\begin{bmatrix} A & v \\ v^T & 0 \end{bmatrix}
\begin{bmatrix} p \\ \lambda \end{bmatrix}
=
\begin{bmatrix} b \\ 0 \end{bmatrix},
```

where `v` is the coarse representation of cell volume on that component.
Shifted diffusion operators and components with a positive open Dirichlet
conductance have no constant nullspace. When no component has a nullspace, the
bottom-grid factorization is ordinary and no mean is projected. The initial
solve state supplies the starting roundoff floor. A current-iterate floor can
increase the stop level only after the residual decreases and while that floor
is less than 1% of the initial residual.

For the manufactured periodic problem

```math
-\Delta p = 8\pi^2\sin(2\pi x)\cos(2\pi y),
```

on the standard two-level mesh, the tests give:

| Block cells | Active cells | PCG iterations | L2 error | L∞ error |
|---:|---:|---:|---:|---:|
| 4×4 | 112 | 8 | 2.882e-2 | 5.587e-2 |
| 8×8 | 448 | 8 | 6.749e-3 | 1.555e-2 |
| 16×16 | 1,792 | 9 | 1.643e-3 | 4.049e-3 |

Both norms converge at approximately second order and the PCG count changes
by one over a factor of sixteen in active cells. Separate tests establish
exact uniform block-partition independence, operator symmetry and nullity,
variable-coefficient duality, staggered shifted solves, and projection on
periodic and impermeable-wall mixed-level trees.

## API

```julia
mesh = BlockTreeMesh(
    block_size=(8, 8), max_level=2,
    refine=(key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
)

operator = PoissonOperator(mesh)
solver = EllipticSolver(operator) # Krylov.jl CG + geometric MG
pressure = CellField(mesh)
rhs = CellField(mesh)
initialize!(rhs, (x, y) -> 8pi^2 * sinpi(2x) * cospi(2y))
stats = solve!(pressure, solver, rhs; rtol=1e-10)

# The layers can also be assembled explicitly.
using Krylov
multigrid = TreeGeometricMultigridPreconditioner(operator)
krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
explicit_solver = EllipticSolver(operator, krylov, multigrid)

# Optional AMG integration.
import AlgebraicMultigrid
amg_solver = EllipticSolver(operator; preconditioner=:amg)

# The initial physical pressure is zero.  For a nonzero old pressure the
# velocity supplied to project! must already include its pressure acceleration.
pressure = CellField(mesh)
velocity = FaceVelocity(mesh, velocity_function)
projection = ProjectionWorkspace(solver)
project!(velocity, pressure, solver, dt, projection; tolerance=1e-8)
```

The standalone algebraic `solve!` retains `rtol` and `atol` for expert use.
The physical `project!` API takes a pressure-predicted velocity and adds its
correction to the supplied old physical pressure. The correction has zero
volume-weighted mean on each unanchored component; an open pressure boundary
instead fixes the increment there. It accepts one dimensionless `tolerance`
and solves with the absolute strong-residual target `tolerance/dt^2`.
Consequently `stats.final_linf/stats.residual_scale` equals
`dt^2*stats.final_linf`, because `stats.residual_scale = 1/dt^2`. This is the
fractional volume defect associated with the linear-solve residual, up to
the roundoff-limited target. `stats.compatible` separately compares `dt^2`
times the largest absolute removed strong component RHS mean with the
requested tolerance, not the roundoff-raised target. Thus `converged` can be
true while `compatible` is false. For non-projection linear solves,
`compatible` does not assess physical compatibility. An incompatible
one-velocity projection warns once per projection workspace, retained across
mesh rebuilds. A shortened simulation step can additionally retain part of
its inherited divergence, as described in
[Incompressible Navier--Stokes solver](navier-stokes.md#common-step).

`preconditioner` may also be a factory accepting the sparse matrix or a
pre-built object implementing `ldiv!(destination, preconditioner, source)`.
`krylov` may be a supported Krylov.jl workspace factory or, for a cell
elliptic solve, a pre-built compatible workspace. CG, BiCGSTAB, GMRES, and
FGMRES workspaces are supported. Method-specific settings such as
`restart=true` and `reorthogonalization=true` are supplied through
`krylov_options`; the package retains control of residual tolerances,
iteration limits, and preconditioner placement. The default pressure method
is CG. Staggered diffusion selects CG for a volume-weighted symmetric operator,
including the embedded conservative operator, and BiCGSTAB otherwise. CG with a
tree-geometric multigrid preconditioner requires equal `pre_sweeps` and
`post_sweeps`; unequal counts raise `ArgumentError`. Other Krylov methods may
use unequal counts.

The operator and solver setup may allocate; `apply!`, a warmed `solve!`, and
the projection kernels reuse their workspaces. The implementation is 2-D,
uses shared-memory threading, and requires isotropic mixed-level cells through
the underlying `GradientOperator`. Periodic and homogeneous-Neumann
pressure-correction boundaries are supported, as are homogeneous Dirichlet
increments on open velocity faces when the operator is constructed with
`velocity_boundary`. General inhomogeneous pressure data and a structured
single-block FFT option remain future extensions.
