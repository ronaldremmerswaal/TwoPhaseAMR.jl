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

`PoissonOperator(mesh)` selects `a=0`, `b=1`. Both this cell operator and the
component-wise `StaggeredEllipticOperator` implement the common
`AbstractEllipticOperator` contract. For implicit midpoint viscosity the
staggered shifted coefficient is `b=dt*nu/2`.

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

The finest unknowns are the active leaf cells. Each cell is represented during
setup by an integer rectangle on the finest dyadic lattice. One coarsening pass
merges every complete aligned 2×2 sibling group. Unmatched cells beside a
level transition remain at their current scale and can merge in a later pass
after finer neighbours have caught up. This reconstructs the quadtree cell
hierarchy inside blocks and across former block boundaries; no tree traversal
occurs during a V-cycle.

The tentative prolongation is constant on each geometric aggregate. One
damped Jacobi smoothing step with the diffusion stiffness widens it enough to
represent smooth error across block and level transitions while retaining
`P*1=1`. Restriction is `Pᵀ` and every coarse matrix is the Galerkin product
`Pᵀ A P`. Consequently symmetry, positivity, and the constant null mode are
inherited at every level. This is a geometric hierarchy with a
smoothed-aggregation transfer, a useful compromise for the wider mimetic AMR
stencil.

V-cycles use pre- and post-damped-Jacobi relaxation. The damping is bounded
from the sparse row sums during setup. Once the remaining problem has at most
`coarse_size` unknowns, a dense factorization is cached as the bottom solver.
All solution, residual, correction, and transfer work vectors are cached.

`GeometricMultigridPreconditioner` is only a preconditioner: it implements
three-argument `ldiv!` and contains no Krylov workspace, convergence policy, or
iteration statistics. `EllipticSolver` composes the operator, a workspace from
[Krylov.jl](https://jso.dev/Krylov.jl/stable/), and an independent
preconditioner. The default composition is `Krylov.CgWorkspace` plus one
complete symmetric geometric V-cycle. The matrix, Galerkin hierarchy,
weighted-Jacobi smoother, restriction/prolongation pair, and bottom solve are
symmetric positive (semi-)definite, as required by preconditioned CG.

This use of a complete V-cycle as a Krylov preconditioner follows standard
multigrid practice. PETSc requires both the operator and preconditioner used by
CG to be symmetric positive (semi-)definite, while hypre likewise requires a
symmetric AMG relaxation when BoomerAMG preconditions CG. The symmetric cell
hierarchy satisfies those conditions; the nonsymmetric staggered hierarchy
below deliberately does not use CG.

## Staggered shifted hierarchy

The viscous problem has two separate sets of unknowns: face-normal velocities
on the x- and y-dual grids. Their integrated midpoint matrices are

```math
A_a = M_a\left(I-\frac{\Delta t\nu}{2}L_a\right),
\qquad a\in\{x,y\}.
```

`StaggeredGeometricMultigridPreconditioner` builds one hierarchy per
component.
Twice-finest-cell integer coordinates represent both normal-face and
tangential half-cell positions exactly. Aligned dyadic bins form tentative
dual-grid aggregates; coarse tree faces remain singletons until the bin size
reaches their scale. The prolongation is smoothed with the symmetric stable
diffusion core.

The widened consistency correction near AMR interfaces makes `L_a` mildly
nonsymmetric. Its Galerkin matrices are therefore *not* symmetrized: doing so
would change the diffusion equation being solved. Damped Jacobi relaxation
uses an absolute-row-sum bound, and LU is used only after geometric coarsening
reaches `coarse_size`. With an unchanged time step the complete hierarchy and
bottom factorization are reused.

For this nonsymmetric system, `StaggeredEllipticSolver` defaults to
`Krylov.BicgstabWorkspace` with right-preconditioned geometric V-cycles. It
owns the two external Krylov workspaces and rebuilds their operators and
preconditioners only when the implicit coefficient changes. The geometric
hierarchies remain unaware of BiCGSTAB.

## Preconditioner choices

The built-in and optional choices use one constructor interface:

| value | package | pressure CG | staggered diffusion BiCGSTAB |
|---|---|---:|---:|
| `:geometric` | TwoPhaseAMR | yes (default) | yes (default) |
| `:amg` | AlgebraicMultigrid.jl | yes | yes |
| `:ichol` | LimitedLDLFactorizations.jl | yes | no: operator is nonsymmetric |
| `:ilu` | IncompleteLU.jl | use a nonsymmetric Krylov method | yes |
| `nothing` | none | yes | yes |

The optional packages are weak dependencies. Import the selected package before
constructing the solver, for example `import AlgebraicMultigrid`. AMG supports
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

Sources:

- [McCormick and Thomas, *The Fast Adaptive Composite Grid Method for Elliptic Equations*](https://doi.org/10.1090/S0025-5718-1986-0829618-X)
- [Teunissen and Ebert, *Afivo*](https://doi.org/10.1016/j.cpc.2018.06.018)
- [AMReX linear-solver documentation](https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html)
- [Zhang et al., *AMReX: Block-structured adaptive mesh refinement for multiphysics applications*](https://doi.org/10.1177/10943420211022811)
- [PETSc `KSPCG` documentation](https://petsc.org/release/manualpages/KSP/KSPCG/)
- [hypre BoomerAMG documentation](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
- [Tatebe, *The Multigrid Preconditioned Conjugate Gradient Method*](https://ntrs.nasa.gov/citations/19940017009)
- [PETSc `KSPBCGS` documentation](https://petsc.org/release/manualpages/KSP/KSPBCGS/)
- [van der Vorst, *Bi-CGSTAB*](https://doi.org/10.1137/0913035)

## Nullspace and convergence

For periodic or impermeable-wall domains, constants are the only pressure
null mode. A non-periodic velocity boundary is set automatically to
`u dot n = 0`, and the normal pressure correction is homogeneous Neumann.
`solve!` fixes the solution by enforcing a volume-weighted zero mean. Its
default `compatibility=:project` removes the volume mean of the strong
right-hand side and records it in `LinearSolverStats.removed_rhs_mean`;
`compatibility=:error` instead rejects incompatible data. The bottom solve
uses the augmented system

```math
\begin{bmatrix} A & v \\ v^T & 0 \end{bmatrix}
\begin{bmatrix} p \\ \lambda \end{bmatrix}
=
\begin{bmatrix} b \\ 0 \end{bmatrix},
```

where `v` is the coarse representation of cell volume. Shifted diffusion
operators have no nullspace and use an ordinary bottom-grid factorization.

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
multigrid = GeometricMultigridPreconditioner(operator)
krylov = Krylov.CgWorkspace(operator.matrix, zeros(ncells(mesh)))
explicit_solver = EllipticSolver(operator, krylov, multigrid)

# Optional AMG integration.
import AlgebraicMultigrid
amg_solver = EllipticSolver(operator; preconditioner=:amg)

velocity = FaceVelocity(mesh, velocity_function)
projection = ProjectionWorkspace(solver)
project!(velocity, pressure, solver, dt, projection; rtol=1e-10)
```

`preconditioner` may also be a factory accepting the sparse matrix or a
pre-built object implementing `ldiv!(destination, preconditioner, source)`.
`krylov` may be a Krylov.jl workspace factory or, for a cell elliptic solve, a
pre-built compatible workspace. The default pressure method remains CG and
the default staggered diffusion method remains BiCGSTAB.

The operator and solver setup may allocate; `apply!`, a warmed `solve!`, and
the projection kernels reuse their workspaces. The current implementation is
2-D and serial and requires isotropic mixed-level cells through the underlying
`GradientOperator`. Periodic and homogeneous-Neumann pressure-correction
boundaries are supported. General pressure Dirichlet/inhomogeneous Neumann
data and a structured single-block FFT option remain future extensions.
