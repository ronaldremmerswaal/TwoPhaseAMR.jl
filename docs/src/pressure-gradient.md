# Compatible pressure gradient

`GradientOperator` keeps the finite-volume divergence as the primary
topological operator. For cell volumes `V`, oriented cell--face incidence `B`,
and leaf-face areas `A`, the existing divergence is

```math
D = V^{-1} B A.
```

The pressure gradient is derived from this operator rather than discretized
independently:

```math
G = -K A B^T,
```

where `K` is a sparse symmetric positive-definite inverse face Riesz map. The
pressure operator therefore satisfies

```math
-V D G = B A K A B^T \succeq 0.
```

Constants are in its null space, and the gradient and divergence are negative
adjoints in the face inner product induced by ``K^{-1}``.

## Vertex-patch construction

Each physical leaf-face segment belongs to its two endpoint patches. For one
patch, let the rows of `Y` contain

```math
Y_f = |f|(x_+ - x_-),
```

where the normal component of the centre difference is positive — the plus cell
lies on the coordinate-normal side of the shared face — and its magnitude is the
shortest periodic image, which also resolves a periodic direction with only two
cells; the tangential component takes its shortest periodic image. `N` contains
half of the coordinate-normal vector for each face. The local metric is

```math
K_v = N (Y^T N)^{-1} N^T + \gamma
      \left[I-Y(Y^TY)^{-1}Y^T\right],
```

with ``\gamma`` chosen so the stabilization and consistent terms have equal
trace. The first term gives `K_v Y = N`, hence exact affine
pressure gradients after assembling both endpoint contributions. The second
term controls every remaining patch mode and makes the local map positive
definite.

On a conforming Cartesian mesh, the assembled map reduces exactly to the
diagonal MAC coefficient ``1/|\omega_f|``. `GradientOperator` detects this case
and uses a direct diagonal kernel. Only coarse--fine vertex patches introduce
off-diagonal entries; the adaptive kernel applies their precomputed sparse
schedule without allocations or tree traversal. The manufactured periodic AMR
test records the expected better-than-``h^{1.4}`` global weighted-L2 convergence
for a first-order interface layer embedded in a second-order regular grid.

```julia
pressure = CellField(mesh)
gradient = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
operator = GradientOperator(pressure)

gradient!(gradient, pressure, operator)
```

The current slice accepts periodic and non-periodic meshes and requires
isotropic cells when levels are mixed. At a physical wall the normal pressure
correction is homogeneous Neumann, so its boundary face gradient is exactly
zero; paired with `enforce_no_permeation!`, this preserves `u dot n = 0` during
projection. For a one-velocity open face, `weighted_gradient!` with a
boundary-configured `PoissonOperator` instead uses the homogeneous-Dirichlet
pressure increment and the local half-cell distance; `gradient!` itself
continues to use the homogeneous-Neumann physical-boundary closure.
General inhomogeneous pressure data and the tensor metric needed for stretched
coarse--fine patches remain explicit future extensions.

## Accuracy on adaptive meshes

On a conforming Cartesian mesh the operator is the ordinary second-order MAC
gradient. Coarse--fine couplings introduce a localized first-order interface
error, while smooth periodic problems retain approximately second-order cell
pressure accuracy in the global L2 norm. The compatible construction preserves
the adjoint relation and constant nullspace across the same interfaces.

## References

- P. H. A. J. M. van der Plas, [*Local grid refinement for free-surface flow
  simulations*](https://research.rug.nl/en/publications/local-grid-refinement-for-free-surface-flow-simulations/),
  Chapter 4, PhD thesis, University of Groningen (2017).
- J. Bonelle and A. Ern, [*Analysis of compatible discrete operator schemes
  for elliptic problems on polyhedral meshes*](https://doi.org/10.1051/m2an/2013104),
  ESAIM: Mathematical Modelling and Numerical Analysis 48 (2014), 553--581.
- M. A. Olshanskii, K. M. Terekhov, and Y. V. Vassilevski, [*An octree-based
  solver for the incompressible Navier--Stokes equations with enhanced
  stability and low dissipation*](https://doi.org/10.1016/j.compfluid.2013.04.027),
  Computers & Fluids 84 (2013), 231--246.
- T. V. Gerya, D. A. May, and T. Duretz, [*An adaptive staggered grid finite
  difference method for modeling geodynamic Stokes flows with strongly
  variable viscosity*](https://doi.org/10.1002/ggge.20078), Geochemistry,
  Geophysics, Geosystems 14 (2013), 1200--1225.
