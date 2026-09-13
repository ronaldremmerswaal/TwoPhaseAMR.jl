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

using the shortest periodic displacement, and let `N` contain half of the
coordinate-normal vector for each face. The local metric is

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
projection. General inhomogeneous pressure data and the tensor metric needed
for stretched coarse--fine patches remain explicit future extensions.

## Mixed-level wave-equation validation

The divergence--gradient pair is also tested dynamically with the acoustic
system

```math
\partial_t p = -D u, \qquad \partial_t u = -G p
```

and the periodic standing wave

```math
p(x,y,t) = \sin(2\pi(x+0.13))\cos(2\pi(y-0.07))
           \cos(2\pi\sqrt{2}\,t).
```

Time integration uses explicit Störmer--Verlet in kick--drift--kick form with
``\Delta t=0.1h_{\min}``. This second-order symplectic method has a genuine
stability interval for the imaginary spectrum of the wave operator; explicit
two-stage RK2 does not. The fixed tree contains both level-1 and level-2 leaf
blocks, with the southwest level-1 block refined. At ``t=0.1`` the regression
test gives:

| Block cells | Pressure L2 | Rate | Pressure LInf | Rate | Velocity L2 | Rate | Velocity LInf | Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 x 8 | 4.0591e-3 | -- | 1.3475e-2 | -- | 4.7349e-3 | -- | 1.8530e-2 | -- |
| 16 x 16 | 9.5005e-4 | 2.10 | 4.0782e-3 | 1.72 | 1.4802e-3 | 1.68 | 8.5197e-3 | 1.12 |
| 32 x 32 | 2.3767e-4 | 2.00 | 1.1421e-3 | 1.84 | 4.8519e-4 | 1.61 | 4.0732e-3 | 1.06 |

The cell pressure is approximately second-order in L2 and better than order
1.7 in LInf. The face velocity is better than order 1.5 in L2, while its
maximum error is first-order because the largest gradient error remains
localized at coarse--fine faces.

This is a compatible/mimetic construction: topology remains in `D`, while
metric closure and refinement coupling live in `K`. It deliberately avoids the
F1 construction's adjoint interpolation correction in the primal divergence.

References:

- P. van der Plas, *Local Grid Refinement for Free-Surface Flow Simulations*,
  Chapter 4, University of Groningen, 2017.
- J. Bonelle and A. Ern, “Analysis of Compatible Discrete Operator schemes for
  elliptic problems on polyhedral meshes,” 2014.
