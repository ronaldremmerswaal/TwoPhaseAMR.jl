# Matrix-free geometric preconditioning for two-phase flow

## Scope and conclusion

This review concerns the two linear systems solved during every two-phase
Navier--Stokes step:

1. the cell-centred, variable-density pressure projection; and
2. the coupled, face-staggered, variable-viscosity implicit diffusion step.

The recommendation is to retain geometric multigrid and the existing Krylov
separation, but replace assembled fine and intermediate matrices by
factorized operator kernels and compact geometric transfer schedules. Only the
bottom problem should be assembled and factorized. The pressure hierarchy
needs coefficient-dependent interpolation or an equivalent local Galerkin
construction; coefficient-independent rediscretization is not a sufficiently
robust final design for density ratios of order `10^3`. The diffusion hierarchy
should apply the complete vector strain operator rather than precondition its
two principal component blocks independently.

This conclusion is specific to the operators in this package. The pressure
operator contains a non-diagonal face Riesz map at AMR transitions, and the
diffusion operator contains normal/shear coupling. A standard scalar BoxMG
formula or two independent scalar Helmholtz cycles cannot be copied unchanged.

The thesis fixes the physical fine-grid operators: inverse mixture density in
the projection and the fraction-weighted geometric mean of dynamic viscosity
at stress locations. It does not prescribe a production multilevel solver.
The literature below is therefore used for the preconditioner while the thesis
operator remains the equation being solved.

### Implemented first stage

The first implementation stage now provides:

- allocation-free, factorized fine-grid actions for the exact pressure and
  variable-stress operators;
- in-place pressure-coefficient updates with a topology-dependent sparse-entry
  map;
- persistent coefficient-dependent pressure transfers;
- a full-vector diffusion hierarchy whose Galerkin levels retain the
  normal/shear cross-component terms; and
- preconditioner-only aggregation of incomplete staggered sibling groups once
  exact geometric coarsening stalls at AMR interfaces.

The factorized kernels match the assembled matrices to roundoff on uniform,
fixed mixed-level, and dynamically adapted meshes. Their hot applications and
multigrid V-cycles allocate zero bytes.

Intermediate Galerkin matrices are still assembled. A trial compact refresh
schedule stored one target index per scalar-product contribution. On the
initial Stokes-3 topology that meant 54,086,426 pressure contributions and
17,063,458 diffusion contributions. It retained about 1.1 GiB and made the
short-run topology stage roughly three times slower. That representation was
removed. The current hybrid keeps the exact fine actions and persistent
transfers, but delegates intermediate `P' * A * P` products to the optimized
sparse kernels. The measured outcome and the rejected design are recorded in
the [Stokes-3 performance report](../../benchmark/stokes3-performance-results.md#matrix-free-follow-up-short-run-diagnosis).

This result completes the exact fine-action and full-vector hierarchy parts of
the plan. It also establishes that an all-level matrix-free implementation
needs a genuinely local coarse operator, as proposed below; expanding generic
sparse products into scalar Julia schedules is not a viable substitute.

## Why this is the next performance target

The exact-mesh Stokes-3 profile in the
[performance report](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/two_phase_flow_extension/benchmark/stokes3-performance-results.md) assigns
the following cost through `t=1` to coefficient refresh:

| operation | time | allocation |
|---|---:|---:|
| diffusion matrix and preconditioner refresh | 20.449 s | 53.849 GiB |
| pressure preconditioner refresh | 6.272 s | 14.205 GiB |

Together these refreshes consume 38.3% of the measured post-optimization time
and allocate 68.054 GiB. The Krylov solves themselves take 8.785 s. The
remaining allocation comes primarily from sparse stiffness construction,
principal-block extraction, smoothed transfer construction, and repeated
Galerkin products. A matrix-free hierarchy attacks this numerical refresh cost
without changing the AMR decisions or reducing the refresh frequency.

Here, *matrix-free* means that a level applies its operator from local geometry,
coefficient arrays, and fixed schedules. It may store a diagonal, a small set of
face or stress weights, transfers, and work vectors. Wang, Sun, and Zhu use the
same terminology for an adaptive octree hierarchy that stores one diagonal and
three face coefficients per three-dimensional cell while never assembling a
generic sparse matrix.[^wang2026] Recomputing every geometric quantity on each
application would save little memory and waste time; topology and numerical
coefficients should have separate lifetimes.

## The two discrete systems

### Variable-density pressure

Let `C` be the cell-to-face incidence, `A_f` the diagonal matrix of face areas,
and `K` the inverse face Riesz map used by the compatible pressure gradient.
With face coefficient `kappa = 1/rho_f`, the integrated positive pressure
stiffness is

```math
S_p = C^T A_f \sqrt{\kappa}\,K\sqrt{\kappa}\,A_f C.
```

It is symmetric positive semidefinite and has the constant vector as its
nullspace for periodic or all-Neumann boundaries. `K` is diagonal in conforming
Cartesian regions and sparse but non-diagonal near AMR transitions. The
square-root placement of `kappa` is part of the fine operator: it keeps the
weighted gradient and divergence dual and is also used by balanced surface
tension and gravity.

The present preconditioner stores `S_p` and every Galerkin matrix
`P_l' * S_l * P_l`. An exact matrix-free fine application instead evaluates
the displayed factorization from right to left. This removes the
coefficient-dependent stiffness assembly while preserving the exact pressure
operator.

### Coupled variable-viscosity diffusion

Let `E` map face velocities to the normal and shear components of the strain,
and let `W_mu` contain stress-control-volume measures multiplied by dynamic
viscosity. The midpoint system is

```math
A_u = M_\rho + \frac{\Delta t}{2} E^T W_\mu E,
```

where `M_rho` is diagonal on face dual volumes. The geometric phase mixture
used by the code and thesis is

```math
\mu(c)=\mu_l^c\mu_g^{1-c},
\qquad
\log\mu(c)=c\log\mu_l+(1-c)\log\mu_g.
```

The exact matrix-free application is three streaming operations:

```text
strain = E * velocity
stress = W_mu .* strain
result = M_rho .* velocity + (dt/2) * E' * stress
```

The diagonal required by Jacobi or Chebyshev smoothing is also local:

```math
(A_u)_{ii}=(M_\rho)_{ii}
 +\frac{\Delta t}{2}\sum_s (W_\mu)_{ss}E_{si}^2.
```

The current geometric preconditioner extracts the `u-u` and `v-v` principal
blocks and builds two scalar hierarchies. The Krylov operator remains coupled,
but the preconditioner omits the shear-induced cross-component block.

## Evidence from the literature

### Discontinuous-coefficient pressure equations

Almgren et al. solve the variable-density projection on block-structured AMR
with intermediate geometric levels, conservative full-weighting restriction,
red-black Gauss--Seidel smoothing, and operator-dependent interpolation. They
coarsen conductivity arithmetically transverse to a flux and harmonically
along it, then rediscretize the same operator form.[^almgren1998] This is a
direct precedent for separating AMR geometry from per-level directional
conductance. The paper also states that more elaborate coarsening may be
needed for very large density discontinuities.

Dendy's BoxMG and de Zeeuw's matrix-dependent transfers construct
interpolation from the operator, rather than assuming that geometrically
linear interpolation represents algebraically smooth error.[^dendy1982]
[^dezeeuw1990] This distinction matters at a material jump: flux is continuous,
while the pressure gradient is not.

MacLachlan, Tang, and Vuik provide the closest direct comparison for two-phase
pressure correction. On uniform staggered grids, coefficient-independent GMG
deteriorates with density contrast and bubble count. BoxMG, which combines
operator-induced interpolation with variational coarsening, remains at 12 CG
iterations as the phase coefficient ratio changes from `10^-1` to `10^-5` in
their reported test, and is their most robust method in evolving bubbly-flow
cases.[^maclachlan2008] These numbers should not be transferred as performance
predictions for this package, but they establish that a purely geometric
coarse model is a material risk at the target density ratios.
Their GMG baseline uses a homogeneous coarse model; the current code already
uses coefficient-dependent smoothed transfer and Galerkin matrices. The result
therefore supports preserving that coefficient dependence in compact form,
rather than replacing it by homogeneous geometric interpolation.

Popinet likewise notes that standard multigrid can slow substantially for the
discontinuous coefficient and source terms of large-density-ratio flow, and
that performance depends on representing interface topology consistently on
coarser grids.[^popinet2009]

For adaptive octrees, Wang, Sun, and Zhu store compact local operator
coefficients, use Galerkin-consistent coarsening in uniform regions, and add a
flux-consistent FAS-style correction at T-junctions.[^wang2026] Their PCG
iteration counts are nearly grid independent. This 2026 work is a preprint,
and its compact seven-point MAC operator is simpler than the present Riesz-map
operator. Its treatment of topology and coefficient storage is a useful design
pattern, not an implementation that can be copied directly.

Fortunato, Rycroft, and Saye show for LDG discretizations that separately
coarsening a factorized gradient and divergence can give much better
multigrid behavior than applying a Galerkin product to the already composed
primal operator, including on adaptive Cartesian and multiphase interface
problems.[^fortunato2019] Their discretization differs from this finite-volume
method. The relevant inference is that preserving the `div--metric--grad`
factorization on coarse levels deserves a direct experiment here.

### Coupled variable-viscosity diffusion

Cai et al. study the same two algebraic subproblems on a uniform staggered
finite-volume grid: `div(rho^-1 grad)` for pressure and a generalized vector
Helmholtz problem containing `div(2 mu D(u))` for velocity. One standard GMG
cycle is used as the inexact subsolve. The velocity multigrid retains the full
stress coupling and uses a `2d`-colour generalization of red-black
Gauss--Seidel; it does not split the vector operator into scalar component
cycles.[^cai2014] They also give staggered restriction stencils and separate
coarsening rules for face inertia, cell-centred normal-stress viscosity, and
node/edge shear-stress viscosity. The limitation is that their grid hierarchy
is uniform.

Clevenger and Heister compare assembled AMG with matrix-free GMG for an
adaptively refined, variable-viscosity Stokes problem.[^clevenger2021] Their
GMG velocity level applies the fully coupled strain-rate operator and uses a
degree-four Chebyshev smoother. Matrix-free evaluation makes the off-diagonal
strain coupling require no additional matrix storage. In their 18.5-million
degree-of-freedom case, matrix-free GMG cuts the reported assembly stage by a
factor of 11.2 and total time by a factor of 3. Their iteration table also
shows deterioration as both viscosity contrast and the number of inclusions
increase, so full coupling and matrix-free evaluation do not by themselves
guarantee contrast-independent convergence.

The adaptive matrix-free framework used in that work supports local smoothing
on refined regions and matrix-free transfer between nonuniform levels.[^clevenger2019]
It is finite-element based, but it establishes that hanging-node constraints,
level transfers, and matrix-free application can be separated cleanly.

If a diagonal polynomial smoother is insufficient, box or patch relaxation is
the established stronger alternative. Borzacchiello et al. apply coupled box
relaxation to staggered finite-volume variable-viscosity Stokes systems and
report robustness up to viscosity contrast `10^7`.[^borzacchiello2017] That
paper smooths a velocity-pressure saddle-point system on structured grids, so
its exact local solve is not the first choice for this shifted SPD velocity
block. It supports a small coupled stress patch as a fallback after the cheaper
Chebyshev design is measured.

## Pressure hierarchy recommended for this code

### Exact fine-level kernel

The Krylov operator must continue to apply the current fine-grid factorization
`C' A_f sqrt(kappa) K sqrt(kappa) A_f C`. The non-diagonal part of `K` is
topological and can be retained as a compact patch schedule or as the existing
sparse Riesz map initially. Neither choice requires construction of `S_p`.
This first step can be checked against the current sparse matrix to roundoff
before any preconditioner changes.

### Coarse levels

Two variants should be implemented and compared on the same hierarchy:

1. **Factorized rediscretization.** Build `C_l` and the compatible `K_l` from
   level geometry once after adaptation. Refresh directional coarse
   conductances in linear time using series/parallel aggregation, following
   Almgren et al. Apply the same factorized form on every level.
2. **Compact local Galerkin/BoxMG coarsening.** Accumulate local coarse
   connection weights directly from fine connections and compact transfer
   schedules. In regular regions this is a local `R A P` calculation. Near
   T-junctions it must preserve the current Riesz-map flux, potentially with a
   flux-consistent correction. No global CSC product is formed.

The first variant is the simplest matrix-free baseline. The second is the
robust candidate for density jumps and interfaces that disappear on a coarse
level. A standard five-point rediscretization is not acceptable at an AMR
transition because it would discard `K`. Conversely, recursively evaluating
`A_l x = P_l' A_{l+1}(P_l x)` avoids coarse matrices but revisits fine levels
during every coarse smoothing step, gives roughly `O(N log N)` work, and makes
the smoother diagonal expensive. It should not be the production design.

### Transfer and smoothing

Pressure prolongation should remain compact and reproduce constants exactly,
but its weights should depend on local conductances near a density jump. A
BoxMG/de Zeeuw-style flux interpolation is a stronger starting point than the
current coefficient-dependent sparse Jacobi product because its weights can be
computed directly into a fixed schedule.

A fixed-degree Chebyshev--Jacobi smoother is the first matrix-free candidate:
it needs only operator applications, a cached diagonal, and a spectral upper
bound. It is parallel and can be applied in symmetric pre/post order. The
current damped Jacobi sweep should remain as a baseline. If either loses
robustness with thin or disconnected phase regions, graph-coloured
Gauss--Seidel or a small coefficient-aware patch smoother is the next step;
the BoxMG literature supports that escalation.

Restriction should be the volume-scaled adjoint of prolongation, and the
pre/post smoothers must be paired symmetrically if the cycle preconditions CG.
The constant null mode must be removed with the level volume weights after
restriction and smoothing. The existing augmented bottom solve remains valid.
Any FAS-style T-junction correction must be shown to define a fixed linear,
symmetric positive preconditioner before it is used with CG; the recent octree
paper's overshoot parameters should not be copied without that check.

## Diffusion hierarchy recommended for this code

### Exact factorized kernel

Retain `E` and the stress interpolation schedule as topology data and stop
forming `E' W_mu E`. On coefficient refresh, update only:

- cell `log(mu)` from the current volume fraction;
- stress `W_mu` through the existing interpolation;
- face inertia `M_rho`; and
- the local diagonal.

The outer operator then uses two applications of `E` and pointwise products.
This preserves the exact dynamic-viscosity placement from the thesis. A later
optimization can replace sparse `E` by structure-of-arrays stress schedules;
that change is independent of the multigrid mathematics.

### Full-vector coarse operators

Each level should contain all staggered velocity components and apply

```math
A_{u,l}=M_{\rho,l}+\frac{\Delta t}{2}E_l^T W_{\mu,l}E_l.
```

This retains normal/shear coupling and is SPD by construction. `E_l` and
staggered transfers are built only when topology changes. Density is restricted
conservatively to face dual volumes. For viscosity, restricting volume fraction
or `log(mu)` and then applying the geometric phase mixture are algebraically
equivalent because `log(mu)` is affine in the fraction. This is the most
faithful first coarse coefficient. A directional energy/homogenized viscosity
can be compared later if high-contrast convergence requires it; it is a
preconditioner coefficient and must not silently replace the thesis mixture in
the fine equation.

The staggered transfer needs separate normal and tangential rules and a
volume-weighted adjoint restriction. Cai et al.'s six-point two-dimensional
restriction is a useful uniform-grid reference. On this mesh, the transfer
must additionally commute with the face ownership and hanging-face constraints
used by `E_l`.

### Smoother choice

Start with a degree-two to degree-four Chebyshev--Jacobi smoother on the full
vector operator, using the exact local diagonal above. It has three advantages:

1. no sparse row traversal or graph-colour synchronization;
2. normal/shear coupling is included in every operator application; and
3. a fixed number of symmetric sweeps gives a fixed linear preconditioner.

The hard limit is the diffusion-dominated regime, where `M_rho` becomes small
relative to `E'WE`, combined with large viscosity contrast. If the refinement
study shows growing iterations there, compare, in this order:

1. component-coloured coupled Gauss--Seidel as in Cai et al.;
2. small cell/vertex stress-patch block Jacobi or Schwarz; and
3. interface-preserving aggregation for stress patches cut by the phase jump.

The present component-split hierarchy should remain in the benchmark as an
ablation. It should not define the target architecture because it throws away
the coupling that matrix-free application makes inexpensive.

## Shared implementation architecture

The hierarchy should separate three kinds of state:

| lifetime | pressure | diffusion |
|---|---|---|
| topology change | cells/faces, `C_l`, `K_l` schedule, transfers, colours | staggered faces, `E_l`, stress interpolation, transfers, patches |
| coefficient change | conductance, diagonal, spectral bound | inertia, stress viscosity, diagonal, spectral bound |
| every cycle | solution, residual, correction, polynomial work | vector solution, residual, correction, strain/stress work |

All arrays should grow in place and be reused. Coefficient refresh should be a
sequence of allocation-free `O(sum_l N_l)` loops. A small bottom operator
should still be assembled and factorized; "matrix-free everywhere" is less
useful than removing matrices from the levels where they dominate work and
memory. This hybrid bottom solve keeps the pressure nullspace treatment exact.

The coefficient hierarchy must refresh every time the VOF field changes. The
research does not support freezing a high-density-ratio hierarchy merely to
avoid setup. Compact direct coefficient updates are the way to make the same
mesh and coefficient evolution cheap.

## Validation and benchmarking plan

The first tests compare kernels, independently of convergence:

- random-vector equivalence with the current assembled pressure and diffusion
  matrices on uniform, fixed mixed-level, and AMR meshes;
- symmetry bilinear identities and non-negative energy;
- exact pressure constant nullity and volume-weighted compatibility;
- exact agreement of the matrix-free weighted gradient used by projection,
  surface tension, and gravity; and
- zero steady diffusion of constant admissible velocities.

The multigrid study then varies refinement and physical contrast rather than
checking errors against ad hoc constants:

| system | axes to vary | required observation |
|---|---|---|
| pressure | cells, AMR levels, density ratio `1` to at least `10^6`, flat/rotated/curved and disconnected interfaces | convergence rate, CG iterations, true residual, nullspace |
| diffusion | cells, AMR levels, viscosity ratio, density ratio, `dt*mu/(rho*h^2)` from inertia- to diffusion-dominated | convergence rate, Krylov iterations, energy dissipation |

For each case record separately:

- topology setup time and allocation;
- coefficient refresh time and allocation;
- one operator application and one V-cycle;
- Krylov iterations and solve time; and
- retained hierarchy and work-vector memory.

Finally, run the existing Stokes-3 benchmark through `t=1` with exactly the
same AMR decisions. The trace comparison must verify block keys, cell/face
counts, refine/coarsen decisions, physical time grid, and pressure/diffusion
residuals. The primary success criterion is removal of sparse refresh
allocation while preserving refinement and contrast trends; a single small
mesh timing is insufficient.

## Proposed implementation order

1. Add exact factorized fine-grid `mul!` kernels and equivalence benchmarks,
   while retaining the current preconditioners.
2. Introduce persistent level geometry, compact transfers, diagonals, and
   workspaces. Assemble only the bottom level.
3. Implement the pressure factorized-rediscretization baseline, then the
   coefficient-dependent local Galerkin/BoxMG variant.
4. Implement the full-vector diffusion hierarchy with Chebyshev--Jacobi
   smoothing; retain the current component split for comparison.
5. Add coloured or stress-patch relaxation only if the contrast/refinement
   study shows that polynomial smoothing is the bottleneck.
6. Re-run the exact-mesh Stokes-3 before/after benchmark and add the new setup,
   refresh, V-cycle, and operator-application rows to its report.

This order isolates operator correctness from preconditioner quality and
allows the first two steps to reduce diffusion stiffness assembly even if a
new coarse hierarchy needs further tuning.

## Sources

[^almgren1998]: A. S. Almgren, J. B. Bell, P. Colella, L. H. Howell, and M. L. Welcome, [*A Conservative Adaptive Projection Method for the Variable Density Incompressible Navier--Stokes Equations*](https://escholarship.org/uc/item/0xg2k57t), Journal of Computational Physics 142 (1998), 1--46. [DOI](https://doi.org/10.1006/jcph.1998.5890).

[^dendy1982]: J. E. Dendy Jr., [*Black Box Multigrid*](https://doi.org/10.1016/0021-9991(82)90057-2), Journal of Computational Physics 48 (1982), 366--386.

[^dezeeuw1990]: P. M. de Zeeuw, [*Matrix-dependent prolongations and restrictions in a blackbox multigrid solver*](https://ir.cwi.nl/pub/13925/), Journal of Computational and Applied Mathematics 33 (1990), 1--27.

[^maclachlan2008]: S. P. MacLachlan, J. M. Tang, and C. Vuik, [*Fast and robust solvers for pressure-correction in bubbly flow problems*](https://www.math.mun.ca/~smaclachlan/research/bubbles.pdf), Journal of Computational Physics 227 (2008), 9742--9761. [DOI](https://doi.org/10.1016/j.jcp.2008.07.022).

[^popinet2009]: S. Popinet, [*An accurate adaptive solver for surface-tension-driven interfacial flows*](https://doi.org/10.1016/j.jcp.2009.04.042), Journal of Computational Physics 228 (2009), 5838--5866.

[^wang2026]: M. Wang, Y. Sun, and B. Zhu, [*Matrix-Free Multigrid with Algebraically Consistent Coarsening on Adaptive Octrees*](https://arxiv.org/abs/2604.18886), arXiv:2604.18886 (2026 preprint).

[^fortunato2019]: D. Fortunato, C. H. Rycroft, and R. I. Saye, [*Efficient Operator-Coarsening Multigrid Schemes for Local Discontinuous Galerkin Methods*](https://arxiv.org/abs/1808.05320), SIAM Journal on Scientific Computing 41 (2019), A3913--A3937. [DOI](https://doi.org/10.1137/18M1206357).

[^cai2014]: M. Cai, A. Nonaka, J. B. Bell, B. E. Griffith, and A. Donev, [*Efficient Variable-Coefficient Finite-Volume Stokes Solvers*](https://arxiv.org/abs/1308.4605), Communications in Computational Physics 16 (2014), 1263--1297. [DOI](https://doi.org/10.4208/cicp.070114.170614a).

[^clevenger2021]: T. C. Clevenger and T. Heister, [*Comparison Between Algebraic and Matrix-free Geometric Multigrid for a Stokes Problem on Adaptive Meshes with Variable Viscosity*](https://arxiv.org/abs/1907.06696), Numerical Linear Algebra with Applications 28 (2021), e2375. [DOI](https://doi.org/10.1002/nla.2375).

[^clevenger2019]: T. C. Clevenger, T. Heister, G. Kanschat, and M. Kronbichler, [*A Flexible, Parallel, Adaptive Geometric Multigrid Method for FEM*](https://arxiv.org/abs/1904.03317), ACM Transactions on Mathematical Software 47 (2021), Article 7. [DOI](https://doi.org/10.1145/3425193).

[^borzacchiello2017]: D. Borzacchiello, E. Leriche, B. Blottiere, and J. Guillet, [*Box-relaxation based multigrid solvers for the variable viscosity Stokes problem*](https://doi.org/10.1016/j.compfluid.2017.08.027), Computers & Fluids 156 (2017), 120--138.
