# Vortex-sheet commutativity development experiment

This directory contains a cheap, independent experiment for the two
commutativity relations in the two-velocity pressure projection. It is
deliberately not part of the test suite.

The default command validates only the three interpolants already in
production:

```sh
julia --project=. -O0 development/vortex_sheet_commutativity/commuting_normal_interpolant.jl
```

This path only applies the interface operators to prescribed fields. It
does not solve a pressure or multiplier system, form a Schur complement, or
construct the matrix product `T inv(K_xi) N'`. To additionally build and
check the experimental interpolant obtained from commutativity, run:

```sh
julia --project=. -O0 development/vortex_sheet_commutativity/commuting_normal_interpolant.jl --construct
```

To compare sparse stencil sizes and construction costs, run:

```sh
julia --project=. -O0 development/vortex_sheet_commutativity/commuting_normal_interpolant.jl --sparsity
```

The script reports its computation time after module loading. On a cold
Julia process, loading and compiling `TwoPhaseAMR` can dominate the visible
wall-clock time; that startup cost is separate from this experiment. The
commands above use optimisation level zero because these small stencil
applications are compilation-bound. On the reference machine this reduced a
cold existing-interpolant run from about 20 seconds to about 11 seconds
without changing any reported error. Do not use `--compile=min`: interpreting
the sparse assembly is substantially slower.

## Fixed data and the only model freedom

For every resolution, the script fixes:

- a periodic uniform Cartesian mesh;
- a circular interface;
- two constant phase densities;
- smooth one-sided phase pressure fields;
- the active staggered jump space and its diagonal kinetic coefficient
  `K_xi`;
- one local, constant-reproducing tangential interpolant `T`.

The two phase pressure functions are equal in the default experiment, while
their densities differ. This avoids a separate pressure-jump lifting and
still gives a nonzero `d_s[p/rho]`. The fixed pressure contribution is the
ordinary uniform-grid gradient of the cell samples of
`p_g/rho_g-p_l/rho_l`. The full jump stationarity equation is

```text
K_xi xi + C' p + N' lambda = 0.
```

Define its fixed pressure contribution by

```text
C' p = -K_xi xi_p.
```

This compatible manufactured pressure block is intentional: the experiment
isolates the remaining freedom in `N`. Substituting the production pressure
block would be a separate pressure-block commutativity experiment and could
not be repaired by changing `N`.

Solving the full three-term stationarity equation for `xi` gives

```text
xi[p, lambda] = xi_p - inv(K_xi) N' lambda.
```

The multiplier commutativity condition must hold for every `lambda`.
Therefore the cheap validation does not solve a multiplier Schur system. It
samples one fixed smooth multiplier on the circular interface and evaluates
`N' lambda`, the diagonal `inv(K_xi)`, and then `T` in sequence. Normal
consistency of `N` is likewise checked by forward evaluation on constant and
smooth vector fields. The condition `N[u]=0` motivates the multiplier, but
need not be solved in order to validate the operator identity.

Thus `N` is the only quantity changed between runs. The pressure-only error
`T xi_p - d_s[p/rho]` is printed separately and must converge independently
of `N`.

## Candidates and diagnostics

The default study compares only the production `:divergence`, `:face`, and
`:cell` matrices. The `--construct` study additionally compares them with:

- `:centred`: the normal component of the same fixed vector reconstruction
  used by `T`;
- `:commuting_r2`: a locally constructed adjoint with a two-cell patch.

The commuting construction forms `Q=N'` one column at a time. It makes the
smallest kinetic change from the centred baseline subject to

```text
T inv(K_xi) q_r = 0,
Z' q_r = n_r,
```

where the two columns of `Z` are constant Cartesian vector fields. The
second relation is the consistency moment needed for first-order accuracy.

Patch discovery uses cell and face adjacency tables rather than scanning the
whole interface for every row. With a fixed patch radius, both the number of
nonzeros and the construction work therefore scale linearly with the number
of interface degrees of freedom. Each subsequent `N*x` or `N'*lambda`
application costs one sparse traversal, namely O(`nnz(N)`).

Linearity splits tangential commutativity into a pressure block and a
multiplier block. For each resolution and candidate, the primary output
therefore reports three independent consistency errors, each in L2 and
Linf:

- normal-consistency L2 and Linf errors for a smooth staggered vector field;
- pressure-block errors in `T xi[p,0] - d_s[p/rho]`;
- multiplier-block errors in `T xi[0,lambda]`, corresponding to the strong
  choice `R_lambda=0`;
- their observed refinement rates.

Here L2 is the root-mean-square discrete interface norm, so errors at
different interface-row counts are directly comparable. The code retains
the combined-field, construction-residual, and stencil-size diagnostics in
each returned `CandidateDiagnostics`, but does not clutter the primary table
with them.

The first three production candidates are included specifically to determine
whether normal consistency and energy compatibility alone yield tangential
commutation. The experiment does not assume that they do.

## Trying another normal interpolant

A candidate builder is simply a function from `CommutativityProblem` to
`NormalCandidate`. Pass builders directly to `run_study`:

```julia
include("development/vortex_sheet_commutativity/commuting_normal_interpolant.jl")
using .VortexSheetCommutativity

builders = [
    problem -> VortexSheetCommutativity.commuting_normal(
        problem; patch_radius=3,
    ),
]

VortexSheetCommutativity.run_study(
    resolutions=(16, 32, 64),
    builders=builders,
)
```

Keep `T`, the manufactured pressures, and the problem construction fixed
when comparing candidates. This is what makes failure or convergence
attributable to `N` alone.
