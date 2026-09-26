# Choosing normal and tangential jump operators together

This is a frozen-geometry operator experiment, not a new projection method.
Run [`co_design_nt.jl`](co_design_nt.jl) from the repository root with
`julia --project=. development/infer/co_design_nt.jl`. It uses the production
active faces, pressure coupling and diagonal jump metric, but does not solve
a Poisson or KKT system. The geometry is a circle; its exact normal and
tangent are used to avoid confusing geometry error with operator error.
All reported stencils have radius 2.5 cells unless noted otherwise.

## The two independent relations

For face jumps `xi`, cell pressure `p` and normal-constraint multiplier
`lambda`, stationarity is

```text
K xi + C' p + N' lambda = 0.
```

`N` must interpolate the physical interface-normal jump; `T` must
interpolate the physical tangential jump. If `Z` maps a constant vector
`v=(v_x,v_y)` to its face-normal samples, constant consistency means
`N Z v=n·v` and `T Z v=tau·v` at every interface row. Bounded coefficients
on an `O(h)` patch then also give first-order interpolation of smooth
velocity jumps. These interpolation properties must be checked separately
from the stationarity relation.

Let `S p` be the desired surface derivative of `[p/rho]`. Substituting
stationarity gives two *different* requirements:

```text
-T K^-1 C'  ~= S                 (pressure block),
 T K^-1 N'  ~= 0                 (normal-multiplier block).
```

The first is independent of `N`: modifying `N` alone cannot fix its error.
The second is a whole-matrix relation. Making each matching pair of rows
orthogonal does not make adjacent, overlapping rows orthogonal.
The natural kinetic lifts, if an explicit reduced basis is wanted, are
`E_T=K^-1 T'` and `E_N=K^-1 N'`. Then
`E_T' K E_N=T K^-1 N'`. These lifts are *determined by* `T,N`; choosing
them independently risks losing the meaning of tangential and normal
interpolation. A symmetric KKT solve, including all cross terms, retains
its usual energy identity even if this block is only approximately zero.
Deleting the cross terms on that basis would not be justified.

There is a two-face obstruction. With one x face and one y face, the
consistent rows are `T=(-n_y,n_x)` and `N=(n_x,n_y)`, but

```text
T K^-1 N' = n_x n_y (1/K_y - 1/K_x).
```

This is generally nonzero for an oblique interface and unequal jump
masses. The current paired `:infer` construction instead makes two
kinetic-orthogonal *trial vectors*; its effective stationarity observation
has weights `(K_x n_y,-K_y n_x)`, not the physical tangential weights.
One cannot satisfy both physical interpolation and exact orthogonality
using only these two face samples. More faces, a changed metric, or an
approximate relation are needed.

## Options tested

The program constructs `N,T` on the same active face set and tests
smooth-vector interpolation, the two operator blocks, coefficient size,
coverage and the rank of their kinetic lifts.

| Choice | Normal-multiplier block | Main outcome |
| --- | --- | --- |
| Geometric local reconstruction | Nonzero, including cross-row terms | `N,T` interpolate smoothly, but the pressure block does not converge on these samples. |
| Adjust `N` against only its matching `T` row | Matching-row zero | Neighboring-row coupling remains large; the pressure block is unchanged. |
| Hold `T` fixed and adjust `N` against **all** overlapping `T` rows | Whole block zero | `N` still converges, but with much larger errors; the bad pressure block is unchanged. |
| Hold `N` fixed and adjust `T` against all overlapping `N` rows | Whole block zero | Both interpolate, but pressure error improves slowly and does not show reliable first-order behavior. |
| Give `T,N` globally disjoint checkerboard face sets | Whole block zero by construction | Both interpolate, but the pressure error plateaus and the kinetic Gram can be poorly conditioned. Requiring both phase apertures open loses most rows. |
| Fit pressure samples exactly, then find `N` | Matching-row zero only | Severe overfitting: coefficient amplification up to 142 at `n=64` for 1000:1 and 165 at `n=32` for 1:1000, with enormous held-out errors. |
| Alternating bounded local fits of `T,N` | Small, not exact | Both interpolate and pressure errors fall substantially. Strengthening the orthogonality penalty trades some pressure accuracy for a smaller block. |

The local constraints in the last option reproduce the two constant-vector
moments **exactly**. Each row is fit near the geometric reconstruction,
using local pressure moments, all overlapping rows of the other operator,
a ridge penalty and a cap on the correction. Since the orthogonality term
is bilinear in `T,N`, three alternating sweeps are used. The `soft1000`
variant below uses an orthogonality weight of 1000 and pressure weight of
1000; these are exploratory, row-normalized weights, not production
parameters. The cap limits each correction to the norm of its geometric
baseline. Smooth velocity interpolation is then an `O(h)` consequence of
exact constant reproduction, bounded coefficients and bounded patch width;
the measured values check this on two nonconstant vector fields.

The first pressure tests use common affine and trigonometric fields in the
two phases. Two additional pressure modes are signed distance in only one
phase, zero in the other. Their interface trace is zero, so their correct
tangential response is zero; these modes reveal leakage from a purely
normal phase-pressure variation. `affine_plus_normal` includes them in the
fit; the trigonometric pressure modes remain held out.

Here are selected results for `soft1000` with `affine_plus_normal`. `p_diag`
is the held-out diagonal trigonometric pressure error. `p_normal` is the
phase-specific normal-pressure leakage. `comm` is the largest normalized
entry of the **full** multiplier block, including cross-row entries. The
L2 errors average over interface rows. All rows were covered.

| Density ratio | `n` | `T` L2 | `N` L2 | `p_diag` L2 | `p_normal` L2 | `comm` | Max coefficient amplification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000:1 | 64 | .0337 | .00986 | .173 | .000532 | .000181 | 1.18 |
| 1000:1 | 128 | .0140 | .00427 | .117 | .00104 | .000271 | 1.27 |
| 1000:1 | 256 | .00701 | .00236 | .0567 | .000787 | .00206 | 1.40 |
| 1:1000 | 64 | .0287 | .0126 | .220 | .00323 | .000720 | 1.79 |
| 1:1000 | 128 | .0145 | .00526 | .0852 | .00109 | .000749 | 1.25 |
| 1:1000 | 256 | .00800 | .00256 | .0639 | .00157 | .00133 | 1.53 |

For comparison, geometric `T,N` at 1000:1, `n=128` have interpolation L2
errors `.00428,.00400`, but `p_diag` L2 is `1.57` and the full normalized
commutator is `.402`. Thus the joint fit sacrifices some interpolation
accuracy while substantially improving both algebraic relations. Its
`p_diag` and interpolation L2 errors decrease under this refinement, but
the `p_diag` Linf errors do **not** exhibit a clean first-order rate in
both density directions. Nor does the maximum commutator yet decrease
monotonically; the data support a small finite-resolution defect, not a
proof of asymptotic `O(h)` commutation. Increasing the orthogonality weight
from 1 to 1000 reduced the finite-resolution multiplier block by roughly
two orders of magnitude but increased pressure error. The weight should
ultimately be chosen by a mesh-scaled error budget, not a magic constant.
At equal densities, where the desired common-pressure tangential jump is
zero, the geometric reconstruction still gives `p_diag` L2 `.520` and
`.744` at `n=64,128`; the affine-plus-normal `soft1000` fit gives `.228`
and `.0787`. This isolates an error in the discrete pressure response,
not an error in the physical density-jump target.

Exact orthogonality by itself is not enough: at 1000:1, `n=128`, fixing
geometric `T` and adjusting `N` gives `p_diag` L2 `1.57`; fixing `N` and
adjusting `T` gives `.704`; a bounded exact-block pressure fit gives
`.351`, but its `p_diag` Linf *worsens* from `.889` to `1.13` between
`n=64` and `128`. Enforcing that both phase apertures be open leaves only
64 of 140 circle rows at `n=64` and 112 of 276 at `n=128`. It is not a
safe rule for choosing `E_T/E_N` faces.

The kinetic-lift rank matters too. At 1000:1, `n=64`, the 140 `T` rows
and 140 `N` rows of the tested full-block designs are individually
independent, and their combined rank is 280 in a space of 420 active face
jumps. Thus `E=[E_T E_N]` alone discards 140 directions. The combined
nonzero Gram condition number is about `1.1e3` for exact `N`-from-`T`,
`7.7e4` for checkerboard partition, and `1.0e4` for the bounded `soft1000`
fit with affine-plus-normal pressure moments. This is an additional representability
and conditioning gate for any reduced implementation. Keeping the full
active jump space while using `N` as the constraint and `T` as a diagnostic
avoids that particular reduction error.

## Provisional construction path

1. Keep the aperture-based divergence and the existing `K` unchanged.
   Start with physical local `N_0,T_0` on a shared patch; do not label
   individual faces exclusively normal or tangential.
2. Enforce constant-vector moments exactly for both operators. Fit `T`
   to the pressure block using common affine pressure and phase-specific
   zero-trace normal modes, as screened here. Regularize and cap
   corrections. General piecewise-affine phase pressure and a prescribed
   pressure-trace jump or surface-tension lifting need additional moments;
   the present experiment has not validated those cases.
3. Adjust `N` against all overlapping `T` rows, and alternate if necessary.
   Target a measured `O(h)` pressure defect and `O(h)` multiplier effect on
   smooth multipliers, while monitoring worst local defects, row coverage,
   coefficient bounds and Gram conditioning. A small same-row product is
   insufficient.
4. On an ill-conditioned or topologically ambiguous patch, widen the patch
   modestly or relax commutation **locally**. Preserve physical normal
   consistency and the symmetric KKT structure. Never silently replace a
   missing normal condition with an axis-aligned one.
5. Before a production switch, validate piecewise-affine manufactured
   pressure, surface-tension pressure jump, the Poisson scaled-gradient
   `L2/Linf` errors, momentum, the usual energy identity, KKT rank and
   conditioning, then KH mesh and time-step refinement at several density
   ratios and Weber/Bond cases. This circle screen cannot certify KH.

The promising option is therefore a **bounded local joint fit**, not an
exact facewise split. It is still a hypothesis: these fits use analytical
circle geometry and analytical pressure-moment targets. The corresponding
node-free pressure trace/target construction, moving-interface continuity,
sliver robustness, and full projection behavior remain unresolved. No
production `K` or projection operator was changed in this experiment.
