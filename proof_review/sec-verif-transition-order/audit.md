# Audit: sec-verif-transition-order — order 1/2 of the stress-versus-Laplacian difference and the solution order at a level transition

- Source: `docs/techref/chapters/14_verification.tex:473-483` (closing paragraph after `tab:verif:orders`), branch `docs/techref-verification`, commit `77f46cb`
- Label: none (the paragraph carries no `\label`; slug `sec-verif-transition-order`; suggestion: add `\label{par:verif:transition-order}`)
- Date: 2026-11-10
- Overall status: ⚠️ partly verified — the number 0.5 and the mechanism (rows of order one on an area $O(h_{\min})$) are verified for the test field, but (E1) the order one in the rows is a property of that field (for a generic field the difference grows like $h^{-1/2}$ in $L^2$), and (E2) the order one of the solution does not follow from the order 1/2 of the difference
- Rerun: `julia --project=@claude-symbolics proof_review/sec-verif-transition-order/scripts/run_all.jl` (the checks run in child processes under `docs/techref/figures`, which resolves TwoPhaseAMR to the worktree `src/`)

## Statement and assumptions

Text (`14_verification.tex:473-483`): *"…the difference between the stress with uniform viscosity and the scalar Laplacian converges with order 0.5 on composite meshes, with and without the wall: the $L^2$ order of a difference of order one in the rows along a level transition, whose cells cover an area of order $h_{\min}$. The solution converges with order one there; the shifted solve pins 0.8 (#112)."*

Reading the claims:

1. (measured) $\|\mathsf D\|_{L^2}\sim h^{1/2}$ for $\mathsf D=\mathcal S_{\mathrm{stress},\mu=1}u-\mathcal L_{\mathrm{scalar}}u$ on composite meshes (test `uniform-viscosity stress versus scalar diffusion (#106)`).
2. (explanation) $\mathsf D=O(1)$ in the rows along the transition, whose dual volumes cover an area $A=O(h_{\min})$, so $\|\mathsf D\|_{L^2}\approx\sqrt{A}\,\|\mathsf D\|_\infty=O(h^{1/2})$.
3. (measured) the solution of the shifted stress solve converges with order one (#112), pinned at 0.8.

Conditions that the explanation needs and that the text does not state:
(a) the transition is a curve of fixed length (fixed or interface-following tree), so $A\propto h_{\min}$;
(b) the norm is the dual-volume-weighted $L^2$ norm (area weighted), not the maximum norm (order 0);
(c) the rows off the transition contribute $o(h^{1/2})$ (the bulk rows are second order, the corner rows at walls cover $O(h^2)$);
(d) the *field* has a vanishing normal velocity on the transition (E1) — this is where the explanation as stated is incomplete.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 $L^2$ order of $\mathsf D$ is 1/2 (no wall) | `:473-476` | ✅ | package run, $n=16,\dots,128$: orders $0.577, 0.517, 0.507$ (the test prints $0.48$–$0.50$ for $n=16,32,64$ in its own fixture) | `code_01_difference_scaling.jl` | $L^2=0.0527, 0.0353, 0.0247, 0.0174$ |
| S2 with the embedded wall | `:473-476` | ✅ | same, test fixture (solid $y\le1$): $0.486, 0.497, 0.499$ | `code_01_difference_scaling.jl` (T8) | |
| S3 rows of order one | `:476-478` | ✅ for the test field | $\max\lvert\mathsf D\rvert=0.3589, 0.3623, 0.3631, 0.3633$ (orders $-0.014,-0.003,-0.001$), all rows $\ge0.1\max$ lie within $2h_c$ of a transition ($n\ge32$) | `code_01_difference_scaling.jl` (T2) | the order one comes from the stress rows: $\max\lvert\tau_{\rm stress}\rvert=0.355,0.361,0.363,0.363$, scalar band rows $O(h)$: $0.065,0.033,0.016,0.008$ (T6) |
| S4 area of the rows of order one is $O(h)$ | `:477-478` | ✅ | rows with $\lvert\mathsf D\rvert\ge0.1\max$: $148,176,378,746$ faces, area $15.1,3.95,2.49,1.23$; last order $1.013$ | `code_01_difference_scaling.jl` (T3) | pre-asymptotic for $n\le32$ |
| S5 $L^2$ over strip $\sim h^{1/2}$, rest decays faster | `:476-478` | ✅ | strip order $0.507$; rest orders $2.24, 2.11, 1.52$ | `code_01_difference_scaling.jl` (T4) | the rest consists of wall-corner rows ($\max0.011$, area $O(h^2)$) |
| S6 the order is a property of the field | not stated | ❌ (omission) | same difference for $\sin(Kx+\varphi)$: $\varphi=0.7$: $L^2$ orders $-0.49$, max orders $-0.9$; $\varphi=\pi/2$: $-0.50$, $-1.00$ | `code_01_difference_scaling.jl` (T7, T9) | see E1 |
| S7 solution of the shifted solve converges with order one | `:478-479` | ✅ (measured) | #112 problem, $n=6,\dots,48$: $L^2$ orders $0.985, 1.002, 1.013$ (chapter 13 quotes $0.985$, $1.002$); max-norm orders $0.59, 0.74, 0.86$ | `code_02_truncation_vs_solution.jl` | pinned $\ge0.8$ in $L^2$ |
| S8 implication $\mathsf D=O(1)\Rightarrow$ solution order | implicit in the juxtaposition | ❌ not an implication | stability gives only $\ge1/2$; observed orders $1$ (#112) and $2$ (#87, with truncation of order one) | `code_02_truncation_vs_solution.jl` (C3, C4) | see E2 |
| S9 chapter 9 transition rows (truncation $O(1)$, solution $\ge1.8$, #87) | `09_viscosity.tex:242-249` | ✅ | scalar cut-cell Helmholtz without band, $n=16,\dots,128$: $\max\lvert\tau\rvert$ on the strip $0.54\to0.41$ (no slip, axis 1), $5.7$ (axis 2), $1.65$/$13.2$ (free slip); strip $L^2$ orders $0.35$–$0.52$; solution max orders $1.88$–$1.99$, $L^2$ orders $1.98$–$2.02$ | `code_02_truncation_vs_solution.jl` | monopole ratio $\lvert\sum V\tau\rvert/\sum V\lvert\tau\rvert$ over the strip $0.255,0.069,0.018,0.0045$ ($\propto h^2$): the rows cancel |

## Errors and consequences

**E1 — the "difference of order one" is specific to the test field (omission with consequences for the explanation).**
The field of `operator_difference` is $\boldsymbol u=\nabla^\perp[\sin(Kx)\sin^2(ay)]$; its velocity component normal to the two transitions ($x=4$ and $x=0\equiv8$) vanishes, $\sin(Kx)=0$. With the phase-shifted field $\sin(Kx+\varphi)$ the same two operators differ by terms that grow under refinement: $L^2$ order $-\tfrac12$, maximum order $-1$ (T7; $\varphi=\pi/2$: $\max\lvert\mathsf D\rvert=1.66,3.31,6.60,13.2$ for $n=16,\dots,128$). The same holds for the truncation of the stress operator alone on a fully periodic two-level mesh with the fields $(\sin2\pi y,0)$ or $(0,\sin2\pi x)$ (T9): $\max\lvert\tau\rvert=264,534,1071,2144$, i.e. $\propto h^{-1}$ (through `CellField` and through `InterfaceField` viscosity, which build the same `VariableStaggeredDiffusionOperator`), against $O(h)$ for the scalar band rows. In the #112 fixture itself (`code_02`, case A) $\max\lvert\tau\rvert=79,385,941,1978$ for $n=6,12,24,48$, alternating in sign along the transition (monopole ratio $\le0.5\,\%$). The stress rows on the transition are therefore inconsistent with $\Delta u$ by $O(h^{-1})$: the *flux* through the hanging dual faces is off by a quantity of order one on an area $O(h)$, not "a difference of order one in the rows". The order $\tfrac12$ of the chapter is the $L^2$ order of a quantity that is of order one only for fields with vanishing normal velocity on the transition.

**E2 — the solution order is not determined by the order of the difference.**
Stability of the shifted problem ($M=\mathsf V+\sigma\mathsf S$, $\mathsf S$ symmetric positive semidefinite, $M e=\sigma\mathsf V\tau$, checked numerically in C4) gives $\lVert e\rVert_{\mathsf V}\le\sigma\lVert\tau\rVert_{\mathsf V}$: a truncation of $L^2$ order $\tfrac12$ implies a solution order of *at least* $\tfrac12$, nothing more. The two cases at hand span the range: (B) truncation $O(1)$ on the strip, $L^2$ order $\tfrac12$ — solution order $2$ (the rows cancel, monopole ratio $\propto h^2$: supraconvergence); (A, #112) truncation $O(h^{-1})$ — solution order $1$. The order one of #112 is explained by the flux-form (energy) argument, not by the truncation: a flux inconsistency $q=O(1)$ on the strip of area $O(h)$ gives $\lvert a_h(u_I,v)-a(u,v)\rvert\le\lVert q\rVert_{L^2(\text{strip})}\lVert\nabla v\rVert_{L^2(\text{strip})}=O(h^{1/2})\,O(h^{1/2})=O(h)$ (Strang), the first-order global convergence that the test comment states. [INFERENCE: this is an upper bound; the numerics ($1.0$ in $L^2$, $0.6$–$0.9$ in the maximum norm) agree but do not prove sharpness.]

### Proposed replacement (lines 473–479)

```latex
With a level transition the two-phase embedded-wall rates stay within
$\log_2 1.1$ of the open-domain rates. The difference between the stress
with uniform viscosity and the scalar Laplacian, applied to the
divergence-free field of the test, converges with order $0.5$ on composite
meshes, with and without the wall. That field has a vanishing normal velocity on
the transitions; there the stress rows differ from the scalar rows (whose
band is $O(\minwidth)$) by terms of order one on cells that cover an area
of order $\minwidth$, whence the $L^2$ order $\tfrac12$ (the maximum norm does
not converge). The order is a property of the field, not of the operator: for
a field whose normal velocity does not vanish on the transition the stress rows
differ from the Laplacian by $O(\minwidth^{-1})$ with alternating signs along
the transition ($L^2$ order $-\tfrac12$ in the same test), so the flux through
the transition faces is inconsistent by a term of order one on an area
$O(\minwidth)$. The solution error of the shifted solve is first order
(\issue{112}: $0.985$ and $1.002$ in $L^2$; the test pins $0.8$); no order of
the solution follows from the order of the difference, since rows of
truncation order one at a transition can give a solution of order two
(\cref{eq:viscosity:cut}).
```

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| $L^2$ order of $\mathsf D$ is $\tfrac12$ (test field) | likely sharp (numerical) | orders $0.507$ (no wall), $0.499$ (wall); the limit $\tfrac12$ follows from $\sqrt{A}\max$ with $A\propto h$ (area order $1.013$) | `code_01_difference_scaling.jl` | none |
| "rows of order one" | not sharp / wrong for generic fields | $O(h^{-1})$ with a phase shift; order one only if $u_n=0$ on the transition | `code_01_difference_scaling.jl` (T7, T9) | replace by the statement of E1 |
| solution order one (#112) | likely sharp (numerical) | orders $0.985, 1.002, 1.013$; the maximum norm lags ($0.59\to0.86$) | `code_02_truncation_vs_solution.jl` | an $L^2$ proof would need the Strang argument with the actual flux error (open point) |
| solution order from the truncation order | not an implication | stability gives $\ge\tfrac12$; observed $1$ (A) and $2$ (B) | `code_02_truncation_vs_solution.jl` | state as measurement |
| pinned bound $0.8$ (#112) | not sharp | measured $\ge0.985$ on all three pairs; the bound leaves room for the pre-asymptotic $n=6$ pair | – | none needed |

## Open points

- Root cause of the $O(h^{-1})$ stress truncation at hanging faces for fields with nonzero normal velocity on the transition (normal-strain rows of the cells or the vertex-shear weights at hanging vertices; `src/two_phase/` variable-viscosity operator, `StaggeredDiffusionOperator(mesh; viscosity=CellField)`). It is more than "first order": the operator does not converge in $L^2$ or in the maximum norm there, while the solution does. For the caller: compare with the text of issue #112 before the chapter or the issue is changed.
- Chapter 13 (`tab:stab:orders`, "variable stress at transitions first order") and chapter 9 refer to the solution and stay valid.
- A proof (not a numerical check) of the solution order one from the flux inconsistency needs the energy-norm Strang argument above plus an Aubin–Nitsche step; not attempted.

## History
- 2026-11-10: first audit (branch `docs/techref-verification`, commit `77f46cb`): S1–S9 run; E1, E2 found; replacement text proposed (not applied: this audit does not edit the source file).
- 2026-11-10 (end of audit): Final `run_all.jl`: all checks passed (T1-T9: 7+2+6+2+8 assertions, C1-C4: 18 assertions; Runic 1.10.0 formatted).
