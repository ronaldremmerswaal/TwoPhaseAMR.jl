# Audit: dev:amr:curvature — thresholds and the hysteresis equivalence $2\rho<1\iff2\varepsilon_c<\varepsilon_r$

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:148-200` (`eq:amr:curvature` l. 153; `dev:amr:curvature` l. 177-198), code `src/mesh/adaptivity/phase_criteria.jl:111-129` (constructor, `0 < 2coarsen_ratio < 1`), 273-327 (`refinement_labels!`), commit `2959760`
- Date: 2026-11-10
- Overall status: ✅ verified (the equivalence is exact algebra; one sharpness remark on the margin)
- Rerun: `julia --project=@claude-symbolics proof_review/dev-amr-curvature/scripts/run_all.jl`

## Statement and assumptions

Thesis: $\varepsilon_c^{\kappa}\le\max_c h_c|\kappa_c|\le\varepsilon_r^{\kappa}$ with $2\varepsilon_c^{\kappa}<\varepsilon_r^{\kappa}$ (usually $\varepsilon_r=3\varepsilon_c$). Code: $E_c=n\,\max(h)|\kappa_c|$ over mixed cells ($\infty$ if unresolved), labelled with `eq:amr:error-labels` with $\tau_r=1$, $\tau_c=\rho$ ($n$ = `cells_per_radius`, $\rho$ = `coarsen_ratio`, default $1/3$): refine iff $\max h|\kappa|>1/n$, coarsen iff all $h|\kappa|<\rho/n$. Claim: with $\varepsilon_r=1/n$, $\varepsilon_c=\rho/n$ the thesis' condition $2\varepsilon_c<\varepsilon_r$ is $2\rho<1$ ($0<2\rho<1$ in the code), and $\rho=\tfrac13$ is the usual ratio.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Thresholds in the code: refine iff $E>1$ and level $<$ max, coarsen iff $E<\rho$ | `phase_criteria.jl:308-318` | ✅ | independent replica of the labels from `interior(interface.curvature)` on 4 static drops (different $R$, levels, $\rho\in\{0.1,0.3,\tfrac13,0.45\}$, $n\in\{4,8,12\}$): 160 labels equal (K1) | `code_01_curvature_hysteresis.jl` | |
| $\varepsilon_r=1/n$, $\varepsilon_c=\rho/n$ | l. 187-191 | ✅ | $E>1\iff h|\kappa|>1/n$, $E<\rho\iff h|\kappa|<\rho/n$ | same | |
| $2\varepsilon_c<\varepsilon_r\iff2\rho<1$ | l. 191-193 | ✅ | $2\rho/n<1/n\iff2\rho<1$ for $n>0$; exhaustive over $n\in\{2,8,40\}$, $\rho\in\{0.1,0.3,0.4999,0.5,0.7\}$ (K2) | same | |
| Constructor enforces exactly $0<2\rho<1$ | `phase_criteria.jl:119` | ✅ | accepts $0.01,\tfrac13,0.4999$; throws for $0.5,0.6,1,0,-0.1$ (K2) | same | |
| Default $\rho=\tfrac13$ is the thesis' $\varepsilon_r=3\varepsilon_c$ | l. 193-195 | ✅ | $\varepsilon_r/\varepsilon_c=1/\rho=3$ | — | |
| Clipping: $E\le n/2$ for a resolved cell | l. 168-171 | ✅ (consistent) | largest observed $E=2.15<4=n/2$ ($n=8$) | same | not a separate test of `eq:interface:clip` (ch. 4) |

## Errors and consequences

None.

## Sharpness

The role of $2\rho<1$ is that coarsening doubles the error (first-order scaling $E\propto h|\kappa|$), so a merged block whose children are below $\rho$ has $E<2\rho<1$ and is not refined again. That is the exact analogue of the factor 4 of `eq-amr-interpolation` (second-order scaling). Measured scaling of the indicator under an actual merge (static circle, curvature re-estimated on the merged mesh, 7 radii):

| quantity | value |
|---|---|
| $E(\text{merged})/\max E(\text{children})$ | $2.04,\,2.03,\,2.37,\,1.98,\,2.16,\,2.01\text{–}2.13,\,2.01\text{–}2.07$ (min $1.98$, max $2.39$) |
| ratio to the ideal 2 | up to $+19.6\,\%$ |
| largest $\rho$ that keeps $\rho\cdot\text{max ratio}<1$ | $\rho<0.418$ |

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| $2\rho<1$ as the hysteresis condition | **sharp for ideal first-order scaling** (necessary and sufficient in that model); with the numerical curvature the effective ratio is 1.98–2.39, so $2\rho<1$ is *not* sufficient for $\rho$ close to $\tfrac12$ (e.g. $\rho=0.45$ with ratio 2.39 gives $E=1.07>1$) | K3, K4 | `code_01_curvature_hysteresis.jl` | require $\rho\lesssim0.4$ or document that the guarantee is for the ideal scaling; the default $\tfrac13$ has a margin of $1.5/1.2$ |
| Default $\tfrac13$ | safe on the tested circles (margin $1.5/1.196=1.25$) | K3 | same | — |
| Equivalence of the two inequalities | **exact** | algebra | same | — |

## Open points

- The static-circle ratio is a single family per radius (the mesh has 4 root blocks); other interface shapes (elongated, thin films) and the unresolved/filament branch ($E=\infty$, which forbids coarsening) are not covered.
- Thesis text not re-read (the quoted condition and "usually $\varepsilon_r=3\varepsilon_c$" are taken from the chapter's citation `\thesis{\S2.5}`).

## History

- 2026-11-10: first audit (commit `2959760`): K1–K4 pass; `run_all.jl` passes.
