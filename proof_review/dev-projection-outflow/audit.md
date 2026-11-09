# Audit: dev:projection:outflow — boundary reset after the projection (and the boundary-value paragraph of §10.5.1)

- Source: `docs/techref/chapters/10_projection.tex:884-918` (boundary values 884-902, check 903-910, deviation 911-918), commit `b1e7b5b`
- Code audited (normative): `src/operators/staggered_diffusion.jl` (`enforce_velocity_boundary!`), `src/navier_stokes/projection.jl` and `src/embedded/operators.jl` (`project!`), `src/single_phase/stepping.jl` (stage order), `src/two_phase/two_velocity/stepping.jl`
- Date: 2026-11-09
- Overall status: ⚠️ the statements of the paragraph and the first three sentences of the deviation are verified; **the last sentence ("The next projection removes the defect again") is misleading** (repeated projection of the same field decays the defect by only a factor $\approx2/3$ per projection, it does not remove it) and the deviation lacks the issue reference (#93)
- Rerun: `julia --project=@claude-symbolics proof_review/dev-projection-outflow/scripts/run_all.jl`

## Statement and assumptions

(Paragraph) The projection applies the velocity boundary conditions before the divergence is formed and again after the update; `Inflow` samples its function, `NoSlip`/`FreeSlip` and the default walls give zero, `Outflow`/`ZeroGradientVelocity` copy the area- and aperture-weighted mean of the parallel interior faces of the boundary cell, `Outflow` adds one uniform outward correction so that the total boundary flux is zero, without an outflow side a nonzero total throws; with a solid the faces with $a_f=0$ are set to zero; the second application changes nothing at walls and inflows but at outflow/zero-gradient sides it copies the corrected interior faces, so the boundary flux of the adjacent cells changes after the solve and the defect identity no longer holds there; the two-velocity step sets the boundary values of both phases before the projection and does not touch them afterwards.
(Deviation) the end-of-step velocity is not discretely solenoidal in the cells next to such sides, the tolerance does not bound their volume defect, "the next projection removes the defect again".

**Assumptions made explicit.** Single phase, no solid for the defect measurement (a channel $2\times1$, composite mesh, parabolic Inflow left, Outflow right, NoSlip walls, $\varepsilon=10^{-12}$, $\delta t=0.01$); the corrected velocity "before the reset" is $v^*-\delta t\,\mathsf R\mathsf Jp$ with the boundary values used in the solve.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| O1 before the reset the corrected velocity is solenoidal to the tolerance; after the reset the Outflow-adjacent cells carry $\delta t|\mathfrak D v|=5.7\times10^{-3}$ (16 cells), the others $1.2\times10^{-13}$ | 903-910, 911-914 | ✅ verified | independent divergence of $v^*-\delta t\mathsf R\mathsf Jp$ and of the returned velocity | `code_01_outflow.jl` | $1.2\times10^{-13}$ before; $5.73\times10^{-3}$ (16 cells) and $1.2\times10^{-13}$ after; the chapter's numbers ($5.7\times10^{-3}$, $1.2\times10^{-13}$, 16 cells) coincide |
| O2 total boundary flux (hence $\sum_c|c|\mathfrak Dv$) stays zero after the reset | 892-895 | ✅ verified | sum of the end-of-step defects | `code_01_outflow.jl` | $1.4\times10^{-16}$ (largest single $|c|\mathfrak Dv=2.2\times10^{-3}$): the defects sum to zero |
| O3 "the next projection removes the defect again" | 915-918 | ⚠️ misleading (F1) | project the same end-of-step velocity again, 4 times | `code_01_outflow.jl` | end-of-step $\delta t\max|\mathfrak Dv|=3.7\times10^{-3},2.5\times10^{-3},1.7\times10^{-3},1.1\times10^{-3}$ after the 2nd…5th projection: reduced by $\approx2/3$ each time, never removed |
| O4 `ZeroGradientVelocity` side + inflow without an Outflow side: nonzero net flux throws | 892-895 | ✅ verified | `ArgumentError` | `code_01_outflow.jl` | |
| O5 walls/inflow sides: the second application changes nothing (closed box with FreeSlip/NoSlip/Inflow(0)) | 896-899 | ✅ verified | $\delta t\max|\mathfrak Dv|=1.8\times10^{-13}$ | `code_01_outflow.jl` | |
| O6 with a solid the faces with $a_f=0$ are zero after the projection (and before: garbage $+1$ removed) | 899-900 | ✅ verified | 203 faces, composite + sloped solid | `code_01_outflow.jl` | |
| O7 Outflow value = area-weighted mean of the parallel interior faces + one uniform correction; total flux zero | 889-895 | ✅ (script) | independent evaluation on the composite channel | `code_01_outflow.jl` | |
| S1 stage order: one-velocity step uses $t^{n+1}$ for the boundary values; two-velocity step sets both phases before and not after | 884-902 | ✅ code reading | `single_phase/stepping.jl:100-131` (`boundary_time = current_time + time_step`), `two_velocity/stepping.jl:326-355` | — | |

## Errors and consequences

**F1 (misleading sentence).** "The next projection removes the defect again." Each projection *solves* for the divergence including the inherited defect (its pre-reset velocity is solenoidal to the tolerance) but ends with the same reset, which re-extrapolates the Outflow values from the corrected interior and re-creates a defect. Measured by projecting the same end-of-step field repeatedly: $5.7\times10^{-3}\to3.7\times10^{-3}\to2.5\times10^{-3}\to1.7\times10^{-3}\to1.1\times10^{-3}$ (factor $\approx0.66$). In time stepping the defect is therefore carried from step to step at the size set by the interior variation near the outflow, and it is part of what issue #90 sees (inherited divergence divided by a short step). It is filed as issue #93 ("Outflow / ZeroGradientVelocity boundary reset after the projection breaks the divergence constraint"), which the box does not cite.

*Proposed replacement of the last two sentences of the box* (macros existing):

```latex
The end-of-step velocity is therefore not discretely solenoidal in the cells next to such sides,
and the tolerance does not bound their volume defect. The next projection treats this defect as
divergence to remove, but ends with the same reset: projecting the same field again reduces the
defect only by a factor of about $2/3$ ($\chkProjOutflow$, $3.7\times10^{-3}$, $2.5\times10^{-3}$,
$\dots$), so it is carried from step to step; the total boundary flux, and hence the sum of the
cell defects, stays zero (\issue{93}).
```
(The numbers after $\chkProjOutflow$ are measured by the audit script on the channel of the chapter (its first value, $5.73\times10^{-3}$, equals the chapter's); if the chapter should print them, `figures/projection.jl` needs a repeated-projection loop.)

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| defect size $5.7\times10^{-3}$ next to the outflow (tolerance $10^{-12}$) | **sharp** (quantified) | reproduced independently; sets in the 16 cells of the outflow side | `code_01` | — |
| "the tolerance does not bound their volume defect" | **sharp** | defect $\sim10^{9}\times$ the tolerance | `code_01` | — |
| total flux preserved by the Outflow correction | **sharp** | $1.4\times10^{-16}$ | `code_01` | — |

## Open points

- The ZeroGradientVelocity-with-Outflow companion case (net flux corrected only on Outflow faces) was not measured separately.
- A fix would project with the boundary values re-extrapolated consistently (outer iteration or pressure Dirichlet) — design decision, not audited.

## History

- 2026-11-09: first audit (script `code_01_outflow`). `run_all.jl`: PASS (1 code script).
