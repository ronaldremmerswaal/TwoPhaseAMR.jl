# Audit: eq:projection:cont-update, cont-gradjump, cont-energy — continuous model of the projection; sign of thesis eq. (7.24)

- Source: `docs/techref/chapters/10_projection.tex:26-69` (cont-update 29, cont-gradjump 40, cont-energy 59), commit `b1e7b5b`; thesis: `~/Documents/phd_project/thesis/pt2_poisson.tex:14-18` (eqs. (7.8), (7.9)), `pt2_gradient_mimetic.tex:5-9` (eq. (7.24)), `basic_navier_stokes.tex:61-66` (definition of $[\![\cdot]\!]$ and $\eta$)
- Date: 2026-11-09
- Overall status: ✅ cont-update/cont-gradjump verified; cont-energy verified **with the plus sign**; the **writer's flag is confirmed: thesis (7.24) prints the interface term with the wrong sign** (a typo; the minus sign holds for $[\![\cdot]\!]$ = liquid minus gas or $\eta$ into the liquid)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-continuous/scripts/run_all.jl`

## Statement and assumptions

1. (cont-update) $\mathbf U^{\pi,n+1}=\mathbf U^{\pi,*}-\tfrac{\delta t}{\rho^\pi}\nabla p^\pi$ in $\Omega^{\rm o}\cap\Omega^\pi$, $\nabla\!\cdot\mathbf U^{\pi,n+1}=0$.
2. (cont-gradjump) imposing $[\![\mathbf U\!\cdot\!\eta]\!]=0$ on (1): $[\![\tfrac1\rho\,\eta\!\cdot\!\nabla p]\!]=[\![\mathbf U^*\!\cdot\!\eta]\!]/\delta t$ (thesis (7.9); $[\![a]\!]=a^{\rm g}-a^{\rm l}$, $\eta$ into the gas).
3. (cont-energy) for $\mathbf U\!\cdot\!\mathbf n=0$ on $\partial\Omega$ and $[\![p]\!]=0$: $-\int_\Omega\mathbf U\!\cdot\!\nabla p\,\mathrm dV=\int_\Omega p\,\nabla\!\cdot\!\mathbf U\,\mathrm dV+\int_{\mathcal I}p\,[\![\mathbf U\!\cdot\!\eta]\!]\,\mathrm dS$, "the thesis prints the interface term with a minus sign; with $[\![\cdot]\!]$ the gas minus the liquid value and $\eta$ pointing into the gas it is a plus".

**Assumptions made explicit.** Piecewise smooth $\mathbf U^\pi$, $p$ continuous across $\mathcal I$ (single-valued), $\mathcal I$ a closed or wall-to-wall interface, divergence theorem applicable in each phase; thesis conventions as quoted from `basic_navier_stokes.tex`: "The interface normal is denoted by $\eta$, and points into the gas phase", "$[\![a]\!](t,\mathbf x)\defeq a^{\rm g}-a^{\rm l}$".

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| C1 (7.9) $\Leftrightarrow$ continuity of $\mathbf U\!\cdot\!\eta$ after the update | 40-44 | ✅ verified | symbolic: $[\![u^{n+1}_\eta]\!]=-\delta t\,([\![\tfrac1\rho\partial_\eta p]\!]-[\![u^*_\eta]\!]/\delta t)$ identically; the convention $[\![\cdot]\!]$ = liquid − gas gives the same equation | `sym_02_gradjump.jl` | the chapter's (cont-gradjump) is thesis (7.9) verbatim |
| C2 per-phase divergence theorem, liquid outward normal $\eta$, gas outward normal $-\eta$ | 59-65 | ✅ verified | derivation (below) and C3 | — | |
| C3 (cont-energy) with **plus** sign | 59-65 | ✅ verified | exact rational integration (Symbolics): $\Omega=[0,1]^2$, interface $y=\tfrac14+\tfrac12x$ (liquid below), random polynomial $p$ (one polynomial), different polynomial $\mathbf U^{\rm l},\mathbf U^{\rm g}$ with $\mathbf U\!\cdot\!\mathbf n=0$ on the four walls | `sym_01_energy_split.jl` | $-\!\int\mathbf U\!\cdot\!\nabla p=-\tfrac{1618849}{1451520}$, $\int p\,\nabla\!\cdot\!\mathbf U=-\tfrac{785123}{725760}$, $\int_{\mathcal I}p[\![U_\eta]\!]=-\tfrac{16201}{483840}$; residual of the chapter's identity $=0$ exactly |
| C4 thesis (7.24) as printed ("$-\int_I p[\![u_\eta]\!]$") | `pt2_gradient_mimetic.tex:6` | ❌ sign error | same integrals | `sym_01_energy_split.jl` | residual of the printed sign $=-\tfrac{16201}{241920}=2\int_{\mathcal I}p[\![U_\eta]\!]\ne0$ |
| C5 "pressure does no work on a solenoidal field with continuous normal velocity" | 66-69 | ✅ | follows from C3 | — | |

**Derivation.** Per phase, $-\int_{\Omega^\pi}\mathbf U^\pi\!\cdot\!\nabla p=\int_{\Omega^\pi}p\nabla\!\cdot\!\mathbf U^\pi-\oint_{\partial\Omega^\pi}p\,\mathbf U^\pi\!\cdot\!\mathbf n^\pi$. On $\mathcal I$: $\mathbf n^{\rm l}=\eta$, $\mathbf n^{\rm g}=-\eta$, so the interface contribution is $-\int_{\mathcal I}p\,U^{\rm l}_\eta+\int_{\mathcal I}p\,U^{\rm g}_\eta=+\int_{\mathcal I}p\,[\![U_\eta]\!]$ with $[\![\cdot]\!]=\mathrm g-\mathrm l$. The wall terms vanish since $\mathbf U\!\cdot\!\mathbf n=0$ there.

## Errors and consequences

**Thesis (7.24) sign (writer's flag): confirmed.** With the thesis' own definitions (`basic_navier_stokes.tex:61-66`: $[\![a]\!]=a^{\rm g}-a^{\rm l}$; $\eta$ into the gas) the interface term of (7.24) is $+\int_I p[\![u_\eta]\!]\,\mathrm dS$, not $-\int_I p[\![u_\eta]\!]\,\mathrm dS$. The printed minus is correct only for the opposite jump convention ($[\![\cdot]\!]=\mathrm l-\mathrm g$) or $\eta$ into the liquid. It is a typo of the thesis with no consequence for the rest of the thesis' argument (it is used qualitatively: "(7.24) shows that the pressure gradient's contribution to kinetic energy vanishes if $\mathbf u$ is divergence free and continuous in the interface normal direction"; (7.38)-(7.44) are discrete statements built from the discrete analogue, not from the printed sign). The chapter's parenthetical remark is **correct** and needs no change.

*Optional strengthening of the remark (macros existing):*

```latex
(the thesis prints the interface term with a minus sign; with $\jump{\cdot}$ the gas minus the
liquid value and $\inormal$ pointing into the gas it is a plus, which was checked by exact
integration on a polynomial example, \code{proof_review/eq-projection-continuous/})
```

**E1 (editorial, line 34-35 of the source).** The sentence after (cont-update) is a fragment: "…, and in the one-velocity model with one velocity $\Velo$ and the piecewise constant density $\dens$. The phase velocities are continuous…" (no predicate). *Proposed replacement:*

```latex
and in the one-velocity model the same update holds with the one velocity $\Velo$ and the piecewise
constant density $\dens$ in place of $\Velo^{\phase}$ and $\dens^{\phase}$. The phase velocities are
continuous in the interface normal direction, $\jump{\Velo\cdot\inormal}=0$~\thesis{eq.~(6.1)}, but not
tangentially; imposed on~\cref{eq:projection:cont-update} this gives the jump condition~\thesis{eq.~(7.9)}
```

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (cont-energy) exact identity | **sharp** (equality) | exact rational arithmetic; the opposite sign misses by $2\int_{\mathcal I}p[\![U_\eta]\!]$ | `sym_01` | — |
| $[\![p]\!]=0$ | **sharp** (necessary) | with a pressure jump $[\![p]\!]\ne0$ the per-phase boundary terms $p^{\rm l}U^{\rm l}_\eta$, $p^{\rm g}U^{\rm g}_\eta$ no longer combine into $p[\![U_\eta]\!]$ (extra $[\![p]\!]\langle U_\eta\rangle$-type term); the chapter's restriction (surface tension enters through the predictor) is exactly what the identity needs | (algebra) | — |
| $\mathbf U\!\cdot\!\mathbf n=0$ on $\partial\Omega$ | sufficient; with prescribed boundary flux the wall term $\oint p\,\mathbf U\!\cdot\!\mathbf n$ appears (the discrete analogue is the boundary work $2\delta tB$ of `eq-projection-one-system`) | | | state as in the one-velocity energy sentence |

## Open points

- The claims of the first paragraph of the section that are not formulas ("each phase velocity on its own part of $\partial c$", eq. (6.15); homogeneous Neumann; zero total flux per component) are standard and were not mechanically tested.

## History

- 2026-11-09: first audit (scripts `sym_01`, `sym_02`). `run_all.jl`: PASS (2 symbolic scripts).
