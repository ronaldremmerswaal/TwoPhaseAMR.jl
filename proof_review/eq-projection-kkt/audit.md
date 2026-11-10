# Audit: eq:projection:kkt (group) — mimetic projection: active/inactive faces, kinetic-energy minimisation, KKT system, Pythagoras identity

- Source: `docs/techref/chapters/10_projection.tex:557-647` (active 566, inactive 576, mimetic-min 589, kkt 603, mimetic-energy 634), commit `b1e7b5b`
- Code audited (normative): `src/two_phase/projection/mimetic_projection.jl` (`_local_active_faces`, `_assemble_mimetic_system!`, `project!`, `_local_independent_rows`), `src/two_phase/projection/common.jl`
- Date: 2026-11-09
- Overall status: ⚠️ all five labelled claims verified; **mimetic-energy** is exact but its stated sufficient conditions can be sharpened to an exact leftover formula (verified), and the sentence "true unless a staggered fraction lies within $\epsilon_f$ of 0 or 1" is not tested on the fixtures (leftover $\le10^{-16}$ there)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-kkt/scripts/run_all.jl`

## Statement and assumptions

(active) $\mathcal F_\xi=\{f\in\mathcal F_{\rm mix}:\ \text{both cells uncut up to the floor},\ \min(m^{\rm l},m^{\rm g})/m\ge\epsilon_{\rm act},\ a^{\rm l}+a^{\rm g}>\epsilon_f\}$, $\epsilon_{\rm act}=10^{-6}$; (inactive) $\xi_f=[u^*]_f/\delta t$, $[u^{n+1}]_f=0$ there; (mimetic-min) the projection minimises $\mathcal E_{\rm def}=\tfrac12\|\delta\bar u\|^2_{\mathsf M^{\rm tv}}+\tfrac12\sum_{\mathcal F_\xi}m^{\rm r}V^{\rm o}\delta[u]^2$ over corrections with $\mathfrak D^{\rm mix}(v^{n+1})=0$ on the retained cells and $\mathsf N[u^{n+1}]_{\mathcal F_\xi}=0$; $\delta\bar u=-\delta t\,\mathsf R^{\rm tv}\mathsf Jp$, $\delta[u]=-\delta t\xi$; (kkt) the symmetric indefinite system $\begin{bsmallmatrix}-\mathsf S&\mathsf C&0\\\mathsf C^\top&\mathsf D&\mathsf N^\top\\0&\mathsf N&0\end{bsmallmatrix}\begin{bsmallmatrix}p\\\xi\\\mu\end{bsmallmatrix}=\tfrac1{\delta t}\begin{bsmallmatrix}b\\0\\\mathsf N[u^*]\end{bsmallmatrix}$, $\mathsf C=-\mathsf J_{\mathcal F_\xi,r}^\top\operatorname{diag}(a^\Xi)$, $b_c=|c^{\rm o}|\mathfrak D^{\rm mix}(v^*)_c+(\mathsf J^\top_{{\rm inact},r}\operatorname{diag}(a^\Xi)[u^*]_{\rm inact})_c$, nonsingular if $\mathsf S$ is and the rows of $\mathsf N$ are independent; (mimetic-energy) $E(v^*)=E(v^{n+1})+E(v^{n+1}-v^*)$ for an exact solution if the boundary flux of every component vanishes and $a^\Xi[u]=0$ on the faces outside $\mathcal F_{\rm mix}$ that carry both phases, $E(v)=\|\bar u\|^2_{\mathsf M^{\rm tv}}+\sum m^{\rm r}V^{\rm o}[u]^2$.

**Assumptions made explicit.** The fitted rows $\mathsf N$ are data here (their construction is audited in `eq-projection-mimetic-rows`); walled box; $\mathsf M^{\rm tv}=\mathsf A\gamma^{{\rm tv}\,-1/2}(\mathsf K^{\rm o}_{++})^{-1}\gamma^{{\rm tv}\,-1/2}\mathsf A$; $E$ is a norm of the *intrinsic* velocities weighted with the open staggered masses, not the physical kinetic energy.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| K1 active set (printed rules) = code's `active_faces`; inactive = mixed $\setminus$ active | 563-576 | ✅ verified | independent rules vs code, 4 fixtures × 2 variants | `code_01_kkt_minimisation.jl` | 72/43/117/73 active, 0/8/2/10 inactive |
| K2 dense KKT matrix and right-hand side from the text = assembled ones | 603-623 | ✅ verified | independent $\mathsf S,\mathsf C,\mathsf D,b$ with the code's fitted rows | `code_01_kkt_minimisation.jl` | $\le1.1\times10^{-16}$ (matrix), $\le2.8\times10^{-16}$ (rhs); the code's matrix is exactly symmetric (0.0) |
| K2' inertia: symmetric indefinite, nonsingular | 624-628 | ✅ verified | eigenvalues | `code_01_kkt_minimisation.jl` | $(+n_\xi,-(n_p+n_{\mathsf N}))$ in all 8 cases, e.g. $(+72,-279)$, $(+117,-678)$; smallest $|\lambda|/|\lambda|_{\max}=8\times10^{-6}$ |
| K3 the KKT solution is the constrained minimiser (and $\delta\bar u=-\delta t\,\mathsf R^{\rm tv}\mathsf Jp$, $\delta[u]=-\delta t\xi$) | 583-602 | ✅ verified | NULL-SPACE method (constraints $\mathsf G z=h$, SVD null space $Z$, $y=-(Z^\top WZ)^{-1}Z^\top Wz_p$), independent of the multiplier formulation; feasibility and null-space gradient of the code's correction | `code_01_kkt_minimisation.jl` | $|z_{\rm opt}-z_{\rm code}|/|z_{\rm code}|\in[1.8\times10^{-13},3.5\times10^{-12}]$; feasibility $\le1.2\times10^{-14}$; $|Z^\top Wz|/|Wz|\le1.1\times10^{-15}$; $E_{\rm code}/E_{\rm opt}-1\le1.2\times10^{-14}$ |
| K4 inactive faces: $\xi=[u^*]/\delta t$, $[u^{n+1}]=0$ | 576-582 | ✅ verified | | `code_01_kkt_minimisation.jl` | $[u^{n+1}]\le4.4\times10^{-16}$ (largest $|[u^*]|=0.6$) |
| E1 Pythagoras $E(v^*)=E(v^{n+1})+E(v^{n+1}-v^*)$ | 629-645 | ✅ verified | independent dense $E$ with $\mathsf M^{\rm tv}$, 4 fixtures × 2 variants | `code_02_mimetic_energy.jl` | $\le7.6\times10^{-16}$ of $E(v^*)$; 2.8-16.4 % of $E$ removed |
| E1' code statistic `energy_identity_error` $=2|\langle v^{n+1},v^{n+1}-v^*\rangle_E|$ | 640-644 | ✅ verified | independent cross term | `code_02_mimetic_energy.jl` | agrees (both $\le2\times10^{-16}$ of $E^*$) |
| E2 conditions: boundary flux per component zero and $a^\Xi[u]=0$ on the faces outside $\mathcal F_{\rm mix}$ carrying both phases | 636-642 | ⚠️ sufficient, exactly quantified (F1) | floor $\epsilon_f=0.1$: 12 such faces; $E(v^*)-E(v^{n+1})-E(\Delta)=-4.1936=-2\delta t\sum_{f\notin\mathcal F_{\rm mix}}a^\Xi_f[u^{n+1}]_f(\mathsf Jp)_f$ | `code_02_mimetic_energy.jl` | predicted $-4.193575207344$ vs actual $-4.193575207343$ ($7\times10^{-4}$ of $E^*$) |
| E3 the CMDGFM has no energy identity | 575-583 (CMDGFM paragraph) | ✅ verified | same $E$ on the CMDGFM result | `code_02_mimetic_energy.jl` | $|E^*-E^{n+1}-E_\Delta|/E^*=8.9\times10^{-3}$ |

## Errors and consequences

No formula is wrong. One precision improvement (F1) and one unverified sentence (U1).

**F1 (mimetic-energy).** The proof in the text gives the identity under the two stated conditions; dropping the second one leaves the exact defect
$E(v^*)-E(v^{n+1})-E(v^{n+1}-v^*)=-2\,\delta t\sum_{f}a^\Xi_f\,[u^{n+1}]_f\,(\mathsf Jp)_f$, the sum over the faces carrying both phases that are not in $\mathcal F_{\rm mix}$ (verified to $2.5\times10^{-13}$ relative with a coarse floor). *Proposed replacement of the end of the sentence introducing (mimetic-energy)*, from "for an exact solution," to "(true unless … without being 0 or 1)":

```latex
for an exact solution if the boundary flux of every component vanishes; otherwise the left-hand side
exceeds the right-hand side by
$-2\dt\sum_{f}\apgfm{f}\jump{\velo^{\timelevel{n+1}}}_f(\jumpmat\pres)_f$, summed over the faces that
carry both phases but are not in $\facesmix$ (a staggered fraction within $\ffloor$ of $0$ or $1$
without being $0$ or $1$; the sum is zero when $\apgfm{f}\jump{\velo}_f=0$ there, as it is on the
fixtures of \cref{tab:projection:counts}, and amounts to $7\times10^{-4}$ of $E(\velo^{\stage{3}})$ for
$\ffloor=0.1$)
```

**U1.** "true unless a staggered fraction lies within $\epsilon_f$ of 0 or 1 without being 0 or 1" — the leftover is $\le10^{-16}$ of $E^*$ on all fixtures (no such face with $a^\Xi[u]\ne0$ occurs at the default floor $\sqrt\epsilon$); the converse (that it fails whenever such a face exists) is shown by the $\epsilon_f=0.1$ experiment.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (mimetic-min) the KKT solution minimises $\mathcal E_{\rm def}$ | **sharp** (global minimum of a convex QP) | null-space gradient $\le10^{-15}$, independent minimiser equal to $10^{-12}$ | `code_01` | — |
| rank condition on $\mathsf N$ | **sharp** (nonsingularity iff) | inertia $(+n_\xi,-(n_p+n_N))$ requires independent rows; with dependent rows the Lagrange multiplier block is singular (the code drops them) | `code_01` | — |
| (mimetic-energy) conditions | **sharp** | exact leftover formula (F1) verified | `code_02` | give the formula |
| zero boundary flux per component | **sharp** | otherwise the anchor row cannot hold (`eq-projection-two-velocity`, F6) and the cross term $\sum_cp_c|c^{\rm o}|\mathfrak D^{\rm mix}$ does not vanish | `eq-projection-two-velocity/code_01` | — |
| Pythagoras for the CMDGFM | **fails** (0.9 % of $E^*$) | as the text says | `code_02` | — |

## Open points

- The dependent-row branch of the rank selection is audited separately on synthetic rows (`eq-projection-mimetic-rows/code_03`); on the fixtures no row is dependent.
- Issue #76 (mimetic hydrostatic) concerns the numbers in `dev:projection:hydrostatic`, not these formulas; not audited here.

## History

- 2026-11-09: first audit (scripts `code_01_kkt_minimisation`, `code_02_mimetic_energy`). `run_all.jl`: PASS (2 code scripts).
