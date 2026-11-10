# Audit: eq:projection:face-data (group) — two-velocity face data, mixed faces, mixture divergence, phase gradients and identities, decomposition, mixture update, component gauge

- Source: `docs/techref/chapters/10_projection.tex:194-314` (face-data 205, mixed 220, mixture-divergence 227, phase-gradient 237, phase-identities 251, decomposition 260, mixture-update 269, gauge 289), commit `b1e7b5b`
- Code audited (normative): `src/two_phase/projection/common.jl` (`_projection_face_data`, `_projection_gradient_core!`, `_projection_correct_phase_velocities!`, `_projection_gauge_partition!`, `_projection_pressure_operator`), `src/two_phase/projection/cmdgfm_cache.jl` (`_cmdgfm_cell_flux!`, `_cmdgfm_pressure_jump`), `src/two_phase/two_velocity/velocity_fields.jl` (`_mark_inactive_phase_velocities!`)
- Date: 2026-11-09
- Overall status: ✅ verified (symbolic for the algebra, package for the discrete statements, 4 chapter fixtures and a 2-component fixture); no error in a formula; two wording remarks (W1, W2)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-two-velocity/scripts/run_all.jl`

## Statement and assumptions

Per face: masses $m^{\rm l}=\rho^{\rm l}\hat C$, $m^{\rm g}=\rho^{\rm g}(1-\hat C)$, $m=m^{\rm l}+m^{\rm g}=\bar\rho_f$, $m^{\rm r}=m^{\rm l}m^{\rm g}/m$, $\Xi^{\rm l}=-m^{\rm g}$, $\Xi^{\rm g}=m^{\rm l}$, $a^\Xi=(a^{\rm l}\Xi^{\rm l}+a^{\rm g}\Xi^{\rm g})/m$; mixed faces $\hat C\in(\epsilon_f,1-\epsilon_f)$, $a^{\rm l}+a^{\rm g}>0$; mixture divergence $|c^{\rm o}|\mathfrak D^{\rm mix}=\sum_f o_{c,f}|f|(a^{\rm l}u^{\rm l}+a^{\rm g}u^{\rm g})$ (boundary faces with $s_f$); phase gradients $G^\pi=\mathsf R^{\rm tv}\mathsf Jp+\tfrac{\Xi^\pi}m\xi$; identities $G^{\rm g}-G^{\rm l}=\xi$, $m^{\rm l}G^{\rm l}+m^{\rm g}G^{\rm g}=m\,\mathsf R^{\rm tv}\mathsf Jp$; decomposition $u^\pi=\bar u+\tfrac{\Xi^\pi}m[u]$, $\sum m^\pi(u^\pi)^2=m\bar u^2+m^{\rm r}[u]^2$, $a^{\rm l}u^{\rm l}+a^{\rm g}u^{\rm g}=a\bar u+a^\Xi[u]$; update $|c^{\rm o}|\mathfrak D^{\rm mix}(v^{n+1})=|c^{\rm o}|\mathfrak D^{\rm mix}(v^*)+\delta t(\mathsf S^{\rm tv}p+\mathsf J^\top\operatorname{diag}(a^\Xi)\xi)$ with $\mathsf J^\top\mathsf A\mathsf R^{\rm tv}\mathsf J=\mathsf S^{\rm tv}$; component identity (gauge) $\sum_{c\in\mathcal K}|c^{\rm o}|\mathfrak D^{\rm mix}=\sum_{f\in\mathcal F_{\rm bnd},\,c(f)\in\mathcal K}s_f|f|(a^{\rm l}u^{\rm l}+a^{\rm g}u^{\rm g})$.

**Assumptions made explicit.** $m^{\rm l},m^{\rm g}>0$ in the decomposition (both phases present); absent phase velocities are NaN and carry weight 0 (never multiplied); $a^{\rm l}+a^{\rm g}=a_f$ (verified here to $1.1\times10^{-16}$ on all fixtures, eq-interface-phase-aperture); walled box.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| F1 face data: $\hat C$ (independent staggered fraction), $m^{\rm l},m^{\rm g},m,\Xi,m^{\rm r},a^\Xi$ vs `_projection_face_data`; $a^{\rm l}+a^{\rm g}=a$; mixed set | 196-226 | ✅ verified | own staggered fraction (volume weights without solid; independent open-strip areas with solid), 6 fixtures | `code_01_data_flux_gauge.jl` | relative $\le3\times10^{-15}$; $a^\Xi$ to $5.6\times10^{-17}$; mixed sets equal |
| F2 presence: liquid present iff $\hat C>0$ (and $V^{\rm o}>0$), gas iff $\hat C<1$ | 196-203 | ✅ verified | NaN pattern of the marked predictor | `code_01_data_flux_gauge.jl` | exact |
| F3 mixture divergence incl. boundary faces ($s_f$, boundary phase apertures) | 227-236 | ✅ verified | independent sum vs `_cmdgfm_cell_flux!` with random boundary values of both phases | `code_01_data_flux_gauge.jl` | difference 0.0 |
| F4a $\sum_fo_{c,f}|f|v_f=-(\mathsf J^\top v)_c$ | 275-279 | ✅ verified | random $v$ | `code_01_data_flux_gauge.jl` | $1.2\times10^{-16}$ |
| F4b $\mathsf J^\top\mathsf A\mathsf R^{\rm tv}\mathsf J=\mathsf S^{\rm tv}$, $\mathsf S^{\rm tv}=\mathsf J^\top\gamma^{{\rm tv}\,1/2}\mathsf K^{\rm o}\gamma^{{\rm tv}\,1/2}\mathsf J$, $\gamma^{\rm tv}=a/\bar\rho$ | 244-250, 275-279 | ✅ verified | dense independent $\mathsf R^{\rm tv}$ | `code_01_data_flux_gauge.jl` | $\le1.8\times10^{-16}$ |
| P1 $G^{\rm g}-G^{\rm l}=\xi$; $m^{\rm l}G^{\rm l}+m^{\rm g}G^{\rm g}=m\mathsf R^{\rm tv}\mathsf Jp$ | 251-256 | ✅ verified | symbolic (exact rational evaluation at 20 random points + `simplify_fractions`); numerically on the CMDGFM solution ($\le2\times10^{-13}$, `eq-projection-cmdgfm-system`) | `sym_01_phase_algebra.jl` | |
| D1-D3 decomposition: $u^\pi$, kinetic energy, mixture flux | 260-267 | ✅ verified | symbolic | `sym_01_phase_algebra.jl` | |
| D4 update $\bar u\leftarrow\bar u-\delta t\,\mathsf R^{\rm tv}\mathsf Jp$, $[u]\leftarrow[u]-\delta t\xi$, flux $\leftarrow-\delta t(a\mathsf R^{\rm tv}\mathsf Jp+a^\Xi\xi)$ | 268-275 | ✅ verified | symbolic face-wise; discrete matrix form on the CMDGFM solution (cell by cell, 4 fixtures) | `sym_01_phase_algebra.jl`, `eq-projection-cmdgfm-system/code_02` | $\le1.1\times10^{-14}$ |
| D5 $a^\Xi=\hat C(1-\hat C)(\rho^{\rm l}-\rho^{\rm g})/m$ without solid, $a^{\rm l}=\hat C$ | 279-281 | ✅ verified | symbolic | `sym_01_phase_algebra.jl` | |
| G1 anchors/partition: smallest index per component of (open cells, faces with $a_f>0$) | 281-288 | ✅ verified | BFS vs `pressure_indices`, `pressure_anchors` (CMDGFM) | `eq-projection-cmdgfm-system/code_02` | equal on 4 fixtures |
| G2 component identity (gauge) with random boundary values; constant of each component is a null vector of $\mathsf S^{\rm tv}$ | 289-299 | ✅ verified | band fixture with two components + 4 fixtures | `code_01_data_flux_gauge.jl` | identity $\le1.2\times10^{-14}$, null vector $\le1.2\times10^{-16}$ |
| G3 "the dropped row holds as soon as the other rows hold if that flux vanishes; otherwise the anchor keeps the whole defect" | 299-304 | ✅ verified | inflow $0.25$ into the lower component only | `code_01_data_flux_gauge.jl` | all other cells $1.8\times10^{-12}$; anchor cell $\delta t|\mathfrak D^{\rm mix}|=0.0343=\delta t|\text{flux}|/|c^{\rm o}|$; the solve reports `converged=false` |

## Errors and consequences

No formula is wrong. Wording remarks:

**W1.** The text says the unknowns "$\gradjump_f=0$ elsewhere (except on the inactive faces of the mimetic projection)" — correct; but the decomposition paragraph says "on a face where both phases are present": for faces that are present with both phases but are not in $\mathcal F_{\rm mix}$ (fraction within $\epsilon_f$ of 0 or 1) the pair $(\bar u,[u])$ is defined and $[u]$ is *not* moved; this is what the energy remark of `mimetic-energy` needs (see `eq-projection-kkt`). No change needed.

**W2.** The gauge paragraph says "(all cells and faces without solid)"; with a solid the graph has the cells of positive open fraction and the interior faces with $a_f>0$. Statement correct as written (verified on the band fixture: 2 components, 2 anchors).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| decomposition identities | **sharp** (algebraic identities) | exact | `sym_01` | — |
| zero boundary flux per component is needed for the dropped row | **sharp** | counterexample F6: anchor keeps defect $=$ flux$/|c^{\rm o}|$ exactly | `code_01` | state the quantity: after the solve $\delta t|\mathfrak D^{\rm mix}|_{\rm anchor}=\delta t\,|\Phi_{\mathcal K}|/|c^{\rm o}_{\rm anchor}|$ |
| $a^{\rm l}+a^{\rm g}=a_f$ | exact on the fixtures ($1.1\times10^{-16}$); clamp inactive away from covered neighbours (`eq-interface-phase-aperture`) | | `code_01` | — |

## Open points

- The numbers of the face-data macros of the chapter are not recomputed here; formulas verified on independent data.
- Components with a hanging-node periodic self-edge (`_projection_incidence!` writes explicit zeros) were not exercised (walled box only).

## History

- 2026-11-09: first audit (scripts `sym_01_phase_algebra`, `code_01_data_flux_gauge`). `run_all.jl`: PASS (1 symbolic + 1 code script).
