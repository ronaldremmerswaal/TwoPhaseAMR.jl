# Audit: eq:momentum:skew — skew-symmetry of the centred convection and its boundary terms

- Source: `docs/techref/chapters/06_momentum_transport.tex:486-512` (equation at 489-492, `eq:momentum:euler-energy` at 505-510), boundary dual faces 364-412; commit `ad10545` (branch `docs/techref-momentum`, PR #66)
- Code audited (normative): `src/operators/advection/staggered_update.jl` (`convection!`), `src/operators/advection/staggered_fluxes.jl` (`_add_boundary_convection!`, `_split_convection_rate!`), `src/operators/advection/staggered_schedules.jl` (`_boundary_dual_faces`)
- Date: 2026-11-08
- Overall status: ⚠️ periodic statement verified (exact); the wall parenthesis is wrong as printed (needs the closed dual divergence); energy statement needs the two wall limitations
- Rerun: `julia --project=@claude-symbolics proof_review/eq-momentum-skew/scripts/run_all.jl`

## Statement and assumptions

**Claim 2 (as printed).** For the all-centred operator (`Centred` as `scheme` and `amr_interface_scheme`), on a periodic mesh "and with walls where the boundary values vanish",
`⟨ψ, C q⟩ + ⟨q, C ψ⟩ = Σ_f V_f q_f ψ_f (div Υ)_f` and `2 ∂(C q)_f/∂q_f = (div Υ)_f`, with `⟨a,b⟩ = Σ V a b`, `C = V^{-1} Σ_g σ_{f,g} Υ_g (q_tail + q_head)/2` (+ boundary dual faces). Consequences: for `div Υ = 0` the semi-discrete energy `½⟨q,q⟩` is conserved on uniform, composite and cut meshes; the default `Centred()` (MonotonizedCentral at interfaces) is not skew on a composite mesh; forward Euler adds `½ δt² ‖C q‖²_V` (`eq:momentum:euler-energy`).

**Assumptions made explicit.**
1. `Υ_g` arbitrary real rates, `V_f > 0`; no use of the rules (N)/(T) for the algebraic identity: it holds on any dual graph (including two dual faces joining the same pair).
2. "`div Υ`" in the displayed identity is the **interior** dual divergence of `eq:momentum:dual-divergence`; the boundary dual faces are not in it (this is the defect of the wall parenthesis).
3. Boundary states are those of `eq:momentum:boundary` (`Centred`: `(q_f + q_b)/2` for normal faces); boundary values `q^∂_b = 0` unless stated.
4. Energy statements assume `div Υ = 0` outside the two wall limitations (identity audit); for cut meshes `div^o υ = 0` (open-solenoidal velocity).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. `⟨ψ,Cq⟩ + ⟨q,Cψ⟩ = Σ V q ψ div Υ` (periodic) | tex 489-492 | ✅ verified | symbolic on an arbitrary 5-unknown/8-dual-face graph with random orientations (S1); package: operator assembled column by column from `convection!` (Centred, `δt = 0`) on periodic uniform, composite and cut meshes, both components, random and streamfunction velocities (K1, K3) | `step01_skew_symbolic.jl` (46 tests), `code_01_skew_periodic.jl` | `V C + (V C)^T = diag(V div Υ)` to 6.0e-16 relative to the largest entry; bilinear form 8.5e-17 |
| 2. `2 ∂C_ff/∂q_f = div Υ` (periodic) | tex 492 | ✅ verified | symbolic S1, package K2 | same | 5.4e-16 |
| 3. Arithmetic mean is necessary for the two-point state | (new) | ✅ verified | symbolic S2: `α q_tail + (1−α) q_head` adds `(1−2α) Υ_g` to the off-diagonal symmetric part; package K7 with `α = 0.4` | `step01_skew_symbolic.jl`, `sharp_01_arithmetic_mean.jl` | off-diagonal symmetric part becomes of the size of the largest entry |
| 4. Energy conservation for `div Υ = 0` (periodic, uniform/composite/cut) | tex 495-498 | ✅ verified | package: `q^T V C q` for streamfunction (open-solenoidal on cut meshes) velocities | `code_01_skew_periodic.jl` (K4) | 7.9e-17 relative |
| 5. Forward Euler: `E¹ − E⁰ = ½ δt² ‖Cq‖²_V ≥ 0` (`eq:momentum:euler-energy`) | tex 505-510 | ✅ verified | symbolic S4 (algebra) + package one centred Euler step (K5) | same | agreement at rounding level (2.6 ε E₀) |
| 6. Default `Centred()` is not skew on a composite mesh | tex 502-503 | ✅ verified | package K6 | `code_01_skew_periodic.jl` | skew defect 3.6 (relative to the largest entry) on composite/cut meshes vs 4.7e-16 on the uniform mesh; Lax–Wendroff at `δt = 0` with centred interface scheme equals the centred operator exactly (0.0) |
| 7. **With walls (boundary values zero)**: displayed identity and `2∂C_ff/∂q_f = div Υ` hold | tex 488 (parenthesis) | ❌ **error** for the printed `div Υ` | symbolic S3; package: walled composite meshes (uncut and with a sloped solid wall), `NoSlip`, `Centred`, `convection!` at `δt = 0` | `step01_skew_symbolic.jl` (S3), `code_02_skew_walls.jl` (W1, W2) | the printed right-hand side fails at wall cells: 847 of 10 036 unknowns, up to 0.83 of the largest entry of `V C`; **corrected** by the closed divergence `div Υ + V_f^{-1} Σ_{b∈B(f)} σ_b Υ_b`: deviation 5.3e-16 |
| 8. Off-diagonal part of `V C` is skew with walls | (new) | ✅ verified | package W1 | `code_02_skew_walls.jl` | 2.4e-16 (boundary dual faces touch only the diagonal when `q^∂ = 0`) |
| 9. Non-zero boundary values: extra term `½ Σ_f Σ_b σ_b Υ_b (q^∂_b ψ_f + ψ^∂_b q_f)` | (new) | ✅ verified | symbolic S3 + package W3 | same | 6.7e-17 |
| 10. Energy with walls for solenoidal `υ` | tex 495-498 | ⚠️ true outside the two wall limitations | package W4 | `code_02_skew_walls.jl` | energy rate `−½ Σ V q² θ_f` (θ the closed divergence): at the unknowns outside limitations (i)/(ii) 4.9e-18; the unknowns of (i)/(ii) carry a rate of 0.0079 relative to `scale·|q|²` (pre-fix), reported not asserted |
| 11. Citations: thesis App. 6.C (6.78)–(6.81); symmetry-preserving property (Verstappen–Veldman) | tex 498-501 | ✅ thesis numbers confirmed; ⚠️ reference unchecked | thesis text (`pt1_*.tex`, PDF text) | — | the thesis proves the continuous/semi-discrete analogue with the footnote "`K = A[m̄] − ½A[m̄]1` is skew-adjoint" (App. 6.C, footnote 2): the exact analogue of step 1 and worth citing; `[verstappen2003]` not opened |

## Errors and consequences

**F1 (step 7): the wall parenthesis is wrong as printed.** With walls the displayed right-hand side `Σ V q ψ (div Υ)` uses the interior dual divergence, but the boundary dual faces of `eq:momentum:boundary` (rate `½|f| υ_f` through the centre of a wall cell, state `½(q_f + q_b)`) add `σ_b Υ_b` to the diagonal: `2 V_f ∂(Cq)_f/∂q_f = V_f (div Υ)_f + Σ_b σ_b Υ_b`. The boundary term is of the size of the interior rates in a wall cell (not a rounding effect). Corrected statement: both displayed formulas hold with the **closed** dual divergence `θ_f = (div Υ)_f + V_f^{-1} Σ_{b∈B(f)} σ_b Υ_b`. Consequences:
- the energy rate of `dq/dt = −Cq` is `−½ Σ_f V_f q_f² θ_f` (equality), which is `0` exactly where the closed divergence vanishes: for a discretely solenoidal velocity at every unknown outside limitations (i), (ii) of the identity audit (not "on cut meshes with walls" unconditionally);
- the chapter's statement "on uniform, composite and cut meshes" is correct for periodic meshes;
- no consequence for the periodic identities, the Euler energy statement and the default `Centred()` remark.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Identity `V C + (V C)^T = diag(V div Υ)` | **sharp** (equality) | residual 6e-16 relative; no inequality | `code_01_skew_periodic.jl` | — |
| Arithmetic mean is the only two-point state | **sharp** (counterexample) | `α = 0.4`: off-diagonal symmetric part `(1−2α) Υ_g`, size of the largest entry | `sharp_01_arithmetic_mean.jl`, `step01_skew_symbolic.jl` S2 | state it (as in the replacement) |
| Requirement "`Centred` as both scheme and interface scheme" | **sharp** | default `Centred()` is not skew (3.6 relative) | `code_01_skew_periodic.jl` (K6) | — |
| Euler growth `½ δt² ‖Cq‖²` | **sharp** (equality, `div Υ = 0`) | exact algebra | S4, K5 | — |
| Walls: closed divergence | **sharp** | printed right-hand side fails by up to 0.83 of the largest entry | `code_02_skew_walls.jl` | replacement below |
| Conservation requires `div Υ = 0` at the unknown | **sharp** | rate `−½ V q² θ_f` | W4 | state the rate |

## Open points

- `[verstappen2003]` (Verstappen & Veldman 2003, symmetry-preserving discretisation) and the Veldman references of the thesis footnote were not checked against the literature; for a literature scout.
- With a solid, the boundary dual faces use uncut measures (issue #61): the closed divergence will then be built with apertures (see below).

## Effect of issues #60–#64

- **#60**, **#61** close limitations (i)/(ii): the energy statement then holds at every unknown of walled meshes (the `[excluded configuration]` rate 0.0079 becomes 0; the hard checks keep passing). After #61 the boundary terms `σ_b Υ_b` carry apertures; the replacement text is written in terms of the rates `Υ_b` and stays correct.
- **#62** (scheme selection on bounded/solid phase schedules): not relevant for the single-phase `convection!` statement.
- **#63**, **#64**: not relevant.

## Proposed replacement text

Replaces from `the state is the mean` up to `is not skew on a composite mesh.` in the paragraph *Kinetic energy* (label `eq:momentum:skew` kept; only existing macros).

```latex
the state is the mean
of the two unknowns and~\cref{eq:momentum:operator} gives, on a periodic mesh,
\begin{equation}\label{eq:momentum:skew}
  \ipstag{\psi}{\advectionstag{\velo}{\avarstag}}+\ipstag{\avarstag}{\advectionstag{\velo}{\psi}}
  =\sum_f\stagvol{f}\,\avarstag_f\psi_f\,(\divstag\dualrate{})_f,
  \qquad 2\,\frac{\partial(\advectionstag{\velo}{\avarstag})_f}{\partial\avarstag_f}=(\divstag\dualrate{})_f :
\end{equation}
the operator is skew-adjoint in $\ipstag{\cdot}{\cdot}$ up to half its dual
divergence on the diagonal. The arithmetic mean is the only two-point state
with this property: $\alpha\avarstag_{\dminus{g}}+(1-\alpha)\avarstag_{\dplus{g}}$
adds $(1-2\alpha)\dualrate{g}$ to the off-diagonal symmetric part of
$\stagvol{}\advectionstag{}{}$ for the pair $\dminus{g},\dplus{g}$. With walls
and vanishing boundary values $\avarstag^{\physbnd}_b$ the same identities hold
with the closed dual divergence
$(\divstag\dualrate{})_f+\stagvol{f}^{-1}\sum_{b\in\dualbnd{f}}\fside{b}\,\dualrate{b}$
in place of $(\divstag\dualrate{})_f$: the boundary dual
faces~\cref{eq:momentum:boundary} of a wall cell carry $\half\meas{f}\velo_f$
and add $\half\fside{b}\dualrate{b}$ to the diagonal, which the interior dual
faces miss (for non-vanishing boundary values the left side gains
$\half\sum_f\sum_{b\in\dualbnd{f}}\fside{b}\,\dualrate{b}\roundpar{\avarstag^{\physbnd}_b\psi_f+\psi^{\physbnd}_b\avarstag_f}$).
For a discretely solenoidal velocity the
semi-discrete system $\dd\avarstag/\dd\timevar=-\advectionstag{\velo}{\avarstag}$
therefore conserves $\tfrac12\ipstag{\avarstag}{\avarstag}$, on uniform,
composite and cut meshes and, with walls, at every unknown outside the two
configurations of \cref{sec:momentum:single} (the energy changes at the rate
$-\tfrac12\sum_f\stagvol{f}\avarstag_f^2\,\theta_f$, $\theta_f$ the closed dual
divergence); this is the conservation of quadratic invariants of
the thesis~\thesis{App.~6.C, eqs.~(6.78)--(6.81) and footnote~2}, where the Lax--Wendroff
state reduces to the centred one as $\dt\to0$, and the symmetry-preserving
property of Verstappen and Veldman~\cite{verstappen2003}. The default
\code{Centred()} uses \code{MonotonizedCentral} on non-regular dual faces and
is not skew on a composite mesh.
```

## History

- 2026-11-08: first audit against `ad10545`; every package-level check run on the clean tree (the foreign working-tree patch was removed before the final runs). Scripts: 1 symbolic (46 tests), 3 package (19 152 tests including the sharpness script). Outcome: periodic statement verified; error F1 (walls) with corrected statement; energy rate formula added.
