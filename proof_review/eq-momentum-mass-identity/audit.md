# Audit: eq:momentum:mass-identity — mass–momentum consistency of one sweep, conditions (a) and (b)

- Source: `docs/techref/chapters/06_momentum_transport.tex:586-650` (conservative form at 621-626, identity at 628-631, conditions (a)/(b) at 633-640); commit `ad10545` (branch `docs/techref-momentum`, PR #66)
- Code audited (normative): `src/two_phase/momentum_workspace.jl` (`_advect_momentum_component!`), `src/vof/fluxes.jl` (`_advect_wy!`: clamp check at line 477, `128ε`), `src/two_phase/phase_face_transport.jl`, `src/embedded/operators.jl`
- Date: 2026-11-08
- Overall status: ⚠️ identity verified under (b) with the stored old density; (a) is vacuous as printed; the old state of the identity must be stated; (b) failure is exactly quantified (issue #63)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-momentum-mass-identity/scripts/run_all.jl`

## Statement and assumptions

**Claim 3 (as printed).** The conservative form `V ρ̄^{[j]} v^{[j]} = V ρ̄^{[j-1]} v^{[j-1]} − Σ_π ρ^π Σ_g σ_{f,g} δt Υ^π_g (F v)^π_g + V (H[Σ_π ρ^π χ^π σ^{[j]}])_f v^n_f` holds *provided* the staggered mass obeys
`V (ρ̄^{[j]} − ρ̄^{[j-1]}) = −Σ_π ρ^π Σ_g σ_{f,g} δt Υ^π_g + V (H[Σ_π ρ^π χ^π σ^{[j]}])_f` (`eq:momentum:mass-identity`). By `eq:momentum:identity` and the fraction update `eq:vof:sweep` this holds exactly up to rounding when **(a)** no fraction is clamped in the sweep and **(b)** every cell `c ∈ C(f)` is uncut or has the face `f` as the single face of its side, so that `|Λ_{c,f}∩c°| = ϖ_{c,f} |c°|`; where (b) fails the staggered mass drifts (issue #63).

**Assumptions made explicit.**
1. One directional sweep along `ξ` with the carrier volumes `Φ^o_h` of the sweep axis (`Φ^l_h + Φ^g_h = Φ^o_h` face by face, one-velocity model); phase schedule: boundary unknowns included for bounded/embedded meshes (no boundary dual faces needed); hanging sides with arbitrary partitions.
2. The hypothesis of `eq:momentum:identity` on `f` (both cells have unknowns on both sides, T-complete) holds in the phase schedule of every mesh because the boundary faces are unknowns there.
3. `ρ̄^{[j]}` is the density of the **predicted** fractions `H clamp(Č)` of `eq:momentum:predicted`; `ρ̄^{[j-1]}` is the density of the fractions **stored at the start of the sweep**, i.e. after the snap `eq:vof:snap` of the previous sweep (for `j = 1` after the previous step's snap). This is the reading under which the identity holds (F1).
4. `V_f ρ̄^{[j-1]}` uses the same `H` weights `½ |Λ_{c,f}∩c°|` as `ρ̄^{[j]}` (`eq:momentum:interpolation`); `σ^{[j]}` is the directional carrier dilation per open volume (`eq:momentum:sweep-dilation`) and `χ` the frozen indicator.

**Exact defect (new; symbolic for arbitrary hanging partitions and the axial and transverse components, verified in the package).** With `w_c = ½ |Λ_{c,f}∩c°|`, `μ_c = |c°|^{-1} Σ_π ρ^π Σ_{f'∈F(c)∩F_ξ} o_{c,f'} δt |f'| υ^π_{f'}` (divergence of the phase mass fluxes of `c` in the sweep) and `Δ_c = clamp(Č_c,0,1) − Č_c` (clamp change of the liquid fraction):

`(LHS − RHS) of eq:momentum:mass-identity = ½ Σ_{c∈C(f)} (ϖ_{c,f} |c°| − |Λ_{c,f}∩c°|) μ_c + Σ_{c∈C(f)} w_c (ρ^l − ρ^g) Δ_c.`

The dilation terms cancel exactly (`R = H[ρ^χ σ]` is the same on both sides).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. Conservative form follows from the fluctuation form `eq:momentum:one-velocity` given the mass identity | tex 621-627 | ✅ verified | package replay: the independent replica (docs `momentum.jl`) reproduces the code's velocity; exact bookkeeping per sweep | `code_01_mass_identity.jl` (M4a) | replica vs code 2.7e-13 relative; bookkeeping `Σ V ρ̄^{[j]}_{pred} v^{[j]} − Σ V ρ̄^{[j-1]}_{stored} v^{[j-1]} = Σ V d_f v^{[j-1]} + Σ V R v^n` holds to 1.8e-16 in all 60 fixtures |
| 2. Mass identity under (a) + (b) | tex 633-637 | ✅ verified | symbolic S2 (arbitrary hanging partitions, both components) + package M1 (60 random fixtures: periodic/walled, with discs and sloped walls, LaxWendroff/MC, both orders, snapping floors) | `step01_mass_identity_symbolic.jl` (25 tests), `code_01_mass_identity.jl` (93 488 tests) | 93 104 unknown-sweeps satisfy (a) and (b): defect ≤ 1.6e-12 ρ_l |
| 3. Exact defect formula | (new) | ✅ verified | symbolic S1 (axial rule (N) and transverse rule (T)); package M5 | same | package: formula deviation 1.6e-16 ρ_l over the 440 unknown-sweeps violating (b) |
| 4. (b) is necessary | tex 637-640 | ✅ verified | symbolic S3 (even a pure-liquid cut coarse cell: defect `−½(L − ϖO) ρ^l σ_c`); package M5 | same | 422 of 440 unknown-sweeps with (b) violated have a non-zero defect, up to 2.8 ρ_l per unit volume (tiny cells); reported (issue #63), never asserted |
| 5. (a) "no fraction is clamped in the sweep" | tex 634-635 | ⚠️ vacuous as printed | code reading + package M2 | `code_01_mass_identity.jl` | `_advect_wy!` stops with an error if a fraction leaves `[0,1]` by more than `128ε`, so `Δ_c ≤ 128ε` always; no unknown with (b) and a clamp occurred (`b_only = 0`), so the clamp term is verified symbolically (S1) and as zero in the package (clamp term 0.0) |
| 6. Old density of the identity | tex 628-631 | ❌ **wording** (F1) | symbolic S4 + package M3 (snaps forced with floor 0.06) | `step01_…` S4, `code_01_…` M3 | stored (snapped) fractions: defect 2.7e-16 (626 snap unknowns); the pre-snap predicted density of the previous sweep (the literal reading of `eq:momentum:predicted`): defect `−½ Σ_c L_c (ρ^l−ρ^g) snap_c`, non-zero at 626 of 626 snap unknowns, up to 0.042 ρ_l |
| 7. Snap changes `ρ̄` but not `v`: momentum change `V δρ̄ v` | tex 641-643 | ✅ verified | same replay (the old state of the next sweep is the snapped one) | `code_01_…` (M3, M4a) | consistent with the stored-state reading |
| 8. Momentum change `Σ_j Σ_f V (H[Σ ρ^π χ^π σ])_f v^n_f`; zero for a carrier solenoidal in every cell; conserved under (a), (b), no snapping, periodic or impermeable | tex 643-650 | ✅ verified | package M4 (10 fixtures with (a), (b), no snap) and M4c (9 solenoidal-carrier fixtures) | `code_01_…` | M4: momentum change equals the dilation sum to 5.9e-17; M4c: relative momentum change 3.3e-17 |
| 9. Thesis citations: mass–momentum consistency (6.45); exact conservation of the sum of the phase momenta (6.53) | tex 638-639, 649 | ✅ numbers confirmed in the thesis text | thesis PDF text | — | (6.53) defines `ū = {{αρu}}/{{αρ}}`; (6.54), (6.55) are the two-velocity updates |

## Errors and consequences

**F1 (step 6): the old state of the mass identity is not defined.** `eq:momentum:predicted` defines `ρ̄^{[j]}` from the *pre-snap* clamped predicted fractions; read literally, `ρ̄^{[j-1]}` of the next sweep is the same quantity of the previous sweep. The code (`field.conditional_fraction`) uses the *stored* fractions, which have been snapped between the sweeps. The two differ by `½ Σ_c L_c (ρ^l − ρ^g) snap_c`; with the literal reading the identity (and the momentum conservation) fails by up to 0.042 ρ_l in the snapped fixtures, with the stored reading it holds to 2.7e-16. Consequence: the text must say that `ρ̄^{[j-1]}` is the density of the stored fractions at the start of the sweep. The following snap sentence ("changes `ρ̄` but not `v`") is correct only in that reading.

**F2 (step 5): condition (a) is vacuous.** Clamps are bounded by `128ε` in the code, larger violations raise an error; so (a) holds up to rounding in every run that completes. The text should say "up to the clamp tolerance `128ε`" instead of treating (a) as a restriction that can be violated.

**F3 (steps 3, 4): the failure of (b) is exactly quantifiable.** The text only says "drift by the difference of the two weightings". The exact defect is the first sum of the formula above (`½ (ϖ|c°| − |Λ∩c°|) μ_c`), so the momentum drifts by `Σ_f V_f d_f v^{[j-1]}_f` per sweep (verified to 1.8e-16). For a pure cell `μ_c = ρ^π σ_c`, so the defect vanishes for a solenoidal carrier; it does not vanish in a mixed cut coarse cell with two segments, nor for a non-solenoidal carrier (symbolic S3: a pure-liquid cell gives `−½ (L − ϖ O) ρ^l σ_c`).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Condition (b) | **sharp** (necessary and sufficient in the cell-wise sense) | exact defect formula; pure-liquid cut coarse cell with two segments gives a non-zero defect; package 422/440 nonzero | `step01_…` S3, `code_01_…` M5 | none (cell-wise necessary; the defect can cancel between two cells of `f` only by coincidence) |
| Condition (a) | **not sharp** (vacuous) | clamp ≤ `128ε`; the clamp term is `Σ w_c (ρ^l−ρ^g) Δ_c` | `code_01_…` M2/M3 | replace by the tolerance statement |
| Old state = stored fractions | **sharp** | literal reading fails by up to 0.042 ρ_l | `code_01_…` M3 | state it |
| Momentum conservation (solenoidal carrier, (a), (b), no snap) | **sharp** (equality up to rounding) | 3.3e-17 | `code_01_…` M4c | — |
| "Carrier solenoidal in every cell" | needed | for non-solenoidal carriers the momentum changes by exactly the dilation sum (agreement 5.9e-17) | `code_01_…` M4 | — |

## Open points

- Whether the stored-fraction reading is the intended one for the `j = 1` sweep of the first step (no preceding sweep): the code uses the initial conditional fraction; the identity holds there (checked in every fixture).
- Condition (b) depends on #63; see below.

## Effect of issues #60–#64

- **#63** (cut coarse cell with two segments): the fix either changes the `H` weights or the `ϖ` of the dual weights so that `|Λ∩c°| = ϖ|c°|` is no longer required. Then the first sum vanishes, (b) can be dropped from the text, and the `[excluded configuration] M5` line of `code_01_mass_identity.jl` will report `nonzero = 0`; the hard checks keep passing.
- **#60/#61**: not involved (the phase schedule has boundary unknowns; no boundary dual faces).
- **#62**: the bulk branch of the face state changes (upwind → selected scheme on bounded/solid schedules); the identity involves only the volumes and is unaffected.
- **#64**: unrelated.

## Proposed replacement text

Replaces from `with the dilation of the split update. By~\cref{eq:momentum:identity} and the` up to `two weightings (\issue{63}).` (new label `eq:momentum:mass-defect`; only existing macros; the notation `\mu_c` is introduced inline).

```latex
with the dilation of the split update, where $\stagdens^{\splitlevel{j-1}}$ is
the density of the fractions stored at the start of the sweep (snapped,
\cref{eq:vof:snap}) and $\stagdens^{\splitlevel{j}}$ that of the predicted
fractions~\cref{eq:momentum:predicted}. By~\cref{eq:momentum:identity} for the
phase volumes and the fraction update~\cref{eq:vof:sweep}, the left side
of~\cref{eq:momentum:mass-identity} minus its right side is
\begin{equation}\label{eq:momentum:mass-defect}
  \half\sum_{c\in\cellset{f}}\roundpar{\sidefrac{c}{f}\meas{\opencell{c}}-\meas{\stripset{c}{f}\cap\opencell{c}}}\,\mu_c
  +\half\sum_{c\in\cellset{f}}\meas{\stripset{c}{f}\cap\opencell{c}}\,(\dens^{\liq}-\dens^{\gas})\,
    \roundpar{\operatorname{clamp}(\preclamp{\liq,\splitlevel{j}}{c},0,1)-\preclamp{\liq,\splitlevel{j}}{c}},
\end{equation}
with $\mu_c=\meas{\opencell{c}}^{-1}\sum_\phase\dens^{\phase}\sum_{f'\in\faceset{c}\cap\facesaxis{\sweepaxis{j}}}\orient{c}{f'}\,\dt\meas{f'}\volumeflux^{\phase}_{f'}$
the divergence of the phase mass fluxes of $c$ in the sweep (the dilation terms
cancel). So~\cref{eq:momentum:mass-identity} holds exactly, up to rounding,
when (a)~no fraction is clamped in the sweep (the code stops if one leaves
$[0,1]$ by more than $128\machineeps$, so the second sum is rounding) and
(b)~every cell $c\in\cellset{f}$ is uncut or has the face $f$ as the single
face of its side, so that $\meas{\stripset{c}{f}\cap\opencell{c}}=\sidefrac{c}{f}\meas{\opencell{c}}$
and the first sum vanishes.
Where (b) fails, in a cut coarse cell whose side carries two segments, the
staggered mass drifts from the cell mass by the first sum, and the momentum by
the sum over $f$ of $\velo^{\splitlevel{j-1}}_f$ times this defect (\issue{63}).
```

## History

- 2026-11-08: first audit against `ad10545`; all package-level checks run on the clean tree after removal of a foreign working-tree patch. Scripts: 1 symbolic (25 tests), 1 package (93 488 tests, 60 fixtures; M4c added). Outcome: identity verified under (a), (b) with the stored old density; F1–F3 with exact defect formula.
