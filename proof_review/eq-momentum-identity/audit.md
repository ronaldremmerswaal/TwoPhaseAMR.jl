# Audit: eq:momentum:identity — weight rules (N)/(T) and the divergence identity of the dual faces

- Source: `docs/techref/chapters/06_momentum_transport.tex:109-228` (weights `eq:momentum:weights`, identity `eq:momentum:identity` at line 169, regularity `eq:momentum:regular`), wall limitations at 399-420 (`Limitations at walls`); commit `ad10545` (branch `docs/techref-momentum`, PR #66; working tree identical to the commit for `src/`)
- Code audited (normative): `src/operators/advection/staggered_schedules.jl` (`_amr_dual_schedule`, `_uniform_dual_schedule`, `_bounded_phase_schedule`, `_boundary_dual_faces`), `src/embedded/operators.jl` (`_cut_cell_dual_schedule`), `src/operators/advection/staggered_fluxes.jl` (`_add_boundary_convection!`)
- Date: 2026-11-08
- Overall status: ⚠️ identity and rules verified under the printed hypothesis; proof sentence incomplete (transverse part), two imprecise phrases, wall limitations (i)/(ii) exactly classified
- Rerun: `julia --project=@claude-symbolics proof_review/eq-momentum-identity/scripts/run_all.jl`
- Shared helpers: `proof_review/_shared/momentum_audit.jl` (package side), `proof_review/_shared/momentum_symbolic.jl` (symbolic patches); independent replica `docs/techref/figures/momentum.jl` (reused, not modified)

## Statement and assumptions

**Claim 1 (as printed).** The weights of the dual faces are the sums of rule (N) (`½ o_{c,f} ϖ_{c,f'}` on the flux of `f`, dual face `(f',f)`, for the other axial faces `f'` of each cell `c ∈ C(f)`) and rule (T) (`½ ϖ_{c⁻,f'} ϖ_{c⁺,f''}` on the flux of a transverse face, between faces of the same side `s` of its two cells), including hanging sides (`ϖ_{c,f} = ½` for a segment of a coarse side). For every unknown `f` whose cells have unknowns on both sides, `V_f (div_t Υ)_f = Σ_{c∈C(f)} (ϖ_{c,f}/2) Σ_{f'∈F(c)∩F_t} o_{c,f'} |f'| υ_{f'}`, `t = 1,2`; where `f` is the single face of its side in every cell this is `V_f` times the interpolated primal divergence (thesis Lemma 6.1, (6.13)).

**Assumptions made explicit.**
1. `υ` is an arbitrary (non-solenoidal) face velocity; the identity is linear in the fluxes `Φ_h = |h| υ_h` (with apertures already inside `υ` for cut meshes: the weights are multiplied by `α_f`).
2. The mesh is 2:1 balanced (used only in the lemma below), `ϖ ∈ {½, 1}`, side sums `S^±_c ∈ {0, 1}` (0 only on a domain wall: wall faces are not unknowns of the single-phase schedule; periodic faces are unknowns).
3. "Unknown" means an interior face of the schedule (single-phase). The phase schedule of a bounded domain or a solid additionally has the boundary faces as unknowns (volume zero) and then needs no boundary dual faces.
4. Notation of the audit: `S_c = Σ_{f'∈F(c)∩F_k} ϖ_{c,f'}` (covered fraction of both axial sides of `c`), `S^s_{c'}` the covered fraction of side `s` of `c'`, `T-complete` unknown: every cell `c'` across a transverse face of a cell of `f` has unknowns on the side that faces `f`.

**Exact residual (new, proved symbolically for arbitrary hanging partitions, and checked in every matrix entry).** For `f ∈ F_k`:

- axial part (`t = k`): `V_f (div_k Υ)_f − RHS_k = −½ Φ_f (S_{c⁻} − S_{c⁺})`, `c^∓ = C(f)` with `o_{c⁻,f} = +1`;
- transverse part (`t ≠ k`): `V_f (div_t Υ)_f − RHS_t = Σ_{c∈C(f)} Σ_{h∈F(c)∩F_t} ½ ϖ_{c,f} o_{c,h} Φ_h (S^{s(c)}_{c'(h)} − 1)`, `c'(h)` the cell across `h`, `s(c)` the side of `c` carrying `f`.

So the identity holds for all fluxes iff `S_{c⁻} = S_{c⁺}` and every `c'(h)` has the side facing `f` covered. The printed hypothesis (`S^±_c = 1` for both cells of `f`) is sufficient; **Lemma (2:1 balance)**: on a 2:1-balanced mesh the printed hypothesis implies T-completeness (a side of `c'` without unknown is a wall; if `c'` is not coarser than `c` then `f` lies on that wall, if `c'` is coarser the other cell of `f` spans from `f` to the wall by 2:1 balance and is itself a wall cell, contradicting the hypothesis). Checked on 48 400 (unknown, direction) rows.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. Rules (N), (T) reproduce the package's weights, including hanging sides | tex 110-123 | ✅ verified | independent construction from the incidence (`MA.rule_weights`) vs. the code's schedules, key by key, on 33 random 2:1 meshes (periodic / walled / mixed, 1–2 levels, with discs and a sloped wall); cut schedule times the aperture; phase schedules of bounded/embedded meshes with boundary unknowns | `code_01_rules_and_identity.jl` (C1) | gap **0** (exact) for the single-phase schedules and the phase schedules; also at T-incomplete unknowns (the rule itself, not the code, omits the missing partner) |
| 2. Hanging fine segment: 8 dual faces, `¼`, `⅛`, `V = 3/8` | tex 146-153 | ✅ verified | package schedule at the chapter's composite fixture | `code_01_rules_and_identity.jl` (C5) | matches `\chkMomCoarseLinks`, `\chkMomCoarseVolume` |
| 3. Regularity flag `eq:momentum:regular` = package `refinement_interface` | tex 189-196 | ✅ verified | 56 444 AMR dual faces (uncut weights and volumes, `64ε`) | `code_01_rules_and_identity.jl` (C6) | flag = negation of the printed regularity on every face outside T-incomplete unknowns |
| 4. Identity `eq:momentum:identity` under the printed hypothesis | tex 169-176 | ✅ verified | symbolic: axial and transverse patches with arbitrary partitions (S1–S3); numeric: `Q − P` entry by entry | `step01_identity_symbolic.jl` (180 tests), `code_01_rules_and_identity.jl` (C3) | 48 400 unknown-direction rows with the hypothesis: max residual 5.6e-17 (unit fluxes) |
| 5. Proof sentence "a primal flux enters the dual faces of its cells with weights that sum to ½ϖ o per unknown f and cancel in the dual faces between two unknowns of the same cell" | tex 177-181 | ⚠️ incomplete | symbolic residual (S1, S2) | `step01_identity_symbolic.jl` | true for fluxes of other faces; the flux of `f` itself needs `S_c = 2` **and** the cancellation `o_{c⁻,f} = −o_{c⁺,f}`; the transverse fluxes need the side of the neighbour `c'` to be covered (T-completeness), which the sentence does not mention |
| 6. 2:1 lemma: hypothesis ⟹ T-completeness | (not in the text) | ✅ numeric + geometric argument | `MA.t_complete` on every unknown satisfying the hypothesis | `code_01_rules_and_identity.jl` (C4) | needed so that the printed hypothesis is sufficient; used nowhere else |
| 7. Closed-form residual = package residual in every entry | (new) | ✅ verified | symbolic (S1, S2, S4, S5) + `Q − P − R̂ = 0` | both scripts | max entry deviation 5.6e-17 outside limitation (i); the closed form also explains the T-incomplete rows (deviation 0) |
| 8. Walls: boundary dual faces repair the axial part exactly | tex 399-405 | ✅ verified | symbolic S5; package with `NoSlip` walls and random non-solenoidal `υ` | `code_02_walls_and_operator_connection.jl` (A1) | every full-aperture wall-adjacent unknown misses exactly `½ Φ_f` without the boundary dual face (1 129 of 1 129 in the code_01 meshes) |
| 9. Limitation (i) (T-incomplete unknowns) and (ii) (cut wall cells with aperture < 1) are the only failures | tex 405-420 | ✅ verified (classification), ⚠️ exhaustiveness is a random search | package, 3 walled families, with a solid | `code_02_walls_and_operator_connection.jl` (A1–A3) | uncut walled meshes: 70 T-incomplete unknowns, **all** fail, nothing else fails; with a solid: 96 failing unknowns = T-incomplete ∪ cut wall cells (20 cut-wall failures); no third configuration found |
| 10. Operator connection `C(υ)1 = H(div^o υ)` where `f` is the single face of its side (condition (b)) | tex 182-188 | ✅ verified | package, periodic composite meshes with an interior solid | `code_02_walls_and_operator_connection.jl` (B1) | 5 432 unknowns with (b): residual 2.9e-16 relative to the rounding scale |
| 11. Where (b) fails the connection fails by `½ Σ_c (ϖ_{c,f}|c°| − |Λ_{c,f}∩c°|) (div° υ)_c` | tex 186-188 ("not related by 𝓗") | ✅ verified (exact formula; text only says "in general not") | package, 76 unknowns violating (b) | same (B1, reported) | 54 of 76 nonzero, formula deviation 2.9e-16; vanishes for `div° υ = 0` |

## Errors and consequences

No error in the identity itself. Precision defects in the text:

1. **Incomplete proof sentence (step 5).** The flux-bookkeeping sentence covers only the fluxes of the other faces and the "cancel between unknowns of the same cell" part; it omits (a) the cancellation of the own flux between the two cells of `f` and (b) that the transverse fluxes reach `f` only through the *neighbour's* side, which is exactly what fails at limitation (i). Consequence: none for the result (the hypothesis is sufficient on 2:1 meshes), but the reader cannot see why limitation (i) occurs; replacement below.
2. **"whose side is a single face"** (operator connection) should read "in which `f` is the single face of its side": condition (b) of `eq:momentum:mass-identity` is on the face `f`, not on the side. Consequence: wording only.
3. **"in general not related by 𝓗"** can be replaced by the exact difference (step 11); this is the quantity that appears in the mass defect of `eq:momentum:mass-identity` (audit `eq-momentum-mass-identity`).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Hypothesis "both cells of `f` have unknowns on both sides" | **sufficient, essentially necessary** (proof) | exact residual: axial part vanishes for all fluxes iff `S_{c⁻} = S_{c⁺}`; transverse part iff every neighbour side is covered. Equal defects at both cells (a face between two wall cells) give a vanishing axial residual although the hypothesis fails | `step01_identity_symbolic.jl` S1–S5 | state the exact condition instead: `S_{c⁻} = S_{c⁺}` and T-complete; or keep the hypothesis plus the 2:1 lemma |
| 2:1 balance | used only in the lemma hypothesis ⟹ T-completeness | argument above; counterexample needs a mesh with a 4:1 jump at a wall (not constructible with the package's trees, so unchecked) | `code_01_rules_and_identity.jl` (C4) | none needed |
| Weights ⟷ package | **exact** | gap 0 over 33 meshes | `code_01_rules_and_identity.jl` | — |
| Wall repair by boundary dual faces | **exact** for the axial part | residual `½ Φ_f` supplied by `Υ_b = ½ |f| υ_f` | `code_02_…` (A1) | — |
| Limitations (i), (ii) complete | **likely** (random search) | 70/70 and 96/96 classified, no extra failures | `code_02_…` | prove by the closed form: failures are exactly `S^s_{c'} = 0` (i) or aperture mismatch (ii) |

## Open points

- Exhaustiveness of "two configurations" is supported by random search over random walled meshes (3 families, 1–2 levels), not by proof; the closed forms (steps 4, 7) prove it for (i) in the uncut case.
- The exact wording of limitation (ii) depends on issue #61 (cut-wall measures): see below.
- Thesis (6.13) is stated for conforming meshes; the composite/cut version is the chapter's own and is verified only through its closed form here.

## Effect of issues #60–#64

- **#60** (transverse flux where a level transition meets a wall): closes limitation (i). The weights at T-incomplete unknowns will gain the missing flux terms, so `weight_gap` over T-incomplete dual faces (reported under `[excluded configuration]`, now 0) becomes non-zero by design; rules (N)/(T) in the text must then add the wall case, and the hard checks (which exclude those unknowns) keep passing.
- **#61** (uncut measures in the boundary dual faces of cut walls): closes limitation (ii); the closed divergence of audit `eq-momentum-skew` then carries apertures. No effect on the interior identity.
- **#62**: bounded/embedded phase schedules flag every dual face as non-regular; independent of the identity.
- **#63** (cut coarse cell with two segments): changes either the interpolation weights or the weights `ϖ`, which removes the exception of step 11 (re-run `code_02` B1: the reported violation count drops to 0).
- **#64**: unrelated.

## Proposed replacement text

Replaces the passage of the paragraph *Divergence identity* from `For every unknown $f$` up to `quantifies the effect.` (label `eq:momentum:identity` kept; only existing macros).

```latex
For every unknown $f\in\facesaxis{k}$ whose cells have unknowns on both sides
($\sum_{f'\in\sidefaces{s}{c}}\sidefrac{c}{f'}=1$ for $s=\pm1$ and
$c\in\cellset{f}$), the weights~\cref{eq:momentum:weights} give
\begin{equation}\label{eq:momentum:identity}
  \stagvol{f}\,(\divstag_{t}\dualrate{})_f
  =\sum_{c\in\cellset{f}}\frac{\sidefrac{c}{f}}{2}
    \sum_{f'\in\faceset{c}\cap\facesaxis{t}}\orient{c}{f'}\,\meas{f'}\volumeflux_{f'},
  \qquad t=1,2.
\end{equation}
For $t=k$ the flux of a face $f'\neq f$ of $c$ enters the dual face $(f',f)$
with weight $\half\sidefrac{c}{f}\orient{c}{f'}$, as on the right, while the
flux of $f$ itself contributes $-\half\orient{c}{f}(S_c-\sidefrac{c}{f})\meas{f}\volumeflux_f$
to the left side, $S_c=\sum_{f'\in\faceset{c}\cap\facesaxis{k}}\sidefrac{c}{f'}$; the
difference to the right side is $-\half S_c\orient{c}{f}\meas{f}\volumeflux_f$, which
cancels between the two cells of $f$ for $S_c=2$. For $t\neq k$ the flux of a
face $h$ of $c$ enters the dual faces between $f$ and the unknowns on the side
of the neighbour $c'$ across $h$ that faces $f$, with weights summing to
$\half\sidefrac{c}{f}\orient{c}{h}\sum_{f''}\sidefrac{c'}{f''}$, which is the
right side when that sum is one. On a $2{:}1$-balanced mesh the hypothesis
implies this: a side of $c'$ without unknown is a wall, so either $f$ lies on
that wall or $c'$ is coarser than $c$ and the other cell of $f$ reaches the
wall. Without the hypothesis the two sides differ by exactly the missing
terms: a wall cell ($S_c=1$) lacks $\half\orient{c}{f}\meas{f}\volumeflux_f$, which the
boundary dual face~\cref{eq:momentum:boundary} supplies, and a neighbour $c'$
without unknowns on its wall side lacks the flux terms of limitation~(i)
below. In a cell that is uncut or in which $f$ is the single face of its
side, $\half\sidefrac{c}{f}\meas{\opencell{c}}=\half\meas{\stripset{c}{f}\cap\opencell{c}}$,
so the right-hand side is $\stagvol{f}$ times the
interpolation~\cref{eq:momentum:interpolation} of the directional primal
divergence: the thesis' operator connection
$\divstag\interpolantflux=\interpolantsbp\divh$~\thesis{Lemma~6.1,
eq.~(6.13)}, on the composite mesh and per sweep direction. In a cut cell
whose side carries two segments the open part of the half strip is in general
not half the open cell: the right side then differs from
$\stagvol{f}(\interpolantsbp\divopen\velo)_f$ by
$\half\sum_{c}\roundpar{\sidefrac{c}{f}\meas{\opencell{c}}-\meas{\stripset{c}{f}\cap\opencell{c}}}(\divopen\velo)_c$,
which vanishes for $\divopen\velo=0$; \cref{sec:momentum:one-velocity}
quantifies the effect on the mass.
```

## History

- 2026-11-08: first audit against `ad10545` (clean tree for `src/`; an unrelated working-tree patch that touched `src/operators/advection/*`, `phase_face_transport.jl` and `embedded/operators.jl` for ≈10 minutes was removed before the final runs, and every package-level check was rerun on the clean tree). Scripts: 1 symbolic (180 tests), 2 package (205 296 + 7 + 8 tests). Outcome: identity and rules verified; precision defects 1–3; exact residual and 2:1 lemma added; wall limitations classified.
