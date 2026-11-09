# Audit: eq:momentum:ctu — the modified flux interpolant (term-wise upwinding) and its CTU interpretation

- Source: `docs/techref/chapters/06_momentum_transport.tex:652-690` (equation at 660-665; interpretation sentence at 674-679), deviation box 700-712; commit `ad10545` (branch `docs/techref-momentum`, PR #66); thesis `pt1_momentum.tex` (6.49)–(6.51), Lemmas 6.2, 6.3 (App. 6.B)
- Code audited (normative): `src/two_phase/phase_face_transport.jl` (`_phase_transport`, `_ctu_phase_transport`, `_phase_bulk_state`), `src/two_phase/momentum_workspace.jl` (`_advect_momentum_component!`), `src/vof/workspace.jl` (`ctu_threshold ∈ (0,1]`, default ½)
- Date: 2026-11-08
- Overall status: ❌ the formula is correct, the **interpretation and the boundedness claim are not**: nothing "bypasses" a control volume, and the update of a draining control volume is not bounded at hanging segments (value 1.95 from data in `[0,1]` at density ratio 1000)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-momentum-ctu/scripts/run_all.jl`

## Statement and assumptions

**Claim 5 (as printed).** `δt Υ^π_g (F q)^π_g = Σ_f w_{g,f} δt|f| υ^π_f q_{u(g,f)}` if `min(Č^π_{g⁻}, Č^π_{g⁺}) < θ`, else `δt Υ^π_g (F q)_g` (bulk state of the selected scheme, no compressive term, Courant number `|δtΥ^π_g|/min V`), with `u(g,f)` the tail if the partial volume `w_{g,f} δt|f|υ^π_f ≥ 0` and the head otherwise. "Keeping the partial interpolated volumes separate lets fluid in transit through a nearly empty control volume, entering through one primal face and leaving through another, bypass its value: the CTU flux of the thesis (6.49)–(6.51), whose switch bounds the update of a draining control volume (Lemmas 6.2 and 6.3)."

**Assumptions made explicit.**
1. Per sweep, per phase `π`; `p_i = w_{g,i} Φ^π_i` partial volumes of the dual face (oriented tail → head); `P = Σ_{p_i>0} p_i`, `N = Σ_{p_i<0} |p_i|`; `O_f`, `I_f` total (density-weighted) partial volumes leaving / entering the control volume of `f`.
2. The update is the fluctuation form `eq:momentum:one-velocity`; with the mass identity (audit `eq-momentum-mass-identity`, condition (b)) `V ρ^new q^new = (V ρ^old − O_f) q_f + Σ_{in} |p_i| q_nb + V R q^n`, with `R = H[ρ^χ σ]` the frozen dilation (zero for solenoidal carriers).
3. Own-value coefficient `γ_f = (V ρ^old − O_f)/(V ρ^new) = 1 − (I_f + V R)/(V ρ^new)`; the update is a convex combination iff `γ_f ≥ 0` (and `R = 0`).
4. "Draining control volume": all dual faces of `f` use the CTU branch (new predicted fraction `< θ`).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. `eq:momentum:ctu` (both branches, `θ`, upwind choice by the sign of each partial volume) equals the code | tex 660-669 | ✅ verified | independent evaluation (replica of `docs/techref/figures/momentum.jl` schedules, states, upstream unknowns) vs `_phase_transport` for every dual face of uniform/composite/walled/cut phase schedules, random signed primal volumes, random values, fractions around `θ = ½`, 6 schemes | `code_01_ctu_formula.jl` | CTU branch: 178 632 faces, relative difference **0.0** (bitwise); bulk branch on periodic schedules: ≤ 7.1e-13 (limiter ratios amplify rounding) |
| 2. `ctu_threshold ∈ (0,1]`, default ½ | tex 669 | ✅ by code inspection | `WYAdvectionWorkspace` validation | — | matches |
| 3. Bulk state is "the bulk state of the selected scheme" | tex 655-657 | ⚠️ true on periodic schedules only (issue #62) | same replica; bounded/embedded schedules flag every dual face non-regular | `code_01_ctu_formula.jl` (reported) | on bounded/solid schedules the package equals the replica that treats every dual face as non-regular (upwind in the bulk), 52 992 of 52 992 evaluations; the chapter's later paragraph already says this |
| 4. Term-wise upwinding = net upwinding + counter-flow exchange: `CTU − net = −min(P,N)(q_head − q_tail)` | (new) | ✅ verified | symbolic S1 (all sign patterns); package: dual faces with opposite-sign partial volumes | `step01_ctu_symbolic.jl` (341 tests), `code_01_ctu_formula.jl` | 76 338 CTU dual faces have opposite-sign partial volumes; the two fluxes differ only there, by a diffusive exchange `min(P,N)` between the two unknowns |
| 5. "lets fluid in transit … bypass its value" | tex 674-677 | ❌ **not what the code does** | symbolic S2/S3: own flux of an unknown enters at one dual face with the upwind neighbour's value, leaves at another with `q_f` | `step01_ctu_symbolic.jl` | in this per-dual-face restriction nothing bypasses the control volume; the only effect is step 4 |
| 6. "the CTU flux of the thesis (6.49)–(6.51)" | tex 677-678 | ⚠️ restriction of it | thesis text (6.49)–(6.51), Lemmas 6.2/6.3 | — | thesis: partial volumes attributed to donor cells `b̂(g,h,k)` over `2×3^{d-1}` faces and unsplit step; code: two unknowns per dual face, per directional sweep (acknowledged in the deviation box, but not in the interpretation sentence) |
| 7. Closed forms `O_f = Σ_c (ϖ_{c,f}/2) out_c + E_f`, `I_f = Σ_c (ϖ_{c,f}/2) in_c + E_f`, `E_f = (2 − ϖ⁻ − ϖ⁺)/2 |Φ_f|` along the sweep, 0 across | (new) | ✅ verified | symbolic S3 (arbitrary hanging partitions, every sign pattern); package P1 on uniform/composite/walled/cut phase schedules | `step01_…`, `code_02_ctu_boundedness.jl` (P1) | package deviation 2.2e-16 relative to the largest flux; net dual volumes never exceed the partial ones |
| 8. Own-value coefficient `γ_f = (Vρ^old − O_f)/(Vρ^new) = 1 − (I_f + VR)/(Vρ^new)` | (new) | ✅ verified | symbolic S2; package P2 on random one-velocity sweeps | `code_02_…` (P2) | identity gap 1.7e-10 (relative scaling by `1/ρ^new`) |
| 9. Boundedness "whose switch bounds the update of a draining control volume" on conforming patches | tex 678-679 | ⚠️ true there, but not because of the switch | package P3: all conforming CTU unknowns | `code_02_…` (P3) | 37 264 conforming CTU unknowns, `γ_f ≥ −2.1e-13` (rounding); it follows from the cell-wise outflow bound `out^π_c ≤` phase volume of `c` (with `E_f = 0`, `O_f = Σ ½ out_c ≤ V ρ^old`), which the WY update provides independently of CTU |
| 10. Boundedness at hanging segments | tex 678-679 | ❌ **fails** | symbolic S4: only the own flux non-zero, donor phase inside its donating rectangle, receiver empty: `γ_f = −(2−ϖ⁻−ϖ⁺)/ϖ⁺`; package weights (P4); package staggered phase scalar (P5); package one-velocity momentum (P5m) | `sharp_01_hanging_overshoot.jl`, `code_02_…` (P5a) | `γ_f = −½` (coarse → fine), `−1` (fine → coarse). Scalar: new fraction 0.054, `γ = −0.981`, value `1.981` from data in `[0,1]`; `γ = −0.5`, value `1.5`. **Momentum, densities (1000, 1)**: value `1.952` (range of all values `[0.026, 1.952]`) and `1.369`; densities (1,1) and (10,1): range stays in `[0,1]` (0.386/0.337 and 0.906/0.414). Random sweeps: 1 negative of 5 372 hanging CTU unknowns (min −0.46): a rare, constructible event |
| 11. CTU never has a larger own coefficient than net upwinding (adds diffusion, no convexity) | (new) | ✅ verified | package P6 | `code_02_…` | `γ^CTU ≤ γ^net` in all 42 636 CTU unknowns, strictly smaller in 3 138 |
| 12. Thesis lemmas' hypotheses hold for the code | tex 678-679 | ⚠️ not verified | — | — | the lemmas assume the thesis' donor-cell attribution (no flux overlap); not re-derived for the directional two-unknown form |

## Errors and consequences

**E1 (step 5): the interpretation is wrong for the code's form.** In the per-dual-face restriction the own flux of an unknown `f` enters its control volume at one dual face carrying the value of the upwind neighbour and leaves at another carrying `q_f`; nothing passes through without meeting `q_f`. What term-wise upwinding changes is only where partial volumes of opposite sign meet in one dual face (step 4): a diffusive counter-flow exchange `min(P,N)(q_tail − q_head)`.

**E2 (steps 9, 10): the boundedness claim is true only on conforming patches and not because of the switch.** On a conforming patch the volume leaving `f` is half the volume leaving each of its two cells, hence at most the phase volume in `f` (cell-wise bound), so `γ_f ≥ 0` for any state choice. At a hanging segment an additional `½(2 − ϖ⁻ − ϖ⁺)|Φ_f|` leaves (and the same enters), the coefficient reaches `−½` (coarse → fine) or `−1` (fine → coarse), and the update leaves the range of the old values. Consequences:
- the thesis lemmas 6.2/6.3 are not transferred to hanging sides; in the one-velocity momentum update at density ratio 1000 a velocity `0` among neighbours `1` becomes `1.95` in one step (no such effect at ratios 1 and 10 in the same configuration);
- the overshoot is independent of issues #60–#64;
- a code comment in `src/two_phase/phase_face_transport.jl` (`# Keeping the partial terms separate cancels fluid in transit`) repeats the interpretation of E1 and should be corrected with the text.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| `γ_f ≥ 0` on conforming patches | **sharp** (attained 0) | extremal state (own flux only, receiver empty): `γ = 0`; `min γ = −2.1e-13` in 37 264 random cases | `sharp_01_…` (P4), `code_02_…` (P3) | attribute it to the cell-wise outflow bound |
| `γ_f ≥ −(2−ϖ⁻−ϖ⁺)/ϖ⁺` at hanging segments | **sharp** (attained) | `−½` and `−1` reproduced through the package weights and by `advect!` (scalar and momentum) | `sharp_01_…` (P4, P5, P5m) | replace the boundedness claim by the closed form |
| CTU vs net upwinding | **CTU is never better** | `γ^net ≤ 0` in the extremal states too; `γ^CTU ≤ γ^net` always | `code_02_…` (P6), `sharp_01_…` (P4) | state "adds diffusion, no convexity" |
| Density dependence of the overshoot | observed, not proved | ratio 1000: 1.95 and 1.37; ratios 1, 10: inside `[0,1]` | `sharp_01_…` (P5m) | quantify the weight of the liquid part: `ρ^l(Vα^l − O^l) + ρ^g(Vα^g − O^g)` |
| `θ = ½` default | unknown | the switch only selects the branch; boundedness of the bulk branch is not claimed | — | — |

## Open points

- The hypotheses of the thesis Lemmas 6.2/6.3 (App. 6.B) were read, but not re-proved for the directional two-unknown form; whether a limiter of the extra `E_f` at hanging segments restores boundedness is a design question for the code owner.
- The statement that term-wise upwinding is "the CTU flux of the thesis" could be checked against the thesis' unsplit derivation (6.49)–(6.51) in more detail by a literature/thesis scout.

## Effect of issues #60–#64

- **#62** (scheme selection on bounded/solid phase schedules): the bulk branch of `eq:momentum:ctu` will then use the selected scheme there; the reported line of `code_01_ctu_formula.jl` (52 992 of 52 992 agreeing with the all-non-regular replica) will drop, the hard checks (CTU branch; periodic bulk) are unaffected.
- **#60, #61, #63, #64**: not involved. The hanging-segment overshoot (E2) is independent of all five fixes.

## Proposed replacement text

Replaces from `Keeping the partial interpolated volumes separate lets` up to `and 6.3}.` (only existing macros).

```latex
Keeping the partial interpolated volumes separate upwinds each of them with
the value of its own upwind unknown. Where partial volumes of opposite sign
meet at a dual face this equals upwinding the net volume plus a counter-flow
exchange $\min(P,N)\,(\avarstag_{\dminus{g}}-\avarstag_{\dplus{g}})$ between its two
unknowns, $P$ and $N$ the sums of the positive and of the negative partial
volumes, and otherwise it is the upwind flux. This is the
corner-transport-upwind (CTU) flux of the thesis~\thesis{eqs.~(6.49)--(6.51)}
restricted to the two unknowns of the dual face (the thesis attributes partial
volumes to their donor cells over $2\times3^{d-1}$ faces): the own flux of an
unknown enters its control volume at one dual face with the value of the
upwind neighbour and leaves at another with its own value, so no fluid
bypasses the control volume. On a conforming patch the volume of a phase
leaving a control volume is half the volume leaving each of its two cells, hence
at most the volume of that phase in it, and the coefficient of the old value in
the update stays non-negative: the update of a draining control volume is
bounded by the cell-wise outflow bound, as in the thesis
lemmas~\thesis{Lemmas~6.2 and 6.3}, and not by the switch. At a hanging
segment an extra $\half(2-\sidefrac{\cminus{f}}{f}-\sidefrac{\cplus{f}}{f})\abs{\dt\meas{f}\volumeflux^{\phase}_f}$
leaves and the coefficient reaches $-\half$ (coarse to fine) or $-1$ (fine to
coarse), so the lemmas do not carry over there: a staggered velocity $0$ among
neighbours $1$ became $1.95$ in one step at the density ratio $1000$.
```

## History

- 2026-11-08: first audit against `ad10545`; all package-level checks run on the clean tree (the foreign working-tree patch was removed before the final runs and the final `run_all` was rerun after the last script edit). Scripts: 1 symbolic (341 tests), 3 package (367 383 tests). Outcome: formula verified (bitwise); interpretation (E1) and boundedness claim (E2) wrong at hanging sides, with a momentum-level demonstration.
