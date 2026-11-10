# Audit: eq:amr:vof, eq:amr:scalar — transfer of liquid fractions and intrinsic phase scalars

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:449-532` (`eq:amr:vof` l. 457, `eq:amr:scalar` l. 491), code `src/interface/interface_field.jl:627-750` (`InterfaceProlongation`, `_restrict_parent!`), `src/vof/scalar_transfer.jl:1-136` (`PhaseProlongation`, `_restrict_parent!`), commit `2959760`
- Date: 2026-11-10
- Overall status: ✅ verified (both claims); one unclaimed property measured (the shift of `eq:amr:scalar` does not preserve local bounds)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-vof/scripts/run_all.jl`

## Statement and assumptions

**Fractions** (`eq:amr:vof`): the liquid fraction of a refined cell's child $r$ is $\varphi_{c_r}=\mathrm{clamp}(4|R_r\cap L_c|,\,0,\,4|R_r\cap O_c|)$, $R_r$ the child's quadrant in the parent's local coordinates, $L_c$ the liquid half-plane of the parent's plane, $O_c$ the open part cut by the parent's solid half-plane; restriction is the mean. Claims: refinement conserves liquid volume "up to the accuracy with which the plane constant reproduces the fraction"; $\varphi_{c_r}\le$ the child's open fraction, so the conditional fraction stays in $[0,1]$; a pure cell gives pure children.

**Phase scalars** (`eq:amr:scalar`): weights $w^{\rm l}=\varphi$, $w^{\rm g}=\mathrm{open}-\varphi$; merge: weighted mean (plain mean if all weights vanish, the common value if all four equal); refine: limited linear values $\ell_{c_r}$ shifted by the constant $\big(4w_ca_c-\sum_sw_{c_s}\ell_{c_s}\big)/\sum_sw_{c_s}$ if $\sum_sw_{c_s}>\sqrt{\varepsilon}$ (children's weights before the snap), no shift if both slopes vanish. Claim: the phase amounts $|c|w a$ are conserved up to rounding (without the shift up to the volume defect) and up to the snap of the following reconstruction.

Implicit assumptions: the parent's stored fraction equals the area of its *own plane* (a planar cell; true after a reconstruction, to the plane-constant accuracy, but not for cells whose fraction cannot be matched by a plane, e.g. over-constrained triple cells); the snap acts after the transfer.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Restriction conserves $\sum|c|\varphi$ | l. 473-476 | ✅ | merge-only regrids: relative change $\le1.4\times10^{-16}$ (V1) | `code_01_vof_scalar.jl` | the mean of four equal volumes |
| Refinement conserves it up to the plane constant | l. 474-476 | ✅ | the plane defect $|{\rm area}(\text{plane}\cap\text{open})-\varphi_{c}|\le2.2\times10^{-16}$ on every mixed parent (independent Sutherland–Hodgman clipping of the stored plane, drops resting on the wavy solid included); refine-only regrids: relative volume change $\le2.4\times10^{-16}$ (V1) | same | the four quadrants partition the parent, so $\sum_r\frac14\cdot4|R_r\cap L|=|L|$ exactly |
| Children = $\mathrm{clamp}(4|R_r\cap L\cap O|,0,4|R_r\cap O|)$ | l. 457-460 | ✅ | independent polygon-clipping replica: $\le3.3\times10^{-16}$ (V2), snapped children excluded (none occurred in these fixtures) | same | |
| $0\le\varphi_{c_r}\le$ open fraction of the child | l. 466-472 | ✅ | violation $0.0$ on all children of 9 sequences (periodic, walled with solid, wetted drops with triple cells) (V2) | same | $L\subset O$, so the clamp is inactive except for rounding; it is the *new* geometry's open fraction that is compared |
| Pure parent gives pure children | l. 465 | ✅ | all 92 536 children of pure parents: conditional fraction exactly the parent's 0 or 1 (V2) | same | |
| `eq:amr:scalar` shift: $\sum_r|c_r|w_ra_r=|c|w_ca_c$ | l. 504-517 | ✅ | amount change per refine-only regrid $\le7.8\times10^{-16}$ (liquid and gas, with solid) (S1); replica of the formula on the halo-filled parent data in 21 808 families (families with snapped children are skipped; none occurred): $\le1.0\times10^{-13}$ (S2) | same | the large end ($10^{-13}$) is the amplification $1/\sum w$ when the weights are small, not a defect |
| Weighted-mean restriction | l. 498-501 | ✅ | replica: exactly $0.0$; amount change of merge-only regrids $\le4.3\times10^{-16}$; the merged value lies in the range of the children ($\le4.4\times10^{-16}$ violation, convexity) (S1, S2) | same | the merged weight is the mean of the children's weights ($\varphi$ is restricted by the mean), so $|c|w_ca_c=\sum|c_r|w_ra_r$ |
| Gas weight with a solid | l. 502-503 | ✅ | uses $\mathrm{clamp}(4|{\rm open\ polygon}|-\varphi_{c_r},0,1)$ as in the code; replica passes for 3 wetted-drop sequences | same | |
| Chapter's own check numbers | l. 535-541 | ✅ | reproduced bit-for-bit by `amr.jl` (`eq-amr-momentum/code_02`) | — | |

## Errors and consequences

None. Remarks:

- The conservation statement for the amounts holds "up to the snap" but *also* up to the plane-constant defect when the shift is not applied (stated l. 520-521). Fine.
- **Not claimed, measured**: the shift of `eq:amr:scalar` can push children outside the range of $\{a_c,$ four neighbours$\}$ (the unshifted limited values never leave it): up to $7\%$, $4\%$, $12\%$, $14\%$ of the stencil range on the random-noise intrinsic fields of the test (S2, "shift overshoot"). The chapter makes no boundedness claim for phase scalars (it does for cell fields, `eq:amr:prolong`), so this is not an error; a reader may assume it, so a sentence "the shift does not preserve local bounds" would be accurate.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Liquid volume conserved in refinement up to plane constant | **sharp** at round-off for planar cells ($2.2\times10^{-16}$); no case with a larger plane defect was found | V1 | `code_01_vof_scalar.jl` | a cell whose fraction is not reproduced by any plane (over-constrained triple cell) would give the defect; not found in 3 wetted-drop sequences |
| $\varphi\le$ open | **sharp** (equality only for liquid-filled children) | V2 | same | — |
| Phase amounts | **sharp** (round-off) | S1 | same | — |
| Boundedness of the shifted values | **not sharp / not available**: loses local bounds (≤14 % overshoot, noise data) | S2 | same | clip the shift per child to the stencil range and redistribute (would break the exact amount identity), or shift with weights $\propto w_{c_r}$ times the local range |

## Open points

- Snapped families never occurred in the random fixtures (the chapter's check has 8 snapped children out of 12 288); the "up to the snap" statement is therefore taken from the chapter's own measurement.
- The effect of the overshoot on advected scalars (e.g. temperature) is not assessed.

## History

- 2026-11-10: first audit (commit `2959760`): V1–V2, S1–S2 pass on 9 sequences; `run_all.jl` passes.
