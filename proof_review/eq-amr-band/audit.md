# Audit: eq:amr:band — the ring of halo slots and the issue #117 mechanism

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:82-146` (`eq:amr:band` l. 89; the #117 paragraph l. 106-116; `dev:amr:band` l. 122), measurement l. 786-790; code `src/mesh/adaptivity/phase_criteria.jl:38-79` (`_fixed_interface_refinement_labels!`), `docs/techref/figures/amr.jl:amr_band_cycle`, commit `2959760`
- Date: 2026-11-10
- Overall status: ⚠️ mechanism verified; the general statement "for a static interface the band has no fixed point" is too strong (it holds for some static interfaces only)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-band/scripts/run_all.jl`

## Statement and assumptions

A block $b$ is *near* iff, over its owned cells plus $w$ rings of halo slots (slots of the block's own level), the conditional liquid fraction has a value $<1-f_{\rm floor}$ and one $>f_{\rm floor}$; near blocks below the target level are refined, at the target kept, above coarsened, non-near blocks coarsened. Claims: (i) "the ring consists of halo slots of the block's own level, so a parent sees $w$ cells of twice the width of those its children see"; (ii) "a family whose children find no interface within their rings while their parent does is merged, refined again by the next attempt, and merged again when the children are `coarsen_delay` steps old: for a static interface the band has no fixed point"; (iii) the measurement: a static drop with $w=1$, `coarsen_delay=2` changes the mesh 8 times in 14 attempts with period `coarsen_delay`+1 = 3; with $w=0$ it does not change.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Package labels = rule `eq:amr:band` | l. 89-100 | ✅ | independent replica on the stored conditional fractions, rings 0, 1, 2 (halo 2), 3 drops, 3 regrids each: 252 labels equal (B1) | `code_01_band_cycle.jl` | the label function `_fixed_interface_refinement_labels!` slices `[h+1-r : h+nx+r]²` of `conditional_fraction.blocks`, i.e. the block's own halo |
| (i) ring width = $w$ cells of the block's own level | l. 106-110 | ✅ | by the code (halo slots have the block's cell width, `eq:mesh:halo-fill`) | — | |
| (iii) static drop, $w=1$, delay 2: 8 changes in attempts 3–14, period 3 | l. 786-790 | ✅ | independent history of the leaf sets (B2): changes 8, period 3; $w=0$: changes 0, period 1 | same | reproduces `chkAmrCycleChanges=8`, `chkAmrCyclePeriod=3` |
| (ii) mechanism | l. 111-116 | ✅ | for the 4 parents produced by a merge and labelled refine at the next attempt: the Chebyshev gap $g$ between the parent rectangle and the nearest mixed cell (independent geometry) is $0.5\,h_p=1\,h_c$ for all four, the parent block contains no mixed cell, and all four parents are in fact split again at the next attempt (B3); with $w=0$ no merged parent is ever split again (B3b) | same | the children's ring (width $h_c=h_p/2$) does not reach a mixed cell at distance $h_c$, the parent's ring ($h_p$) does |
| (ii) "for a static interface the band has no fixed point" | l. 115-116 | ❌ too strong | 12 random static drops and 12 random tilted planes per ring (`coarsen_delay`=2, target level 2): $w=1$: 3/12 drops and 7/12 planes have no fixed point (period 3), the others reach a fixed leaf set; $w=2$: 6/12 and 11/12; $w=0$: 0/12 and 0/12 (B5) | same | the mechanism needs a mixed cell at a gap $g$ with $w\,h_c\le g<w\,h_p$ from a whole family; the gaps are multiples of $h_c$ so this depends on the position of the interface relative to the block lattice |

## Errors and consequences

**F1 (overstatement).** l. 115-116 "for a static interface the band has no fixed point (\issue{117})" is not true for every static interface (B5): with $w=1$, 5 of 12 planes and 9 of 12 drops reach a fixed point; with $w=0$ the band always has one. The correct statement is that the band *can* have no fixed point, namely exactly when some family of children has no mixed cell within its rings while the parent block has one within its ring: the gap $g$ to the nearest mixed cell satisfies $w\,h_c\le g<w\,h_p=2w\,h_c$ for all children. Consequence: only this sentence (and the "(\issue{117})" tag) changes; the deviation box and the measurement paragraph (a specific drop, 8 changes in 14 attempts) are correct. Suggested LaTeX:

```latex
A family whose children find no interface within their rings while their parent
does (the nearest mixed cell lies at a distance $g$ from the block with
$\ringwidth\,h_c\le g<\ringwidth\,h_p$, $h_c=h_p/2$) is merged, refined again by
the next attempt, and merged again when the children are $\coarsendelay$ steps
old: such an interface has no fixed point of the band (\issue{117}); it
depends on the position of the interface relative to the block lattice
whether some family is in that situation, and for $\ringwidth=0$ none is.
```

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Existence of a cycle (mechanism) | **sharp** (necessary and sufficient pattern) | gap windows $[w\,h_c,w\,h_p)$; every cycling parent has $g=h_c$ for $w=1$ | `code_01_band_cycle.jl` | — |
| Period `coarsen_delay`+1 | **sharp for this fixture** | period 3 for delay 2 | same | other delays not run |
| $w=0$ has a fixed point | **sharp** (0 of 24 fixtures cycle) | B3b, B5 | same | — |
| Fix suggestion | an idea, not tested: apply the ring of the *children's* level to the parent (a ring of $2w$ cells at the parent level) or use hysteresis on the geometric distance | — | — | open |

## Open points

- A fix of the cycle (design decision) is outside this audit; issue #117 is open.
- The survey uses `coarsen_delay=2`; for `coarsen_delay=4` (the default of `MeshAdaptationOptions`) the period is 5 by the same mechanism (not run).

## History

- 2026-11-10: first audit (commit `2959760`): B1–B5 pass; `run_all.jl` passes. Finding F1.
