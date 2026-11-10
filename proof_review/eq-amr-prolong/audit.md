# Audit: eq:amr:prolong, eq:amr:mc (and eq:amr:restrict) — conservative limited-linear cell transfer

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:395-440` (`eq:amr:restrict` l. 403, `eq:amr:prolong` l. 406, `eq:amr:mc` l. 413), code `src/mesh/transfer.jl:_mc_slope`, `src/mesh/adaptivity/field_transfer.jl:_prolong_child!, _transfer_cell_field!`, commit `2959760`
- Date: 2026-11-10
- Overall status: ⚠️ verified; one precision remark (affine exactness at borders next to a coarser block and at walls) and one sharpness finding (the bound is attained)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-prolong/scripts/run_all.jl` (code script: `docs/techref/figures` project, worktree `src/`)

## Statement and assumptions

Merged cell: mean of the four children (`eq:amr:restrict`). Refined cell: child $r$ gets $a_c+\tfrac14\sum_k s^r_k\,\sigma_k$ with $\sigma_k=\mathrm{MC}(a_c-a_{c-e_k},a_{c+e_k}-a_c)$ and $\mathrm{MC}(a,b)=0$ for $ab\le0$, else $\mathrm{sgn}(a+b)\min\{\tfrac12|a+b|,2|a|,2|b|\}$, the neighbours on the parent block's lattice (halo slots `eq:mesh:halo-fill`, zero-gradient wall rule by default). Claims: (a) both maps conserve $\sum_c|c|a_c$ ("the offsets sum to zero"); (b) each child lies between the smallest and largest of $a_c$ and its four neighbours; (c) constants and, "whenever the two differences of each direction are equal, as on a uniform neighbourhood", affine data are reproduced exactly; at extrema the slope is zero and the reconstruction first order.

Implicit assumptions: the neighbour values are *those read through the halo*: a slot inside a coarser cell is a copy of the coarse value (piecewise constant, `eq:mesh:halo-fill`, ch. 2), and a wall slot copies the owned cell (zero gradient). The text says "uniform neighbourhood" but not that these two cases make the stencil non-affine.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Offsets $\pm\tfrac14\sigma_x\pm\tfrac14\sigma_y$ sum to zero over the four children; children have equal volume | l. 436-437 | ✅ | exact; $\int$ change $\le9.6\times10^{-17}\int|a|$ for noise data on 3 mesh types × 3 seeds × 6 random regrids (P2) | `code_01_cell_transfer.jl` | merged cells: mean, exact |
| Package = chapter's formula | `_prolong_child!`, `_mc_slope` | ✅ | independent replica on the halo-filled old data: children differ by exactly $0.0$, merged cells by $0.0$, kept blocks by $0.0$ (P3) | same | the replica reads the halo values from the package (halo fill is audited in ch. 2) |
| MC properties: sign, $\tfrac14|\sigma|\le\tfrac12\min(|a|,|b|)$, $\mathrm{MC}(a,a)=a$, $ab\le0\Rightarrow0$ | l. 438-440 | ✅ | 2 × 10⁶ random pairs spanning 13 decades (P1) | same | |
| Each child between min and max of $\{a_c,$ four neighbours$\}$ | l. 439-441 | ✅ | proved: child $-a_c\le\tfrac14(\sigma_x+\sigma_y)\le\tfrac12(b_x+b_y)\le\max(b_x,b_y)$ with $b$ the one-sided difference towards the extreme neighbour; 2 × 10⁶ random draws, 0 violations (P1), and 0 violations on 120 832 children of the random meshes (P4) | same | the text's reason ("$\tfrac14|\sigma_k|\le\tfrac12\min$") is per direction; the sum of the two directions needs the extra step $\tfrac12(b_x+b_y)\le\max(b_x,b_y)$ |
| Constants reproduced exactly | l. 441 | ✅ | $0$ error (P5) | same | |
| Affine data reproduced exactly "whenever the two differences of each direction are equal" | l. 441-442 | ✅ as stated, ⚠️ in application | $\mathrm{MC}(a,a)=a$ (P1). On random composite walled/periodic/channel meshes: exact to $0.0$ at all 8 732 … 20 112 children whose parent stencil is exact for the affine data through the halo; error $0.039$–$0.078$ at the remaining 1 000–2 100 children per mesh: stencils through a piecewise-constant coarse-to-fine halo slot, a zero-gradient wall slot or the periodic seam (P5) | same | the clause is true; it should say that the *differences are not equal* at a block border next to a coarser block and at a wall |
| "At extrema the slope is zero and the reconstruction first order" | l. 442-443 | ✅ (qualified) | RMS error of the prolongation of smooth periodic data (exact cell averages) $2.3\times10^{-2}\to2.5\times10^{-4}$ for $n=16\to128$: overall order 2.17; the max-norm error is $h^2u''/8$ at the extremum cells and shows no clean rate (alignment of the extrema with the grid), $0.047,0.0055,0.0057,0.0013$ (P6) | same | "first order" refers to the local reconstruction (slope zero); the effect on the value is $O(h^2)$ |
| Conservation of $\sum|c|a_c$ across the regrid | l. 434-438 | ✅ | as above | same | |
| Chapter's own check numbers ($\chkAmrCellReplica=0$, pressure integral $4.7\times10^{-15}$) | l. 445-449 | ✅ | reproduced bit-for-bit by re-running `amr.jl` (`eq-amr-momentum/code_02`) | — | |

## Errors and consequences

None. Precision remarks (not errors):

1. l. 426-428: after "as on a uniform neighbourhood" add that the one-sided differences are generally *unequal* in the layer of cells next to a coarser block (the halo slot is a piecewise-constant copy of the coarse cell) and at a wall with the zero-gradient rule; affine data are reproduced exactly only away from those layers. Suggested LaTeX:

```latex
reproduced exactly, and affine data whenever the two differences of each
direction are equal, as on a uniform neighbourhood; they are not equal in the
layer next to a coarser block (the halo slot copies the coarse cell,
\cref{eq:mesh:halo-fill}) nor next to a wall with the zero-gradient rule, where
the reconstruction is first order. At extrema the slope is zero and the
reconstruction first order.
```

2. l. 425-426: the proof of "each child lies between…" needs the final step $\tfrac12(b_x+b_y)\le\max(b_x,b_y)$ (the two offsets are added).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Child within $[\min,\max]$ of the five values | **sharp** | extremiser: neighbours $(-3,+1)$ in both directions, $\sigma=2$ in both, child$(+,+)=\tfrac14\cdot2+\tfrac14\cdot2=1=$ the largest neighbour; random search reaches child/extreme $=0.99999$ (P1) | `code_01_cell_transfer.jl` | none: the bound cannot be improved without changing the limiter (the factor 2 of MC is what lets a child reach the neighbour) |
| $\tfrac14|\sigma|\le\tfrac12\min$ | **sharp** | equality for $|a|\ge3|b|$ (MC $=2|b|$) | same | — |
| Conservation | **sharp** (exact to round-off) | $\le10^{-16}$ relative | same | — |
| Order of the reconstruction | likely sharp: 2 in the mean (RMS 2.17), alignment-dependent max-norm | numerical only | same | a non-limited central slope would be second order everywhere except that it can violate the bound (not provided) |
| Exactness for affine data | **sharp** (the equal-difference premise is necessary: where a piecewise-constant or zero-gradient halo slot enters the stencil the child error is $O(\lvert\nabla a\rvert h)$, measured $0.039$–$0.078$ for $\nabla a=(2,-3)$) | P5 | same | a halo that interpolates coarse-to-fine (instead of injecting) would remove the first-order layer at coarse–fine borders |

## Open points

- Coarse-to-fine halo injection makes the transfer first order in the layer next to a coarser block even for smooth data: not quantified here (the chapter's mesh-halo statement in ch. 2 says it). A bound on how much accuracy is lost in the global error is open.

## History

- 2026-11-10: first audit (commit `2959760`): P1–P6 pass; `run_all.jl` passes.
