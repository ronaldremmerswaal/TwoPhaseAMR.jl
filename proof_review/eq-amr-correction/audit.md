# Audit: eq:amr:correction, eq:amr:identities — divergence-preserving refinement of a face velocity

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:561-613` (`eq:amr:correction` l. 579, `eq:amr:identities` l. 599), code `src/mesh/adaptivity/field_transfer.jl:584-665` (`_preserve_refined_divergence!`), commit `2959760` (branch `docs/techref-amr`, chapter pinned to `b300a0f`)
- Date: 2026-11-10
- Overall status: ✅ verified (both claims; two wording remarks)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-correction/scripts/run_all.jl` (runs the symbolic script in `@claude-symbolics` and the code script in `docs/techref/figures`, which resolves `TwoPhaseAMR` to this worktree's `src/`; the worktree has no root `Manifest.toml`, so `--project=/tmp/techref-ch12` itself does not instantiate)

## Statement and assumptions

For a `FaceVelocity` (and each phase velocity), $\mathfrak F_c=|c|(\mathrm{div}_h u)_c$ is the net integrated outflow. For every cell $c$ of a refined block the children's defects $\delta_r=\tfrac14\mathfrak F_c-\mathfrak F^{\rm new}_{c_r}$ are removed by corrections $F_{12},F_{34},F_{13},F_{24}$ of the integrated fluxes of the four internal faces (children numbered low-low, high-low, low-high, high-high), given by `eq:amr:correction`.

Claims:

1. the four numbers solve the four-cell flux system $\Delta\mathfrak F_r=\delta_r$ and are its minimum-norm solution; $\sum_r\delta_r=0$ because the line map keeps the flux through the parent's sides;
2. `eq:amr:identities`: $\mathrm{div}_h$ is unchanged in kept cells, equals the parent value in the children of a refined cell, and is the mean of the four children in a merged cell; hence $\max_c|\mathrm{div}_h u|$ does not grow, and a solenoidal velocity stays solenoidal.

Implicit assumptions that should be (and mostly are) stated: Cartesian divergence (no solid apertures), the 2:1 graded forest of Ch. 2 (every new leaf is an old leaf, a child, or a parent of four old leaves), the data of *all* faces including the boundary faces enter (the identities are for arbitrary face values, no smoothness), the minimum norm is taken in the *integrated* flux $F_f$ (for rectangular children the velocity-norm minimiser is different, see Sharpness), and with a periodic integral correction the line-flux preservation no longer holds but the divergence is unchanged (claim 7, `eq-amr-momentum`).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Incidence of the four internal faces on the four children (outflow changes $F_{12}+F_{13}$, $-F_{12}+F_{24}$, $F_{34}-F_{13}$, $-F_{34}-F_{24}$) | l. 586-589 | ✅ | symbolic (S1) and by the package: the extra correction of the code is solved against this incidence (C1, $\lvert AF-\delta\rvert\le6\times10^{-17}$) | `sym_01_min_norm.jl`, `code_01_correction_identities.jl` | the signs follow the orientation $+k$ of the face velocity, outflow of the lower cell |
| The four numbers of `eq:amr:correction` solve $AF=\delta$ when $\delta_4=-\delta_1-\delta_2-\delta_3$ | l. 579-586, 590-592 | ✅ | Symbolics: `expand`/`simplify` of $AF-\delta$ is identically 0 (S1) | `sym_01_min_norm.jl` | the fourth equation is the only place where $\sum\delta_r=0$ is used |
| Null space of $A$ is the circulation $(1,-1,-1,1)$ (rank 3) | l. 592 ("differ by a circulation") | ✅ | exact (S2) | `sym_01_min_norm.jl` | |
| The chapter's $F$ is the minimum Euclidean-norm solution | l. 592-593 | ✅ | $F\cdot n=0$ for the null vector $n$ symbolically (S3); equals `pinv(A)*delta` to $2\times10^{-17}$ on 30 016 refined cells of random meshes (C1) | both | norm of the integrated fluxes; for unequal face areas the velocity-norm minimiser differs (S4, checked) |
| Code equals the chapter's formulas | `field_transfer.jl:632-647` | ✅ | line by line: `vertical_lower=(2r1+r3-r2)/4`, `vertical_upper=r3+r1-vl`, `horizontal_left=r1-vl`, `horizontal_right=r2+vl`; the face keys `(1,xmid,ylo,ymid)`, `(1,xmid,ymid,yhi)`, `(2,ymid,xlo,xmid)`, `(2,ymid,xmid,xhi)` are $f_{12}$, $f_{34}$, $f_{13}$, $f_{24}$; numerically: package correction minus chapter formula $\le7\times10^{-18}$ (C1) | `code_01_correction_identities.jl` | the correction is obtained as (corrected `FaceVelocity`) − (the same data in two `FaceField`s, which get no correction); all other faces differ by exactly 0 |
| $\sum_r\delta_r=0$ (flux through the sides of $c$ is kept by the line map) | l. 593-596 | ✅ | $\lvert\sum\delta_r\rvert\le10^{-12}\max\lvert\delta\rvert$ on every refined cell (C1); the preservation itself is C3 | `code_01_correction_identities.jl` | follows from `eq:amr:face-line` (see `eq-amr-face-line`) |
| `eq:amr:identities` (kept, children of refined, merged) | l. 597-613 | ✅ | random 2:1 graded meshes (walled, doubly periodic, periodic-in-$x$ channel), **random-noise face velocity including the boundary faces**, random labels, 6 regrids × 9 seeds, 30 016 refined cells: all three identities hold to $\le1.1\times10^{-15}$ relative to $\max\lvert D\rvert$ (C2) | `code_01_correction_identities.jl` | no solenoidality or smoothness used |
| $\max_c\lvert D u\rvert$ does not grow; solenoidal stays solenoidal | l. 610-613 | ✅ | relative growth $\le4.2\times10^{-16}$ (C2); the merged value is a mean, so $\le$ | `code_01_correction_identities.jl` | the chapter's own check (`chkAmrSolenoidal`) is the stream-function case |
| "kept and merged cells only see sides whose integrated fluxes are preserved; the corrections of different cells touch disjoint faces" | l. 609-610 | ✅ | integrated flux through every cluster of old/new line segments covering the same union (split halves; weighted-mean merge), computed on the lines independently of the package line machinery: $\le1.1\times10^{-16}$ over $5.3\times10^5$ surviving clusters (C3) | `code_01_correction_identities.jl` | see `eq-amr-face-line` for the line formulas |

## Errors and consequences

None. Wording remarks:

- l. 592-593: say "least Euclidean norm *in the integrated fluxes* $F_f$" (it is the norm of the vector $(F_{12},F_{34},F_{13},F_{24})$). For rectangular (anisotropic) children the areas of the two normals differ, and the minimiser of $\sum(F_f/\lvert f\rvert)^2$ is $F=WA^\top(AWA^\top)^+\delta$, which differs from the chapter's formula (symbolic/numeric check S4: equal for equal areas, different for areas $(1,1,2,2)$). The code is the integrated-flux version.
- l. 597-613: the three identities are *exact up to rounding for arbitrary face data*; the text could say so (it is what makes the check a random-noise check and not a smoothness one).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Minimum norm (in $F$) | **sharp** (exact) | KKT: $F\perp\ker A$, equals the pseudo-inverse; the only freedom is the circulation $t(1,-1,-1,1)$ | `sym_01_min_norm.jl` | the circulation could be fixed by another functional (e.g. minimum kinetic-energy change, minimum velocity norm for unequal areas, or minimum change of the discrete vorticity); not needed for the divergence |
| Identities (equalities) | **sharp** (equalities to round-off; the "no growth" is an inequality that is strict when the merged children differ) | 30 016 refined cells, noise data | `code_01_correction_identities.jl` | none for $\mathrm{div}_h$; the open (cut) and the two-velocity mixture divergence are *not* kept (Ch. 12 §divergence) |
| 2:1 grading, one level per regrid | necessary for the families being complete | the transfer throws otherwise | — | — |
| Cartesian divergence (no apertures) | necessary | open-divergence defects up to $2.9\times10^{-2}$ in the cut flow (`chkAmrFlowCutTransferred`, reproduced in `eq-amr-momentum/code_02`) | — | an aperture-weighted correction would be needed to keep $\mathrm{div}^{\rm o}$ |

## Open points

- None for these two claims. (Not audited here: the dependence of the faces' *orientation* convention on `mesh.faces` orientation `minus -> plus` for periodic seams, which the random periodic meshes exercise but which is not separately asserted.)

## History

- 2026-11-10: first audit (commit `2959760`). Symbolic algebra (11 tests), 60 132 code assertions on 30 016 refined cells of random meshes; `run_all.jl` passes.
