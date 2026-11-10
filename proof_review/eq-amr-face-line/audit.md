# Audit: eq:amr:face-line, eq:amr:face-bracket — line transfer of face values

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:534-575` (`eq:amr:face-line` l. 545, `eq:amr:face-bracket` l. 557), code `src/mesh/adaptivity/field_transfer.jl:177-385` (`_build_face_transfer_geometry!`, `_tangential_slope`, `_line_value`, `_line_point_value`, `_face_transfer_value`), `src/mesh/topology/mesh.jl:349-410` (`_bracketing_line_segments`), commit `2959760`
- Date: 2026-11-10
- Overall status: ✅ verified (flux preservation, affine exactness, order)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-face-line/scripts/run_all.jl` (also reruns the line-flux check of `eq-amr-correction`)

## Statement and assumptions

Old faces are grouped into lines (faces of axis $k$ on one grid line, tangential order, positions on the finest lattice). A new face $f$ on an existing, fully covered line receives $u_g+s_g\,(\mathrm{pos}\,f-\mathrm{pos}\,g)\cdot t$ if $f\subseteq g$, the length-weighted mean $\frac1{|f|}\sum_{g\subset f}|g|u_g$ otherwise (`eq:amr:face-line`); any other new face (an internal face of a refined cell) is interpolated linearly between the nearest line coordinates below and above that have an old face containing the midpoint (`eq:amr:face-bracket`). The slope $s_g$ is the MC slope (`eq:amr:mc`) of the two difference quotients to the neighbours on the line (one-sided at an end, wrapped at a periodic seam, across gaps of the line).

Claims (4 of the task): (i) the integrated flux through each surviving line piece is preserved (halves of a split face carry offsets of equal length and opposite sign; a merge carries the length-weighted mean); (ii) the face transfer is exact for affine data.

Implicit assumptions: positions in lattice units (the face length is proportional to the lattice length along a line, so $|f|$-weights equal lattice-length weights); the slope uses the *face centres* as abscissae (the centre of a face is where an affine function's face average is attained); "surviving" means that old and new segments cover the same union on the line.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Code = chapter for the three branches | `_line_value`, `_line_point_value`, `_face_transfer_value` | ✅ | line by line (slope per lattice unit multiplied by the lattice offset of the centres; weighted mean over $g\subset f$; bracket value interpolated by lattice distance $-d_-/(d_+-d_-)$ with wrapping) | — | the face transfer value is also checked below numerically |
| Offsets of the two halves of a split face are opposite and of equal length, so $|f_1|u_{f_1}+|f_2|u_{f_2}=|g|u_g$; a merged face carries the weighted mean | l. 566-571 | ✅ | for the clusters of old/new segments covering the same union on every line: relative difference of the integrated flux $\le1.1\times10^{-16}$ on 9 seeds × 6 regrids of random walled, periodic and channel meshes with *random noise* face data, computed independently of the package line map (C3) | `../eq-amr-correction/scripts/code_01_correction_identities.jl` | requires no periodic integral correction (it shifts all faces by a constant and then preserves the divergence only, not the line fluxes); registered with `conserve_integral=false` in the test |
| "the integrated flux through every piece of an old line that survives is unchanged" | l. 569-572 | ✅ | same; surviving clusters are the ones with equal old and new coverage; the number of clusters is in the log (≥ 38 000 per case), the others (new internal faces of refined cells, vanished internal faces of merged cells) are by definition not surviving | same | |
| Every branch is exact for affine data (difference quotients are equal) | l. 572-574 | ✅ | all faces (interior and boundary, 83 736–91 456 per seed) of random walled meshes after 8 random regrids, affine data re-initialised before each regrid: $|u-\text{affine}|=0.0$ for `FaceField{1}`, `FaceField{2}` and `FaceVelocity` (F1) | `code_01_face_affine_order.jl` | for `FaceVelocity` the divergence correction also vanishes (affine data have constant divergence, the parent flux is exact) |
| Order for smooth data | not claimed | ✅ (measured) | RMS error at the new faces under uniform refinement $n=16\to128$: `FaceField` 2.17, divergence-corrected `FaceVelocity` 2.08 (F3) | same | the correction is $O(h^2)$ in the velocity: the child divergence is forced to the parent's, the true one differs by $O(h)$, so $\delta=O(h^3)$ in flux, $O(h^2)$ in velocity |

## Errors and consequences

None.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Flux through surviving pieces preserved | **sharp** (equality to round-off) | $\le1.1\times10^{-16}$ | `code_01_correction_identities.jl` (C3) | none |
| Affine exactness | **sharp** (exact, $0.0$) | F1 | `code_01_face_affine_order.jl` | quadratic data would need three-point slopes; the MC limiter caps exactness at affine data |
| Order | likely sharp: 2 in the mean | numerical, 2.17 / 2.08 | same | — |
| Dependence on the limiter | the MC slope is first order at tangential extrema of the face data | not separately measured | — | — |

## Open points

- The face prolongation is not energy-preserving: the kinetic-energy change per regrid is second order (see `eq-amr-momentum`).
- Not exercised separately: the "one point value that exists" fallback of `eq:amr:face-bracket` (only one of the two bracketing lines has an old face containing the midpoint); it is reached next to a wall or a gap of the lines and covered by the random meshes only implicitly through the identities.

## History

- 2026-11-10: first audit (commit `2959760`): F1, F3 pass; line-flux check passes; `run_all.jl` passes.
