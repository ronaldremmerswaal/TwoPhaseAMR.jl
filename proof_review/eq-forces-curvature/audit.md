# Audit: eq:forces:curvature — face curvature, its extension and the unresolved faces (eq:forces:extension, eq:forces:unresolved)

- Source: `docs/techref/chapters/08_forces.tex:214-307` (curvature 223, extension 232, unresolved 245, deviation 281), thesis §4.7.1; commit `03bad7a`
- Code audited (normative): `src/two_phase/forces.jl` (`_face_curvature`, `surface_tension!`)
- Date: 2026-11-09
- Overall status: ✅ verified (three labels): code = formulas on every face of six fixtures; the quoted counts are reproduced by an independent evaluation
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-curvature/scripts/run_all.jl`

## Statement and assumptions

**`eq:forces:curvature`.** $\varkappa_f=\sum_c w_c\varkappa_c/\sum_c w_c$ over the two cells of $f$ with $w_c=C_c(1-C_c)>0$ and $\varkappa_c$ finite, $C$ the conditional liquid fraction.
**`eq:forces:extension`.** If that set is empty: $\varkappa_f=\sum_{c'}\omega_{c'}\varkappa_{c'}/\sum\omega_{c'}$, $\omega_{c'}=w_{c'}/\max(|x_{c'}-x_f|_2^2,(h_{c'}/4)^2)$, over all leaf cells with $|x_{c'}-x_f|_2\le3\max_{c\in\{c^\pm\}}h_c+h_{c'}$ ($h=\max_k$ width), $w_{c'}>0$, $\varkappa_{c'}$ finite, nearest periodic image; the block prefilter changes nothing.
**`eq:forces:unresolved`.** $\mathcal F_{\rm unres}=\{f:\sigma>0,(\mathcal D\psi)_f\ne0,a_f>0,\text{no curvature}\}$, $\mathcal J^\sigma_f=0$ there, counted.

**Assumptions made explicit.** $C$ is `conditional_fraction`; "leaf cells in block and cell order" matters only for the order of floating-point sums.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. neighbours rule | 223-231 | ✅ verified | independent evaluation on **every** face of: drop uniform (480 faces), composite (1240), composite+solid (1240), thin film (112), doubly periodic composite drop straddling the seam (1296), composite drop with 60 % of the cell curvatures removed (1240) | `code_01_face_curvature.jl` (K1) | difference **0.0**; `nothing` exactly where the code returns `nothing` |
| 2. extension rule, radius $3\max h+h_{c'}$, weight with $(h_{c'}/4)^2$ floor | 232-243 | ✅ verified | same; nearest periodic image by explicit search over the 9 images (the code uses `_wrapped_delta`); no block prefilter | K1 | the prefilter is exact: 0.0 on all faces incl. the periodic and the thinned composite meshes |
| 3. classification counts | 258-276 | ✅ verified | K2: neighbours / extension / none over all faces: 107/326/47, 168/591/481 (composite and cut identical), film 0/0/112, periodic 121/431/744, thinned 72/584/584 | K2 | sums for the three drop fixtures: **2960** faces, **1508** extension, **1009** undefined = the numbers of the chapter (`chkForceAllFaces`, `chkForceAllExtension`, `chkForceAllUndefined`), 168 neighbour faces on the composite jump faces (`chkForceNeighbourFaces`) |
| 4. convexity: $\varkappa_f\in[\min,\max]$ of the cell values it averages (neighbours rule) | 223-231 | ✅ verified | K3 | K3 | holds on all fixtures |
| 5. unresolved set and zero jumps; count equals `surface_tension!`'s | 245-257 | ✅ verified | independent count for CSF and CGFM jumps; jumps on those faces are exactly 0 | K4 | thin film: CSF 9, CGFM **5** (= `chkForceFilmUnresolved`); thinned composite drop: CSF 1, CGFM 0; resolved drops: 0 |
| 6. $\sigma=0$: nothing interpolated and nothing counted | 258-262, 650-655 | ✅ verified | `unresolved_curvature_faces` $=0$ for $\sigma=0$ on all six fixtures | K4 | — |
| 7. radius is a hard limit (K5) | 232-243 | ✅ verified (behaviour) | a CSF jump face of the thinned drop with a valid cell at distance 1.01 and reach $3h+h=1.00$ is unresolved | K5 | the unresolved set depends discontinuously on the radius; not a defect of the text |
| 8. reset/sum/print of the counter | 262-275 | ⚠️ read, not run | code inspection: reset in the one- and two-velocity `step!`, summed in `simulate!`, field `unresolved_curvature_faces` of `FlowSimulation`, key `total_unresolved_curvature_faces` printed by `print_simulation_summary` | — | names agree with the text (`flow_simulation.jl:48,406-443`, `flow_diagnostics.jl:216-220`, `simulate.jl:138`, `stepping.jl`) |
| 9. thesis' weights $\alpha^{\rm l}\alpha^{\rm g}$ vs conditional fractions: "the same in uncut cells" | 281-290 | ✅ plausible | $C=\alpha$ in an uncut cell by definition (`eq:interface:fractions`) | — | the difference is in cut cells only |

## Errors and consequences

No error. The chapter's numbers (2960, 1508, 1009, 168, 5) are reproduced by an independent implementation.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| formulas = code | **sharp** (equality) | difference 0.0 on 5 608 faces (6 fixtures) | `code_01` | — |
| prefilter changes nothing | **sharp** | 0.0 on composite periodic and thinned meshes | `code_01` | — |
| "convex combination" property of the neighbours rule (not stated) | **sharp** | K3 | `code_01` | could be stated |
| extension radius $3h$ | **hard cut, not tuned** | one unresolved jump face at distance $1.01>1.00$ in the thinned test | K5 | a smooth weight (no cut) would remove the discontinuity; a design question, not an error |
| faces beyond the radius of every curvature are `none` | **sharp** | 481 of 1240 faces of the drop are `none` (they carry no jump) | K2 | — |

## Open points

- The print/accumulate chain of the counter (step 8) was read, not run; the existing tests (`test/two_phase/forces.jl`) cover it.

## History

- 2026-11-09: first audit (`code_01_face_curvature.jl`). `run_all.jl`: PASS (57 tests).
