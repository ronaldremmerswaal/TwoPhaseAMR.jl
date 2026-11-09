# Audit: eq:forces:aperture-density — gravity model: potential point, aperture density, integrated jumps (with eq:forces:potential, eq:forces:gravity-jump)

- Source: `docs/techref/chapters/08_forces.tex:399-439` (potential 403, aperture-density 415, gravity-jump 420) and deviation box 468-475; thesis (7.61), (7.64) (`pt2_gravity.tex`), commit `03bad7a`
- Code audited (normative): `src/two_phase/forces.jl` (`gravity!`), `src/interface/height_functions.jl` (`_interface_centroid`), `src/two_phase/two_velocity/transport_geometry.jl` (`_current_phase_face_apertures!`)
- Date: 2026-11-09
- Overall status: ✅ verified (three labels); the claim about the thesis force and the "pure phase gives $g\cdot e_k$" reduction hold
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-gravity/scripts/run_all.jl`

## Statement and assumptions

**`eq:forces:potential`.** $\phi_c=g\cdot\gpoint{c}$, $\gpoint{c}=x_c+\mathrm{cellscale}_c\circ\mathrm{segmid}_c$ for $0<C^{\rm l}_c<1$ (midpoint of the PLIC segment in the whole reference cell, not clipped by the solid, also in a triple cell), $=x_c$ otherwise (also in a pure cut cell).
**`eq:forces:aperture-density`.** $\bar\rho^{\rm a}_f=(\rho^{\rm l}a^{\rm l}_f+\rho^{\rm g}a^{\rm g}_f)/a_f$ ($a_f>0$), $a^{\rm l}+a^{\rm g}=a$.
**`eq:forces:gravity-jump`.** $\mathcal J^g_f=\bar\rho^{\rm a}_f\,|f|(\phi_{c^+}-\phi_{c^-})$, 0 for $a_f=0$; the force is $\mathsf R\mathcal J^g$. On a face in one phase $\bar\rho^{\rm a}_f$ is that phase's density, also when cut; without solid and on a conforming patch $(\mathsf R\mathcal J)_f=F_f/\bar\rho_f$ with the thesis' $F=(\rho^{\rm l}a^{\rm l}+\rho^{\rm g}a^{\rm g})\nabla_h(g\cdot x^I)$ (7.64); in a pure phase on a uniform mesh it is $g\cdot e_k$. A gravity component along a periodic direction is rejected.

**Assumptions made explicit.** The phase apertures are the package's (`eq:interface:phase-aperture`, audited); $V_f=|f|h$ on uniform meshes; the independent potential uses my own clipping of the reference square.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. `_interface_centroid(n,a)` = midpoint of $\{n\cdot\ell=a\}\cap[-\tfrac12,\tfrac12]^2$ | 403-410 | ✅ verified | 200 random lines (seeded) vs own segment-in-square midpoint | `code_01_potential_density_jumps.jl` (G1) | $2.3\times10^{-16}$ |
| 2. the potential vector the code forms ($=$ scratch `work.input`) equals $g\cdot(x_c+h\circ\mathrm{segmid})$ for $0<C<1$ and $g\cdot x_c$ otherwise (pure cut and triple cells included) | 403-410 | ✅ verified | 12 fixtures: drops (uniform, composite, cut), planes (uniform/composite × none/gas/liquid, oblique), gravity $(0.3,-0.9)$ for the drops | G2 | $\le2.2\times10^{-16}$ relative |
| 3. $a^{\rm l}+a^{\rm g}=a_f$ ($=1$ without solid) on every face | 413-420 | ✅ verified | package `_current_phase_face_apertures!` | G3 | $0.0$ |
| 4. on a face in one phase $\bar\rho^{\rm a}=$ that phase's density, also cut (113 cut one-phase faces over the 12 fixtures); always $\rho^{\rm g}\le\bar\rho^{\rm a}\le\rho^{\rm l}$ | 428-433 | ✅ verified | package | G3 | $1.1\times10^{-16}$ relative |
| 5. code jumps $=\bar\rho^{\rm a}|f|\,\mathcal D\phi$ (independent potential, package apertures), also with $g$ not normal to the interface (drop) | 420-427 | ✅ verified | independent evaluation on all faces of 12 fixtures | G4 | $\le2.1\times10^{-15}$ relative |
| 6. no solid: $\bar\rho^{\rm a}=\rho^{\rm l}a^{\rm l}+\rho^{\rm g}a^{\rm g}$, $a^{\rm l}+a^{\rm g}=1$; thesis (7.64) $\{\!\{a\rho\}\!\}=a^{\rm l}\rho^{\rm l}+a^{\rm g}\rho^{\rm g}$ | 434-438; thesis `pt2_gravity.tex` (7.64), the identity $\{\!\{a\rho\}\!\}=-[\rho]a^{\rm l}+\rho^{\rm g}$ | ✅ verified | G3 ($a^{\rm l}+a^{\rm g}=a=1$) and the thesis text | G3 | with a solid the thesis has no counterpart; the division by $a_f$ is the code's |
| 7. $(\mathsf R\mathcal J)_f=\mathcal J_f/(\bar\rho_fV_f)$ (uniform, no solid) and $=g\cdot e_k$ in a pure phase (839 faces, vertical and oblique $g$) | 436-438 | ✅ verified | package | G5 | $4.0\times10^{-16}$; $1.4\times10^{-15}$ |
| 8. gravity along a periodic direction rejected (`ArgumentError`), normal to it accepted | 438 | ✅ verified | package: doubly periodic in $x$, $g=(1,0)$ vs $(0,-1)$ | G6 | also model-level check in `src/two_phase/model.jl:34` |
| 9. thesis equation numbers: (7.53) $F=\{\!\{\chi\rho\}\!\}g$, (7.55) $p^\dagger$, (7.59) energy balance, (7.61) $x^I$, (7.64) MGM, (7.65) Lemma, (7.67) momentum | citations 399-467 | ✅ verified | thesis PDF text (`thesis.pdf`, `pdftotext`): all seven numbers exist with the quoted content | — | the chapter's "(7.61)" is the equation that approximates $\int x\,u_{\eta}\,dS\approx x^I\int u_\eta\,dS$ |

## Errors and consequences

No error. Remarks:
- The deviation box says the thesis uses "the centroid of a pure one"; the code uses the cell centre of a pure *cut* cell — any pure-cell point gives the same balance (`eq-forces-hydrostatic` step 11), but the potential-energy and momentum identities of the thesis (not claimed) need the thesis' choice.
- The line "Until issue 74 the code used … which put liquid next to a sloped solid out of equilibrium" is correct but only on composite meshes in the fixtures (one cut hanging face; on the uniform mesh all cut faces with a jump have $a_f=1$ up to rounding), see `eq-forces-hydrostatic` step 12.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| segment midpoint = `_interface_centroid` | **sharp** (identity) | $2.3\times10^{-16}$ | G1 | — |
| one-phase faces carry the phase density | **sharp** | 113 cut one-phase faces (12 fixtures) to $10^{-16}$; the pre-#74 formula fails (`eq-forces-hydrostatic`) | G3, O1 | — |
| $\rho^{\rm g}\le\bar\rho^{\rm a}\le\rho^{\rm l}$ (convexity of the mixture) | **sharp** (holds on all faces; equality on one-phase faces) | G3 | — | add as a remark |
| periodic rejection | **sharp** | $g=(1,0)$ on an $x$-periodic mesh throws | G6 | — |

## Open points

- `_current_phase_face_apertures!` is audited by `eq-interface-phase-aperture` (previous round); this audit takes it as given.

## History

- 2026-11-09: first audit (`code_01_potential_density_jumps.jl`). `run_all.jl`: PASS.
