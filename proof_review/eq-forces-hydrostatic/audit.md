# Audit: eq:forces:hydrostatic — hydrostatic balance of the mimetic gravity model (planar interface, with and without solid)

- Source: `docs/techref/chapters/08_forces.tex:399-467` (potential 403, aperture-density 415, gravity-jump 420, hydrostatic 440, deviation box 468-475); thesis Lemma 7.1 (7.65), App. 7.A (`pt2_gravity.tex`), commit `03bad7a`
- Code audited (normative): `src/two_phase/forces.jl` (`gravity!`), `src/two_phase/two_velocity/transport_geometry.jl` (`_current_phase_face_apertures!`), `src/interface/height_functions.jl` (`_interface_centroid`)
- Date: 2026-11-09
- Overall status: ⚠️ the iff statement and the planar-state claim verified with and without solid; the claim is **conditional on exact PLIC reproduction of the plane**, which fails near a solid/wall for oblique planes and in triple cells (F1–F3); the text's checks sentence over-reads the oblique case (F2); the proof for the solid case, "not proved here", can be given (replacement)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-hydrostatic/scripts/run_all.jl`

## Statement and assumptions

**Claim.** (a) By eq:forces:balance, gravity is balanced iff there is a cell function $a$ with $\bar\rho^{\rm a}_f(\mathcal D\phi)_f=(\mathcal Da)_f$ for every face with $\gamma_f>0$; the condition is face by face, does not depend on the metric $\mathsf K$ and holds at level transitions whenever it holds. (b) For a planar interface normal to $g$ that the PLIC reproduces exactly and without solid, $a=p^\dagger$ (thesis Lemma 7.1, $p^\dagger=-\{\!\{\chi\rho\}\!\}|g|q(x^I)$). (c) With a solid the aperture density of a cut one-phase face is that phase's density, "so the same state is balanced also next to a sloped solid in either phase (checks below; not proved here for every cut configuration)". (d) Deviation box: the balance condition does not depend on the choice of the potential point in cut cells.

**Assumptions made explicit** (Lemma 7.1, case analysis of App. 7.A): (A1) every mixed cell has its potential point on the plane ($q(x^I_c)=0$) — i.e. the PLIC line of $c$ **is** the plane; (A2) a face with a pure neighbour lies in one phase (phase aperture $\in\{0,a\}$), so $\bar\rho^{\rm a}_f$ equals that phase's density (the thesis uses $a^\pi_f\in\{0,1\}$; with a solid the code needs the open-face mixture eq:forces:aperture-density); (A3) a face between a pure liquid and a pure gas cell (interface on a grid line, **not covered by the thesis' case analysis**, which assumes adjacent pure cells have $a^\pi,\chi_{c_1},\chi_{c_2}$ all 0 or all 1) has phase apertures ½, ½; the balance then holds, and $a=p^\dagger$ if the two cells have the same width normal to the face.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. iff, face by face, metric-independent | 439-447 | ✅ verified | $\mathsf C\mathsf K\mathsf C|_{\gamma>0}$ SPD (injectivity) — see `eq-forces-balance` step 5; $\mathsf K$ symmetric, smallest eigenvalue $0.0038$ | `../eq-forces-balance/scripts/code_03_iff_injectivity.jl` | holds on uniform, composite, solid fixtures |
| 2. Vertical $g$, planar interface $y=4.6$: explicit $a_c=\rho^\pi g\cdot(x_c-x_0)$ (pure), 0 (mixed) satisfies the condition, solid in neither / gas / liquid, uniform and composite | 448-455 | ✅ verified | package `gravity!` jumps vs $\mathcal D a$ on every face with $a_f>0$, errors in units of the acceleration $\bar\rho^{\rm a}|f||g|h$ | `code_01_planar_state.jl` (H1) | acceleration error $1.8\times10^{-14}$ in all 6 fixtures; least-squares consistency of $J$ over all cell functions $3\times10^{-15}$–$5\times10^{-15}$ |
| 3. Oblique $g$ without solid, random tilts (11 planes, angles up to 35°, random offsets, uniform and composite) | 448-455 | ✅ verified | same | H2 | acceleration error $1.4\times10^{-13}$, consistency $5\times10^{-15}$ |
| 4. interface exactly on a grid line (pure liquid / pure gas faces; 16 uniform, 24 composite; and 32 faces at a horizontal level transition: coarse liquid cell $h_1=0.5$ below, fine gas cells $h_2=0.25$ above) | (thesis case 2 not covered) | ✅ verified in the code; (A3) refined | package: phase apertures of these faces $(0.5,0.5)$, consistency residual $3.7\times10^{-15}$, $4.9\times10^{-15}$, $3.3\times10^{-15}$; the jump equals that of the continuous $p^\dagger$ exactly for equal widths (ratio 1), and **0.750375** of it at the level transition, $=(\rho^{\rm l}+\rho^{\rm g})(h_1+h_2)/(2(\rho^{\rm l}h_1+\rho^{\rm g}h_2))$ | `code_02_special_configurations.jl` (G1, G2) | still balanced (a cell function exists: no closed loop crosses the interface with different jumps), but then $a\ne p^\dagger$ sampled at the cell points: the pressure has a step at the interface |
| 5. Cut one-phase faces: $\bar\rho^{\rm a}_f$ is that phase's density (113 such faces over the 12 fixtures) | 428-433 | ✅ verified | package: $(\rho^{\rm l}a^{\rm l}+\rho^{\rm g}a^{\rm g})/a$ vs phase density, $|a^{\rm l}+a^{\rm g}-a|=0$ | `../eq-forces-gravity/scripts/code_01_potential_density_jumps.jl` (G3) | $1.1\times10^{-16}$ |
| 6. "balanced next to a sloped solid in either phase (not proved here)" | 456-461 | ⚠️ true under (A1)–(A3), proof below; **not true in general** (F3) | proof; package counterexamples | proof in the replacement; `code_02` (T1) | the argument does not use the solid at all |
| 7. Checks sentence: "planar interface through (4.3,4.6) normal to $g=(0,-1)$ and to $(0.6,-0.8)$ … without solid and with a sloped solid in the gas or in the liquid" | 482-492 | ❌ over-reads (F2) | package: oblique $g$ **with** a solid: consistency residual $3.6\times10^{-4}$ (uniform, gas), $5.9\times10^{-3}$ (uniform, liquid), $8.6\times10^{-3}$ (composite, liquid); composite + gas $4.9\times10^{-15}$ | `code_01_planar_state.jl` (H4) | the figure script runs oblique $g$ only without solid |
| 8. Failure ⟺ PLIC off the plane: balanced iff all interface-cell centroids lie on the plane (both directions, all oblique cases + the wall-meeting plane) | — | ✅ verified | centroid offset $|g\cdot(x^I-x_0)|/|g|$ vs consistency residual: $(2\times10^{-15},4.9\times10^{-15})$ ↔ $(6.6\times10^{-3},3.6\times10^{-4})$ etc. | H4, H5 | the printed assumption "PLIC reproduces the plane exactly" is the exact condition |
| 9. Plane meeting a *horizontal* wall in a corner (37.8°, $y(0)=0$) | — | ✅ verified | consistency $9.8\times10^{-3}$, offset $1.2\times10^{-2}$ | H5 | PLIC not exact in the corner |
| 10. Plane meeting the sloped solid (triple cells), embedded contact angle $\pi/2$, $\arctan0.1$, $\pi-\arctan0.1$ | — | ✅ verified (failure) | consistency $1.17\times10^{-2}$, offset $2.7\times10^{-2}$, **identical for the three angles** (the embedded angle enters the curvature only) | `code_02` (T1) | no hydrostatic balance in triple cells |
| 11. Deviation box: the choice of $\phi_c$ does not matter for the balance | 468-475 | ✅ verified | package: any random point of the cell in a pure cell (cut or not) and any random point of the PLIC segment in a mixed cell (3 draws per fixture, 6 fixtures) | `code_02` (P1) | consistency $\le4.8\times10^{-15}$; the condition holds for any $\phi$ whose mixed-cell points lie on the plane |
| 12. Issue #74 text: "until #74 the code used $\rho^{\rm l}a^{\rm l}+\rho^{\rm g}(1-a^{\rm l})$, which put liquid next to a sloped solid out of equilibrium" | 462-466 | ✅ verified, with precision | package: pre-#74 density on the liquid-solid fixtures | `code_02` (O1) | composite mesh: consistency $2.1\times10^{-3}$, **one** face differs (cut hanging face, 23 cut liquid faces); uniform mesh: no face differs (all cut faces with a jump have $a=1$ up to rounding) |
| 13. Code jumps equal the independent jumps (own potential, package phase apertures) | 420-427 | ✅ verified | independent evaluation on every fixture | `code_02` (O2), `eq-forces-gravity` | 0.0 |

## Errors and consequences

**F1 (sharp statement of the assumptions).** "PLIC reproduces the plane exactly" is necessary and sufficient (step 8): with every interface-cell point on the plane the balance holds with a solid of any shape in either phase, and without it it fails at the cells where the PLIC deviates. This happens for an oblique plane within reach of a solid or a corner (height-function stencils truncated by the solid), and in triple cells where a planar interface meets a solid at an angle (steps 7–10). A solid does not obstruct balance as such.

**F2 (oblique + solid).** The chapter's check sentence suggests that oblique $g$ was checked with solids; it was checked only without solid. With a solid the planar interface of the fixture meets the solid or the walls and the PLIC is not exact; the balance then fails by $4\times10^{-4}$–$9\times10^{-3}$ (consistency residual), tied to the centroid offset (step 8).

**F3 (proof of the solid case; thesis case 2).** The thesis proof of Lemma 7.1 does not use the apertures $a^\pi_f$ of cut faces and excludes (case 2) the interface on a grid line implicitly. The code balances both (steps 4, 5; for the grid-line case $a=p^\dagger$ only at equal cell widths across the interface); the proof below is complete for any solid.

**Proof of the planar state (replacement text).** Let $\phi_c=\gravacc\cdot\gpoint{c}$, $q(\coord)=\gravacc\cdot(\coord-\coord^{\interface})$, and $\avar_c=\dens^{\phase}q(\gpoint{c})$ for a pure cell of phase $\phase$, $\avar_c=0$ for a mixed cell (then $q(\gpoint{c})=0$ by (A1)). For a face $f$ with cells $c_1,c_2$: if both are mixed, both sides vanish ($\phi_{c_1}=\phi_{c_2}$, $\avar_{c_1}=\avar_{c_2}=0$); if both are pure of the same phase, $\apdens{f}=\dens^{\phase}$ and $\dens^{\phase}(\phi_{c_2}-\phi_{c_1})=\avar_{c_2}-\avar_{c_1}$ since $\avar=\dens^{\phase}(\phi-\gravacc\cdot\coord^{\interface})$; if one is mixed and one pure of phase $\phase$, the face lies in the phase $\phase$ (a line through the mixed cell cannot cross a face of a pure neighbour), $\apdens{f}=\dens^{\phase}$ and $\dens^{\phase}(\phi_{c_2}-\phi_{c_1})=-\dens^{\phase}q(\gpoint{c_1})=\avar_{c_2}-\avar_{c_1}$; if both are pure of different phases (interface on a grid line) the phase apertures are ½ and ½, $\apdens{f}=\tfrac12(\dens^{\liq}+\dens^{\gas})$, and the identity holds with this $\avar$ when the two cells have the same width normal to the face (the planar interface lies midway between the points); for unequal widths (a level transition at the interface) $\jumpmat\avar$ and $\apdens{f}\jumpmat\gpot$ differ by the factor $0.750375$ (for the fixtures' $h_1=2h_2$) and the balance holds only with a different cell function: it was found to hold on the composite mesh with a horizontal level transition at the interface, with a pressure step; nothing else uses the metric or the solid, only $\apdens{f}\in\{\dens^{\liq},\dens^{\gas},\tfrac12(\dens^{\liq}+\dens^{\gas})\}$ as above.

**Proposed replacement** (the paragraph "Hydrostatic balance", lines 439-467):

```latex
\emph{Hydrostatic balance.} By~\cref{eq:forces:balance} gravity is balanced if and only if there is a cell function $\avar$ with
\begin{equation}\label{eq:forces:hydrostatic}
  \apdens{f}\,(\jumpmat\gpot)_f=(\jumpmat\avar)_f\quad\text{for every }f\in\facesint\text{ with }\facecoef_f>0 ;
\end{equation}
since this condition is face by face, it does not depend on the metric, and holds at level transitions whenever it holds.
For a planar interface normal to $\gravacc$ through $\coord^{\interface}$ it holds, with a solid of any shape and in either
phase, as soon as the PLIC line of every mixed cell is the plane (then $\gpoint{c}$ lies on it), with
$\avar_c=\dens^{\phase}\gravacc\cdot(\gpoint{c}-\coord^{\interface})$ in a pure cell of phase $\phase$ and $\avar_c=0$ in a mixed cell
(the thesis' $\pres^{\dagger}$~\thesis{eq.~(7.65), App.~7.A}): a face with a pure neighbour lies in one phase and then
$\apdens{f}$ is that phase's density by~\cref{eq:forces:aperture-density}, also when it is cut; a face between a liquid and a gas cell
(interface on a grid line) has the phase apertures $\tfrac12,\tfrac12$ and $\apdens{f}=\tfrac12(\dens^{\liq}+\dens^{\gas})$ (the state is balanced, with $\avar=\pres^{\dagger}$ if the two cells have the same width normal to the interface);
a face between two mixed cells has $\jumpmat\gpot=0=\jumpmat\avar$. The proof does not use the solid, nor the metric, nor the
choice of the potential point in a pure cell. The assumption is also necessary: where the PLIC is not exact, for an
oblique plane within reach of a solid or a corner and for an interface that meets the solid in a triple cell, the interface-cell points are off the plane and there is no
such $\avar$ (checks below: consistency residual $4\times10^{-4}$--$10^{-2}$). The energy and momentum identities of the
thesis~\thesis{eqs.~(7.59), (7.67)} are derived for the diagonal metric and are not claimed here; the diagnostic
\apitype{PotentialEnergy} integrates the exact first moments of the phase polygons, not $\gpot$.
```
and in the checks paragraph replace "…, without solid and with a sloped solid in the gas or in the liquid" by "… (vertical $\gravacc$ with a sloped solid in the gas or in the liquid, and the oblique $\gravacc$ without solid)".

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| "if and only if" condition (a) | **sharp** (iff) | injectivity of $\mathsf R$ on the support | `eq-forces-balance` | — |
| PLIC reproduces the plane (A1) | **sharp: necessary and sufficient** | residual ↔ centroid offset in every case incl. corner and triple cells; $\ne0$ exactly when off the plane | H4, H5, T1 | state it as the condition |
| (A2) one-phase faces carry the phase density | **sharp** (counterexample) | pre-#74 density: consistency $2.1\times10^{-3}$ on the composite fixture with solid in the liquid | `code_02` (O1) | — |
| (A3) pure–pure interface faces have apertures ½ | **balance holds, $a=p^\dagger$ only for equal widths** | G1: balanced and $=p^\dagger$; G2: balanced at a level transition but the jump is $0.750375\times$ the one of $p^\dagger$ (an interface-aligned level transition gives a pressure step) | G1, G2 | an aperture split proportional to the cell widths ($h_1:h_2$ weights) would restore $a=p^\dagger$ at such faces; a design question |
| independence of the potential point | **sharp** | 18 random draws, residual $\le4.8\times10^{-15}$; pure cells free, mixed cells: any point of the PLIC segment | `code_02` (P1) | state in the deviation box |
| Planar interface in a triple cell | **not balanced** | 1.2 % residual, independent of the embedded angle | T1 | open: a PLIC that is exact for a plane through a triple cell would restore it |
| Solver tolerance of the one-velocity projection | **likely sharp** | $2.2\times10^{-7}$ of $\delta t|g|$ (`eq-forces-stages` S1) | — | — |

## Open points

- A balanced oblique plane with a solid needs a fixture whose interface stays outside the height-function reach of the solid (interface plane through the box centre does not: it hits the walls). A planar interface that crosses a vertical wall at a non-right angle is balanced only because height functions near that wall reduce to one-sided differences; not exercised with a solid.
- The thesis' (7.65) uses $\{\!\{\chi\rho\}\!\}$ with $\chi$ in pure cells; the conditional-fraction variant for cut cells is the code's, covered by (A2).

## History

- 2026-11-09: first audit (`code_01_planar_state.jl`, `code_02_special_configurations.jl`). `run_all.jl`: PASS (two runs: before and after adding the level-transition case G2 to `code_02`; the final `code_02` was rerun alone, 12/12 tests).
