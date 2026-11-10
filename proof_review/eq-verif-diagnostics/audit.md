# Audit: eq:verif:energy, eq:verif:energy-two, eq:verif:energies, eq:verif:divergence-norm — the diagnostics of Chapter 14

- Source: `docs/techref/chapters/14_verification.tex:22-118` (section "Diagnostics" and its implementation note), branch `docs/techref-verification`, commit `77f46cb`; code `src/simulation/diagnostics/interface.jl`, `src/two_phase/forces.jl` (`_staggered_liquid_fraction`, `_phase_first_moments`, `_gravitational_potential_energy`), `src/visualization/plic_geometry.jl`, `src/embedded/geometry_utils.jl`, `src/embedded/operators.jl`, `src/operators/advection/cell_advection.jl`
- Labels: `eq:verif:energy`, `eq:verif:energy-two`, `eq:verif:energies`, `eq:verif:divergence-norm` (one folder; the four equations form one section)
- Date: 2026-11-10
- Overall status: ✅ verified — all four formulas equal the code (independent re-implementations from the text, 11 random fixtures); three wording/notation points (W1–W3) and one measured limitation of the interface measure (L1) for the text
- Rerun: `julia --project=@claude-symbolics proof_review/eq-verif-diagnostics/scripts/run_all.jl` (child processes under `docs/techref/figures`)

## Statement and assumptions

The diagnostics map the end-of-step state to numbers:

- $E_k=\tfrac12\sum_{f\in\mathcal F^{\rm int}}\omega^{\rm o}_f\bar\rho_f u_f^2$ and $\mathsf P_k=\sum_{f\in\mathcal F_k}\omega^{\rm o}_f\bar\rho_f u_f$ (one velocity); with two velocities the weights $\rho^{\rm l}\hat C_f$, $\rho^{\rm g}(1-\hat C_f)$ multiply the phase velocities; boundary faces do not enter; the clamped staggered fraction $\hat C_f$ is that of `eq:forces:density` (strip weights, clamp).
- $E_p=-\mathbf g\cdot(\rho^{\rm l}M^{\rm l}+\rho^{\rm g}(M^{\rm o}-M^{\rm l}))$ with the first moments of the open region and of the reconstructed liquid region (PLIC polygons).
- $|\Gamma|=\sum_{0<C_c<1}|\Gamma_c|+\sum_{f\ \text{separating}}|f|a_f$, $E_\sigma=\sigma|\Gamma|$.
- $\|\mathfrak D v\|_2=(\sum_{a_c>0}|c^{\rm o}|(\mathfrak D v)_c^2)^{1/2}$ with $\mathfrak D=\mathfrak D$ (one velocity, no solid), $\mathfrak D^{\rm o}$ (one velocity, solid), $\mathfrak D^{\rm mix}(v^{\rm l},v^{\rm g})$ (two velocities); this is the constraint of the projection, aperture-weighted volume flux (post-#101), not density weighted.

Implicit assumptions that the text should state or that hold by construction:
(i) $\hat C_f$ is the *open-strip* weighted fraction with the clamp inactive in practice (K7: raw fraction in $[0,1]$ to $10^{-12}$ on all fixtures; the clamp only removes round-off);
(ii) the fractions are refreshed: after `_refresh_conditional_fraction!` every mixed cell has $\varepsilon^{\varphi}<C<1-\varepsilon^{\varphi}$, which is what makes the segment sum and the face sum disjoint (S2);
(iii) the PLIC liquid side of a cell is $\{\mathbf n\cdot\mathbf x\le a\}$ in local coordinates (checked through $|{\rm liquid\ polygon}|=C\,|c^{\rm o}|$ in every mixed cell, gap $\le10^{-12}$);
(iv) the potential energy has an origin-dependent additive constant (stated in the text: zero at the origin) and gravity along a periodic direction is rejected by the model.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| D1 $E_k$, $\mathsf P_k$ one velocity | `eq:verif:energy`, `:24-34` | ✅ | independent sums over the 11 fixtures (uniform/composite, drops, grid-line planes, sloped/aligned/disc solids), density ratios 1000 and 6.6: relative gap $\le10^{-12}$ (K1); single phase with and without solid (K6) | `code_01_kinetic_momentum.jl` | |
| D2 open dual volume $\omega^{\rm o}_f$ equals `geometry.face_dual_volumes` (`mesh.face_dual_volumes` without a solid) | `:24-27` | ✅ | strip-area re-evaluation, $\le10^{-12}\max$ (K3) | `code_01_kinetic_momentum.jl` | clipping round-off of covered faces ($10^{-29}$) is treated as zero |
| D3 two-velocity weights $\rho^{\rm l}\hat C_f$, $\rho^{\rm g}(1-\hat C_f)$, zero weight $\Rightarrow$ zero product | `eq:verif:energy-two`, `:35-42` | ✅ | independent sums with NaN at absent phases (22 958 NaN entries exercised), $\le10^{-12}$ (K2); algebraic consequence $u^{\rm l}=u^{\rm g}$ gives the one-velocity values (K4) | `code_01_kinetic_momentum.jl` | the rule also covers faces with $\omega^{\rm o}_f=0$ (W1) |
| D4 boundary faces do not enter | `:42-43` | ✅ | boundary values set to $10^6$ leave $E_k$ unchanged (K5) | `code_01_kinetic_momentum.jl` | |
| D5 phase volumes, masses | `:43-48` | ✅ | $V^{\rm l}=\sum|c^{\rm o}|C_c$, $V^{\rm g}=|\Omega^{\rm o}|-V^{\rm l}$ from clipped polygons, 11 fixtures, $\le10^{-12}$ (V1) | `code_02_potential_surface_volume.jl` | $|\Omega^{\rm o}|=\sum_c|c|a_c$ |
| D6 $E_p$ from first moments of PLIC polygons | `eq:verif:energies`, `:49-66` | ✅ | independent polygon moments (open cell clipped by the solid plane, liquid side of the PLIC line), 11 fixtures, $\le10^{-12}$ (P1); the 4$\times$4 unit-test values $1.75$ (grid line), $1.27$ (mixed) and $0.875$ (cut) exact; the mass-at-centre alternative gives $1.30$ for the mixed case (so the sentence is a real statement); oblique plane $0.6x+0.8y\le4.4$ on a composite mesh: $E_p=24449.050$ against the exact clipped-box moments $24448.708$ (relative $1.4\times10^{-5}$, not exact) | `code_02_potential_surface_volume.jl` | |
| D7 $|\Gamma|$: PLIC chords of mixed cells, clipped to the open part | `:49-66` | ✅ | independent chord of the open polygon (solid-clipped), 11 fixtures, $\le10^{-12}$ (S1) | `code_02_potential_surface_volume.jl` | |
| D8 grid-line face term $\sum_{\rm sep}|f|a_f$ | `:49-66` | ✅ | planes $x=4$ ($8$), $y=3.5$ ($8$), $x=2$ with an aligned solid $y\le1$ ($7$), $x=5$ with a sloped solid ($8-1.4=6.6$): exact to $10^{-13}$; offsets of $0.1\ldots0.9$ floors (face term) and $1.1\ldots10$ floors (PLIC chords, 16 mixed cells) both give $8$ (S3) | `code_02_potential_surface_volume.jl` | the measure is continuous across the floor |
| D9 no double counting | `:61-66` | ✅ | no neighbour of a separating face is a mixed cell, 0 violations on 11 fixtures (S2) | `code_02_potential_surface_volume.jl` | depends on assumption (ii) |
| D10 $E_{\rm tot}=E_k+E_p+E_\sigma$, one phase $E_p=E_\sigma=0$ | `:67-68` | ✅ | `diagnostic_value(TotalEnergy)` is the sum (source), one-phase returns zero (unit test) | – | read |
| D11 $\|\mathfrak D v\|_2$ one velocity (no solid / solid) | `eq:verif:divergence-norm`, `:69-79` | ✅ | independent cell sums with random interior and boundary velocities, 11 fixtures, $\le10^{-12}$ (D1) | `code_03_divergence_norm.jl` | |
| D12 $\|\mathfrak D v\|_2$ two velocities: aperture-weighted volume flux $a^{\rm l}v^{\rm l}+a^{\rm g}v^{\rm g}$ | `eq:verif:divergence-norm`, `:69-79` | ✅ | independent sums incl. boundary faces; independent of the densities ($1000$ vs $6.6$) to $10^{-13}$; $a^{\rm l}+a^{\rm g}=a$ on all interior and boundary faces to $10^{-13}$ (D2) | `code_03_divergence_norm.jl` | |
| D13 this is the projection constraint | `:77-80` | ✅ | after one projected step (CMDGFM, mimetic; uniform and cut; density ratio 1000, slip state) the norm is $2\times10^{-14}$ (CMDGFM) and $1.3\times10^{-9}$ (mimetic) $\le\sqrt{|\Omega^{\rm o}|}\,\text{tol}/\delta t=10^{-6}$ while it was $0.10$ before, and the density-weighted mixture divergence is $0.087$/$0.067$ (D3); one velocity with a solid: $10^{-15}$ after the step against $0.079$ for the aperture-free divergence (D4) | `code_03_divergence_norm.jl` | the foil quantities show that the diagnostic would otherwise fail to detect the constraint |
| D14 boundary faces enter the divergence | `:69-79` | ✅ | uniform through-flow with NaN phases: $\le128\varepsilon$; perturbing one boundary face gives $>10^{-3}$ (D5) | `code_03_divergence_norm.jl` | |
| D15 $\max|v_f|$: interior and boundary faces; two velocities over active faces | `:80-85` | ✅ | $10^3$ planted on absent-phase faces is ignored; interior and boundary maxima equal the independent values (V2) | `code_02_potential_surface_volume.jl` | |
| D16 implementation note (recording rules, test list) | `:86-118` | ✅ (read) | `_diagnostic_due!` (same state recorded once; `:completed` always unless identical), `record_diagnostics!` loop; test titles and values (4$\times$4 values, $2\times10^{-13}$) match `test/diagnostics.jl` | – | `check_links.py` guards the symbols |

## Errors and consequences

No error in the four formulas.

- **W1 (wording).** "where a product with a vanishing weight is zero, also where the phase velocity is undefined" does not cover the covered faces of a cut mesh: there the *volume* $\omega^{\rm o}_f$ vanishes, both phase velocities are NaN and only the code's `iszero(volumes[index]) && continue` prevents $0\cdot\mathrm{NaN}$ (my first independent implementation produced NaN there). Replacement: ``where a product with a vanishing weight ($\opendualvol{f}$, $\clampfrac{f}$ or $1-\clampfrac{f}$) is zero, also where the phase velocity is undefined''.
- **W2 (notation).** `\mathfrak{D}` is the rendering of `\divh` (`techref-notation.sty:304`), so the first case of `eq:verif:divergence-norm` reads $\mathfrak D\boldsymbol v=\mathfrak D\boldsymbol v$. Replace the left-hand side (and the squared factor) by a different symbol, e.g.
  ```latex
  \divnorm=\Big(\sum_{c:\ \openfrac{c}>0}\meas{\opencell{c}}\,(\mathfrak{D}^{\star}\velo)_c^2\Big)^{1/2},\qquad
  \mathfrak{D}^{\star}\velo=\begin{cases}\divh\velo, &\ldots\end{cases}
  ```
- **W3 (completeness).** The text does not say that boundary faces enter the divergence (they carry prescribed fluxes; D14), while they do not enter the energies. Add after the equation: ``The boundary faces enter the divergence with their prescribed values (and, with two velocities, the boundary phase apertures), unlike the energies.''
- **L1 (not an error; limitation worth one sentence).** The interface measure is the length of the PLIC reconstruction. On a circle (radius $2.21$, uniform meshes $16,32,64$) its error is $+0.054,-0.034,+0.021$ (orders $0.64$, $0.72$), i.e. at best first order with a changing sign, whereas the potential energy from the polygon moments is far more accurate (errors $2.5\times10^{-3}$, $4.7\times10^{-6}$, $1.4\times10^{-7}$ on $E_p\approx2\times10^{4}$). Even a straight line of slope $3/4$ ($0.6x+0.8y=4.4$, composite mesh, cells $0.5$ and $0.25$) is measured $9.311$ against the exact $9.167$ ($+1.6\,\%$). Hence $E_\sigma$ is accurate to $\approx0.15\,\%$ at 64 cells across a drop of radius $2.21$ in the box of side 8 and its time series carries that bias. Suggested sentence after `eq:verif:energies`: ``The measure $\meas{\interface}$ is the length of the reconstruction; on a circle it converges with an observed order of about $0.7$, so $\surfenergy$ is a diagnostic of changes, not a second-order approximation of the exact surface energy.''

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| Formulas as definitions (equalities with the code) | sharp | equality to $10^{-12}$ relative on all fixtures | all | none |
| Clamp of $\hat C_f$ | needed only for round-off | raw fraction in $[0,1]$ on all fixtures | `code_01` (K7) | the clamp could be stated as "protects against round-off" |
| Disjointness of the segment and face sums | holds under the refresh invariant | S2; floor continuity S3 | `code_02` | state assumption (ii) |
| Divergence norm $=$ projection constraint | sharp (equality) | norms after projection $=$ solver defect ($2\times10^{-14}$, $1.3\times10^{-9}$), foils $0.09$, $0.08$ | `code_03` | none |
| Bound $\sqrt{|\Omega^{\rm o}|}\,\mathrm{tol}/\delta t$ (test) | not sharp for the mimetic projection | actual $1.3\times10^{-9}$ against $10^{-6}$ (factor 800) | `code_03` | none |
| $E_p$ exact for planes | not exact, very accurate | oblique plane (slope 3/4, 16 cells per 8): relative error $1.4\times10^{-5}$; grid-line planes exact | `code_02` (P3) | state "accurate", not "exact" for oblique planes |
| $|\Gamma|$ second order | not sharp (not claimed) | observed order $0.7$ on a circle | `code_02` (S4) | L1 sentence |

## Open points

- The post-#101 aperture weighting is a *code* fact verified here and in `test/diagnostics.jl`; the text of the chapter does not mention #101 (not needed).
- `CFLTimeStep` equals `stable_timestep(...; cfl=0.45)` of `eq:stab:flow-bound` (read, not run).
- `MaximumVelocity` with two velocities uses the boundary activity rule (`_phase_cell_is_active`), which the chapter states only through "faces on which a phase is active"; checked in V2.

## History
- 2026-11-10: first audit (branch `docs/techref-verification`, commit `77f46cb`): D1–D16 run (K1–K7, P1–P3, S1–S4, V1–V2, D1–D5 of the scripts); W1–W3, L1 proposed; source file not edited.
- 2026-11-10 (end of audit): Final `run_all.jl`: all checks passed (K: 221, P/S/V: 183, D: 129 assertions; Runic 1.10.0 formatted).
