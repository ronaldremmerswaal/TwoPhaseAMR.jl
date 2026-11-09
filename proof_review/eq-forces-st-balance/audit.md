# Audit: eq:forces:st-balance — balanced surface tension (constant curvature), and the jump formula eq:forces:st-jump

- Source: `docs/techref/chapters/08_forces.tex:308-366` (st-jump 313, st-balance 342); sign conventions 24-80 (`eq:forces:stress-balance`, `eq:forces:steady`), commit `03bad7a`
- Code audited (normative): `src/two_phase/forces.jl` (`surface_tension!`, `_surface_tension_jump`, `_face_curvature`)
- Date: 2026-11-09
- Overall status: ✅ verified (balance, jump formula, signs, zero rules); two quantitative additions proposed (parasitic currents of the real curvature; level-transition coupling)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-forces-st-balance/scripts/run_all.jl`

## Statement and assumptions

**`eq:forces:st-jump`.** $\mathcal J^\sigma_f=\sigma\,\varkappa_f\,(\mathcal D\psi)_f$, $\psi=C$ (CSF) or the indicator $[C>\tfrac12]$ (CGFM); zero if $\sigma=0$, $(\mathcal D\psi)_f=0$, $a_f=0$ or $f$ unresolved. At a face with liquid below and gas above $\mathcal J_f=-\sigma\varkappa_f|f|$.
**`eq:forces:st-balance`.** If $\varkappa_f=\varkappa_0$ on every face with $(\mathcal D\psi)_f\ne0$, $a_f>0$ (none unresolved), then $\mathsf R\mathcal J^\sigma=\mathfrak S_\gamma(\sigma\varkappa_0\psi)$ (or $\mathfrak S^{\rm o}$), $p=\sigma\varkappa_0\psi+$const; the one-velocity projection leaves a solenoidal predictor unchanged on uniform, composite and cut meshes; for CGFM $p$ jumps by $\sigma\varkappa_0$ across the indicator (discrete Young–Laplace, steady state (a)), for CSF $p$ follows $C$.

**Assumptions made explicit.** Same operator in the force and in the projection (eq:forces:balance); the fixture "cut" is the composite mesh with the sloped solid of the drop fixtures (the solid does not touch the drop).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. jumps written by the code $=\sigma\varkappa_0\mathcal D\psi$ on faces with $a>0$, zero elsewhere | 313-320, 342-346 | ✅ verified | package: `surface_tension!(…; curvature=κ0)`, CSF and CGFM, uniform / composite / cut | `code_01_const_curvature.jl` (S1) | difference 0.0 |
| 2. $\mathsf R\mathcal J=\mathfrak S(\sigma\varkappa_0\psi)$ | 342-346 | ✅ verified | package vs `weighted_gradient!`/`_cut_cell_weighted_gradient!` | S2 | $\le2.7\times10^{-16}$ |
| 3. projection leaves a nonzero solenoidal predictor unchanged, $\mathsf S(p-\sigma\varkappa_0\psi)=0$ | 347-352 | ✅ verified | package `project!` (tolerance $10^{-12}$) | S3 | leftover $\le1.5\times10^{-6}$ of $\delta t\max|\mathsf R\mathcal J|$ (CSF drop, density ratio 1000; a solver tolerance, $\le4.5\times10^{-8}$ on the cut mesh) |
| 4. CGFM: $p_{\rm liquid}-p_{\rm gas}=\sigma\varkappa_0$; $p$ constant in the liquid core and in the gas; CSF: $p-\sigma\varkappa_0C$ constant | 350-355 | ✅ verified | package: pressure levels of the projection | S4 | level jump $0.313901=\sigma\varkappa_0$ (to $10^{-6}$); spread of $p-\sigma\varkappa_0\psi$ $\le5\times10^{-8}$ |
| 5. $\mathcal J^\sigma$ equals the independent evaluation (conditional fractions, face curvature of the code, zero rules), CSF and CGFM | 313-320 | ✅ verified | independent re-evaluation on all faces of the three fixtures | `code_02_jump_formula_and_limits.jl` (J1) | difference 0.0; the unresolved count equals the independent count (0 on the drops) |
| 6. sign: low liquid / high gas: $\mathcal J_f=-\sigma\varkappa_f|f|<0$; force into the liquid | 321-323 | ✅ verified | package: CGFM faces (18 uniform, 33 composite): $\mathcal J_f=-\sigma\varkappa_f|f|$ bitwise | J2 | — |
| 7. $\sigma=0$ gives zero jumps and force | 319 | ✅ verified | package | J (before J1) | exact zeros |
| 8. $(\mathsf R\mathcal J^\sigma)_f=\sigma\varkappa_f(\mathfrak S\psi)_f$ on a conforming patch; not at a level transition unless $\varkappa$ is constant over the patch | 325-337 | ✅ verified | rows of $\mathsf K$ with one entry (480 / 1162 faces) vs the 78 coupled rows of the composite fixtures | J3 | diagonal rows $\le2.8\times10^{-16}$; coupled rows differ by 2.2 %–5.5 % of the largest value |
| 9. Laplace pressure of the code's curvature is positive and $\approx\sigma/R$ (convex drop, $p^{\rm l}>p^{\rm g}$) | 36-52 | ✅ verified | package: projected real CGFM force, composite drop | J4 | 0.31715 vs $\sigma/R=0.31390$ (+1.0 %) |
| 10. "the computed force of a drop is not exactly balanced; the parasitic velocity is a property of the curvature estimate, not of the map" | 356-361 | ✅ verified, quantified (F1) | package: consistency of the real jumps (does a cell function exist?) and leftover velocity after one projection | J4 | consistency residual 1.1–1.5 % (uniform, composite), 4.3–5.7 % (cut); leftover/$|\mathsf R\mathcal J|$ 0.8–3 % (CGFM 1.5, 0.8; CSF 2.9, 3.0 %) and **5.7 % (CGFM), 14.5 % (CSF) on the cut mesh** |

## Errors and consequences

No error. Two additions (not errors):

**F1 (quantify the parasitic current; the solid matters).** The statement "a property of the curvature estimate" is right (the force map reproduces the weighted gradient to rounding, step 2), but on the cut fixture the parasitic leftover is five times larger than on the composite fixture although the solid does not touch the drop (0.145 vs 0.030 for CSF). The cause lies in the curvature estimate near the solid (not investigated: the map and the projection are the same in both).

**F2 (clarify "conforming patch").** The reduction $\mathsf R\mathcal J=\sigma\varkappa_f\mathfrak S_\gamma\psi$ holds at exactly the faces whose row of $\mathsf K$ has a single entry (1162 of 1240 on the composite fixture); the 78 coupled faces lie in $x\in[3.75,4.125]$ around the level transition.

**Proposed replacement** (the paragraph after `eq:forces:st-balance`, lines 356-361):

```latex
The pressure solve itself imposes no value jump: the Young--Laplace jump enters only through the
explicit increment and is carried by the pressure that balances it. The curvature of a discrete circle
varies from face to face, so the computed force of a drop is not exactly balanced; the remaining (parasitic)
velocity is a property of the curvature estimate, not of the map. In the drop fixtures one projection of
$\dt$ times the force with the computed curvature leaves 0.8--3\,\% of $\dt\max_f\abs{(\forcemap\fjump{\surftens})_f}$
on the uniform and composite meshes and 6--15\,\% (CGFM, CSF) on the cut mesh, although the solid does not touch the drop;
the Laplace pressure of the projected CGFM force is within 1\,\% of $\surftens/R$.
```
(and, in the sentence before `eq:forces:st-balance`: "only where $\curvface$ is constant over the patches of $f$" → "only at the faces whose row of $\rieszmat$ is diagonal or where $\curvface$ is constant over the patch").

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| $\varkappa_f=\varkappa_0$ on the faces with a jump | **sharp** (iff, by eq:forces:balance) | any variation of $\varkappa_f$ makes $\mathcal J$ not the jump of a cell function; real curvature: consistency residual 1–6 % | `code_02` (J4) | state "if and only if" |
| "uniform, composite and cut meshes" | **sharp** (3 fixtures) | S2/S3 hold on all three | `code_01` | — |
| CSF: $p$ follows the conditional fraction | **sharp** | spread of $p-\sigma\varkappa_0C$ $\le5\times10^{-8}$ | `code_01` (S4) | — |
| CGFM: pressure jump equals $\sigma\varkappa_0$ | **sharp** (equality to $10^{-6}$, solver tolerance) | level jump 0.313901 | `code_01` | — |
| Parasitic velocity independent of the map | **likely sharp** | S2: map error $10^{-16}$; leftover $10^{-2}$–$10^{-1}$ comes from the estimate | `code_02` | — |

## Open points

- Cause of the larger parasitic current on the cut fixture (curvature near a nearby solid; the drop's lowest point is 0.2 above the cut cells): not investigated here. A literature check is not needed.

## History

- 2026-11-09: first audit (`code_01_const_curvature.jl`, `code_02_jump_formula_and_limits.jl`). `run_all.jl`: PASS.
