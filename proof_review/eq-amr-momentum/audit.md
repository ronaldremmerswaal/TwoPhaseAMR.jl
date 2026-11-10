# Audit: eq:amr:momentum — periodic integral correction; the quoted wall-momentum and kinetic-energy changes

- Source: `docs/techref/chapters/12_amr_criteria_transfer.tex:615-626` (`eq:amr:momentum` l. 618), l. 646-652 (check: wall axes 8.3e-3 to 1.2e-2, energy 8.6e-3), l. 741-760 and 796-800 (table `tab:amr:conservation`, flows); code `src/mesh/adaptivity/field_transfer.jl:445-466` (`_conserve_face_momentum!`), 495-532 (`_transfer_face_field!`), `docs/techref/figures/amr.jl`, commit `2959760`
- Date: 2026-11-10
- Overall status: ⚠️ claim verified; **the quoted $0.8$–$1.2\,\%$ and $0.86\,\%$ are not a transfer defect but a diagnostic artefact of the fixture** (first order in $h$, vanishes to second order for impermeable walls); normalisation and fixture need to be stated
- Rerun: `julia --project=@claude-symbolics proof_review/eq-amr-momentum/scripts/run_all.jl` (the second script reruns `amr.jl` in full, about 3 minutes)

## Statement and assumptions

For a `FaceVelocity`/`FaceField` registered with the integral correction (default), each *periodic* axis $k$ restores $M_k=\sum_{f\in\mathcal F^{\rm int}\cap\mathcal F_k}\omega_fu_f$ by one additive constant (`eq:amr:momentum`); this "leaves `eq:amr:identities` intact: in a periodic direction a constant normal component has zero divergence". On an axis with walls no correction is made ("keeps affine exactness and the divergence of the wall cells"); $M_k$ then changes.

Reported (l. 646-652, table): on the walled static fixtures $M_k$ changes by 8.3e-3 to 1.2e-2 **of $\sum\omega_f|u_f|$** and the kinetic energy by up to 8.6e-3 **of itself** per regrid; in the flows the density-weighted momentum $\sum\rho_f\omega_fu_f$ changes by 4.2e-6, 3.8e-6, 5.4e-5 **of $\sum\rho_f\omega_f|u_f|$** and the energy by 1.4e-5, 9.2e-6, 3.2e-5.

Implicit assumptions that the text does not state: $M_k$ and $E=\sum\omega_fu_f^2/2$ use the **interior faces only** (`facesint`); the static walled fixtures use velocity fields (`general`, `solenoidal` of `amr.jl`) whose normal component **does not vanish at the walls**; the flows use impermeable walls (stream function $\sin^2(\pi x/2)\sin^2(\pi y)$).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| Constant correction restores $M_k$ on periodic axes | l. 618-624 | ✅ | random-noise velocity, random 2:1 meshes, doubly periodic and $x$-periodic channel: $|\Delta M_k|/\sum\omega|u|\le6.0\times10^{-16}$ (M1) | `code_01_momentum_energy.jl` | the target is the *old* integral with the old dual volumes; the constant is $(M_{\rm old}-M_{\rm new})/\sum_{f'}\omega_{f'}$ |
| ... and leaves every cell divergence unchanged | l. 625 | ✅ | identities of `eq:amr:identities` after the correction (default registration): $\le7.3\times10^{-16}$ relative (M1) | same | the cell's two $x$-sides have equal total area, all faces of a periodic axis are interior faces |
| On a walled axis the correction is not made; this "keeps affine exactness and the divergence of the wall cells" | l. 626-628, code comment `field_transfer.jl:519-525` | ✅ | a constant added to the interior faces of a walled axis changes the divergence of every cell with a boundary face on that axis (the boundary faces are not corrected) and breaks affine exactness (the interior values are exact); the interior-only $M_k$ is resolution dependent because the boundary half-control-volumes are missing | — | argument; the numerical evidence for the resolution dependence is M2–M4 |
| Reported static numbers (8.3e-3…1.2e-2; 8.6e-3) | l. 650-652 | ✅ reproduced | a fresh run of `amr_transfer_check()`: `chkAmrWallMomentumMin = 0.00826`, `chkAmrWallMomentum = 0.01205`, `chkAmrEnergy = 0.00864` (all chkAmr macros equal the committed values) | `code_02_reproduce_checks.jl` | the numbers are what the script measures |
| Reported flow numbers | l. 796-800, table | ✅ reproduced | fresh runs of the three flow cases (60 + 60 + 40 steps): momentum $4.16\times10^{-6}$, $3.85\times10^{-6}$, $5.38\times10^{-5}$; energy $1.43\times10^{-5}$, $9.17\times10^{-6}$, $3.20\times10^{-5}$; transferred defects $3.7\times10^{-11}$, $2.1\times10^{-4}$, $2.9\times10^{-2}$ | same | |
| **Interpretation of the static wall numbers** (task question) | l. 650-652, 752-754 | ⚠️ not a defect: an artefact of the interior-only integral and of wall-crossing fixture velocities | see "Errors and consequences" and M2–M5 | `code_01_momentum_energy.jl` | |

## Errors and consequences

**Finding E1 (interpretation, not an error in an equation).** Chapter 12 reports a wall-axis momentum change of 0.8–1.2 % and an energy change of up to 0.86 % per regrid "on the walled axes". Both numbers are produced by fixtures whose velocity does not vanish at the walls, with integrals over the interior faces only. Measured on the walled $2\times1$ box (blocks of fixed physical size, base meshes $h_0=1/16,1/32,1/64$ with the same regridding sequence; $\max_k|\Delta M_k|/\sum\omega|u|$ and $|\Delta E|/E$):

| field | integral | $h_0=1/16$ | $1/32$ | $1/64$ | observed order in $h_0$ ($M$ / $E$) |
|---|---|---|---|---|---|
| `general` (u·n ≠ 0 on the walls) | interior faces (as in the chapter) | $1.3\times10^{-2}$ / $4.0\times10^{-3}$ | $6.1\times10^{-3}$ / $2.1\times10^{-3}$ | $3.0\times10^{-3}$ / $1.1\times10^{-3}$ | **1.03–1.05 / 0.92–0.96** |
| `general` | closed with the wall faces ($\omega=|f|V/(2h_t)$) | $9.9\times10^{-4}$ / $5.8\times10^{-4}$ | $2.5\times10^{-4}$ / $1.5\times10^{-4}$ | $6.1\times10^{-5}$ / $3.7\times10^{-5}$ | 2.0 / 2.0 |
| `solenoidal` (through-flow at the walls) | interior | $8.0\times10^{-3}$ / $6.0\times10^{-3}$ | $3.9\times10^{-3}$ / $2.4\times10^{-3}$ | $2.0\times10^{-3}$ / $1.0\times10^{-3}$ | 1.0 / 1.2–1.3 |
| `solenoidal` | closed | $8.4\times10^{-5}$ / $2.4\times10^{-3}$ | $2.2\times10^{-5}$ / $6.3\times10^{-4}$ | $5.6\times10^{-6}$ / $1.6\times10^{-4}$ | 1.9–2.0 / 1.9–2.0 |
| impermeable ($u\cdot n=0$ at all walls: $\psi=\sin^2\frac{\pi x}2\sin^2\pi y$) | interior = closed | $9.1\times10^{-5}$ / $4.2\times10^{-3}$ | $2.4\times10^{-5}$ / $1.2\times10^{-3}$ | $6.0\times10^{-6}$ / $3.0\times10^{-4}$ | 1.9–2.0 / 1.9–2.0 |
| doubly periodic, smooth solenoidal | (all faces interior; $M$ restored) | — / $3.2\times10^{-3}$ | — / $8.5\times10^{-4}$ | — / $2.1\times10^{-4}$ ($h_0=1/128$) | — / 1.96 |

Interpretation:

1. **Expected behaviour, no transfer defect.** The face transfer conserves the integrated flux through every surviving line piece to round-off (`eq-amr-face-line`) and the cell divergence exactly (`eq-amr-correction`). The change of the *interior-only* staggered integral at a wall is a quadrature effect: the half dual volumes of the boundary faces are not part of the sum, and the size of the wall cell changes under a regrid, so $M_k$ changes by $O(h)\,u_{\rm wall}$: first order, observed 1.03–1.05, and removed (order 2, factor ≥ 10 at $h_0=1/16$) by closing the integral with the wall faces or by choosing a velocity that is impermeable at the walls. The code comment `field_transfer.jl:519-525` says exactly this ("its missing boundary half-control-volumes make the interior-only staggered integral resolution dependent").
2. **Order of the kinetic-energy change for the face transfer used (limited linear in the tangential direction, linear in the normal direction, divergence correction $O(h^2)$): second order,** $|\Delta E|/E\approx c\,(kh)^2$ with $c\approx0.1$ for the smooth fields tested ($k\sim\pi$; observed 1.85–2.0 on the impermeable box and 1.96 on the periodic square, 4.2e-3 at $h_0=1/16$ → 3.0e-4 at $1/64$). The change is positive when refining (periodic square, $h_0=1/32$: $+1.1\times10^{-3}$, $+3.2\times10^{-3}$ for the two refining regrids), and the two coarsening regrids that follow undo it exactly: the energy change of the whole refine–refine–coarsen–coarsen cycle is $\le1.1\times10^{-16}$ (M5), so a static refine/coarsen cycle does not drift. The prolongation interpolates a smooth field with an $O(h^2)$ quadrature error of the energy; no energy-conserving face prolongation is claimed or provided. It is therefore the same order as the discretisation error, not a conservation defect. With a wall-crossing velocity and the interior-only sum the measured energy change is first order (0.92–1.3).
3. The 0.83–1.2 % and 0.86 % of the chapter belong to $h_0=1/16$, level 2, **with a velocity that does not vanish at the walls**. In the flows (impermeable walls, smooth velocity) the changes are $10^{-5}$–$5\times10^{-5}$ (momentum relative to $\sum\rho\omega|u|$), $1\times10^{-5}$–$3\times10^{-5}$ (energy), as expected for second-order consistency on a nearly uniform field.

**Suggested rewording** (l. 646-652; add the field and the integral):

```latex
and~\cref{eq:amr:momentum} restores $\faceintegral{k}$ on the periodic axes to
$\chkAmrPeriodicMomentum$ of $\sum\dualvol{f}\abs{\velo_f}$. On the walled axes
no correction is made: for these fixtures, whose velocity does not vanish at the
walls, the staggered integral over the interior faces changes by
$\chkAmrWallMomentumMin$ to $\chkAmrWallMomentum$ and the kinetic energy by up
to $\chkAmrEnergy$ per regrid, a first-order effect of the half dual volumes of
the omitted wall faces (it is second order, about $10^{-4}$ at $h_0=1/16$,
for a velocity with $\velo\cdot n=0$ at the walls); the kinetic-energy change
of the transfer itself is second order in the cell width.
```

and the table header "Momentum $\sum\stagdens\dualvol{f}\velo_f$" should say relative to $\sum\stagdens\dualvol{f}\abs{\velo_f}$ (static rows: relative to $\sum\dualvol{f}\abs{\velo_f}$, interior faces, no density).

**Consequence for the cut flow** (not asked, but visible in the reproduced run): the open-divergence defect after the transfer in the cut case is $2.9\times10^{-2}$ (dimensionless $\delta t\max|D^{\rm o}|$), $2.9\times10^{8}$ times the tolerance and not reduced by an extra projection (as the chapter states: "left to the next step"). That is a known limitation of the Cartesian correction with apertures (issue/limitation documented in §divergence), not a new finding.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| Periodic correction restores $M_k$ and keeps the divergence | **sharp** (round-off) | M1 | `code_01_momentum_energy.jl` | — |
| No correction on walled axes | **necessary** for the divergence of the wall cells | argument; a constant added to interior faces changes the divergence of every wall cell of that axis | — | a correction on the walled axis that also adjusts the boundary face (not available for a prescribed wall velocity) |
| $M_k$ not conserved on walled axes | **sharp as a statement about the interior-only integral with wall-crossing velocity** (first order); for impermeable walls it is a second-order quadrature effect | table above | same | state the fixture and the integral (above) |
| Energy change | second order for smooth data (**likely sharp**: orders 1.85–2.0 numerically) | M4, M5 | same | an energy-preserving prolongation would need a modified face map; not required |

## Open points

- The static wall fixtures could be replaced by an impermeable field so that the quoted numbers measure the transfer, not the diagnostic; the chapter's generator (`amr.jl`) would then print $\sim10^{-4}$ for $M_k$ on the walled axes. (Requires an edit of `docs/techref/figures/amr.jl`, outside this audit.)
- The thesis does not state a field transfer; no literature comparison was needed. A comparison with Popinet's/Berger–Colella divergence-free prolongations (Tóth–Roe [`toth2002`], cited in the chapter) of the *energy* change is open (suggest `literature-scout`).

## History

- 2026-11-10: first audit (commit `2959760`): M1–M5 pass; `code_02_reproduce_checks.jl` reproduces every `chkAmr*` macro tested (22 static/indicator macros, 14 flow assertions); `run_all.jl` passes.
