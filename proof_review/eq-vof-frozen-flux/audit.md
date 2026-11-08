# Audit: eq:vof:frozen-flux — equal-velocity reduction of the frozen subface carrier to the one-velocity rectangles

- Source: `docs/techref/chapters/05_vof_transport.tex:586-644` (subfaces (5.22) `eq:vof:subface` at 595-607, fluxes (5.23) `eq:vof:frozen-flux` at 618-624, reduction sentence 638-644 "Without a solid and with $u^{\rm l}_f=u^{\rm g}_f$ the two subface rectangles tile the one-velocity rectangle and the fluxes coincide with those of Section 5.4; with a solid they differ in shape"), commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/two_phase/two_velocity/interface_transport.jl` (`_frozen_subface_interval`, `_frozen_subface_swept_fraction`, `_compute_directional_wy_fluxes!` for `FrozenInterfaceFluxes`), `src/vof/fluxes.jl` (`_swept_fraction`, one-velocity fluxes).
- Date: 2026-11-08
- Overall status: ✅ **verified** (hypotheses: no solid, $u^{\rm l}_f=u^{\rm g}_f$ on every face). On 16 random graded meshes (14 444 faces, 1 328 hanging) the frozen path with equal phase velocities reproduces the one-velocity path to round-off in apertures, subface tiling, fluxes of both sweeps, fractions, scalars and stable step. The exclusion "with a solid" is correct and quantifiable (fluxes at cut faces differ by up to $5.6\times10^{-4}$ of the donor volume, $5.9\,\%$ of the flux, in the cut fixtures).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-vof-frozen-flux/scripts/run_all.jl`. Last run: all checks passed (symbolic group and package group).
- Label: `eq:vof:frozen-flux` is the nearest label; the reduction sentence has no label of its own (label now in the source).

## Statement and assumptions

For a face segment $f$ with donor $c^{\rm u}$ and phase apertures $a^{\rm l}_f$, $a^{\rm g}_f$ (frozen for the step), each phase $\pi$ with $a^\pi_f>0$, $u^\pi_f\neq0$ sweeps the rectangle (5.9) over its subface $f^\pi\subseteq f^{\rm o}$ of width $\min\{a^\pi_f,a^{\rm o}_f\}\,w$, upwind of $u^\pi_f$, with the depth for which the open part of the rectangle holds the volume $\delta t|f|a^\pi_f|u^\pi_f|$ (no solid: $\delta t|u^\pi_f|/h$); the carrier is $\upsilon^{\rm o}_f=\sum_\pi a^\pi_fu^\pi_f$ and the liquid volume $\sum_\pi C^{\dr}_{f,\pi}\delta t|f|a^\pi_fu^\pi_f$ (5.23). The liquid subface is at the lower end of the open interval iff the averaged tangential interface normal is $\ge0$. Claim: with no solid ($a^{\rm o}_f=1$) and $u^{\rm l}_f=u^{\rm g}_f=u_f$ the two subface rectangles tile the one-velocity rectangle of (5.9)–(5.13) and the fluxes coincide.

Assumptions needed, used where indicated: (a) no solid: $a^{\rm o}_f=1$ and the donor is uncut, so the depth is the explicit $\delta t|u|/h$ independent of the aperture (S1); (b) equal velocities **on the face** (each segment, also at hanging faces); (c) $a^{\rm l}_f+a^{\rm g}_f=a^{\rm o}_f$: in the code $a^{\rm g}:=a^{\rm o}-a^{\rm l}$ (`_phase_face_apertures`), so the widths add up to the segment up to one rounding (S2, S3); (d) the subface positions follow the current reconstruction while the widths are frozen, so the tiling persists in the second sweep (S6).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 without a solid the depth of a subface rectangle is $\delta t|u|/h_k$, independent of $a^\pi$ and of the segment width | 609-616 (otherwise-case), code `distance = target / transverse_width` | ✅ | symbolic: $\text{target}/(a\,w)=\delta t|u|/h_k$ | `step01` S1 | target $=\delta t|u||f|a/V$, width $a\,w$, $w=|f|/h_t$, $V=h_kh_t$ |
| S2 the phase volumes add up to the one-velocity carrier iff $a^{\rm l}+a^{\rm g}=1$ | 618-621 | ✅ | symbolic; code: $a^{\rm l}+a^{\rm g}-1=0$ **exactly** on all 14 444 faces | `step01` S2, `code_01` | |
| S3 the subface intervals tile the segment (either phase at the lower end) | 595-607 | ✅ | symbolic S3 (adjacent end points, widths add to $w$); independent evaluation of the intervals (polygon-free, from the apertures and normals) on all faces: gap/overlap $\le1.1\times10^{-16}\,w$ | `step01`, `code_01` | also at 1 328 hanging faces |
| S4 the liquid area in the union of two adjacent rectangles is the sum of the liquid areas (plane/rectangle area is additive over the tiling) | implicit | ✅ | code: `_plane_fraction`$\cdot$area, 20 000 random planes and splits: $\le10^{-15}$ | `code_01` | |
| S5 stable time steps coincide | 632-637 vs 5.15 | ✅ | package: frozen `stable_timestep((u,u))` equals one-velocity `stable_timestep(u)` on all 16 meshes | `code_01` | relative difference exactly $0$ |
| S6 fluxes (open, liquid, gas) of **both** sweeps, fractions and scalars after a full step coincide | 638-640 | ✅ | package vs package on copies: 16 random periodic meshes (levels 1 and 2, isotropic and anisotropic), both orders, Upwind and Lax–Wendroff scalars | `code_01` | fluxes $\le2.1\times10^{-15}$ (relative to the largest flux), fractions $\le2.2\times10^{-16}$, scalars $\le9.0\times10^{-14}$ |
| S7 with a solid the fluxes differ ("differ in shape") | 640-641 | ✅ | docs fixtures `:cut`, `:film`, equal velocities, same step: flux difference at cut faces up to $5.6\times10^{-4}$ donor volumes ($5.9\,\%$ of the flux; 224 faces); at faces away from the solid $5.6\times10^{-17}$ (512 faces) | `code_01` | the frozen path needs the stable step halved on `:film` (strip error, see `vof-boundedness`) |
| S8 unequal velocities: fluxes differ | trivial | ✅ | $>10^{-6}$ difference at $\delta t=0.01$ | `code_01` | confirms hypothesis (b) is used |

## Errors and consequences

No error. The statement is correct as printed; it can be made precise. The reduction is exact in exact arithmetic and holds to rounding: $a^{\rm l}+a^{\rm g}=1$ held exactly in all 14 444 faces (complement defined as $a^{\rm o}-a^{\rm l}$), the tiling error is at most $1.1\times10^{-16}$ of the segment, the flux difference $\le2.1\times10^{-15}$.

Remark on the with-solid case: the printed reason ("the one-velocity rectangle spans the whole face") is only half of it: the frozen subfaces are placed in the *open interval* of the face (width $a^{\rm o}_fw$) and each has the depth that holds its own volume in its own open strip, whereas the one-velocity rectangle spans the whole segment with a single depth; both the position and the depth differ. In uncut donors with $a^{\rm o}_f<1$ the frozen depth is $\delta t|u|/h$ (no aperture factor) while the one-velocity depth is $a^{\rm o}_f\delta t|u|/h$.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| reduction without solid, equal velocities | **sharp** (an identity up to rounding) | $\le2.1\times10^{-15}$ on 14 444 faces; fails with a solid (S7) and for unequal velocities (S8) | `code_01` | state "to rounding, on every face, in both sweeps" |
| $a^{\rm l}+a^{\rm g}=1$ | **used and exact** in floating point | max $\lvert a^{\rm l}+a^{\rm g}-1\rvert=0$ | `code_01` | — |
| "with a solid they differ" | **sharp** (cannot be weakened to equality) | differences up to $5.6\times10^{-4}$ donor volumes | `code_01` | give both reasons (position and depth) |
| equal velocities on every face | **needed** (a single differing face changes that face's fluxes) | S8 | `code_01` | — |

## Open points

- The reduction was verified for the carrier fluxes, the fractions and the cell scalars. The staggered phase scalars and phase momentum (Chapter 6) use interpolants of these phase volumes; the statement of Section 5.8.1 does not claim them.
- Pure donors: the one-velocity path returns the swept fraction $0$ or $1$ directly, the frozen path evaluates the plane (without solid) and the bisection (with a solid, even in pure donors). Without a solid both give the same value (checked implicitly by the equality of the fluxes on all faces); with a solid the frozen path can refuse a step the one-velocity path accepts (audit `vof-boundedness`).

## Proposed replacement text

`05_vof_transport.tex:638-641`, replace the sentence by

```latex
Without a solid and with $\velo^{\liq}_f=\velo^{\gas}_f$ on every face the two subface rectangles tile
the one-velocity rectangle: their widths add up to the segment because
$\phaseaperture{\liq}{f}+\phaseaperture{\gas}{f}=\aperture{f}=1$ (to rounding), both have the depth
$\dt\abs{\velo_f}/\cellwidth{\donor{f}}{k}$, and in the second sweep the widths are still the frozen ones
while the positions follow the current reconstruction. The carrier, the phase volumes and the
time-step bound then coincide with those of \cref{sec:vof:donating} up to rounding. With a solid
the subfaces lie in the open interval of the segment and each has the depth that holds its own volume
in its own open strip, whereas the one-velocity rectangle spans the whole segment with one depth, so
the fluxes differ.
```

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `662-670`. The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic tiling algebra (8 checks); package frozen vs one-velocity on 16 random graded meshes (14 444 faces, both sweeps, scalars), additivity of the area function (20 000 draws), hypotheses used (solid fixtures, unequal velocities). Outcome: ✅ verified, wording refinement.
