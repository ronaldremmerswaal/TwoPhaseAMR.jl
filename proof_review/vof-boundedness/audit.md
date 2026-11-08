# Audit: vof-boundedness — boundedness of the WY sweeps (transfer of thesis Lemma 3.3) and the cut-donor depth example (claims 1 and 2)

- Source: `docs/techref/chapters/05_vof_transport.tex:344-366` (paragraph "Boundedness." of Section 5.5, including the example with depth $1/\sqrt2$ at 361-366), the paragraph "Limitation at level transitions" `268-282`, the CFL bounds (5.15) `eq:vof:cfl` (378-386) and the sweep update (5.6) `eq:vof:sweep` (115-125); thesis Lemma 3.3 (`int_advection_split.tex`, printed *Lemma 3.3 (Boundedness of the WYA)* with (3.50)–(3.56)); issue #52. Commit `ec20986` (branch `docs/techref-vof`, tree clean). Code: `src/vof/fluxes.jl` (`_swept_fraction`, `_advect_wy!`), `src/vof/workspace.jl` (`stable_timestep`), `src/embedded/geometry.jl` (`_interior_open_aperture`, `_open_subcell_area`).
- Date: 2026-11-08
- Overall status: ⚠️ **claim 1: the argument is correct in the classes it covers, and boundedness really fails outside them; claim 2 verified.** Uncut donors (any carrier) and cut donors with single-face sides (one-velocity and projected carriers) satisfy the argument (numerics: no overlap, no violation). Outside, the proof gap is a real gap: with the face-trace averaging artefact (the donor's own plane closes a side, trace $0$, the uncut neighbour behind it is open, trace $1$, so $a_f=\tfrac12$) there are explicit counterexamples **at the stable time step**: a cut coarse donor with a hanging side (one-velocity carrier, $C=-0.107$ after the sweep, `advect!` raises "WY conditional fraction out of bounds") and the frozen carrier ($C=1.207$); the run-time check catches them, so "The run-time check is the only guarantee in general" is exactly right. With consistent straight walls no violation was found (only refusals of the step by the strip condition, issue #52). The example of claim 2 is exact in the package (depth $0.7071067811865475$) and not extremal (depth $1-\epsilon+\epsilon/\sqrt2\to1$; $>\tfrac12$ also for straight walls, up to $0.67$).
- Rerun: `julia --project=@claude-symbolics proof_review/vof-boundedness/scripts/run_all.jl` (symbolic group, then package group of six scripts incl. the randomised searches and the explicit counterexamples, about 4 minutes; flags `--heap-size-hint=3G -t 1`). Last run: all checks passed (see History).
- Label: none. Suggested: `\label{prop:vof:boundedness}` on the paragraph "Boundedness." (it states a result with hypotheses) and `\label{ex:vof:depth-triangle}` on the example.

## Statement and assumptions

**Claim 1 (as printed).** The argument of thesis Lemma 3.3 carries over when, in the sweep, the rectangles $\hat R_f(\hat d_f)$ of the faces through which a cell donates are pairwise disjoint: the outflow of the phase that the cell moves conservatively (liquid if $\chi_c=0$, gas if $\chi_c=1$) is at most the volume of that phase in the reconstruction, so that phase fraction stays $\ge0$; its inflow over both sweeps is at most $\nu^{\rm in}_c\le\nu^\dagger\le\tfrac12$, so it stays $\le\tfrac12+\nu^{\rm in}_c\le1$; up to rounding and the at most $\varepsilon^C$ added by a snap after the first sweep $0\le C\le1$ before clamping. The rectangles are disjoint in uncut donors ($\hat d_f\le\nu^{\rm ext}_f\le\tfrac12$: opposite sides in opposite halves, segments of one side with disjoint tangential extents) and in a cut donor whose two sides normal to the sweep are single faces (two overlapping full-width rectangles would cover the cell, so their open volumes would sum to $\ge|c^{\rm o}|$, whereas $\nu^{\rm out}_c\le\nu^\dagger$). "For a cut coarse donor with two segments on a side neither argument applies." "The run-time check is the only guarantee in general." The related limitation: the depth (5.10) exists (bisection succeeds) for uncut donors and for cut donors with single-face sides because of the outgoing bound; for a coarse cut donor the half-strip behind a segment can hold less open volume than the segment's carrier volume (fixture of issue #52).

**Claim 2 (as printed).** In cut donors $\hat d_f$ can exceed $\tfrac12$: for the open part the triangle below the anti-diagonal and an uncut neighbour behind the face, $a_f=\tfrac12$, the open area of the full-width rectangle is $\hat d^2/2$, so $\nu^{\rm out}_c=\nu^{\rm ext}_f=\tfrac12$ gives $\hat d_f=1/\sqrt2$.

**Hypotheses made explicit** (the *exact* sufficient set verified here): (H1) $\nu^{\rm in}_c,\nu^{\rm out}_c,\nu^{\rm ext}_f\le\nu^\dagger\le\tfrac12$ at the sweep's step (all three are imposed by the code, `stable_timestep`); (H2) the carrier moves the physical liquid with full-width rectangles (one-velocity; projected-liquid for the physical liquid; the auxiliary gas carrier has its own bound, see below); the *frozen* carrier is covered only in uncut donors; (H3) every donor of the sweep satisfies disjointness in the moved phase — implied by "uncut", or "cut with a single face on both sides normal to the sweep and full-width rectangles"; (H4) the depth exists, i.e. no strip refusal. Without (H4) the step is refused; without (H3) the bound can fail (S6, S7: explicit counterexamples at the stable step).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 geometric reading of (5.6): $\chi=1$: conservative update of the gas with volumes $O_f-L_f$ (5.7); $\chi=0$: conservative update of the liquid | 128-141 | ✅ | symbolic S1 | `step01` | |
| S2 bound chain: $C'=C-{\rm out}+{\rm in}$ with $0\le{\rm out}\le C$, $0\le{\rm in}\le\nu\le\tfrac12$, $C\le\tfrac12$ gives $0\le C'\le1$, both extremes attained | 347-353 | ✅ | symbolic S2 (slack identities) | `step01` | |
| S3 outflow of the moved phase $\le$ its volume under disjointness; inflow $\le$ incoming carrier; chain $\chi=0$: $C\le\tfrac12+\nu^{\rm in}_{\le j}+\varepsilon^C$, $\chi=1$: $C\ge\tfrac12-\nu^{\rm in}_{\le j}-\varepsilon^C$ | 347-353 | ✅ | independent replica sweeps (polygon fluxes) on 78 random steps (plain and walled, 0–2 levels, anisotropic, one-velocity / frozen / projected carriers, step $=\delta t_{\rm WY}$): max excess of outflow over the phase volume $1.3\times10^{-16}$ (disjoint records), chain violation $0$ in every class, $C\in[-1.0\times10^{-16},\,1]$ | `code_01` | 78 steps: 77 accepted by replica and package alike, 1 refused alike (frozen strip) |
| S4 disjointness in uncut donors, any level structure and carrier | 353-356 | ✅ | proof: depth $=\nu^{\rm ext}$-type $\le\tfrac12$; numerics: 0 overlaps in 14 746 + 738 (hanging) one-velocity, 9 575 + 444 projected, 11 557 + 523 frozen donor records; observed max depth $0.433$ | `code_01` | |
| S5 disjointness in cut donors with single-face sides (full-width rectangles) | 357-360 | ✅ | proof (volume argument, needs $\nu^{\rm out}_c<1$, weaker than printed); numerics: 0 overlaps in 560 one-velocity and 450 projected records, max depth $0.432$ | `code_01`, `code_04` (uniform/one: 6 918 feasible random samples, 0 overlaps) | restricted to full-width rectangles (P1) |
| S6 cut donors with a hanging side, one-velocity carrier: not covered by the proof | 360-361 | ❌ **bound fails** (artefact apertures); ⚠️ none with straight walls | explicit counterexample (`code_06` (1)): donor open fraction $0.238$, $C=0.1597$ ($\chi=0$), only the upper right segment ($a_f=0.5$, depth $0.9925$) and the left face ($a_f=0.967$, depth $0.083$) donate, $\nu^{\rm out}=\tfrac12$ at the stable step $0.2057646$ (independent (5.15) equal), rectangles overlap in the moved phase (area $0.0255$), pre-clamp $C=-0.1071$; the package raises "WY conditional fraction out of bounds: -0.1071"; the replica stays in $[0,1]$ only for $\delta t\le0.65\,\delta t_{\rm WY}$. Search (`code_04`): artefact apertures 3 000 geometries: 584 samples with overlap in the moved phase, worst violation $0.107$ (hill climbing), confirmed by the package; straight walls: 3 000 geometries, 33 398 feasible samples, 0 overlaps, no violation | `code_04`, `code_06` | abstract model (`code_05`): convexity of the open part plus the outgoing bound alone allow overlap and double counting (458 overlapping samples of 38 574, 1 636 violations, worst $0.099\,\lvert c^{\rm o}\rvert$), so a proof needs the aperture/trace consistency |
| S7 frozen carrier in cut donors: not covered by the proof | 357-361, §5.8.1 | ❌ **bound fails** (artefact apertures); ⚠️ none with straight walls | explicit counterexample (`code_06` (2)): $C=0.756$ ($\chi=1$), pre-clamp $C=1.2068$ at the stable step $0.40359$ (independent (5.24) equal), overlap in the moved phase $0.0286$, package error "out of bounds: 1.2068"; search: artefact apertures worst violation $0.207$ (confirmed); straight walls: uniform cut mesh 3 000 geometries, 30 777 feasible samples, 730 with overlapping open rectangles (153 geometries), 5 inside the moved phase (min margin $0.039\lvert c^{\rm o}\rvert$), no violation; hanging mesh 492 open overlaps, none in the phase | `code_04`, `code_06` | 4 005 / 5 421 straight-wall samples refused (strip) |
| S8 depth existence (Limitation paragraph): holds for uncut and single-face cut donors, fails for coarse cut donors | 268-282 | ✅ | issue #52 reproducer: `advect!` raises "cut-cell face flux exceeds the donor's open subcell volume" at the step `stable_timestep` returns ($0.059375$); half-strip $0.025312$ vs target $0.026719$, $a_f=0.90$, donor open fraction $0.113$, mixed ($C=0.5014$); critical step $0.947368421054\,\delta t_{\rm WY}=({\rm strip}/{\rm target})\delta t_{\rm WY}$ ($0.9473684210527$, agreement $2\times10^{-12}$); random cut meshes (levels 0, 1, 2, 10 each): 0 refusals for the one-velocity path | `code_03` | the refusal is a per-segment condition $\delta t\le\delta t_{\rm WY}\cdot A_{\rm strip}/\tau_{\rm seg}$ not implied by (5.15) |
| S9 "(the donor is mixed; pure donors need no rectangle)" | 281-282 | ⚠️ one-velocity only | same geometry with pure liquid / pure gas interface: one-velocity runs at $\delta t_{\rm WY}$; the **frozen** carrier raises the strip error also for pure donors | `code_03` | P3 |
| S10 the example: strip area $d^2/2$, $a_f=\tfrac12$, $\nu^{\rm out}=\nu^{\rm ext}=\tfrac12$, $d=1/\sqrt2$ | 361-366 | ✅ | symbolic S3; package on a hand-made donor (stored solid plane $\{-(\xi+\eta)/\sqrt2\le0\}$, uncut neighbour): aperture $0.5$ (exactly), open fraction $0.5$, $\nu^{\rm out}=\nu^{\rm ext}=0.5$, independent depth $0.7071067811865475$, package area function at that depth $=$ target to $10^{-12}$, package flux $=$ independent flux to $10^{-14}$, step within bounds | `step01`, `code_02` (ii) | the plane is stored by hand; aperture, area, flux and step are the package's |
| S11 sup of the depth is $1$, not $1/\sqrt2$ | 361-366 | ✅ (extension) | symbolic S4 and package: open triangle with legs $\epsilon$ in the far corner: depth $1-\epsilon+\epsilon/\sqrt2$ for $\epsilon=0.5,\ldots,0.02$ ($0.8536\to0.9941$), $a_f=\tfrac12$, $\nu^{\rm out}=\tfrac12$, $\nu^{\rm ext}$ down to $2\times10^{-4}$ | `step01`, `code_02` (iii) | |
| S12 natural occurrence | 361-366 | ✅ (extension) | 3 000 random straight walls on a 6×6 mesh, 28 800 cut-donor faces at the stable step: depth $>\tfrac12$ on 3 faces, max $0.670$ (aperture exactly $0.5$: donor trace $0$, neighbour trace $1$, neighbour almost covered, open fraction $5\times10^{-4}$) | `code_02` (iv) | the artefact of the face-trace average (S10) occurs with consistent walls, rarely |

## Errors and consequences

No error in the argument for the classes it covers. Precisions and one caution:

**P1 (full-width rectangles).** The cut-donor volume argument ("two overlapping full-width rectangles would cover the cell") holds for the one-velocity carrier and for the physical liquid update of the projected carrier. For the frozen carrier the rectangles are over *phase subfaces* (partial width, placed in the open interval); the covering step fails, and so does the bound: counterexample `code_06` (2) ($C=1.207$ at the stable step). With straight walls overlapping open rectangles occur in 730 of 30 777 feasible samples (5 inside the moved phase, margin $\ge0.039\lvert c^{\rm o}\rvert$) without a violation. In uncut donors the frozen carrier is covered ($\hat d\le\nu^{\rm ext}\le\tfrac12$: 11 557 + 523 records, no overlap).

**P2 (what happens outside).** For cut coarse donors with two segments on a side the failure is not only a refusal (issue #52) but also a genuine loss of boundedness: `code_06` (1) shows $C=-0.107$ at the stable step with all of (5.15) satisfied, caught by the run-time check (an error, not silent). The mechanism is the face-trace average: a segment whose side is closed in the donor's plane but open in the uncut neighbour has $a_f=\tfrac12$, its rectangle (depth $0.99$) reaches through the cell and overlaps the rectangle of the opposite face in the open moved phase. This needs inconsistent traces; with consistent straight walls (3 000 geometries, 33 398 feasible samples) no overlap occurred and the only failure was the strip refusal (2 264 samples). So "neither argument applies" is a real gap, and the text should say that the bound can fail there.

**P3 (pure donors).** "pure donors need no rectangle" is true of the one-velocity (and projected) path; the frozen path always bisects when a solid is present, so a pure donor can trigger the strip error (reproduced with an all-liquid and an all-gas interface on the fixture of issue #52). On random cut meshes at the stable step the frozen path refused 11 % (single level) to 15 % (graded) of the random velocity samples of the search, the one-velocity path none on single-level meshes and 6.3 % on the graded mesh (2 264 of 35 662; random per-face velocities of the search, not typical flow fields) — the limitation is wider for the frozen carrier than the text of Section 5.4 suggests (the frozen section only says "with the error of Section 5.4 if the strip over the subface is too small").

**P4 (example not extremal).** $1/\sqrt2$ is one value of a family; the depth reaches $1-\epsilon+\epsilon/\sqrt2\to1$. For natural straight walls it exceeded $\tfrac12$ on 3 of 28 800 faces (max $0.670$).

**Remark on the thesis' lemma.** Lemma 3.3 states only the incoming condition (3.50)–(3.51); its lower-bound proof uses (3.55), the outgoing form that is equivalent to the incoming one only for divergence-free velocities (thesis text after (3.54)). The techref's added outgoing and extrusion bounds are therefore needed for its non-solenoidal carriers (deviation box of Section 5.6). On equal-size uncut meshes the incoming bound of *every* cell already limits each face's depth, because the volume leaving a cell through a face enters its neighbour: this is why the printed outgoing bound is not sharp there (`sharp_01` (B)).

Consequence: none for the numerics reported in the chapter (all fixtures run at the stable step); only wording.

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| $\nu^{\rm in}\le\tfrac12$ for the upper bound | **sharp** | $C=\tfrac12$ cell between two liquid cells: $C'=\tfrac12+\nu^{\rm in}$ exactly; reaches $1$ at $\nu^{\rm in}=\tfrac12$, `advect!` raises the out-of-bounds error at $\nu^{\rm in}=0.5000005$ | `sharp_01` (A) | — |
| outgoing / extrusion bound $\nu\le\tfrac12$ for the lower bound | **not sharp**; threshold = sum of depths $\le1$ | uniform mesh, both faces donating, diagonal interface: $C'=0.1$ at $s=0.4$ ($\nu^{\rm out}=0.8>\tfrac12$, still bounded), $C'=0$ at $s=0.5$ (rectangles touch), $C'=-0.05$ at $s=0.55$ (package error) | `sharp_01` (B) | state the lower bound with per-axis $\sum$ of depths $\le1$ (uncut) or $\nu^{\rm out}_c<1$ (cut, single faces); on equal-size meshes the receivers' $\nu^{\rm in}\le\tfrac12$ already forces depth $\le\tfrac12$ |
| single-face sides for cut donors | **needed** (the bound fails for a hanging side with artefact apertures) | counterexample `code_06` (1); control: uniform mesh with artefact apertures, 1 500 geometries, 16 951 samples, 0 overlaps | `code_04`, `code_06` | for consistent traces the question is open |
| full-width rectangles | **needed** (the frozen carrier violates the bound) | counterexample `code_06` (2) | `code_04`, `code_06` | a bound argument for subface rectangles (not found) |
| depth example $1/\sqrt2$ | **exact but not extremal** (sup $=1$) | S10, S11 | `code_02` | quote the family or the supremum |
| (ε added by a snap after sweep 1) | **consistent** | chain with $\varepsilon^C$ allowance holds, max $C=1.000000$ | `code_01` | — |

## Open points

- Whether boundedness holds when the traces are consistent (aperture equal to the donor's own trace: straight walls, convex corners resolved): no counterexample in 33 398 straight-wall samples, no proof; the counterexamples use the trace-averaging artefact. Literature search for the caller (`literature-scout`): boundedness proofs of geometric VOF on cut-cell/AMR meshes and of Weymouth–Yue [2010] with a general CFL. Possible code-side remedies (not audited): a per-segment strip/overlap check in `stable_timestep`, or merging the two segments' donating regions of a coarse donor.
- Boundedness of the *auxiliary gas carrier* (projected path): its lower bound is checked by the package ("WY auxiliary gas fraction out of bounds") and no violation occurred in the 26 projected steps of `code_01`; a proof was not attempted (it has no dilation term, so the argument is the $\chi=0$ branch for every cell).
- The frozen carrier's strip limitation (P3) may deserve its own issue, referencing #52.

## Proposed replacement text

`05_vof_transport.tex:357-366`, replace "They are also disjoint in a cut donor ... $\drdepth{f}=1/\sqrt2$." by

```latex
They are also disjoint in a cut donor whose two sides normal to $\unitvec{k}$ are single faces
when the rectangles span the whole side (one-velocity and projected carriers): two overlapping
full-width rectangles would cover the cell, so their open volumes would sum to at least
$\meas{\opencell{c}}$, whereas the outgoing bound limits them to
$\wycflval\meas{\opencell{c}}<\meas{\opencell{c}}$. Neither argument applies to a cut coarse
donor with two segments on a side, nor to the frozen carrier in cut donors, whose subface
rectangles span only part of the side, and there the bound can fail: if the donor's own plane
closes a side (trace $0$) while the uncut neighbour behind it is open (trace $1$), the face has
$\aperture{f}=\half$, the rectangle of such a segment can reach almost through the cell, overlap the
rectangle of the opposite face in the open moved phase and make its outflow exceed its volume with
all of~\cref{eq:vof:cfl} satisfied; the run-time check then stops the run
(\href{https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/issues/56}{issue~56} collects explicit
configurations at the stable step, for one velocity and for the frozen carrier).
\phantomsection\label{ex:vof:depth-triangle}In cut donors $\drdepth{f}$ can
exceed $\half$, and approach $1$: if the open part of the donor is the triangle below its
anti-diagonal and the neighbour behind the face is uncut, the traces of the face are $0$ and $1$, so
$\aperture{f}=\half$, and the open area of the full-width rectangle is $\hat d^2/2$, so
$\cflout{c}=\cflext{f}=\half$ gives $\drdepth{f}=1/\sqrt2$; for the triangle with legs $\epsilon$ in the
far corner, $\cflout{c}=\half$ gives $\drdepth{f}=1-\epsilon+\epsilon/\sqrt2\to1$. The bound
$\cflin{c}\le\half$ cannot be relaxed: a cell with $\condfrac{\liq,\timelevel{n}}{c}=\half$ that receives
liquid from both sides reaches $\condfrac{\liq}{c}=\half+\cflin{c}$.
```

`05_vof_transport.tex:281-282` (Limitation paragraph), replace "(the donor is mixed; pure donors need no rectangle)" by

```latex
(the donor is mixed: in the one-velocity path pure donors need no rectangle, whereas the frozen carrier
evaluates the strip over each phase subface also in pure donors and can refuse a step on a single-level cut
mesh as well); the largest accepted step is $\dtwy$ times the ratio of the open area of the strip to the carrier volume of the
segment ($0.947$ in this fixture; see \href{https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/issues/56}{issue~56})
```

**Applied** on 2026-11-08 to `docs/techref/chapters/05_vof_transport.tex` (uncommitted, branch `docs/techref-vof`), new-file lines `281-284` (Limitation paragraph), `346` (label `prop:vof:boundedness`), `359-379` (replacement with label `ex:vof:depth-triangle`); labels `prop:vof:boundedness`, `ex:vof:depth-triangle`; references to issue 56. The techref builds with 0 undefined references and `check_links.py` passes.

## History

- 2026-11-08: first audit. Symbolic chain, triangle and far-corner algebra (15 checks); independent replica sweeps on 78 random steps for three carriers (class statistics); the triangle example and its far-corner family realised in the package; issue #52 reproducer with critical-step identity; threshold experiments; randomised search with hill climbing for overlap-induced violations (4 classes, 9 600 geometries). Outcome: ⚠️ claim 1 verified in the covered classes with precisions P1–P4, claim 2 ✅.
- 2026-11-08 (later): randomised searches extended with the artefact apertures of the face-trace average (only the donor cut, stored plane by hand, neighbours uncut; `code_04`), an abstract single-cell model (`code_05`) and two explicit counterexamples reproduced in the package (`code_06`: one-velocity $C=-0.107$, frozen $C=1.207$ at the stable step). Outcome: the bound is not guaranteed outside the covered classes (status of claim 1 revised to ⚠️ with real failures); replacement text rewritten accordingly.
