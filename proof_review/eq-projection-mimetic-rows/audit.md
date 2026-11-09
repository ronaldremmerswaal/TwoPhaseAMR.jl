# Audit: eq:projection:joint-fit (group) — mimetic rows: aperture vector, smoothed rows, face rows, joint fit, rank selection

- Source: `docs/techref/chapters/10_projection.tex:649-767` (aperture-vector 652, smoothed-rows 662, face-rows 681, joint-fit 715; "Rank and fallback" 756), commit `b1e7b5b`
- Code audited (normative): `src/two_phase/projection/mimetic_smoothed.jl` (`_local_aperture_normal_tangent_reconstruction`, `_local_one_ring_aperture_vectors`), `mimetic_face_rows.jl` (`_local_joint_face_rows`, `_local_physical_normal_row`), `mimetic_joint_fit.jl` (`_local_joint_constraint_operator`, `_local_joint_soft_row!`), `mimetic_projection.jl` (`_local_integrated_aperture_vectors`, `_local_independent_rows`)
- Date: 2026-11-09
- Overall status: ✅ verified — an independent re-implementation of the rows and of the whole fit from the text reproduces the code's fitted rows to $\le8\times10^{-12}$ on 4 chapter fixtures and 6 stress fixtures (both variants); two small precision remarks (W1 planar length, W2 counts in the sentence on moments)
- Rerun: `julia --project=@claude-symbolics proof_review/eq-projection-mimetic-rows/scripts/run_all.jl`

## Statement and assumptions

(aperture-vector) $\mathbf A_c=\sum_fo_{c,f}|f|a^{\rm l}_f\mathbf n_f=|c|\,\mathfrak D(a^{\rm l}\mathbf n)_c$ (boundary faces with $s_f$), the interface area vector up to the sign; its one-ring sum $\mathbf A^{\rm s}_c$ over face neighbours (once per shared face) that are mixed with $\mathbf A_c\!\cdot\!\mathbf A_{c'}>0$; (smoothed-rows) rows for every mixed cell with open fraction $\ge1-\epsilon_f$ and $|\mathbf A_c|>64\epsilon\max(h_{\max},1)$: $\rho_c=|\mathbf A_c|\mathbf A^{\rm s}_c/|\mathbf A^{\rm s}_c|$, $\tau_c=(-\rho_{c,2},\rho_{c,1})$, $(\mathsf N^0[u])_c=\sum_k\rho_{c,k}\sum_{f\in c\cap\mathcal F^k\cap\mathcal F_\xi}|f|[u]_f/\sum|f|$, scale $|\mathbf A_c|$ ("the interface length of a planar cut"); (face-rows) $\mathsf N^0=[u]_f+R_f\sum w_{f,h}[u]_h$ at the active faces with $\beta_f\ge\tfrac12$ and all transverse faces active, scale $\sqrt{1+R_f^2}$, fallback rows from the face aperture vector; (joint-fit) supports by graph radius 2, geometric base rows with weights $\propto(\sum_i(\Delta x_i/h_i)^2+\tfrac1{16})^{-1}$, exact moments, 4 pressure observations and commutation observations (weights $10^3$), ridge $\tfrac1{10}$, cap 1, 3 alternating sweeps, fallback to the base row; rank selection by sparse QR with column pivoting of the normalised transpose.

**Assumptions made explicit.** The active set is that of `eq-projection-kkt` (verified); walled box (the periodic "nearest image" is not exercised); the fit is *soft* (the pressure/commutation conditions hold approximately by design).

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| R1 aperture vector (faces + boundary faces) = code's integrated vector | 649-661 | ✅ verified | independent sum with `data.liquid_aperture` and boundary apertures | `code_01_rows_fit.jl` | $\le10^{-13}$ |
| R1' $|\mathbf A_c|$ = interface length of a planar cut | 646-648 | ⚠️ planar-exact for shallow tilts (W1) | PLIC segment length per cell | `code_02_apvec_length.jl` | $3\times10^{-13}$ for tilts $(0,-1),(0.3,-1)$ (uniform and composite); $5.7\,\%$ for the steep tilt $(1,-0.6)$; on the drop $|\mathbf A|/\text{length}-1$ up to $0.26$ (mean $0.04$) |
| R1'' $\mathbf A_c$ is antiparallel to the PLIC normal $\eta_c$ (into the gas), "up to the sign" | 649-652 | ✅ verified | $\cos(\mathbf A,\eta_c)$ | `code_02_apvec_length.jl` | $\le-0.9994$ (planar), $-0.976$ on the drop |
| R2 smoothed rows: one-ring sum, $\rho,\tau$, active-face area averages, unresolved rule, scales | 662-677 | ✅ verified | independent rows vs `reconstruction.normal/tangent`, 10 fixtures | `code_01_rows_fit.jl` | max entry difference exactly $0.0$ (15-48 rows) |
| R3 face rows: selection ($\beta\ge\tfrac12$, all transverse faces active), coefficients, scales; aperture-aligned fallback rows | 678-701 | ✅ verified | independent rows vs `_local_joint_face_rows`; the fallback path occurs in the thin-slab fixtures (64 and 96 rows) | `code_01_rows_fit.jl` | max entry difference exactly $0.0$ |
| R4 joint fit: supports, geometric base rows, observations, weights, ridge, cap, sweeps | 702-751 | ✅ verified | complete independent implementation (graph supports, base weights with the $\tfrac1{16}$ offset, 4 pressure + commutation observations, exact moment origin, ridge fit in the moment null space, 3 sweeps) | `code_01_rows_fit.jl` | $\max|\mathsf N_{\rm indep}-\mathsf N_{\rm code}|/\max|\mathsf N|\le8.4\times10^{-12}$ (joint_face), $\le1.8\times10^{-12}$ (joint_smoothed); on the thin slab ($0$ difference) both count the geometric fallbacks (64/96) |
| R4' fitted rows reproduce the moments exactly (so a constant jump satisfying the base rows satisfies the fitted ones) | 746-751 | ✅ verified | $\mathsf N\mathsf c$ vs $\rho$ | `code_01_rows_fit.jl` | $\le6.2\times10^{-15}$ |
| R4'' "the same $\mathsf N$ appears in the constraint and in the stationarity row" so symmetry, momentum, energy are unaffected | 750-755 | ✅ verified | KKT symmetric to 0.0; Pythagoras $\le10^{-15}$ (`eq-projection-kkt`) | | |
| R5 rank selection: retained rows independent, span the candidates, original order, earlier of dependent rows kept | 756-767 | ✅ verified | QR-pivot retention on synthetic dependent rows; all fixture row sets have full rank (no dependent rows occur) | `code_03_rank_selection.jl`, `code_01_rows_fit.jl` | rank 4 of 7 rows, retained $[1,2,3,4]$, dropped later duplicates/combinations |
| R6 pressure observation: for an affine $p$, $\xi_f=\mathsf D^{-1}a^\Xi\mathsf Jp$ gives the jump of the scaled tangential derivative $[\![\rho^{-1}]\!]\tau\!\cdot\!\nabla p$ (thesis (7.20)) | 735-741 | ✅ verified (algebra) | with $a^{\rm l}=\hat C$ on a uniform mesh $a^\Xi/(m^{\rm r}V)=(\rho^{\rm g\,-1}-\rho^{\rm l\,-1})/V$, $(\mathsf Jp)_f=|f|\,\mathrm{disp}_f\!\cdot\!\nabla p$ | (derivation) | the code's target $(\rho^{\rm g\,-1}-\rho^{\rm l\,-1})\tau_j$ is exactly this |

## Errors and consequences

No error. Remarks:

**W1 (aperture-vector, last sentence of the smoothed-rows paragraph).** "the interface length of a planar cut" is exact (to $3\times10^{-13}$) for the shallow tilts tested; with a steep tilt $(1,-0.6)$ the per-cell deviation reaches 5.7 % (cells whose boundary face apertures come from the wall closure), and on the curved drop $|\mathbf A_c|$ differs from the PLIC segment length by up to 26 % (mean 4 %). The row *scale* is therefore only a normalisation, not a length measurement; the text's wording "of a planar cut" is right. *Optional precision:* replace "the interface length of a planar cut" by "approximately the interface length in the cell ($3\times10^{-13}$ for a shallow planar interface, a few per cent on the curved fixtures)".

**W2.** In the sentence introducing the moments of the face rows ("The moments are the row of $\mathsf N^0\mathsf c$ and its rotation") the moment of a face row is $(1,R_f)$ for $\mathrm{axis}(f)=1$ and $(R_f,1)$ for axis 2 — not unit vectors; no change needed, but a reader might expect unit normals (the code normalises in the kink modes only).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| exact moment reproduction | **sharp** (round-off) | $\le6\times10^{-15}$ | `code_01` | — |
| the fit is soft: commutation/pressure observations only approximate | **likely sharp** (design) | table of the chapter (0.013-0.18 commutator; 58-88 % pressure rows within $h/L$) not recomputed | — | — |
| rank condition on $\mathsf N$ | sharp via `code_03` (dependent rows are removed; retained rank = numerical rank) | | `code_03` | — |
| patch radius 2, ridge $\tfrac1{10}$, weights $10^3$ | design constants (not optimised; no claim of optimality in the text) | | | — |

## Open points

- Periodic domains: the geometric weights use the "nearest periodic image" (`_wrapped_delta`); the walled-box fixtures do not exercise it. The `Periodic face displacement is ambiguous with 2 cells per period` issue (#36) is related.
- The fallback branch of the fit (`geometry_fallback`, thin slab) agrees between code and independent implementation; the branch "fit not accepted (moment error $>10^{-8}$)" never triggered in the 10 fixtures.

## History

- 2026-11-09: first audit (scripts `code_01_rows_fit` (10 fixtures × 2 variants), `code_02_apvec_length`, `code_03_rank_selection`). `run_all.jl`: PASS (3 code scripts; 10 fixtures).
