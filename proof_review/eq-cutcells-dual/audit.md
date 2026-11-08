# Audit: eq:cutcells:dual — the open staggered volume (7.5) versus its definition (3.7)

- Source: `docs/techref/chapters/07_cutcells.tex:133-161` (eq. 7.5 at 146-149) and the definition `docs/techref/chapters/03_discrete_setting.tex:105-113` (eq. 3.7, `eq:setting:open-dual`; the no-solid limit is (3.3), 43-46), commit `1e1886c` on branch `docs/techref-geometry` (chapters unchanged in the working tree; only `proof_review/` is untracked).
- Date: 2026-11-08
- Overall status: ✅ **verified.** (7.5) is (3.7) written in the local coordinates of each cell: equal to $\le5.1\times10^{-15}$ on the cylinder fixture, on 2:1-graded AMR meshes with 88 hanging faces and on anisotropic cells ($1.7\times0.6$), evaluated independently in physical coordinates; without a solid $V^{\rm o}_f=V_f$ exactly on 2140 faces (80 hanging); the quoted example is reproduced ($V^{\rm o}_f=0.4903$, $a_fV_f=0.4294$); boundary half strips verified.
- Rerun: `julia --project=@claude-symbolics proof_review/eq-cutcells-dual/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`). Last run: all passed.

## Statement and assumptions

Definition (3.7): $V^{\rm o}_f=\frac12\sum_{c\in\mathcal C(f)}\lvert\Lambda_{c,f}\cap c^{\rm o}\rvert$, $f\in\mathcal F_{\rm int}$, where the strip $\Lambda_{c,f}$ of $c$ spans the full width of $c$ normal to $f$ and the tangential extent of $f$ ("bisects every open line segment normal to the face", Verstappen–Dröge). (7.5):

$$V^{\rm o}_f=\tfrac12\sum_{c\in\mathcal C(f)}\lvert c\rvert\,(\hat r^+_t-\hat r^-_t)\Big(1-\mathcal A\big(\hat s_c,\hat\phi^{\rm s}_c;[-\tfrac12,\tfrac12]\times[\hat r^-_t,\hat r^+_t]\big)\Big),$$

"with $\mathcal A\equiv0$ for uncut and the contribution zero for covered cells. A boundary face gets the half strip of its single cell. Without a solid $V^{\rm o}_f=V_f$, but in general $V^{\rm o}_f\ne a_fV_f$: for the face of Figure 7.2, $V^{\rm o}_f=0.490$ and $a_fV_f=0.429$." The cut-cell elliptic operator rescales its metric by $(V_f/V^{\rm o}_f)^{1/2}$.

Assumptions used: the open cell is the complement of the solid half-plane $\hat s_c\!\cdot\!\hat\xi\le\hat\phi^{\rm s}_c$ in local coordinates $\hat\xi=\hat S_c^{-1}(x-x_c)$; the strip's local image is the rectangle with axes ordered $(k,t)$, $k$ the face normal (for $k=2$ the roles of the two intervals swap); a face segment at a level transition has tangential extent a half side of the coarse cell.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1 affine map $x=x_c+\operatorname{diag}(h)\hat\xi$: area of a polygon $=\lvert c\rvert\times$ local area (Jacobian $h_1h_2=\lvert c\rvert$) | 146-149 | ✅ | symbolic (shoelace of a generic quadrilateral) | `step01` | |
| S2 strip image: $[-\frac12,\frac12]\times[\hat r^-_t,\hat r^+_t]$, physical area $\lvert c\rvert(\hat r^+_t-\hat r^-_t)=h_k\lvert f\rvert$; the solid half-plane is $(\hat s_1/h_1)x_1+(\hat s_2/h_2)x_2\ge\hat\phi^{\rm s}+\hat s\!\cdot\!\hat S^{-1}x_c$ in physical coordinates | 140-146 | ✅ | symbolic | `step01` | |
| S3 no solid: $\frac12\sum_c\lvert c\rvert(\hat r^+_{t,c}-\hat r^-_{t,c})=\lvert f\rvert\tfrac12(h_{c^-,k}+h_{c^+,k})=V_f$ of (3.3), since $\lvert f\rvert=h_{c,t}(\hat r^+_{t,c}-\hat r^-_{t,c})$ for both cells | 152 | ✅ | symbolic + package | `step01`, `code_01` | exact (0.0) on 2140 faces, 80 hanging |
| S4 (7.5) $=$ (3.7): code `geometry.face_dual_volumes` and (7.5) as printed (with the package's closed-form $\mathcal A$ on $[-\frac12,\frac12]\times[\hat r^-_t,\hat r^+_t]$) equal the strip $\cap$ open cell evaluated **in physical coordinates** (physical rectangle clipped by the physical half-plane, Sutherland–Hodgman + shoelace; no $\mathcal A$, no local frame) | 146-149; 03:105-109 | ✅ | package on three fixtures (below) | `code_01` | all faces: 1.1e-15, 2.2e-15, 5.1e-15 |
| S5 covered cells contribute zero; uncut cells the full strip | 150-151 | ✅ | package | `code_01` | 14 / 345 faces with $V^{\rm o}_f=0$ (cylinder / AMR) |
| S6 boundary faces: half strip of the single cell | 151-152 | ✅ | package (`boundary_dual_volumes`, per-axis order) | `code_01` | 32 / 92 boundary faces; $\le8.9\times10^{-16}$ |
| S7 $V^{\rm o}_f\ne a_fV_f$ in general; counterexample: minus cell cut by a plane parallel to the face ($\hat s=(1,0)$, $\hat\phi^{\rm s}=0$, plus cell uncut): $V^{\rm o}_f=\frac34V_f$, $a_f=1$ | 152-157 | ✅ | symbolic (exact) + package | `step01`, `code_01` | differs on 60 of 112 faces (cylinder), 198 of 1672 (AMR); up to 44.7 % of $V_f$ |
| S8 quoted numbers of Figure 7.2: face between cells (6,6) and (7,6) of the cylinder fixture: $V^{\rm o}_f=0.490$, $a_fV_f=0.429$ | 154-157 | ✅ | package (independent strip areas) | `code_01` | $0.4903$, $a_f=0.4294$, $V_f=1.0$; equals `geometry.face_dual_volumes` to $10^{-14}$ |

Fixtures of S4/S7 (`code_01`): (a) the cylinder fixture of the techref (8×8 unit cells, centre (4.3,3.6), radius 2.6): 112 interior faces (74 with a partially open strip, 14 closed), 32 boundary faces; (b) a 2:1-graded AMR mesh (16×8 cells at level 0, blocks of 4×4, refinement window, `max_level` 2) with an elliptic solid: 1672 interior faces (88 hanging, 543 partially open strip, 345 closed), 92 boundary faces, unit cells and anisotropic cells ($h_x\times h_y=1.7\times0.6$); (c) the AMR mesh with a level set that is positive everywhere (no solid): 2140 faces (80 hanging).

## Errors and consequences

No error. (7.5) is the definition (3.7) in local coordinates; the statements after it are correct, including for hanging faces (strip of the coarse cell over the half side) and anisotropic cells.

Remarks (no correction needed):
- (7.5) writes the rectangle as $[-\frac12,\frac12]\times[\hat r^-_t,\hat r^+_t]$ "axes ordered $k,t$": for faces with normal $k=2$ the interval $[\hat r^-_t,\hat r^+_t]$ belongs to the first local axis. The code handles both (`bounds = Axis == 1 ? (-0.5,0.5,lo,hi) : (lo,hi,-0.5,0.5)`); the text says so.
- 345 of the 1672 faces of the AMR fixture have $V^{\rm o}_f=0$ (both cells covered, or both strips fully inside the solid); `03_discrete_setting.tex:402-405` already documents the replacement of a vanishing $V^{\rm o}_f$ by 1 (issue 42, units-dependent).

## Sharpness

| Claim / assumption | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| (7.5) $=$ (3.7) | **sharp** (an identity: affine image of an area) | symbolic S1/S2, $\le5.1\times10^{-15}$ on 1672 faces incl. hanging and anisotropic | `step01`, `code_01` | — |
| $V^{\rm o}_f=V_f$ without a solid | **sharp** (exact) | 0.0 on 2140 faces (80 hanging) | `code_01` | — |
| $V^{\rm o}_f\ne a_fV_f$ | **sharp** (non-equality attained) | $V^{\rm o}_f=\frac34V_f$ vs $a_fV_f=V_f$ for a cut parallel to the face; up to 44.7 % deviation on the fixtures | `step01`, `code_01` | — |
| boundary face = half strip | **sharp** | $\le8.9\times10^{-16}$ | `code_01` | — |
| hanging-face strip extent (half side of the coarse cell) | **needed** | 88 hanging faces, all equal to the physical evaluation | `code_01` | — |

## Open points

- None. (Out of scope: the replacement of a vanishing $V^{\rm o}_f$ by one in the cut-cell metric, issue 42.)

## Proposed replacement text

None required. Optional clarification after (7.5) in `07_cutcells.tex:150`: "(for a face normal to axis 2 the interval $[\rlo{t},\rhi{t}]$ is the first local axis)".

## History

- 2026-11-08: first audit (symbolic affine-map and no-solid reduction, counterexample for $V^{\rm o}\ne aV$; package vs an independent physical-coordinate evaluation of (3.7) on the cylinder, a 2:1-graded AMR mesh and an anisotropic AMR mesh; quoted numbers reproduced). Outcome: ✅ verified.
