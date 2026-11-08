# Audit: eq:setting:patch-metric — symmetry of the patch moment tensor $\mathsf Y_v^\top\mathsf N_v$ on square cells

- Source: `docs/techref/chapters/03_discrete_setting.tex:253-305` (definition 256-259; the claim, 304-305: "That the tensor is symmetric on every patch of square cells is observed on the test meshes; a proof is pending (audit scheduled)"), commit `40004ba` on branch `docs/techref-notation` (the file is unchanged in the working tree; only `proof_review/` is untracked).
- Label: the claim has no `\label`; the folder is named after the equation it concerns. Suggest `\label{lem:setting:patch-symmetry}` for the lemma proposed below.
- Date: 2026-11-08
- Overall status: ⚠️ **the claim is verified** (proof, exhaustive enumeration, package); **two implementation facts violate its hypotheses in corner cases** and are reported as findings E1, E2 (neither is an error of the mathematics in the `.tex`).
- Rerun: `julia --project=@claude-symbolics proof_review/eq-setting-patch-metric/scripts/run_all.jl` (two sequential child processes: Symbolics group under `@claude-symbolics`, package group under the repository environment; flags `--heap-size-hint=3G -t 1`).

## Statement and assumptions

Notation of the source: $\mathcal F(v)$ the interior face segments ending in the vertex $v$; $\mathsf Y_v$ has rows $|f|\,\mathbf d_f^\top$ with $\mathbf d_f=\mathbf x_{c^+(f)}-\mathbf x_{c^-(f)}$ (shortest periodic image), $\mathsf N_v$ has rows $\tfrac12\mathbf e_{k(f)}^\top$. Hence

$$\mathsf T_v:=\mathsf Y_v^\top\mathsf N_v=\tfrac12\sum_{f\in\mathcal F(v)}|f|\,\mathbf d_f\,\mathbf e_{k(f)}^\top,\qquad
(\mathsf T_v)_{12}=\tfrac12\sum_{k(f)=2}|f|\,(\mathbf d_f)_1,\quad (\mathsf T_v)_{21}=\tfrac12\sum_{k(f)=1}|f|\,(\mathbf d_f)_2 ,$$

i.e. the off-diagonal entries are face-measure-weighted **tangential** displacements.

**Lemma (to be stated in the source).** Let all cells be square, $h_{\ell,1}=h_{\ell,2}$, and let two cells that share a face segment differ by at most one level. Let $v$ be a vertex that is not on a wall (equivalently $\#\mathcal F(v)\ge 3$; a wall vertex has exactly one face segment and takes the diagonal closure). Then $\mathsf T_v$ is symmetric and positive definite, with eigenvalues in $[\tfrac12h^2,\,3h^2]$ ($h$ = smallest cell width at $v$), so $\operatorname{sym}(\mathsf T_v)=\mathsf T_v$, the positive-definiteness guard never fires, $\mathsf K_v\mathsf Y_v=\mathsf N_v$ holds exactly and $(\nabla_h p)_f=\partial_kp$ for affine $p$ on every face whose two endpoints are not on a wall.

Assumptions actually used, with where they enter:

| # | Assumption | Used in | Stated in the source? |
|---|---|---|---|
| A1 | dyadic quadtree of isotropic splits; every leaf block has the same number of cells, so $h_{\ell,k}=2^{-\ell}h_{0,k}$ | the two vertex types (hanging at the midpoint of a coarse side; four corner cells) | implicitly (ch. 2) |
| A2 | square cells: `extent[1]/base_mesh[1] == extent[2]/base_mesh[2]` | $r=1$ in the telescoping step | yes (the claim) |
| A3 | **face grading**: cells sharing a face segment differ by $\le1$ level. (`eq:mesh:grading` in ch. 2 also covers corner contacts and periodic seams; the proof does not need the corner part.) | $\min(a,b)(a-b)/2=(a^2-b^2)/6$ | in ch. 2 (`eq:mesh:grading`), not referred to in the claim |
| A4 | $\mathbf d_f$ is the covering-space centre difference: $\lvert (\mathbf d_f)_k\rvert <L_k/2$ (fails only for a periodic direction of exactly two level-0 cells) | periodic seams | no |
| A5 | $v$ not on a wall | the tensor only exists for $\#\mathcal F(v)\ge3$ | yes (248) |

`block_size` and the absolute level play no role (sharpness table).

## Configurations (per-configuration verdict)

Complete list of vertex neighbourhoods on a 2:1-graded quadtree. Every quadrant at $v$ is covered by one leaf; either four leaves have a corner at $v$ (type C, 16 oriented patterns), or one leaf has $v$ in the relative interior of a side (type H, 4 orientations; by dyadic alignment and grading the two cells opposite it are exactly half as wide); a vertex inside a leaf has no face. Sides $\in\{h,2h\}$, F = fine, C = coarse, order (NE,NW,SW,SE). $\mathsf T/h^2$ for one orientation (the others permute/flip the sign of the off-diagonal):

| Configuration | Pattern | #faces | $\mathsf T/h^2$ | eig | Symmetric | Patches in the exhaustive enumeration | Package (interior / seam) |
|---|---|---|---|---|---|---|---|
| conforming interior (FFFF = CCCC) | FFFF | 4 | $\mathrm{diag}(1,1)$ | 1, 1 | ✅ | 1 175 753 | 38 256 / 794 |
| re-entrant corner of a refined region (3 fine, 1 coarse) | CFFF | 4 | $\frac14\begin{pmatrix}5&1\\1&5\end{pmatrix}$ | 1, 3/2 | ✅ | 171 024 | 207 / 40 |
| straight interface at a coarse-cell corner (2+2 adjacent) | CCFF | 4 | $\mathrm{diag}(\frac52,\frac32)$ | 3/2, 5/2 | ✅ | 169 232 | 3 684 / 506 |
| saddle / checkerboard corner (coarse diagonal) | CFCF | 4 | $\frac12\begin{pmatrix}3&\mp1\\\mp1&3\end{pmatrix}$ | 1, 2 | ✅ | 84 616 | 117 / 24 |
| convex corner of a refined block (1 fine, 3 coarse) | CCCF | 4 | $\frac14\begin{pmatrix}11&\mp1\\\mp1&11\end{pmatrix}$ | 5/2, 3 | ✅ | 169 008 | 695 / 136 |
| hanging vertex (coarse cell on one side) | H | 3 | $\mathrm{diag}(\frac32,\frac12)$ or $\mathrm{diag}(\frac12,\frac32)$ | 1/2, 3/2 | ✅ (any aspect ratio) | 934 048 | 5 098 / 611 |
| vertex on a wall | — | 1 | none (diagonal closure) | — | n/a | — | max #faces at a wall vertex = 1 |
| wall corner | — | 0 | none | — | n/a | — | — |
| periodic seam / image | translate of the above | same | same | same | ✅ if A4 | — | all six classes occur on seams |

(The enumeration counts are patches over all 83 522 quadtrees of depth ≤3 on a window of side 8, all interior lattice vertices; the "package" columns count the graded patches of the 89 meshes of `code_01`.) Corner-balanced vertices realise exactly these six classes and nothing else.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| S1: $\mathsf T=\frac12\sum\lvert f\rvert \mathbf d_f\mathbf e_{k(f)}^\top$; off-diagonals = weighted tangential displacements | tex 253-259; `gradient.jl:79-86,139-160` | ✅ | symbolic: $\mathsf Y^\top\mathsf N$ built from the faces equals the closed form $\frac14[h_xh_yF_{11},h_x^2F_{12};h_y^2F_{21},h_xh_yF_{22}]$ in all 20 configurations ($h_x,h_y$ symbols) | `step01` | the face rule (minus = lower/left cell, overlap segment, centre difference) mirrors `_match_sides!`, `_face_displacement` |
| S2: $\mathsf T$ symmetric $\Rightarrow\mathsf K\mathsf Y=\mathsf N$ | tex 261-266 | ✅ | symbolic in $h$ for all 20 configurations (square) and in $r$ for the symmetric stretched classes; algebra: $\mathsf P\mathsf Y=0$, $\mathsf N\mathsf S^{-1}\mathsf N^\top\mathsf Y=\mathsf N\mathsf S^{-1}\mathsf T^\top=\mathsf N$ when $\mathsf T^\top=\mathsf T=\mathsf S$ | `step01` | needs $\operatorname{rank}\mathsf Y=2$, implied by $\mathsf T$ positive definite |
| S3: only types C and H exist; sides $\in\{h,2h\}$ | claim 304-305 | ✅ | argument above + exhaustive enumeration of all 83 522 trees × all vertices (2 783 937 patches): every face-graded patch is type C or H | `step03` | the code's patches coincide with this model (S9) |
| S4: hanging: $\mathsf T=h^2\mathrm{diag}(\frac32,\frac12)$; $\mathsf T_{12}=\mathsf T_{21}=0$ for every aspect ratio | tex 273-278 | ✅ | symbolic, 4 orientations | `step01` | the two coarse-fine faces have equal length and tangential displacements $\pm h/2$ |
| S5: corner, one face per half-line: $\lvert f\rvert \,d_t=\min(a,b)(a-b)/2=g(a)-g(b)$, $g(s)=s^2/6$, for $a/b\in\{\frac12,1,2\}$ | proof step | ✅ | symbolic (identity for the three ratios; **fails for 4:1**); all 64 half-line faces of the 16 patterns read off the face construction | `step02` | $\lvert f\rvert =\min(a,b)$, tangential offset $(a-b)/2$ because both cells have a corner at $v$ |
| S6: $2(\mathsf T_{12}-\mathsf T_{21})=(r^2-1)\,[g(NE)+g(SW)-g(NW)-g(SE)]$ with $r=h_{\ell,1}/h_{\ell,2}$ | proof step | ✅ | symbolic; $=0$ for $r=1$ and any four sides | `step01`, `step02` | horizontal faces carry the extra $r^2$ |
| S7: $\mathsf T$ positive definite, $\lambda\in[\frac12h^2,3h^2]$ | tex 323-324 | ✅ | exact eigenvalues of the six classes | `step03` | the guard `isposdef` never fires on graded square meshes |
| S8: wall vertices carry no tensor | tex 248-251 | ✅ | model (all wall configurations: exactly one face) + package (max 1 face over all wall vertices) | `step04`, `code_01` | `_physical_vertex || length<=2` is redundant for walls |
| S9: periodic seams = covering-space translate | tex 238-239 | ✅ (under A4) | package: at every vertex (seams included) the patch equals the integer model built from the touching cells; all classes occur on seams | `code_01`, `code_03` | caveat E2 |
| S10: numerical spot check of the claim | tex 292-296 | ✅ | 89 meshes (walls / periodic in x, y, both; block sizes (2,2),(4,4),(2,4),(4,2),(6,6); $L=2,3$; dyadic and non-dyadic extents): 53 469 vertex patches, 50 199 tensor patches. On every graded patch: $\max\lvert \mathsf T_{12}-\mathsf T_{21}\rvert /\max\lvert \mathsf T\rvert \le4.8\times10^{-15}$ (exactly 0 on dyadic meshes), $\max\lvert \mathsf K\mathsf Y-\mathsf N\rvert \le2.6\times10^{-15}$, package patch = integer model to $1.6\times10^{-14}$, Cartesian patch = diagonal to $1.3\times10^{-14}$, assembled operator = sum of local blocks to $1.9\times10^{-16}$, affine jumps/pressures exact to $9\times10^{-15}$ on 95 823 faces with two complete patches | `code_01` | 31 ungraded patches (4 meshes, E1) are counted separately |
| S11: stretched counterexample consistent | tex 296-303 | ✅ | formula $\mathsf T_{12}-\mathsf T_{21}=\frac{r^2-1}{4}h_y^2(\sigma_{NE}+\sigma_{SW}-\sigma_{NW}-\sigma_{SE})$ equals the package on all stretched meshes (error 0.0); asymmetric classes are exactly CFFF, CFCF, CCCF; the fixture gives 121 patches / 8 asymmetric, **all 8 of class CCCF**, residual $3.45\times10^{-2}$ — equal to `generated/checks.tex` (121, 0, 121, 8, 3.5e-2) | `code_02` | "all at refinement corners" is correct; more precisely convex corners of refined blocks |
| S12: numbers of eq:setting:patch-example | tex 273-288 | ✅ | exact rational arithmetic: $\mathsf T=\mathrm{diag}(\frac32,\frac12)$, $\tau=\frac56$, $\mathsf K=[\frac49,-\frac19,\frac5{18};\dots;\frac79]$, two-point value $\frac13$ | `step05` | side check |
| S13: hypothesis "2:1-graded" holds for every mesh `BlockTreeMesh` builds | `02_mesh_amr.tex:64-74` (eq:mesh:grading) | ❌ for some extents | package | `code_04`, `code_05` | **E1** |

## Errors and consequences

No error in the mathematics of the `.tex`. Two implementation findings that falsify a hypothesis of the lemma in corner cases:

**E1 — `BlockTreeMesh` does not always produce a 2:1-graded mesh; `isbalanced` does not detect it** (`src/mesh/topology/mesh.jl:466-473` `_tree_box_intersects`, `500-516` `_root_window_indices`, used by `_balance!` 605-646; `isbalanced` 1007-1027 uses the same exact comparisons).
- Reproduction: `BlockTreeMesh(base_mesh=(8,8), block_size=(2,2), extent=(0.7,0.7), periodic=(true,true), max_level=4, refine=(key,o,e)->key.i==0&&key.j==0)` contains faces between blocks whose levels differ by 2 (size ratio 4), and `isbalanced(mesh)==true`. Same for `extent=3.3`, `0.3`, `0.1`, `0.37*8`, `5.353681183789219` (also without periodicity). Extents 1.0, 2.0, 0.5, 4.0, 1.5, 3.0, 5.0, 10.0, 12.0, 100.0, 1e-3, π, 2π were never affected (180 random refinement patterns each, `code_05`). For the affected extents 50–77 of 80 random refinement patterns (63–96 %) end with a face of ratio 4 (`code_05`); 160 of 300 random extents in [0.3, 12.3] do (`code_04`).
- Mechanism: the neighbour search prunes with strict float comparisons (`upper < low`, and `floor`/`ceil` of a float quotient for the root window). Touching block boundaries computed along different arithmetic paths differ by one ulp (e.g. `-8.9e-16`), the neighbour is treated as disjoint, the 2:1 refinement is never triggered. The tolerant `_touch` is applied only after the pruning.
- Diagnosis confirmed by an in-process experiment (no file edited): a tolerance of $10^{-9}$ of the node / root width in both functions removes every violation in all 640 scanned meshes (`code_05`).
- Consequence for the audited claim: on such 4:1 faces the square-cell tensor is **not** symmetric (explicit example: $\mathsf T=[\frac52,0;-1,\frac12]h^2$, $\max|\mathsf K\mathsf Y-\mathsf N|=0.625$), `_local_inverse_metric` symmetrises it silently and affine exactness is lost there; in the 4 offending meshes of `code_01`, 22 of 31 ungraded patches are asymmetric, and `GradientOperator(mesh)` even throws `ArgumentError("degenerate vertex patch…")` on 9 of the 160 offending random meshes of `code_04` (sym$(\mathsf T)$ not positive definite). The lemma itself is untouched: every graded patch of those meshes is symmetric.
- Suggested fix: do the neighbour search in integer lattice coordinates (`BlockKey` level/i/j scaled by $2^{L-\ell}$), or add a tolerance to both functions; make `isbalanced` tolerant or integer-based; add a regression test with `extent=0.7`.

**E2 — a periodic direction with exactly two level-0 cells makes `GradientOperator` throw** (`src/operators/gradient.jl:74-77` `_wrapped_delta`, used by `_face_displacement` 79-86). The seam face between the two cells has covering-space displacement $+h$, but the raw difference is exactly $-L/2$ and `round(-0.5)==-0.0`, so the wrapped value keeps the wrong sign ($-h$); $\mathsf T_{11}<0$ and `isposdef` fails (`code_03` (d); uniform `base_mesh=(2,2)` periodic, `(2,4)` with x periodic, `(4,2)` with y periodic all throw; $\ge4$ cells or a refined stripe are fine). Not silent (the guard catches it), but it is a spurious failure of a legal mesh. Fix: take the normal component of $\mathbf d_f$ from the cell widths, $(h_{c^-}+h_{c^+})/2>0$, instead of wrapping it.

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| $\mathsf T$ symmetric on graded square meshes | **sharp** (an identity, no slack) | $\mathsf T_{12}-\mathsf T_{21}=0$ exactly (integer model: all 2.7 M patches; observed 0.0 on dyadic package meshes, $\le4.8\times10^{-15}$ relative otherwise) | `step01`, `step03`, `code_01` | none |
| square cells | **sharp** (necessary) | $\mathsf T_{12}-\mathsf T_{21}=\frac{r^2-1}{4}h_y^2(\sigma_{NE}+\sigma_{SW}-\sigma_{NW}-\sigma_{SE})$: for every $r\ne1$ the convex corner of a single refined block (CCCF) is asymmetric; for a given mesh symmetry holds iff $r=1$ or every corner has equally many coarse cells on its two diagonals (CCFF, FFFF, hanging are symmetric for all $r$). Stretched fixture: $r=2$ gives $\frac34h_y^2$ at CCCF | `sharp_01`, `code_02` | no relaxation of the hypothesis in general |
| face grading (2:1) | **sharp** (necessary) | 4:1 hanging vertex at a quarter point: $F=[10,0;-4,2]$, $\mathsf T_{21}=-h^2\ne\mathsf T_{12}=0$; corner $(4,2,1,1)$: $F_{12}-F_{21}=-2$; 64 192 of 78 208 non-graded patches asymmetric; the identity $\min(a,b)(a-b)/2=(a^2-b^2)/6$ fails for $a/b=4$ | `sharp_02`, `step02`, `step03`, `code_03` | — |
| corner balance of `_balance!` | **not needed** | 2 048 face-graded patches with a 4:1 *diagonal* (e.g. sides (4,2,1,2)) are symmetric; all 35 face-graded corner patterns with sides in {1,2,4} symmetric | `step03`, `sharp_02` | state the hypothesis as face grading only |
| `block_size` | **irrelevant** | patches depend only on the cells touching $v$; all classes occur for block sizes (2,2),(4,4),(2,4),(4,2),(6,6) and package patch = integer model on each | `code_01` | — |
| level / `max_level` | **irrelevant** | only size ratios enter; $\mathsf T\propto h^2$ | `step03`, `code_01` | — |
| dyadic structure | needed | the hanging vertex is at the midpoint only because of alignment; off-midpoint hanging vertices are the 4:1 case | `sharp_02` | — |
| periodic images (A4) | needed | two level-0 cells: sign of seam displacement lost (E2) | `code_03` | fix `_face_displacement` |
| eigenvalue bounds $[\frac12,3]h^2$ | **sharp** (both attained) | hanging gives $\frac12h^2$, CCCF gives $3h^2$ | `step03` | — |
| positive definiteness of sym$(\mathsf T)$ for $r\ne1$ | thresholds known | CFFF $r\in(0.101,9.899)$, CFCF $(0.1716,5.828)$, CCCF $(0.0455,21.95)$; `GradientOperator` builds iff inside (tested at $\pm3$ %) | `sharp_01`, `code_02` | — |

## Open points

- For the `literature-scout`: (a) is the symmetric vertex-patch moment construction (affine-exact, energy-compatible) a known mimetic reconstruction (Bonelle–Ern 2014 is cited in the `.tex`; also Van der Plas 2017); (b) known consistency results for the MAC/staggered pressure gradient on 2:1-graded quadtrees (Losasso–Gibou–Fedkiw 2004; Popinet 2003) to cite next to the lemma.
- E1 contradicts ch. 2 (`02_mesh_amr.tex:70-74`, "The code enforces eq:mesh:grading by refinement-only balancing") for extents such as 0.1, 0.3, 0.7, 3.3; fix upstream (or qualify the sentence) before the lemma is published; after the fix `code_04`/`code_05` still pass and simply report zero violations.
- E2: decide between fixing `_face_displacement` and documenting a minimum of three level-0 cells per periodic direction (multigrid already needs ≥4).
- Suggest a unit test in `test/operators/gradient.jl`: random graded square meshes, `max|T12-T21| <= 1e-14` over all patches (the existing test checks `R*jump` only on one mesh).
- Wording check of tex 266-268 ("every face whose two endpoint patches are of type (patch-metric)") is right and necessary: a coarse--fine face with one wall endpoint gets the diagonal closure there and the gradient of an affine pressure is $g_k+\frac12g_t\,d_t/d_k$ (error $\frac12|g_t|/3=0.117$ for $0.3x_1-0.7x_2$ and $d_t/d_k=\pm\frac13$, independent of $h$); conforming wall-adjacent faces are exact (`code_03` (e)). A sentence in the text would help readers.
- Not checked: Float32 meshes; three dimensions (the package is 2-D); anisotropic refinement (not supported by the quadtree).

## Proposed replacement text for `03_discrete_setting.tex:304-305`

Replace "That the tensor is symmetric on every patch of square cells is observed on the test meshes; a proof is pending (audit scheduled)." by:

```latex
On square cells the tensor is symmetric on every patch of a mesh that satisfies
\cref{eq:mesh:grading} (only contacts across faces are used). Its off-diagonal
entries
$\frac12\sum_{\faxis{f}=2}\meas{f}\,(\disp{f})_1$ and
$\frac12\sum_{\faxis{f}=1}\meas{f}\,(\disp{f})_2$
collect the tangential displacements, which are non-zero only on faces between
cells of different levels. A vertex that is not on a wall has $\#\vpatch{v}\ge3$
and is one of two kinds. At a hanging vertex the two coarse--fine faces have the
same measure $h$ and tangential displacements $\pm\frac h2$, and the face between
the two fine cells has none, so $\patchmom{v}^{\transpose}\patchsel{v}=h^2\diag{\frac32,\frac12}$
(or $h^2\diag{\frac12,\frac32}$ for a horizontal interface). Otherwise four cells with
widths $s_{\mathrm{NE}},s_{\mathrm{NW}},s_{\mathrm{SW}},s_{\mathrm{SE}}\in\{h,2h\}$ have a
corner at $v$; the face on each of the four half-lines from $v$ joins cells of
widths $a,b$ with $a/b\in\{\frac12,1,2\}$, has measure $\min(a,b)$ and tangential
displacement $\frac12(a-b)$, and $\frac12\min(a,b)\cdot\frac12(a-b)=\frac1{12}(a^2-b^2)$. Hence
\begin{equation*}
  12\,(\patchmom{v}^{\transpose}\patchsel{v})_{12}
  =(s_{\mathrm{NE}}^2-s_{\mathrm{SE}}^2)+(s_{\mathrm{SW}}^2-s_{\mathrm{NW}}^2)
  =(s_{\mathrm{NE}}^2-s_{\mathrm{NW}}^2)+(s_{\mathrm{SW}}^2-s_{\mathrm{SE}}^2)
  =12\,(\patchmom{v}^{\transpose}\patchsel{v})_{21}.
\end{equation*}
The eigenvalues of $\patchmom{v}^{\transpose}\patchsel{v}$ lie in $[\frac12h^2,3h^2]$
($h$ the smallest cell width at $v$), so the positive-definiteness check never
fails, the symmetrisation changes nothing and $\rieszloc{v}\patchmom{v}=\patchsel{v}$
holds exactly. For cells with aspect ratio $r=\meshwidth{\lev}{1}/\meshwidth{\lev}{2}$ the
two off-diagonal entries differ by
$\frac{r^2-1}{12}\big(s_{\mathrm{NE}}^2+s_{\mathrm{SW}}^2-s_{\mathrm{NW}}^2-s_{\mathrm{SE}}^2\big)$
($s$ the cell heights), which is non-zero exactly at corners whose two
diagonals carry unequally many coarse cells; hanging vertices and corners with
two adjacent coarse cells stay symmetric. A single face between cells that
differ by two levels already breaks the symmetry for square cells.
```

(Optional, line 298: "all at refinement corners" $\to$ "all at convex corners of refined blocks".)

## History

- 2026-11-08: first audit of the claim at commit `40004ba` (source unchanged). Symbolic proof for all 20 oriented configurations, exhaustive enumeration of 83 522 quadtrees, 89 package meshes, stretched and hypothesis experiments. Outcome: claim verified; findings E1 (grading not guaranteed by `BlockTreeMesh` for some extents) and E2 (two-cell periodic direction throws). Final `run_all.jl` (47 s with warm caches): symbolic/exact-model group 332 tests pass, package group 1671 tests pass, exit code 0. Nothing in `src/`, `docs/` or `test/` was edited; `proof_review/` is untracked and uncommitted.
