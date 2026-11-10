# Audit: ch:solvers — Chapter 11 (operators and linear solvers): null space, projected V-cycle, smoother, GMRES stop and round-off floor, CMDGFM scaling, active-domain transfer, CG bound

- Source: `docs/techref/chapters/11_solvers.tex` (labels `eq:solvers:null`, `eq:solvers:energy`, `eq:solvers:floor`, `eq:solvers:gmres`, `eq:solvers:gershgorin`, `eq:solvers:jacobi`, `eq:solvers:active-transfer`, `eq:solvers:projectors`, `eq:solvers:cmdgfm-scaling`, `sec:solvers:vcycle`; no label on the proof-like paragraphs, the folder is named after the chapter label `ch:solvers`) and `docs/techref/figures/solvers.jl`, branch `docs/techref-solvers`. Audited at `df7eb6c`; the branch moved to `5c31fca` during the audit (one edit: the CMDGFM residual-defect reduction factors, `eq:solvers:cmdgfm-defect`, after l. 909; outside the claims below and not audited). Lines 1–909 of the chapter are identical at both commits. Code audited (normative, same tree): `src/operators/elliptic.jl`, `src/operators/staggered_diffusion.jl`, `src/embedded/operators.jl`, `src/solvers/multigrid/{cycles,hierarchy,geometric_preconditioner,elliptic_solver,options,stats}.jl`, `src/two_phase/projection/{common,cmdgfm_projection}.jl`, `src/navier_stokes/implicit_diffusion.jl`.
- Date: 2026-11-10
- Overall status: ⚠️ all seven claims hold (claims 1, 3, 6 exactly; claim 2 holds and is **stronger** than stated, it needs no restriction); **five wording defects with replacement LaTeX** (E1 the V-cycle paragraph relies on a citation where a six-line proof is available and omits that equal sweep counts are a hypothesis the code does not enforce; E2 `eq:solvers:gmres` differs from the code by the cap $\min\{1,\cdot\}$; E3 claim 5 "changes only the norm" is incomplete; E4 the condition-number paragraph compares quantities in different norms; E5 the composite-mesh viscous sentence holds only without a solid) and one **observation about the code** (embedded one-phase viscous matrices are symmetric but routed to BiCGSTAB)
- Rerun: `julia --project=docs/techref/figures proof_review/ch-solvers/scripts/run_all.jl` (no Symbolics environment is needed; flags for the shared machine: `--heap-size-hint=3G -t 1`, `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_PRECOMPILE_TASKS=1`; the driver sets them for its child processes; about 6 minutes)

## Statement and assumptions

Claims audited (line numbers of the chapter at `5c31fca`):

1. (l. 107–150, `eq:solvers:null`) $\ker\mathsf A=\operatorname{span}\{\mathbb 1_{\mathcal C}\}$ over the unanchored components $\mathcal C$ of the graph of active cells joined by interior faces with $\kappa_f\ne0$; anchored = a cell with $\gamma_{cc}>0$ (open boundary faces) or sharing a face with $\kappa_f\ne0$ with an inactive cell; and $\mathsf A\mathsf x=\mathsf b$ is solvable iff $\mathbb 1_{\mathcal C}^{\top}\mathsf b=0$ on every unanchored $\mathcal C$.
2. (l. 525–539) The V-cycle with projections is symmetric positive definite on the mean-free space without Galerkin coarse operators (Tatebe §§4–5, plus the paired projections); checked in the chapter on 16²/32², a 2560-cell composite mesh, the drop and the split domain.
3. (l. 439–446, 465–478) $\omega_l\lambda\le\tfrac43$ for the eigenvalues $\lambda$ of $\mathsf D^{-1}\mathsf A_l$ ($\omega_l=\min\{1,4/(3g_l)\}$, $g_l$ the Gershgorin bound); $g=2$, $\omega=2/3$ on the uniform five-point operator.
4. (l. 214–226, 244–258) The GMRES/FGMRES stop `eq:solvers:gmres`; the factorised round-off floor uses an entrywise larger $|\mathsf A|$ than the assembled one, so it is never smaller.
5. (l. 740–752) The CMDGFM scaling $\hat{\mathsf A}=\mathsf D_r\mathsf A\mathsf D_c$, $\hat{\mathsf P}=\mathsf D_c^{-1}\mathsf P\mathsf D_r^{-1}$ leaves $\hat{\mathsf A}\hat{\mathsf P}$ similar to $\mathsf A\mathsf P$; "the scaling changes the norm in which BiCGSTAB measures its residual, not the preconditioned spectrum".
6. (l. 447–462, `eq:solvers:active-transfer`) The restricted row-normalised transfer satisfies $\mathsf P\mathbb 1^{\rm act}_{l+1}=\mathbb 1^{\rm act}_l$; an unrestricted transfer would send coarse corrections into covered cells.
7. (l. 540–560) Reading of the condition numbers $\kappa(\mathsf B\mathsf A)=1.27$–$1.52$ against the CG bound $((\sqrt\kappa-1)/(\sqrt\kappa+1))^k$ and the measured residual reductions $0.08$, $0.12$.
8. Side question: the chapter says the composite-mesh one-phase viscous matrix is nonsymmetric (BiCGSTAB); is that expected after the diffusion fixes #79/#87?

**Assumptions made explicit.** (a) $\kappa_f\ge0$, $\mathsf R_{\rm open}$ symmetric positive definite on the faces with $\kappa_f\ne0$ (for a solid $\mathsf R_{\rm open}=\mathsf S\mathsf R\mathsf S$ with $\mathsf S$ the dual scaling and $\kappa_f=\mathrm{material}\cdot a_f^2$, so $\kappa_f=0$ exactly where $a_f=0$); "$\ne0$" is an exact test in floating point (`iszero` on `face_coefficient`). (b) Claim 2: the level operators are symmetric positive semidefinite with positive diagonals (isolated cells excepted), Jacobi with $\omega_l\lambda_{\max}(\mathsf D^{-1}\mathsf A_l)<2$ on every level, $\nu_1=\nu_2$, restriction $=$ prolongation$^{\top}$; nothing else (see the proof below). (c) Claim 7 uses the Ritz extremes of the PCG Lanczos matrix as $\kappa$ (equal to the dense values of the chapter to the printed digits). (d) Claim 8 concerns `StaggeredEllipticSolver` with `krylov=nothing`.

**Proof of claim 2 (new; `scripts/step02a_vcycle_algebra.jl`).** Fix a level with SPSD $\mathsf A$, $\mathcal W=\{\mathsf x:\mathsf v_{\mathcal C}^{\top}\mathsf x=0\ \forall\mathcal C\}$ and $\langle\mathsf x,\mathsf y\rangle_A=\mathsf x^{\top}\mathsf A\mathsf y$, an inner product on $\mathcal W$ (since $\ker\mathsf A\cap\mathcal W=\{0\}$). Let $\mathsf S=\Pi(\mathsf I-\omega\mathsf D^{-1}\mathsf A)$ ($\Pi$ the mean removal) and let $\mathsf B_c$ be the preconditioner of the next level, symmetric positive semidefinite. The error of one cycle on $\mathcal W$ is $\mathsf I-\mathsf B\mathsf A=\mathsf S^{\nu_2}\mathsf T\mathsf S^{\nu_1}$ with $\mathsf T=\mathsf I-\Pi\mathsf P\mathsf B_c\mathsf P^{\top}\mathsf A$. (i) $\mathsf A\Pi=\mathsf A$ (the indicators are null vectors), so $\mathsf S$ is $\langle\cdot,\cdot\rangle_A$-selfadjoint ($\mathsf A\mathsf S=\mathsf A(\mathsf I-\omega\mathsf D^{-1}\mathsf A)$ is symmetric) and $\mathsf A\mathsf T=\mathsf A-\mathsf A\mathsf P\mathsf B_c\mathsf P^{\top}\mathsf A$ is symmetric; (ii) $\langle\mathsf T\mathsf x,\mathsf x\rangle_A=\|\mathsf x\|_A^2-(\mathsf A\mathsf x)^{\top}\mathsf P\mathsf B_c\mathsf P^{\top}(\mathsf A\mathsf x)\le\|\mathsf x\|_A^2$; (iii) $\|\mathsf S\mathsf x\|_A<\|\mathsf x\|_A$ for $\mathsf x\in\mathcal W\setminus\{0\}$ iff $0<\omega\lambda<2$ on the nonzero spectrum of $\mathsf D^{-1}\mathsf A$. With $\nu_1=\nu_2=\nu$: $\langle\mathsf B\mathsf A\mathsf e,\mathsf e\rangle_A\ge\|\mathsf e\|_A^2-\|\mathsf S^{\nu}\mathsf e\|_A^2>0$ and $\mathsf B\mathsf A$ is $A$-selfadjoint, so $\mathsf B$ is symmetric positive definite on the range of $\mathsf A$ and vanishes on $\mathsf v_{\mathcal C}$. The coarsest level is the top-left block of the inverse of the symmetric bordered matrix (`eq:solvers:augmented`), which is symmetric positive semidefinite; induction downward. **No relation between $\mathsf P$ and $\mathsf A$ and no Galerkin product is used.** Tatebe's Theorems 2–3 (§§4–5, read in the NTRS scan) are the same argument for nonsingular levels, with an arbitrary SPD coarse operator and $\mathsf r=b\mathsf p^{\top}$, $b$ a scalar; the chapter's "paired projections" are the extension proved here.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1a. $\ker\mathsf A$ = span of unanchored indicators; $\dim\ker$ = number of unanchored components; spectrum separated from 0 | `eq:solvers:null`, l. 107–140 | ✅ verified | independent union-find (`predicted_components`) vs dense eigen-decomposition of the assembled matrix: split cut-cell 32² (closed: 2 components; Outflow top: 1; Outflow right, bottom+top: 0), composite 2-level mesh with a solid and a **non-diagonal metric** (closed/Outflow), composite periodic, 60 random cases ($:$dirichlet / $:$decoupled, random inactive masks (25 %), random zero face coefficients (30 %), random open sides and zero boundary coefficients, uniform and composite; 127 unanchored components in total); $\max|\mathsf A\mathbb 1_{\mathcal C}|/|\mathsf A|\le6\times10^{-17}$, kernel gap $\ge10^{-5}$ | `step01_null_space.jl` | also equals the code's `components.anchors` (smallest cell index of each unanchored component) and `operator.singular` in every case |
| 1b. solvable iff $\mathbb 1_{\mathcal C}^{\top}\mathsf b=0$ | same | ✅ verified | $\mathsf b$ made compatible ⇒ pseudo-inverse residual $\le10^{-9}$; one broken compatibility condition ⇒ residual equals the broken mean | same | symmetric ⇒ range $=\ker^{\perp}$; inactive rows give no condition (identity rows) |
| 1c. anchoring rules (open-face conductance; nonzero face to an inactive cell) | l. 111–116 | ✅ verified | Outflow top anchors exactly one of the two components, right anchors both; in $:$dirichlet mode the active–inactive faces with $\kappa_f\ne0$ anchor, in $:$decoupled mode with $\kappa_f\ne0$ also (the matrix treatment is the same) | same | matches `_refresh_elliptic_components!` (`src/solvers/multigrid/elliptic_solver.jl:39-103`) |
| 2a. symmetric positive semidefinite $\mathsf B$, mean-free range, positive definite there, for arbitrary $\mathsf P$ and coarse operators | `sec:solvers:vcycle` | ✅ proved (above) and verified | 120 random 3–4-level hierarchies (singular and anchored, arbitrary signed $\mathsf P$, coarse operators unrelated to $\mathsf P^{\top}\mathsf A\mathsf P$): asymmetry $\le1.4\times10^{-15}$, spectrum of $\mathsf B\mathsf A$ in $[1.3\times10^{-2},5.7\times10^3]$ (SPD but not a contraction) | `step02a_vcycle_algebra.jl` | coarse operator scaled by $10^{-3}$: SPD, $\lambda_{\max}=2.8\times10^4$ |
| 2b. the package's V-cycle, larger cases | l. 525–560 | ✅ verified | 19 dense cases ($\mathsf B$ from $n$ V-cycles, exact spectrum) up to 7552 cells: uniform 16²–64² (κ 1.27–1.37), periodic 32², composite base 16 with 2 and 3 levels (1.38, 1.39), drop at ratio $10^3$ and $10^6$ (1.52–1.83), split cut-cell 32² and 64² (1.43, 1.44), cut cells with open fraction $10^{-3}$ (1.69), composite + solid (1.47), split with Outflow top (**κ=2.01**, $\lambda_{\max}=1.45$) and Outflow right (2.03), Dirichlet-/decoupled-inactive blob without geometry (**κ=2.86, $\lambda_{\max}=2.205$**): asymmetry $\le6.2\times10^{-13}$ (ratio $10^6$; otherwise $\le10^{-15}$), range $\le2\times10^{-16}$, active–inactive coupling exactly 0, $\lambda_{\min}>0$ throughout. Large: random-vector symmetry $\le5\times10^{-17}$, Rayleigh quotients $>0.19$, Ritz values of $\mathsf B\mathsf A$ in $[0.46,1.03]$ on 128², 256² uniform, composite base 32/64 with 2–3 levels (19 648–49 600 cells), drop $10^3$/$10^6$ composite and 128², split 128², composite + solid. Matrix-free finest level equals the assembled one to $5\times10^{-16}$ | `step02b_vcycle_code.jl` | the chapter's $[0.673,1.035]$ is a property of its five fixtures; $\lambda_{\max}(\mathsf B\mathsf A)>1$ occurs and reaches 2.2 |
| 2c. hypotheses in the code: equal pre/post sweeps | `MultigridOptions`, l. 525–531 | ✅ verified (necessity) | $(\nu_1,\nu_2)=(1,1),(2,2),(3,3)$: asymmetry $4\times10^{-16}$, SPD (spectrum of $\mathsf B\mathsf A$ in $[0.55,1.06]$, $[0.76,1.02]$, $[0.83,1.00]$); $(1,2),(2,1)$: asymmetry $7.7\times10^{-3}$, $(0,2),(2,0)$: $3.1\times10^{-2}$ | `step02c_sweep_options.jl`; `step02a` (1,2): 0.29 | `MultigridOptions` accepts $\nu_1\ne\nu_2$; CG with such a hierarchy is outside the theory |
| 2d. $\omega\lambda<2$ is the operative condition | l. 535 | ✅ verified | $\omega g=2.6$, one sweep: $\mathsf B$ indefinite in 32 of 40 random hierarchies; $\omega g=1.95$: SPD in 40 of 40 | `step02a_vcycle_algebra.jl` | |
| 2e. citation "Tatebe §§4–5" | l. 536 | ✅ verified (read) | the NTRS scan (19940017009): Thm 2 (two-grid, pre+post, damped Jacobi, arbitrary SPD $L_{l-1}$, $\mathsf r=b\mathsf p^{\top}$), Thm 3 (V-/W-cycle by induction); hypotheses: $L=P-Q$, $P$ symmetric, symmetric part of $P+Q$ positive definite (⇔ $\omega\rho(\mathsf D^{-1}\mathsf A)<2$) | — | nonsingular levels only |
| 3a. $g_l=\max_i\sum_j|A_{ij}|/A_{ii}$, code weight $=\min\{1,4/(3g)\}$, $\operatorname{spec}(\mathsf D^{-1}\mathsf A)\subset[0,g]$, $\omega\lambda_{\max}\le\tfrac43$ | `eq:solvers:gershgorin`, `eq:solvers:jacobi` | ✅ verified | every level of 12 hierarchies (uniform, periodic, composite, drop $10^6$, split, composite+solid, Outflow, reaction 0.5/50/$5\times10^4$), dense spectra: $g\le2$ in all, $g=2$ except with a reaction term ($g=1.0051$–$1.9999$); $\omega\rho(\mathsf D^{-1}\mathsf A)$ between $1.0048$ and $4/3$ (equal to $4/3$ wherever $g=2$ and the $(\pi,\pi)$ mode is resolved) | `step03_smoother_bound.jl` | Gershgorin $\Rightarrow$ $|\lambda|\le g$ is Saad Thm 4.6 (checked in the book) |
| 3b. uniform five-point: $g=2$, $\omega=2/3$ | l. 445–446, 477–478 | ✅ verified | levels of walled and periodic 32²: $g=2.0000$, $\omega=0.6667$, $\rho(\mathsf D^{-1}\mathsf A)=2.0000$ (the $(\pi,\pi)$ mode) | same | $\omega\rho=\tfrac43$ attained |
| 4a. $\|\mathsf r\|_2\le\tau\min V\Rightarrow\max|r_c|/V_c\le\tau$ | `eq:solvers:gmres` | ✅ verified (the implication), ❌ differs from the code | 2000 random draws over 8 decades of volumes; code `_gmres_absolute_tolerance` $=\tau\cdot\min\{1,\min_{V>0}V\}$ (`init = one(T)`): equals the text when some $V<1$, is $\tau\cdot1<\tau\min V$ when all $V>1$ | `step04_gmres_and_floor.jl` | E2: stricter than the text, still sufficient |
| 4b. GMRES/FGMRES reach the strong target | l. 244–258 | ✅ verified | uniform 32² and split cut-cell, GMRES and FGMRES: final strong residual $7\times10^{-12}$–$2\times10^{-11}$ vs target $5\times10^{-10}$; one iteration more than CG | same | conservative by a factor 10–70 |
| 4c. factorised floor $\ge$ assembled floor, entrywise | `eq:solvers:floor`, l. 218–222 | ✅ verified | 7 operators (uniform, composite, drop $10^3$, split, composite+solid, Outflow, reaction), 6 draws each incl. $|x|$ over 6 decades: $(|\mathsf A||x|)_{\rm asm}\le(|\mathsf A_{\rm fact}||x|)_{\rm fact}$ on every cell; $\roundfloor_{\rm fact}/\roundfloor_{\rm asm}\in[1.000,1.271]$ (exactly 1 on diagonal metrics) | same | triangle inequality $|\mathsf J^{\top}\mathsf M\mathsf J|\le|\mathsf J|^{\top}|\mathsf M||\mathsf J|$, $\mathsf C\ge0$; the identity rows in the code's floor are irrelevant (excluded by $V>0$) |
| 4d. the 128 ε scale bounds the evaluation error of the factorised action | l. 214–217 | ✅ verified | exact BigFloat evaluation of the same factors on 3 composite operators: error $=0.42$–$0.45\ \varepsilon\,\max(|\mathsf A_{\rm fact}||x|/V)$ | same | margin ≈ 300 |
| 5a. $\hat{\mathsf A}\hat{\mathsf P}=\mathsf D_r(\mathsf A\mathsf P)\mathsf D_r^{-1}$, same spectrum | `eq:solvers:cmdgfm-scaling` | ✅ verified | 50 random $24\times24$ draws with scalings over 4 decades: spectra equal to $1.2\times10^{-14}$; code constants ($(\Delta t/L)^2,\Delta t^2/L$; $\rho_0(L/\Delta t)^2,L/\Delta t^2$) and `_ScaledProjectionPreconditioner` $=\mathsf D_c^{-1}\mathsf P\mathsf D_r^{-1}$ checked against the text | `step05_cmdgfm_scaling.jl` | |
| 5b. "changes only the norm BiCGSTAB measures" | l. 749–752 | ⚠️ incomplete | `Krylov.bicgstab!` on the scaled system equals, to $10^{-14}$ over 1, 3, 6 iterations, BiCGSTAB on the unscaled right-preconditioned system in the inner product $\mathsf u^{\top}\mathsf D_r^2\mathsf v$; it differs from the Euclidean BiCGSTAB by 0.2, 0.2, 0.02 | same | E3: iterates (shadow residual, residual-minimising step) change, the spectrum does not |
| 6a. $\mathsf P\mathbb 1^{\rm act}_{l+1}=\mathbb 1^{\rm act}_l$ | `eq:solvers:active-transfer` | ✅ verified | every level of the split cut-cell 64², composite 16 + solid, solid disc 32², and a blob without geometry ($:$dirichlet and $:$decoupled): $\max|\mathsf P\mathbb 1_c-\mathbb 1_f|\le2.2\times10^{-16}$; exact zeros in covered rows and columns | `step06_transfer_null_vector.jl` | |
| 6b. an unrestricted transfer violates it and feeds covered cells | l. 458–462 | ✅ verified | recomputed from the tentative transfer and the level stiffness: $\max|\mathsf P_{\rm raw}\mathbb 1_c-\mathbb 1_f|=0.333$, 52–80 covered cells receive a correction | same | |
| 6c. $\mathbb 1^{\rm act}$ is a null vector of the fine and coarse operators (single component, closed) | l. 459 | ✅ verified | $|\mathsf A\mathbb 1^{\rm act}|/|\mathsf A|\le1.2\times10^{-16}$ on all 3 levels of the solid disc | same | |
| 7a. CG energy bound $\|e_k\|_A\le2q^k\|e_0\|_A$ holds; $\kappa$ from the Ritz extremes | l. 553–557 | ✅ verified | exact solution by the bordered system; uniform 32², 64², drop $10^3$ 64², composite base 32, split 64²: envelope holds for all $k$ | `step07_cg_bound.jl` | |
| 7b. predicted vs measured iteration count and factors | l. 553–560 | ✅ verified; ⚠️ wording | bound count $\lceil\ln(2/10^{-10})/\ln(1/q)\rceil$ = measured count ($10,10,13,10$; split $10$ vs $11$); energy-error factor at the stop $q\cdot(0.92\ldots1.02)$; residual factors relative to the energy factor: B-norm equal, strong-L² $0\ldots+10\,\%$, strong-L∞ $+10\ldots+20\,\%$ | same | E4 |
| 7c. unpreconditioned CG: $\kappa(\mathsf A)=O(h^{-2})$, iterations double | l. 802–806 | ✅ verified | $\kappa(\mathsf A)/n^2=0.805,0.809,0.810$ at $n=16,32,48$ | same | |
| 8. one-phase viscous matrix on composite meshes | l. 652–662; `09_viscosity.tex:117-135` | ✅ nonsymmetric without a solid; ⚠️ not with a solid | $\dualvolmat(\idmat-s\lapstag{k})$: uniform walled/periodic exactly symmetric, CG; composite base 16: asymmetry $0.22$ of the matrix, $0.27$ of $\dualvolmat\lapstag{k}$ (the band rows; $\viscstiff{k}$ itself symmetric to $0$), BiCGSTAB; embedded constructor (#79/#87 rows, no band): asymmetry $0$ (uniform cut) and $2\times10^{-19}$ (composite cut), **BiCGSTAB anyway** | `step08_viscous_symmetry.jl` | E5 |

## Errors and consequences

None of the seven claims is false. Defects of statement, each with a replacement (LaTeX macros of the chapter):

**E1 (claim 2; paragraph "Symmetry and definiteness", l. 525–539).** The paragraph defers to a citation for a result that holds for nonsingular levels only and does not say that the *equal sweep count* is a hypothesis that `MultigridOptions` does not enforce. Replace the last two sentences ("For such a cycle … a priori") by:

```latex
Let $\precond_{\mgindex}$ be the cycle started on level $\mgindex$ and
$\mathcal W_{\mgindex}$ the volume-mean-free space of the unanchored
components of $\mgmesh{\mgindex}$, on which
$\langle\mathsf x,\mathsf y\rangle_{\mgindex}=\mathsf x^{\transpose}\mgmat{\mgindex}\mathsf y$
is an inner product. With
$\mathsf S=\compproj{\mgindex}(\idmat-\jacobiweight{\mgindex}\mathsf D^{-1}\mgmat{\mgindex})$
and $\nu=\nu_1=\nu_2$ the error of one cycle is
\begin{equation}
  \idmat-\precond_{\mgindex}\mgmat{\mgindex}=\mathsf S^{\nu}\,\mathsf T\,\mathsf S^{\nu},
  \qquad
  \mathsf T=\idmat-\compproj{\mgindex}\mgprolong{\mgindex}\,\precond_{\mgindex+1}\,{\mgprolong{\mgindex}}^{\transpose}\mgmat{\mgindex}.
\end{equation}
Since $\mgmat{\mgindex}\compproj{\mgindex}=\mgmat{\mgindex}$, both factors are
selfadjoint for $\langle\cdot,\cdot\rangle_{\mgindex}$, and
$\langle\mathsf T\mathsf x,\mathsf x\rangle_{\mgindex}=\norm{\mathsf x}_{\mgindex}^{2}-\mathsf r^{\transpose}\mgprolong{\mgindex}\precond_{\mgindex+1}{\mgprolong{\mgindex}}^{\transpose}\mathsf r\le\norm{\mathsf x}_{\mgindex}^{2}$
with $\mathsf r=\mgmat{\mgindex}\mathsf x$ whenever $\precond_{\mgindex+1}$ is
symmetric positive semidefinite, as the exact coarse solve~\cref{eq:solvers:augmented} is.
The smoother contracts, $\norm{\mathsf S\mathsf x}_{\mgindex}<\norm{\mathsf x}_{\mgindex}$ on
$\mathcal W_{\mgindex}\setminus\{0\}$, exactly when
$0<\jacobiweight{\mgindex}\lambda<2$ on the nonzero spectrum of
$\mathsf D^{-1}\mgmat{\mgindex}$, which~\cref{eq:solvers:jacobi} guarantees
($\jacobiweight{\mgindex}\lambda\le\tfrac43$). Hence
$\langle\precond_{\mgindex}\mgmat{\mgindex}\mathsf e,\mathsf e\rangle_{\mgindex}\ge\norm{\mathsf e}_{\mgindex}^{2}-\norm{\mathsf S^{\nu}\mathsf e}_{\mgindex}^{2}>0$,
and by induction from the coarsest level $\precond$ is symmetric, positive
definite on the mean-free space and zero on the $\mathsf v_{\compcells}$. The
argument uses no relation between $\mgprolong{\mgindex}$ and $\mgmat{\mgindex}$,
hence no Galerkin product; for nonsingular levels it is
Theorems~2 and~3 of~\cite{tatebe1993}. It needs $\nu_1=\nu_2$ (the code accepts
$\nu_1\neq\nu_2$, for which $\precond$ is nonsymmetric and CG is not justified)
and fails for $\jacobiweight{\mgindex}\lambda>2$. It does not make the cycle a
contraction: $\lambda_{\max}(\precond\sysmat)$ exceeds one on an operator with
an open side ($1.45$) and on a Dirichlet-inactive operator without geometry
($2.2$), so the cycle is a preconditioner, not a stand-alone solver, and its
convergence factor is not bounded a priori; it is measured
(\cref{sec:solvers:measured}).
```

and after "$\chkSolvKappaSplit$ (split)" in the checks paragraph add: "with larger values with an open side or a Dirichlet-inactive region (cases outside the table: $\kappa=2.0$ and $2.9$)" (optional). The claim "needs restricting?": **no**; the chapter's checks (32², one composite mesh) are now backed by a proof and by 12 dense cases up to 7552 cells and 11 sampled cases up to 65 536 cells.

**E2 (claim 4; `eq:solvers:gmres`, l. 252–255).** The code's tolerance is $\stoplevel\min\{1,\min_{V>0}V\}$ (`_gmres_absolute_tolerance`, `init = one(T)`). Replace by

```latex
      \norm{\resvec}_2\le\stoplevel\,\min\curlypar{1,\ \min_{c:\,\volmat_{cc}>0}\volmat_{cc}}
      \quad\Longrightarrow\quad\strongnorm{\resvec}\le\stoplevel,
```
and "because $|r_c|\le\norm{\resvec}_2$ for every $c$" by "because $|r_c|\le\norm{\resvec}_2$ for every $c$ (the cap at one only tightens the test when every volume exceeds one)".

**E3 (claim 5, l. 749–752).** Replace "is similar to $\sysmat\precond$: the scaling changes the norm in which BiCGSTAB measures its residual, not the preconditioned spectrum." by

```latex
is similar to $\sysmat\precond$, so the preconditioned spectrum is unchanged.
BiCGSTAB itself is not invariant under a nonuniform $\mathsf D_{\mathrm r}$: on
the scaled system it is BiCGSTAB on the unscaled right-preconditioned system
in the inner product $\mathsf u^{\transpose}\mathsf D_{\mathrm r}^{2}\mathsf v$
(shadow residual, residual-minimising step and stopping norm), so the scaling
weighs the pressure and the jump rows in these.
```

**E4 (claim 7, l. 553–557).** The bound is for the energy norm of the error; the measured factor is for the strong residual. Replace "The CG bound … for the energy norm of the error then predicts a reduction by $\chkSolvBoundUniform$ (uniform, $32\times32$) and $\chkSolvBoundDrop$ (drop) per iteration; preconditioned CG measures mean residual reductions $\chkSolvCgUniformFactor$ and $\chkSolvCgDropFactor$ in $\chkSolvCgUniform$ and $\chkSolvCgDrop$ iterations on the same systems." by

```latex
The CG bound $\norm{\mathsf e_{\itcount}}_{\sysmat}\le2\roundpar{(\sqrt\kappa-1)/(\sqrt\kappa+1)}^{\itcount}\norm{\mathsf e_0}_{\sysmat}$
for the energy norm of the error gives the factor $\chkSolvBoundUniform$
(uniform, $32\times32$) and $\chkSolvBoundDrop$ (drop) per iteration, and
$\lceil\ln(2/\reltol)/\ln(1/q)\rceil$ iterations, within one of the measured
$\chkSolvCgUniform$ and $\chkSolvCgDrop$. The energy norm of the error reduces
by this factor to within $8\,\%$ (never above $1.02$ times it); the mean reduction of the strong residual
\cref{eq:solvers:factor}, $\chkSolvCgUniformFactor$ and $\chkSolvCgDropFactor$,
is in another norm, for which the bound does not apply, and exceeds the factor
by up to $15\,\%$.
```
(the audit scripts give energy factors $0.072$ (32²), $0.078$ (64²), $0.137$ (drop 64²) against $q=0.073,0.079,0.149$, strong-L² residual factors $0.077,0.080,0.151$; the numbers $\chkSolvCgDrop=12$ vs $\lceil\cdot\rceil=11$ for the drop at 32² are within one.)

**E5 (claim 8, l. 652–659).** Replace the two sentences "The matrix is symmetric when … BiCGSTAB for all shifts. The corrected rows … BiCGSTAB (\issue{80})." by

```latex
The matrix $\dualvolmat(\idmat-s\lapstag{k})$ is symmetric for every $s$ iff
$\dualvolmat\lapstag{k}$ is. The code tests the sufficient condition that every
off-diagonal pair has equal volumes and equal Laplacian entries, entry by
entry for both components; then the default Krylov method is CG, otherwise
BiCGSTAB for all shifts. The corrected rows of~\cref{eq:viscosity:laplacian}
at level transitions are nonsymmetric ($\dualvolmat\lapstag{k}$ misses
symmetry by $27\,\%$ of its largest entry on the composite fixture), so on composite meshes without a solid
the one-phase viscous step uses BiCGSTAB (\issue{80}). The embedded operator
has no corrected band: $\dualvolmat\lapstag{k}$ is symmetric to round-off
($4\times10^{-17}$ on a composite mesh with a solid), but the unequal open volumes fail the
test, so it is solved by BiCGSTAB as well, although CG would be valid.
```
Consequence: Table `tab:solvers:systems` "SPD or nonsymmetric" is right; the measured composite-mesh case (no solid) is genuinely nonsymmetric. An embedded one-phase viscous solve could use CG with the symmetric tree multigrid (code observation, not an error of the chapter).

**Chapter vs code, further small points (no action required).** (i) The floor (`eq:solvers:floor`, factorised variant) omits the identity rows of the inactive cells, which the code includes in $|\mathsf A_{\rm fact}||x|$; they are excluded from the maximum by $V>0$. (ii) The range $[\chkSolvLambdaMin,\chkSolvLambdaMax]$ of $\lambda(\mathsf B\mathsf A)$ and $\kappa\le1.52$ are properties of the five fixtures; other admissible operators give $\lambda_{\max}$ up to 2.2 (E1).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| 1. $\ker\mathsf A$ = unanchored indicators; "$\ne0$" criterion | **sharp** (iff, proved and verified); the criterion is an exact-zero test | the kernel has the full dimension predicted in 127 random unanchored components; with a coefficient $\epsilon>0$ joining two halves the matrix is nonsingular but $\lambda_2=\epsilon/2$ (nearly decoupled components: $\lambda_2=4.9\times10^{-3},5.0\times10^{-7},5.0\times10^{-11}$ for $\epsilon=10^{-2},10^{-6},10^{-10}$) | `step01_null_space.jl` | nothing to improve in the statement; remark: a tiny $\kappa_f$ gives a nearly singular system whose compatibility is not enforced (cf. #97) |
| 2. SPD of the V-cycle without Galerkin | **sharp**: necessary hypotheses exhibited | $\nu_1\ne\nu_2$ ⇒ nonsymmetric (0.29; package: $\ge10^{-3}$); $\omega g=2.6$ ⇒ indefinite (32/40); $\omega g<2$ suffices (40/40 at 1.95) | `step02a`, `step02c` | unneeded: Galerkin, any relation of $\mathsf P$ to $\mathsf A$, nonnegative $\mathsf P$, SPD (SPSD suffices) coarse operators; $4/3$ may be replaced by any value $<2$ |
| 3. $\omega\lambda\le4/3$, $g=2$ | **sharp** for the bound on the uniform operator ($\omega\rho=\tfrac43$ attained); **not sharp** as the definiteness threshold | $\rho(\mathsf D^{-1}\mathsf A)=g=2$ at $(\pi,\pi)$; energy convergence holds for every $\omega<2/g=1$ on the uniform operator; smoothing factor of the five-point operator: $\mu(2/3)=0.667$, $\mu(4/5)=0.600$, $\mu(1)=1$ | `step03` | the weight $4/(3g)$ is a smoothed-aggregation convention (Vaněk et al.), not optimal for smoothing ($4/5$ would give 0.6); changing it is an efficiency question |
| 3. Gershgorin vs spectral radius | **likely sharp** on uniform; slack $\le0.3\,\%$ on composite levels | $\rho=1.998$, $1.9976$, $1.9979$ against $g=2$ on the composite fixtures | `step03` | none worthwhile |
| 4. `eq:solvers:gmres` | **not sharp** (sufficient, conservative) | final strong residual $10$–$70\times$ below the target; one iteration more than CG; the code's cap $\min\{1,\cdot\}$ makes it stricter when all volumes exceed 1 | `step04` | none needed with the multigrid preconditioner; a strong-norm callback is not available for GMRES until the end of an Arnoldi pass |
| 4. factorised floor $\ge$ assembled floor | **sharp** (equality on diagonal metrics); up to $1.27\times$ larger on composite meshes | ratios $[1.000,1.271]$ | `step04` | none: the factorised action does compute with the larger scale |
| 4. $128\varepsilon$ | **not sharp**, margin $\approx300$ for the factorised action | actual evaluation error $0.42$–$0.45\,\varepsilon\,|\mathsf A||x|/V$ | `step04` | 128 covers the recursively updated residual and the preconditioned recurrences as well; not examined |
| 5. similarity | **sharp** (exact identity) | $1.2\times10^{-14}$ | `step05` | — |
| 5. "changes only the norm" | **wrong as an invariance statement** (E3) | iterate differences 0.2/0.2/0.02 against the Euclidean BiCGSTAB; $<10^{-13}$ against the $\mathsf D_r^2$-weighted one | `step05` | replacement given |
| 6. $\mathsf P\mathbb 1_c=\mathbb 1_f$ | **sharp** (exact, $2\times10^{-16}$); the restriction is necessary | unrestricted transfer violates by $1/3$ and feeds 52–80 covered cells | `step06` | normalised entries can be slightly negative on composite meshes with a non-diagonal metric (min entry $-0.000$ to three digits); not harmful |
| 7. CG bound | **sharp for the energy norm** on these systems | $\|e_k\|_A$ reduction $=q\times(0.92\ldots1.02)$ for noise right-hand sides | `step07` | no bound for the strong residual factor; measured up to $15\,\%$ above $q$ (L²) and $20\,\%$ (L∞) |
| 8. composite-mesh viscous matrix nonsymmetric | **correct for meshes without a solid**; the code's symmetry criterion is sufficient, not necessary | embedded constructor: symmetric to $10^{-19}$, BiCGSTAB selected | `step08` | test symmetry of $\dualvolmat\lapstag{k}$ (i.e. $V_iL_{ij}=V_jL_{ji}$) instead of equal volumes to allow CG for embedded geometry |

## Open points

- Whether CG with the (symmetric) staggered multigrid is wanted for embedded one-phase viscous solves (E5 consequence): a code decision for the owners of `src/navier_stokes/implicit_diffusion.jl`, not a documentation matter.
- `MultigridOptions` accepts $\nu_1\ne\nu_2$ with CG: document or reject (E1).
- The new paragraph on CMDGFM residual-defect reduction factors (commit `5c31fca`, after l. 909) is not audited here.
- Not examined: Float32 behaviour of the V-cycle symmetry; the two-phase staggered (variable stress) multigrid, whose symmetry is not asserted by the chapter beyond the unsmoothed transfer.
- Literature (for `literature-scout`, optional): a published statement of the $\mathsf T\le\mathsf I$ argument for non-Galerkin coarse correction with singular (projected) levels.

## History

- 2026-11-10: first audit at `df7eb6c` (branch moved to `5c31fca`: CMDGFM reduction-factor edit, outside the claims). Verified the seven claims and the viscous side question; found E1–E5; created 10 scripts and `run_all.jl`. Final run of `scripts/run_all.jl` (10 child processes, one at a time, wall time 429 s): **all checks passed** (471 + 9 + 361 + 3 + 42 + 111 + 10 + 138 + 2011 + 84 + 3 + 4 + 63 + 14 931 + 15 + 14 assertions in 16 testsets, exit 0). Runic 1.10.0 applied to the scripts.
