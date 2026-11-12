# Spurious tangential phase-velocity jump in the CMDGFM projection

Reproduce: `julia --project=benchmark/validation/viscous_kh benchmark/validation/viscous_kh/cmdgfm_jump/projection_only.jl`
(≈ 3 min, one thread) → `results.txt` (tables A–D) and `faces.txt` (every resolved x-face row).

Setup as `jump_reproducer.jl`: n×n uniform mesh, periodic in x, free-slip walls, equal densities and
viscosities, σ = 0; both phases carry u = 0.5 tanh(y/δ), v = 0, δ = 0.05 (u_g − u_l = 0 on every face).
Only the two-velocity CMDGFM projection is called (`_project_two_velocity!`,
src/two_phase/two_velocity/flow_workspace.jl:97). Variants V1–V5 re-solve the assembled system with
sparse LU after changing stencils or the right-hand side in the script; V0 is that direct solve with no change and
agrees with the library projection in every case (Table D vs Table A). src/ is unchanged.

## Verdict: H1b (inference term of the jump row) is the sole cause (measured)

On an x-face the stencil ratio is n_y/n_x (cmdgfm_projection.jl:610–611). For a near-horizontal interface
|ratio| ≫ 1 (16 at slope 0.063, 163 near the crests of the cosine), so the row takes the inference branch
(:656–708) with β = (1 − γ_lg − γ_gl)/(|r| − γ_lg − γ_gl) ≈ 1/|r|. **Every** resolved x-face row in all
tilted cases took this branch (`x-rows` = `infer` in Table A). The two inference faces are the parallel
x-faces exactly **+1 h and −1 h** away (gas face above, liquid face below; `faces.txt`, column
`Δy/h` = +1.0/−1.0 on every row). The row (:721–739, assembled at :797–830) contains
(β − 1)(u_g[gas face] − u_l[liquid face]). For a continuous shear field this term is
(β − 1)(U(y+h) − U(y−h)) ≈ −2U′h. That is the only nonzero part of every row: the face and transverse
parts are exactly 0 (Table A). The projection then enforces row = 0 by setting
[u]_f ≈ (1 − β)(U(y+h) − U(y−h)) ≈ 2U′h.

| case | δ/h | jump created | max row residual before (all from inference) | max dt·\|ξ\| | pressure part of Δ(u_g−u_l) | V1 inference rhs removed | V2 β=1, no inference | V3 same-face inference |
|---|---|---|---|---|---|---|---|---|
| flat y = 0 (control) | 1.6 / 3.2 / 6.4 | 0 / 0 / 0 | 0 (no x-face row) | 0 | 0 | 0 | 0 | 0 |
| flat y = 0.37h (control) | 1.6 / 3.2 / 6.4 | 0 / 0 / 0 | 0 (no x-face row) | 0 | 0 | 0 | 0 | 0 |
| cos a = 0.01 | 1.6 | 0.533 | 0.512 | 0.533 | 1.1e-16 | 0 | 0 | 0 |
|  | 3.2 | 0.304 | 0.294 | 0.304 | 1.1e-16 | 0 | 0 | 0 |
|  | 6.4 | 0.158 | 0.148 | 0.158 | 5.6e-17 | 0 | 0 | 0 |
| triangle s = 0.06 | 1.6 / 3.2 / 6.4 | 0.519 / 0.296 / 0.155 | 0.495 / 0.282 / 0.145 | = jump | ≤ 1.1e-16 | 0 | 0 | 0 |
| triangle s = 0.3 | 1.6 / 3.2 / 6.4 | 0.503 / 0.289 / 0.152 | 0.367 / 0.209 / 0.109 | = jump | ≤ 1.1e-16 | ≤ 3e-17 | 0 | 0 |
| cos, real step-1 input (state (f) of jump_trace.jl) | 1.6 / 3.2 / 6.4 | 0.528 / 0.302 / 0.158 | 0.512 / 0.294 / 0.147 | = jump | ≤ 2.2e-16 | ≤ 4e-14 | ≤ 4e-14 | 0 |

"Jump created" = max |u_g − u_l| over x-faces carrying both phases (triangle: faces > 0.1 from the kinks).
The last row reproduces the jump_reproducer/jump_trace numbers (0.528/0.302/0.158), so the
projection-only experiment captures the whole effect. The jump is ≈ 2|U′|h (Table B: 2|U′|h = 0.568/0.305/0.155
at the max face), i.e. first order, independent of slope from 0.063 down to the crests (β = 0.003 at
|r| = 325, still 0.147 at δ/h = 6.4) and from 0.06 to 0.3. It is not created on the flat controls. Being ≈ 1 − β times
the inference residual plus coupling through neighbouring ξ, the created jump is slightly larger than
|row residual| (s = 0.3: 0.152 vs 0.109).

## Mechanism (measured unless marked)

1. For a tilted interface every mixed x-face has |n_y/n_x| > 1 and a resolved row with inference faces
   at ±h (cmdgfm_projection.jl:656–708; flat interface: n_x = 0 → unresolved at :603, no x-row at all).
2. For the jump-free input the row residual is exactly (β−1)(u_g(y+h) − u_l(y−h)) = (β−1)·ΔU ≈ −2U′h
   (face part 0, transverse part 0, Table A). The inference term thus reads the shear of the continuous
   field over 2h as a phase jump. It compares the gas value and the liquid value at two different points.
3. The jump rhs is this residual / dt (:828–829); the solve returns ξ ∝ 1/dt, and the correction
   (projection/common.jl:299–302, applied at :310–322) changes u_g − u_l by exactly −dt(ζ_g − ζ_l)/m·ξ = −dt ξ
   (ζ_l = −m_g, ζ_g = m_l, common.jl:188–189). The common pressure gradient cancels: the pressure part of
   Δ(u_g − u_l) is ≤ 2e-16 even though dt|∇p| is 3e-3…3e-2 (Table C). Hence dt-independence.
4. ξ is distributed by the *other* phase's mass (u_l −= −dt m_g/m ξ, u_g −= dt m_l/m ξ), so the minority
   phase takes almost all of the jump. At the max face (c_face = 0.016, half a cell above the interface) this is the liquid,
   as observed in jump_trace.
5. The row is the thesis' n·J^β[u] = 0 with J^β_n = [u]_f/β + (1 − 1/β)(u_g(h_g) − u_l(h_l))
   (docs/techref/chapters/10_projection.tex:407–458). For |r| ≫ 1 it is dominated by the inference term,
   so its O(h·∂u/∂n) consistency error is imposed as a tangential jump of size (1 − β)·2U′h on the face.

## Other hypotheses (refuted, measured)

- **H1 (first-order c(1−c) transverse interpolant):** the transverse part of every row is 0 for this input
  (v = 0 in both phases, Table A). Removing it from the rhs (V4) leaves the jump unchanged to all printed digits. The
  weights are therefore irrelevant here.
- **H2 (correction driven by cut-cell divergence):** the mixture divergence is ≤ 1.4e-14 (a_l + a_g = 1 exactly).
  The per-phase aperture-weighted divergence in cut cells is O(1) (0.3…19.5, Table C), but it is not a constraint of
  the method. Zeroing the pressure-row rhs (V5) leaves the jump unchanged.
- **H3 (inconsistent apertures / face locations between divergence and gradient):** |a_l + a_g − 1| = 0
  on every face, and the pressure part of the jump is roundoff. No evidence. The common pressure correction exists only
  because ξ couples back into the divergence rows (coefficient = (c_face − a_l)·area for equal densities, common.jl:190–192 and cmdgfm_cache.jl:374) and through the
  (β−1)(Gp(h_g) − Gp(h_l)) columns. It moves both phases equally.

## What a fix would have to change (description only)

The inferred jump must vanish for a continuous field. That means comparing the gas and liquid values at the
same location: either extrapolate both phases to the face or interface position (using the normal derivative of each
phase), or subtract the mixture-velocity difference between the two donor faces. Variant **V3** does the latter
in-script: it uses (u_g − ū)(h_g) − (u_l − ū)(h_l) = γ_lg[u]_{h_g} + γ_gl[u]_{h_l}. This keeps the
existing ξ columns (β−1)γ, drops the (β−1)(Gp(h_g) − Gp(h_l)) pressure columns, and leaves the diagonal-dominance
bound (10_projection.tex:502) intact. It removes the jump completely (Table D). Simply disabling inference
(V2: β = 1, no inference) also gives zero jump here. But it gives up the dominance bound, because off-diagonals
reach |r| ≈ 16…325. That variant is not recommended.

## Open / not done

- [INFERENCE] V1–V3 give zero jump for a jump-free input by construction (linearity), so they confirm the source but do
  not validate a fix. Not tested: whether V3 still removes a *genuine* normal-velocity jump, or whether it reproduces the
  Fortran CMDGFM reference cases (the purpose of the inference branch). V3 down-weights the donor jumps by γ ≤ 1/2.
- No AMR/level-transition, solid or σ > 0 cases were run; the per-face log covers x-faces only. y-face rows are
  axis-aligned (flat) or have |r| = |n_x/n_y| ≤ 0.3 (tilted; β = 1 and no inference by :656, not logged).
