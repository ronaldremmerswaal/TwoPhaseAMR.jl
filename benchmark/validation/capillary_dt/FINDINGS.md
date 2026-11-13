# #178: the two-velocity capillary-wave error grows under dt refinement

Reproduce (fresh process, ≈ 2 min on one thread; writes `results.txt`, every number below is taken from it):

```
OPENBLAS_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 julia --heap-size-hint=3G -t 1 \
    --project=benchmark/validation/capillary_dt benchmark/validation/capillary_dt/driver.jl
```

The case is the test's `wave_case(:uniform, n, DR, :capillary; formulation = CGFM())`
(test/integration/two_phase_waves.jl:86-223, copied verbatim to `wave_case.jl`). dt factor = multiple of the
initial stable step; factor 1 = the automatic step, as in the test. The two-velocity runs use `budget.jl`, a copy of
`step!(::TwoPhaseVelocity, …)` (src/two_phase/two_velocity/stepping.jl:277-364) that takes a snapshot after every
stage. It reproduces the test errors to all printed digits (0.0065537 / 0.0078230 / 0.0113736). The variants are
in-script only. `variants.jl` is a hook after the projection. `inference_override.jl` is an in-session copy of the
CMDGFM jump-row assembly (cmdgfm_projection.jl:721-853) with a mode switch. Its `:baseline` mode gives the
src error to 16 digits. **src/ is unchanged.**

Labels: **[M]** measured here, **[I]** inferred.

## Verdict

1. **The baseline step has no per-step, dt-independent defect [M].** Every stage changes the phase velocities by
   O(dt) per step (table 1: all /dt rows are constant). The projection's change of the mixture velocity is
   O(dt²). The error does not grow without bound. It *saturates*: the signed amplitude error at T/4 converges at
   first order in dt (successive differences −0.0111, −0.0056, −0.0028, −0.0014) to a dt → 0 limit of ≈ −0.029. The
   test metric goes 0.0066 → 0.0078 → 0.0114 → 0.0133 → 0.0144 towards ≈ 0.015.
2. **The increase comes from a temporal error cancelling against a spatial (semi-discrete) error of opposite
   sign [M].** The temporal error is the same in both velocity models: the signed error at factor 1 minus its
   dt → 0 limit is ≈ +0.022 in both. At DR 1 the spatial error is negative. In the one-velocity model it is
   ≈ −0.013, so |E| is smallest between factors ½ and ¼, and that model **also increases** under further
   refinement: 0.00302 → 0.00418 → 0.00501 at factors ¼ → ⅛ → 1/16. The test stops at ¼ and does not see this.
   The two-velocity spatial error is ≈ 2.2× larger (≈ −0.029), so its minimum lies at a factor ≥ 1 and every
   refinement increases the error.
3. **The entire excess spatial error of the two-velocity model is the CMDGFM inference term of #171 [M].**
   Replacing that term by a same-location comparison (V3 of the #171 study) makes the two-velocity model
   reproduce the one-velocity model to ≤ 1e-5 (≈ 1e-7 at n = 16, DR 1) at DR 1 and DR 1000, n = 16 and n = 32 (table 2). The test's factors
   1, ½, ¼ then give 0.01257 / 0.00455 / 0.00302, which is non-increasing. Removing the jump on mixed faces
   *without* a jump row (#175 class) changes nothing (≤ 1e-5).
4. **DR 1000 converges because there the spatial error has the same sign as the temporal error [M].** In the
   one-velocity model (= V3) the dt → 0 limit is ≈ +0.009. The inference term shifts the limit by ≈ −0.004,
   against −0.016 at DR 1. At DR 1000 that shift reduces the error.

So the dt dependence reported in #178 has no time-stepping cause. A spatial consistency error of the CMDGFM
jump row (#171) shows up in the fixed-mesh dt study. Under the rule "refining dt must never increase the error" the
one-velocity model also fails on this mesh (factors ⅛, 1/16 at n = 16 and n = 32). A consistent first-order
time integrator cannot satisfy that rule when the spatial and temporal errors have opposite signs.

## 1. Per-step budget, two-velocity, DR 1, n = 16, baseline [M]

Mean over all steps of the max over faces of the change each stage makes (`results.txt` section A).
`d[u]x`, `d[u]y` = change of u_g − u_l on x-/y-faces that carry both phases; for this near-horizontal interface x is
tangential and y is normal. `dU` = change of the mass-weighted mixture velocity. The stages are the `step!` body:
transport (WY interface + phase momentum, boundary enforcement), diffusion (common increment of the mixture
solve), forces (CGFM surface tension, common acceleration), old_gradient (−dt·G p_old, common), and projection
(CMDGFM pressure increment + jump ξ).

| stage | quantity | f = 1 | ½ | ¼ | ⅛ | 1/16 |
|---|---|---|---|---|---|---|
| transport | d[u]x | 3.39e-3 | 1.65e-3 | 7.76e-4 | 3.62e-4 | 1.69e-4 |
| | d[u]x / dt | 0.545 | 0.531 | 0.498 | 0.465 | 0.433 |
| | d[u]y | 1.26e-3 | 6.54e-4 | 2.53e-4 | 1.13e-4 | 5.57e-5 |
| | dU / dt | 0.0121 | 0.0122 | 0.0124 | 0.0124 | 0.0125 |
| diffusion | d[u]x, d[u]y | ≤ 3e-18 | ≤ 3e-18 | ≤ 2e-18 | ≤ 2e-18 | ≤ 2e-18 |
| | dU / dt | 0.361 | 0.366 | 0.373 | 0.374 | 0.376 |
| forces | d[u]x, d[u]y | ≤ 8e-18 | ≤ 7e-18 | ≤ 5e-18 | ≤ 4e-18 | ≤ 2e-18 |
| | dU / dt | 4.04 | 4.04 | 4.09 | 4.09 | 4.10 |
| old_gradient | d[u]x, d[u]y | ≤ 6e-18 | ≤ 7e-18 | ≤ 7e-18 | ≤ 7e-18 | ≤ 7e-18 |
| | dU / dt | 3.22 | 3.24 | 3.29 | 3.29 | 3.30 |
| projection | d[u]x | 4.69e-3 | 2.33e-3 | 1.17e-3 | 5.85e-4 | 2.93e-4 |
| | d[u]x / dt | 0.752 | 0.746 | 0.752 | 0.751 | 0.752 |
| | d[u]y | 3.89e-4 | 1.92e-4 | 9.67e-5 | 4.82e-5 | 2.41e-5 |
| | dU / dt | 0.391 | 0.197 | 0.100 | 0.050 | 0.025 |
| | steps | 23 | 46 | 91 | 182 | 363 |
| end of run | max \|u_g − u_l\| x-faces | 0.0994 | 0.0987 | 0.0984 | 0.0983 | 0.0982 |
| | max \|u_g − u_l\| y-faces | 0.0048 | 0.0071 | 0.0175 | 0.0207 | 0.0210 |

Contrast, same table with the jump removed after every projection (section D, `equalize all faces`):
projection d[u]x per step = 0.076 / 0.076 / 0.079 / 0.086 and hook dU per step = 1.2e-3 / 1.2e-3 / 1.1e-3 / 1.0e-3 for
f = 1 / ½ / ¼ / ⅛. Both are **independent of dt**, so d[u]x/dt grows as 1/dt (12 → 111). Only in this variant does a
per-step O(1) quantity exist. The error then grows with the number of steps: 0.0088 / 0.0121 / 0.0288 / 0.0591.

The DR 1000 budget (section E) has the same structure as the baseline: /dt rows bounded, projection dU = O(dt²),
end x-jump 0.242–0.243 at every dt. The projection d[u]x/dt steps from 1.66 to 3.13 between ½ and ¼. It is constant
within each pair, so this is a max-norm event on one face, not a 1/dt trend.

## 2. Error series [M]

Test metric (L2 in time of the normalized amplitude error); in brackets the signed error at T/4.

| case, n = 16 | f = 1 | ½ | ¼ | ⅛ | 1/16 |
|---|---|---|---|---|---|
| DR 1 two-velocity, baseline | 0.00655 (−0.0065) | 0.00782 (−0.0176) | 0.01137 (−0.0232) | 0.01334 (−0.0260) | 0.01435 (−0.0274) |
| DR 1 two-velocity, V3 inference | 0.01257 (+0.0092) | 0.00455 (−0.0018) | 0.00302 (−0.0073) | 0.00418 (−0.0101) | 0.00501 (−0.0115) |
| DR 1 one-velocity | 0.01257 (+0.0092) | 0.00455 (−0.0018) | 0.00302 (−0.0073) | 0.00418 (−0.0101) | 0.00501 (−0.0115) |
| DR 1 two-velocity, jump removed after every projection | 0.00878 | 0.01214 | 0.02875 | 0.05906 | |
| DR 1 … removed on inference-row faces only | 0.01090 | 0.00435 | 0.00900 | 0.01751 | |
| DR 1 … removed on faces without a jump row only | 0.00655 | 0.00782 | 0.01137 | 0.01334 | |
| DR 1000 two-velocity, baseline | 0.02598 (+0.0327) | 0.01546 (+0.0186) | 0.01021 (+0.0116) | 0.00759 (+0.0080) | |
| DR 1000 two-velocity, V3 inference | 0.02883 (+0.0370) | 0.01837 (+0.0230) | 0.01315 (+0.0159) | 0.01054 (+0.0124) | |
| DR 1000 one-velocity | 0.02883 (+0.0370) | 0.01837 (+0.0230) | 0.01315 (+0.0159) | 0.01054 (+0.0124) | |

| case, n = 32, DR 1 | f = 1 | ½ | ¼ | ⅛ | 1/16 |
|---|---|---|---|---|---|
| two-velocity, baseline | 0.00375 (−0.0072) | 0.00627 (−0.0108) | 0.00763 (−0.0127) | 0.00833 (−0.0136) | 0.00867 (−0.0140) |
| two-velocity, V3 inference | 0.00427 (+0.0042) | 0.00146 (+0.0005) | 0.00062 (−0.0013) | 0.00104 (−0.0023) | 0.00134 (−0.0027) |
| one-velocity | 0.00427 (+0.0042) | 0.00146 (+0.0005) | 0.00062 (−0.0013) | 0.00104 (−0.0023) | 0.00135 (−0.0027) |

dt → 0 limits of the signed T/4 error (last value + last successive difference) [M, Richardson-type estimate]:

| | DR 1, n = 16 | DR 1, n = 32 | DR 1000, n = 16 |
|---|---|---|---|
| one-velocity (= V3) | −0.0129 | −0.0032 | +0.0088 |
| two-velocity baseline | −0.0288 | −0.0145 | +0.0045 |
| shift caused by the inference term | −0.0159 | −0.0113 | −0.0043 |

The one-velocity spatial error converges at second order in h (−0.0129 → −0.0032). The two-velocity baseline
converges at only about first order (−0.0288 → −0.0145) because of the inference shift. [I] The test's two-velocity
*spatial* rate at fixed CFL (≥ 1.8) is therefore helped by the same temporal/spatial cancellation.

The temporal self-convergence at fixed mesh is non-increasing in every row (`|A_dt − A_finest|` = L2 difference to
the finest dt of the same row). Examples: two-velocity baseline n = 16: 0.0166 / 0.0077 / 0.0033 / 0.0011; one-velocity:
0.0163 / 0.0076 / 0.0033 / 0.0011.

## 3. Mechanism (file:line)

1. **Jump row with inference** [M; structure as in #171]. On an x-face cut by the near-horizontal interface,
   |n_y/n_x| ≈ 113 (cmdgfm_projection.jl:610-611). The row takes the inference branch (:656-708) with
   β = 0.0088 ≈ 1/|r|. Its donors are the parallel faces at +h (gas) and −h (liquid) (section C2: donor offsets
   (+1, −1) h on every listed face). The row (:721-739, assembled at :797-830) imposes
   [u]_f = (1 − β)(u_g(y + h) − u_l(y − h)) + transverse terms. It reads the shear of a continuous velocity over 2h
   as a phase jump. The viscous layer is δ = √(2ν/ω₀) ≈ 0.057 ≈ 0.9 h [I, arithmetic]. The tangential velocity
   changes sign across it, so the imposed jump (0.098; u_l = −0.054 and u_g = +0.045 on face 605) is of the size
   of the wave velocity aω₀ ≈ 0.11.
2. **The jump is a slaved constraint and does not accumulate** [M]. Every projection solves the row to tolerance.
   Between projections only transport changes [u], by O(dt) per step (table 1). Diffusion, forces and the old
   gradient are common to both phases (two_velocity/stepping.jl:191-202, :121-154). The following projection therefore corrects
   only O(dt). If the jump is removed after every projection, the projection re-creates all of it every step: this
   is the #171 property. The rhs is row/dt (:828-829) and Δ(u_g − u_l) = −dt ξ (common.jl:298-302, ζ at
   :188-189), so the created jump is dt-independent. The error then grows ∝ number of steps (section D). The
   baseline does not show that signature.
3. **How the jump reaches the amplitude** [I]. ξ is distributed by the *other* phase's mass
   (common.jl:298-302). At DR 1 the minority phase on each cut face takes the full jump: the liquid sliver above
   the interface (c_face = 0.0023, u_l = −0.054 while u_g = +0.045) and symmetrically the gas sliver below. The WY
   transport moves each phase subface with its own velocity (two_velocity/interface_transport.jl:294-339), and the
   phase momentum transport uses these values as donors. The VOF field, and with it the cosine moment, is
   therefore advected with a spurious tangential shear at the interface. The path was not decomposed further.
   V3 proves causality for the whole excess, not for the individual paths.
4. **DR 1 vs DR 1000** [M for the numbers, I for the reason]. At DR 1000 the jump goes almost entirely into the
   gas velocity, because m_l/m ≈ 1 except on liquid-sliver faces. The end x-jump of 0.24 is a gas value, and the
   heavy liquid barely moves. The inference shift of the dt → 0 error is therefore small (−0.004 vs −0.016). The
   sign decides the trend: the DR 1000 spatial error (+0.009 one-velocity, +0.0045 baseline) has the same sign as
   the temporal error. |E| then decreases monotonically to the spatial error, and the negative inference shift
   even lowers it.
5. **Faces without a jump row (#175 class) and minority values** [M]. The growing y-face jump in table 1
   (0.005 → 0.021) sits only on mixed faces without a row: unresolved stencils are dropped
   (cmdgfm_projection.jl:914-921), their minority aperture is closed (a = 0) and c = 2e-5 or 1 − 2e-5 (section C2).
   Equalizing them leaves the error unchanged to 1e-5 (section D). The residual V3 jumps (0.08 at f ≥ ¼, 0 at f ≤ ⅛)
   sit on the same kind of face (c = 7e-5, a_min = 0) and do not affect the error (V3 = one-velocity to ≤ 1e-5).
6. **One-velocity contrast** [M]. There is no phase jump. The temporal error is identical (+0.022 at f = 1) and
   the spatial error is −0.013 at n = 16, so the minimum of |E| lies between f = ½ and ¼. The model increases at
   f = ⅛ and 1/16, at n = 16 and at n = 32 (0.00062 → 0.00104 → 0.00135). Over the test's three factors it happens
   to be non-increasing.

## 4. In-script confirmation [M]

| DR 1, n = 16 | f = 1 | ½ | ¼ | trend over the test's factors |
|---|---|---|---|---|
| baseline | 0.00655 | 0.00782 | 0.01137 | increasing (×1.74) |
| V3 same-location inference (`inference_override.jl`, mode `:same_location`) | 0.01257 | 0.00455 | 0.00302 | non-increasing |

V3 (from #171's `cmdgfm_jump/FINDINGS.md`) replaces (β − 1)(u_g(h_g) − u_l(h_l)) by
(β − 1)(γ_lg [u]_{h_g} + γ_gl [u]_{h_l}). It keeps the ξ columns (β − 1)γ and drops the
(β − 1)(G p(h_g) − G p(h_l)) pressure columns. This term vanishes for a continuous field. It is a diagnostic
substitution, not a validated fix. [I] It down-weights genuine donor jumps by γ ≤ ½, and the Fortran reference cases
were not run. Extended to f = ⅛ and 1/16, V3 increases exactly like the one-velocity model (0.00418, 0.00501). That
is the opposite-sign cancellation of verdict 2, which no change to the jump row can remove.

The linear-inference option of #171 was not run here (other checkout); its numbers from the issue are
0.00615 / 0.00933 / 0.01305. [I] It reduces but does not remove the consistency error: with δ ≈ h the velocity is
far from linear over the 2h donor span, so the imposed jump stays O(1).

## 5. What a fix would have to change (description only)

1. **Jump row (cmdgfm_projection.jl:656-708, :721-739, :797-830).** The inference contribution must vanish for a
   velocity field that is continuous across the interface. Compare gas and liquid at the *same* location: either the
   jumps at the donor faces (V3), or each phase extrapolated to the face with its own normal derivative. Do not
   compare u_g(y + h) with u_l(y − h). Regression check: at DR 1 with equal kinematic viscosities the two-velocity
   wave must match the one-velocity wave (with V3 they agree to ≤ 1e-5 at every dt and both meshes).
2. **Test gate (test/integration/two_phase_waves.jl:268-333).** "Non-increasing under dt refinement at fixed mesh"
   does not hold for a correct discretization: the one-velocity model violates it at f = ⅛ and 1/16 on n = 16 and
   n = 32, and the test only avoids that by stopping at ¼. To test time-stepping correctness for both models, measure
   the temporal self-convergence on the fixed mesh. Examples: the L2 amplitude difference to the finest-dt run
   (non-increasing and ≈ halving in every case here), or the halving of successive signed differences. Separately,
   compare the dt-converged two-velocity error with the one-velocity error at DR 1 to catch the inference
   defect.
3. Not needed for #178: the treatment of faces without a jump row (#175) and the minority-phase
   marking/extension. Equalizing the rowless faces changes the error by ≤ 1e-5, and the transport stage changes
   the phase velocities by O(dt) per step.
