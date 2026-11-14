# Momentum transport in the Gram metric 𝓜 (MimeticProjection) — parked study

Worktree `/home/ronald/Documents/wt-mm-momentum`, branch `study/mm-momentum-transport` (from `feat/mimetic-mm-projection` @d3544498).
Note: `/home/ronald/Documents/mimetic-projection-audit/notes/mimetic_momentum_transport.{tex,pdf}` (propositions, proofs, tables).
Prototype (non-default): `src/two_phase/projection/mimetic_transport.jl` (`_MIMETIC_TRANSPORT[]`, default `:none`), mass matching in
`mimetic_metric.jl` (`_MIMETIC_MASS_MATCH[]`, default `false`), hook in `two_velocity/stepping.jl::_transport_substep!`.
Drivers here: `common.jl`, `freestream.jl`, `smallcell.jl`, `drop.jl`, `geometry.jl`, `smoke.jl` (+ unrun `cases.jl`, `queue.sh`).
Study parked by the owner before the KH/RT/wave/combined/cost runs.

## Designs (all keep the W sweeps: VOF volume fluxes, LW, CTU switch, drain limiter, floor; one 𝓜⁺ band solve per step)

| design | update after the W sweeps | constants | 𝓜-momentum | semi-discrete energy | small cells |
|---|---|---|---|---|---|
| D0 (current) | u⁺ = u_W | exact | drifts (obstruction + Zᵀ(E⁺ū−Eu)) | no | as now |
| DF flux form | 𝓜⁺u⁺ = 𝓜u − Δt C_W(u) | **fails** (Prop. flux) | exact | no | **unbounded** |
| DS skew form (θ=½) | 𝓜⁺u⁺ = 𝓜⁺ū + W⁺(u_W−ū) + ½(Eu − E⁺ū) | fails | no | exact | **unbounded** |
| DG redistribution | DS + skew band Y (LSQR) + remainder | exact in arithmetic | exact if Gram mass = VOF mass | exact | **unbounded** |
| DI increment form | 𝓜⁺(u⁺−ū) = W⁺(u_W−ū) | **exact, every sweep** | Zᵀ(E⁺−E)u per step (telescopes) | no | **bounded** (λmax(W,𝓜) ≤ 2.7) |
| DP | DI + mass-proportional momentum restoration | exact (round-off) | exact with mass matching | no | bounded map; see open items |

E = 𝓜 − W (band only), Z the per-phase constant fields, ū the extended start values.

## Main results

1. **Obstruction (Prop. obstruction).** Exact 𝓜-momentum and exact free stream together need Zᵀ𝓜Z (the Gram mass Σ_c G_c) to be invariant.
   For the assembled metric it differs from the VOF mass by O(h²) and moves with the interface (drop R=0.2, sliding one cell):

   | n | (ΣG − V)/V | range over slide | mass-matched | max slide |
   |---|---|---|---|---|
   | 16 | −3.5e-3 | 6.4e-3 | 6.6e-16 | 3.1e-2 h |
   | 32 | −4.2e-4 | 6.9e-4 | 8.8e-16 | 7.7e-3 h |
   | 64 | −1.1e-4 | 8.3e-5 | 6.6e-16 | 2.5e-3 h |

   **Mass matching**: least-norm slides of the cut-cell pressure points along their symmetry lines (one global 3×3 solve per assembly)
   make Σ_c G^π_c = ρ^π V^π I exactly; the static drop changes ≤ 3 %.
2. **A flux form with the VOF fluxes cannot preserve constants** (Prop. flux): per value 𝓜Z and WZ differ by O(1) (slaved values, cross entries).
3. **θ-family identities** (Prop. theta) verified to 2e-17 in one-step checks: constants exact iff θ=1, momentum iff θ=0, energy iff θ=½.
4. **Free stream** (U in both phases, drop/band, ρ_g ∈ {1, 1e-3}, CFL 0.1/0.45 of the full-cell WY bound, fractions down to 5e-10):
   D0 and DI bitwise exact after every sweep and step; DP+mm 1.4e-16–2.6e-16; DG 5e-5…310 (round-off over slaved-value inertia);
   DS/DF 4.6e3…1.6e7. 𝓜-momentum change of the constant field over 30 steps: D0/DI 2.1e-4, DP+mm 3e-12 (drop).
5. **Small cut volumes** (drop + vortex crossing the box, floors √ε and 1e-12, CFL 0.1/0.25/0.5 of the full-cell bound):
   DI and DP run at every CFL up to the WY maximum 0.5 — no small-cell restriction; DG and DF blow up in 1–126 steps at every CFL.
   ρ_g = 1: wet maxima ≤ 1.03 u0 for D0/DI/DP; DI makes new extrema on small-aperture values (≤ 2.4 u0; 𝓜⁻¹W not monotone, min entry −0.8).
   ρ_g = 1e-3: wet maxima D0 3.7–6.3, DI 1.7–3.4, DP 1.7–5.3; gas values on faces with aperture < 0.05 reach 1e2–1e4 u0 for **all** designs
   (D0 included); DI/DP need 1.6–3.4× more steps than D0 at the same CFL fraction (the bound sees larger gas velocities). 𝓜-energy max/initial:
   D0 ≤ 1.008, DI ≤ 1.0096, DP ≤ 1.0000.
6. **Static drop** (test case, n=16/32, DR 1/1000, La ∞/1200): D0, D0+mm, DI, DP+mm agree within a few % (inviscid n=32 energy growth 2.18–2.21 at DR 1).
   **Translating drop** U=(1,0.37), n=32: max|u−U| 0.18 (DR 1) and 1.3–1.8 (DR 1000) for every design vs 0.0096/0.028 at rest — the loss of
   Galilean invariance is not from the transport metric (constants are exact) and is open.
7. **Cost** (n=32, band 324–354 values): band back-substitution 6–10 µs; prototype overhead (sparse rebuilds) ~0.6 ms; DG's LSQR 2–3.5 ms;
   wall/step with diagnostics D0 8–9.5 ms, DI 8.6–10.3 ms, DP 8.8–11.2 ms.
8. **Derivation check** (independent, NumPy, 653 checks, `mimetic-projection-audit/proof_review/mimetic-momentum-transport/report.md`):
   algebra verified; corrections incorporated in the note — Prop. constant (b) excludes identified values; the mass-matching 3×3 system
   has rank = number of distinct normal directions (≤ 2 directions: only partial matching); Prop. flux's conclusion holds for a moving
   interface (not purely algebraically); the θ trilemma holds when all three defects are nonzero; a *local* skew Y needs the mass
   condition per band component (one global match is not enough with several components); the λmax bound's cut-volume independence is
   measured, not proved.

## Recommendation (for the owner to decide)

- If a 𝓜-consistent transport is wanted, use **DI** (bounded, exact free stream per sweep, one back-substitution with the existing factor),
  optionally **DP + mass matching** for exact 𝓜-momentum. Do not pursue flux/skew/redistribution forms with the present assembly:
  they are unbounded at slaved values for every Δt.
- Tradeoffs: DI/DP do not give exact semi-discrete energy (R3) and are not monotone near the interface; at ρ_g = 1e-3 they take more steps.
  Mass matching moves pressure points by O(h³) (affects hydrostatics at large density ratio by that amount; not measured on RT).

## Open items

- R3 with a bounded design: needs a metric continuous in time per value (no donor switches, continuous slaving/lumping/grid-line identification).
- Gas values on faces of aperture < 0.05 at DR 1000 unbounded (D0 and DP alike); DP reached 5.2e3 u0 (floor 1e-12, CFL 0.1).
- DP's momentum restoration verified only for constant fields (not logged on non-constant runs).
- Not run: inviscid KH momentum/energy drift and energy vs Δt, RT DR3/DR1000 n=32, capillary waves DR1/DR1000 n=16/32/64, combined case, per-case
  timing (`cases.jl`, `queue.sh` ready).
- Default-path tests (`two_phase` group) not run after the wrap-up order; default path unchanged by construction (hook returns `nothing`,
  assembly refactor semantics-preserving, flags default off) — run `Pkg.test(test_args=["two_phase"])` before any merge.

## Production needs

- Walls: the metric still throws for cut cells touching a wall; DI acts only on metric dofs (boundary unknowns keep the W values).
- AMR: MimeticProjection is uniform-mesh only; DI needs nothing beyond 𝓜 on the composite mesh; mass matching is a 3×3 solve per band component.
- Implementation: replace the sparse rebuilds by `_mimetic_metric_product!`/`_mimetic_metric_solve!`, gather the extended sources directly,
  keep serial (fixed-order) reductions for thread-count bitwise reproducibility; tests for constants per sweep, the θ identities, the bound.

---

# Aperture activity (revived study item)

Switch `_PHASE_ACTIVITY[] = :aperture` (env `MM_ACTIVITY=aperture` in the drivers): a phase value is active iff its phase
aperture is > 0 (instead of staggered phase volume > 0). Applied consistently to the velocity marking
(`_mark_inactive_phase_velocities!`, `_phase_face_is_active`), the metric assembly (projection, capillary force and viscous
step act on its dofs), the transport preparation (aperture-inactive values are extended like absent ones) and the WY velocity
check. Staggered masses of values without a dof (volume, no wetted face) and the excess above κ𝓜_ff of small values are
merged into the same-axis, same-phase ring donor with the largest aperture (`_mimetic_merged_masses`, κ = 1).
`_MIMETIC_SMALL_TIES[]` switches the (1−2a)·m ties of values with a < ½ (default on). `_APERTURE_PLAIN[]`
(`MM_APERTURE=plain`) uses the plain ½–½ average of the two cell traces without snaps. All default off; default path unchanged.
New design DGP = DG + momentum restoration; the restoration (DP, DGP) now uses the momentum of the actual update
(Zᵀ(𝓜⁺ū + rhs)), so it stays exact under the merge.

## Results

Free stream (max |u−U|/|U| after the step; 𝓜-momentum change of the constant field over 30 steps):

| design | volume activity (drop, floor 1e-12) | aperture activity (drop, floor 1e-12) | ΔP/(ρV|U|) aperture |
|---|---|---|---|
| D0, DI | 0 | 5e-17 | 2.2e-4 (obstruction) |
| DP+mm | 1.4e-16 | 5e-17 (2.4e-14 plain) | 2.4e-14 |
| DG+mm | **310** | **1.4e-14** | 2.4e-14 |
| DS / DF | 1.6e7 / 1.2e5 | 2.6e5 / 4.3e6 | 2e-2 |

One-step identities (drop, ρ_g = 1e-3, vortex): DG+mm largest correction 0.16 (volume activity 4e3–6e4); DI/DP 0.03; DS/DF 18–23.

Small-cell sweep (drop + vortex across the box, CFL 0.1/0.25/0.5 of the full-cell bound, floors √ε and 1e-12; completed runs of
6, worst wet max|u|/u0):

| design | volume ρ_g=1 / 1e-3 | aperture + small ties | aperture, no small ties |
|---|---|---|---|
| D0 | 6/6 1.01 / 5/6 6.33 | 6/6 1.03 / 6/6 3.35 | 0/6 / 1/6 |
| DI | 6/6 1.03 / 6/6 3.35 | 6/6 1.03 / 6/6 2.28 | 0/6 / 0/6 |
| DP+mm | 6/6 1.03 / 6/6 5.31 | 6/6 1.03 / 6/6 1.89 | 0/6 / 0/6 |
| DG+mm | 0/6 (stops at step 1–21) | 1/6 / 0/6 (stops at 46–761) | 0/6 |
| DS, DF | 0/6 | 0/6 | 0/6 |

Without the small ties every design fails, the current transport included: newborn values with Gram inertia O(b) get O(1/b)
increments from the projection. The small ties are needed by the projection and bring an O(m_f) jump at each birth.

Static/translating drop (n = 32): D0/DI/DP+mm keep the static energy growth (DR 1: 2.18–2.22) and DR 1 currents; at DR 1000
the viscous static gas current doubles (2.9e-2 → 5.8e-2, Ca 7.4e-4 < 5e-3) and the translating drop is 20–60 % worse.
DG+mm stops (non-finite) in the DR 1000 translating drop at n = 16.

Drop impact (pool + drop at −2, DR 1000, σ = 0.05, n = 32, t ≤ 0.25, package CGFM):

| run | steps | max E/E0 | max |u| | where |
|---|---|---|---|---|
| D0 volume | 779 | **186** | 4.6e4 | slaved gas value (a = 0, α = 1.1e-8) |
| D0 aperture | 3092 | 1.12 | 1.2e4 | gas, a = 5e-5, α = 7e-9 |
| DI aperture | — | 1e14 | 5e8 | blew up at t = 0.207 |
| DP+mm aperture | 586 | 1.19 | 1.2e4 | gas, a = 1.3e-4, α = 4e-8 |
| DG+mm aperture | — | ∞ | — | non-finite at t = 0.065 |

Time continuity of the apertures (translating drop/band, 60 steps, CFL 0.45): births ≈ deaths ≈ 18/step (drop), 73/step
(band), equal to the births/deaths of staggered volume; aperture at birth median 0.20 (min 2e-4); surviving values jump by up
to 0.997 in one step; ~30 (drop) / 136 (band) small-tie switches per step. A value is born with a finite aperture, not a→0.

Plain ½–½ aperture: the current definition already is the ½–½ mean of the two PLIC traces; the only clip exceptions are the
64ε round-off snap (`_snap_phase_apertures`), the pure-phase shortcut (identical to the mean), the VOF fraction floor (cells
within √ε of 0/1 snapped, indirect) and, in the metric only, the grid-line identification (and the same snap on boundary
faces). No pure-neighbour rule exists on this branch. Removing the snap changed nothing measurable: identical transitions
(one death on a 1e-16 opening), free stream, small-cell sweep (identical step counts) and impact (same run).

Energy as dt → 0: LW without the CTU switch is unstable on the vortex drop for every design (current included) within 2–60
steps at ρ_g = 1e-3; completed runs at ρ_g = 1: DI ΣΔE_transport/E0 = 5.5e-5, 1.8e-4, 1.9e-4 for dt, dt/2, dt/4 (not
decreasing); DGP 2.2e-3, 3.2e-3. The same runs: |ΣΔP|/S ≤ 6e-11 for DP+mm and DGP+mm vs 1e-4–1e-3 for D0, DI.

Derivation check (Python, 379 checks, `mimetic-projection-audit/proof_review/mimetic-momentum-transport-aperture/`):
rows O(b) only for closed data; metric not continuous at cut→uncut, lumping, non-closed data; birth/death changes O(b), not
O(𝓜_ff); the merge does not bound λmax(W̃,𝓜) without small ties (~1e-3/b); θ-family momentum/energy and DG lose exactness by
the merge term μ (constants stay exact); corrections incorporated in the note.

## What works, what does not

- Works: aperture activity + small ties keeps D0/DI/DP bounded at every CFL ≤ ½ with cut volumes to 1e-12, removes the slaved
  outliers and the 186·E0 energy excursion of the current method in the impact; DG becomes exact for constants to round-off;
  DP/DGP with mass matching give exact 𝓜-momentum under the merge.
- Does not: a time-continuous metric (births at a ≈ 0.2, aperture jumps ~1/step, cut→uncut jumps); boundedness of DG/DS/DF;
  semi-discrete energy; the small-aperture (1e-5–1e-4) gas outliers of 1e4 in the impact; DI is not robust in the impact.

## Recommendation (owner decides)

Aperture activity is worth keeping only as a fix of the current method's slaved-value pathology (with small ties). It does not
make an 𝓜-native transport with explicit metric-change terms viable. For a metric-consistent transport DP + mass matching
remains the candidate. A time-continuous metric would need geometry that moves continuously (e.g. advected reconstruction
normals or blocks from the VOF donating regions).

## Open risks

- Small-aperture gas values (a ~ 1e-5–1e-4, α ~ 1e-8) reach 1e4 in the impact with every design.
- The merge has uncovered cases (no donor, identified donor) and adds the momentum/energy term μ.
- DR 1000 static/translating drop currents are 1.2–2× worse under aperture activity.
- Not run: KH, RT, waves, combined, cost.
