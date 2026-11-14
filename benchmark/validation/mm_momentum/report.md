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
