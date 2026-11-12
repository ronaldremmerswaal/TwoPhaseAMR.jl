# Audit: sec:momentum:cfl — time-step restriction and the uniform-mesh dual Courant bound

- Source: `docs/techref/chapters/06_momentum_transport.tex:542-575` (`eq:momentum:cfl` at 545-548; two-phase paragraph at 560-566); commit `ad10545` (branch `docs/techref-momentum`, PR #66); chapter 5 `eq:vof:cfl` (`wycflval`)
- Code audited (normative): `src/operators/advection/staggered_update.jl` (`stable_timestep`, `_accumulate_staggered_rates!`), `src/two_phase/stepping.jl` (`_shared_velocity_timestep_bound`, `_transport_substep!`: `cfl = 0.5`, `dt ≤ bound (1 + 32ε)`), `src/vof/fluxes.jl`, `src/two_phase/phase_face_transport.jl`
- Date: 2026-11-08
- Overall status: ✅ verified, and **sharper than stated**: the bound holds on every uncut 2:1 mesh (not only uniform) and is attained; "without solid" is necessary; one wording imprecision; the single-phase convexity range is larger than printed
- Rerun: `julia --project=@claude-symbolics proof_review/sec-momentum-cfl/scripts/run_all.jl`

## Statement and assumptions

**Claim 4 (as printed).** "On a uniform mesh without solid" the volume-of-fluid time-step bound `in_c, out_c ≤ ν† |c|` (`ν† = wycflval ≤ ½`) bounds the dual Courant numbers `ν_g = |δt Υ^π_g| / min(V_{g^-}, V_{g^+})` of the phase transport: a dual face through a cell centre carries "half of an incoming and half of an outgoing phase volume of that cell, each at most ν†|c|", a transverse dual face half of a face volume of each of two cells, so `ν_g ≤ ν† ≤ ½` with `V_f = |c|`.

**Assumptions made explicit.**
1. `Φ_h = δt |h| υ^π_h` are the time-integrated phase volumes (carrier, liquid, gas) of the sweep axes, `|Φ^π_h| ≤ |Φ^o_h|`, and the WY bound caps for every cell the total volume entering and the total volume leaving through all its faces by `ν† |c|`.
2. Dual-face volumes are the uncut `|c|` (no solid). With a solid `V_f` are open volumes (`min(V)` of the two unknowns).
3. Both phases and both carriers of the two-velocity model are covered (frozen and projected carrier).
4. The convexity sentence concerns the single-phase unsplit Upwind update `q^{n+1} = q − δt C(υ) q` at `δt = c δt^stag(cfl = 1)` for a discretely solenoidal `υ`.

## Steps

| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| 1. `dtstag` formula, `step!` uses `cfl = ½` with `(1+32ε)`, `simulate!` `0.45`; boundary dual faces excluded | tex 543-554 | ✅ by code inspection (not a script) | reading `stable_timestep`, `_transport_substep!`, `simulate!` | — | matches the text |
| 2. The two-phase transport has no bound of its own (WY bound + capillary + gravity) | tex 556-558 | ✅ by code inspection | `_shared_velocity_timestep_bound` | — | `stable_timestep(velocity, advection)` of the WY workspace, then the capillary/gravity minima |
| 3. `|Υ_g| ≤ ν†|c|` on the uniform mesh (normal and transverse dual faces) | tex 560-566 | ✅ verified | symbolic certificates for every sign pattern of the two face volumes (S1); package: random non-solenoidal velocities scaled to the WY step for `ν† ∈ {0.5, 0.45, 0.3}` with random interfaces, carrier/liquid/gas volumes (B1); two-velocity frozen/projected carriers (D1) | `step01_courant_symbolic.jl` (132 tests), `code_01_courant.jl` | max `ν_g/ν†` = 0.807 (random fields), 0.789 (frozen) and 0.860 (projected) |
| 4. Wording "half of an incoming and half of an outgoing phase volume" | tex 562-563 | ⚠️ imprecise | symbolic S1 | `step01_courant_symbolic.jl` | it describes only the sign pattern in/out; both faces carrying inflow or both outflow give `½(|a|+|b|) ≤ ½ in_c` or `½ out_c`. Correct sentence: half of the volume through each of two faces of the cell, each ≤ ν†|c| |
| 5. The bound is attained (sharp) | (new) | ✅ verified | translation on the uniform mesh; diagonal translation | `sharp_01_courant_extremizer.jl` (B2) | `ν_g = ν†` exactly for an axis-parallel translation, `ν†/2` for a diagonal one |
| 6. Extension: the bound holds on every uncut 2:1-balanced mesh (periodic, walled, mixed; 1–2 levels) | (new) | ✅ numerical (exact LP per configuration type); ⚠️ no closed proof | for every distinct local configuration of dual faces: exact supremum of `|Υ|/min V` over all carrier volumes admitted by the WY bound, by linear programs over sign patterns | `sharp_01_courant_extremizer.jl` (A1), `step01_courant_symbolic.jl` S2 (the hanging-segment mechanism) | 69 072 dual faces, 101 configuration types, supremum exactly 1.0 (never more); smallest configuration 1/6. A fine control volume is smaller (`V = 3/8 |c|`) but its small cell caps the volumes through its faces: symbolic S2 gives exactly `ν V` |
| 7. "without solid" is necessary | tex 560 | ✅ verified | same LPs on cut meshes | `sharp_01_…` (A2, reported) | exact suprema per mesh `[1.0, 3.298, 1.0, 1.65, 2.416, 1.0, 1.027, 1.0, 1.0, 1.0, 1.097, 1.0]`: up to 3.3 ν† in small open volumes |
| 8. Convexity of the unsplit upwind update of a solenoidal velocity | tex 554-556 | ✅ verified, **range larger than printed** | symbolic S3 (`Out = In = ½ Σ|U|`, coefficient `1 − c/2`); package E1: periodic composite meshes, convex combination for every `c ≤ 2`, negative diagonal for `c > 2`, row sums one | `step01_…` S3, `code_01_courant.jl` (E1) | the printed range `cfl ≤ 1` of `eq:momentum:cfl` is not needed for convexity; with `step!` (`cfl = ½`) and `simulate!` (0.45) the margin is a factor 4 |
| 9. Same with walls (boundary dual faces excluded from the bound) | tex 546-548 | ✅ verified for `c ≤ 1` | package E2: walled composite meshes, `NoSlip`, solenoidal velocity, unknowns outside limitation (i) | `code_01_courant.jl` (E2) | smallest diagonal coefficient at `c = 1`: 0.489 over 10 492 unknown-checks; the boundary dual faces add to the outflow, so the factor 2 of step 8 is not available with walls |
| 10. No TVD/stability statement for split second-order updates | tex 556-558 | ✅ (nothing claimed) | — | — | consistent |

## Errors and consequences

No mathematical error. Items:

1. **Wording (step 4)**: the sentence "half of an incoming and half of an outgoing phase volume" is only one of four sign cases; the bound is nevertheless true in all four (symbolic certificates). Replacement below.
2. **Not sharp (step 6)**: "uniform" is not needed; the bound `ν_g ≤ ν†` holds on all uncut 2:1 meshes tested. Reported as a numerical, per-configuration result (exact LPs), not as a proof; a case-by-case proof over the dual-face types (N inside a cell, N across a hanging side, T) is possible and not written.
3. **Not sharp (step 8)**: the convexity range is `cfl ≤ 2` on periodic meshes (and `cfl ≤ 1` guaranteed with walls in the tests), not `≤ 1`.
4. **Hypothesis to keep (step 7)**: with a solid the dual Courant number can exceed `ν†` by up to 3.3 in small open volumes: the restriction "without solid" must stay (and the single-phase `δt^stag` uses the open volumes for exactly this reason).

## Sharpness

| Claim / assumption | Classification | Evidence (extremizer, counterexample, rate) | Script | Possible improvement |
|---|---|---|---|---|
| `ν_g ≤ ν†` (uniform, no solid) | **sharp** | axis-parallel translation attains 1 (extremizer); diagonal translation ½ | `sharp_01_courant_extremizer.jl` (B2) | — |
| "uniform mesh" | **not needed** (improvement shown numerically) | exact LP supremum 1.0 on all uncut 2:1 meshes, 101 configuration types | `sharp_01_…` (A1) | state the 2:1-balanced version (replacement below) |
| "without solid" | **necessary** | LP supremum up to 3.3 ν† (cut meshes) | `sharp_01_…` (A2) | keep |
| `ν† ≤ ½` | trivial (`cfl = ½` in `step!`) | code inspection | — | — |
| Convex combination for `cfl ≤ 1` (single phase) | **not sharp** by a factor 2 on periodic meshes | coefficient `1 − c/2`; negative for `c > 2` | `step01_…` S3, `code_01_…` E1 | state `cfl ≤ 2` (periodic) |

## Open points

- A closed proof of step 6 for the three dual-face types (the LP exhausts the configurations found in random meshes, not all conceivable ones).
- After issue #62 the Lax–Wendroff/MC states are used on bounded and solid schedules; `ν_g` up to 3.3 ν† there (step 7) then matters for the sign of the `(1 − ν_g)` coefficient (to be checked once #62 lands).

## Effect of issues #60–#64

- **#62** (scheme selection on bounded/solid schedules): the bulk state of the selected scheme becomes active there, so the exceedance of the Courant bound with a solid (step 7) becomes relevant for stability; the bound proved here (no solid) is unaffected.
- **#64** (cell-scalar `stable_timestep` ignores apertures): affects the cell-scalar bound, not the staggered bound audited here; the statement "small open volumes reduce the bound" is about the staggered `stable_timestep` and stays true.
- **#60/#61/#63**: not involved.

## Proposed replacement text

Replaces from `On a uniform mesh` up to `with $\stagvol{f}=\meas{c}$.` (second paragraph of the section; only existing macros) and, optionally, the convexity sentence.

```latex
On a uniform mesh
without solid this bounds the dual Courant numbers: a dual face through a
cell centre carries half of the volume through each of two faces of that
cell, each at most $\wycflval\meas{c}$ (the volume through a face is at most
the volume entering or leaving the cell), and a transverse dual face half of a
face volume of each of two cells, so $\mathcfl_g\le\wycflval\le\half$
with $\stagvol{f}=\meas{c}$; a uniform translation attains $\wycflval$. The same
bound holds on every $2{:}1$-balanced mesh without solid, periodic or with
walls: a hanging fine control volume is smaller, but the volumes through the
faces of its small cell are capped by its own volume (checked as exact linear
programs over every local configuration of random meshes, not proved here).
With a solid it can fail, because the open volumes are small: $\mathcfl_g$
reached $3.3\,\wycflval$ in random cut meshes.
```

```latex
Under the bound the unsplit upwind update of a solenoidal velocity is a convex
combination of old values: the coefficient of the old value is at least
$1-\half\stagcfl$ on a periodic mesh, so $\stagcfl\le2$ suffices there (with
walls the boundary dual faces, which are not in the bound, add to the outflow,
and the smallest coefficient at $\stagcfl=1$ was $0.49$ in the tests).
```
(The second block replaces `Under the bound\nthe unsplit upwind update is a convex combination of old values when the\nvelocity is solenoidal (test below).`)

## History

- 2026-11-08: first audit against `ad10545`; package-level checks on the clean tree. Scripts: 1 symbolic (132 tests), 2 package (21 221 tests incl. the exact LPs). Outcome: bound verified, sharp, and extended to uncut 2:1 meshes (numerical); wording and range improvements.
- 2026-11-12: issue #72 resolved. After #139 the cut-donor bound limits every outgoing flux by its donor's open strip, and the phase schedules weigh the fluxes by those same strips (#68 and the open-strip schedule shared with the two-velocity momentum in this branch), so the dual Courant number obeys `nu_g <= nu^dagger` with a solid as well: measured on the code's schedules over 40 random cut meshes with a level transition at `cfl` 0.5 and 0.3 the largest ratio is 1.0 to round-off, at regular and at interface dual faces, and the bound is attained. The `A2` supremum of 3.3 `nu^dagger` is therefore a supremum over states the older per-cell bound admitted, not over the current admissible set; the excluded line must not be read as attainable.
