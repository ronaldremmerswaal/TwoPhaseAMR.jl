# Two-velocity verification

The two-velocity verification suite, `benchmark/two_velocity/`, compares the
two-velocity pressure projections on problems with exact or reference answers
and tracks the progress on the small phase values of the mimetic projection.
It **measures and reports**; the contracts of the package stay in `test/`.

| Label | Method |
| --- | --- |
| CMDGFM-L | [`CMDGFMProjection`](@ref)`()` with [`LinearJumpInference`](@ref)`()` (the default) |
| CMDGFM-C | [`CMDGFMProjection`](@ref)`(jump_inference = ConstantJumpInference())`, the published rows (thesis reference) |
| MM | [`MimeticProjection`](@ref)`()` |
| 1-vel | [`OneVelocityModel`](@ref)`()`, the reference of the time-dependent tiers |

| Tier | Content | Cost (one thread) | Where |
| --- | --- | --- | --- |
| 1 | single projections with exact answers, T1–T7, n = 32, 64, 128 | 45 s for all methods, plus compilation | CI (job `two-velocity-verification`, required by `ci-ok`), with `--compare` |
| 2 | short flows: rest, static drop, capillary wave, Kelvin–Helmholtz, translation, label exchange | 20 min for all methods (the Kelvin–Helmholtz ensemble 15 of them) | local |
| 3 | drop impact, n = 32 members 1–5 and n = 64 | 2 min for all methods | local |

## Running the suite

From the repository root, once per checkout:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

then

```sh
julia --project=benchmark -t 1 benchmark/two_velocity/run.jl --tier=1 --compare
julia --project=benchmark -t 1 benchmark/two_velocity/run.jl --tier=2,3 --methods=mimetic,cmdgfm --compare
```

The run writes `results.toml` and `results.md` (all values, with observed
orders) to `benchmark/two_velocity/output/`. `--compare` judges every metric
against `benchmark/two_velocity/baseline.toml`: *worse* when it moved in the
wrong direction by more than `max(rtol |baseline|, atol)`, *better* when it
moved the other way by as much. The relative tolerance is `1e-3` in tier 1
(single projections, deterministic up to the projection tolerance), `2e-2` in
tier 2 and `0.25` in tier 3 (the impact is singular); exact cases carry an
absolute tolerance at their solver level. Orders, counts and step numbers are
reported only. The run fails (exit status 1) when a metric is worse, a
measured metric of a case that ran is missing, or a case failed (an exception
or a projection that did not converge). `--record` writes the run's tiers and
methods into the baseline and `--update-docs` regenerates the tables below
from it; re-record only for an intentional change and explain it in the pull
request.
`TWO_VELOCITY_PERTURB=1` loosens the projection tolerance to `1e-3` to show
the failure path. Further options: `--methods`, `--resolutions`, `--impact`,
`--output` (see the header of `run.jl` and `benchmark/README.md`).

## Tier 1: single projections

One projection with `Δt = 1` per case and method (tolerance `1e-8` on
`Δt max|div u_mix|`), on the package geometry (PLIC apertures and staggered
fractions) of a uniform `n × n` mesh of the unit square, periodic in `x`,
walls at `y = 0` and `y = 1`, liquid below the graph `y = η(x)` or inside the
circle. The interface lies off the grid lines (`0.37h` above, circle centres
shifted by `(0.21h, 0.37h)`). These are the laboratory tests T1–T7 of the
mimetic-projection audit, run with every two-velocity method. Norms follow
the audit: the rms over the mixed faces is weighted by the relative mass
`K = m_l m_g/(m_l + m_g)` (`m_π` the phase mass of the staggered volume).

- **T1 continuous shear.** `u = ½ tanh((y − ½)/0.05) e_x` in both phases is
  single valued and solenoidal: the projection must keep it and create no
  phase-velocity jump. Flat, cosine (`a = 0.01`) and triangle (slopes 0.06 and
  0.3) interfaces, DR 1 and 1000.
- **T2 vortex sheet** (liquid `−½`, gas `+½` in `x`) and **T3 gravity** (both
  phases `−1` in `y`, `g Δt = 1`) over the wavy interface
  `y = ½ + 0.37h + ε cos 2πx`, `ε = 0.005`, against the exact first-order
  per-phase fields (harmonic pressures, extended analytically beyond the
  interface): the jump error divided by `U ε k` (`g Δt ε k`), split into
  faces with both values wetted and faces with a value without aperture, and
  each value against its own phase's exact field. T3 at DR 1 is exact rest.
- **T4 sheet over tilted and curved interfaces** (triangle slopes 0.3 and 1,
  circle `R = 0.25`, DR 1): the projected jump must be the tangential part
  `J − n(n·J)` of the sheet jump `J`; ±0.1 around the kinks are excluded.
- **T5 single-valued gradient predictor** `u* = −Gφ`, `φ = cos 2πx cos πy`
  (DR 1): exact answer rest.
- **T6 exact two-phase gradient** `∇p_π/ρ_π` of harmonic pressures that are
  continuous with a continuous normal flux at a flat interface: exact answer
  rest.
- **T7 manufactured Poisson problem** below, with walls (predictor
  `∇p_π/ρ_π`, exact answer rest and `p`; DR 1 and 1000) and periodic in both
  directions with the divergence-free rotation of the former gallery example
  12 (DR 1000).

### T7: manufactured pressure with a gradient jump

This is the former gallery example 12. Let `r² = (x − ½)² + (y − ½)²`, liquid
inside the circle `r < 0.3`. The pressure is continuous at the interface but
has a phase-dependent gradient:

```math
p_*(x,y)=\sin(2\pi x)+\cos(2\pi y)
+40\,(r^2-0.3^2)(x-y)\,\mathbf 1_{r<0.3}.
```

Equal `x`/`y` amplitudes make the normal and tangential surface-rms parts of
the scaled jump `ξ = [∇p/ρ]` nearly equal on the circle (the suite checks
their ratio, 0.8–1.25). In the periodic case the predictor is the
divergence-free liquid rotation `u = 2 cos(2πr²) (y − ½, −(x − ½))` (gas at
rest) plus `∇p_*/ρ`, so one projection with `Δt = 1` should return the
rotation and `p_*`. Periodic pressure has an arbitrary constant: the pressure
errors subtract its volume-weighted mean. The phase-gradient errors are
errors of the **physical** gradients realised by the projection,
`ρ_π (u*_π − u⁺_π)/Δt` (rms weighted by the phase fraction of the dual
volume; the maximum excludes nearly absent phases), and `xi` is the error of
their scaled jump on the mixed faces. At `n = 32` the published rows give the
numbers the example stated (pressure rms/maximum `1.04/8.99`, phase gradient
`90.5/5305`).

### Tier-1 results

Values at `n = 32 / 64 / 128` and, in parentheses, the observed order of the
finest pair.

<!-- tier1 table: begin -->

_Generated from `benchmark/two_velocity/baseline.toml` by `run.jl --update-docs` (Julia 1.12.7, revision 189ba44f)._

| test | case | metric | n | CMDGFM-L | CMDGFM-C | MM |
|---|---|---|---|---|---|---|
| T1 | cos a=0.01 DR=1 | `max_change` | 32 / 64 / 128 | 0.217 / 0.0481 / 0.00747 (2.69) | 0.52 / 0.297 / 0.155 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | cos a=0.01 DR=1 | `max_jump` | 32 / 64 / 128 | 0.222 / 0.0493 / 0.0075 (2.71) | 0.533 / 0.304 / 0.158 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | cos a=0.01 DR=1000 | `max_change` | 32 / 64 / 128 | 0.332 / 0.0737 / 0.0131 (2.49) | 0.693 / 0.397 / 0.207 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | cos a=0.01 DR=1000 | `max_jump` | 32 / 64 / 128 | 0.334 / 0.0742 / 0.0132 (2.49) | 0.697 / 0.399 / 0.208 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | flat 0.37h DR=1 | `max_change` | 32 / 64 / 128 | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] |
| T1 | flat 0.37h DR=1 | `max_jump` | 32 / 64 / 128 | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] |
| T1 | flat 0.37h DR=1000 | `max_change` | 32 / 64 / 128 | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] |
| T1 | flat 0.37h DR=1000 | `max_jump` | 32 / 64 / 128 | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.06 DR=1 | `max_change` | 32 / 64 / 128 | 0.217 / 0.0481 / 0.00745 (2.69) | 0.483 / 0.276 / 0.147 (0.90) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.06 DR=1 | `max_jump` | 32 / 64 / 128 | 0.223 / 0.0492 / 0.00755 (2.71) | 0.519 / 0.296 / 0.155 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.06 DR=1000 | `max_change` | 32 / 64 / 128 | 0.332 / 0.0737 / 0.0112 (2.72) | 0.613 / 0.352 / 0.183 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.06 DR=1000 | `max_jump` | 32 / 64 / 128 | 0.334 / 0.0742 / 0.0112 (2.72) | 0.616 / 0.354 / 0.184 (0.94) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.3 DR=1 | `max_change` | 32 / 64 / 128 | 0.585 / 0.282 / 0.166 (0.76) [L] | 0.418 / 0.244 / 0.128 (0.93) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.3 DR=1 | `max_jump` | 32 / 64 / 128 | 0.597 / 0.289 / 0.171 (0.75) [L] | 0.503 / 0.289 / 0.152 (0.93) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.3 DR=1000 | `max_change` | 32 / 64 / 128 | 0.612 / 0.266 / 0.156 (0.77) [L] | 0.509 / 0.293 / 0.153 (0.93) [C1] | 0 / 0 / 0 [E] |
| T1 | triangle s=0.3 DR=1000 | `max_jump` | 32 / 64 / 128 | 0.632 / 0.276 / 0.164 (0.74) [L] | 0.528 / 0.303 / 0.159 (0.93) [C1] | 0 / 0 / 0 [E] |
| T2 | sheet wavy ε=0.005 DR=1 | `e2` | 32 / 64 / 128 | 0.0141 / 0.00846 / 0.00478 (0.83) | 0.0119 / 0.00777 / 0.00995 (-0.36) [G] | 0.0508 / 0.0271 / 0.0144 (0.92) |
| T2 | sheet wavy ε=0.005 DR=1 | `e2_slaved` | 32 / 64 / 128 | 0.00549 / 0.00129 / 4.69e-04 (1.46) | 0.00541 / 0.00123 / 0.0105 (-3.09) [G] | 0.0644 / 0.0335 / 0.0173 (0.96) |
| T2 | sheet wavy ε=0.005 DR=1 | `e2_wetted` | 32 / 64 / 128 | 0.0218 / 0.0142 / 0.00845 (0.75) | 0.0181 / 0.0131 / 0.00866 (0.59) | 0.00477 / 0.00324 / 0.00283 (0.20) |
| T2 | sheet wavy ε=0.005 DR=1 | `extension` | 32 / 64 / 128 | 0.0202 / 0.0135 / 0.0108 (0.32) | 0.02 / 0.0132 / 0.0112 (0.23) | 0.0322 / 0.0193 / 0.0135 (0.51) |
| T2 | sheet wavy ε=0.005 DR=1000 | `e2` | 32 / 64 / 128 | 0.0242 / 0.0155 / 0.0101 (0.62) | 0.168 / 0.116 / 0.142 (-0.29) [G] | 0.0703 / 0.0391 / 0.02 (0.97) |
| T2 | sheet wavy ε=0.005 DR=1000 | `e2_slaved` | 32 / 64 / 128 | 0.00541 / 0.00122 / 0.00419 (-1.77) [G] | 0.00559 / 0.00153 / 0.135 (-6.46) [G] | 0.0817 / 0.0436 / 0.0222 (0.97) |
| T2 | sheet wavy ε=0.005 DR=1000 | `e2_wetted` | 32 / 64 / 128 | 0.0397 / 0.0259 / 0.0171 (0.60) | 0.28 / 0.194 / 0.158 (0.29) | 0.0432 / 0.0294 / 0.0139 (1.08) |
| T2 | sheet wavy ε=0.005 DR=1000 | `extension` | 32 / 64 / 128 | 0.0278 / 0.0143 / 0.00504 (1.51) | 0.0457 / 0.0249 / 0.0118 (1.08) | 0.00234 / 0.00143 / 7.93e-04 (0.85) |
| T3 | gravity wavy ε=0.005 DR=1 | `e2` | 32 / 64 / 128 | 2.98e-14 / 4.60e-14 / 6.16e-14 [E] | 2.98e-14 / 4.26e-14 / 6.15e-14 [E] | 4.90e-11 / 1.09e-10 / 8.57e-11 [E] |
| T3 | gravity wavy ε=0.005 DR=1 | `e2_slaved` | 32 / 64 / 128 | 0 / 4.96e-16 / 1.65e-15 [E] | 0 / 4.96e-16 / 1.61e-15 [E] | 1.71e-11 / 5.57e-11 / 2.78e-11 [E] |
| T3 | gravity wavy ε=0.005 DR=1 | `e2_wetted` | 32 / 64 / 128 | 4.84e-14 / 7.79e-14 / 1.09e-13 [E] | 4.84e-14 / 7.23e-14 / 1.09e-13 [E] | 7.65e-11 / 1.68e-10 / 1.47e-10 [E] |
| T3 | gravity wavy ε=0.005 DR=1 | `extension` | 32 / 64 / 128 | 2.06e-10 / 3.13e-12 / 2.23e-11 [E] | 2.06e-10 / 3.11e-12 / 2.23e-11 [E] | 2.51e-10 / 5.52e-10 / 5.05e-10 [E] |
| T3 | gravity wavy ε=0.005 DR=1 | `max_u` | 32 / 64 / 128 | 2.16e-11 / 2.55e-12 / 3.40e-12 [E] | 2.16e-11 / 2.55e-12 / 3.40e-12 [E] | 5.30e-11 / 6.76e-11 / 4.55e-11 [E] |
| T3 | gravity wavy ε=0.005 DR=1000 | `e2` | 32 / 64 / 128 | 0.12 / 0.112 / 0.767 (-2.77) [G] | 0.12 / 0.112 / 0.638 (-2.51) [G] | 0.0746 / 0.0485 / 0.172 (-1.82) [G] |
| T3 | gravity wavy ε=0.005 DR=1000 | `e2_slaved` | 32 / 64 / 128 | 0.00598 / 0.00388 / 0.525 (-7.08) [G] | 0.00598 / 0.00388 / 0.18 (-5.53) [G] | 0.0516 / 0.0438 / 0.0805 (-0.88) [G] |
| T3 | gravity wavy ε=0.005 DR=1000 | `e2_wetted` | 32 / 64 / 128 | 0.2 / 0.187 / 1.14 (-2.60) [G] | 0.2 / 0.187 / 1.12 (-2.58) [G] | 0.104 / 0.0559 / 0.285 (-2.35) [G] |
| T3 | gravity wavy ε=0.005 DR=1000 | `extension` | 32 / 64 / 128 | 0.157 / 0.326 / 0.421 (-0.37) [G] | 0.157 / 0.326 / 0.42 (-0.37) [G] | 0.0726 / 0.0415 / 0.0601 (-0.53) [G] |
| T4 | circle R=0.25 DR=1 | `e2` | 32 / 64 / 128 | 0.0744 / 0.0528 / 0.0409 (0.37) | 0.197 / 0.192 / 0.186 (0.04) [C2] | 0.0424 / 0.0225 / 0.0115 (0.96) |
| T4 | triangle s=0.3 DR=1 | `e2` | 32 / 64 / 128 | 0.0119 / 0.00822 / 0.00609 (0.43) | 0.0484 / 0.048 / 0.0484 (-0.01) [C2] | 0.0156 / 0.00819 / 0.00393 (1.06) |
| T4 | triangle s=1 DR=1 | `e2` | 32 / 64 / 128 | 0.198 / 0.119 / 0.143 (-0.26) [L] | 0.292 / 0.292 / 0.292 (0.00) [C2] | 0.018 / 0.00869 / 0.00415 (1.06) |
| T5 | circle R=0.25 DR=1 | `max_jump` | 32 / 64 / 128 | 2.51e-12 / 2.90e-15 / 6.50e-15 [E] | 1.57e-12 / 2.20e-15 / 4.97e-15 [E] | 0.201 / 0.103 / 0.0664 (0.64) [M] |
| T5 | circle R=0.25 DR=1 | `max_u` | 32 / 64 / 128 | 4.49e-12 / 5.09e-15 / 7.53e-14 [E] | 5.22e-12 / 2.62e-14 / 1.65e-13 [E] | 0.148 / 0.0783 / 0.0427 (0.88) [M] |
| T5 | triangle s=0.3 DR=1 | `max_jump` | 32 / 64 / 128 | 3.92e-13 / 3.12e-15 / 2.90e-15 [E] | 2.14e-12 / 1.54e-15 / 3.29e-15 [E] | 0.102 / 0.0537 / 0.0272 (0.98) [M] |
| T5 | triangle s=0.3 DR=1 | `max_u` | 32 / 64 / 128 | 6.39e-12 / 9.84e-15 / 7.85e-14 [E] | 6.27e-12 / 3.04e-14 / 1.94e-13 [E] | 0.107 / 0.0567 / 0.0287 (0.98) [M] |
| T5 | wavy ε=0.02 DR=1 | `max_jump` | 32 / 64 / 128 | 1.03e-15 / 1.52e-12 / 5.48e-13 [E] | 1.10e-15 / 1.19e-12 / 6.27e-15 [E] | 0.0838 / 0.0483 / 0.0248 (0.96) [M] |
| T5 | wavy ε=0.02 DR=1 | `max_u` | 32 / 64 / 128 | 2.27e-15 / 1.30e-11 / 8.96e-12 [E] | 1.98e-15 / 5.04e-12 / 1.53e-13 [E] | 0.0826 / 0.0509 / 0.0261 (0.97) [M] |
| T6 | flat 0.37h DR=1 | `jump_ratio` | 32 / 64 / 128 | 0.00234 / 6.19e-04 / 1.55e-04 (2.00) | 0.0203 / 0.0102 / 0.00509 (1.00) | 0.0966 / 0.0498 / 0.0253 (0.98) [M] |
| T6 | flat 0.37h DR=1 | `mean_ratio` | 32 / 64 / 128 | 0.0846 / 0.0439 / 0.0224 (0.97) | 0.0846 / 0.0439 / 0.0224 (0.97) | 0.064 / 0.0315 / 0.0156 (1.01) [M] |
| T6 | flat 0.37h DR=1000 | `jump_ratio` | 32 / 64 / 128 | 0.0182 / 0.00522 / 0.0014 (1.90) | 0.613 / 0.607 / 0.603 (0.01) [C2] | 0.0857 / 0.0429 / 0.0215 (1.00) [M] |
| T6 | flat 0.37h DR=1000 | `mean_ratio` | 32 / 64 / 128 | 0.00425 / 5.83e-04 / 7.96e-05 (2.87) | 0.141 / 0.0663 / 0.032 (1.05) | 0.00386 / 0.00104 / 2.70e-04 (1.95) [M] |
| T7 | circle R=0.3 walls DR=1 | `jump_ratio` | 32 / 64 / 128 | 0.0673 / 0.0312 / 0.0566 (-0.86) [?] | 0.318 / 0.38 / 0.345 (0.14) [C2] | 0.356 / 0.173 / 0.0901 (0.94) [M] |
| T7 | circle R=0.3 walls DR=1 | `mean_ratio` | 32 / 64 / 128 | 0.0341 / 0.0261 / 0.0285 (-0.13) [?] | 0.0469 / 0.088 / 0.0752 (0.23) [C2] | 0.076 / 0.0493 / 0.0234 (1.07) [M] |
| T7 | circle R=0.3 walls DR=1 | `p_rms_centre` | 32 / 64 / 128 | 0.00919 / 0.00232 / 8.51e-04 (1.45) | 0.00872 / 0.00186 / 7.83e-04 (1.25) | 0.0151 / 0.00735 / 0.00247 (1.57) |
| T7 | circle R=0.3 walls DR=1 | `p_rms_point` | 32 / 64 / 128 | 0.00919 / 0.00232 / 8.51e-04 (1.45) | 0.00872 / 0.00186 / 7.83e-04 (1.25) | 0.00638 / 0.00334 / 5.42e-04 (2.62) |
| T7 | circle R=0.3 walls DR=1000 | `jump_ratio` | 32 / 64 / 128 | 0.369 / 0.278 / 0.395 (-0.50) [?] | 0.71 / 0.721 / 0.717 (0.01) [C2] | 0.0597 / 0.032 / 0.0157 (1.03) [M] |
| T7 | circle R=0.3 walls DR=1000 | `mean_ratio` | 32 / 64 / 128 | 0.525 / 0.344 / 0.85 (-1.31) [?] | 0.764 / 0.822 / 0.832 (-0.02) [C2] | 0.0108 / 0.00858 / 0.0042 (1.03) [M] |
| T7 | circle R=0.3 walls DR=1000 | `p_rms_centre` | 32 / 64 / 128 | 1.01 / 0.318 / 0.114 (1.48) | 1.04 / 0.727 / 0.259 (1.49) | 0.0171 / 0.00974 / 0.00276 (1.82) |
| T7 | circle R=0.3 walls DR=1000 | `p_rms_point` | 32 / 64 / 128 | 1.01 / 0.318 / 0.114 (1.48) | 1.04 / 0.727 / 0.259 (1.49) | 0.00539 / 0.00345 / 5.43e-04 (2.67) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `grad_l2` | 32 / 64 / 128 | 70.2 / 31.3 / 25.2 (0.32) | 90.5 / 87.2 / 50.1 (0.80) | 0.297 / 0.176 / 0.053 (1.73) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `grad_linf` | 32 / 64 / 128 | 3.38e+03 / 2.66e+03 / 5.55e+03 (-1.06) [?] | 5.3e+03 / 5.32e+03 / 5.9e+03 (-0.15) [?] | 7.83 / 6.16 / 3.15 (0.97) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `p_l2` | 32 / 64 / 128 | 1.01 / 0.318 / 0.114 (1.48) | 1.04 / 0.727 / 0.259 (1.49) | 0.0171 / 0.00975 / 0.00276 (1.82) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `p_linf` | 32 / 64 / 128 | 10.8 / 4.43 / 1.88 (1.24) | 8.99 / 9.76 / 4.96 (0.97) [?] | 0.163 / 0.186 / 0.0938 (0.99) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `xi_l2` | 32 / 64 / 128 | 2.15 / 2.16 / 2.39 (-0.15) [?] | 3.23 / 3.33 / 3.32 (0.00) [?] | 0.416 / 0.255 / 0.118 (1.11) |
| T7 | ex12 periodic circle R=0.3 DR=1000 | `xi_linf` | 32 / 64 / 128 | 5.81 / 6.16 / 6.1 (0.01) [?] | 6.29 / 6.29 / 6.29 (0.00) [?] | 1.35 / 1.23 / 0.531 (1.21) |

- T1 `max_change`: largest change of a phase value, exact 0
- T1 `max_jump`: largest created phase-velocity jump, exact 0
- T2 `e2`: K-weighted rms of (j⁺ − exact)/(U ε k)
- T2 `e2_slaved`: e2 over the faces with a value without aperture
- T2 `e2_wetted`: e2 over the faces with both values wetted
- T2 `extension`: mass-weighted rms of each value against its own phase's exact field
- T3 `e2`: K-weighted rms of (j⁺ − exact)/(g Δt ε k)
- T3 `e2_slaved`: e2 over the faces with a value without aperture
- T3 `e2_wetted`: e2 over the faces with both values wetted
- T3 `extension`: mass-weighted rms of each value against its own phase's exact field
- T3 `max_u`: largest projected value at equal densities, exact rest
- T4 `e2`: K-weighted rms of j⁺ − (J* − n(n·J*))·n_f, kinks ±0.1 excluded
- T5 `max_jump`: max|j⁺|/max|Gφ|, exact rest
- T5 `max_u`: max|u⁺|/max|Gφ|, exact rest
- T6 `jump_ratio`: rms(j⁺)/rms(j*), exact rest
- T6 `mean_ratio`: max|ū⁺|/max|ū*| of the mass-weighted mean, exact rest
- T7 `jump_ratio`: rms(j⁺)/rms(j*), exact rest
- T7 `mean_ratio`: max|ū⁺|/max|ū*| of the mass-weighted mean, exact rest
- T7 `p_rms_centre`: rms pressure error at the cell centres, mean removed
- T7 `p_rms_point`: rms pressure error at the method's pressure points, mean removed
- T7 `grad_l2`: phase-fraction-weighted rms error of the realised phase pressure gradients ρ(u* − u⁺)/Δt
- T7 `grad_linf`: largest realised phase-gradient error (phase fraction > √eps)
- T7 `p_l2`: volume-weighted rms pressure error at the cell centres, gauge aligned
- T7 `p_linf`: largest pressure error at the cell centres, gauge aligned
- T7 `xi_l2`: rms error of the scaled gradient jump ξ = [∇p/ρ] on the mixed faces
- T7 `xi_linf`: largest error of the scaled gradient jump on the mixed faces
- `[?]` does not converge; cause not established
- `[C1]` CMDGFM-C one-donor rows: the two-point difference ±2hU' of the shear, first order (#171)
- `[C2]` CMDGFM-C: mixed faces without a regular row keep their predictor jump; no convergence
- `[E]` exact by construction (round-off or projection tolerance)
- `[G]` n = 128: the wave crests cross a grid line and leave liquid slivers (T2, T3); not solver limited
- `[L]` CMDGFM-L limitation pinned as @test_broken (#171)
- `[M]` MM: the sampled gradient is not a gradient of the mimetic metric in cut cells, O(h)

<!-- tier1 table: end -->

### Reading the tier-1 table

Exact by construction (`[E]`), at round-off or at the projection tolerance:

- MM keeps the shear of T1 exactly: a single-valued solenoidal field is a
  fixed point of a projection orthogonal in the metric `𝓜`.
- CMDGFM keeps the T1 shear over the flat interface (CMDGFM-L imposes the
  increment of the own jump on an exactly tangential face, CMDGFM-C gives
  such a face no row, so it keeps its jump) and removes the single-valued
  gradient of T5 to round-off.
- T3 at DR 1 is at rest to the projection tolerance for every method.

Not convergent, or first order only, by construction (method limitations,
not regressions); the brackets name the marks of the table:

- `[C1]` CMDGFM-C infers each phase from one donor face, the two-point
  difference `±2hU'` of a resolved shear (issue #171): T1 converges at first
  order, close to `20h` for this shear (DR 1 cosine 0.53 / 0.30 / 0.16).
- `[C2]` CMDGFM-C leaves the mixed faces without a regular row at their
  predictor jump ([`ConstantJumpInference`](@ref)). In T6 these are the cut
  faces parallel to the flat interface, and they carry all of the remaining
  jump: the ratio stays at 0.61 at DR 1000 for every `n`. The plateaus of T4
  and T7 come partly from such faces and partly from the one-donor rows.
- `[L]` CMDGFM-L falls back to the one-donor rule where the crossing-speed
  limiter sees the shear cross a steep interface (T1 triangle `s = 0.3`, first
  order, 0.60 / 0.29 / 0.171), and its `β = 0` rows on faces with
  `|n_t| > |n_a|` keep the donors' jump on the slope-1 triangle of T4. Both are
  pinned as `@test_broken` in
  `test/two_phase/two_velocity/projection/cmdgfm_jump_inference.jl`
  (issue #171).
- `[M]` MM leaves an `O(h)` jump in T5–T7: the sampled gradient of a
  potential is not a gradient of the mimetic metric in cut cells.
- `[G]` T2 and T3 rise from `n = 64` to `n = 128` (T3 at DR 1000 for every
  method). This is a change of geometry, not of the solve: `ε = 0.64h` at
  `n = 128`, so the crests rise `0.37h + 0.64h = 1.01h` above a grid line and
  the troughs `0.27h` below the one beneath, leaving slivers of one phase in
  the neighbouring cell rows (at `n = 32, 64` the wave stays within one row).
  A few sliver faces carry most of the K-weighted error. T3: CMDGFM-L two
  crest faces of liquid fraction 0.016 and 0.002 (error 4.2–4.5, weight `K`
  0.93 and 0.67), MM two crest faces with error 1.78 (68 % of `Σ K e²`); T2
  (CMDGFM-C): faces of liquid fraction 0.002 on the crossed grid lines. The
  T3 metrics agree to eight digits between projection tolerances `1e-7` and
  `1e-8` and to four digits down to `1e-11`; T3 at DR 1000 uses `1e-7`,
  because the round-off floor of `Δt max|div u_mix|` under the hydrostatic
  pressure `ρ_l g y` reaches `3e-9` (MM) to `5e-8` (CMDGFM) at `n = 128`.
- `[?]` Observed, cause not established here: CMDGFM-L's T7 ratios with
  walls and the scaled jump `xi` and the largest phase-gradient error of the
  periodic T7 case for both CMDGFM rules do not decrease with `n`. CMDGFM-L's
  T4 circle error converges at order 0.4 only.

## Tier 2: short flows

Each method, and the one-velocity model as the reference, runs:

- **hydrostatic rest**: a flat pool at `y = ½ + h/3` under `g = 1`, DR 1000,
  both label orders, 50 steps of `Δt = 1e-3` (exact rest);
- **static drop**: inviscid drop `R = 0.2`, `σ = 1` (CGFM), free-slip box,
  `n = 32`, `t = 0.4` with [`simulate!`](@ref), DR 1 and 1000, both label
  orders: late raw and resolved spurious currents (resolved: mixed faces with
  phase aperture > 0.05) and the energy behaviour (largest kinetic energy,
  and that of the last third over the first third);
- **capillary wave** against Prosperetti's solution: the standing wave of the
  integration test (`λ = 1`, `a₀ = λ/100`, La 3000), a quarter period,
  `n = 16, 32, 64`, DR 1 and 1000;
- **Kelvin–Helmholtz linear growth** (two-velocity methods only: one velocity
  field cannot carry the sheet's velocity jump, its error is 7.7): the
  inviscid sheet of the integration test (DR 1000, We 0.5, 32 cells per
  wavelength) against linear theory, as an ensemble: the interface on a grid
  line with the amplitude scaled by 1, `1 ± 0.0025` and `1 ± 0.005` (the noise
  of the error), shifted by `h/3`, a four times larger amplitude, and the flow
  moving at `U = 2` (Galilean);
- **translation**: a sheet translated normally at speed 1 with tangential
  streams `∓½` (an exact solution), DR 1000, a resolved layer and a film of
  1.5 cells, until `t = 2h`;
- **label symmetry**: a static drop and a capillary wave stepped eight times
  with the phase labels exchanged (level set, densities and viscosities):
  the values must exchange (issue #200).

<!-- tier2 table: begin -->

_Generated from `benchmark/two_velocity/baseline.toml` by `run.jl --update-docs` (Julia 1.12.7, revision 189ba44f)._

| test | case | metric | n | CMDGFM-L | CMDGFM-C | MM | 1-vel |
|---|---|---|---|---|---|---|---|
| hydrostatic | pool DR=1000 normal | `max_u` | 32 | 5.64e-12 [E] | 5.64e-12 [E] | 8.20e-12 [E] | 3.88e-10 [E] |
| hydrostatic | pool DR=1000 swapped | `max_u` | 32 | 2.61e-12 [E] | 2.76e-12 [E] | 7.84e-12 [E] | 3.88e-10 [E] |
| static drop | R=0.2 DR=1 normal | `K_growth` | 32 | 1.02 | 1.02 | 0.715 | 1.06 |
| static drop | R=0.2 DR=1 normal | `K_max` | 32 | 7.35e-07 | 7.39e-07 | 6.48e-07 | 7.92e-07 |
| static drop | R=0.2 DR=1 normal | `late_u` | 32 | 0.00584 | 0.0058 | 0.0162 | 0.00685 |
| static drop | R=0.2 DR=1 normal | `late_u_resolved` | 32 | 0.00584 | 0.0058 | 0.00455 | 0.00685 |
| static drop | R=0.2 DR=1 normal | `steps` | 32 | 227 | 227 | 227 | 227 |
| static drop | R=0.2 DR=1 swapped | `K_growth` | 32 | 1.02 | 1.02 | 0.715 | 1.06 |
| static drop | R=0.2 DR=1 swapped | `K_max` | 32 | 7.35e-07 | 7.39e-07 | 6.48e-07 | 7.92e-07 |
| static drop | R=0.2 DR=1 swapped | `late_u` | 32 | 0.00584 | 0.0058 | 0.0162 | 0.00685 |
| static drop | R=0.2 DR=1 swapped | `late_u_resolved` | 32 | 0.00584 | 0.0058 | 0.00455 | 0.00685 |
| static drop | R=0.2 DR=1 swapped | `steps` | 32 | 227 | 227 | 227 | 227 |
| static drop | R=0.2 DR=1000 normal | `K_growth` | 32 | 1.03 | 1.03 | 1.01 | 0.976 |
| static drop | R=0.2 DR=1000 normal | `K_max` | 32 | 7.71e-07 | 7.74e-07 | 7.05e-07 | 7.64e-07 |
| static drop | R=0.2 DR=1000 normal | `late_u` | 32 | 0.00961 | 0.00935 | 0.0274 | 0.00901 |
| static drop | R=0.2 DR=1000 normal | `late_u_resolved` | 32 | 0.00961 | 0.00935 | 0.00846 | 0.00901 |
| static drop | R=0.2 DR=1000 normal | `steps` | 32 | 321 | 321 | 321 | 321 |
| static drop | R=0.2 DR=1000 swapped | `K_growth` | 32 | 1.03 | 1.03 | 1.01 | 0.976 |
| static drop | R=0.2 DR=1000 swapped | `K_max` | 32 | 7.71e-07 | 7.74e-07 | 7.05e-07 | 7.64e-07 |
| static drop | R=0.2 DR=1000 swapped | `late_u` | 32 | 0.00961 | 0.00935 | 0.0274 | 0.00901 |
| static drop | R=0.2 DR=1000 swapped | `late_u_resolved` | 32 | 0.00961 | 0.00935 | 0.00846 | 0.00901 |
| static drop | R=0.2 DR=1000 swapped | `steps` | 32 | 321 | 321 | 321 | 321 |
| capillary wave | La=3000 DR=1 | `error` | 16 / 32 / 64 | 0.00563 / 0.00468 / 0.00154 (1.60) | 0.00559 / 0.0047 / 0.00156 (1.60) | 0.0228 / 0.0117 / 0.00399 (1.55) | 0.00921 / 0.00312 / 0.00234 (0.42) |
| capillary wave | La=3000 DR=1 | `steps` | 16 / 32 / 64 | 29 / 80 / 227 | 29 / 80 / 227 | 29 / 80 / 227 | 29 / 80 / 227 |
| capillary wave | La=3000 DR=1000 | `error` | 16 / 32 / 64 | 0.0217 / 0.00559 / 0.00241 (1.22) | 0.0218 / 0.00562 / 0.0024 (1.22) | 0.0187 / 0.0048 / 0.00149 (1.69) | 0.0246 / 0.00683 / 0.00274 (1.32) |
| capillary wave | La=3000 DR=1000 | `steps` | 16 / 32 / 64 | 29 / 80 / 227 | 29 / 80 / 227 | 29 / 80 / 227 | 29 / 80 / 227 |
| Kelvin-Helmholtz | DR=1000 We=0.5 amplitude 0.04 | `error` | 32 | 0.848 | 0.292 | 0.262 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 amplitude 0.04 | `steps` | 32 | 5.28e+03 | 3.31e+03 | 3.26e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ensemble | `grid_mean` | 32 | 0.323 | 0.34 | 0.351 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ensemble | `grid_spread` | 32 | 8.76e-05 | 2.02e-04 | 0.112 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×0.995 | `error` | 32 | 0.323 | 0.34 | 0.421 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×0.995 | `steps` | 32 | 3.01e+03 | 3.01e+03 | 3.04e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×0.9975 | `error` | 32 | 0.323 | 0.34 | 0.388 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×0.9975 | `steps` | 32 | 3.01e+03 | 3.01e+03 | 3.04e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1 | `error` | 32 | 0.323 | 0.339 | 0.309 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1 | `steps` | 32 | 3.01e+03 | 3.01e+03 | 3.04e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1.0025 | `error` | 32 | 0.323 | 0.34 | 0.321 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1.0025 | `steps` | 32 | 3.01e+03 | 3.01e+03 | 3.04e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1.005 | `error` | 32 | 0.323 | 0.339 | 0.317 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 grid ×1.005 | `steps` | 32 | 3.01e+03 | 3.01e+03 | 3.04e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 moving U=2 | `error` | 32 | 0.475 | 0.465 | 0.649 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 moving U=2 | `steps` | 32 | 4.63e+03 | 4.63e+03 | 4.59e+03 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 shift 0.333h | `error` | 32 | 0.562 | 0.558 | 0.554 |  |
| Kelvin-Helmholtz | DR=1000 We=0.5 shift 0.333h | `steps` | 32 | 2.99e+03 | 2.98e+03 | 3.04e+03 |  |
| translation | sheet DR=1000 film 1.5h | `fraction` | 32 | 1.22e-14 | 1.22e-14 | 1.22e-14 | 1.23e-14 |
| translation | sheet DR=1000 film 1.5h | `normal` | 32 | 0 [E] | 0 [E] | 0 [E] | 0 |
| translation | sheet DR=1000 film 1.5h | `tangential` | 32 | 0 [E] | 0 [E] | 0 [E] | 1 [1V] |
| translation | sheet DR=1000 resolved | `fraction` | 32 | 1.23e-14 | 1.23e-14 | 1.23e-14 | 1.21e-14 |
| translation | sheet DR=1000 resolved | `normal` | 32 | 0 [E] | 0 [E] | 0 [E] | 0 |
| translation | sheet DR=1000 resolved | `tangential` | 32 | 0 [E] | 0 [E] | 0 [E] | 0.92 [1V] |
| label symmetry | capillary wave DR=1 | `fraction` | 32 | 4.44e-16 | 4.44e-16 | 4.44e-16 | 4.44e-16 |
| label symmetry | capillary wave DR=1 | `velocity` | 32 | 4.53e-13 | 3.17e-10 | 3.27e-10 [E] | 2.98e-13 |
| label symmetry | capillary wave DR=1000 | `fraction` | 32 | 4.44e-16 | 4.44e-16 | 4.44e-16 | 4.44e-16 |
| label symmetry | capillary wave DR=1000 | `velocity` | 32 | 3.00e-13 | 2.62e-13 | 3.12e-10 [E] | 2.92e-13 |
| label symmetry | static drop DR=1 | `fraction` | 32 | 4.44e-16 | 4.44e-16 | 4.44e-16 | 4.44e-16 |
| label symmetry | static drop DR=1 | `velocity` | 32 | 4.09e-12 | 2.12e-12 | 7.32e-14 [E] | 2.31e-12 |
| label symmetry | static drop DR=1000 | `fraction` | 32 | 4.44e-16 | 4.44e-16 | 2.22e-15 | 4.44e-16 |
| label symmetry | static drop DR=1000 | `velocity` | 32 | 1.31e-12 | 1.61e-12 | 8.50e-12 [E] | 1.69e-11 |

- hydrostatic `max_u`: largest value over 50 steps of Δt = 1e-3, exact rest
- static drop `K_growth`: largest kinetic energy of the last third over that of the first third
- static drop `K_max`: largest kinetic energy (staggered phase masses)
- static drop `late_u`: largest value for t > T/2 (all present values)
- static drop `late_u_resolved`: largest value for t > T/2 on mixed faces with phase aperture > 0.05
- static drop `steps`: time steps of `simulate!` (package step controller)
- capillary wave `error`: time-L2 error of the normalised amplitude against Prosperetti over a quarter period
- capillary wave `steps`: time steps of `simulate!` (package step controller)
- Kelvin-Helmholtz `error`: normalised time-L2 error of the complex mode amplitude against linear theory
- Kelvin-Helmholtz `grid_mean`: mean error of the five grid-line members (amplitude ×1, ×(1 ± 0.0025), ×(1 ± 0.005))
- Kelvin-Helmholtz `grid_spread`: largest minus smallest error of the grid-line members
- translation `fraction`: largest error of the volume fraction against the translated layer
- translation `normal`: largest error of a normal value with phase aperture > 0 (exact 1)
- translation `tangential`: largest error of a tangential value with phase aperture > 0 (exact ∓½)
- label symmetry `fraction`: largest |C + C' − 1| of the two labellings
- label symmetry `velocity`: largest difference of the exchanged values over their largest value (Inf: a value present in one labelling only)
- `[1V]` one velocity field cannot carry the sheet's tangential jump: O(1) by construction
- `[E]` exact by construction (round-off or projection tolerance)

<!-- tier2 table: end -->

### Reading the tier-2 table

- Every method keeps the hydrostatic pool and the translated sheet at rest
  or exact to round-off, and exchanges its values with the phase labels to
  round-off; the one-velocity model cannot carry the sheet's tangential jump
  (`[1V]`).
- Static drop: CMDGFM's late currents are its resolved ones, while MM's raw
  maxima (0.016 at DR 1, 0.027 at DR 1000) are 3.5 and 3.2 times its resolved
  ones, which exclude the values with phase aperture ≤ 0.05. The kinetic
  energy of the last third stays within 0.72–1.06 times that of the first.
- Kelvin–Helmholtz: MM's grid-line members spread over 0.309–0.421 (mean
  0.351) for amplitude changes of at most 0.5 %, the noise band of the MM
  studies, while the CMDGFM members agree to `2e-4`. The larger amplitude
  costs CMDGFM-L accuracy (0.848 in 5280 steps against 0.292 for CMDGFM-C and
  0.262 for MM), and the moving frame costs MM (0.649 against 0.47).
- The tier-2 values are deterministic on one machine and Julia version, but
  the Kelvin–Helmholtz errors of MM change by a quarter under an amplitude
  change of 0.25 %: compare tier 2 only on the machine that recorded the
  baseline.

## Tier 3: drop impact

A drop of radius 0.08 falls at `V = 2` onto a pool of depth 0.3 in the unit
box (`ρ = (1000, 1)`, `μ = (0.32, 0.0064)`, `σ = 6.4` CGFM, `g = 9.81`,
interphase drag, free slip) until `t = 0.12`, with [`simulate!`](@ref) and its
step controller (CFL 0.45); members 1–5 shift drop and pool by fractions of a
cell of the `n = 32` mesh. Inviscid impact is singular, so the maxima are
compared with a loose tolerance. Raw maxima take every present value;
resolved maxima the values with phase fraction at least ½ and a positive
aperture. The gas values are classified by their own aperture `a`: slaved
(`a = 0`), small (`0 < a ≤ 0.05`, which contains the bottleneck apertures
`a ≈ √(ρ_g/ρ_l)` of the MM spikes) and regular. The anatomy columns give the
aperture, fraction and time of the value of the raw gas peak, the mean number
of slaved and small gas values per step, and the steps in which such a value
exceeds `5V`.

<!-- tier3 table: begin -->

_Generated from `benchmark/two_velocity/baseline.toml` by `run.jl --update-docs` (Julia 1.12.7, revision 189ba44f)._

| test | case | metric | n | CMDGFM-L | CMDGFM-C | MM | 1-vel |
|---|---|---|---|---|---|---|---|
| impact | V=2 member 1 | `fast_small_steps` | 32 / 64 | 0 / 0 | 0 / 0 | 68 / 138 | 0 / 0 |
| impact | V=2 member 1 | `gas_raw` | 32 / 64 | 2.97 / 5.44 | 3.39 / 5.39 | 66.4 / 90.4 | 5.34 / 8.16 |
| impact | V=2 member 1 | `gas_regular` | 32 / 64 | 2.97 / 5.44 | 3.39 / 5.39 | 34.1 / 70.1 | 5.34 / 8.16 |
| impact | V=2 member 1 | `gas_resolved` | 32 / 64 | 2.92 / 5.44 | 3.39 / 5.39 | 17.2 / 28.1 | 5.34 / 8.16 |
| impact | V=2 member 1 | `gas_slaved` | 32 / 64 | 2.59 / 3.51 | 2.46 / 3.87 | 51.8 / 74.4 | 2.09 / 3.06 |
| impact | V=2 member 1 | `gas_small` | 32 / 64 | 2.27 / 3.27 | 2.4 / 3.66 | 66.4 / 90.4 | 2.08 / 3.2 |
| impact | V=2 member 1 | `liquid_raw` | 32 / 64 | 3.46 / 4.89 | 2.89 / 3.61 | 2.51 / 2.91 | 4.92 / 7.45 |
| impact | V=2 member 1 | `liquid_resolved` | 32 / 64 | 2.39 / 3.41 | 2.46 / 3.18 | 2.51 / 2.91 | 2.54 / 3.8 |
| impact | V=2 member 1 | `min_dt` | 32 / 64 | 4.18e-04 / 9.67e-04 | 0.00338 / 5.32e-04 | 2.06e-04 / 6.78e-05 | 0.00257 / 6.75e-04 |
| impact | V=2 member 1 | `peak_aperture` | 32 / 64 | 0.497 / 0.587 | 0.924 / 0.485 | 0.0119 / 0.00389 | 1 / 1 |
| impact | V=2 member 1 | `peak_fraction` | 32 / 64 | 0.478 / 0.631 | 0.763 / 0.647 | 0.115 / 4.31e-04 | 1 / 1 |
| impact | V=2 member 1 | `peak_time` | 32 / 64 | 0.0434 / 0.0237 | 0.0126 / 0.0203 | 0.0262 / 0.0236 | 0.00257 / 0.0136 |
| impact | V=2 member 1 | `slaved_per_step` | 32 / 64 | 48.3 / 99.5 | 48.1 / 96.3 | 39.7 / 83.5 | 51.9 / 105 |
| impact | V=2 member 1 | `small_per_step` | 32 / 64 | 5.29 / 17.4 | 4.68 / 9.57 | 4.3 / 9.58 | 5.28 / 11.2 |
| impact | V=2 member 1 | `steps` | 32 / 64 | 38 / 93 | 34 / 93 | 106 / 245 | 43 / 124 |
| impact | V=2 member 2 | `fast_small_steps` | 32 | 0 | 0 | 127 | 0 |
| impact | V=2 member 2 | `gas_raw` | 32 | 3.9 | 4.84 | 47.1 | 5.75 |
| impact | V=2 member 2 | `gas_regular` | 32 | 3.9 | 4.84 | 30.1 | 5.75 |
| impact | V=2 member 2 | `gas_resolved` | 32 | 3.9 | 4.84 | 16.1 | 5.75 |
| impact | V=2 member 2 | `gas_slaved` | 32 | 2.46 | 3.02 | 47.1 | 2.14 |
| impact | V=2 member 2 | `gas_small` | 32 | 2.82 | 2.61 | 46.4 | 2.22 |
| impact | V=2 member 2 | `liquid_raw` | 32 | 3.44 | 3.69 | 2.51 | 5.74 |
| impact | V=2 member 2 | `liquid_resolved` | 32 | 2.64 | 2.57 | 2.51 | 2.67 |
| impact | V=2 member 2 | `min_dt` | 32 | 0.00261 | 0.00196 | 2.77e-04 | 0.00208 |
| impact | V=2 member 2 | `peak_aperture` | 32 | 1 | 0.823 | 0 | 1 |
| impact | V=2 member 2 | `peak_fraction` | 32 | 0.897 | 0.757 | 5.07e-05 | 1 |
| impact | V=2 member 2 | `peak_time` | 32 | 0.0181 | 0.0314 | 0.0285 | 0.0177 |
| impact | V=2 member 2 | `slaved_per_step` | 32 | 50.7 | 47.4 | 39.9 | 50 |
| impact | V=2 member 2 | `small_per_step` | 32 | 7.92 | 7.63 | 4.79 | 6.98 |
| impact | V=2 member 2 | `steps` | 32 | 38 | 38 | 156 | 45 |
| impact | V=2 member 3 | `fast_small_steps` | 32 | 0 | 0 | 141 | 0 |
| impact | V=2 member 3 | `gas_raw` | 32 | 4.17 | 4.25 | 58.1 | 5.12 |
| impact | V=2 member 3 | `gas_regular` | 32 | 4.14 | 4.25 | 38.1 | 5.12 |
| impact | V=2 member 3 | `gas_resolved` | 32 | 3.9 | 4.25 | 18 | 5.12 |
| impact | V=2 member 3 | `gas_slaved` | 32 | 4.17 | 3.77 | 58.1 | 2.12 |
| impact | V=2 member 3 | `gas_small` | 32 | 2.56 | 3.79 | 56.1 | 2.09 |
| impact | V=2 member 3 | `liquid_raw` | 32 | 2.75 | 3.29 | 2.52 | 5.12 |
| impact | V=2 member 3 | `liquid_resolved` | 32 | 2.49 | 2.44 | 2.52 | 2.27 |
| impact | V=2 member 3 | `min_dt` | 32 | 1.87e-04 | 0.00141 | 1.50e-05 | 0.00216 |
| impact | V=2 member 3 | `peak_aperture` | 32 | 0 | 0.592 | 0 | 1 |
| impact | V=2 member 3 | `peak_fraction` | 32 | 0.0903 | 0.666 | 2.58e-07 | 0.999 |
| impact | V=2 member 3 | `peak_time` | 32 | 0.0557 | 0.0301 | 0.0277 | 0.00453 |
| impact | V=2 member 3 | `slaved_per_step` | 32 | 49.2 | 50.2 | 38.8 | 48.7 |
| impact | V=2 member 3 | `small_per_step` | 32 | 9.68 | 12.2 | 3.39 | 8.43 |
| impact | V=2 member 3 | `steps` | 32 | 40 | 40 | 171 | 47 |
| impact | V=2 member 4 | `fast_small_steps` | 32 | 0 | 0 | 124 | 0 |
| impact | V=2 member 4 | `gas_raw` | 32 | 3.55 | 3.54 | 75.2 | 5.29 |
| impact | V=2 member 4 | `gas_regular` | 32 | 3.55 | 3.54 | 39.9 | 5.29 |
| impact | V=2 member 4 | `gas_resolved` | 32 | 3.55 | 3.54 | 15 | 5.29 |
| impact | V=2 member 4 | `gas_slaved` | 32 | 2.5 | 2.97 | 75.2 | 2.09 |
| impact | V=2 member 4 | `gas_small` | 32 | 2.22 | 2.58 | 62.7 | 2.07 |
| impact | V=2 member 4 | `liquid_raw` | 32 | 3.52 | 3.3 | 2.37 | 5.28 |
| impact | V=2 member 4 | `liquid_resolved` | 32 | 2.21 | 2.33 | 2.37 | 2.3 |
| impact | V=2 member 4 | `min_dt` | 32 | 0.00115 | 7.90e-05 | 1.82e-04 | 0.00242 |
| impact | V=2 member 4 | `peak_aperture` | 32 | 0.621 | 0.893 | 0 | 1 |
| impact | V=2 member 4 | `peak_fraction` | 32 | 0.757 | 0.859 | 3.69e-05 | 1 |
| impact | V=2 member 4 | `peak_time` | 32 | 0.00699 | 0.00705 | 0.028 | 0.00242 |
| impact | V=2 member 4 | `slaved_per_step` | 32 | 52.5 | 52.4 | 39.1 | 49 |
| impact | V=2 member 4 | `small_per_step` | 32 | 6.46 | 7.37 | 4.25 | 5.64 |
| impact | V=2 member 4 | `steps` | 32 | 35 | 35 | 161 | 45 |
| impact | V=2 member 5 | `fast_small_steps` | 32 | 0 | 0 | 123 | 0 |
| impact | V=2 member 5 | `gas_raw` | 32 | 3.54 | 3.54 | 83.8 | 5.76 |
| impact | V=2 member 5 | `gas_regular` | 32 | 3.54 | 3.54 | 30.4 | 5.76 |
| impact | V=2 member 5 | `gas_resolved` | 32 | 3.54 | 3.54 | 20.2 | 5.76 |
| impact | V=2 member 5 | `gas_slaved` | 32 | 2.51 | 2.84 | 83.8 | 2.11 |
| impact | V=2 member 5 | `gas_small` | 32 | 2.53 | 2.75 | 77.8 | 2.13 |
| impact | V=2 member 5 | `liquid_raw` | 32 | 3.37 | 3.11 | 2.48 | 5.55 |
| impact | V=2 member 5 | `liquid_resolved` | 32 | 2.6 | 2.84 | 2.48 | 2.76 |
| impact | V=2 member 5 | `min_dt` | 32 | 7.19e-04 | 0.00113 | 1.45e-04 | 3.27e-05 |
| impact | V=2 member 5 | `peak_aperture` | 32 | 1 | 0.847 | 0 | 1 |
| impact | V=2 member 5 | `peak_fraction` | 32 | 0.935 | 0.756 | 2.52e-06 | 1 |
| impact | V=2 member 5 | `peak_time` | 32 | 0.0193 | 0.035 | 0.0271 | 0.0191 |
| impact | V=2 member 5 | `slaved_per_step` | 32 | 51.6 | 50 | 39.9 | 51.4 |
| impact | V=2 member 5 | `small_per_step` | 32 | 8.38 | 7.54 | 4.8 | 6.89 |
| impact | V=2 member 5 | `steps` | 32 | 39 | 35 | 160 | 47 |
| impact | V=2 members 1-5 | `ensemble_gas_raw_median` | 32 | 3.55 | 3.54 | 66.4 | 5.34 |
| impact | V=2 members 1-5 | `ensemble_gas_resolved_median` | 32 | 3.55 | 3.54 | 17.2 | 5.34 |

- impact `fast_small_steps`: steps in which a slaved or small gas value exceeds 5V
- impact `gas_raw`: largest gas value over the run
- impact `gas_regular`: largest gas value with a > 0.05
- impact `gas_resolved`: largest gas value with gas fraction ≥ ½ and aperture > 0
- impact `gas_slaved`: largest gas value without aperture (a = 0)
- impact `gas_small`: largest gas value with 0 < a ≤ 0.05 (small and bottleneck values)
- impact `liquid_raw`: largest liquid value over the run
- impact `liquid_resolved`: largest liquid value with liquid fraction ≥ ½ and aperture > 0
- impact `min_dt`: smallest time step
- impact `peak_aperture`: gas aperture of the value of the raw gas peak
- impact `peak_fraction`: gas fraction of the value of the raw gas peak
- impact `peak_time`: time of the raw gas peak
- impact `slaved_per_step`: mean number of gas values without aperture per step
- impact `small_per_step`: mean number of gas values with 0 < a ≤ 0.05 per step
- impact `steps`: time steps of `simulate!`
- impact `ensemble_gas_raw_median`: median over the n = 32 members of the raw gas maximum
- impact `ensemble_gas_resolved_median`: median over the n = 32 members of the resolved gas maximum

<!-- tier3 table: end -->

### Reading the tier-3 table

- MM's raw gas maxima are 47–84 at `n = 32` (median 66.4, 33 V) and 90.4 at
  `n = 64`; in every member the peak sits on a gas value without aperture or
  with `a ≤ 0.012`, and 68–141 steps carry a slaved or small gas value faster
  than `5V`. These are the bottleneck values of the MM studies (the n = 32
  member-1 peak, 66.44 on a value of aperture 0.0119 and fraction 0.115 at
  `t = 0.0262`, reproduces their anatomy).
- MM's resolved gas maxima, 15–20 at `n = 32` and 28 at `n = 64`, are still
  four to six times CMDGFM-L's; CMDGFM's raw maxima (3.0–5.4) are within 7 %
  of its resolved ones, and no CMDGFM or one-velocity step has a fast slaved
  or small value.
- The spikes cost MM 2.6–4.6 times the steps of CMDGFM-L (smallest step
  `1.5e-5`).
