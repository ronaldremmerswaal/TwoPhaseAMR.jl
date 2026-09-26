# Frozen KH response and jump-metric exploration

This is a development study, not a new production kinetic metric. The two
reproducible probes are [`frozen_kh_response.jl`](../frozen_kh_response.jl) and
[`metric_feasibility.jl`](metric_feasibility.jl); the manufactured-Poisson
face audit is [`poisson_jump_audit.jl`](../poisson_jump_audit.jl). Run each with
`julia --project=. development/infer/<name>.jl` from the repository root.

## What the cheap KH test does and does not predict

`study_predictor` builds the **same transported and forced KH predictor** for
all methods, freezes its geometry and time step, and replays only the pressure
projection. It compares each projected face velocity and jump to CMDGFM on
fraction-mixed faces. It also advects a copy of the frozen interface for one
step with each projected field and compares the first Fourier-mode speed.
No separate instability integration is needed for each candidate. The
optional `snapshot_steps` first advances a CMDGFM reference to inspect a
later geometry.

This is a screen for large *instantaneous* differences. It cannot reproduce
the feedback of subsequent transport, curvature, or accumulated phase error.
In particular, its ranking of two candidates that are already close to
CMDGFM need not agree with their final wave-amplitude ranking. The earlier
analytic frozen-wave variant (`study`) is cheaper still, but did not rank the
infer variants reliably, so the transported-predictor replay is the useful
default.

At `n=16`, `rho_l/rho_g=1000`, the first *post-transport* KH predictor, the
following values are from the current interface-normal-row implementation.
`velocity relative L2`
is the fractional face-velocity distance to CMDGFM, using interface weights
`c_f(1-c_f) A_f`. `mode budget` is the one-step difference in Fourier-mode
speed multiplied by the analytical horizon and divided by the initial wave
amplitude; it is only a screening scale, **not** a wave-error estimate.

| We | Method | Mass cutoff | Velocity relative L2 | Mode budget | Mass-deactivated faces | Energy-identity defect |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | infer PLIC | default | 5.682 | 4.382 | 16 | 1.6e-19 |
| 0.5 | infer PLIC | 1e-6 | 0.152 | 0.102 | 4 | 2.2e-16 |
| 0.5 | infer smoothed-stencil | default | 3.772 | 1.785 | 16 | 6.1e-17 |
| 0.5 | infer smoothed-stencil | 1e-6 | 0.323 | 0.135 | 4 | 7.7e-21 |
| 5 | infer PLIC | default | 5.681 | 1.025 | 16 | 1.0e-18 |
| 5 | infer PLIC | 1e-6 | 0.151 | 0.0268 | 4 | 1.3e-19 |
| 5 | infer smoothed-stencil | default | 3.772 | 0.414 | 16 | 2.2e-15 |
| 5 | infer smoothed-stencil | 1e-6 | 0.321 | 0.0335 | 4 | 5.7e-20 |

CMDGFM is the zero-difference reference, but is **not** a homogeneous
physical-kinetic-energy projection: its final velocity/correction kinetic
inner-product cosine is about `0.0154` on both frozen states. Consequently
no change of the infer normal constraint alone can produce *exactly* the
CMDGFM field while retaining the current orthogonal-projection identity.
Approximation within the requested accuracy margin is still possible.

The cheap screen correctly flags the disastrous effect of the default active
set at high density ratio and the improvement when it is relaxed. It does
not certify a nonlinear KH result. In particular, the exploratory full
`n=16`, `rho_l/rho_g=1000` amplitude errors after the normal-row correction
were approximately:

| We | CMDGFM | infer PLIC, cutoff 1e-6 | infer smoothed-stencil, cutoff 1e-6 |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.465 | 1.148 | 1.100 |
| 5 | 0.169 | 0.200 | 0.182 |

Thus We=5 is close to the 10–20% target, while We=0.5 remains far from it.
The two methods also swap fine rankings between the frozen and integrated
tests. Further choices require **mesh and time-step** KH sweeps, not only
frozen tests. None of these exploratory cutoff values changes the production
default.

The same frozen replay after 64 CMDGFM steps confirms that the cutoff effect
persists on a moving sheet: at We=0.5, PLIC-infer's relative velocity
distance is `7.00` with the default cutoff and `0.185` at `1e-6`;
smoothed-stencil gives `4.83` and `0.252`. At We=5 the corresponding pairs
are `6.86`/`0.234` and `4.61`/`0.225`. The default excludes 15 faces in
this later state, versus two at `1e-6`. The later snapshot still does not
predict which relaxed method wins the integrated We=0.5 wave, reinforcing
the distinction between a cheap screen and the full validation gate.

## What the mass cutoff actually tests

In `MimeticProjectionSolver`, the default
`active_mass_fraction=sqrt(interface.fraction_floor)` is about `1.22e-4`
for Float64. A mixed face is excluded from the jump solve if

`min(m_l,m_g)/(m_l+m_g) < active_mass_fraction`,

where `m_pi=rho_pi c_f^pi` is its staggered phase mass; the aperture-open and
embedded-geometry checks are separate. The threshold first appears in the
original local-mimetic commit `9ef8c5b` without a derivation or a KH
resolution criterion. The defensible motivation is numerical: a vanishing
phase mass makes the jump block poorly scaled. The conditioning note
`wellposedness_analysis.tex` discusses that risk, but does not select this
cutoff or show that it is needed for the present KH runs.

The criterion is **density-biased**. If `rho_l/rho_g=1000` and gas occupies
a fraction `g` of a face, the gas mass share falls below `1.22e-4` whenever
`g` is below about `0.109`. A face with around 10% gas is not geometrically
unresolved. At the configured KH start, `n=16` has 50 fraction-mixed faces:
the default mass cutoff excludes 12, while `1e-6` excludes none. After the
first transport step there are 56 mixed faces: the default excludes 16 and
`1e-6` excludes four. Those four have `(a_l,a_g)=(1,0)`, but some of the
default-excluded faces have **both** phase apertures open. After 64 CMDGFM
steps and one more predictor, `1e-6` excludes two faces, again aperture-pure.
`underresolved_constraint_rows` is a separate algebraic
stencil-support diagnostic, not a statement that the flat sheet has
insufficient physical mesh resolution.

### Density-scaled cutoff trial

The proposed alternative is
`fraction_floor * min(rho_l/rho_g, rho_g/rho_l)`. For the 1000:1 Float64 KH
case it is `1.4901161193847657e-11`. It compensates the small mass share
of a light minority phase, making the cutoff roughly comparable to the
geometric fraction cleanup scale. It is not a conditioning bound.

At the configured `n=8,16,32` KH starts, `1e-6` already retains all
fraction-mixed faces. After the first n=16 transport predictor, the
density-scaled value retains all 56 fraction-mixed faces, including four
aperture-pure faces that `1e-6` excludes. The frozen projected fields still
satisfy the energy identity to roundoff. Their velocity-distance-to-CMDGFM
values at `We=0.5/5` are `0.119/0.116` for PLIC-infer and `0.304/0.302`
for smoothed-stencil, compared with `0.152/0.151` and `0.323/0.321` at
`1e-6`. The stencil version still reports 11 algebraically underresolved
face stencils; changing the mass cutoff does not repair stencil geometry.

The full `n=16`, 1000:1 KH results are mixed:

| We | PLIC-infer with scaled cutoff | Smoothed-stencil with scaled cutoff |
| ---: | --- | ---: |
| 0.5 | At 3000 cycles, fails the `1e-7` normal-defect gate at `t=2.2793` (defect `1.12e-7`); at 5000 cycles, completes with error `1.244` | Amplitude error `1.127` |
| 5 | At 3000 cycles, fails the same gate at `t=0.0453` (defect `2.18e-7`); at 5000 cycles, completes with error `0.2013` | Amplitude error `0.1722` |

The PLIC failures occur with tiny algebraic residuals and roundoff-level
energy-identity defects. Increasing the iteration cap to 5000 resolves
both PLIC failures, but at extra cost; very small active jump masses appear
to amplify the physical normal-defect normalization. Relative to `1e-6`,
the density-scaled cutoff **worsens** both We=0.5 full-wave errors and
improves only the smoothed-stencil We=5 error. It is therefore not a clear
production replacement. The default remains unchanged.

The current production treatment of excluded mixed faces is to prescribe
`xi_f=j_f^*/dt`, making their **final velocity jump zero**. It does not set
the pressure-gradient jump to zero. This was a deliberate way to keep the
feasible velocity set homogeneous, and thereby keep the usual orthogonal
kinetic-energy identity. It is nevertheless not the fallback requested by
the user. Setting `xi_f=0` instead would keep the velocity jump unchanged,
but the aperture divergence still couples that jump to pressure. The
projection would then be onto an affine set; the missing term is pressure
work against the retained jump, so the usual homogeneous energy identity
does not generally hold. A trial implementation failed the energy-identity
tests and was reverted. This is an unresolved modeling choice, not a mere
one-line fallback fix.

## What `E_T` and `E_N` selection means in the current code

This is not literally a binary label attached to every physical face. First
the active set above determines which faces carry a jump unknown. Then the
chosen `infer_face_selection` determines the normal rows. Directions left in
the nullspace of those rows are inferred by stationarity. The rank filter
tests rows after multiplication by `K^{-1/2}` with tolerance `sqrt(eps(T))`
(`1.49e-8` for Float64); reported candidate-row counts precede that filter.

For `:paired` (aperture normal), `:smoothed` (one-ring averaged cell aperture
normal), or `:plic` (PLIC normal), each fraction-mixed cell is visited once.
It can form an x/y face pair only if both normal components exceed
`1e-3 * |q|`, and each axis has an unused incident active face. The face on
each axis whose staggered fraction is closest to one half is chosen. For a
pair with normal `n=(n_x,n_y)` and physical jump masses `K_x,K_y`, the two
columns, up to harmless nonzero scaling, are

```
E_T = ( n_y,       -n_x     )
E_N = ( K_y*n_x,    K_x*n_y )
```

They are `K`-orthogonal, and `E_N' K j` is proportional to
`n_x j_x + n_y j_y`. Thus one *combination* is inferred and one is
constrained; neither original face is intrinsically T or N. A normal row
on an unpaired face is considered only when
`|q_transverse| <= min(1,h/L)*|q|`, with `h` the maximum local cell spacing
and `L` the minimum domain extent. It also needs a nonnegligible axial
component (`> sqrt(eps(T))` after normalizing) and active transverse faces
for `c_f(1-c_f)` interpolation, unless the transverse normal component is
zero to `64*eps(T)`. If these conditions fail, no normal row is made there.
Where it is made, the row now uses the *full* interface normal, not a
Cartesian jump. The `h/L` angle gate is inherited from the former Cartesian
fallback and is a tunable selection heuristic, not a current consistency
requirement. One implementation asymmetry: `:smoothed` uses its smoothed
normal for forming pairs, but its **unpaired** face check currently uses the
unsmoothed aperture vector.

For `:stencil` (PLIC normal) and `:smoothed_stencil` (smoothed aperture
normal), the jump extension is the identity: there are no explicit paired
`E_T,E_N` columns. A CMDGFM-style face stencil must first be available.
It requires a nonzero axial normal component, the needed transverse faces
(normally two; one near a boundary or AMR-level change), and, when the
transverse/axial normal ratio exceeds one, suitable gas- and
liquid-dominated parallel inference faces. A resolved face computes CMDGFM's
`beta`: it is one for `|n_transverse/n_axial| <= 1`; otherwise, with `s`
the sum of the two opposite-phase mass fractions on those parallel faces,
`beta=clamp((1-s)/(|n_transverse/n_axial|-s),0,1)`.
`beta >= 0.5` creates an interface-normal row, while `beta < 0.5`
leaves that face's direction to stationarity. An unresolved stencil may
still get the full-normal fallback row if it passes the `h/L`/support test.
Here `E_T` is best understood as `ker(N)` and `E_N` as `K^{-1}N'` after all
rows are assembled; no exclusive facewise E-label exists. The statistic
`underresolved_constraint_rows` counts unresolved stencil encounters, **not**
necessarily unconstrained final faces.

On the first post-transport n=16, 1000:1, We=0.5 predictor with the
density-scaled cutoff, all 56 mixed faces are active. `:plic` forms 18
pairs and reports 35 candidate normal rows and 21 inferred directions;
`:smoothed` forms 18 pairs and reports 33 and 23. `:smoothed_stencil`
reports 35 and 21, together with 11 unresolved-stencil encounters. These
numbers describe algebraic dimensions, not geometric sheet resolution.

## Manufactured-Poisson jump audit

The face audit records the worst error in the **scaled gradient jump**
`xi_f=[(grad p)/rho]_f` for the 1000:1 manufactured case. The production
cutoff and current normal-row implementation were used here.

| n | Infer variant | Worst face apertures `(a_l,a_g)` | Exact `xi_f` | Solved `xi_f` | Active? |
| ---: | --- | --- | ---: | ---: | --- |
| 16 | paired aperture | `(1,0)` | 6.28303 | -0.001540 | yes |
| 16 | smoothed-stencil | `(1,0)` | 6.28303 | -0.003332 | yes |
| 32 | paired aperture | `(1,0)` | -6.15074 | 0.012869 | yes |
| 32 | smoothed-stencil | `(1,0)` | 6.16419 | -0.001507 | yes |

These maxima are **not** due to faces dropped by the mass threshold. They
are fraction-mixed but aperture-pure on the measured face. The pressure-only
response at the same solved pressure has the same opposite sign as the
reported solved jump. At *fixed pressure*, no positive **diagonal** rescaling
of `K` can reverse that sign. A different `K` could also change the solved
pressure, so this is not a proof against every diagonal metric. Moreover,
requiring a missing-phase face gradient in full mixed-face `Linf` demands
some extension/reconstruction from nearby faces where that phase exists.
The aperture divergence alone does not sample it locally.

## How much freedom is there in `K`?

For a target tangential pressure response `Y` and fixed pressure forcing
`R=-B'P`, an orthogonal projection in any symmetric positive-definite jump
metric `K` must satisfy the work relation `Y' K Y = Y' R`. This is a
**necessary**, not sufficient, condition. The metric-feasibility script
fits only these finite-sample work moments with a bounded diagonal
reweighting of the current `K`; it does not produce a convergent or physical
discrete Hodge.

For the 1000:1 circular-interface samples, the relative work-moment defect
at `n=16,32,64` was about `0.070, 0.051, 0.032` with the present `K`.
Allowing diagonal factors in `[0.5,2]` reduced it to approximately
`0.0028, 0.0028, 0.0022`. This establishes algebraic room in the **pressure
moments**. It does not establish `Linf` convergence of individual face
gradients; the Poisson audit above is the counterwarning.

Changing only the jump block `K` in production would also change what
"energy preserving" means unless it is induced by a new phase-velocity
kinetic Hodge. A viable next construction must keep the aperture divergence
fixed, make the pressure gradient its adjoint in the new Hodge, preserve
constant-velocity momentum modes, be symmetric positive-definite with
mesh-uniform bounds, and reproduce at least local linear phase-pressure
fields. An off-diagonal, short-range Hodge or a one-sided phase-gradient
reconstruction might address aperture-pure faces; a purely diagonal
pointwise change is not supported by the present audit. Such a change
requires a separate design decision before production implementation.

## Production status and validation

The only production edit in this exploration changes infer fallback rows:
an imposed row now evaluates the **complete interface-normal velocity jump**
using transverse staggered neighbors; a face-axis jump is used only when the
normal is axis-aligned to roundoff. The new focused regression checks this.
`test/two_phase/two_velocity/projection/mimetic_projection_variants.jl`
passed all six testsets (99, 12, 6, 6, 62, and 23 assertions). No production
`K` or active-set threshold was changed.

Next validation gates before adopting a new cutoff or Hodge are: (1) no
excluded normal rows on the resolved KH sheet, unless the geometric and
algebraic reason is explicitly diagnosed; (2) pressure-Poisson scaled-jump
`L2` **and** physical-face `Linf` refinement; (3) momentum and the correct
kinetic-energy identity; and (4) KH mesh and time-step refinement at density
ratios and Weber/Bond cases used in the existing integration comparison.
