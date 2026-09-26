# First execution of the `:infer` plan

Historical note: this report records the original strict G1 stopping point.
The user subsequently asked to proceed despite the pointwise pressure bound,
and `:infer` has since been added to production. See
[`INFER_IMPLEMENTATION_PLAN.md`](../../INFER_IMPLEMENTATION_PLAN.md) and the
current KH development diagnostics. Statements below that no production
option was added describe the state *at the time of this report*.

The approved [implementation plan](../../INFER_IMPLEMENTATION_PLAN.md) stops
before production changes if its cheap operator gate G1 cannot demonstrate
first-order pressure and tangent consistency with the fixed physical \(K,D\).
That gate fails for the current face-to-face target. No `:infer` solver
option was added.

## Baseline (G0)

The existing forward development script, `--pressure-first` variant,
manufactured Poisson comparison, and one-step KH diagnostic completed.
Representative outputs are:

| Study | Coarse | Fine | Observation |
| --- | ---: | ---: | --- |
| Existing fixed-\(T\) pressure \(L_\infty\) error, \(n=16\to48\) | 2.361 | 1.998 | Same pressure error for `:divergence`, `:face`, `:cell`; \(N\) cannot fix it. |
| Existing `--pressure-first` pressure \(L_\infty\), \(n=16\to48\) | 1.177 | 1.035 | Still lacks a convincing \(L_\infty\) rate. |
| Manufactured Poisson relative velocity \(L_2\), \(n=8\to32\), CMDGFM | 3.882 | 1.207 | Recorded comparison, not an `:infer` result. |
| Manufactured Poisson relative velocity \(L_2\), \(n=8\to32\), `:face` | 0.850 | 0.286 | `:face` is better on this particular Poisson measure. |
| One-step KH relative projection change, \(n=8\), CMDGFM | 0.00502 | — | Existing baseline. |
| One-step KH relative projection change, \(n=8\), `:face` | 0.339 | — | Large geometry-related projection correction. |

The one-step mimetic energy-identity errors were about \(10^{-17}\) for the
completed baseline runs. The projection unit-test baselines passed; their
counts are recorded below.

## Cheap feasibility test (G1)

Run

```sh
julia --project=. -O0 development/infer/jump_inference_feasibility.jl
```

The script uses the production mixed-face active set, face kinetic mass
\(K_{ff}=|\omega_f|m_f^\ell m_f^g/(m_f^\ell+m_f^g)\), and the production
\(-K^{-1}D^Tp\) response. It uses the exact analytic normal of a circle
to form tangential targets, which removes normal-reconstruction error
from this gate. Pressure samples include the periodic field
\(p(x,y)=\sin(2\pi x)\); thus the failure below does not depend on an
affine pressure crossing a periodic boundary.

There is a bound that applies to **every** \(K\)-orthogonal projector,
regardless of its basis, support, or normal operator. For one pressure
sample set \(x=-K^{-1}D^Tp\), let \(y\) be its analytical tangential target
and \(z=Tx\). Orthogonality of the retained and removed parts gives

\[
 \langle z,x-z\rangle_K=0,\qquad
 \|z-x/2\|_K=\|x\|_K/2 .
\]

Thus \(z\) lies on a \(K\)-sphere centred at \(x/2\). Its distance to \(y\)
is at least

\[
 e_{\min}=
 \left|\|y-x/2\|_K-\|x\|_K/2\right|.
\]

The table divides this by \(\sqrt{\sum_f K_{ff}}\), so it is a lower bound
for the face-weighted RMS error. The face \(L_\infty\) error is at least as
large. The last column is the measured \(L_\infty\) error from a
**global** projector onto the span of all sample targets. It is a reference
candidate, not the proposed sparse method or the optimal global projector.

| \((\rho^\ell,\rho^g)\) | \(n\) | Minimum pressure RMS error of any \(T\) | Global reference pressure \(L_\infty\) |
| --- | ---: | ---: | ---: |
| \((1000,1)\) | 16 | 0.164 | 0.881 |
| \((1000,1)\) | 32 | 0.125 | 0.817 |
| \((1000,1)\) | 64 | 0.0844 | 0.703 |
| \((1000,1)\) | 96 | 0.0923 | 0.687 |
| \((1000,1)\) | 128 | 0.0683 | 0.575 |
| \((1000,1)\) | 192 | 0.0692 | 0.584 |
| \((1,1000)\) | 32 | 0.0363 | 0.287 |
| \((1,1000)\) | 96 | 0.0283 | 0.389 |
| \((1,1000)\) | 128 | 0.0299 | 0.435 |

The worst periodic sample at \(1000{:}1\) is \(p=\sin(2\pi x)\).
The pressure bound decreases by only a factor of about \(2.4\) over the
\(12\times\) refinement from \(n=16\) to \(n=192\), giving an endpoint
order about \(0.35\); it even rises slightly from \(n=128\) to \(192\).
The reverse density order has a similar plateau. The declared G1 target
was an endpoint order of at least \(0.8\) for both \(L_2\) and
\(L_\infty\). The lower bound alone fails that finite-mesh gate.
This is stronger evidence than failure of any one candidate \(E_T\).

The calculation tests the specific face-to-face tangential target in the
note, with its usual smooth extension to active faces. It does not prove
that *every conceivable* target extension or revised pressure
reconstruction is impossible. It establishes that changing \(E_N\), \(N\),
or a local patch size cannot repair the tested target with the current
\(K,D\). Because G1 fails, fitting sparse \(E_T,E_N\) and running the
production, Poisson, and wave gates would not be a valid execution of
the approved plan. A next design step would revisit the target
representation or the physical gradient/kinetic derivation together,
then repeat this cheap feasibility gate.

## Unit-test baseline

The two directly relevant test files passed when run with `using Test,
TwoPhaseAMR`:

- `test/two_phase/two_velocity/projection/mimetic_projection.jl`: 483/483
  assertions across the five testsets, including AMR and embedded geometry.
- `test/two_phase/two_velocity/projection/mimetic_projection_variants.jl`:
  141/141 assertions across five testsets, including time refinement and
  configuration validation.

The full `test/runtests.jl` command was started, then interrupted after G1
failed while it was in the unrelated single-velocity Navier--Stokes tests.
It did not report a failure before interruption. The full suite remains a
G6 regression gate if the model is redesigned and reaches production.
