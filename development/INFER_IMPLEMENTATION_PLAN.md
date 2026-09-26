# Historical plan: jump inference in MimeticProjectionSolver

This plan predates the current `joint_smoothed` and `joint_face` mimetic
variants. It is retained as research history, not as the active API.

Status: executing with the user's revised acceptance criteria. The
production `:infer` option and its alternate face-selection switch have been
added. The strict pointwise G1 pressure gate failed, but the user explicitly
asked to proceed to KH mesh and time studies. See the
[feasibility report](infer/archive/FEASIBILITY_REPORT.md).

The algorithmic construction in sections 1--3 below is the original
sample-fitting plan, not a claim that it has all been implemented. The
KH-focused branch instead uses the explicit disjoint-pair basis described
in [the implementation progress note](infer/archive/IMPLEMENTATION_PROGRESS.md).
Remaining gates and the fixed-\(K\) versus revised-Hodge decision are
recorded there.

The proposed option is `jump_interpolant = :infer` in both
`MimeticProjectionSolver(interface; ...)` and the high-level
`MimeticProjection(jump_interpolant = :infer)`. It implements the construction
in [jump_inference.tex](../../conditionering_cmdgfm/jump_inference.tex):
choose the physical jump metric \(K\), construct a tangential projection
\(T\), find tangential columns \(E_T\), fit normal columns \(E_N\), and set
\(N=E_N^T K\). This is a new option; the defaults and the existing
`:divergence`, `:face`, `:cell`, and `:commuting` methods stay as they are.

## Key objectives

The method is successful only if all four objectives below are demonstrated:

1. **Momentum conservation.** For a periodic pressure projection without
   external forcing, the total phase-mass-weighted momentum in each Cartesian
   direction is unchanged to the scaled linear-solve tolerance. With physical
   boundaries, account for the pressure impulse at the walls when comparing
   momentum.
2. **The usual energy identity.** At fixed geometry, the phase kinetic norm
   must satisfy
   \(\lVert u^*\rVert_M^2
     =\lVert u^+\rVert_M^2+\lVert u^+-u^*\rVert_M^2\)
   up to the corresponding solve residual. This is the orthogonal-projection
   identity; the pressure step need not leave kinetic energy unchanged.
   Measure it for the *total* correction, including the fixed
   \(\xi_\perp\) part. This is a claim about the pressure substep, not an
   exact energy law for the entire forced, transported KH flow.
3. **KH mesh refinement.** The KH amplitude error must decrease when the
   mesh is refined for both density ratios \(1\) and \(1000\), and both
   Weber numbers \(0.5\) and \(5\).
4. **KH time-step refinement without divergence.** On a fixed mesh and final
   time, KH runs at smaller time steps must remain finite, solver-converged,
   and comparably accurate for the same four density/Weber pairs. The
   analytical error need not decrease monotonically around a spatial-error
   floor. The failure to exclude is divergence of the solution or its error
   as \(\Delta t\) is reduced.

The accuracy goal is **within 10--20% of CMDGFM**, currently the strongest KH
accuracy reference. At every matched KH mesh and time-step setting that CMDGFM completes, compare
the normalized amplitude \(L_2\) error and the maximum-in-time amplitude
error with CMDGFM, using the same physical data and transport method.
Accept \(E_{\mathrm{infer}}\leq 1.2E_{\mathrm{CMDGFM}}+\delta\), where
\(\delta\) is an independently measured solver/quadrature uncertainty,
fixed before comparing methods. Compare manufactured Poisson pressure,
phase-gradient and componentwise scaled-jump errors with CMDGFM too.
Energy compatibility and temporal stability should improve on CMDGFM
without buying that improvement by a meaningful loss of accuracy.

## What must hold

On the active mixed-face space, \(K\) is the **existing physical** diagonal
jump mass, including staggered volume weights. \(D\) is the **existing**
pressure/jump coupling, with its production sign. Let
\(E=[E_T\ E_N]\), \(\xi=E\widehat\xi\), and \(G=E^T K E\).

For every geometry assembled, require the following algebraic properties
to solver accuracy:

1. \(E\) has independent columns and \(G>0\).
2. \(E_T^T K E_N=0\), \(N=E_N^T K\), and hence \(NE_T=0\).
3. \(T=E_T(E_T^T K E_T)^{-1}E_T^T K\), so
   \(T^2=T\), \(T^T K=KT\), and \(T K^{-1}N^T=0\).
4. The KKT blocks use \(DE\), \(G\), \(NE\), and their **actual transposes**.
   The divergence constraint and phase-gradient reconstruction remain in
   force. The multiplier may influence pressure and normal coefficients,
   but tangential stationarity gives
   \(\xi_T=-T K^{-1}D^T p\) independently of it.
5. For the physical predictor \(j^*\), remove the \(K\)-orthogonal remainder
   \(j_\perp=j^*-E G^{-1}E^T K j^*\) as a known correction
   \(\xi_\perp=j_\perp/\Delta t\). Shift **both** constraint right-hand sides
   by its \(D\) and \(N\) contributions; update the phase velocities with this
   correction separately. This matters when \(E\) changes with the interface.

The physical consistency tests are separate:
\(Nz(v)/\ell \simeq n\cdot v\),
\(Tz(v)\simeq z_T(v)\), and
\(-T K^{-1}D^T p\simeq z_{p,T}\), all at least first order on
resolved cuts. The target \(z_{p,T}\) represents
\(\tau\partial_s[p/\rho]\). These three tests must be reported separately;
the first two do not imply the third. Check both \(L_2\) and \(L_\infty\)
errors, with the normal rows divided by their interface measure \(\ell\).

## Findings from the existing code

| Existing path | What it provides | Change needed for `:infer` |
| --- | --- | --- |
| `src/two_phase/two_velocity/projection/mimetic_foundation.jl` | Production face masses, aperture coupling, pressure gradient, and incidence. | Reuse its values and signs. Check gas versus liquid aperture-normal orientation and physical-boundary contributions explicitly. |
| `src/two_phase/two_velocity/projection/mimetic_projection.jl` | Active faces, symmetric three-block KKT solve, normal rows, diagnostics, and a sliver extension. | Generalize its jump space to arbitrary sparse full-column-rank \(E\). The current sliver extension has diagonal \(E^TKE\); an inferred basis normally does not. |
| `src/two_phase/two_velocity/projection/methods.jl` and `src/two_phase/two_velocity/flow_workspace.jl` | Low- and high-level method selection. | Thread through `:infer` and its own configuration. Do not silently inherit the `:commuting` sliver reduction or `c(1-c)` weights. |
| `development/vortex_sheet_commutativity/commuting_normal_interpolant.jl` | Fast forward tests for normal, pressure, and multiplier terms; sparse patch searches. | Its \(T\) is a rectangular **face-to-cell observation**. The note's \(T\) is a **square face-to-face projection**. Keep old results as baselines; add new measurements and a new constructor rather than relabel the old \(T\). |
| `development/kh_single_step/kh_single_step_diagnostic.jl` | Production KH step split into transport, force, and projection; \(\Delta t\) factors \(1,1/2,1/4,1/8\). | Add `:infer`, reduced-space rank/conditioning, compatibility certificates, and a measure of the fixed \(j_\perp\) correction. Its present condition-number calculation assumes diagonal jump mass. |
| `development/manufactured_poisson_comparison.jl` and `examples/manufactured_poisson.jl` | Manufactured pressure/velocity, \(L_\infty\) jump decomposition, pressure and gradient errors, and a visual example. | Add `:infer`; replace the diagnostic's diagonal reduced-mass division with a solve using \(G\). Record gauge-aligned pressure and phase-gradient \(L_2,L_\infty\), and scaled-jump errors by face component. |
| `test/integration/two_velocity_instabilities.jl` and `test/integration/two_phase_waves.jl` | KH/RT and capillary mesh/time refinement helpers. | The existing refinement loops select CMDGFM even though the helpers accept a projection argument. Add named `:infer` runs and assertions; do not count a CMDGFM run as validation of `:infer`. |
| `test/integration/two_velocity_rotation.jl` | Moving circular interface and projection-correction components. | Its helper currently accepts a `MimeticProjection` value but constructs a default low-level mimetic solver, losing the requested interpolant. Forward the selected policy before adding `:infer` to this test. |
| `test/integration/two_velocity_transport.jl` and `two_velocity_vortex.jl` | Interface transport and reversal studies. | Keep them as full-suite regressions; they do not directly select the two-velocity pressure model. |

The additional focused unit-test files are
`test/two_phase/two_velocity/projection/mimetic_projection.jl` and
`mimetic_projection_variants.jl`. The integration runners are
`test/integration/runtests.jl` and `test/runtests.jl`.

## Construction and implementation sequence

### 1. Establish the reference data and feasibility

Record reproducible baselines from the existing forward-only development
script, its `--pressure-first` study, the Poisson comparison, and the KH
single-step diagnostic. Preserve the outputs for all existing options.
Existing failures are recorded, not turned into assertions about a new method.

Extend the cheap development experiment with a distinct `:infer` study.
Build its \(K\), \(D^Tp\), aperture vectors, and active set directly from
the production data. At small sizes also form a dense reference projector,
so the sparse implementation can be compared with an independent result.
Use smooth constant/affine vector samples \(Z\), smooth pressure samples
\(P\), and analytical tangential targets \(Z_T,Y_T\). Include held-out smooth
fields. Keep the old rectangular observation and its error reports available.

Before designing stencils, test whether the desired samples can fit a
\(K\)-orthogonal projection. With
\(X=[Z,-K^{-1}D^TP]\) and \(Y=[Z_T,Y_T]\), exact fitting requires
\(Y^TK(X-Y)=0\) without support restrictions. Measure its scaled residual,
then test whether a bounded local fit is possible. This was originally an
early stop gate. The user explicitly overrode it for the KH-focused trial:
the facewise pointwise target may be too strict for a method that converges
under mesh refinement. Retain the obstruction as a diagnostic, not proof
that the KH trial cannot work.

Use a circle at shifted positions, a nearly flat periodic sheet (including
the KH initial alignment), and a wavy sheet. Check both density orders
\((1000,1)\), \((1,1000)\) and equal density. Empty active-face or tangential
spaces must be reported as missing coverage, never as a zero-error success.

### 2. Construct \(E_T\), then \(E_N\), on interface patches

Start with local aperture-based tangential samples as candidate columns of
\(E_T\). Fit their coefficients on bounded face patches against the constant
and affine vector samples and the **production** pressure responses. The
algebraic way to enforce an exact sample fit is to keep the columns
\(K\)-orthogonal to the sampled residuals \(X-Y\) while retaining the span
of \(Y\); for the accepted first-order fit use a scaled \(O(h)\) tolerance.
Use rank-revealing solves and a bound on coefficient amplification. Form
\(T\) from the accepted basis through a solve with
\(G_T=E_T^T K E_T\), not by storing a dense square matrix in production.
Measure the full assembled operator: acceptable individual patches do not
prove an acceptable global projection.

Next fit each normal column in \(\ker T\) to the integrated aperture-normal
moments, with constants exact where feasible and smooth fields first order:
\(E_T^T K E_N=0\) and \(E_N^T K Z\simeq B_N\). The candidate
\(E_N=(I-T)K^{-1}N_a^T\) is worth trying first, but its lost moments
\(N_aTZ\) must be measured. Fit directly if this candidate fails.
Set \(N=E_N^T K\); drop only genuinely dependent rows after checking that
their physical normal moments remain represented.

The initial candidate uses a bounded patch radius. Grow a patch only when
the declared local fit fails; log every growth and failure. A local fit may
prove impossible even though the unrestricted compatibility check passes.
Do not present an uncertified local row as a successful `:infer` row.
Initial default behavior is to report such a geometry as unsupported.
An optional whole-component `:cell` fallback could later be offered with
its own label and counters, but it must preserve the KKT pairing and cannot
count toward the `:infer` accuracy gates. Mixing fallback rows into \(N\)
while retaining an incompatible \(E_T\) is not allowed.

### 3. Add the production option

Add `:infer` to the low-level solver and high-level projection policy, with
separate `infer_*` controls for patch size, rank/accuracy certificates, and
amplification. Keep the existing `rank_tolerance` and active-mass rules.
Document that `:infer` uses aperture geometry, not `c(1-c)`; it does not
inherit `:commuting_sliver_epsilon`. Retain current defaults for other
methods. Configuration tests must check both constructors and forwarding
through `FlowSimulation`.

Give the assembly one canonical active-face vector, sparse \(E\), sparse
\(N\), reduced \(G=E^T K E\), and \(H=NE\). Use \(DE\) for pressure coupling
and the exact transposes in the KKT matrix. Keep `workspace.jump` as the
represented pressure correction \(E\widehat\xi\), and expose the fixed
\(\xi_\perp\) separately. The existing `active_faces` field doubles as
reduced-unknown labels for slivers; this assumption must be removed for
`:infer` because a basis column need not identify one face.
Keep the existing treatment of inactive face jumps and physical boundaries,
with their right-hand-side shifts included before solving.

The current block preconditioner assumes a diagonal reduced jump metric.
For `:infer`, use the actual sparse SPD \(G\) in its jump block and a
positive sparse approximation of \(H G^{-1}H^T\) for the multiplier block
(initially \(H\operatorname{diag}(G)^{-1}H^T\), with the current
preconditioner-only shift if needed). Test iteration growth. Avoid explicitly
inverting \(G_T\), \(G\), or forming a dense \(T\) in the production path.
Update physical residual scaling and
`MimeticProjectionStats`: number of face unknowns, reduced tangent/normal
unknowns, rank, maximum patch radius, failed certificates, normal/tangent/
pressure fit residuals, \(j_\perp\) norm, and the existing
momentum/energy/normal defects must be distinguishable. Do not reuse
`:commuting` counters to suggest this new certificate has passed.

## Validation gates

Each gate produces a small table of actual values and a pass/fail decision.
On failure, stop before the next, more expensive gate; record which equation
failed and which geometries did so. Numerical thresholds are set before
looking at the wave results.

| Gate | Cases and checks | Pass condition |
| --- | --- | --- |
| G0: baseline | Existing development forward/`--pressure-first`, Poisson comparison, KH single step; existing unit tests. | Reproducible outputs and any existing failure identified. This gate does **not** demand that older methods pass new criteria. |
| G1: cheap operator study | Uniform \(h\) series on shifted circles, flat and wavy periodic interfaces; densities above. Compare sparse builder with small dense reference. Record coverage, \(\operatorname{nnz}(E_T,E_N,N,G)\), build time, patch radius, condition estimates. | Algebraic identities hold, normal consistency is measured, and tangent/pressure \(L_2,L_\infty\) trends and the fixed-\(K\) compatibility obstruction are reported separately. Pointwise pressure-gradient failure is diagnostic, not an automatic KH stop. No hidden zero-coverage rows. |
| G2: assembly, momentum and energy | Static uniform circle/plane, unequal densities, periodic and physical boundaries; then a fixed mixed-level mesh. Inspect \(E,G,H\) and the full KKT matrix. Use predictors both inside and outside \(\operatorname{range}(E)\). | Full-column-rank \(E\), independent accepted normal rows, positive \(G\), symmetric KKT, correct \(D\xi_\perp,N\xi_\perp\) shifts, and projected normal/divergence defects at requested solve tolerance. In periodic no-force cases, phase-mass-weighted momentum is conserved to the scaled tolerance. The total-correction Pythagorean kinetic identity meets the same scaled tolerance and tightens when solver tolerance tightens; with walls, include their pressure impulse. No dense \(T\) in the solve. |
| G3: moving interface and KH single step | Extend the existing KH diagnostic for \(\Delta t\) factors \(1,1/2,1/4,1/8\), density ratios \(1,1000\), \(We=0.5,5\), with CMDGFM and other mimetic controls. Compare raw initial data and data projected once onto the initial admissible space. Translate a planar cut across face/sliver positions. | Report rank, selector changes, normal defect, pressure impulse, and sheet circulation/enstrophy after transport and projection. A nonzero first projection of the raw initial data is not by itself temporal divergence; judge boundedness of the full wave under step refinement separately. Do not hide large local impulses behind a converged linear solve. |
| G4: manufactured Poisson | Use the existing circular manufactured problem at density ratios \(1000\) and \(1/1000\); resolutions \(8,16,32,64\). Include `:cmdgfm`, `:face`, and `:commuting` references. Run both the development summary and the visual example. | Report gauge-aligned pressure, each phase gradient, and each Cartesian component of the scaled jump in \(L_2,L_\infty\), with active-face maxima separate from all-face values. Require finite errors and convergence in physically weighted \(L_2\) where the solution is smooth; a nonconvergent facewise \(L_\infty\) alone does not reject the KH trial. Compare all measures with CMDGFM and identify any common-gradient accuracy floor without crediting \(N\) or \(T\) for curing it. |
| G5: full wave studies | Explicitly select `MimeticProjection(jump_interpolant=:infer)` in capillary and KH/RT helpers. KH: density ratios \(1,1000\) crossed with \(We=0.5,5\), both interface transports, meshes \(8,16,32\), and time factors \(1,1/2,1/4,1/8\) on the middle mesh. Capillary: density ratios \(1,1000\), both transports, same mesh/time series. Then include the existing RT and combined cases. Run matched CMDGFM controls. | Every run finishes with finite diagnostics, controlled momentum/energy/constraint defects, and decreasing mesh error. Under time-step refinement the KH solution and amplitude error remain bounded, without the divergence seen in older KKT methods; monotonic decrease of analytical error is not required. Compare full wave histories and note changes of interface rank/face selection. At matched settings where CMDGFM completes, KH and capillary amplitude errors should remain within 20% of CMDGFM, allowing independently measured solver/quadrature uncertainty \(\delta\). Record a CMDGFM failure explicitly; it is not a free accuracy pass for `:infer`. Do not weaken current CMDGFM assertions. |
| G6: topology, performance, regression | Repeat on the mixed-level and adaptive interface cases used by the wave/instability suites; run targeted projection, configuration, example smoke, and full unit/integration suites; measure sparse storage/build/apply cost at \(16,32,64,128\) after Julia warm-up. | No silent missing interface modes, identities and accuracy survive supported coarse/fine and topology changes, existing methods regress no further, and construction/apply cost grows in proportion to interface size for bounded patches. If a global \(G_T\) solve or stencil overlap defeats that scaling, report it and choose a measured patch/block strategy before release. |

Estimate \(\delta\) by repeating representative solves at tighter linear tolerance
and finer time sampling; record the resulting change before judging accuracy
comparisons. Do not choose an allowance after inspecting whether `:infer`
beats CMDGFM.

For temporal studies, retain the analytical dispersion/amplitude comparison
already in the integration tests, but use a finer-time numerical reference
to distinguish temporal convergence from a spatial error floor. State the
actual step sizes if the existing stability controller changes them, and use
the same end time and interface transport across methods. Report all
\(L_2,L_\infty\), rank, fallback/unsupported, and cost data by resolution and
method, not just a pass flag. Circulation and sheet enstrophy from the
single-step diagnostic are informative observables, not additional
conservation claims for this pressure projection.

## Decisions if a gate fails

- If G1 fails because \(Y^TK(X-Y)\) does not tend to zero, changing \(N\)
  cannot repair the pressure/tangent incompatibility. Revisit the target
  sample representation or the physical gradient/\(K,D\) derivation in a
  separate proposal. Do not tune \(K\) merely to make a fit pass.
- If unrestricted fitting works but local fitting fails, report required
  patch growth, conditioning and cost. A nonlocal construction is a possible
  research result, not a silently accepted local method.
- If G2 passes but G3 fails, examine how the **subspace projector** changes
  with transported geometry, including the fixed \(j_\perp\) correction.
  A stable matrix at each fixed geometry does not imply temporal stability.
- If G3 passes but G4 or G5 fails, keep separate reports for inferred jump,
  normal constraint, common pressure gradient, phase velocities, transport
  and forcing. Do not attribute every error to \(N\).

On approval, execute the stages in order. G1 must pass before adding the
production branch; any failed gate pauses progression and returns the data
for a model-level decision. A change to the physical kinetic metric, pressure
gradient, or declared accuracy target requires a separate design decision.
