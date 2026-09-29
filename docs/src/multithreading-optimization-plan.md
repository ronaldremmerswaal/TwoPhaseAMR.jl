# Phased optimization plan for shared-memory parallelism

Prepared 29 September 2026 from repository revision
`157448c59ad564f28a7130778a5ce4c94eacfe64` and the accompanying
[literature review](multithreading-literature.md).

Source-code links below point to that reviewed Git revision so the findings
remain traceable as implementation work proceeds.

**Status: research and implementation plan only.** No numerical implementation,
toy package, benchmark harness, or optimization is introduced by this document.
All performance gates below await measurement. Hardware inspection and source
review are the evidence available now.

The user clarified that the toy transports a **cell-centred field with staggered
Lax–Wendroff (LW) fluxes**, and that a moving mesh means the existing
**criterion-driven tree refinement/coarsening**, not physical cell motion.
The pressure block and geometric multigrid remain in place. The toy's storage
and threading design are independent of the main package.

The recommended order is to establish a credible machine/measurement baseline,
select an approach using the tiny LW package, and then extend it through the
eight requested stages. Preserve numerical contracts at every transition.
Optimize complete timesteps and time to an accurate solution; a faster isolated
loop is evidence only for that loop.

## Current package and machine: facts that affect the plan

| Area | Observed state | Consequence |
|:--|:--|:--|
| Machine | AMD Ryzen 9 5950X, 16 cores, 32 logical CPUs, one socket/OS NUMA node, two L3 domains totaling 64 MiB; approximately 31.3 GiB RAM | Test physical cores first, then SMT; compare placement within/across L3 domains. Do not assume a two-socket NUMA system. |
| Runtime | Direct installed binary reports Julia 1.12.7; `Project.toml` permits Julia 1.10 onward | Pin one runtime for performance comparisons; retain a compatibility test on the minimum supported version. |
| Threading | [src/threading.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/threading.jl) uses Polyester, a 1,024-item threshold, four target batches per default-pool thread, block-column tiling, and dynamic Julia tasks for tree work | Existing choices are baselines to test. Equal iteration counts and more batches do not by themselves establish good balancing of mixed-phase work. |
| Worker selection | Installed Polyester is 0.7.19; its macro defaults to core-based selection when `per` is unspecified; package wrappers do not specify `per` | A launch with 32 Julia threads is not proof that a dense kernel used 32 workers. Verify actual participation and explicitly test `per=thread` in later experiments. |
| Mesh | [BlockTreeMesh](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/mesh/topology/mesh.jl) is a 2-D block quadtree with owned face segments and a topology-cache generation | Reuse compact numerical schedules. Any new cached schedule must respect the existing generation/lifecycle. |
| Scalar transport | [cell_advection.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/operators/advection/cell_advection.jl) implements first-order upwind/forward Euler; its face-flux/gather update is already parallel | It is not a cell-centred LW implementation. A main-package LW comparison requires a separately validated numerical addition. |
| Interface transport | [vof_fluxes.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/indicator/phase/vof_fluxes.jl) performs directional WY transport with reconstruction and compatible transported fields | Preserve frozen sweep data and shared phase fluxes. Unlimited LW is not a replacement for VOF boundedness. |
| Staggered transport | [staggered_schemes.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/operators/advection/staggered_schemes.jl) includes `LW()`, with `MC()` as its default AMR-interface scheme | Record bulk and interface schemes separately; do not describe a hybrid AMR result as unrestricted LW everywhere. |
| AMR | [adapt.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/mesh/adaptivity/adapt.jl) stages fields, edits/rebalances topology, rebuilds schedules, and transfers registered fields | This is a dependency pipeline, not one loop to thread wholesale. Preserve field identity and retarget workspaces. |
| Poisson/GMG | [cycles.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/solvers/multigrid/cycles.jl) has damped Jacobi, threaded row actions, and a direct bottom solve; sparse transfers and several reductions remain candidates | Profile level-by-level application, transfer, reductions, setup, and refresh. |
| Hierarchy extent | [geometric_preconditioner.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/solvers/multigrid/geometric_preconditioner.jl) prunes positive tree levels and stops at the permanent base grid | A large level-zero grid can become the direct bottom problem. Check hierarchy sizes before timing GMG. |
| CMDGFM | [cmdgfm_projection.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/two_phase/two_velocity/projection/cmdgfm_projection.jl) uses BiCGSTAB and a lower block preconditioner: pressure GMG, jump ILU | Profile assembly, full-system multiplication, jump factors/triangular solves, and gauge mapping separately. |
| Cache validity | CMDGFM checks block keys/geometry; pressure coefficients refresh; jump-factor reuse checks numeric values as well as structure | Preserve these checks. The older cache-reuse description in `navier-stokes.md` is less strict than the implementation and should be reconciled during solver work. |
| Benchmarks | [common.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/benchmark/common.jl) uses a 16×16 base mesh and 8×8 blocks: 256 uniform cells, 448 for its one refined root | Useful smoke/allocation fixtures, but below many threading cutoffs. Add a scaling ladder rather than using these as 16-core evidence. |
| Tests | [threading.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/test/threading.jl) checks scheduling and conservation in its current process; integration tests already cover uniform/mixed/adaptive vortices | Add comparisons across fresh processes with different thread counts, while retaining numerical integration studies. |

These are inspected code paths, not measured bottleneck rankings. For example,
serial `divergence!`, CFL maxima, sparse transfers, and CMDGFM loops are
**profiling candidates**; none is declared dominant without timings.

## Common measurement and acceptance contract

### Define scaling before selecting an implementation

For fixed workload `W`, let `T_p(W)` be warmed elapsed wall time using `p`
active Julia compute workers. Define:

```math
S_p = T_1/T_p, \qquad E_p = S_p/p.
```

Use the same mesh, initial state, final time or step sequence, spatial scheme,
precision, CFL, solver tolerances, diagnostics, and output policy for a scaling
curve. Keep the same algorithm/backend configuration across its thread counts,
apart from declared thread-count-dependent work partitioning. Plot different
algorithms separately. Changing mesh size with `p` is weak scaling.

Record two baselines: the candidate's own one-worker time for scaling, and the
fastest validated serial implementation at the same numerical settings for
absolute benefit. Also compare the main package against the recorded starting
revision. A slow serial baseline must not manufacture impressive efficiency.
When a numerical method or preconditioner changes, additionally compare time
to the same error/tolerance; separate that algorithmic improvement from
parallel speedup.

**Primary target:** `S_16 ≥ 8`, equivalently `E_16 ≥ 0.50`, on each stage's
predeclared large production-relevant fixtures. Also publish `E_2`, `E_4`, and
`E_8`, targeting at least 50% on the same fixtures. Small workloads have a
serial-fallback gate instead. This target is a project decision informed by
[Afivo's roughly 10–12× speedup using up to 26 cores](https://homepages.cwi.nl/~ebert/2018-CPC-afivo-Jannis.pdf),
not a guarantee from that paper.

**SMT target:** measure 24 and 32 workers in addition to 16, with physical-core
placement first. Report `S_32`, `E_32=S_32/32`, and `T_16/T_32`. The literal
50%-at-32-threads target is `S_32 ≥ 16`; keep it as an explicit stretch target.
SMT shares core resources, so it must not be equated to 32 physical cores.
Also report `S_32/16` as physical-core-normalized efficiency, labelled clearly.
Do not substitute it for `E_32`. Recommend 32 threads only if its runtime
improvement is repeatable; otherwise recommend the best measured count.

### Target met, explained shortfall, or failure

Every stage receives one of three performance outcomes:

1. **Target met:** numerical and measurement gates pass and the speedup target
   is met on all declared mandatory large fixtures.
2. **Explained shortfall:** numerical gates pass, the target is missed, but a
   measured bandwidth/latency ceiling or quantified serial dependency explains
   the limit. Record the miss and propose a revised goal explicitly before
   treating that stage as accepted. Do not silently redefine 50% efficiency.
3. **Failure:** invalid comparisons, unexplained performance loss, or a
   correctness regression. Do not propagate that design to the next stage.

Use a traffic model `Q` and measured bandwidth `B_p` to estimate `Q/B_p`; for
unchanged memory traffic in a bandwidth-dominated kernel, `B_p/B_1` estimates
the bandwidth scaling ceiling. STREAM alone is insufficient for irregular
AMR: also measure a gather/stencil control with representative indices and
working-set size. A proposed diagnostic is reaching at least 80% of a credible
kernel-specific throughput ceiling, accompanied by low scheduling overhead.
The 80% threshold is a project proposal, not a published universal rule.
See [Roofline](https://aiichironakano.github.io/cs596/Williams-Roofline-CACM09.pdf)
and [STREAM's traffic conventions](https://www.cs.virginia.edu/stream/ref.html).

As an additional diagnostic, under ideal Amdahl scaling,
`S_p = 1/(f + (1-f)/p)`. Reaching `S_16=8` requires `f ≤ 1/15 ≈ 6.7%` of the
one-thread time to be serial; overhead and bandwidth leave less room.
This is a derived necessary bound under that model, not a measured serial
fraction. Apply it to adaptation and coarse solves before investing in tiny
kernels.

### Reproducible benchmark protocol

1. Record commit, Julia binary/version, resolved dependencies, CPU topology,
   affinity mask, RAM, OS, governor/boost state, default/interactive/GC worker
   counts, BLAS configuration, scheduler, tile size, and active worker count.
   The inspected CPU IDs 0–15 are distinct physical cores and 16–31 their SMT
   siblings; rediscover this mapping rather than hardcoding it.
2. Use fresh processes at `p = 1, 2, 4, 8, 16, 24, 32`; add 3/7 workers for
   correctness/partition stress. Set BLAS to one thread initially and disable
   unrelated output. Test pinned and ordinary placement, with both compact
   and spread L3 placement at intermediate counts. Affinity is a harness
   option, not a library side effect. [ThreadPinning](https://carstenbauer.github.io/ThreadPinning.jl/stable/refs/api_pinning/)
   is one suitable measurement tool.
3. Compile and warm every timed path, including regrids and new solver
   branches. Report startup/first-call latency separately. Time complete
   synchronized work; task submission is not completion.
4. For mutating microbenchmarks, restore state outside timing and use
   `evals=1`. For long runs, restore once per independent run and time the
   complete prescribed trajectory. Never repeatedly solve an already solved
   RHS or advect an increasingly smeared field by accident. Match warm-start
   policy at every count. Follow [BenchmarkTools' setup rules](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/).
5. Use at least five independent process-level repetitions for reported
   finalists, with enough work per sample to dominate timer/launch noise
   (initial target: 0.2–2 seconds). Randomize/interleave counts across repeats.
   Publish medians, dispersion, and a 95% interval for speedup estimated from
   independent runs. If the interval straddles a gate, collect more data or
   report inconclusive. Do not count many correlated inner samples as many
   independent experiments.
6. Retain GC in application timings. Record allocations and GC time separately;
   diagnose allocation scaling with cells, workers, levels, and regrids. A
   GC-disabled kernel experiment is supplementary, not the acceptance result.
7. Collect profiles at 1 and 16 workers, and at 32 if SMT is useful. Measure
   wall-time fractions, bytes/cell, updates/s, barriers, worker busy/idle time,
   and coarse-level sizes. Use CPU/cache counters where accessible; if hardware
   counters are unavailable, label bandwidth models as estimates.
8. Publish time for kernel, complete transport/projection step, and complete
   simulation separately. Include velocity updates, CFL, reconstruction,
   adaptation, coefficient refresh, and diagnostics in the appropriate totals.
   Isolate visualization/I/O in an additional production configuration.

Start with cache-resident, cache-transition, and DRAM-resident problems.
For the toy use square sizes 128, 256, 512, 1024, 2048, and 4096, plus a
rectangular/non-divisible shape for correctness. Six Float64 arrays alone
occupy `48N²` bytes, about 768 MiB at 4096²; determine the actual full working
set, including halos and temporaries. For tree/solver cases, select sizes by
measured memory consumption, leaving substantial headroom and avoiding swap.
Freeze mandatory cases before comparing candidate timings; do not select only
the sizes on which a candidate wins.

Raw results belong in a machine-local results directory or an attached research
artifact, following [benchmark/README.md](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/benchmark/README.md). Commit
the runner, configurations, plotting script, and decision record later; do not
present machine timings as portable CI assertions.

### Numerical and software gates shared by all stages

- Preserve conservation through one signed flux per physical face segment, or
  prove equivalent cancellation for duplicate evaluation of a toy flux. Measure
  volume-weighted mass errors and boundary balance. Keep mesh transfer budgets
  separate from advection budgets.
- Compare against an independent serial reference on small grids, then compare
  thread counts on fixed trajectories. Require bitwise equality only for
  operations with deliberately fixed arithmetic order. For reductions, define
  a fixed chunk tree if reproducibility is needed; otherwise use explicit,
  problem-scaled tolerances and report variation.
- Before tuning, record existing test thresholds, serial errors and convergence
  rates. Preserve all established thresholds. An initial additional acceptance
  budget is no more than 5% increase in non-roundoff numerical error and no more
  than 0.1 decrease in observed order over three resolutions. Roundoff-level
  results need absolute bounds, not relative ratios. Algorithm changes require
  a separate numerical review; a threading change should usually do much better.
- On pure rescheduling changes, compare field differences using a budget based
  on unit roundoff, step count, accumulated absolute flux, and reduction depth;
  freeze that budget from the serial analysis before inspecting parallel results.
  Never increase a tolerance just to make a race disappear.
- Exercise repeated runs, odd sizes, seams, very few blocks, unequal chunks,
  and empty phase/interface sets. Audit ownership of scratch and callbacks.
  Independent workspaces must be usable concurrently; a mutable workspace need
  not support simultaneous calls on itself, but this limitation must be stated.
- No concurrent `push!`, resizing, dictionary mutation, lazy cache construction,
  or packed-bit writes without a designed ownership scheme. Distinct `BitVector`
  indices can share a storage word. Build/finalize shared schedules before
  publishing them to workers; task migration must not invalidate scratch.
- Maintain type-stable inner loops and contiguous traversal of Julia's first
  array index. Do not trade correctness for `@fastmath` or unproved `@simd`
  independence. Time the serial kernel after every layout change.
- Initial performance-regression gate: no more than 5% one-worker slowdown on
  mandatory cases, beyond noise, unless an explicit documented tradeoff is
  accepted. Tiny-work fallback should stay within that budget. Hot-loop numerical
  buffers must not allocate with cell count after warm-up; bounded scheduler
  allocations are recorded rather than automatically disqualifying Julia tasks.

## Stage 1 — Research, baseline design, and measurement readiness

**Purpose.** Translate the literature into testable choices and establish what
can reasonably scale on this machine.

The [literature review](multithreading-literature.md) is the research deliverable
for the current planning stage. Its principal conclusions are: output ownership
before parallel execution; block/tile locality before a scheduler choice;
explicit reductions; separate topology changes from numerical execution; and
time-to-solution evaluation of multigrid/block preconditioners.

**Implementation-stage work.** Inventory the hot paths with profiles and
dependency diagrams; capture the starting-revision baselines; implement the
common harness and results schema. Measure launch/barrier overhead versus
tile size, STREAM-like bandwidth, an LW-like stencil control, and a sparse
gather control. Audit runtime/native-library worker counts and Polyester's
core/thread selection. Run the existing numerical suites before changing code.

| Gate | Evidence needed | If it fails |
|:--|:--|:--|
| G1.1 Research | Linked primary sources, relevant quantitative evidence with hardware/problem context, explicit limits of transfer | Resolve unsupported claims before using them as design premises. |
| G1.2 Baseline | Reproducible one-worker and multiworker timings with valid workloads, errors, allocations, and active-worker evidence | Fix the harness or environment; do not tune against invalid measurements. |
| G1.3 Feasibility | Bandwidth/overhead curves, initial serial-fraction estimates, declared sizes and numerical tolerances | Revise experiments, not numerical accuracy, before proceeding. |

G1.1 is addressed by this research. G1.2 and G1.3 are future work; source
inspection alone does not pass them.

## Stage 2 — Independent uniform periodic LW toy package

**Purpose.** Establish the practical strong-scaling envelope of a conservative
cell-centred transport step without tree, VOF, or main-package abstractions.

### Numerical scope

Solve `∂q/∂t + ∂(u q)/∂x + ∂(v q)/∂y = 0` on `[0,1)²`, storing cell averages
and prescribed face-normal velocities. Use Float64 initially, periodic faces,
and the requested LW flux. For constant speed `a`, a one-dimensional substep
of duration `τ` uses

```math
F_{i+1/2}=a\frac{q_i+q_{i+1}}{2}
 -\frac{a^2\tau}{2\Delta x}(q_{i+1}-q_i),\qquad
q_i^{new}=q_i-\frac{\tau}{\Delta x}(F_{i+1/2}-F_{i-1/2}).
```

Use a documented **Strang composition** of directional conservative LW
substeps, `x(Δt/2), y(Δt), x(Δt/2)`, for the initial 2-D implementation.
It supplies a small, unambiguous numerical reference. Its constant-coefficient
Fourier factor is the product of the three directional LW factors; require
each directional Courant number to be at most one, and use a conservative
starting CFL of 0.4. Do not substitute an unsplit sum of 1-D LW increments:
that omits cross-direction contributions.

For the spatially varying prescribed-flow extension, derive and validate the
conservative variable-coefficient predictor. One candidate freezes all carrier
coefficients at the global step midpoint and uses, in each directional substep,
`q*_(i+1/2)=(q_i+q_(i+1))/2 - τ[(a q)_(i+1)-(a q)_i]/(2Δx)`, with cell speeds
formed consistently from staggered speeds and `F=a_face q*`. This reduces to
the stated LW flux for constant `a`. Require manufactured-solution and
refinement checks before including it in accepted results. A constant-speed CFL
argument is insufficient for arbitrary variable coefficients.

A discretely divergence-free velocity can be generated from a periodic
vertex streamfunction using face differences. Directional conservative
splitting need not preserve a constant scalar exactly for a spatially varying
divergence-free field: quantify the splitting error and verify its convergence.
Require roundoff-level constant preservation for constant carriers. If exact
free-stream preservation for the variable carrier is made a requirement,
derive an appropriate unsplit/transverse LW variant as a separately validated
candidate, rather than adding an unexplained correction.

Use smooth periodic sine/cosine profiles and oblique constant translation as
the primary accuracy/scaling cases, plus zero/negative velocities, time reversal,
and a smooth deforming flow. Initialize analytic **cell averages**, not point
samples when evaluating convergence. Include a discontinuous profile only as
an oscillation characterization. Unlimited LW is dispersive and does not
provide a general positivity or maximum-principle guarantee; do not clip it to
pass a VOF-style bounds test. [Clawpack's LW demonstration](https://www.clawpack.org/gallery/_static/apps/notebooks/classic/advection_1d/advection_1d.html)
illustrates this distinction.

### Package and experiment design

Future suggested location: `research/UniformLWAdvection/`, with its own
`Project.toml`, small `src/`, `test/runtests.jl`, benchmark driver, and README.
No dependency on `TwoPhaseAMR`, RegionTrees, or its threading macros. Start with
ordinary dense arrays and the Julia standard library; keep BenchmarkTools,
pinning tools, and optional backend candidates in a benchmark environment.
Do not build a general CFD framework. A state/workspace, velocity setter,
`step!`, and `simulate!` are sufficient conceptual interfaces.

Compare these bounded alternatives in order:

| Candidate | What it tests | Main cost/risk |
|:--|:--|:--|
| Optimized serial dense loops | Best scalar baseline, loop order, bounds checks, SIMD eligibility | A poor baseline contaminates every speedup claim. |
| Julia `@threads :dynamic` over spatial tiles | Standard composable scheduler with enough work per launch | Launch/barrier overhead and tile granularity. |
| Chunk-owned `@spawn` tasks | Explicit scratch ownership and coarse composition | Task creation; avoid one task per cell. |
| Polyester batches | Persistent-worker overhead and worker selection | Backend-specific behavior, nesting, and SMT policy. |
| Persistent tasks spanning complete timesteps | Amortization across multiple LW sweeps | More complex barriers, lifecycle, cancellation, and exception handling; attempt only if launches dominate. |

For the first two viable backends compare stored face fluxes plus gather against
output-owned recomputation of cheap fluxes. Recompute a face with identical
orientation, inputs, and arithmetic on both sides; verify conservation.
Keep old and new state separate through each substep. Try contiguous column
slabs and 2-D tiles, halo copies versus explicit seam loops, and a small
predeclared tile-size sweep. Preserve the contiguous first index even for
y-direction stencils. Count the extra buffers, memory passes, and barriers.
SIMD, layout, and loop fusion are independent experimental factors.

Keep the comparison bounded: reject clearly inferior backends after the first
representative large size; run the full matrix only for finalists. Pinning,
chunk size, and 16/32-worker policies must not change the mathematical method.
Keep prescribed velocity arrays present in all primary comparisons; separately
report the cost of refreshing a time-dependent field so repeated trigonometric
evaluation does not masquerade as stencil scalability.

| Gate | Required evidence |
|:--|:--|
| G2.1 Serial numerics | Conservation, periodic seams, constant-state tests, analytic translation; observed smooth error order at least 1.8 over the finest two of three suitable resolutions; temporal and spatial errors separated where necessary. |
| G2.2 Thread correctness | Serial agreement at 1/2/3/4/7/8/16/32 workers, repeated executions, uneven tiles, no output aliasing or unsafe shared scratch. |
| G2.3 Scaling | Complete-step `S_16 ≥ 8` on predeclared large cache/DRAM cases, or a formally recorded explained shortfall; report kernel and full-run curves and all SMT metrics. |
| G2.4 Simplicity/cost | No cell-count-dependent allocations in warmed steps, memory model verified, serial/small-work regression budget met, backend choice supported across at least two sizes. |

**Exit artifact:** a tiny usable package, reproducible raw results/plots,
correctness report, chosen backend and tile/fusion policy, measured crossover
sizes, and a record of rejected alternatives. If the dense stencil already
hits bandwidth limits below 8×, stop scheduler proliferation and decide whether
traffic reduction or an explicitly revised target is justified before Stage 3.

## Stage 3 — Uniform cell-centred advection in the main package

**Purpose.** Transfer the demonstrated execution ideas while isolating abstraction
and geometric-interface costs on a fixed uniform mesh.

Establish three workloads: (A) the existing plain `CellField` upwind operator,
(B) a plain scalar LW operator using the toy's mathematical method, and
(C) WY interface transport with compatible centred phase scalars in vortex
reversal. A and C measure threading of existing numerics. B is a small,
separately reviewed numerical extension because the current centred API is
upwind. Avoid coupling that addition to the threading refactor. Compare B to
the toy only under identical numerics; do not compare a VOF step to a scalar
stencil as if they had equal work.

Construct both a single large block and many fixed-size uniform leaf blocks
covering the same grid. Use the same cell count, precision, prescribed face
velocities, CFL/time sequence, and boundary conditions in matched comparisons.
Profile coordinate/face lookup, halo updates, face fluxes, gather/update, CFL,
PLIC/LHF work, field copies, and scalar diagnostics separately.

For vortex reversal, reuse the discrete-streamfunction face sampling from
[vortex_reverse.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/examples/vortex_reverse.jl), but disable adaptation.
Keep its closed-wall setup distinct from the periodic toy. Advect the
interface, a nonconstant intrinsic scalar in each phase, and constant-one
tracers; leave staggered transport for Stage 6. Record the reconstruction
method, `fraction_floor`, sweep order, and peak-velocity CFL policy. Measure
the full interval including initial deformation, thin filaments, and reversal.

Implementation experiments: port the selected tile/workspace ownership policy;
compare generic face schedules with a uniform specialization only where profiles
justify it; remove redundant passes; amortize launches across compatible work.
Keep phase volumes and scalar amounts on the same frozen face partition.
Do not parallelize x/y WY sweeps simultaneously: the later sweep consumes the
earlier sweep's updated reconstruction. Build PLIC/height caches before parallel
readers, and give geometry scratch an explicit owner.

| Gate | Required evidence |
|:--|:--|
| G3.1 Numerical | Existing scalar/VOF tests pass; vortex volume and per-phase scalar balance, constant-one identities, phase bounds, and return-shape/scalar errors preserve the baseline budgets and convergence thresholds. |
| G3.2 Layout | Single-block and many-block layouts agree numerically; both supply enough independent work, with no huge regression for the few-block case. |
| G3.3 Performance | Complete uniform transport, including required reconstruction and CFL, targets `S_16 ≥ 8` on the declared large cases; separately disclose smooth-LW and interface-rich outcomes. |
| G3.4 Cost attribution | Explain the measured toy/main LW gap by geometry, indirection, passes, or launch costs; record fractions rather than impose an arbitrary equality with the toy. |

If the plain scalar scales but VOF does not, retain the proven scalar policy
and investigate interface imbalance, geometry work, and serial sweep overhead.
Do not loosen PLIC accuracy or reduce reconstruction frequency to pass a
threading gate. The output is a uniform transport decision record and numerical
baseline for the following mesh stages.

## Stage 4 — Non-uniform, fixed-in-time mesh advection

**Purpose.** Quantify irregular connectivity and work imbalance while excluding
topology changes from the timed transport interval.

Build each mesh once and freeze it. Use 2:1-balanced layouts with (a) a compact
refined patch, (b) a refined band intersecting the transported interface,
(c) several separated patches, and (d) a strongly asymmetric distribution of
mixed cells. Include periodic seams, multiple tree levels, and anisotropic
cells where the existing operators support them. Record cells, faces, blocks,
level histogram, coarse–fine face count, interface-cell fraction, and stencil
length distribution. For each strong-scaling curve the entire mesh is fixed.

Use cell-centred smooth advection and the Stage 3 interface/scalar vortex.
Freeze refinement over the region needed for the full trajectory so transport
through coarse–fine connections is actually exercised. An equal-cell-count
uniform comparison helps diagnose overhead, but must also show error and face
count; equal cells are not equal accuracy or equal work.

**Work packages.**

1. Audit conservation at every coarse–fine segment. In this repository the
   owned leaf-level segments already connect the coarse and fine cells: one
   integrated flux enters both sides with opposite signs. Preserve this
   mechanism. Do not add a second overlapping coarse-face flux. Classical
   flux-register refluxing would be relevant to a different overlapping-patch
   or subcycled design; introducing local time stepping is outside this plan.
2. Separate regular block interiors from irregular boundary/halo schedules
   only if profiles justify the extra work lists. Compare the generic gather
   with dense interior tiles plus a compact irregular-face pass. Maintain
   identical numerical states and coarse–fine closures across the comparison.
3. Evaluate spatially ordered contiguous chunks, weighted contiguous chunks,
   and dynamically scheduled spatial tiles. A proposed cost model is
   `w_b = a*n_cells + b*n_mixed + c*n_coarse_fine + d*n_expensive_geometry`.
   Fit nonnegative coefficients to measured work or use moving-average block
   costs. Include weight computation and partitioning costs in amortized time.
4. Measure maximum worker busy time divided by mean busy time for each
   expensive phase, targeting at most 1.25 when enough work exists. Distinguish
   a few indivisible expensive items from a poor scheduler. Equal-size static
   batches cannot redistribute already assigned heavy work; test this directly.
5. Try spatial leaf/face ordering as a separate experiment. Reorder every
   associated field and index consistently, or use a schedule permutation.
   Account for its construction cost and extra indirection. Do not replace the
   tree representation merely because spatial ordering helped another code.

The scheduling hypotheses are informed by [AMReX tiling](https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-with-tiling)
and [p4est's weighted partitioning](https://p4est.github.io/api/p4est-latest/p4est_8h.html);
their implementations are not transplanted directly.

For any new non-uniform cell-centred LW path, review unequal-cell reconstruction,
transverse/split treatment, and interface stabilization separately. The existing
AMR LW/MC staggered policy does not automatically define a valid centred LW
scheme. Keep the validated upwind and WY paths as threading controls while
that numerical work is evaluated.

| Gate | Required evidence |
|:--|:--|
| G4.1 Topology/conservation | No missing or duplicate segments; exact ownership audit; coarse–fine mass balance, constant-state behavior, periodic seams, and fixed-mesh return errors pass. |
| G4.2 Accuracy | Existing mixed-mesh convergence thresholds remain met; each changed interpolation/flux policy is identified and independently validated. |
| G4.3 Balance | Worker-time distributions for clustered and dispersed interface work; imbalance target met or an indivisible-work limit quantified. |
| G4.4 Scaling | Complete fixed-mesh transport targets `S_16 ≥ 8` for the mandatory large layouts; setup and schedule-build amortization also reported. |
| G4.5 Regression | Stage 3 uniform performance remains within its accepted budget; memory growth stays proportional to entities and small per-task scratch. |

**Exit artifact:** reusable topology schedules, measured scheduling policy,
serial/parallel fixed-mesh reference trajectories, and a cost model for the
next adaptation stage. If irregular scheduling loses locality, choose a hybrid
of spatial tiles and coarse dynamic work rather than one task per face.

## Stage 5 — Dynamic tree AMR advection

**Purpose.** Preserve the Stage 4 transport gains while including criterion
evaluation, refinement/coarsening, transfer, and invalidation in total cost.

Use the existing AMR criteria, update interval, hysteresis/coarsening delay,
minimum level, and 2:1 balance rules. The optimization must not gain speed by
adapting less often, coarsening more aggressively, or changing the accepted
mesh error. Parameter studies may explore those choices separately.

### Explicit adaptation phases

| Phase | Candidate parallel work | Ordering/ownership constraint |
|:--|:--|:--|
| A. Criteria | Independent block/cell estimates, local maxima, label proposals | Fields/topology remain immutable; reduce proposals with the existing priority rules. |
| B. Decisions and balance | Disjoint sibling-family decisions, discovery of required neighbors | Resolve conflicts deterministically; initially retain a short serial topology-commit phase. |
| C. Stage old state | Independent field snapshots and transfer descriptors | Old connectivity remains valid until all consumers finish. |
| D. Rebuild schedules | Counts, prefix sums, disjoint fills for cells/faces/halos | Never append to a shared vector from several workers; publish only complete schedules. |
| E. Transfer | New-block-owned conservative restriction/prolongation, phase geometry, face transfer | Prevent two writers to shared faces; preserve conservation and phase-velocity rules. |
| F. Retarget and refresh | Reusable capacities, changed-region schedules and geometry | Advance/validate topology generation, preserve field identity, invalidate all dependent caches before next step. |

This table describes logical dependencies, not permission to reorder the
current implementation blindly. In particular, its old-field staging depends
on retained old schedules even after tree edits. First map that actual lifetime
in the adaptation code. Reuse the existing topology generation rather than
inventing an unrelated counter; distinguish topology validity from evolving
phase geometry/coefficient validity.

Avoid concurrent mutation of RegionTrees until thread-safety and family-level
ownership are established. Parallel decision discovery and transfer may be
enough; only pursue concurrent topology edits if their remaining serial cost
prevents the target. Test equal-size but different-topology updates, since
array length alone cannot validate a cache.

Retain growable workspace capacity and count steady-state allocations over many
regrids. Separate unchanged, refined, coarsened, and balanced-only regions.
Measure whether incremental repair is cheaper than rebuilding; retain a full
rebuild as a numerical oracle. Update work weights as the interface moves,
but reweight/repartition only at a cadence whose total cost is justified.

### Two complementary timing modes

**Recorded mesh schedule:** record a serial run's refinement/coarsening labels,
time steps, and event times; replay the same decisions from the same initial
mesh at every worker count. Include applying edits, balancing, schedule rebuild,
transfer, and refresh. Criterion evaluation can be measured against its recorded
answers, but replay alone is not evidence about live adaptive decisions.

**Live AMR:** run actual criteria at every count, including all costs. Record
leaf-key hashes, counts, minimum spacing, step sequence, and errors. Where
floating-point thresholds change a decision, investigate and either make the
decision deterministic or label the workloads different; do not call different
mesh trajectories a clean strong-scaling comparison. Hysteresis remains the
same across counts.

For each run report
`T_total = T_transport + T_criteria + T_edit/balance + T_schedules + T_transfer + T_refresh + T_diagnostics`,
plus regrid count, changed-block fraction, median/p95 regrid latency, and
per-cell-step throughput `sum_k(N_cells,k)/T_total`. Throughput supplements,
but does not replace, fixed-workload speedup. Measure no-change checks as well
as actual regrids. Compute adaptation's one-worker time fraction and attainable
total speedup using the common Amdahl model.

| Gate | Required evidence |
|:--|:--|
| G5.1 Transfer/lifecycle | Repeated refine–coarsen cycles preserve mass/constant fields; correct topology generations, registered field identities, and workspace retargeting; no stale reads. |
| G5.2 Adaptive accuracy | Full vortex deformation/reversal and moving refinement patches meet baseline error/conservation budgets at equal AMR criteria. |
| G5.3 Equal work | Replay is identical across counts; live mesh/step histories match or their differences are disclosed and explained. |
| G5.4 Scaling | Target `S_16 ≥ 8` for whole adaptive trajectories, including all regrid costs; fixed-epoch stepping must retain Stage 4 gains. Report replay and live results separately. |
| G5.5 Memory/latency | No unbounded workspace growth or allocation churn across repeated cycles; regrid latency and serial fraction meet the declared end-to-end budget. |

**Exit artifact:** an AMR lifecycle/ownership diagram, an invalidation test
matrix, replay/live benchmark fixtures, and an accepted adaptation policy.
If serial topology work prevents the target, quantify the maximum total benefit
before attempting parallel tree mutation. A fast transport-only curve cannot
pass G5.4.

## Stage 6 — Staggered passive advection and convection

**Purpose.** Extend ownership/scheduling to the dual-volume transport of
staggered quantities for single-phase, one-velocity two-phase, and intrinsic
two-velocity models.

Start with prescribed carriers and passively transported centred **and**
staggered fields in vortex reversal. Keep the carrier separate from the
transported velocity-like field, even when both have the same initial values.
This isolates convection from nonlinear feedback and physical pressure
projection. Actual coupled momentum evolution follows in Stage 8.

Use [staggered_schedules.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/operators/advection/staggered_schedules.jl),
[staggered_fluxes.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/operators/advection/staggered_fluxes.jl),
[staggered_update.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/operators/advection/staggered_update.jl),
[phase_face_transport.jl](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/two_phase/phase_face_transport.jl), and the
[two-velocity transport workspace](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/src/two_phase/two_velocity/transport_workspace.jl)
as the initial implementation map.

| Substage | Carriers and transported quantities | Essential checks |
|:--|:--|:--|
| 6a Single phase | One prescribed staggered carrier; smooth cell scalar and both face components, including constant-one fields | Dual-volume conservation, axis symmetry, reversal error, alias-safe source snapshots. |
| 6b Two phase, one velocity | WY interface plus intrinsic cell/face scalars and mass-weighted momentum equivalents | Same phase mass fluxes for centred and dual updates; density ratios 1, 10, 1000, with higher-ratio stress cases if supported. |
| 6c Two velocities | Separate liquid/gas carriers; equal-carrier validation and nontrivial tangential slip; both interface transport policies | Per-phase compatibility, activity masks, gas-reset diagnostics, normal/tangential behavior, each component/phase's error. |

Repeat all substages on uniform, frozen mixed, and dynamically adaptive meshes.
For unequal phase velocities, use compatible prescribed data and a separately
verified return/reference solution; a carrier with a jump is not automatically
the same analytic reversible problem as the continuous-carrier vortex. Reuse
the existing rotation/transport integration cases to test true discontinuous
velocities in addition to the equal-carrier vortex limit.

**Core invariants and work packages.**

- Retain the centred-to-dual flux compatibility, conventionally written
  `D_s I_F = I_C D`: a staggered one-function must reproduce the interpolated
  phase-volume update. Construct or test the identity on small AMR graphs.
- Compute each expensive geometrical phase flux once per sweep, publish it,
  then consume it through output-owned dual-volume gathers. Separate x/y
  component workspaces before considering concurrent component execution.
  Concurrent components sharing transport scratch are not independent tasks.
- Preserve snapshots of old momentum/velocity. Shared fluxes, phase fractions,
  donor states, dilation terms, and reconstruction must correspond to the same
  sweep; more fusion is useful only within that dependency boundary.
- Test active, newly active, nearly empty, and exactly absent phases. The
  package stores `NaN` for exactly absent intrinsic velocities; skip inactive
  values before arithmetic, including reductions. Do not evaluate `0*NaN`
  or change an activity threshold to make a threaded calculation easier.
- `ProjectedLiquidVelocity()` includes a carrier projection and an auxiliary
  gas-carrier reset. Measure that projection separately **and include it in
  complete-policy time**; retain it unchanged until Stage 7. Report the reset
  discrepancy and its convergence. Do not incorrectly impose exact physical
  gas-amount conservation on this policy where the current split formulation
  explicitly reports a reset. `FrozenInterfaceFluxes()` has a different balance
  contract. See [two-velocity advection](two-velocity-advection.md).
- Exercise LW in smooth bulk regions, its documented MC AMR closure, and the
  established Fromm/limited choices used in reference studies. Preserve CTU
  fallback near phase interfaces. Parallelization must not remove a fallback
  or replace the shared mass-flux construction by unrelated momentum fluxes.

[Remmerswaal and Veldman](https://arxiv.org/abs/2209.14934) motivate compatible
mass/momentum transport and reversible-flow validation. Exact split-AMR
requirements here come from the inspected package and its tests.

| Gate | Required evidence |
|:--|:--|
| G6.1 Algebra | Constant-one centred/staggered identities, conservation appropriate to each policy, and dual-flux compatibility hold at all thread counts. |
| G6.2 Accuracy | All three model families and mesh regimes pass vortex/reference tests; preserve existing per-phase order gates (including the documented 1.2 finest-pair exception for projected gas in the current test). |
| G6.3 Robustness | Correct phase masks and newly active states, no contamination by absent-phase NaNs, supported aliasing and field transfer work across regrids. |
| G6.4 Scaling | Complete transport targets `S_16 ≥ 8` on each model's mandatory large cases; publish liquid/gas and scalar/momentum breakdowns and projection overhead. |

If the projected-carrier solve dominates, record G6.4 as pending Stage 7 rather
than excluding that solve. The correctness-approved transport design can supply
Stage 7 fixtures without claiming that its overall performance gate has passed.

## Stage 7 — Poisson, geometric multigrid, and CMDGFM projection

**Purpose.** Reach approximately 50% physical-core scaling efficiency for
accurate complete solves/projections while retaining geometric multigrid and
the CMDGFM pressure block.

### 7a. Single-phase Poisson and geometric multigrid

Start with uniform constant-coefficient manufactured Poisson problems, then
fixed mixed meshes, then replayed/live AMR updates. Include periodic/all-Neumann
nullspaces and Dirichlet/mixed boundary cases. Use both repeated fixed RHS
solves with a reset guess and evolving RHS sequences with a matched warm-start
policy. Test the actual public projection path as well as direct `solve!`.

**Hierarchy prerequisite:** print unknowns, nonzeros, storage type, and work
per level before timing. Build the primary large cases by refining a modest
permanent base mesh, producing a genuinely multilevel hierarchy. Also include
a many-root uniform layout as a stress case. If its base problem remains too
large, additional geometric coarsening within/across base-grid blocks is a
separate numerical subproject, with compatible transfers and operator tests.
It retains GMG; replacing it with AMG or a large direct solve does not satisfy
the intended solver stage. Do not hide this limitation by dropping root-grid
configurations from the final coverage claim.

Profile and optimize in this order, adjusting only when measurements disagree:

1. Fine/coarse operator application, including exact factorized fine actions
   and stored sparse rows. Compare row-oriented actions against the existing
   transpose-CSR-like traversal; no CSC-column scatter with shared outputs.
2. Jacobi update, residual, restriction/prolongation, vector copies, and
   nullspace/compatibility reductions. The current sparse transfers call `mul!`;
   establish their actual thread behavior rather than assuming it from the
   surrounding threaded code. Destination-owned transfer rows are a candidate.
3. Per-level launch overhead and active worker counts. Use fewer workers or
   serial execution on small levels; choose the crossover from measurements.
   The finest-level threshold is not necessarily the right coarse-level cutoff.
4. Bottom factorization and solve. Reuse valid factors; keep a genuinely small
   bottom problem serial if that minimizes whole-cycle time. Include rebuilds
   when coefficients or topology change.
5. Krylov dot products, norms, axpy operations, and stopping diagnostics.
   A threaded matrix action alone does not thread Krylov.jl's vector work.
   Test thread-aware vector operations or a suitable supported integration if
   these become dominant; changing BLAS thread counts inside every iteration
   is not a default solution.

Keep existing damped Jacobi as the first smoother. If time-to-tolerance remains
poor, evaluate a bounded set of geometric-MG smoother options, such as a fixed
Chebyshev polynomial or block Jacobi, with validated spectral bounds and
convergence. Do not assume red–black coloring of a Cartesian nearest-neighbor
graph is valid for the AMR Riesz couplings. Preserve the symmetry/positive
definiteness required by the chosen outer method, or use an appropriate outer
method and validate it explicitly.

The immutable numerical contract includes compatible divergence/gradient,
the non-diagonal AMR metric, coefficient insertion, integrated versus strong
residual conventions, and the pressure gauge. Replacing the actual AMR
operator by a cheap five-point stencil fails this contract.

### 7b. Two-phase pressure and evolving coefficients

Add one-velocity two-phase projection with the existing variable coefficients,
including density ratios 1, 10, and 1000, a planar and curved interface,
coarse–fine/interface intersections, and supported cut-cell stress cases.
First freeze geometry/coefficient values for application timings; then move
the interface on a fixed mesh; finally include AMR. Separate topological
hierarchy reuse from numerical refresh on every level.

Measure setup, symbolic/schedule reuse, coefficient refresh, factorization,
V-cycles, Krylov iterations, true residuals, and velocity correction. Preserve
force/projection consistency, especially balanced surface tension. Validate
with manufactured variable-coefficient solutions, divergence reduction,
stationary-droplet behavior, and wave cases. A factorization that is fast only
because it uses stale coefficients is not a valid optimization.

### 7c. CMDGFM with the pressure block fixed

Write the actual gauge-reduced two-block system as

```math
A\begin{bmatrix}p\\\xi\end{bmatrix}
=\begin{bmatrix}K&B\\C&J\end{bmatrix}
 \begin{bmatrix}p\\\xi\end{bmatrix}
=\begin{bmatrix}b_p\\b_\xi\end{bmatrix}.
```

Here `K` is the existing pressure block, and the pressure preconditioning action
is the geometric-MG action established in 7a/7b. **Hold that pressure
discretization, gauge, and validated GMG strategy fixed while comparing CMDGFM
strategies.** Optimize its execution through the shared GMG work; do not
replace it with AMG, a different pressure equation, or whole-system ILU.
The jump/coupling preconditioner and outer solver may change.

The current lower block action approximately computes
`z_p = M_K^(-1) r_p`, then `z_ξ = M_J^(-1)(r_ξ - C z_p)`, with jump ILU.
This gives a controlled comparison: initially replace only `M_J`.

| Candidate | Expected threading advantage | Numerical/performance risk and decision test |
|:--|:--|:--|
| Existing jump ILU | Baseline, possibly fastest for tiny interfaces | Factorization and triangular dependencies; measure their fraction before replacing it. |
| Diagonal or fixed-count damped Jacobi on `J` | Independent row work, low setup | Weak diagonal dominance is not a guarantee of fast convergence; test contraction and reject stalled cases. |
| Small geometric/interface block Jacobi | Independent local factors per patch | Partition choice and long-range interface coupling; compare iterations and setup amortization. |
| Overlapping additive/ restricted Schwarz on interface patches | Independent local solves with stronger coupling | Overlap work and assembly; gather weighted corrections by output ownership rather than racing scatters. |
| Approximate interface Schur preconditioner | Accounts for `S_ξ = J - C K^(-1) B` | Extra pressure applications may outweigh iteration savings; apply approximations within the preconditioner while retaining the exact outer system. |

Start with a single simple diagonal/fixed-iteration candidate and one patch
candidate. Add Schur work only if coupling demonstrably limits convergence.
Keep a serial small-jump fallback. Interface unknowns may scale with perimeter
rather than cell count in 2-D; also test fragmented interfaces so the chosen
method is not tuned only to one smooth circle.

For a fixed linear block action compare existing BiCGSTAB with restarted GMRES
on time, memory, robustness, and true residual. If inner solves stop adaptively
or the preconditioner otherwise changes within an outer solve, use FGMRES
instead of assuming standard BiCGSTAB/GMRES remains valid. FGMRES incurs basis
storage and orthogonalization traffic; count these costs. This follows
[flexible Krylov guidance](https://petsc.org/release/manualpages/KSP/KSPFGMRES/)
and can use [Krylov.jl's existing methods](https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/).
It does not require a PETSc dependency.

Also optimize the full projection pipeline:

- Build jump stencils/triplets with count–prefix-sum–fill or task-owned buffers,
  then deterministic assembly. The current serial `push!` loops cannot simply
  acquire a thread macro. Keep resolved/unresolved-stencil behavior unchanged.
- Cache row-oriented full-system and coupling actions, or use exact block
  operator actions, if their measured benefit pays for extra storage and
  coefficient refresh. Verify each action against the assembled matrix.
- Group gauge-anchor RHS contributions before parallelization: several retained
  pressure unknowns may contribute to the same anchor. The present serial
  `full_right_hand_side[anchor] -= value` cannot be threaded directly.
- Preserve topology/geometry/numeric cache validation. Reuse a jump factor only
  under its validated numeric policy; equal sparsity does not mean equal factors.
  Test interfaces moving without changing the number of active faces.
- Include normal-jump construction, physical defect checks, velocity correction,
  masking, and diagnostics in projection time. Preserve the final **true**
  residual and dimensionless defect criteria, not only Krylov's internal norm.

### Solver acceptance gates

| Gate | Required evidence |
|:--|:--|
| G7.1 Operator | Serial/parallel actions agree; manufactured convergence, constant/nullspace modes, compatibility, and gauges pass on all mesh types; pressure block unchanged. |
| G7.2 Solve | Same physical tolerances, strong/volume-weighted residuals, and corrected divergence; cycle/iteration counts reported. A faster cycle with slower time-to-tolerance fails. |
| G7.3 Two-phase/CMDGFM | Mixture divergence, normal-velocity continuity, pressure-jump behavior, tangential-slip retention, and pressure-work/energy diagnostics match validated references; include density contrast and underresolved/cut-cell cases. |
| G7.4 Reuse | Cold setup, unchanged solve, moving coefficients, moving active set, and topology change all validated and timed. Existing pressure-reuse tests remain effective. |
| G7.5 Scaling | Target `S_16 ≥ 8` for complete single-phase solves, two-phase projections, and CMDGFM projections on mandatory large cases, with setup/refresh-inclusive trajectories as a second curve. Report all 32-worker metrics. |
| G7.6 Robustness | No systematic new failures or hidden tolerance relaxation; memory/iteration-growth limits specified for the selected Krylov method; retained fallback covers small or pathological interfaces. |

Use the existing [Poisson tests](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/test/operators/poisson.jl),
[solver tests](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/test/solvers.jl),
[CMDGFM tests](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/test/two_phase/two_velocity/projection/cmdgfm_projection.jl),
[pressure-reuse tests](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/test/two_phase/two_velocity/projection/cmdgfm_pressure_reuse.jl),
and two-velocity rotation/wave integration cases. Compare tiny coupled systems
against a tightly solved assembled reference with matching gauge.

If jump preconditioning is a negligible fraction, keep ILU and optimize the
actual dominant work. If a parallel preconditioner increases iteration count,
accept it only when complete projection time improves across the mandatory
robustness matrix. The selected pressure/jump strategy and crossover must be
documented, not inferred from one successful vortex snapshot.

## Stage 8 — Full-package integration and representative flows

**Purpose.** Demonstrate that the accepted transport, AMR, and projection
changes improve real simulation time and retain numerical behavior.

Choose and freeze moderate and large configurations of the following cases;
the existing example defaults are starting points, not scaling sizes.

| Mandatory case family | Model/mesh coverage | Primary validation | Likely profiling questions |
|:--|:--|:--|:--|
| [Vortex reversal](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/examples/vortex_reverse.jl) | Uniform, frozen mixed, and live AMR; centred/staggered passive fields for all three velocity models | Return shape, phase volume, scalar/momentum errors and convergence | Reconstruction, transport, imbalance, regrid cost. |
| [Lid-driven cavity](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/examples/lid_driven_cavity.jl) | Single-phase full flow; fixed mesh and a declared AMR configuration | Baseline velocity profiles, divergence, pressure gauge, transient or steady-state criterion | Wall conditions, viscous solve, projection, CFL and diagnostics. |
| [Stokes-3 wave](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/examples/stokes3_wave.jl) | One-velocity and CMDGFM two-velocity models; fixed and adaptive meshes | Wave phase/amplitude, mass, pressure/velocity defects, energy behavior | Evolving coefficients, curvature, forces, momentum transport, cache refresh. |
| [Taylor–Green](https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/157448c59ad564f28a7130778a5ce4c94eacfe64/examples/taylor_green.jl) | Smooth single-phase reference at several resolutions | Analytic decay/error and temporal/spatial convergence | Whether general changes preserve the smooth fast path. |

Retain a small stationary/capillary droplet test as a force-balance regression,
and a cylinder/cut-cell case for boundary/geometry coverage if those paths are
part of the claimed supported performance envelope. These need not expand into
a full Cartesian product of every parameter.

Use the public `FlowSimulation`/`simulate!` path for full-flow cases. Vortex
reversal remains a prescribed-carrier transport case; its success alone does
not demonstrate scaling of Navier–Stokes. Keep timestep selection, solver
stopping rules, AMR criteria, and prescribed physical duration identical.
For steady-state runs report both a fixed-step comparison and time to the same
convergence criterion; differing iteration/step counts need explanation.

At 1, 16, and the best SMT count, partition complete runtime into convection,
VOF/geometry, diffusion, forces, projection, AMR, reductions, diagnostics, and
output. Optimize in decreasing order of possible whole-run benefit. Revisit
implicit/staggered diffusion, variable-stress preconditioners, boundary updates,
height/curvature evaluation, phase extensions, and serial field-copy paths only
as profiles warrant. Apply the same ownership and solver gates to those kernels.

Run a compute-only configuration and a declared production diagnostics/output
configuration. Include required diagnostics in the latter. Measure visualization
or checkpoints separately and in total when enabled; disabling them is a useful
measurement control, not proof of production speedup. Preserve snapshot
ownership if asynchronous visualization is active.

| Gate | Required evidence |
|:--|:--|
| G8.1 Physics | Existing relevant unit/integration tests and baseline observable/error budgets pass; no hidden numerical method, timestep, or stopping-tolerance change. |
| G8.2 End-to-end scaling | Each mandatory large case targets `S_16 ≥ 8`; publish every per-case result and explained miss. A geometric mean cannot conceal a failed family. |
| G8.3 Generality | Small/moderate cases retain sensible serial crossover; uniform/fixed/adaptive layouts, one/two velocities, and supported boundaries remain correct. |
| G8.4 Reliability | Long adaptive runs have bounded memory; repeated runs and concurrent independent simulations do not race; no new solver breakdown pattern. |
| G8.5 Delivery | Reproducible configurations/results, chosen thread recommendations, documented compatibility/fallbacks, and an explicit list of remaining bottlenecks. |

Report both self-speedup and speedup against the original package at 1 and 16
workers. At a fixed error target, also report how much total wall time and
memory AMR saves relative to uniform refinement; that is an application benefit,
distinct from threading efficiency.

## Execution order and decision records

The default progression is `1 → 2 → 3 → 4 → 5 → 6 → 7 → 8`. Numerical gates
must pass before the corresponding execution policy spreads. Stage 7a's
single-phase solver profiling can be prepared once Stage 4 is stable, because
Stage 6's projected-carrier transport may depend on its eventual performance.
This dependency does not waive Stage 6's complete-policy gate.

Use small reviewable implementation changes in the later stage:

| Change group | Concrete deliverable | Must be available before moving on |
|:--|:--|:--|
| Measurement foundation | Baseline capture, configuration/results schema, plotting and correctness driver | G1.2–G1.3 evidence. |
| Standalone numerics | Serial periodic LW package and independent reference tests | G2.1. |
| Standalone threading | Selected backend, storage/tiling experiments, scaling report | G2.2–G2.4. |
| Uniform transport | Execution changes; centred LW addition reviewed separately | G3 results. |
| Frozen AMR transport | Ownership schedules, irregular-work policy, cost model | G4 results. |
| Dynamic AMR | Lifecycle-safe parallel phases and replay/live timing | G5 results. |
| Staggered/two-velocity transport | Dual/phase compatibility and full transport timing | G6 numerical gates; any solver-related performance dependency explicit. |
| GMG and CMDGFM | Threaded solve pipeline and selected jump/outer strategy | G7 and outstanding G6 results. |
| Full flows | Profile-driven remaining kernels and representative end-to-end report | G8 results and recommendations. |

Do not combine a new numerical discretization, storage rewrite, scheduler
replacement, and solver-policy change in one experiment. Make causal
comparisons possible. If a simpler implementation is within measurement noise
of a complex one, choose the simpler one.

For every gate, preserve a compact record with: configuration/commit, numerical
checks, raw-result location, `T_1/T_16/T_32`, `S/E`, active workers, confidence
interval, allocations/peak memory, setup/refresh cost, dominant remaining work,
decision, and the next justified experiment. Current entries are all **not yet
measured** except the Stage 1 research deliverable.

## Suggested future validation commands and artifacts

Use separate Julia processes; the following are existing entry points, not
commands executed as part of this planning task:

```sh
julia --startup-file=no --project=. --threads=1,0 -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. --threads=16,0 -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. --threads=32,0 test/integration/runtests.jl
julia --startup-file=no --project=. --threads=16,0 test/performance/runtests.jl
```

Confirm the spawned test process's effective worker counts; make counts explicit
through the test invocation/environment when necessary. Run selected numerical
integration studies at 1 and 16 as well as 32, and reserve the full suite for
milestone changes. The future scaling driver must add the problem-size ladder,
reset/replay semantics, affinity records, and result schema described above;
the existing tiny benchmark suite remains useful for smoke checks.

Required plots are wall time and speedup versus workers, efficiency versus
workers, throughput versus working-set size, component runtime fractions,
worker imbalance, and (for solvers) iterations/time versus achieved residual.
Include errors beside timings. Stage 5 additionally needs regrid time and
memory versus simulation time. Generate standalone plot artifacts and save the
data/plotting script together so every acceptance decision can be reproduced.

The immediate next implementation deliverable is the measurement foundation
and serial standalone LW package. Backend selection and all numerical-package
optimizations remain decisions to be made from those experiments.
