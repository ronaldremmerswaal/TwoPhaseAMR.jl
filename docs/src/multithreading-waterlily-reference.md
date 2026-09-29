# WaterLily.jl as a fixed-grid reference for the threading plan

Reviewed 29 September 2026 against WaterLily.jl **v1.8.0**, commit
[`2a2b2ba2eeacc885d84e346f743b72a34e0b713f`](https://github.com/WaterLily-jl/WaterLily.jl/tree/2a2b2ba2eeacc885d84e346f743b72a34e0b713f).
This is a source review and an implementation comparison, not a performance
measurement on the TwoPhaseAMR workstation. The local reference checkout is
in `/tmp/twophase-waterlily-reference` and is disposable. The 2025 paper used
WaterLily 1.4; its timing numbers must not be attributed to v1.8.0.

## What the reference establishes

WaterLily solves incompressible flow on a **uniform Cartesian grid** with
staggered velocities. Its paper says identical cells require no stored grid
metadata, and it uses finite-volume pressure and diffusion stencils plus a
limited QUICK convection scheme. That makes it a useful control for dense
loop structure and launch overhead, but not an AMR, VOF, two-velocity, or
CMDGFM performance proxy. The published CPU tests used **Float32**, Julia
1.11, WaterLily 1.4, and an Intel Xeon Platinum 8460Y. Across 1–16 threads,
the paper reports speedup approaching **5×**, rather than 8× at 16. Its
suggestion that KernelAbstractions CPU translation may limit scaling is a
hypothesis in the paper, not a demonstrated cause on this machine. It reports
that pressure projection and convection/diffusion consume most of the measured
GPU step, which does not establish their fractions in TwoPhaseAMR's CPU step.
See the [2025 paper, methods and performance sections](https://arxiv.org/html/2407.16032v3).

The pinned [loop implementation](https://github.com/WaterLily-jl/WaterLily.jl/blob/2a2b2ba2eeacc885d84e346f743b72a34e0b713f/src/core.jl#L77-L145)
selects a SIMD serial backend or a KernelAbstractions backend through a
preference. Its `@loop` maps a Cartesian range to a kernel with each iteration
writing a designated output location. This supports the plan's explicit
serial baseline and output ownership experiments. It does **not** justify
copying its `@fastmath` or `@simd` annotations into a conservative AMR/VOF
kernel without proving dependence and numerical error bounds.

In [convection/diffusion](https://github.com/WaterLily-jl/WaterLily.jl/blob/2a2b2ba2eeacc885d84e346f743b72a34e0b713f/src/Flow.jl#L25-L47),
WaterLily stores a face flux in `Φ`, then applies the two sides in separate
parallel launches. This is an instructive cost/control case for Stage 2's
stored-flux versus locally recomputed output-owned stencil. TwoPhaseAMR's
owned face segments and gathers already encode coarse–fine conservation, so
recomputation must be proved equivalent at each segment before it replaces
the shared flux path.

WaterLily's [Poisson operator](https://github.com/WaterLily-jl/WaterLily.jl/blob/2a2b2ba2eeacc885d84e346f743b72a34e0b713f/src/Poisson.jl#L20-L71)
stores compact neighbor coefficients and applies a matrix-free Cartesian
stencil. Its [multilevel solver](https://github.com/WaterLily-jl/WaterLily.jl/blob/2a2b2ba2eeacc885d84e346f743b72a34e0b713f/src/MultiLevelPoisson.jl#L24-L115)
uses destination-owned restriction/prolongation, serial recursion over levels,
and red–black Gauss–Seidel as the default smoother. It can coarsen rectangular
axes independently. The [2026 solver paper](https://arxiv.org/html/2607.07687v1)
reports 1.6–2.3× lower cost for its adaptive under-relaxed RBGS GMG than its
PCG-smoother GMG in a specific GPU jellyfish case, and fewer V-cycles for
elongated Cartesian domains. The paper's near-ideal MPI scaling concerns
single-thread SIMD ranks and cannot be quoted as shared-memory scaling.

## Bounded transfer into the eight-stage plan

| Stage | Useful experiment or control | Transfer limit and gate |
|:--|:--|:--|
| 1–2 | Keep a simple dense serial kernel, compare Julia tasks and Polyester against a Cartesian output-owned control, and separate the cost of storing face fluxes from recomputation. | Match Float64, complete-step work, and physical error. WaterLily's 5× CPU figure is contextual evidence, not a TwoPhaseAMR target or ceiling. |
| 3 | Compare the toy's dense passes with a uniform main-package path; count launches, face indirection, and PLIC work. | WaterLily's limited QUICK flux is not the plan's LW flux or WY/VOF transport. Preserve frozen sweep state and bounds. |
| 4–5 | Build reusable spatially ordered schedules and test tile/weighted work queues only after measuring irregular work. | A fixed Cartesian index has no topology-generation or coefficient-refresh lifecycle. Conservative face ownership and transfer are mandatory. |
| 6 | Use component-specific output ownership and barrier accounting for staggered transport. | The WaterLily momentum update cannot substitute for compatible phase mass/momentum fluxes, activity masks, or separate two-velocity carriers. |
| 7 | Profile destination-owned sparse transfers, reductions, coarse-level crossover, and hierarchy depth. Compare a validated smoother only after whole-solve timing. | Cartesian RBGS coloring, semi-coarsening, and compact stencil coefficients are not automatically valid for AMR Riesz couplings, variable coefficients, gauges, or CMDGFM's fixed pressure block. |
| 8 | Report complete flow time and per-routine fractions, including projection and convection, as WaterLily does. | Its GPU routine fractions and FP32 results cannot rank this package's CPU bottlenecks. |

The next bounded implementation group after the toy numerical/scaling gates is
the Stage 3 **plain scalar control**, using the existing upwind operator and a
matched prescribed carrier. Compare one large block with many 8×8 blocks,
record kernel and complete transport times at 1/2/4/8 workers, and instrument
face lookup, flux construction, gather, halo/CFL, and allocation fractions.
Only then add the separately validated LW path, followed by the existing WY
interface case. This isolates dense-loop transfer from new numerical-method
work and supplies evidence for whether Stage 4 needs a special interior tile
path. The plan's conservation, small-work regression, and fixed-trajectory
gates still decide acceptance.

For Stage 7, the first solver-specific group should print fine-to-coarse
unknown counts and operator/transfer storage, then time each V-cycle phase
and a complete solve at the same tolerance. In the current package,
[`_restrict!` and `_prolongate!`](../../src/solvers/multigrid/hierarchy.jl)
call sparse `mul!`; [`_smooth!` and residual norms](../../src/solvers/multigrid/cycles.jl)
have different threading and reduction behavior. Compare a destination-owned
row schedule against those transfers only when they matter, retaining the
existing AMR operator and valid coarse factors. A large permanent base grid
can make the direct bottom solve dominant, so report that case separately.
The Stage 7 CMDGFM jump-block experiments should follow this pressure baseline;
WaterLily has no analogue of the jump system.

The primary performance goal is therefore reasonably sought at **4–8 workers**
for each predeclared large fixture: speedups of at least 2× at four and 4× at
eight, with 16/24/32 still measured for diagnosis and a best-count
recommendation. The same numerical and whole-step gates apply. A plateau or
shortfall is recorded against this revised goal, with bandwidth, serial work,
and launch costs measured rather than inferred from WaterLily.
