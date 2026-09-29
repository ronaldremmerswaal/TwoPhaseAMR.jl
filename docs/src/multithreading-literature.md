# Literature review for multithreading optimization

Research date: 29 September 2026. This review supports the
[eight-stage optimization plan](multithreading-optimization-plan.md).
It distinguishes published results, documented implementation behavior, and
proposed experiments. No package scaling results were measured for this review.

## Main findings

### Julia: ownership, task granularity, and composition

Julia provides shared-memory tasks and threaded loops, but correctness still
requires identifying which task owns each output. Tasks can migrate; scratch
indexed by `threadid()` is therefore unsafe as a general library design.
Use work-item or task-owned scratch and explicit completion before consuming
outputs. This is documented in the [Julia threading manual][J1] and the
developers' [2023 task-local-state advisory][J3].

The scheduling choices are materially different. `@threads :dynamic` uses a
bounded number of tasks and contiguous iteration regions; `:greedy`, available
since Julia 1.11, suits uneven iteration costs. `:static` has calling-context
and nesting restrictions. These distinctions matter because this package
supports Julia 1.10. A newer scheduler may be an experiment, but adopting it
requires either a compatible alternative or an explicit version-policy change.
See the [Base threading reference][J2].

Polyester reuses tasks and provides a useful low-overhead comparison for short,
regular loops. Its worker-selection and nesting policies must be measured,
including the distinction between physical cores and logical threads. Its
documented default favors cores. This is especially relevant to a requested
32-thread run on a 16-core CPU. It is a candidate backend, not a reason to
preselect the toy solver's design. See [Polyester's documentation][J4].

The [2019 Julia threading design article][J5] explains task composition and
work scheduling. Its historical thread-local advice should be read together
with the later migration advisory. Ordinary Julia task composition also does
not establish that independently threaded native libraries or custom worker
pools will compose without oversubscription.

**Implication for this project:** compare serial, coarse Julia tasks, threaded
tiles, and persistent-worker loops over the same numerical kernel. Separate
launch/barrier costs from useful work. Choose task sizes experimentally, and
keep ownership independent of the executing worker.

### Finite volume: shared faces create dependencies

Computing a face flux is naturally independent; adding it simultaneously to
both adjacent cell residuals creates conflicting writes. Giuliani and
Krivodonova describe edge/face coloring, including nonconforming adaptive
meshes, as a way to resolve this dependency. Their measured examples are DG
on GPUs, so the paper supplies an ownership technique, not a CPU finite-volume
scaling prediction. See [*Edge coloring in unstructured CFD codes*][F1].

Reguly et al.'s production CFD study of OP2/Hydra is a useful counterweight to
small-loop benchmarks. On their two-socket machine, the 24-thread OpenMP
configuration underperformed MPI; the hybrid configuration was also about 10%
slower than MPI alone. The authors discuss locality, coloring, and threading
overheads. This is evidence that race avoidance and whole-application locality
must be assessed together, not evidence that MPI should replace threading on
this workstation. See [the paper, section 4.1.1][F2].

Basilisk documents automatic parallel field iteration when stencil writes
are local, plus explicit reductions for nonlocal scalar results. It also
notes the performance cost of a quadtree compared with a simple Cartesian
grid. See [Basilisk C, parallel programming][T1].

**Experiments inferred for this project:** compare (a) a face-flux array followed
by an output-owned gather, (b) local recomputation of inexpensive LW fluxes in
an output-owned stencil, and, where worthwhile, (c) coloring. Global atomics
and complete residual arrays per worker are reference alternatives, not
defaults. The first approach favors conservation auditing and shared use of
expensive geometric fluxes; the second can save memory traffic for a cheap
uniform stencil. Neither is universally faster.

### Tree meshes: blocks, tiles, and work weights

Afivo is the closest research match: quadtree/octree finite-volume AMR,
OpenMP shared memory, and geometric multigrid. It parallelizes loops over
blocks and discusses ghost-cell overhead and bandwidth limits. In section
4.2, a **uniform 512³ Poisson problem**, using two Xeon E5-2680v4 processors,
achieves maximum speedups of about **10–12 using up to 26 cores**. Dividing
those speedups by 26 gives approximately **38–46% efficiency** as an indicative
comparison, not a digitized measurement at a specific plotted point. Larger
blocks reduced time per unknown. These are FMG-cycle results,
not full adaptive-flow timings. See [Teunissen and Ebert (2018)][T2].

AMReX's documented box/tile distinction allows a large box to supply several
work items. It also supports dynamic box/tile scheduling and warns that using
an entire box where a tile is expected duplicates work. This provides a
concrete model for testing block scheduling versus within-block tiling.
See [AMReX, MFIter and tiling][T3].

p4est develops scalable forest refinement, coarsening, partitioning, and 2:1
balancing. Its API permits user-defined leaf weights. Those results and
partitions concern distributed memory; they cannot be quoted as shared-memory
speedups. The transferable ideas are compact ordered leaf sets, explicit
adaptation stages, and work weights. See [Burstedde et al. (2011)][T4] and
[the partition API][T5].

**Implication for this project:** use the tree to manage topology, but test
compact block/face schedules in hot numerical loops. Compare unweighted
contiguous partitions with interface-aware weights and a queue of spatial
tiles. Include schedule construction, conservative transfer, and invalidation
in dynamic-AMR measurements. Equal cell counts do not imply equal PLIC work.

### Interpreting scaling on this machine

The Roofline model relates attainable arithmetic performance to memory
bandwidth and operational intensity. It uses traffic reaching the relevant
memory level, not a count of every source-level load. The original paper
also discusses serial-work limits. See [Williams, Waterman, and Patterson
(2009)][P1]. [STREAM][P2] provides a reproducible bandwidth reference for large
contiguous arrays, with documented byte-counting rules.

Thus there is **no literature-backed universal requirement of 50% efficiency**
for every stencil, mesh size, or thread count. Afivo makes it a credible,
ambitious engineering target for sufficiently large tree-based workloads.
A simple LW kernel can saturate memory bandwidth before 16 cores. The plan
therefore keeps 8× speedup at 16 physical cores as its target, reports misses
honestly, and requires evidence before accepting a bandwidth-limited outcome.

[ThreadPinning's API][P3] and [its BLAS guidance][P4] provide controls for
affinity and nested-library threads. They are measurement tools, not required
runtime dependencies. The installed CPU has one OS NUMA node but two L3 cache
domains; placement experiments should reflect that actual topology.

### Solvers and numerical invariants

PETSc documents block factorizations/Schur approaches and flexible GMRES for
preconditioners that vary during a solve. Krylov.jl documents BiCGSTAB, GMRES,
and FGMRES interfaces already relevant to this repository. These establish
candidate algorithms; they provide no evidence that a particular CMDGFM
replacement will converge or scale here. See [PCFIELDSPLIT][S1],
[KSPFGMRES][S2], and [Krylov.jl][S3].

The numerical reference for the toy's requested flux is Lax and Wendroff's
[*Systems of conservation laws*][N1]. Clawpack's author-maintained
[advection demonstration][N2] illustrates LW's dispersive oscillations at
discontinuities. Consequently, the toy must use smooth accuracy tests, and
unlimited LW must not replace bounded geometric VOF transport.

Remmerswaal and Veldman study compatible mass and staggered-momentum
transport, including two velocities and prescribed reversible flows. Their
paper supports the choice of vortex reversal and mass/momentum compatibility
as validation principles. It does not establish every detail of this
repository's split-sweep or AMR extensions. Those differences are explicitly
tracked in the existing [literature coverage audit](literature-todos.md).
See [the transport paper][N3].

## Annotated primary sources

| ID | Source | Use and limitation |
|:--|:--|:--|
| J1 | [Julia manual: Multi-Threading][J1] | Runtime semantics, races, migration, and thread pools; version-sensitive documentation. |
| J2 | [Julia Base threading reference][J2] | Exact scheduler contracts and version requirements; do not assume latest APIs exist in Julia 1.10. |
| J3 | Protter, Churavy, Butterworth et al. (2023), [PSA: Thread-local state is no longer recommended][J3] | Scratch ownership and migration hazards. |
| J4 | JuliaSIMD, [Polyester.jl README][J4] | Persistent task reuse, batching, worker selection, and nesting controls; verify the resolved version. |
| J5 | Bezanson, Nash, Pamnany (2019), [Composable multithreaded parallelism][J5] | Design rationale; historical guidance is superseded where noted above. |
| F1 | Giuliani and Krivodonova (2016 preprint), [Edge coloring in unstructured CFD codes][F1] | Conflict-free face work on adaptive meshes; performance evidence is GPU/DG. |
| F2 | Reguly et al. (2014 preprint), [Acceleration of a Full-scale Industrial CFD Application with OP2][F2] | Whole-code costs, data ordering, coloring, and shared-memory limitations. |
| T1 | Popinet and contributors, [Basilisk C][T1] | Primary implementation documentation for local writes and reductions on adaptive grids. |
| T2 | Teunissen and Ebert (2018), [Afivo: A framework for quadtree/octree AMR with shared-memory parallelization and geometric multigrid][T2], *CPC* 233, 156–166; [DOI](https://doi.org/10.1016/j.cpc.2018.06.018) | Direct shared-memory tree-AMR comparison; 3-D, different hardware, mostly cycle-level scaling evidence. |
| T3 | AMReX developers, [MFIter and tiling][T3] | Block versus tile ownership and scheduling; implementation guidance rather than a speedup guarantee. |
| T4 | Burstedde, Wilcox, Ghattas (2011), [p4est][T4], *SISC* 33(3), 1103–1133 | Tree topology and balancing algorithms; distributed-memory results. |
| T5 | p4est developers, [p4est_partition API][T5] | User-defined work weights; adapt the idea, not the MPI implementation. |
| P1 | Williams, Waterman, Patterson (2009), [Roofline][P1], *CACM* 52(4), 65–76; [DOI](https://doi.org/10.1145/1498765.1498785) | Bandwidth/intensity model and performance ceilings. |
| P2 | McCalpin, [STREAM reference information][P2] | Sustainable bandwidth and traffic conventions; not an irregular-stencil ceiling by itself. |
| P3 | Bauer and contributors, [ThreadPinning API][P3] | Pinning/querying worker placement on the target OS. |
| P4 | ThreadPinning developers, [Pinning BLAS threads][P4] | Oversubscription and native-library worker configuration. |
| P5 | JuliaCI, [BenchmarkTools manual][P5] | Warm-up, setup, interpolation, and `evals=1` for mutating benchmarks. |
| S1 | PETSc developers, [PCFIELDSPLIT][S1] | Block/Schur design choices; no proposed PETSc dependency. |
| S2 | PETSc developers, [KSPFGMRES][S2] | Flexible outer iteration for changing/nonlinear preconditioning. |
| S3 | JuliaSmoothOptimizers, [Krylov.jl nonsymmetric solvers][S3] | Available Julia solver APIs and storage tradeoffs. |
| N1 | Lax and Wendroff (1960), [Systems of conservation laws][N1], *CPAM* 13, 217–237 | Foundational numerical method; not threading evidence. |
| N2 | Clawpack authors, [Periodic advection demonstration][N2] | LW accuracy/oscillation interpretation. |
| N3 | Remmerswaal and Veldman (2022), [Two-velocity transport of mass and momentum][N3] | Compatible transport and reversible-flow validation. |

Primary papers and official project documentation were preferred over forum
performance anecdotes. Failed publisher/PDF fetches were not used to infer
unseen results. In particular, the Afivo numbers above come from the accessible
paper's section 4.2 text. Numerical thresholds in the plan, such as a 5%
regression budget or a 1.25 load-imbalance target, are **proposed project gates**,
not thresholds claimed by these sources.

[J1]: https://docs.julialang.org/en/v1/manual/multi-threading/
[J2]: https://docs.julialang.org/en/v1/base/multi-threading/
[J3]: https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
[J4]: https://github.com/JuliaSIMD/Polyester.jl
[J5]: https://julialang.org/blog/2019/07/multithreading/
[F1]: https://arxiv.org/pdf/1601.07613
[F2]: https://arxiv.org/pdf/1403.7209
[T1]: https://basilisk.fr/Basilisk%20C#parallel-programming
[T2]: https://homepages.cwi.nl/~ebert/2018-CPC-afivo-Jannis.pdf
[T3]: https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-with-tiling
[T4]: https://doi.org/10.1137/100791634
[T5]: https://p4est.github.io/api/p4est-latest/p4est_8h.html
[P1]: https://aiichironakano.github.io/cs596/Williams-Roofline-CACM09.pdf
[P2]: https://www.cs.virginia.edu/stream/ref.html
[P3]: https://carstenbauer.github.io/ThreadPinning.jl/stable/refs/api_pinning/
[P4]: https://carstenbauer.github.io/ThreadPinning.jl/stable/examples/ex_blas/
[P5]: https://juliaci.github.io/BenchmarkTools.jl/stable/manual/
[S1]: https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/
[S2]: https://petsc.org/release/manualpages/KSP/KSPFGMRES/
[S3]: https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/
[N1]: https://doi.org/10.1002/cpa.3160130205
[N2]: https://www.clawpack.org/gallery/_static/apps/notebooks/classic/advection_1d/advection_1d.html
[N3]: https://arxiv.org/abs/2209.14934
