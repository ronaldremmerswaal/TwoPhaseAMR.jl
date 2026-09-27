# Stokes3 time-step selection: implementation and measurements

This follows [the focused plan](TIMESTEP_PLAN.md). The development benchmark
is [`timestep_analysis.jl`](timestep_analysis.jl). Measurements use Julia
1.12.7, one Julia thread and 16 BLAS threads. The frozen-state timings below
are medians of eight warm calls; allocations are checked on every call.

## Why CFL was expensive

The two-velocity CFL query called `reconstruct!` on every step, even though
the interface already stores PLIC normals, intercepts and curvature. On the
Stokes3 mesh that repeated reconstruction took about 7--9 ms and allocated
0.5--2.8 MB per query, depending on the type-specialization state. The phase
aperture freeze and phase-flux rate scan were already allocation-free in the
uncut case. In addition, the solver-level query fetched a second WY workspace
through a lazy `Ref{Any}`, causing allocation and preventing direct reuse of
the generic phase workspace. The one-velocity CFL scan had a separate `Any`
geometry dispatch in its cell loop.

## What changed

`InterfaceField` now holds a `geometry_is_valid` flag, the exact fraction
snapshot and leaf-block keys from the last reconstruction, and the fixed
solid's geometry revision. `reconstruct!` returns immediately when all of
these still match. This catches direct writes to the public fraction arrays;
`force=true` remains available when intentionally changing derived PLIC
fields. After a WY direction sweep changes fractions, the call at the end of
that sweep performs the full reconstruction and refreshes the snapshot.
An AMR topology change or a solid-geometry refresh invalidates the saved
state. The validity check reads existing arrays and allocates nothing.

The phase workspace's WY advection object now uses the model's interface
transport policy and is shared with the lazy intrinsic-velocity state.
The solver-level CFL query accesses this typed shared object directly,
freezes its apertures and scans its rates; the following transport sweep
consumes those same apertures. No second WY workspace is constructed. The
one-velocity scan and both cut-geometry kernels narrow the geometry type
before their hot loops. The capillary/gravity bound uses an explicit
allocation-free minimum-spacing loop.

| Frozen query | Before, ms / bytes | After, ms / bytes |
| --- | ---: | ---: |
| CMDGFM, initial 214-block mesh | 11.56 / 2,827,456 | 2.28 / 0 |
| CMDGFM, after four AMR updates, 169 blocks | 9.88 / 2,317,568 | 1.76 / 0 |
| One velocity, initial mesh | 0.194 / 11,776 | 0.194 / 0 |
| One velocity, after AMR | 0.156 / 11,776 | 0.153 / 0 |

The CMDGFM initial and post-AMR bounds remained respectively
`0.0024180526666743507` and `0.003418299185777767`; one velocity returned
`0.0024316967068556444` and `0.0034071516757014936`. The no-solid and
embedded-solid solver-level queries for both models have been checked to
allocate zero bytes on warmed, unchanged states. A query immediately after
*modifying* fractions may legitimately reconstruct and allocate; subsequent
queries on that reconstructed state do not.

On the initial Stokes3 mesh, the separate CMDGFM component measurements were
0.009 ms for an already-valid `reconstruct!`, 1.606 ms to freeze phase face
apertures, and 0.873 ms for the two-phase rate scan (each with zero
allocations). The aperture step is **not a copy**: for every face it clips the
two adjacent PLIC planes (and any solid plane), averages their liquid opening,
and stores liquid and complementary gas apertures for the transport step.
Component timings are repeated measurements, so their sum need not equal the
2.288 ms complete-query timing. The remaining one- versus two-velocity cost
is therefore dominated by geometric face-aperture evaluation and scanning two
phase velocities rather than one.

## Shared face-aperture cache

The above component measurements exposed duplicate face clipping. The phase
divergence/projection assembly separately recomputed the same physical liquid
and gas face apertures after transport. The interface now owns those vectors,
with validity tied to its exact reconstructed fraction, topology and solid
revision. Projection reads the shared vectors; the gravity-force face loop
also uses the liquid vector. The WY workspace keeps **separate** frozen
vectors: CFL copies the current cached apertures into them, because both
directional sweeps must use the pre-transport openings even after the first
sweep changes the interface. Once transport finishes, projection fills the
cache for the new interface; the next time step's CFL copies it without
repeating PLIC clipping. Thus geometric face-aperture evaluation occurs once
for each new interface state, normally once per completed physical step after
the initial state. AMR invalidates it and requires another evaluation.

The integrated two-step regression checks the cache's update count: initial
CFL = one evaluation, each post-transport projection = one further evaluation,
and intervening repeated CFL queries = zero further evaluations. Frozen
transport arrays remain unchanged until the next explicit freeze. Full-run
timings for this subsequent cache change are pending below.

The post-change default `t=0.5` one-velocity run took 25.271 s for 206 steps
and 45 mesh updates. Its sampled outer CFL cost was 0.037 s (previously
0.033 s), and it ended with 97 blocks, 12,504 faces and 10 final pressure
iterations, matching the prior run's discrete history.

The corresponding pre-aperture-cache CMDGFM run took 62.385 s for 219 steps
and 32 mesh updates, ending with 166 blocks, 21,448 faces and seven final
pressure iterations. Its sampled outer CFL cost was 0.467 s, down from
4.78 s in the earlier baseline; sampled aperture freezing was 0.325 s.

With the shared aperture cache, the warmed initial-mesh CMDGFM query takes
0.702 ms and zero bytes; its aperture *copy* is 0.019 ms, its two-phase rate
scan 0.669 ms, and its returned bound is unchanged. The default CMDGFM
`t=0.5` advance then takes 51.131 s, 58.7 GB allocated, and 10.459 s GC,
with the same 219 steps, 32 AMR updates, final mesh and seven projection
iterations. Sampled outer CFL cost is 0.221 s. The cache also removes repeated
face clipping from the gravity force and projection assembly: sampled phase
forces fall from 11.449 to 0.710 s. These are separate-profile estimates,
so treat fine-grained differences as approximate.

## Reconstruction call-site audit

| Caller | Actual full reconstruction with the validity check |
| --- | --- |
| Interface construction and explicit fraction initialization | Once after assigning fractions. |
| WY directional transport | Once after each fraction-changing sweep; the pre-sweep check is a no-op when the stored PLIC is current. |
| CFL preparation, transport guard and standalone `advect!` | No-op if current; rebuild if the caller edited fractions. |
| AMR indicator and pressure/projection setup | No-op unless inputs changed. |
| AMR transfer | Pre-transfer check is normally a no-op; post-transfer topology change requires a rebuild. |
| Fixed-solid refresh | Solid PLIC rebuilds if its fractions changed; the revision then invalidates the fluid-interface PLIC. |

**AMR locality remains a separate issue.** The current field transfer moves
fractions but does not remap every derived normal/intercept/curvature cache by
leaf key. Consequently its post-transfer reconstruction is still global, not
restricted to changed blocks. A safe local AMR update would have to remap
unchanged cached blocks and include the full height-function/GHF stencil
closure around changed blocks, including periodic and coarse/fine neighbours.
Merely skipping unchanged blocks without that remap would associate old PLIC
data with the wrong leaf. This is not part of the per-step CFL optimization.

## Validation status

- The direct fraction-edit/floor test, cut-geometry revision test, transport
  test file, and diagnostics test file pass with the validity check.
- The solver-level zero-allocation checks pass on small uncut and cut meshes;
  the focused Stokes3 benchmark confirms both models' initial and post-AMR
  gates (all eight repeated calls allocated zero bytes).
- The Navier--Stokes test file passes. The default `t=0.5` CMDGFM Stokes3 run
  and full suite remain to be rerun after the final changes.
