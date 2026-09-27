# Stokes3 diagnostics: measured costs and reusable geometry

The reproducible focused benchmark is
[`diagnostics_analysis.jl`](diagnostics_analysis.jl). It runs Julia 1.12.7
with one Julia thread and 16 BLAS threads. The figures below are medians of
12 warm calls on frozen states, plus one warm, compilation-free paired advance
per model. They identify the dominant work; they are not confidence intervals.
The first sections are the **pre-change baseline**; implementation and
post-change measurements follow below.

## Cost on a frozen state

The initial states use the same 214-block, 27,576-face mesh and interface.
Both evolved states reach `t=0.05` after four AMR changes and have 169 blocks
and 21,832 faces. Their evolved interface/velocity fields are not identical.

| State and operation | One velocity, ms / MB allocated | CMDGFM, ms / MB allocated |
| --- | ---: | ---: |
| Initial: kinetic energy | 27.5 / 66.39 | 39.2 / 138.53 |
| Initial: linear momentum | 29.8 / 66.39 | 41.3 / 138.53 |
| Initial: face dual volumes alone | 29.3 / 66.17 | 39.6 / 66.17 |
| Initial: mixture density / staggered-fraction scan | 0.36 / 0.23 | 11.0 / 72.36 |
| Evolved: kinetic energy | 18.1 / 49.02 | 23.4 / 106.13 |
| Evolved: linear momentum | 18.2 / 49.02 | 25.3 / 106.13 |
| Evolved: face dual volumes alone | 18.0 / 48.84 | 18.1 / 48.84 |
| Evolved: mixture density / staggered-fraction scan | 0.28 / 0.19 | 4.91 / 57.29 |

`PhaseVolume` and `LeafBlockCount` cost at most a few microseconds here.
`DivergenceNorm` takes about 0.12–0.16 ms / 0.16–0.20 MB for one velocity,
and 0.27–0.35 ms / 0.51–0.65 MB for two velocities. Thus it is not the main
cost of the default diagnostic set.

For each energy and momentum evaluation, `_diagnostic_face_volumes` constructs
two entire AMR dual-advection schedules and then maps their volumes into
global-face order. On the initial mesh, the two schedule builds each allocate
roughly 33 MB and cost roughly 12–20 ms; the final mapping is only about
0.14 ms / 0.66 MB. These schedules also construct flux connectivity that
diagnostics never use. Energy and momentum separately repeat this work on
the same unchanged topology.

The CMDGFM-specific surplus is a type-specialization issue in the per-face
staggered-fraction loop. The existing scan allocates 72.36 MB initially and
57.29 MB after AMR. Moving the same loop into a function that receives the
concrete `InterfaceField` and geometry types, without changing the formula,
produces the same checksum in about **0.10–0.14 ms and 5 KB**. The energy and
momentum loops each currently repeat the expensive form; `DivergenceNorm`
also constructs a mixture velocity and repeats fraction interpolation.

## Whole short advance with and without diagnostics

These timings exclude initialization and initial diagnostic registration.
Both on/off runs within each model have identical step counts, four AMR
changes, and the same final mesh. The first diagnostic-enabled measured run
still compiled some code; the values here use the later compilation-free run.

| Advance to `t=0.05` | Diagnostics on | Diagnostics off | Difference |
| --- | ---: | ---: | ---: |
| One velocity, wall time | 5.00 s | 3.71–3.80 s | ~1.25 s |
| One velocity, allocation | 6.95 GB | 4.66 GB | ~2.28 GB |
| CMDGFM, wall time | 9.32 s | 7.33–7.37 s | ~1.97 s |
| CMDGFM, allocation | 16.93 GB | 11.76 GB | ~5.17 GB |

One-velocity took 21 steps; CMDGFM took 22. These are evolving-run costs,
not a matched-state ratio of the diagnostic algorithms.

## Are the volumes already stored?

Yes, but there is no single reusable source used by all consumers:

- Each AMR staggered-advection schedule stores per-axis `volumes`. The
  one- and two-velocity momentum workspaces retain such schedules. Uniform
  schedules store a scalar volume instead. Some bounded schedules also hold
  boundary dual volumes, whereas diagnostics want interior faces only.
  The interior entries compared bit-for-bit with the diagnostic vector in
  both directions on the initial Stokes3 mesh and after CMDGFM's four AMR
  updates.
- The two-phase staggered-diffusion operator stores `volumes` in global-face
  order. For both Stokes3 models, this vector is **bitwise identical** to
  `_diagnostic_face_volumes` initially and after four AMR updates. It is
  already present throughout these runs because viscosity is nonzero.
- Embedded geometry stores per-axis `dual_volumes` and boundary dual
  volumes. The conversion to global-face order is currently repeated by
  some consumers.

The diffusion operator cannot be the universal owner: it is absent when
viscosity is zero, and standalone diagnostics can be called without a
simulation workspace. The topology-only vector now belongs to the **mesh**;
the physical open-volume counterpart belongs to fixed **geometry**. Both are
refreshed when their defining data change. Per-axis and boundary-volume
views are still separate. Consumers that resize or overwrite their volume
arrays make their own copy. For an uncut face, the direct formula is
`V_f = face.area * (h_minus + h_plus) / 2`, with each `h` the adjacent cell
spacing normal to the face; this is also the formula previously used by the
diffusion operator. The mesh cache serves standalone diagnostic calls too.

## Implemented diagnostic path and measured result

`BlockTreeMesh.face_dual_volumes` is created with the face schedule and
unconditionally refreshed on every topology rebuild, including cases where
the new schedule has the same length. `GeometryField.face_dual_volumes`
stores physical open volumes (zero on covered faces) and is refreshed with
the solid geometry. The diffusion operator copies the cache into its mutable
storage; the advection schedule reads the same mesh measure.

Energy and momentum now compute the staggered liquid fraction and mixture
weight **locally, once per face as needed**. They do not materialize a
fraction or density vector. The one- and two-velocity divergence diagnostics
reuse projection-workspace face and cell scratch when evaluating a flow state;
standalone states without a workspace retain their allocating fallback. The
five default Stokes3 diagnostic *value evaluations* are allocation-free after
warm-up, including the embedded-solid case in the focused `cut` benchmark.
Diagnostic-history vector growth and the surrounding simulation are distinct
from this per-value target; do not interpret the whole-run byte delta as zero.

| Frozen-state diagnostic | One, initial / evolved ms | CMDGFM, initial / evolved ms | Bytes/call |
| --- | ---: | ---: | ---: |
| Kinetic energy | 0.283 / 0.227 | 0.184 / 0.144 | 0 |
| Linear momentum | 0.180 / 0.135 | 0.174 / 0.135 | 0 |
| Phase volume | 0.008 / 0.007 | 0.008 / 0.007 | 0 |
| Divergence norm | 0.143 / 0.111 | 0.322 / 0.254 | 0 |
| Leaf blocks | <0.001 / <0.001 | <0.001 / <0.001 | 0 |

The initial mesh has 27,576 faces; both evolved runs have 21,832. These are
12-call medians, not paired confidence intervals. The CMDGFM face traversal
is no longer intrinsically slower than the one-velocity traversal; its
divergence calculation still has to form the mixture velocity.

| Warm short advance to `t=0.05` | Diagnostics on | Diagnostics off | Difference |
| --- | ---: | ---: | ---: |
| One velocity, wall time | 4.09 s | 3.91 s | ~0.18 s |
| One velocity, allocation | 4.748 GB | 4.747 GB | ~0.40 MB |
| CMDGFM, wall time | 7.47 s | 7.34 s | ~0.13 s |
| CMDGFM, allocation | 11.902 GB | 11.902 GB | ~0.45 MB |

These are the later compilation-free passes in the same-process paired
benchmark. They retain the original step counts and four AMR updates: 21
steps for one velocity, 22 for CMDGFM. Wall-time differences are noisy at
this scale; the allocation reduction is more decisive. Before the change the
diagnostic-on surplus was about 2.28 GB and 5.17 GB respectively.

The default `t=0.5` sampled profiles also kept their baseline step/AMR
histories. Relative to the preceding transport-optimized benchmark, the
diagnostic bucket and whole-run allocation changed as follows:

| Full Stokes3 `t=0.5` | One before | One after | CMDGFM before | CMDGFM after |
| --- | ---: | ---: | ---: | ---: |
| Steps / AMR changes | 206 / 45 | 206 / 45 | 219 / 32 | 219 / 32 |
| Sampled diagnostics time | ~7.77 s | ~0.07 s | ~19.23 s | ~0.13 s |
| Whole-run wall time | 36.62 s | 29.33 s | 83.50 s | 68.99 s |
| Whole-run allocation | 46.27 GB | 34.32 GB | 141.49 GB | 101.58 GB |

The sampled bucket is an estimate from one profiling run; small differences
are not precise timings. The large allocation reduction and matched step/AMR
histories support the cost attribution. The CMDGFM outer time-step bucket is
still about 4.78 s and belongs to [point 3](TIMESTEP_PLAN.md), not to
diagnostics.

## Next implementation gates

1. Introduce the common volume source without changing numerical values.
   Compare it elementwise with existing diagnostic, diffusion and per-axis
   advection volumes on uniform, mixed-AMR and embedded-solid meshes,
   including immediately after refinement/coarsening and geometry refresh.
   In Stokes3 no-solid states, exact equality is an appropriate gate.
2. Make energy and momentum use the shared vector and put the two-velocity
   face loop behind a concrete interface/geometry function barrier. Preserve
   the existing summation order and verify diagnostic histories. Avoid a
   cache that is refreshed only by matching array length; topology can change
   without changing the number of faces.
3. Then consider sharing same-state phase fractions/density across energy,
   momentum and divergence, and reusing divergence scratch. These are
   smaller costs and should be measured after the first two changes.
4. Repeat the frozen-state benchmark, paired on/off runs and default
   `t=0.5` Stokes3 benchmark. Run `test/diagnostics.jl`, Stokes3 example tests,
   AMR/embedded-geometry tests, then the full suite. The full suite has two
   independently reproduced pre-existing failures, recorded in
   [`TRANSPORT_FINDINGS.md`](TRANSPORT_FINDINGS.md).
