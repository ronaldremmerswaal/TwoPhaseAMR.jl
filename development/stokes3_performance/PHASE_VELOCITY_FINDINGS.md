# Absent phase velocities and vorticity-based AMR

## Cause

The old two-velocity storage retained a finite liquid velocity in a staggered
volume containing only gas (and conversely). The Stokes3 vorticity criterion
read that dormant extension as if it were physical liquid velocity. In the
default CMDGFM run at `t=0.5`, 27 of 93 pure-gas blocks with a two-cell
pure-gas halo were prevented from coarsening. The maximum fictitious liquid
vorticity in those blocks was about 160. This is not physical vorticity.

## Storage and numerical stencils

An intrinsic phase velocity is now `NaN` if its open staggered phase volume
is exactly zero. There is no density-dependent cutoff: a positive phase
volume, however small, still requires a finite physical value. Fully solid
faces are absent for both phases. A non-finite value on an active face is an
error, not an instruction to substitute zero.

Transport, diffusion, pressure projection, forcing, boundary conditions,
diagnostics, visualization and AMR respect this distinction. Zero phase
weights exclude absent values without evaluating `0 * NaN`. Pure-phase
apertures are exactly zero for the absent phase; polygon roundoff must not
create a nonzero flux coefficient multiplying an absent velocity. CMDGFM
phase-side inference donors must have a physical open dual volume.

Missing interpolation values are computed from valid neighbours in working
storage only. They are never saved as physical velocities in an absent phase.
The centered curl stencil uses valid phase cells, with a local affine fit
within two neighbour rings where the stencil extends past the interface.
A rank-deficient donor set uses a constant extension; an unresolved active
curl remains `NaN` and conservatively prevents coarsening. An absent phase
cell has no curl and does not block coarsening. The extrapolation does not
modify the velocity field and does not accept stale finite absent-phase
values as donors.

AMR similarly uses a finite working extension from valid same-component
faces. Its affine fit preserves the existing linear-exact transfer near
interfaces and solids. After transfer and interface reconstruction, the
new mesh's absent velocities are marked `NaN` again. Explicit registration
is `register_adaptive_field!(amr, velocity_tuple, interface)`; the usual
`AMRWorkspace(mesh, velocity_tuple, interface)` associates them automatically.

## Reproduction and validation

```sh
julia --project=. development/stokes3_performance/phase_velocity_amr.jl 0.5
TWOPHASE_TEST_PROGRESS=true julia --project=. test/runtests.jl
```

The new regression file `test/two_phase/two_velocity/inactive_velocity.jl`
checks refine/coarsen transfer, finite active and NaN absent values,
linear-exact phase curl, independence from poisoned dormant values,
pure-phase coarsening, and transport/diffusion/projection for both projection
models and both interface transport policies. Existing cut-cell, AMR,
conservation and allocation tests remain part of the full suite.

## Default CMDGFM Stokes3 validation to `t=0.5`

| Quantity | Before the correction | NaN phase storage and phase curl |
| --- | ---: | ---: |
| Initial faces / blocks | 27,576 / 214 | 27,576 / 214 |
| Final faces / blocks | 21,448 / 166 | 9,016 / 70 |
| Steps | 219 | 219 |
| AMR changes | 32 | 79 |
| Eligible pure-gas blocks blocked by the liquid curl, final | 27 | 0 |
| Invalid active / incorrectly finite absent velocities, final | Not enforced | 0 / 0 |
| Final pressure solve converged | Yes | Yes |

The validation advance took 73.304 s, including 22.305 s compilation, and
allocated 80.583 GB. It ran concurrently with tests using Julia 1.12.7,
`--pkgimages=no --cpu-target=generic`; these are **not** comparable warm timing
measurements against the earlier pressure table. The changed AMR trajectory
must not be attributed solely to a pressure-kernel speedup. Pressure
optimization follows this correction, with a new matched-state baseline.

The native package-image loader intermittently segfaulted before simulation
started. Disabling package images allowed this validation to finish; that
runtime issue is separate from the numerical NaN checks.

The focused transport/CMDGFM suite passed all 1,918 checks. The rotation
integration sweep also recovered three formerly broken frozen-method
convergence gates: the interface L1 errors at `N=16,32,64` are
`2.315e-3, 7.941e-4, 4.061e-4` (aggregate order 1.256), and the final staggered
L1 refinement order is 1.044. Those checks are now required tests rather than
expected failures. These integral results do not establish Linf convergence.

All 2,418 phase-storage, AMR, phase-curl and diffusion checks also passed
with `--check-bounds=yes`. The rotation rerun passed all 194 checks. A
package-image-free full-suite attempt stopped in Julia's code generator
while compiling the mimetic projection, so disabling package images is not
a general guarantee against the runtime crashes. Full-suite results are
recorded separately from these completed numerical checks.

The final full suite completed: **15,451 passed, 2 failed, 1 broken**.
There are no new failures. The two failures match the documented pre-change
baseline: `mimetic_projection_variants.jl:65` (pressure commuting residual
`0.9 < 0.2`) and `cylinder_example.jl:13` (non-finite pressure drag). The
broken test concerns optional solver packages. In particular, the core
CMDGFM/mimetic projections, cut-cell/AMR transport, phase diffusion,
visualization and zero-allocation diagnostics/CFL gates pass.
