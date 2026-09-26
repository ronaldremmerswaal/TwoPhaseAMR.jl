# Temporal continuity of the inferred jump projection

For a fixed velocity predictor, pressure time step, and mesh, move only the
interface by a displacement `s`. A useful temporal-regularity diagnostic is
whether the *projected velocity* changes continuously with `s`, particularly
when a staggered face first becomes mixed. Smooth aperture coefficients alone
do not guarantee this: a new hard equation changes the feasible space.

If `N(s) j = 0` is imposed with a Lagrange multiplier, replacing one row by
`w(s) N(s)` has no gradual-enforcement effect for any `w(s) != 0`. The row
can be rescaled by changing its multiplier. When `w` reaches zero, its rank
disappears abruptly. In this implementation, candidate rows are constructed
only in mixed cells from active mixed faces, and the independent-row test
normalizes rows in the `K^{-1}` metric before rank-revealing QR. Thus the
interface-length factor in `D(a n)` is a natural geometric scaling but is not
a soft constraint.

The active-face test also uses a hard minority-mass threshold. At an inactive
mixed face, the present mimetic assembly fixes the *final velocity jump* to
zero by applying its entire predictor jump as a known correction. This is
separate from setting the pressure-gradient jump to zero, which would leave
the velocity jump unchanged but generally violate the existing kinetic-energy
projection identity. Both effects deserve explicit attention when studying
time continuity.

## Frozen KH geometry-shift probe

[`kh_operator_continuity.jl`](../kh_operator_continuity.jl) holds the analytic KH
predictor fixed and translates a sinusoidal interface vertically. These
measurements use `n=16`, density ratio 1000, Weber number 5, and the same
pressure time step for every shift. The response is the RMS change of both
projected phase velocities on faces that remain at least 1% mixed in *both*
geometries, weighted by the smaller `c(1-c)` factor. It is a spatial response
diagnostic, not a wave-amplitude error.

| Projection | shift / h | active jump faces | normal rows | RMS projected-velocity change |
| --- | ---: | ---: | ---: | ---: |
| CMDGFM | 0.00001 | 46 | 46 jump equations | 0.0000145 |
| CMDGFM | 0.0001 | 44 | 44 jump equations | 0.0182 |
| mimetic cell, `c(1-c)` weights | 0.0001 | 54 | 18 | 1.9305 |
| mimetic face, `c(1-c)` weights | 0.0001 | 54 | 52 | 0.00826 |
| joint, raw aperture direction | 0.00001 | 50 | 16 | 0.0000553 |
| joint, raw aperture direction | 0.0001 | 54 | 18 | 0.2936 |
| joint, smoothed aperture direction | 0.00001 | 50 | 16 | 0.0000517 |
| joint, smoothed aperture direction | 0.0001 | 54 | 18 | 0.0135 |

The new rows appear between the two positive shifts. Normal smoothing reduces
their effect by about 22 times and brings it to the size of the CMDGFM event,
but does not make the operator mathematically continuous. Lowering the
mass-fraction cutoff from `1e-6` to `1e-8` does not alter this event; the
particular transition is caused by newly mixed geometric support.
The legacy `c(1-c)` coefficient weighting does not by itself ensure a smooth
*projection*: cell rows exhibit the largest event, whereas the face rows have
the smallest one. The spatial arrangement and rank of the equations matter.

## KH time refinement

The full 1000:1, We=5, `n=16` wave test gives:

| Projection | dt factor | amplitude error | difference from previous time-step history |
| --- | ---: | ---: | ---: |
| joint, raw aperture direction | 1 | 0.1469 | — |
| joint, raw aperture direction | 1/2 | 0.0848 | 0.1026 |
| joint, raw aperture direction | 1/4 | 0.2320 | 0.2899 |
| joint, smoothed aperture direction | 1 | 0.0997 | — |
| joint, smoothed aperture direction | 1/2 | 0.0934 | 0.1504 |
| joint, smoothed aperture direction | 1/4 | 0.1121 | 0.0418 |

At the coarser time step, the smoothed variant's mesh errors at 8, 16,
and 32 cells per wavelength are 0.2940, 0.0997, and 0.1220. This is
not monotone mesh convergence, and the `n=32` error is still larger than
CMDGFM's 0.0725. Temporal improvement has not yet met the entire KH
accuracy objective.

Thus the smoother direction avoids the raw variant's observed refined-step
error growth in this physics case, although three time steps do not establish
asymptotic convergence. CMDGFM also has an operator-support switch in the
frozen probe, so support switching alone is not proof of the KH failure.
At the finest time step, the smoothed run's maximum per-projection kinetic
energy identity defect is `9.9e-11`; its total kinetic-plus-surface energy
nevertheless decreases by 1.32% over the full coupled run. The first number
checks the algebra of the projection; the second includes transport, surface
forces, and their time discretisation.

## What would make an activation genuinely gradual?

For a hard projection, the relevant object is the constrained subspace (or
its projector in the kinetic metric), not the magnitude of a row. A new
independent constraint cannot be switched on gradually by scaling it. The
practical options are: keep the effective constraint rank fixed while moving
its support continuously; make new faces inherit a jump represented by
already constrained neighbours before they gain an independent degree of
freedom; or use a finite penalty/compliance, accepting that this changes the
exact kinetic-energy projection identity. The first two can retain the
existing symmetric KKT structure and aperture divergence, but need careful
tests at changes of interface topology. No change to `K` was made here.
