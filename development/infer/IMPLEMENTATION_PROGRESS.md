# `:infer` implementation and validation status

The current production option is `MimeticProjection(jump_interpolant=:infer)`.
It keeps the existing aperture divergence, diagonal physical jump mass `K`,
and symmetric three-block pressure projection. The default selector is
`infer_face_selection=:paired`; the explicit comparisons `:smoothed`, `:plic`,
and `:stencil` change only jump-face selection/normal geometry. **No change to
`K` has been made.** Further metric changes require discussion with the user.

## What is imposed and inferred

For each resolved cut cell, one unused x-face and one unused y-face are
selected using staggered fractions closest to 1/2. With the selected normal
`n=(n_x,n_y)` and diagonal physical masses `K_x,K_y`, the two columns are

```
E_T = ( n_y,       -n_x       ) / max(|n_x|,|n_y|)
E_N = ( K_y*n_x,    K_x*n_y   ) / max(|K_y*n_x|,|K_x*n_y|).
```

Thus `E_T' K E_N=0`, and `N=E_N'K` is a positive multiple of the local
normal jump. The multiplier enforces `N[u]=0`; tangential stationarity gives
the pressure-inferred correction along `E_T`. The two columns span the
original two-face space. An unpaired face is imposed only when its omitted
normal component is `O(h)`; otherwise it stays inferred. The selected
normal comes from the cell aperture vector `D(a_l n)` by default, its
one-ring integrated average for `:smoothed`, or the reconstructed PLIC normal
for `:plic`. `:stencil` instead classifies each face by CMDGFM's `beta`:
normal-dominant faces impose a consistent normal row, tangential-dominant
faces remain free. It uses the same symmetric KKT system but not an explicit
paired basis. The CMDGFM pressure-dependent inference term is not copied.

The implementation reports paired cells, imposed/inferred directions, and
unpaired/unresolved fallbacks in `MimeticProjectionStats` and the KH
single-step diagnostic. It does not silently count fallback as a certified
paired normal row.

## Evidence obtained before the Julia runtime failure

The targeted mimetic projection variant tests passed, including 39 new
assertions for all four `:infer` selectors: symmetric KKT, independent
constraint rows, positive reduced mass, `E'KE` orthogonality, periodic
momentum, and the physical kinetic projection identity. The existing
projection tests also passed in that run.

The first raw-aperture paired prototype, before the stricter `O(h)`
unpaired-face fallback, gave these **full-final-time** KH amplitude errors:

| density ratio | Weber | mesh | raw `:infer` | CMDGFM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5 | 8 | 0.774 | 0.710 |
| 1 | 0.5 | 16 | 0.474 | 0.414 |
| 1 | 5 | 8 | 0.229 | 0.255 |
| 1 | 5 | 16 | 0.121 | 0.130 |
| 1000 | 0.5 | 8 | 1.588 | 1.748 |

At density ratio 1, Weber 0.5, mesh 16, raw `:infer` errors for time-step
factors 1, 1/2, 1/4 were 0.474, 0.434, 0.424. At density ratio 1000, mesh
8, the factors gave 1.588, 1.527, 1.875. The user clarified that temporal
acceptance means *absence of divergence*, not monotone analytical error.
These three high-density values are bounded but not a sufficient asymptotic
study. Results predating the `O(h)` fallback must not be used as current-code
regressions.

At density ratio 1000, after a half-sized KH transport step, the raw
aperture pair selector produced a spurious local normal direction near
`(0.78,-0.63)` on an almost flat interface. Its normal-jump defect was
2.65. Both one-ring aperture smoothing and direct PLIC normals reduced the
largest local defect to about 0.11 without changing `K`. Nevertheless, their
one-step velocity corrections remained about 0.19 of the initial norm,
compared with CMDGFM's 0.0038. Local normal consistency therefore does not
by itself reproduce CMDGFM's dynamical behavior.

After the `O(h)` fallback change, the **full-final-time** high-density,
Weber-0.5, mesh-8 error was 3.00 for `:smoothed` and 1.72 for direct `:plic`;
CMDGFM was 1.75. The smoothed mesh-16 error was 1.52. A mesh-16 direct-PLIC
run was attempted but not obtained: Julia began segfaulting while loading
its system image, even for `println(1)` before package code runs. The
unsandboxed direct Julia binary failed the same way. Further numerical
validation requires a working Julia runtime.

The forward-only circular normal test in `normal_consistency.jl` shows the
default aperture-pair **normal equation** is consistent on static geometry:
for density `(1000,1)`, `L2` errors at n=8,16,32,64 were 0.129, 0.0495,
0.0289, 0.0183 and `Linf` errors were 0.161, 0.124, 0.0559, 0.0352.
The reverse density order behaved similarly. This does not validate the
pressure-inferred tangent or moving-interface KH dynamics.

## Unfinished gates and decision point

The current code still needs post-change Poisson, KH mesh/time at all four
density/Weber combinations and both transports, capillary/RT, performance,
and full integration regressions. In particular, no claim of high-density KH
mesh convergence or non-divergent asymptotic time refinement is justified
yet. If the fixed-`K` selectors fail these gates after Julia is repaired,
the next step is **not** an automatic metric edit: discuss with the user how
to alter the physical kinetic Hodge and its adjoint pressure coupling
together, while retaining aperture divergence and the energy identity.
