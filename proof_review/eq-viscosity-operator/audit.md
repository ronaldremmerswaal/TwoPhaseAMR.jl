# Audit: eq:viscosity:operator — integrated stiffness and operator of the two-phase stress

- Source: `docs/techref/chapters/09_viscosity.tex:277-286` (and the sentence on $\mu\Delta u$, l. 31), commit `03bad7a`; code `diffusion!` (`variable_diffusion_operator.jl`), `_factorized_variable_stiffness_mul!`, `_geometry_face_dual_volumes`
- Date: 2026-11-09
- Overall status: ✅ verified
- Rerun: `julia --project=@claude-symbolics proof_review/eq-viscosity-operator/scripts/run_all.jl`

## Statement and assumptions
Stiffness $\mathsf E^\top\mathsf W\mathsf E$, operator $-\dualvolmat^{-1}\mathsf E^\top\mathsf W\mathsf E\,u$ with $\dualvolmat=\operatorname{diag}(\meas{\omega_f})$ or $\operatorname{diag}(\meas{\omega^{\rm o}_f})$ with 1 where it vanishes (solid); for constant viscosity and $\nabla\!\cdot u=0$ the continuous operator is $\mu\Delta u$.

## Steps
| Step | Source | Status | Method | Script | Comment |
|---|---|---|---|---|---|
| stiffness $=\mathsf E^\top\mathsf W\mathsf E$ (with the code's wall rows for a solid) | l. 278-280 | ✅ | independent $\mathsf E$, $\mathsf W$: $\le1.8\times10^{-16}$ relative on walled, cut, periodic | `code_01_operator.jl` | |
| volumes: `mesh.face_dual_volumes`; open dual volumes with 1 where they vanish | l. 281-283 | ✅ | `op.volumes` equal exactly | same | |
| `diffusion!` $=-\dualvolmat^{-1}\mathsf Su$, boundary rates 0; factorised action $=\mathsf Su$ | l. 280, 283 | ✅ | $\le3.4\times10^{-16}$ (relative); factorised action $\le10^{-11}$ | same | |
| $-\dualvolmat^{-1}\mathsf Su\to\mu\Delta u$ for constant $\mu$, $\nabla\!\cdot u=0$ | l. 31 | ✅ | uniform periodic mesh, solenoidal $u$: max relative error $1.28\times10^{-2},3.2\times10^{-3},8.0\times10^{-4}$ ($n=16,32,64$), order 1.99, 2.00 | same | |

## Errors and consequences
None.

## Sharpness
| Claim | Classification | Evidence | Script | Possible improvement |
|---|---|---|---|---|
| second order consistency with $\mu\Delta u$ | **likely sharp** (observed 2.00 on uniform mesh) | refinement table | `code_01_operator.jl` | at hanging vertices/levels: not tested in this script (quadratic form: `eq-viscosity-quadratic`) |
| constant-$\mu$, div-free condition | needed | for non-solenoidal $u$ the stress form equals $\mu(\Delta u+\nabla\nabla\!\cdot u)$ (not tested) | — | none |

## Open points
None.

## History
- 2026-11-09: first audit.
