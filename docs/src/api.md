# Public API

```@docs
TwoPhaseAMR
```

The standard workflow needs only [`BlockTreeMesh`](@ref),
[`FlowSimulation`](@ref), and [`simulate!`](@ref). Initial data use
[`FaceVelocity`](@ref) and [`CellField`](@ref); wall behavior is selected with
[`NoSlip`](@ref) or [`FreeSlip`](@ref). Supplying an [`InterfaceField`](@ref)
to the same simulation constructor selects two-phase flow. Construct physical
interfaces from a level set, and use [`PhaseScalar`](@ref) for intrinsic scalar
values that differ by phase. [`CSF`](@ref) and [`GFM`](@ref) select the surface
tension formulation.

Advanced users can work directly with the following layers:

- fields, placements, halo exchange, and transfer;
- adaptive indicators, arbitration, and regridding;
- staggered transport and diffusion operators;
- geometric interface reconstruction, compatible phase transport, and
  interface-aware refinement indicators;
- variable dynamic-viscosity stress and variable-density projection;
- compatible gradients, pressure projection, and elliptic solvers; and
- visualization registries, checkpoint sessions, and GUI controls.

No separate data representation is introduced by the high-level API:
`flow.velocity`, `flow.pressure`, `flow.solver`, and `flow.workspace` are the
same public low-level objects documented below.

## Complete documented API

```@autodocs
Modules = [TwoPhaseAMR]
Private = false
Order = [:type, :function, :constant]
```
