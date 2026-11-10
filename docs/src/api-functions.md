# Public API functions

The public types and the high-level workflow are listed in the
[Public API](api.md). This page contains the remaining exported function
reference; [`locate_face`](@ref) is documented with [`FaceRef`](@ref) there.

```@autodocs
Modules = [TwoPhaseAMR]
Private = false
Filter = t -> t !== TwoPhaseAMR.locate_face
Order = [:function]
```
