# Types owned by the intrinsic two-velocity transport model.
"""
    TwoPhaseVelocity

A liquid/gas pair of existing [`FaceVelocity`](@ref) fields, stored as the
ordinary tuple `(liquid_velocity, gas_velocity)`. The alias adds dispatch and
documentation without introducing another field container or duplicating
velocity storage.

`FlowSimulation` and two-velocity updates store NaN for an absent phase
(exactly zero open staggered phase volume). Stencil extensions are computed
from valid phase values in working storage only. Phase-weighted operations
must exclude absent values before arithmetic, not multiply them by zero.
"""
const TwoPhaseVelocity = Tuple{<:FaceVelocity, <:FaceVelocity}
