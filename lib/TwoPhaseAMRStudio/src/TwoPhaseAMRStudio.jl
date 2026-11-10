"""
    TwoPhaseAMRStudio

Authenticated local HTTP/WebSocket access to a `TwoPhaseAMR.Recorder`.
The server owns simulation controls and optional recording jobs; timeline and
other view state belong exclusively to each browser client.

Protocol v1 uses one authenticated `/ws?token=…` connection and JSON text
messages with a `type` member. A client receives `hello` then `state`, followed
by `frame` and retained-index `frames` updates. `control` actions are `pause`,
`resume`, `stop`, `keyframe`, and `continue`; `recording` actions are `start` and `stop`.
`get` supports `frames`, `field`, `topology`, `planes`, `diagnostics`, and
`steplog`; optional UInt32 `request` ids are echoed in responses and errors.

Field requests include `frame`, `field`, `encoding` (`f32` or `u16`), and a
UInt32 `request`. A two-phase field also requires `phase` (`liquid` or `gas`).
The optional `mode` (`cell` or `plic`) is a validated display hint, not a
server-side transform: each request returns the selected intrinsic raw array.
Request both arrays and `get planes` for majority masks and sparse interface
planes. `get topology` supplies blocks, per-axis face tiles and solid geometry.
Binary replies use the 32-byte little-endian TPAS header described by
[`encode_field`](@ref) and [`decode_field`](@ref).
"""
module TwoPhaseAMRStudio

import HTTP
import JSON
import Random
import TwoPhaseAMR
import CairoMakie

export studio, StudioDriver, url, encode_field, decode_field

const StudioDriver = TwoPhaseAMR.StudioDriver

include("protocol.jl")
include("metadata.jl")
include("recording.jl")
include("server.jl")

end
