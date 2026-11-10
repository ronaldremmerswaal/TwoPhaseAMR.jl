"""
    AbstractObserver

Supertype for objects that watch a running [`simulate!`](@ref). Numerical code
dispatches on this interface without depending on any graphics package. An
observer implements [`observe!`](@ref) and may implement
[`stop_requested`](@ref). [`Recorder`](@ref) is the standard observer.
"""
abstract type AbstractObserver end

"""
    NoObserver()

Zero-cost observer used when `simulate!` has nothing to report to.
"""
struct NoObserver <: AbstractObserver end

"""
    StepInfo

Solver information passed to [`observe!`](@ref) together with the
[`SimulationState`](@ref) of each event:

- `dt`: size of the step that just completed (zero at `:initial`);
- `projection`: latest pressure-projection statistics;
- `topology_changed`: whether mesh adaptation after this step changed the leaf
  set;
- `final_time`: the time requested from `simulate!`;
- `step_procedure`: at `:initial`, an independent-step procedure for observers
  that advance copied state (see [`Recorder`](@ref)'s `exact_keyframes`),
  otherwise `nothing`;
- `exception`: the error that stopped the run (`:failed` only);
- `wall_time`: wall-clock seconds spent by the solver on this step, including
  mesh adaptation and diagnostics, excluding the observer.

Custom time loops construct it as `StepInfo(; dt, final_time,
projection=nothing, topology_changed=false, step_procedure=nothing,
exception=nothing, wall_time=0.0)`.
"""
struct StepInfo{T, S}
    dt::T
    projection::S
    topology_changed::Bool
    final_time::T
    step_procedure::Any
    exception::Union{Nothing, Exception}
    wall_time::Float64
end

function StepInfo(;
        dt, final_time, projection = nothing, topology_changed::Bool = false,
        step_procedure = nothing, exception = nothing, wall_time::Real = 0.0,
    )
    T = promote_type(typeof(dt), typeof(final_time))
    return StepInfo{T, typeof(projection)}(
        T(dt), projection, topology_changed, T(final_time), step_procedure,
        exception, Float64(wall_time),
    )
end

"""
    observe!(observer, event, state, info)

Called by [`simulate!`](@ref) with `event` `:initial` (before the first step),
`:step` (after every completed step, mesh adaptation, and diagnostic
recording), `:completed` (once at the end, also after a requested stop), or
`:failed` (when the run throws). `state` is the read-only
[`SimulationState`](@ref) and `info` the [`StepInfo`](@ref). The observer must
not modify solver-owned data. An exception thrown at `:initial` aborts the run
before its first step. Later observer exceptions are logged and the simulation
continues, except `InterruptException`, which propagates to the caller.
"""
function observe! end

observe!(::NoObserver, event::Symbol, state, info) = nothing

_observe_safely!(::NoObserver, event, state, info) = nothing
function _observe_safely!(observer::AbstractObserver, event, state, info)
    try
        observe!(observer, event, state, info)
    catch exception
        exception isa InterruptException && rethrow()
        @error "Observer event failed; the simulation continues" event exception = (
            exception, catch_backtrace(),
        )
    end
    return nothing
end

"""
    stop_requested(observer)::Bool

Polled by [`simulate!`](@ref) before the first step and after every completed
step. Returning `true` makes `simulate!` finish at that step boundary: it emits
`:completed` and returns with `stopped = true`. Exceptions during polling are
logged and treated as `false`, except `InterruptException`, which propagates.
"""
stop_requested(::AbstractObserver) = false

_observer_stop_requested(::NoObserver) = false
function _observer_stop_requested(observer::AbstractObserver)
    try
        return stop_requested(observer)::Bool
    catch exception
        exception isa InterruptException && rethrow()
        @error "Observer stop poll failed; the simulation continues" exception = (
            exception, catch_backtrace(),
        )
        return false
    end
end

# Dynamic mesh adaptation is incompatible only with observers that advance
# copied state over several steps (exact keyframes).
_supports_dynamic_topology(::AbstractObserver) = true
