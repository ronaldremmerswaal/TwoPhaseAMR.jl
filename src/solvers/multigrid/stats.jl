"""Convergence statistics from the latest external Krylov solve."""
mutable struct LinearSolverStats{T}
    cycles::Int
    converged::Bool
    initial_l2::T
    initial_linf::T
    final_l2::T
    final_linf::T
    target_linf::T
    residual_scale::T
    removed_rhs_mean::T
    method::Symbol
    preconditioner::Symbol
end

LinearSolverStats(::Type{T}; method = :none, preconditioner = :none) where {T} =
    LinearSolverStats(
    0, false, T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), one(T), zero(T),
    method, preconditioner
)

function Base.show(io::IO, stats::LinearSolverStats)
    return print(
        io,
        "LinearSolverStats(",
        "cycles=", stats.cycles,
        ", converged=", stats.converged,
        ", initial_l2=", stats.initial_l2,
        ", initial_linf=", stats.initial_linf,
        ", final_l2=", stats.final_l2,
        ", final_linf=", stats.final_linf,
        ", target_linf=", stats.target_linf,
        ", residual_scale=", stats.residual_scale,
        ", removed_rhs_mean=", stats.removed_rhs_mean,
        ", method=:", stats.method,
        ", preconditioner=:", stats.preconditioner,
        ")",
    )
end

function _dimensionless_tolerance(::Type{T}, tolerance) where {T}
    value = T(tolerance)
    isfinite(value) && zero(T) < value < one(T) || throw(
        ArgumentError("tolerance must be finite and lie in (0, 1)"),
    )
    return value
end

function _strong_maximum(values, weights)
    T = promote_type(eltype(values), eltype(weights))
    result = zero(T)
    for index in eachindex(values, weights)
        weight = @inbounds weights[index]
        weight > zero(weight) || continue
        result = max(result, abs(@inbounds(values[index])) / weight)
    end
    return result
end

function _dimensionless_residual(stats::LinearSolverStats{T}) where {T}
    scale = stats.residual_scale
    scale > zero(T) && return stats.final_linf / scale
    return iszero(stats.final_linf) ? zero(T) : T(Inf)
end

# Composite linear solvers expose component statistics through `x` and `y`.
# Model-specific projection statistics extend this function in their own domain.
_dimensionless_residual(stats) =
    map(_dimensionless_residual, (stats.x, stats.y))

function _solver_failure_message(
        problem, stats, tolerance; context = nothing, details = nothing
    )
    prefix = isnothing(context) ? string(problem) :
        "$(context) $(problem)"
    achieved = _dimensionless_residual(stats)
    detail_text = isnothing(details) ? "" : ", $(details)"
    return "$(prefix) failed to converge " *
        "(tolerance=$(tolerance), " *
        "dimensionless_residual=$(achieved)$(detail_text), " *
        "stats=$(repr(stats)))"
end

