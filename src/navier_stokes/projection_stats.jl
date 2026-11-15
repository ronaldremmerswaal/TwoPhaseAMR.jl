"""
    ProjectionRunStats

Run-level accounting of the pressure-projection solves performed by one
[`simulate!`](@ref) call. `simulate!` returns it as the `projection` field of
its result and [`print_simulation_summary`](@ref) reports it.

The absolute target of a projection solve is `tolerance / dt^2` in the strong
norm; a developed flow often satisfies it before the iterative solve starts,
and such a solve reports no iterations and an unchanged residual. The final
step alone therefore does not describe the projection work of a run.
`solves` counts the projected steps (one per physical step; the extra
two-velocity projection after a regrid is not included), `iterations` sums the
iterations of all solves, `without_iterations` counts the solves that did no
work, `max_iterations` is the largest count of a single solve, and `converged`
records whether every solve met its target. A solve *worked* when it performed
an iteration or reduced its residual; a two-velocity warm start can reduce the
residual without an iteration. `last_iterations`, `last_initial_residual`, and
`last_final_residual` describe the most recent solve that worked; they stay
zero when none did. Residuals use the native norm of the projection solver
(`initial_residual` of `LinearSolverStats`, `CMDGFMProjectionStats` or
`MimeticProjectionStats`). For a [`MimeticProjection`](@ref)
`defective_cells` and `lumped_values` sum the per-solve counts of
`MimeticProjectionStats` over the run and `max_defective_cells`,
`max_lumped_values` are their largest per-solve values; they stay zero for
the other projections.
"""
mutable struct ProjectionRunStats{T}
    solves::Int
    iterations::Int
    without_iterations::Int
    max_iterations::Int
    converged::Bool
    last_iterations::Int
    last_initial_residual::T
    last_final_residual::T
    defective_cells::Int
    max_defective_cells::Int
    lumped_values::Int
    max_lumped_values::Int
end

ProjectionRunStats(::Type{T}) where {T} =
    ProjectionRunStats(0, 0, 0, 0, true, 0, zero(T), zero(T), 0, 0, 0, 0)

"""Defective cut cells and lumped phase values of one projection solve (zero
for projections without a cut-cell metric)."""
_cut_cell_counts(stats) = (0, 0)

"""Record one pressure-projection solve in `run`."""
function _accumulate_projection!(run::ProjectionRunStats{T}, stats) where {T}
    count = iterations(stats)
    initial = T(initial_residual(stats))
    final = T(final_residual(stats))
    run.solves += 1
    run.iterations += count
    run.max_iterations = max(run.max_iterations, count)
    run.converged &= converged(stats)
    defective, lumped = _cut_cell_counts(stats)
    run.defective_cells += defective
    run.max_defective_cells = max(run.max_defective_cells, defective)
    run.lumped_values += lumped
    run.max_lumped_values = max(run.max_lumped_values, lumped)
    if iszero(count) && !(final < initial)
        run.without_iterations += 1
        return run
    end
    run.last_iterations = count
    run.last_initial_residual = initial
    run.last_final_residual = final
    return run
end

"""Whether any solve of `run` did work: iterated or reduced its residual."""
_has_projection_reduction(run::ProjectionRunStats) =
    run.solves > run.without_iterations
