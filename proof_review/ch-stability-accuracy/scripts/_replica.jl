# Literal replica of the step-selection loop of src/navier_stokes/simulate.jl (lines 90-91, 122-149, 166-172), Float64.
# Shared by step02_selection_replica.jl (the rules, on synthetic bounds) and code_02_selection_real.jl (replica vs the package).

# One call of simulate!(...; dt) over [t0, T]; `bound(n)` is the flow bound at the start of step n (n = 1, 2, ...), `retained` the proposal from a
# previous call (Inf if none), `defect0` the defect step of the previous call (0 at the first step of a run).
function replica_call(t0, T, bound; dt = nothing, retained = Inf, defect0 = 0.0, regrid = n -> false)
    current = t0
    selected = isnothing(dt) ? retained : float(dt)
    defect = defect0
    time_tolerance = 16eps() * max(1.0, abs(T), abs(current))
    rec = NamedTuple{(:step, :proposed, :defect, :reference), NTuple{4, Float64}}[]
    n = 0
    while T - current > time_tolerance
        if isnothing(dt)
            b = bound(n + 1)
            selected = min(selected, b)
            if n == 0 && !isfinite(b) && !isfinite(selected)
                selected = (T - current) / 128
            end
        end
        remaining = T - current
        proposed = isfinite(selected) ? selected : remaining
        step = remaining - proposed < 1.0e-6 * proposed ? remaining : proposed
        reference = max(proposed, defect, step)
        push!(rec, (; step, proposed, defect, reference))
        defect = step
        current = step == remaining ? T : min(T, current + step)
        n += 1
        regrid(n) && (defect = proposed)
    end
    return (; rec, time = T, retained = selected)
end
